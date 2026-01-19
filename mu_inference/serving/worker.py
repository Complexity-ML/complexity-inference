"""
Mu Worker
=========

GPU worker for model execution.

Handles:
- Model loading and management
- Forward pass execution
- KV cache management
- Batch processing
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import torch

from mu_inference.core.config import MuConfig, ModelConfig, EngineConfig
from mu_inference.core.cache import MuKVCache, MuPagedCache, create_cache
from mu_inference.models.base import MuModelBase, MuModelOutput
from mu_inference.models.loader import load_model

logger = logging.getLogger(__name__)


@dataclass
class WorkerRequest:
    """Request for the worker to process."""
    request_id: str
    input_ids: torch.Tensor  # [1, seq_len]
    position_ids: Optional[torch.Tensor] = None
    is_prefill: bool = True  # True for first pass, False for decode


@dataclass
class WorkerOutput:
    """Output from the worker."""
    request_id: str
    logits: torch.Tensor  # [1, vocab_size]
    finished: bool = False
    error: Optional[str] = None


class MuWorker:
    """
    GPU worker for model execution.

    Manages the model and KV cache on a single GPU.
    """

    def __init__(
        self,
        config: EngineConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.dtype = self._get_dtype(config.dtype)

        # Model and cache
        self.model: Optional[MuModelBase] = None
        self.cache = None

        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "prefill_tokens": 0,
        }

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def load_model(self, model_path: str) -> None:
        """
        Load model from path.

        Args:
            model_path: Local path or HuggingFace model ID
        """
        logger.info(f"Loading model: {model_path}")

        self.model = load_model(
            model_path,
            config=self.config.model,
            mu_config=self.config.mu,
            device=self.device,
            dtype=self.dtype,
        )

        self.model.eval()

        # Initialize cache
        self._init_cache()

        logger.info(f"Model loaded: {self.model}")

    def _init_cache(self) -> None:
        """Initialize KV cache."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        config = self.model.config

        self.cache = create_cache(
            config=self.config.cache,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads or config.num_attention_heads,
            head_dim=config.head_dim,
            device=self.device,
            mu_config=self.config.mu,
        )

        logger.info(f"Cache initialized: {self.cache.get_stats()}")

    def allocate_request(self, request_id: str) -> bool:
        """
        Allocate resources for a new request.

        Returns:
            True if allocated, False if at capacity
        """
        if not self.cache.allocate(request_id):
            return False

        self.active_requests[request_id] = {
            "past_key_values": None,
            "seq_len": 0,
        }

        return True

    def free_request(self, request_id: str) -> None:
        """Free resources for a completed request."""
        self.cache.free(request_id)
        self.active_requests.pop(request_id, None)

    @torch.inference_mode()
    def execute(self, requests: List[WorkerRequest]) -> List[WorkerOutput]:
        """
        Execute forward pass for a batch of requests.

        Supports continuous batching:
        - Decode requests (single token) are batched together for efficiency
        - Prefill requests are processed individually (variable length)

        Args:
            requests: List of WorkerRequest

        Returns:
            List of WorkerOutput
        """
        if not requests:
            return []

        # Separate decode (single token) vs prefill (variable length)
        decode_requests = [r for r in requests if not r.is_prefill]
        prefill_requests = [r for r in requests if r.is_prefill]

        outputs_map: Dict[str, WorkerOutput] = {}

        # Batch decode requests together (efficient)
        if decode_requests:
            try:
                decode_outputs = self._execute_batched_decode(decode_requests)
                for output in decode_outputs:
                    outputs_map[output.request_id] = output
            except Exception as e:
                logger.error(f"Batched decode error: {e}")
                # Fallback to individual processing
                for request in decode_requests:
                    try:
                        output = self._execute_single(request)
                        outputs_map[output.request_id] = output
                    except Exception as ex:
                        outputs_map[request.request_id] = WorkerOutput(
                            request_id=request.request_id,
                            logits=torch.zeros(1, self.model.config.vocab_size),
                            finished=True,
                            error=str(ex),
                        )

        # Process prefill requests individually (variable length)
        for request in prefill_requests:
            try:
                output = self._execute_single(request)
                outputs_map[output.request_id] = output
            except Exception as e:
                logger.error(f"Error processing {request.request_id}: {e}")
                outputs_map[request.request_id] = WorkerOutput(
                    request_id=request.request_id,
                    logits=torch.zeros(1, self.model.config.vocab_size),
                    finished=True,
                    error=str(e),
                )

        # Return in original order
        return [outputs_map[r.request_id] for r in requests]

    @torch.inference_mode()
    def _execute_batched_decode(self, requests: List[WorkerRequest]) -> List[WorkerOutput]:
        """
        Execute batched decode for multiple requests.

        All decode requests have sequence length 1, so we can batch them efficiently.
        """
        batch_size = len(requests)

        # Collect input tensors and position info
        input_ids_list = []
        position_ids_list = []
        request_ids = []

        for request in requests:
            req_state = self.active_requests.get(request.request_id)
            if req_state is None:
                raise RuntimeError(f"Request {request.request_id} not allocated")

            input_ids_list.append(request.input_ids.to(self.device))
            request_ids.append(request.request_id)

            # Position ID based on past sequence length
            past_kv = req_state["past_key_values"]
            past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
            position_ids_list.append(
                torch.tensor([[past_len]], dtype=torch.long, device=self.device)
            )

        # Stack into batches: [batch_size, 1]
        batched_input_ids = torch.cat(input_ids_list, dim=0)
        batched_position_ids = torch.cat(position_ids_list, dim=0)

        # For batched decode, we need to handle KV cache per-request
        # This is a simplified approach - process individually but prepare for future optimization
        # TODO: Implement true batched attention with separate KV caches

        outputs = []
        for i, request in enumerate(requests):
            req_state = self.active_requests[request.request_id]
            past_key_values = req_state["past_key_values"]

            # Single forward pass per request (still needed due to separate KV caches)
            model_output, new_kv = self.model(
                input_ids=input_ids_list[i],
                position_ids=position_ids_list[i],
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Update state
            req_state["past_key_values"] = new_kv
            req_state["seq_len"] += 1
            self.stats["tokens_generated"] += 1

            # Extract last token logits (handle both 2D and 3D output)
            logits = model_output.logits
            if logits.dim() == 3:
                logits = logits[:, -1, :]  # [batch, seq, vocab] -> [batch, vocab]

            outputs.append(WorkerOutput(
                request_id=request.request_id,
                logits=logits,
                finished=False,
            ))

        return outputs

    def _execute_single(self, request: WorkerRequest) -> WorkerOutput:
        """Execute single request."""
        request_id = request.request_id
        input_ids = request.input_ids.to(self.device)

        # Get cached state
        req_state = self.active_requests.get(request_id)
        if req_state is None:
            raise RuntimeError(f"Request {request_id} not allocated")

        past_key_values = req_state["past_key_values"]

        # Position IDs
        if request.position_ids is not None:
            position_ids = request.position_ids.to(self.device)
        else:
            past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long, device=self.device
            ).unsqueeze(0)

        # Forward pass
        model_output, new_kv = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Update state
        req_state["past_key_values"] = new_kv
        req_state["seq_len"] += input_ids.shape[1]

        # Update stats
        if request.is_prefill:
            self.stats["prefill_tokens"] += input_ids.shape[1]
        else:
            self.stats["tokens_generated"] += 1

        # Extract last token logits (handle both 2D and 3D output)
        logits = model_output.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [batch, seq, vocab] -> [batch, vocab]

        return WorkerOutput(
            request_id=request_id,
            logits=logits,
            finished=False,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        cache_stats = self.cache.get_stats() if self.cache else {}

        return {
            **self.stats,
            "active_requests": len(self.active_requests),
            "cache": cache_stats,
            "device": str(self.device),
            "model_memory_mb": self.model.get_memory_footprint()["total_mb"] if self.model else 0,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "prefill_tokens": 0,
        }
