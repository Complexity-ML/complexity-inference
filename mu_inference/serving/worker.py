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

        Currently processes one request at a time (batching coming soon).

        Args:
            requests: List of WorkerRequest

        Returns:
            List of WorkerOutput
        """
        outputs = []

        for request in requests:
            try:
                output = self._execute_single(request)
                outputs.append(output)
            except Exception as e:
                logger.error(f"Error processing {request.request_id}: {e}")
                outputs.append(WorkerOutput(
                    request_id=request.request_id,
                    logits=torch.zeros(1, self.model.config.vocab_size),
                    finished=True,
                    error=str(e),
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

        return WorkerOutput(
            request_id=request_id,
            logits=model_output.logits,
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
