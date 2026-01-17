"""
Mu Engine
=========

Main inference engine orchestrating all components.

Features:
- Multi-user request handling
- Automatic scheduling
- Streaming generation
- Mu dynamics integration
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, List, Dict, AsyncIterator, Callable, Any
from dataclasses import dataclass, field

import torch

from mu_inference.core.config import (
    EngineConfig,
    MuConfig,
    SamplingParams,
    ModelConfig,
)
from mu_inference.core.scheduler import MuScheduler, Request, RequestStatus
from mu_inference.serving.worker import MuWorker, WorkerRequest, WorkerOutput
from mu_inference.utils.sampling import sample_token, prepare_logits
from mu_inference.utils.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from generation."""
    request_id: str
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None  # "stop", "length", "error"
    usage: Optional[Dict[str, int]] = None


@dataclass
class StreamOutput:
    """Streaming output chunk."""
    request_id: str
    text: str
    token_id: int
    finished: bool
    finish_reason: Optional[str] = None


class MuEngine:
    """
    Main Mu inference engine.

    Orchestrates model loading, scheduling, and generation.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or EngineConfig()
        self.model_path = model_path

        # Components
        self.worker: Optional[MuWorker] = None
        self.scheduler: Optional[MuScheduler] = None
        self.tokenizer = None

        # Request tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}

        # State
        self._initialized = False
        self._running = False

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "start_time": None,
        }

    async def initialize(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the engine.

        Args:
            model_path: Path to model (overrides config)
        """
        if self._initialized:
            return

        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError("model_path is required")

        logger.info(f"Initializing Mu Engine with model: {model_path}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = load_tokenizer(model_path)

        # Initialize worker
        logger.info("Initializing worker...")
        self.worker = MuWorker(
            config=self.config,
            device=self.config.device,
        )
        self.worker.load_model(model_path)

        # Initialize scheduler
        logger.info("Initializing scheduler...")
        self.scheduler = MuScheduler(
            config=self.config.scheduler,
            mu_config=self.config.mu,
        )

        self._initialized = True
        self.stats["start_time"] = time.time()

        logger.info("Mu Engine initialized successfully")

    def generate_sync(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> GenerationOutput:
        """
        Synchronous generation.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Optional request ID

        Returns:
            GenerationOutput
        """
        return asyncio.get_event_loop().run_until_complete(
            self.generate(prompt, sampling_params, request_id)
        )

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> GenerationOutput:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Optional request ID

        Returns:
            GenerationOutput
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = sampling_params or SamplingParams()
        request_id = request_id or str(uuid.uuid4())

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]

        # Allocate
        if not self.worker.allocate_request(request_id):
            raise RuntimeError("Failed to allocate request - at capacity")

        try:
            # Generate tokens
            generated_ids = []
            finish_reason = None

            for step in range(sampling_params.max_tokens):
                # Prepare input
                if step == 0:
                    # Prefill: full prompt
                    worker_input = WorkerRequest(
                        request_id=request_id,
                        input_ids=input_ids,
                        is_prefill=True,
                    )
                else:
                    # Decode: last token only
                    last_token = torch.tensor([[generated_ids[-1]]])
                    worker_input = WorkerRequest(
                        request_id=request_id,
                        input_ids=last_token,
                        is_prefill=False,
                    )

                # Execute
                outputs = self.worker.execute([worker_input])
                output = outputs[0]

                if output.error:
                    finish_reason = "error"
                    break

                # Sample next token
                logits = output.logits

                # Apply sampling
                logits = prepare_logits(
                    logits,
                    temperature=sampling_params.temperature,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                    repetition_penalty=sampling_params.repetition_penalty,
                    past_tokens=input_ids[0].tolist() + generated_ids,
                )

                next_token = sample_token(logits)
                generated_ids.append(next_token)

                # Check stop conditions
                if next_token == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break

                if sampling_params.stop_token_ids and next_token in sampling_params.stop_token_ids:
                    finish_reason = "stop"
                    break

            if finish_reason is None:
                finish_reason = "length"

            # Decode output
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_tokens_generated"] += len(generated_ids)

            return GenerationOutput(
                request_id=request_id,
                text=output_text,
                token_ids=generated_ids,
                finished=True,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_len,
                    "completion_tokens": len(generated_ids),
                    "total_tokens": prompt_len + len(generated_ids),
                },
            )

        finally:
            # Free resources
            self.worker.free_request(request_id)

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[StreamOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Optional request ID

        Yields:
            StreamOutput for each token
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = sampling_params or SamplingParams()
        request_id = request_id or str(uuid.uuid4())

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Allocate
        if not self.worker.allocate_request(request_id):
            raise RuntimeError("Failed to allocate request - at capacity")

        try:
            generated_ids = []
            finish_reason = None

            for step in range(sampling_params.max_tokens):
                # Prepare input
                if step == 0:
                    worker_input = WorkerRequest(
                        request_id=request_id,
                        input_ids=input_ids,
                        is_prefill=True,
                    )
                else:
                    last_token = torch.tensor([[generated_ids[-1]]])
                    worker_input = WorkerRequest(
                        request_id=request_id,
                        input_ids=last_token,
                        is_prefill=False,
                    )

                # Execute
                outputs = self.worker.execute([worker_input])
                output = outputs[0]

                if output.error:
                    yield StreamOutput(
                        request_id=request_id,
                        text="",
                        token_id=-1,
                        finished=True,
                        finish_reason="error",
                    )
                    return

                # Sample
                logits = prepare_logits(
                    output.logits,
                    temperature=sampling_params.temperature,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                    repetition_penalty=sampling_params.repetition_penalty,
                    past_tokens=input_ids[0].tolist() + generated_ids,
                )

                next_token = sample_token(logits)
                generated_ids.append(next_token)

                # Decode token
                token_text = self.tokenizer.decode([next_token])

                # Check stop
                finished = False
                if next_token == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    finished = True
                elif sampling_params.stop_token_ids and next_token in sampling_params.stop_token_ids:
                    finish_reason = "stop"
                    finished = True

                yield StreamOutput(
                    request_id=request_id,
                    text=token_text,
                    token_id=next_token,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break

                # Yield control
                await asyncio.sleep(0)

            # Final token if not already finished
            if finish_reason is None:
                yield StreamOutput(
                    request_id=request_id,
                    text="",
                    token_id=-1,
                    finished=True,
                    finish_reason="length",
                )

            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_tokens_generated"] += len(generated_ids)

        finally:
            self.worker.free_request(request_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0

        worker_stats = self.worker.get_stats() if self.worker else {}

        return {
            "total_requests": self.stats["total_requests"],
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "uptime_seconds": uptime,
            "tokens_per_second": self.stats["total_tokens_generated"] / uptime if uptime > 0 else 0,
            "worker": worker_stats,
            "mu_config": {
                "enabled": self.config.mu.enabled,
                "mu": self.config.mu.mu,
                "clamp_min": self.config.mu.clamp_min,
                "clamp_max": self.config.mu.clamp_max,
            },
        }

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("Shutting down Mu Engine...")

        if self.worker:
            # Clear cache
            if self.worker.cache:
                self.worker.cache.clear()

        self._initialized = False
        self._running = False

        logger.info("Mu Engine shutdown complete")
