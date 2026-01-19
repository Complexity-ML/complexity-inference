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


# =============================================================================
# Continuous Batching Engine
# =============================================================================

@dataclass
class BatchedRequest:
    """Internal request tracking for continuous batching."""
    request_id: str
    prompt: str
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    generated_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    is_prefill_done: bool = False
    result_future: Optional[asyncio.Future] = None
    stream_queue: Optional[asyncio.Queue] = None
    prompt_len: int = 0


class MuBatchEngine:
    """
    Continuous Batching Engine for high-throughput inference.

    Unlike MuEngine which processes one request at a time, MuBatchEngine:
    - Collects incoming requests over a time window
    - Batches decode steps together for GPU efficiency
    - Allows new requests to enter while others are generating
    - Requests can finish and exit independently

    This is the recommended engine for production serving.
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
        self.tokenizer = None

        # Request management
        self._pending_requests: Dict[str, BatchedRequest] = {}  # Waiting for prefill
        self._active_requests: Dict[str, BatchedRequest] = {}   # In decode phase
        self._request_lock = asyncio.Lock()

        # Background loop
        self._batch_loop_task: Optional[asyncio.Task] = None
        self._running = False
        self._initialized = False

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "start_time": None,
        }

    async def initialize(self, model_path: Optional[str] = None) -> None:
        """Initialize the batch engine."""
        if self._initialized:
            return

        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError("model_path is required")

        logger.info(f"Initializing Mu Batch Engine with model: {model_path}")

        # Load tokenizer
        self.tokenizer = load_tokenizer(model_path)

        # Initialize worker
        self.worker = MuWorker(
            config=self.config,
            device=self.config.device,
        )
        self.worker.load_model(model_path)

        self._initialized = True
        self.stats["start_time"] = time.time()

        # Start batch processing loop
        self._running = True
        self._batch_loop_task = asyncio.create_task(self._batch_loop())

        logger.info("Mu Batch Engine initialized with continuous batching enabled")

    async def _batch_loop(self) -> None:
        """
        Main continuous batching loop.

        Runs in background, collecting and processing requests.
        """
        batch_wait_ms = self.config.scheduler.batch_wait_time_ms
        min_batch_size = self.config.scheduler.min_batch_size
        max_batch_size = self.config.scheduler.max_batch_size

        while self._running:
            try:
                # Wait for requests or timeout
                start_wait = time.time()
                while (time.time() - start_wait) * 1000 < batch_wait_ms:
                    async with self._request_lock:
                        total_requests = len(self._pending_requests) + len(self._active_requests)

                    if total_requests >= min_batch_size:
                        break

                    await asyncio.sleep(0.001)  # 1ms check interval

                # Process one batch step
                await self._process_batch_step()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch loop error: {e}")
                await asyncio.sleep(0.01)

    async def _process_batch_step(self) -> None:
        """Process one step of batch generation."""
        async with self._request_lock:
            if not self._pending_requests and not self._active_requests:
                return

            worker_requests = []
            request_mapping: List[BatchedRequest] = []

            # 1. Process pending requests (prefill)
            for req_id, req in list(self._pending_requests.items()):
                if len(worker_requests) >= self.config.scheduler.max_batch_size:
                    break

                # Allocate if not done
                if not self.worker.allocate_request(req_id):
                    # At capacity, will retry next step
                    continue

                worker_requests.append(WorkerRequest(
                    request_id=req_id,
                    input_ids=req.input_ids,
                    is_prefill=True,
                ))
                request_mapping.append(req)

                # Move to active
                self._active_requests[req_id] = req
                del self._pending_requests[req_id]

            # 2. Process active requests (decode)
            for req_id, req in list(self._active_requests.items()):
                if req_id in [r.request_id for r in request_mapping]:
                    continue  # Already added as prefill

                if len(worker_requests) >= self.config.scheduler.max_batch_size:
                    break

                if not req.is_prefill_done:
                    continue  # Will be processed after prefill completes

                # Decode: use last generated token
                if req.generated_ids:
                    last_token = torch.tensor([[req.generated_ids[-1]]])
                    worker_requests.append(WorkerRequest(
                        request_id=req_id,
                        input_ids=last_token,
                        is_prefill=False,
                    ))
                    request_mapping.append(req)

        if not worker_requests:
            return

        # Execute batch
        try:
            outputs = self.worker.execute(worker_requests)
        except Exception as e:
            logger.error(f"Worker execution error: {e}")
            return

        # Process outputs
        async with self._request_lock:
            for i, output in enumerate(outputs):
                req = request_mapping[i]

                if output.error:
                    req.finish_reason = "error"
                    await self._finish_request(req)
                    continue

                # Sample next token
                logits = prepare_logits(
                    output.logits,
                    temperature=req.sampling_params.temperature,
                    top_k=req.sampling_params.top_k,
                    top_p=req.sampling_params.top_p,
                    repetition_penalty=req.sampling_params.repetition_penalty,
                    past_tokens=req.input_ids[0].tolist() + req.generated_ids,
                )

                next_token = sample_token(logits)
                req.generated_ids.append(next_token)
                req.is_prefill_done = True

                # Stream output if requested
                if req.stream_queue is not None:
                    token_text = self.tokenizer.decode([next_token])
                    await req.stream_queue.put(StreamOutput(
                        request_id=req.request_id,
                        text=token_text,
                        token_id=next_token,
                        finished=False,
                    ))

                # Check stop conditions
                finished = False
                if next_token == self.tokenizer.eos_token_id:
                    req.finish_reason = "stop"
                    finished = True
                elif req.sampling_params.stop_token_ids and next_token in req.sampling_params.stop_token_ids:
                    req.finish_reason = "stop"
                    finished = True
                elif len(req.generated_ids) >= req.sampling_params.max_tokens:
                    req.finish_reason = "length"
                    finished = True

                if finished:
                    await self._finish_request(req)

            # Update stats
            self.stats["batches_processed"] += 1
            batch_size = len(outputs)
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["batches_processed"] - 1) + batch_size)
                / self.stats["batches_processed"]
            )

    async def _finish_request(self, req: BatchedRequest) -> None:
        """Complete a request and notify waiters."""
        req_id = req.request_id

        # Remove from active
        self._active_requests.pop(req_id, None)

        # Free worker resources
        self.worker.free_request(req_id)

        # Update stats
        self.stats["total_requests"] += 1
        self.stats["total_tokens_generated"] += len(req.generated_ids)

        # Decode output
        output_text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)

        # Notify future if set
        if req.result_future is not None and not req.result_future.done():
            req.result_future.set_result(GenerationOutput(
                request_id=req_id,
                text=output_text,
                token_ids=req.generated_ids,
                finished=True,
                finish_reason=req.finish_reason,
                usage={
                    "prompt_tokens": req.prompt_len,
                    "completion_tokens": len(req.generated_ids),
                    "total_tokens": req.prompt_len + len(req.generated_ids),
                },
            ))

        # Notify stream if set
        if req.stream_queue is not None:
            await req.stream_queue.put(StreamOutput(
                request_id=req_id,
                text="",
                token_id=-1,
                finished=True,
                finish_reason=req.finish_reason,
            ))

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> GenerationOutput:
        """
        Generate completion (non-streaming).

        Request is added to batch and processed with others.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = sampling_params or SamplingParams()
        request_id = request_id or str(uuid.uuid4())

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]

        # Create result future
        loop = asyncio.get_event_loop()
        result_future = loop.create_future()

        # Create request
        req = BatchedRequest(
            request_id=request_id,
            prompt=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            result_future=result_future,
        )

        # Add to pending
        async with self._request_lock:
            self._pending_requests[request_id] = req

        # Wait for result
        return await result_future

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[StreamOutput]:
        """
        Stream generation token by token.

        Request is batched with others for efficient processing.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = sampling_params or SamplingParams()
        request_id = request_id or str(uuid.uuid4())

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]

        # Create stream queue
        stream_queue: asyncio.Queue = asyncio.Queue()

        # Create request
        req = BatchedRequest(
            request_id=request_id,
            prompt=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            stream_queue=stream_queue,
        )

        # Add to pending
        async with self._request_lock:
            self._pending_requests[request_id] = req

        # Yield from queue until finished
        while True:
            output = await stream_queue.get()
            yield output
            if output.finished:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        worker_stats = self.worker.get_stats() if self.worker else {}

        return {
            "engine_type": "continuous_batching",
            "total_requests": self.stats["total_requests"],
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "batches_processed": self.stats["batches_processed"],
            "avg_batch_size": round(self.stats["avg_batch_size"], 2),
            "uptime_seconds": uptime,
            "tokens_per_second": self.stats["total_tokens_generated"] / uptime if uptime > 0 else 0,
            "pending_requests": len(self._pending_requests),
            "active_requests": len(self._active_requests),
            "worker": worker_stats,
            "continuous_batching": {
                "enabled": self.config.scheduler.enable_continuous_batching,
                "batch_wait_time_ms": self.config.scheduler.batch_wait_time_ms,
                "max_batch_size": self.config.scheduler.max_batch_size,
            },
        }

    async def shutdown(self) -> None:
        """Shutdown the batch engine."""
        logger.info("Shutting down Mu Batch Engine...")

        self._running = False

        # Cancel batch loop
        if self._batch_loop_task:
            self._batch_loop_task.cancel()
            try:
                await self._batch_loop_task
            except asyncio.CancelledError:
                pass

        # Clear pending/active requests
        async with self._request_lock:
            for req in self._pending_requests.values():
                if req.result_future and not req.result_future.done():
                    req.result_future.set_exception(RuntimeError("Engine shutdown"))
            for req in self._active_requests.values():
                if req.result_future and not req.result_future.done():
                    req.result_future.set_exception(RuntimeError("Engine shutdown"))
                self.worker.free_request(req.request_id)

            self._pending_requests.clear()
            self._active_requests.clear()

        # Clear worker cache
        if self.worker and self.worker.cache:
            self.worker.cache.clear()

        self._initialized = False

        logger.info("Mu Batch Engine shutdown complete")
