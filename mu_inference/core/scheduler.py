"""
Mu Request Scheduler
====================

Multi-user request scheduling with Mu-guided prioritization.

Supports:
- FCFS (First-Come-First-Served)
- Priority scheduling with Mu-smoothed priorities
- Fair scheduling
- Preemption for long-running requests
- Chunked prefill for large prompts
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable, Any
from collections import deque
import heapq

from mu_inference.core.mu import MuDynamics, soft_clamp
from mu_inference.core.config import SchedulerConfig, SamplingParams


class RequestStatus(str, Enum):
    """Request lifecycle status."""
    PENDING = "pending"         # Waiting in queue
    PREFILL = "prefill"         # Processing prompt
    DECODE = "decode"           # Generating tokens
    PREEMPTED = "preempted"     # Preempted, waiting to resume
    FINISHED = "finished"       # Completed
    FAILED = "failed"           # Error occurred


@dataclass
class Request:
    """
    Single inference request.

    Tracks all state needed for generation.
    """

    # Identity
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    # Input
    prompt: str = ""
    prompt_token_ids: List[int] = field(default_factory=list)

    # Output
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""

    # Sampling
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # Status
    status: RequestStatus = RequestStatus.PENDING

    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    # Priority (lower = higher priority)
    priority: float = 0.0

    # Position tracking
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    max_tokens: int = 512

    # Preemption
    num_preemptions: int = 0

    # Callback
    on_token: Optional[Callable[[int, str], None]] = None
    on_finish: Optional[Callable[["Request"], None]] = None

    def __lt__(self, other: "Request") -> bool:
        """For priority queue ordering."""
        return self.priority < other.priority

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.FAILED)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def latency(self) -> Optional[float]:
        if self.finish_time and self.start_time:
            return self.finish_time - self.start_time
        return None

    @property
    def tokens_per_second(self) -> Optional[float]:
        lat = self.latency
        if lat and lat > 0:
            return self.num_generated_tokens / lat
        return None


@dataclass
class SchedulerOutput:
    """Output from scheduler for one step."""

    # Requests to prefill (process prompt)
    prefill_requests: List[Request] = field(default_factory=list)

    # Requests to decode (generate next token)
    decode_requests: List[Request] = field(default_factory=list)

    # Requests that were preempted
    preempted_requests: List[Request] = field(default_factory=list)

    # Total tokens in this batch
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class MuScheduler:
    """
    Request scheduler with Mu-guided dynamics.

    Features:
    - Multi-policy scheduling (FCFS, priority, fair)
    - Mu-smoothed priority updates (no sudden jumps)
    - Preemption support
    - Chunked prefill for large prompts
    - Adaptive batch sizing via Mu dynamics
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

        # Request queues
        self.waiting: deque[Request] = deque()  # FCFS queue
        self.running: Dict[str, Request] = {}   # Currently processing
        self.finished: Dict[str, Request] = {}  # Completed (for retrieval)

        # Priority queue (for priority scheduling)
        self.priority_queue: List[Request] = []

        # Mu dynamics for priority smoothing
        self.priority_dynamics = MuDynamics(
            mu=0.0,
            clamp_min=-10.0,
            clamp_max=10.0,
            ema_alpha=0.3,
        )

        # Mu dynamics for batch size adaptation
        self.batch_dynamics = MuDynamics(
            mu=config.max_batch_size / 2,
            clamp_min=1,
            clamp_max=config.max_batch_size,
            ema_alpha=0.2,
        )

        # Stats
        self._num_processed = 0
        self._total_tokens = 0

    def add_request(self, request: Request) -> str:
        """
        Add a new request to the scheduler.

        Returns:
            Request ID
        """
        request.status = RequestStatus.PENDING
        request.arrival_time = time.time()

        if self.config.policy == "priority":
            # Smooth priority with Mu dynamics
            request.priority = self.priority_dynamics.update(request.priority)
            heapq.heappush(self.priority_queue, request)
        else:
            self.waiting.append(request)

        return request.request_id

    def schedule(self) -> SchedulerOutput:
        """
        Schedule next batch of requests.

        Returns:
            SchedulerOutput with prefill and decode requests
        """
        output = SchedulerOutput()

        # First, continue decoding running requests
        decode_budget = self.config.max_num_batched_tokens
        for request in list(self.running.values()):
            if request.status == RequestStatus.DECODE:
                output.decode_requests.append(request)
                output.num_decode_tokens += 1
                decode_budget -= 1

        # Check if we can add new requests
        available_slots = self.config.max_batch_size - len(output.decode_requests)
        prefill_budget = min(
            self.config.max_prefill_tokens,
            decode_budget,
        )

        # Get new requests based on policy
        new_requests = self._get_next_requests(available_slots)

        for request in new_requests:
            # Check token budget
            tokens_needed = request.num_prompt_tokens
            if self.config.enable_chunked_prefill:
                tokens_needed = min(tokens_needed, prefill_budget)

            if tokens_needed <= prefill_budget:
                request.status = RequestStatus.PREFILL
                request.start_time = time.time()
                self.running[request.request_id] = request
                output.prefill_requests.append(request)
                output.num_prefill_tokens += tokens_needed
                prefill_budget -= tokens_needed

        # Handle preemption if needed
        if self._should_preempt(output):
            preempted = self._preempt_requests(output)
            output.preempted_requests = preempted

        return output

    def _get_next_requests(self, max_count: int) -> List[Request]:
        """Get next requests based on scheduling policy."""
        requests = []

        if self.config.policy == "priority":
            while self.priority_queue and len(requests) < max_count:
                request = heapq.heappop(self.priority_queue)
                requests.append(request)

        elif self.config.policy == "fair":
            # Round-robin style fair scheduling
            # TODO: Implement fair scheduling
            while self.waiting and len(requests) < max_count:
                requests.append(self.waiting.popleft())

        else:  # FCFS
            while self.waiting and len(requests) < max_count:
                requests.append(self.waiting.popleft())

        return requests

    def _should_preempt(self, output: SchedulerOutput) -> bool:
        """Check if preemption is needed."""
        if not self.config.enable_preemption:
            return False

        # Preempt if we're over token budget
        return output.total_tokens > self.config.max_num_batched_tokens

    def _preempt_requests(self, output: SchedulerOutput) -> List[Request]:
        """Preempt requests to free resources."""
        preempted = []

        # Sort by tokens generated (preempt those with most progress first)
        candidates = sorted(
            output.decode_requests,
            key=lambda r: r.num_generated_tokens,
            reverse=True,
        )

        tokens_to_free = output.total_tokens - self.config.max_num_batched_tokens

        for request in candidates:
            if tokens_to_free <= 0:
                break

            request.status = RequestStatus.PREEMPTED
            request.num_preemptions += 1
            output.decode_requests.remove(request)

            # Re-add to waiting queue
            if self.config.policy == "priority":
                # Boost priority after preemption
                request.priority = soft_clamp(
                    request.priority - 1.0,
                    -10.0, 10.0
                )
                heapq.heappush(self.priority_queue, request)
            else:
                self.waiting.appendleft(request)

            preempted.append(request)
            tokens_to_free -= 1

        return preempted

    def finish_request(self, request_id: str, success: bool = True):
        """Mark a request as finished."""
        request = self.running.pop(request_id, None)
        if request:
            request.status = RequestStatus.FINISHED if success else RequestStatus.FAILED
            request.finish_time = time.time()
            self.finished[request_id] = request
            self._num_processed += 1
            self._total_tokens += request.total_tokens

            # Callback
            if request.on_finish:
                request.on_finish(request)

    def update_request(self, request_id: str, token_id: int, token_text: str):
        """Update a running request with a new token."""
        request = self.running.get(request_id)
        if request:
            request.output_token_ids.append(token_id)
            request.output_text += token_text
            request.num_generated_tokens += 1
            request.status = RequestStatus.DECODE

            # Callback
            if request.on_token:
                request.on_token(token_id, token_text)

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return (
            self.running.get(request_id) or
            self.finished.get(request_id)
        )

    def abort_request(self, request_id: str):
        """Abort a request."""
        # Remove from running
        if request_id in self.running:
            request = self.running.pop(request_id)
            request.status = RequestStatus.FAILED
            self.finished[request_id] = request
            return

        # Remove from waiting
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)

        # Remove from priority queue
        self.priority_queue = [r for r in self.priority_queue if r.request_id != request_id]
        heapq.heapify(self.priority_queue)

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "waiting": len(self.waiting) + len(self.priority_queue),
            "running": len(self.running),
            "finished": len(self.finished),
            "total_processed": self._num_processed,
            "total_tokens": self._total_tokens,
            "policy": self.config.policy,
            "batch_dynamics": self.batch_dynamics.get_stats(),
        }

    def clear_finished(self, max_age: float = 300.0):
        """Clear old finished requests."""
        cutoff = time.time() - max_age
        self.finished = {
            k: v for k, v in self.finished.items()
            if v.finish_time and v.finish_time > cutoff
        }
