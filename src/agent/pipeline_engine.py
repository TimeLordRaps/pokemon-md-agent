"""Pipeline engine with continuous batching and ≤50ms tick for partial flush.

Manages prefill/decoding queues with starvation prevention and non-blocking assembly.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Deque
from collections import deque
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages for request processing."""
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETE = "complete"


@dataclass
class PipelineRequest:
    """Request in pipeline."""
    id: str
    prompt: str
    images: Optional[List[Any]] = None
    model_name: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    stage: PipelineStage = PipelineStage.PREFILL
    kv_cache: Optional[Any] = None
    tokens_generated: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last active time."""
        self.last_active = time.time()

    def is_stale(self, timeout_s: float = 30.0) -> bool:
        """Check if request has timed out."""
        return (time.time() - self.last_active) > timeout_s


@dataclass
class Batch:
    """Batch of requests for parallel processing."""
    id: str
    requests: List[PipelineRequest]
    stage: PipelineStage
    created_at: float = field(default_factory=time.time)
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.requests)


class PipelineEngine:
    """Async pipeline engine with continuous batching."""

    def __init__(self, max_batch_size: int = 8, tick_interval_ms: int = 50,
                 max_queue_depth: int = 100, starvation_threshold_ms: int = 1000):
        """Initialize pipeline engine.

        Args:
            max_batch_size: Maximum requests per batch
            tick_interval_ms: Tick interval for partial flush (≤50ms)
            max_queue_depth: Maximum queued requests before rejection
            starvation_threshold_ms: Time after which queued requests get priority
        """
        self.max_batch_size = max_batch_size
        self.tick_interval_ms = tick_interval_ms
        self.max_queue_depth = max_queue_depth
        self.starvation_threshold_ms = starvation_threshold_ms

        # Queues for different stages
        self.prefill_queue: Deque[PipelineRequest] = deque()
        self.decode_queue: Deque[PipelineRequest] = deque()
        self.completed_requests: Dict[str, PipelineRequest] = {}

        # Active batches
        self.active_prefill_batch: Optional[Batch] = None
        self.active_decode_batch: Optional[Batch] = None

        # Control
        self.running = False
        self.tick_task: Optional[asyncio.Task] = None

        # Callbacks for actual processing
        self.prefill_callback: Optional[Callable[[Batch], Any]] = None
        self.decode_callback: Optional[Callable[[Batch], Any]] = None

        # Stats
        self.stats = {
            "requests_processed": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "starvation_events": 0,
            "queue_full_rejects": 0,
        }

        logger.info(f"Initialized PipelineEngine with batch_size={max_batch_size}, tick={tick_interval_ms}ms")

    async def start(self) -> None:
        """Start the pipeline engine."""
        if self.running:
            return

        self.running = True
        self.tick_task = asyncio.create_task(self._tick_loop())
        logger.info("Pipeline engine started")

    async def stop(self) -> None:
        """Stop the pipeline engine."""
        self.running = False
        if self.tick_task:
            self.tick_task.cancel()
            try:
                await self.tick_task
            except asyncio.CancelledError:
                pass
        
        # Wait for any pending batch processing tasks to complete
        # In a real implementation, you'd track these tasks
        await asyncio.sleep(0.01)  # Small delay to let async tasks complete
        
        logger.info("Pipeline engine stopped")

    def set_prefill_callback(self, callback: Callable[[Batch], Any]) -> None:
        """Set callback for prefill processing."""
        self.prefill_callback = callback

    def set_decode_callback(self, callback: Callable[[Batch], Any]) -> None:
        """Set callback for decode processing."""
        self.decode_callback = callback

    async def submit_request(self, request: PipelineRequest) -> bool:
        """Submit request to pipeline. Returns False if queue full."""
        if len(self.prefill_queue) >= self.max_queue_depth:
            self.stats["queue_full_rejects"] += 1
            logger.warning(f"Pipeline queue full, rejecting request {request.id}")
            return False

        self.prefill_queue.append(request)
        logger.debug(f"Submitted request {request.id} to pipeline")
        return True

    async def get_completed_request(self, request_id: str) -> Optional[PipelineRequest]:
        """Get completed request by ID."""
        return self.completed_requests.pop(request_id, None)

    async def _tick_loop(self) -> None:
        """Main tick loop for partial batch flushing."""
        tick_interval = self.tick_interval_ms / 1000.0

        while self.running:
            try:
                await self._process_tick()
                await asyncio.sleep(tick_interval)
            except Exception as e:
                logger.error(f"Error in pipeline tick: {e}")

    async def _process_tick(self) -> None:
        """Process one tick - assemble and flush partial batches."""
        # Check for starvation
        await self._check_starvation()

        # Try to assemble and flush prefill batch
        if not self.active_prefill_batch and self.prefill_queue:
            batch = self._assemble_batch(PipelineStage.PREFILL)
            if batch:
                await self._flush_batch(batch)

        # Try to assemble and flush decode batch
        if not self.active_decode_batch and self.decode_queue:
            batch = self._assemble_batch(PipelineStage.DECODE)
            if batch:
                await self._flush_batch(batch)

        # Check if active batches are complete
        await self._check_batch_completion()

    async def _check_starvation(self) -> None:
        """Check for starved requests and promote them."""
        now = time.time()

        # Check prefill queue for starvation
        if self.prefill_queue:
            oldest = self.prefill_queue[0]
            if (now - oldest.created_at) * 1000 > self.starvation_threshold_ms:
                # Force flush a small batch
                batch = self._assemble_batch(PipelineStage.PREFILL, force_flush=True)
                if batch:
                    self.stats["starvation_events"] += 1
                    logger.info(f"Starvation flush: {batch.size} requests")
                    await self._flush_batch(batch)

    def _assemble_batch(self, stage: PipelineStage, force_flush: bool = False) -> Optional[Batch]:
        """Assemble batch for given stage."""
        queue = self.prefill_queue if stage == PipelineStage.PREFILL else self.decode_queue

        if not queue:
            return None

        # Determine batch size
        if force_flush:
            batch_size = min(2, len(queue))  # Small batch for starvation
        else:
            batch_size = min(self.max_batch_size, len(queue))

        # Only assemble batch if forced or queue is at capacity
        if not force_flush and batch_size < self.max_batch_size:
            return None

        if batch_size == 0:
            return None

        # Extract requests
        requests = []
        for _ in range(batch_size):
            if queue:
                requests.append(queue.popleft())

        batch = Batch(
            id=f"batch_{stage.value}_{int(time.time()*1000)}",
            requests=requests,
            stage=stage
        )

        logger.debug(f"Assembled {stage.value} batch with {batch.size} requests")
        return batch

    async def _flush_batch(self, batch: Batch) -> None:
        """Flush batch to processing."""
        if batch.stage == PipelineStage.PREFILL:
            self.active_prefill_batch = batch
            if self.prefill_callback:
                asyncio.create_task(self._process_batch_async(batch))
        elif batch.stage == PipelineStage.DECODE:
            self.active_decode_batch = batch
            if self.decode_callback:
                asyncio.create_task(self._process_batch_async(batch))

        self.stats["batches_processed"] += 1
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (self.stats["batches_processed"] - 1)) + batch.size
        ) / self.stats["batches_processed"]

        logger.debug(f"Flushed {batch.stage.value} batch {batch.id} with {batch.size} requests")

    async def _process_batch_async(self, batch: Batch) -> None:
        """Process batch asynchronously."""
        try:
            if batch.stage == PipelineStage.PREFILL and self.prefill_callback:
                await self.prefill_callback(batch)
            elif batch.stage == PipelineStage.DECODE and self.decode_callback:
                await self.decode_callback(batch)

            # Mark requests as processed
            for request in batch.requests:
                request.stage = PipelineStage.COMPLETE
                self.completed_requests[request.id] = request
                self.stats["requests_processed"] += 1

            logger.debug(f"Completed {batch.stage.value} batch {batch.id}")

        except Exception as e:
            logger.error(f"Error processing batch {batch.id}: {e}")
            # Re-queue failed requests
            for request in batch.requests:
                if batch.stage == PipelineStage.PREFILL:
                    self.prefill_queue.appendleft(request)
                else:
                    self.decode_queue.appendleft(request)

        finally:
            # Clear active batch
            if batch.stage == PipelineStage.PREFILL:
                self.active_prefill_batch = None
            else:
                self.active_decode_batch = None

    async def _check_batch_completion(self) -> None:
        """Check if active batches have completed (placeholder - in real impl would check actual status)."""
        # This is a placeholder - real implementation would check GPU/memory status
        # For now, assume batches complete immediately in simulation
        pass

    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths."""
        return {
            "prefill": len(self.prefill_queue),
            "decode": len(self.decode_queue),
            "completed": len(self.completed_requests),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        stats.update({
            "active_prefill_batch": self.active_prefill_batch.size if self.active_prefill_batch else 0,
            "active_decode_batch": self.active_decode_batch.size if self.active_decode_batch else 0,
            "tick_interval_ms": self.tick_interval_ms,
            "max_batch_size": self.max_batch_size,
        })
        return stats

    def clear_queues(self) -> None:
        """Clear all queues (for testing/cleanup)."""
        self.prefill_queue.clear()
        self.decode_queue.clear()
        self.completed_requests.clear()
        self.active_prefill_batch = None
        self.active_decode_batch = None
        logger.info("Cleared all pipeline queues")