"""Async micro-batching inference queue for Qwen3-VL models."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.agent.timebudgets import ROUTER_FLUSH_TICK_MS, ROUTER_MAX_WALL_S

logger = logging.getLogger(__name__)


class HybridFuture:
    """Future compatible with asyncio await and blocking consumption."""

    def __init__(
        self,
        poller: Optional[Callable[[], None]] = None,
        poll_interval: float = 0.01,
    ) -> None:
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._done = False
        self._waiters: List[asyncio.Future] = []
        self._event = threading.Event()
        self._poller = poller
        self._poll_interval = max(poll_interval, 1e-3)

    def set_result(self, value: Any) -> None:
        if self._done:
            return
        self._result = value
        self._done = True
        self._event.set()
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if self._done:
            return
        self._exception = exc
        self._done = True
        self._event.set()
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_exception(exc)

    def result(self, timeout: Optional[float] = None) -> Any:
        if self._done:
            if self._exception:
                raise self._exception
            return self._result

        start_time = time.time()
        while not self._done:
            if self._poller:
                try:
                    self._poller()
                except Exception as poll_exc:  # pragma: no cover - defensive path
                    self.set_exception(poll_exc)
                    break

            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    break
                wait_slice = min(self._poll_interval, max(remaining, 0))
            else:
                wait_slice = self._poll_interval

            self._event.wait(wait_slice)

        if not self._done:
            raise RuntimeError("Result not ready")
        if self._exception:
            raise self._exception
        return self._result

    def done(self) -> bool:
        return self._done

    def __await__(self):
        if self._done:
            if self._exception:
                raise self._exception
            return self._result

        loop = asyncio.get_event_loop()
        waiter = loop.create_future()
        self._waiters.append(waiter)
        result = yield from waiter.__await__()
        return result


@dataclass
class PendingQuery:
    """Represents a queued inference request."""

    query: Any
    future: HybridFuture
    timestamp: float
    metadata: Dict[str, Any]
    budget_remaining_s: Optional[float] = None  # Seconds of budget remaining for this waiter


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""

    total_batches_processed: int = 0
    total_queries_processed: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0
    avg_throughput_inferences_per_sec: float = 0.0


class InferenceTimeoutError(Exception):
    """Raised when inference exceeds the maximum wall time."""

    def __init__(self, batch_size: int, timeout_s: float):
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        super().__init__(f"Inference timed out after {timeout_s}s for batch of {batch_size} queries")


class InferenceQueue:
    """Accumulates inference queries for batched processing to amortize GPU setup costs.

    Features:
        - Async micro-batching with per-query metadata awareness
        - Optional warm-up batches prior to servicing live traffic
        - Tracing hook for external instrumentation (perf counters, logging)
        - Bounded queue with timestamps per item
        - Partial-flush policy: flush when batch full, oldest item age ≥ ROUTER_FLUSH_TICK_MS,
          or any waiter has ≤2s budget left
        - Timeout protection with structured errors
        - Partial result delivery on single request failures
    """

    def __init__(
        self,
        batch_size: int = 4,
        timeout_ms: int = 50,
        micro_batch_size: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
        warmup_batches: int = 0,
        trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.micro_batch_size = micro_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.trace_hook = trace_hook
        self._clock = clock

        self.pending_queries: List[PendingQuery] = []
        self.last_batch_time = self._clock()
        self.metrics = BatchMetrics()
        self._total_processing_time = 0.0

        self._timeout_task: Optional[asyncio.Task] = None
        self._batch_infer_func: Optional[Callable[..., Any]] = None
        self._supports_metadata: Optional[bool] = None
        self._last_callable_id: Optional[int] = None
        self._warmup_remaining = max(warmup_batches, 0)

    def add_query_async(
        self,
        query: Any,
        batch_infer_func: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
        budget_remaining_s: Optional[float] = None,
    ) -> HybridFuture:
        """Add query to batch queue and return future for result.

        Args:
            query: Inference query to process
            batch_infer_func: Function to call for batched inference
            metadata: Optional metadata dict for the query
            budget_remaining_s: Optional time budget remaining for this waiter (seconds)
        """
        poll_interval = max(self.timeout_ms / 1000.0 / 4.0, 0.005)
        future = HybridFuture(poller=self._check_and_process_batch, poll_interval=poll_interval)
        pending = PendingQuery(
            query=query,
            future=future,
            timestamp=self._clock(),
            metadata=metadata or {},
            budget_remaining_s=budget_remaining_s,
        )
        self.pending_queries.append(pending)

        self._batch_infer_func = batch_infer_func
        self._check_and_process_batch()

        try:
            loop = asyncio.get_running_loop()
            if self._timeout_task is None or self._timeout_task.done():
                self._timeout_task = loop.create_task(self._timeout_checker())
        except RuntimeError:
            # No running event loop (expected for sync tests)
            pass

        return future

    def add_query(
        self,
        query: Any,
        batch_infer_func: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
        budget_remaining_s: Optional[float] = None,
    ) -> Any:
        """Synchronous helper that blocks until result is ready.

        Args:
            query: Inference query to process
            batch_infer_func: Function to call for batched inference
            metadata: Optional metadata dict for the query
            budget_remaining_s: Optional time budget remaining for this waiter (seconds)
        """
        future = self.add_query_async(query, batch_infer_func, metadata=metadata, budget_remaining_s=budget_remaining_s)
        while not future.done():
            self._check_and_process_batch()
            if not future.done():
                time.sleep(self.timeout_ms / 1000.0)
        return future.result()

    async def _timeout_checker(self) -> None:
        """Background task that checks for timeouts and processes batches."""
        while self.pending_queries:
            await asyncio.sleep(self.timeout_ms / 1000.0)
            self._check_and_process_batch()

    def _check_and_process_batch(self) -> None:
        """Check if batch should be processed and execute if ready."""
        if not self.pending_queries or self._batch_infer_func is None:
            return

        current_time = self._clock()
        time_since_last_batch = (current_time - self.last_batch_time) * 1000
        oldest_query_time = min(p.timestamp for p in self.pending_queries)
        time_since_oldest_query = (current_time - oldest_query_time) * 1000

        # Check if any waiter has ≤2s budget left
        budget_trigger = any(
            p.budget_remaining_s is not None and p.budget_remaining_s <= 2.0
            for p in self.pending_queries
        )

        if (
            len(self.pending_queries) >= self.batch_size
            or time_since_last_batch >= ROUTER_FLUSH_TICK_MS
            or time_since_oldest_query >= ROUTER_FLUSH_TICK_MS
            or budget_trigger
        ):
            logger.info(
                "queue.flush_partial: batch_size=%d, time_since_last=%.1fms, oldest_age=%.1fms, budget_trigger=%s",
                len(self.pending_queries),
                time_since_last_batch,
                time_since_oldest_query,
                budget_trigger,
            )
            self._process_batch(self._batch_infer_func)

    async def _process_batch_async(self, batch_infer_func: Callable[..., Any]) -> None:
        """Process accumulated queries in a batch (async version)."""
        if not self.pending_queries:
            return

        pending_batch = self.pending_queries[:]
        self.pending_queries.clear()

        total_batch_size = len(pending_batch)
        total_latency = 0.0
        processed_queries = 0
        micro_count = 0

        try:
            for micro_count, micro_batch in enumerate(self._split_micro_batches(pending_batch), start=1):
                queries = [p.query for p in micro_batch]
                metadata_list = [p.metadata for p in micro_batch]

                if self._warmup_remaining > 0:
                    try:
                        await self._invoke_batch(batch_infer_func, queries, metadata_list, warmup=True)
                    except Exception as warmup_exc:  # pragma: no cover - warmup failure path
                        logger.debug("Warmup batch failed: %s", warmup_exc)
                    finally:
                        self._warmup_remaining = max(self._warmup_remaining - 1, 0)

                results, latency = await self._invoke_batch(
                    batch_infer_func, queries, metadata_list, warmup=False
                )
                total_latency += latency

                if len(results) != len(micro_batch):
                    raise RuntimeError(
                        f"Batch returned {len(results)} results but expected {len(micro_batch)}"
                    )

                for pending, result in zip(micro_batch, results):
                    if not pending.future.done():
                        pending.future.set_result(result)

                processed_queries += len(micro_batch)

                if self.trace_hook is not None:
                    prompt_tokens, decode_tokens = self._aggregate_token_counts(metadata_list)
                    self.trace_hook(
                        {
                            "timestamp": self._clock(),
                            "batch_size": len(micro_batch),
                            "total_batch_size": total_batch_size,
                            "micro_batch_index": micro_count,
                            "latency_ms": latency * 1000.0,
                            "prefill_tokens": prompt_tokens,
                            "decode_tokens": decode_tokens,
                        }
                    )

        except InferenceTimeoutError as exc:
            # Deliver results for successfully processed queries, timeout remaining
            logger.warning("Partial batch timeout: %s", exc)
            for pending in pending_batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
        except Exception as exc:
            logger.error("Batch processing failed: %s", exc)
            for pending in pending_batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return
        finally:
            self.last_batch_time = self._clock()

        if processed_queries == 0:
            return

        self.metrics.total_batches_processed += 1
        self.metrics.total_queries_processed += processed_queries
        self.metrics.avg_batch_size = (
            (self.metrics.avg_batch_size * (self.metrics.total_batches_processed - 1)) + processed_queries
        ) / self.metrics.total_batches_processed

        effective_latency = max(total_latency, 1e-6)
        self._total_processing_time += effective_latency
        self.metrics.avg_processing_time = self._total_processing_time / self.metrics.total_batches_processed
        self.metrics.avg_throughput_inferences_per_sec = (
            self.metrics.total_queries_processed / max(self._total_processing_time, 1e-6)
        )

        logger.info(
            "Processed batch of %d queries in %.3fs (%.1f inferences/sec) via %d micro-batches",
            processed_queries,
            total_latency,
            processed_queries / effective_latency,
            micro_count or 1,
        )

    async def _invoke_batch(
        self,
        batch_infer_func: Callable[..., Any],
        queries: List[Any],
        metadata_list: List[Dict[str, Any]],
        warmup: bool,
    ) -> Tuple[List[Any], float]:
        """Invoke batch inference callable and measure latency."""
        start = self._clock()
        try:
            call_result = self._call_batch_function(batch_infer_func, queries, metadata_list)
            if asyncio.iscoroutine(call_result):
                results = await asyncio.wait_for(call_result, timeout=ROUTER_MAX_WALL_S)
            else:
                results = call_result
        except asyncio.TimeoutError:
            batch_size = len(queries)
            logger.warning("inference.timeout: batch_size=%d, timeout_s=%.1f", batch_size, ROUTER_MAX_WALL_S)
            raise InferenceTimeoutError(batch_size, ROUTER_MAX_WALL_S)
        latency = self._clock() - start

        if warmup:
            return [], latency
        return list(results or []), latency

    def _call_batch_function(
        self,
        batch_infer_func: Callable[..., Any],
        queries: List[Any],
        metadata_list: List[Dict[str, Any]],
    ) -> Any:
        """Call batch inference function with or without metadata based on signature."""
        func_id = id(batch_infer_func)
        if func_id != self._last_callable_id:
            self._supports_metadata = self._detect_metadata_support(batch_infer_func)
            self._last_callable_id = func_id

        if self._supports_metadata:
            return batch_infer_func(queries, metadata_list)
        return batch_infer_func(queries)

    def _detect_metadata_support(self, func: Callable[..., Any]) -> bool:
        """Detects whether callable accepts metadata argument."""
        try:
            signature = inspect.signature(func)
            return len(signature.parameters) >= 2
        except (TypeError, ValueError):
            return False

    def _aggregate_token_counts(self, metadata_list: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Aggregate token counts (prefill/decode) from metadata."""
        prompt_tokens = 0
        decode_tokens = 0
        for meta in metadata_list:
            prompt_tokens += int(meta.get("prompt_tokens") or meta.get("input_tokens") or meta.get("tokens") or 0)
            decode_tokens += int(meta.get("decode_tokens") or meta.get("output_tokens") or 0)
        return prompt_tokens, decode_tokens

    def _split_micro_batches(self, pending_batch: List[PendingQuery]) -> Iterable[List[PendingQuery]]:
        """Split batch into micro-batches if thresholds are defined."""
        if self.micro_batch_size is None and self.max_tokens_per_batch is None:
            yield pending_batch
            return

        micro_batch: List[PendingQuery] = []
        token_budget = 0
        for pending in pending_batch:
            query_tokens = int(pending.metadata.get("prompt_tokens") or pending.metadata.get("tokens") or 1)

            projected_size = len(micro_batch) + 1
            projected_tokens = token_budget + query_tokens
            size_limit = self.micro_batch_size is not None and projected_size > self.micro_batch_size
            token_limit = self.max_tokens_per_batch is not None and projected_tokens > self.max_tokens_per_batch

            if micro_batch and (size_limit or token_limit):
                yield micro_batch
                micro_batch = []
                token_budget = 0

            micro_batch.append(pending)
            token_budget += query_tokens

        if micro_batch:
            yield micro_batch

    def _process_batch(self, batch_infer_func: Callable[..., Any]) -> None:
        """Process accumulated queries in a batch."""
        try:
            # Try to get the current running loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop, schedule the task on it
            asyncio.create_task(self._process_batch_async(batch_infer_func))
        except RuntimeError:
            # No running loop, create one
            asyncio.run(self._process_batch_async(batch_infer_func))

    def check_timeouts(self) -> None:
        """Manually check for and process timed-out batches (for testing)."""
        self._check_and_process_batch()

    def get_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics."""
        return {
            "total_batches_processed": self.metrics.total_batches_processed,
            "total_queries_processed": self.metrics.total_queries_processed,
            "avg_batch_size": self.metrics.avg_batch_size,
            "avg_processing_time": self.metrics.avg_processing_time,
            "avg_throughput_inferences_per_sec": self.metrics.avg_throughput_inferences_per_sec,
            "pending_queries": len(self.pending_queries),
            "warmup_remaining": self._warmup_remaining,
        }
