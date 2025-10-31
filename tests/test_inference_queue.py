"""Tests for InferenceQueue async micro-batching."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock

from src.agent.inference_queue import InferenceQueue, PendingQuery, BatchMetrics, HybridFuture


class TestInferenceQueue:
    """Test InferenceQueue functionality."""

    def test_initialization(self):
        """Test queue initializes correctly."""
        queue = InferenceQueue(batch_size=4, timeout_ms=50)
        assert queue.batch_size == 4
        assert queue.timeout_ms == 50
        assert len(queue.pending_queries) == 0
        assert isinstance(queue.metrics, BatchMetrics)

    def test_sync_add_query(self):
        """Test synchronous query addition."""
        queue = InferenceQueue(batch_size=1)  # Immediate processing

        def mock_infer(queries):
            return [f"result_{q}" for q in queries]

        result = queue.add_query("test_query", mock_infer)
        assert result == "result_test_query"

    def test_batch_processing(self):
        """Test batch processing when batch size reached."""
        queue = InferenceQueue(batch_size=2, timeout_ms=1000)  # Long timeout

        results = []

        def mock_infer(queries):
            results.extend([f"result_{q}" for q in queries])
            return results[-len(queries):]

        # Add first query (should not process yet)
        future1 = queue.add_query_async("query1", mock_infer)
        assert len(queue.pending_queries) == 1
        assert not future1.done()

        # Add second query (should trigger batch processing)
        future2 = queue.add_query_async("query2", mock_infer)

        # Manually trigger processing
        queue.check_timeouts()

        # Wait for results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result1 = loop.run_until_complete(future1)
            result2 = loop.run_until_complete(future2)
        finally:
            loop.close()

        assert result1 == "result_query1"
        assert result2 == "result_query2"
        assert len(results) == 2

    def test_timeout_processing(self):
        """Test timeout-based batch processing."""
        queue = InferenceQueue(batch_size=10, timeout_ms=10)  # Small timeout

        def mock_infer(queries):
            return [f"result_{q}" for q in queries]

        # Add query
        future = queue.add_query_async("test", mock_infer)

        # Wait for timeout processing - the HybridFuture will poll automatically
        result = future.result(timeout=1.0)  # 1 second timeout for the result

        assert result == "result_test"

    def test_batch_metrics(self):
        """Test batch processing metrics."""
        queue = InferenceQueue(batch_size=2, timeout_ms=1000)

        def mock_infer(queries):
            time.sleep(0.01)  # Simulate processing time
            return [f"result_{q}" for q in queries]

        # Process a batch - add both queries first, then they should batch together
        future1 = queue.add_query_async("q1", mock_infer)
        future2 = queue.add_query_async("q2", mock_infer)

        # Wait for both results
        result1 = future1.result(timeout=1.0)
        result2 = future2.result(timeout=1.0)

        assert result1 == "result_q1"
        assert result2 == "result_q2"

        stats = queue.get_stats()
        assert stats["total_batches_processed"] >= 1
        assert stats["total_queries_processed"] >= 2
        assert stats["avg_batch_size"] >= 2.0

    def test_error_handling(self):
        """Test error handling in batch processing."""
        queue = InferenceQueue(batch_size=1)

        def failing_infer(queries):
            raise RuntimeError("Inference failed")

        # Should raise exception
        with pytest.raises(RuntimeError, match="Inference failed"):
            queue.add_query("test", failing_infer)


class TestPendingQuery:
    """Test PendingQuery dataclass."""

    def test_creation(self):
        """Test PendingQuery creation."""
        future = HybridFuture()
        query = PendingQuery(
            query="test",
            future=future,
            timestamp=123456.789,
            metadata={}
        )
        assert query.query == "test"
        assert query.future is future
        assert query.timestamp == 123456.789
        assert query.metadata == {}


class TestBatchMetrics:
    """Test BatchMetrics dataclass."""

    def test_initialization(self):
        """Test BatchMetrics initializes with zeros."""
        metrics = BatchMetrics()
        assert metrics.total_batches_processed == 0
        assert metrics.total_queries_processed == 0
        assert metrics.avg_batch_size == 0.0
        assert metrics.avg_processing_time == 0.0
        assert metrics.avg_throughput_inferences_per_sec == 0.0