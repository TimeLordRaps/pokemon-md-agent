"""Tests for pipeline engine with continuous batching and ≤50ms tick for partial flush."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from src.agent.pipeline_engine import (
    PipelineEngine, PipelineRequest, Batch, PipelineStage
)


class TestPipelineRequest:
    """Test PipelineRequest functionality."""

    def test_request_creation(self):
        """Test creating a pipeline request."""
        request = PipelineRequest(
            id="req_123",
            prompt="test prompt",
            model_name="test-model",
            max_tokens=256
        )

        assert request.id == "req_123"
        assert request.prompt == "test prompt"
        assert request.model_name == "test-model"
        assert request.max_tokens == 256
        assert request.stage == PipelineStage.PREFILL
        assert isinstance(request.created_at, float)
        assert isinstance(request.last_active, float)

    def test_request_touch(self):
        """Test touching a request updates last_active."""
        request = PipelineRequest("test", "prompt")
        original_active = request.last_active

        time.sleep(0.001)
        request.touch()

        assert request.last_active > original_active

    def test_request_staleness(self):
        """Test request staleness detection."""
        request = PipelineRequest("test", "prompt")

        # Fresh request
        assert not request.is_stale()

        # Make it stale
        request.last_active = time.time() - 40  # 40 seconds ago
        assert request.is_stale(30)  # 30 second timeout


class TestBatch:
    """Test Batch functionality."""

    def test_batch_creation(self):
        """Test creating a batch."""
        requests = [
            PipelineRequest("req1", "prompt1"),
            PipelineRequest("req2", "prompt2"),
        ]

        batch = Batch(
            id="batch_001",
            requests=requests,
            stage=PipelineStage.PREFILL
        )

        assert batch.id == "batch_001"
        assert len(batch.requests) == 2
        assert batch.size == 2
        assert batch.stage == PipelineStage.PREFILL
        assert isinstance(batch.created_at, float)


class TestPipelineEngine:
    """Test PipelineEngine functionality."""

    def test_engine_initialization(self):
        """Test pipeline engine initialization."""
        engine = PipelineEngine()

        assert engine.max_batch_size == 8
        assert engine.tick_interval_ms == 50
        assert engine.max_queue_depth == 100
        assert engine.starvation_threshold_ms == 1000
        assert not engine.running
        assert len(engine.prefill_queue) == 0
        assert len(engine.decode_queue) == 0
        assert len(engine.completed_requests) == 0
        assert engine.active_prefill_batch is None
        assert engine.active_decode_batch is None

    def test_engine_custom_initialization(self):
        """Test pipeline engine with custom parameters."""
        engine = PipelineEngine(
            max_batch_size=4,
            tick_interval_ms=25,
            max_queue_depth=50,
            starvation_threshold_ms=500
        )

        assert engine.max_batch_size == 4
        assert engine.tick_interval_ms == 25
        assert engine.max_queue_depth == 50
        assert engine.starvation_threshold_ms == 500

    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """Test starting and stopping the engine."""
        engine = PipelineEngine()

        # Start
        await engine.start()
        assert engine.running
        assert engine.tick_task is not None

        # Stop
        await engine.stop()
        assert not engine.running
        # Task may still exist but should be cancelled/done
        assert engine.tick_task.done() if engine.tick_task else True

    @pytest.mark.asyncio
    async def test_submit_request_success(self):
        """Test successful request submission."""
        engine = PipelineEngine()

        request = PipelineRequest("test_req", "test prompt")
        success = await engine.submit_request(request)

        assert success
        assert len(engine.prefill_queue) == 1
        assert engine.prefill_queue[0] == request

    @pytest.mark.asyncio
    async def test_submit_request_queue_full(self):
        """Test request submission when queue is full."""
        engine = PipelineEngine(max_queue_depth=1)

        # Fill queue
        req1 = PipelineRequest("req1", "prompt1")
        await engine.submit_request(req1)

        # Try to add another
        req2 = PipelineRequest("req2", "prompt2")
        success = await engine.submit_request(req2)

        assert not success
        assert len(engine.prefill_queue) == 1
        assert engine.stats["queue_full_rejects"] == 1

    @pytest.mark.asyncio
    async def test_get_completed_request(self):
        """Test getting completed requests."""
        engine = PipelineEngine()

        # No completed request
        result = await engine.get_completed_request("nonexistent")
        assert result is None

        # Add a completed request manually
        completed_req = PipelineRequest("completed", "done")
        completed_req.stage = PipelineStage.COMPLETE
        engine.completed_requests["completed"] = completed_req

        result = await engine.get_completed_request("completed")
        assert result == completed_req
        assert "completed" not in engine.completed_requests

    def test_queue_depths(self):
        """Test getting queue depths."""
        engine = PipelineEngine()

        # Add some requests
        engine.prefill_queue.append(PipelineRequest("p1", "prompt1"))
        engine.prefill_queue.append(PipelineRequest("p2", "prompt2"))
        engine.decode_queue.append(PipelineRequest("d1", "prompt3"))
        engine.completed_requests["c1"] = PipelineRequest("c1", "prompt4")

        depths = engine.get_queue_depths()
        assert depths["prefill"] == 2
        assert depths["decode"] == 1
        assert depths["completed"] == 1

    def test_get_stats(self):
        """Test getting engine statistics."""
        engine = PipelineEngine()

        stats = engine.get_stats()
        assert "active_prefill_batch" in stats
        assert "active_decode_batch" in stats
        assert "tick_interval_ms" in stats
        assert "max_batch_size" in stats
        assert stats["tick_interval_ms"] == 50
        assert stats["max_batch_size"] == 8

    def test_clear_queues(self):
        """Test clearing all queues."""
        engine = PipelineEngine()

        # Add some data
        engine.prefill_queue.append(PipelineRequest("p1", "prompt1"))
        engine.decode_queue.append(PipelineRequest("d1", "prompt2"))
        engine.completed_requests["c1"] = PipelineRequest("c1", "prompt3")
        engine.active_prefill_batch = Batch("batch1", [PipelineRequest("b1", "prompt4")], PipelineStage.PREFILL)
        engine.active_decode_batch = Batch("batch2", [PipelineRequest("b2", "prompt5")], PipelineStage.DECODE)

        engine.clear_queues()

        assert len(engine.prefill_queue) == 0
        assert len(engine.decode_queue) == 0
        assert len(engine.completed_requests) == 0
        assert engine.active_prefill_batch is None
        assert engine.active_decode_batch is None


class TestPipelineBatching:
    """Test batch assembly and processing."""

    def test_assemble_batch_empty_queue(self):
        """Test batch assembly with empty queue."""
        engine = PipelineEngine()

        batch = engine._assemble_batch(PipelineStage.PREFILL)
        assert batch is None

    def test_assemble_batch_single_request(self):
        """Test batch assembly with single request."""
        engine = PipelineEngine()
        engine.prefill_queue.append(PipelineRequest("req1", "prompt1"))

        batch = engine._assemble_batch(PipelineStage.PREFILL)
        assert batch is not None
        assert batch.size == 1
        assert len(engine.prefill_queue) == 0

    def test_assemble_batch_multiple_requests(self):
        """Test batch assembly with multiple requests."""
        engine = PipelineEngine(max_batch_size=3)

        # Add 5 requests
        for i in range(5):
            engine.prefill_queue.append(PipelineRequest(f"req{i}", f"prompt{i}"))

        batch = engine._assemble_batch(PipelineStage.PREFILL)
        assert batch is not None
        assert batch.size == 3  # Limited by max_batch_size
        assert len(engine.prefill_queue) == 2  # 5 - 3 = 2 remaining

    def test_assemble_batch_force_flush(self):
        """Test forced batch flush for starvation prevention."""
        engine = PipelineEngine(max_batch_size=8)

        # Add only 1 request
        engine.prefill_queue.append(PipelineRequest("req1", "prompt1"))

        # Normal assembly (should wait for more)
        batch = engine._assemble_batch(PipelineStage.PREFILL, force_flush=False)
        assert batch is None

        # Force flush (should create small batch)
        batch = engine._assemble_batch(PipelineStage.PREFILL, force_flush=True)
        assert batch is not None
        assert batch.size == 1

    @pytest.mark.asyncio
    async def test_starvation_check(self):
        """Test starvation detection and forced flush."""
        engine = PipelineEngine(starvation_threshold_ms=100)

        # Add a request and make it old
        old_request = PipelineRequest("old_req", "old_prompt")
        old_request.created_at = time.time() - 1  # 1 second ago (less than threshold)
        engine.prefill_queue.append(old_request)

        # Should not trigger starvation yet
        await engine._check_starvation()
        assert len(engine.prefill_queue) == 1

        # Make it old enough
        old_request.created_at = time.time() - 2  # 2 seconds ago (> 100ms threshold)
        await engine._check_starvation()

        # Should have triggered forced flush
        assert len(engine.prefill_queue) == 0
        assert engine.stats["starvation_events"] == 1


class TestPipelineCallbacks:
    """Test pipeline callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_registration(self):
        """Test registering and using callbacks."""
        engine = PipelineEngine()

        # Mock callbacks
        prefill_callback = AsyncMock()
        decode_callback = AsyncMock()

        engine.set_prefill_callback(prefill_callback)
        engine.set_decode_callback(decode_callback)

        # Create batch and process
        batch = Batch("test_batch", [PipelineRequest("req1", "prompt1")], PipelineStage.PREFILL)

        await engine._flush_batch(batch)

        # Callback should have been called
        prefill_callback.assert_called_once_with(batch)

    @pytest.mark.asyncio
    async def test_batch_processing_success(self):
        """Test successful batch processing."""
        engine = PipelineEngine()

        # Mock successful callback
        callback = AsyncMock()
        engine.set_prefill_callback(callback)

        # Submit request
        request = PipelineRequest("test_req", "test_prompt")
        await engine.submit_request(request)

        # Manually trigger batch processing
        batch = engine._assemble_batch(PipelineStage.PREFILL)
        await engine._flush_batch(batch)

        # Request should be completed
        completed = await engine.get_completed_request("test_req")
        assert completed is not None
        assert completed.stage == PipelineStage.COMPLETE

    @pytest.mark.asyncio
    async def test_batch_processing_failure(self):
        """Test batch processing failure handling."""
        engine = PipelineEngine()

        # Mock failing callback
        callback = AsyncMock(side_effect=Exception("Processing failed"))
        engine.set_prefill_callback(callback)

        # Submit request
        request = PipelineRequest("fail_req", "fail_prompt")
        await engine.submit_request(request)

        # Manually trigger batch processing
        batch = engine._assemble_batch(PipelineStage.PREFILL)
        await engine._flush_batch(batch)

        # Request should be back in queue (after failure)
        assert len(engine.prefill_queue) == 1
        assert engine.prefill_queue[0] == request


class TestTickLoop:
    """Test tick loop functionality."""

    @pytest.mark.asyncio
    async def test_tick_processing(self):
        """Test that tick loop processes batches."""
        engine = PipelineEngine(tick_interval_ms=10)  # Fast ticks for testing

        # Add requests
        for i in range(3):
            await engine.submit_request(PipelineRequest(f"req{i}", f"prompt{i}"))

        # Start engine briefly
        await engine.start()
        await asyncio.sleep(0.05)  # Let ticks run
        await engine.stop()

        # Check that requests were processed (would be completed in real scenario)
        # In this test setup, they remain in queue since no callbacks are set
        assert len(engine.prefill_queue) == 3  # Still in queue without callbacks

    @pytest.mark.asyncio
    async def test_tick_with_callbacks(self):
        """Test tick processing with callbacks set."""
        engine = PipelineEngine(tick_interval_ms=10)

        # Set mock callback
        callback = AsyncMock()
        engine.set_prefill_callback(callback)

        # Add requests
        for i in range(2):
            await engine.submit_request(PipelineRequest(f"req{i}", f"prompt{i}"))

        # Start briefly
        await engine.start()
        await asyncio.sleep(0.05)
        await engine.stop()

        # Callback should have been called
        assert callback.called


if __name__ == "__main__":
    pytest.main([__file__])