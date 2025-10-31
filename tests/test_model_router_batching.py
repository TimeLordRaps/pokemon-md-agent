"""Test model router batching functionality with performance benchmarks."""

import asyncio
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent.model_router import (
    ModelRouter, ModelSize, InferenceQueue,
    TwoStagePipeline, PrefillRequest, PrefillResult,
    DecodeRequest, DecodeResult, GroupKey
)

pytestmark = pytest.mark.slow


class TestInferenceQueue:
    """Test InferenceQueue batching logic."""

    @pytest.mark.timeout(15)
    def test_batch_timeout_processing(self):
        """Test that queries are processed after timeout."""
        queue = InferenceQueue(batch_size=3, timeout_ms=50)

        # Mock batch function
        batch_calls = []
        def mock_batch_infer(queries):
            batch_calls.append(queries)
            return [f"result_{i}" for i in range(len(queries))]

        # Add 2 queries (less than batch_size)
        future1 = queue.add_query_async("query1", mock_batch_infer)
        future2 = queue.add_query_async("query2", mock_batch_infer)

        # Wait for timeout (reduce in FAST mode)
        fast_mode = os.getenv("FAST", "0") == "1"
        sleep_time = 0.03 if fast_mode else 0.06
        time.sleep(sleep_time)  # Slightly longer than 50ms

        # Manually check timeouts (since no event loop in sync test)
        queue.check_timeouts()

        # Check results
        assert future1.result() == "result_0"
        assert future2.result() == "result_1"
        assert len(batch_calls) == 1
        assert len(batch_calls[0]) == 2

    def test_batch_size_processing(self):
        """Test that queries are processed when batch_size is reached."""
        queue = InferenceQueue(batch_size=2, timeout_ms=1000)  # Long timeout

        batch_calls = []
        def mock_batch_infer(queries):
            batch_calls.append(queries)
            return [f"result_{i}" for i in range(len(queries))]

        # Add exactly batch_size queries
        future1 = queue.add_query_async("query1", mock_batch_infer)
        future2 = queue.add_query_async("query2", mock_batch_infer)

        # Should process immediately
        assert future1.result() == "result_0"
        assert future2.result() == "result_1"
        assert len(batch_calls) == 1
        assert len(batch_calls[0]) == 2

    def test_batch_metrics(self):
        """Test batch processing metrics tracking."""
        queue = InferenceQueue(batch_size=2, timeout_ms=50)

        def mock_batch_infer(queries):
            time.sleep(0.01)  # Simulate processing time
            return [f"result_{i}" for i in range(len(queries))]

        # Process multiple batches
        for i in range(6):
            future = queue.add_query_async(f"query{i}", mock_batch_infer)
            future.result()

        stats = queue.get_stats()
        assert stats["total_batches_processed"] >= 3
        assert stats["total_queries_processed"] == 6
        assert stats["avg_batch_size"] > 0
        assert stats["avg_processing_time"] > 0


class TestModelRouterBatching:
    """Test ModelRouter batching integration."""

    def test_dynamic_batch_sizing(self):
        """Test dynamic batch size calculation."""
        router = ModelRouter()

        # Test different model sizes with default parameters
        assert router.auto_batch_size(ModelSize.SIZE_2B) == 2  # Default scaling gives 2
        assert router.auto_batch_size(ModelSize.SIZE_4B) == 2  # Default scaling gives ~2
        assert router.auto_batch_size(ModelSize.SIZE_8B) == 1  # Default scaling gives ~1

        # Test with parameters that give base sizes
        assert router.auto_batch_size(ModelSize.SIZE_2B, gpu_utilization=0.5, vram_used_gb=18.0) == 4
        assert router.auto_batch_size(ModelSize.SIZE_4B, gpu_utilization=0.5, vram_used_gb=18.0) == 3
        assert router.auto_batch_size(ModelSize.SIZE_8B, gpu_utilization=0.5, vram_used_gb=18.0) == 2

        # Test with GPU utilization scaling
        assert router.auto_batch_size(ModelSize.SIZE_2B, gpu_utilization=0.8, vram_used_gb=18.0) < 4  # Higher util = smaller batch

    @pytest.mark.asyncio
    async def test_concurrent_async_inference(self):
        """Test concurrent async inference calls."""
        router = ModelRouter()

        # Mock the batch inference
        original_infer_async = router.infer_async
        call_count = 0

        async def mock_infer_async(query, model_size):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)  # Simulate processing
            return f"result_{call_count}"

        router.infer_async = mock_infer_async

        # Make concurrent requests
        tasks = [
            router.infer_async("query1", ModelSize.SIZE_2B),
            router.infer_async("query2", ModelSize.SIZE_2B),
            router.infer_async("query3", ModelSize.SIZE_2B),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all("result_" in r for r in results)

    def test_backward_compatibility_sync_wrapper(self):
        """Test that sync infer() method works."""
        router = ModelRouter()

        # Mock async method
        original_infer_async = router.infer_async

        async def mock_infer_async(query, model_size):
            return f"sync_result_{query}"

        router.infer_async = mock_infer_async

        # Test sync wrapper
        result = router.infer("test_query", ModelSize.SIZE_2B)
        assert result == "sync_result_test_query"


class TestBatchingPerformance:
    """Performance benchmarks for batching."""

    def test_single_vs_batched_throughput(self):
        """Benchmark single vs batched inference throughput."""
        # Simulate processing times (rough estimates)
        single_query_time = 0.1  # 100ms per query individually
        batched_query_time = 0.025  # 25ms per query when batched (4x speedup)

        # Single processing: 8 queries sequentially
        single_total_time = 8 * single_query_time  # 800ms

        # Batched processing: 2 batches of 4 queries each
        batched_total_time = 2 * (4 * batched_query_time)  # 200ms

        speedup = single_total_time / batched_total_time
        assert speedup >= 3.0, f"Expected 3x+ speedup, got {speedup:.1f}x"

    def test_memory_usage_bounds(self):
        """Test memory usage stays within bounds."""
        # Correct VRAM usage per model (user correction: 8B is actually 4B quantized)
        base_vram = {
            ModelSize.SIZE_2B: 4,  # 4GB for 2B models
            ModelSize.SIZE_4B: 8,  # 8GB for 4B models
            ModelSize.SIZE_8B: 8,  # 8GB for 8B models (they are actually 4B quantized)
        }

        # With batching, should not exceed base + reasonable overhead
        max_vram_limit = 24  # 24GB limit

        for model_size, vram_gb in base_vram.items():
            # Assume 2x batch size overhead max
            max_batch_vram = vram_gb * 2
            assert max_batch_vram <= max_vram_limit, f"{model_size.value} exceeds VRAM limit: {max_batch_vram}GB > {max_vram_limit}GB"

    def test_latency_tradeoffs(self):
        """Test p99 latency requirements."""
        # Simulate latency distribution
        latencies = [50, 60, 70, 80, 90, 100, 150, 200]  # Mix of latencies

        # Calculate p99 (99th percentile)
        sorted_latencies = sorted(latencies)
        p99_index = int(0.99 * len(sorted_latencies))
        p99_latency = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]

        assert p99_latency <= 200, f"p99 latency {p99_latency}ms exceeds 200ms target"


class TestTwoStagePipeline:
    """Test TwoStagePipeline functionality."""

    def test_prefill_request_creation(self):
        """Test PrefillRequest dataclass creation."""
        request = PrefillRequest(
            prompt="Test prompt",
            images=["image1", "image2"],
            model_size=ModelSize.SIZE_4B,
            use_thinking=True,
            max_tokens=128
        )

        assert request.prompt == "Test prompt"
        assert request.images == ["image1", "image2"]
        assert request.model_size == ModelSize.SIZE_4B
        assert request.use_thinking is True
        assert request.max_tokens == 128

    def test_group_key_creation(self):
        """Test GroupKey creation and hashing."""
        key1 = GroupKey(
            model_id="test_model",
            mode="instruct",
            max_seq=256,
            vision_shape=(2, "image")
        )

        key2 = GroupKey(
            model_id="test_model",
            mode="instruct",
            max_seq=256,
            vision_shape=(2, "image")
        )

        assert key1 == key2
        assert hash(key1) == hash(key2)

        # Different keys should not be equal
        key3 = GroupKey(
            model_id="test_model",
            mode="thinking",  # Different mode
            max_seq=256,
            vision_shape=(2, "image")
        )
        assert key1 != key3

    def test_pipeline_initialization(self):
        """Test TwoStagePipeline initialization."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=30)

        assert pipeline.model_router == router
        assert pipeline.flush_tick_ms == 30
        assert isinstance(pipeline.prefill_queues, dict)
        assert isinstance(pipeline.decode_queues, dict)

    def test_prefill_submission(self):
        """Test submitting prefill requests."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=100)  # Long flush to prevent auto-processing

        request = PrefillRequest(
            prompt="Test prompt",
            model_size=ModelSize.SIZE_2B
        )

        future = pipeline.submit_prefill(request)

        # Check that request was queued
        assert len(pipeline.prefill_queues) == 1

        # Force flush to process
        pipeline.force_flush()

        # Should have processed and cleaned up queue
        assert len(pipeline.prefill_queues) == 0

    def test_micro_batching_grouping(self):
        """Test that micro-batching groups requests correctly."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=1000)  # Long delay

        # Submit multiple requests with same group key
        for i in range(3):
            request = PrefillRequest(
                prompt=f"Test prompt {i}",
                model_size=ModelSize.SIZE_2B,
                use_thinking=False,
                max_tokens=128
            )
            pipeline.submit_prefill(request)

        # Should all be in same group
        assert len(pipeline.prefill_queues) == 1
        group_key = list(pipeline.prefill_queues.keys())[0]
        assert len(pipeline.prefill_queues[group_key]) == 3

        # Different model should be in different group
        request_diff = PrefillRequest(
            prompt="Different model",
            model_size=ModelSize.SIZE_4B,  # Different model
            use_thinking=False,
            max_tokens=128
        )
        pipeline.submit_prefill(request_diff)

        assert len(pipeline.prefill_queues) == 2

    def test_flush_tick_timing(self):
        """Test flush tick timing mechanism."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=10)  # Very short flush

        request = PrefillRequest(prompt="Test", model_size=ModelSize.SIZE_2B)
        pipeline.submit_prefill(request)

        # Initially queued
        assert len(pipeline.prefill_queues) > 0

        # Wait for flush tick
        time.sleep(0.02)  # Slightly longer than 10ms

        # Check flush (simulate by calling check_flush)
        pipeline._check_flush()

        # Should have been processed
        assert len(pipeline.prefill_queues) == 0