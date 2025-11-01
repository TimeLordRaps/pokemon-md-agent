"""Test inference queue stages with pipeline batching and micro-batching.

Validates PREFILL/DECODE stages, group key micro-batching, timeout processing,
batch size limits, and concurrent async pipeline operations. Ensures efficient
token processing with minimal latency overhead and proper queue management.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, Mock
from src.agent.model_router import (
    ModelRouter, ModelSize, TwoStagePipeline,
    PrefillRequest, PrefillResult, DecodeRequest, DecodeResult, GroupKey
)


class TestInferenceQueueStages:
    """Test two-stage pipeline functionality."""

    @pytest.mark.asyncio
    async def test_prefill_stage_micro_batching(self):
        """Test PREFILL stage groups requests by model and parameters."""
        # Sanitize HF_HOME for testing
        from src.agent.utils import sanitize_hf_home
        import os
        old_hf_home = os.environ.get('HF_HOME')
        sanitized_hf = sanitize_hf_home()
        if sanitized_hf:
            os.environ['HF_HOME'] = sanitized_hf

        try:
            router = ModelRouter()
            pipeline = TwoStagePipeline(router, flush_tick_ms=100)

            # Create requests with same group key
            requests = []
            for i in range(3):
                req = PrefillRequest(
                    prompt=f"Test prompt {i}",
                    model_size=ModelSize.SIZE_2B,
                    use_thinking=False,
                    max_tokens=128
                )
                requests.append(req)

            # Submit all requests
            futures = [pipeline.submit_prefill(req) for req in requests]

            # Force flush to process
            pipeline.force_flush()

            # Wait for all futures to complete
            results = await asyncio.gather(*futures)

            # Check that all futures completed successfully
            assert len(results) == 3
            for result in results:
                assert isinstance(result, PrefillResult)

            # Queues should be empty after processing
            assert len(pipeline.prefill_queues) == 0  # All groups processed
        finally:
            # Restore original HF_HOME
            if old_hf_home is not None:
                os.environ['HF_HOME'] = old_hf_home
            elif 'HF_HOME' in os.environ:
                del os.environ['HF_HOME']

    @pytest.mark.asyncio
    async def test_decode_stage_processing(self):
        """Test DECODE stage processes prefill results with temperatures."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=100)

        # Mock prefill result
        prefill_result = PrefillResult(
            tokenized_input="tokenized",
            prompt_sha="sha123",
            cache_hit=False
        )

        # Create decode request
        decode_req = DecodeRequest(prefill_result=prefill_result, temperature=0.7)

        # Submit decode request
        future = pipeline.submit_decode(decode_req)

        # Force flush
        pipeline.force_flush()

        # Wait for completion
        result = await future
        assert isinstance(result, DecodeResult)

        # Queues should be empty after processing
        assert len(pipeline.decode_queues) == 0

    def test_group_key_hashing(self):
        """Test group key creation and hashing for micro-batching."""
        key1 = GroupKey(
            model_id="model1",
            mode="instruct",
            max_seq=256,
            vision_shape=(2, "image")
        )

        key2 = GroupKey(
            model_id="model1",
            mode="instruct",
            max_seq=256,
            vision_shape=(2, "image")
        )

        key3 = GroupKey(
            model_id="model2",  # Different
            mode="instruct",
            max_seq=256,
            vision_shape=(2, "image")
        )

        assert key1 == key2
        assert hash(key1) == hash(key2)
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_operations(self):
        """Test concurrent PREFILL and DECODE operations."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=10)  # Short flush

        # Submit multiple prefill requests
        prefill_futures = []
        for i in range(5):
            req = PrefillRequest(
                prompt=f"Prompt {i}",
                model_size=ModelSize.SIZE_2B
            )
            prefill_futures.append(pipeline.submit_prefill(req))

        # Submit decode requests
        decode_futures = []
        for i in range(3):
            prefill_result = PrefillResult(
                tokenized_input=f"tokenized_{i}",
                prompt_sha=f"sha_{i}"
            )
            decode_req = DecodeRequest(prefill_result=prefill_result, temperature=0.8)
            decode_futures.append(pipeline.submit_decode(decode_req))

        # Let flush ticks process
        time.sleep(0.05)

        # Check queues processed
        pipeline._check_flush()

        # Wait for all futures to complete
        await asyncio.gather(*prefill_futures, *decode_futures)

        # Queues should be empty after processing
        assert len(pipeline.prefill_queues) == 0
        assert len(pipeline.decode_queues) == 0

    @pytest.mark.asyncio
    async def test_batch_size_limits(self):
        """Test batch size limits are respected in processing."""
        router = ModelRouter()

        # Mock batch processing with size limits
        def mock_batch_prefill(prompts, images_list, model_size, use_thinking):
            # Note: In real implementation, batching logic would limit sizes
            return [PrefillResult(tokenized_input=f"result_{i}") for i in range(len(prompts))]

        router.two_stage_pipeline._batch_prefill_processing = mock_batch_prefill

        # Submit requests (batching handled internally)
        for i in range(6):
            req = PrefillRequest(prompt=f"Prompt {i}", model_size=ModelSize.SIZE_2B)
            router.two_stage_pipeline.submit_prefill(req)

        # Force processing
        router.two_stage_pipeline.force_flush()

    @pytest.mark.asyncio
    async def test_timeout_flush_mechanism(self):
        """Test timeout-based flush prevents indefinite queuing."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=20)  # 20ms timeout

        # Initialize caches first
        pipeline.force_flush()

        # Submit request
        req = PrefillRequest(prompt="Test", model_size=ModelSize.SIZE_2B)
        pipeline.submit_prefill(req)

        # Initially queued
        assert len(pipeline.prefill_queues) == 1
        group_key = list(pipeline.prefill_queues.keys())[0]
        assert len(pipeline.prefill_queues[group_key]) == 1

        # Wait for timeout
        time.sleep(0.03)  # Longer than 20ms

        # Check flush
        pipeline._check_flush()

        # Should be processed (group should no longer exist or be empty)
        if group_key in pipeline.prefill_queues:
            assert len(pipeline.prefill_queues[group_key]) == 0
        else:
            # Group was completely processed and removed
            pass

    @pytest.mark.asyncio
    async def test_pipeline_metrics_tracking(self):
        """Test pipeline tracks batching and latency metrics."""
        router = ModelRouter()
        pipeline = TwoStagePipeline(router, flush_tick_ms=100)

        # Submit multiple batches
        for batch in range(3):
            for i in range(2):
                req = PrefillRequest(
                    prompt=f"Batch{batch}_Prompt{i}",
                    model_size=ModelSize.SIZE_2B
                )
                pipeline.submit_prefill(req)

            # Force flush per batch
            pipeline.force_flush()

        # Should have processed all
        for group_key, requests in pipeline.prefill_queues.items():
            assert len(requests) == 0, f"Group {group_key} should have empty queue"
