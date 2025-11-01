"""Unit tests for deadline-aware ModelRouter functionality.

Tests deadline budget checking, model selection fallbacks, and truncation behavior.
"""

import sys
import asyncio
from pathlib import Path
import time
from unittest.mock import Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.model_router import (
    ModelRouter, ModelSize, DeadlineExceededError,
    PrefillRequest, DecodeRequest, PrefillResult
)


class TestModelRouterDeadline:
    """Test deadline-aware routing logic."""

    def test_select_model_with_budget_8b(self):
        """Test model selection prefers 8B when budget allows."""
        router = ModelRouter()
        selected = router.select_model(remaining_budget_s=4.0)
        assert selected == ModelSize.SIZE_8B

    def test_select_model_with_budget_4b(self):
        """Test model selection falls back to 4B when 8B exceeds budget."""
        router = ModelRouter()
        selected = router.select_model(remaining_budget_s=2.5)
        assert selected == ModelSize.SIZE_4B

    def test_select_model_with_budget_2b(self):
        """Test model selection falls back to 2B when larger models exceed budget."""
        router = ModelRouter()
        selected = router.select_model(remaining_budget_s=1.0)
        assert selected == ModelSize.SIZE_2B

    def test_select_model_no_budget(self):
        """Test DeadlineExceededError when no model fits budget."""
        router = ModelRouter()
        with pytest.raises(DeadlineExceededError):
            router.select_model(remaining_budget_s=0.1)

    def test_select_model_preferred_model_fits(self):
        """Test preferred model selection when it fits budget."""
        router = ModelRouter()
        selected = router.select_model(remaining_budget_s=3.0, preferred_model=ModelSize.SIZE_4B)
        assert selected == ModelSize.SIZE_4B

    def test_select_model_preferred_model_fallback(self):
        """Test fallback when preferred model exceeds budget."""
        router = ModelRouter()
        selected = router.select_model(remaining_budget_s=1.0, preferred_model=ModelSize.SIZE_8B)
        assert selected == ModelSize.SIZE_2B

    def test_estimate_inference_time(self):
        """Test inference time estimation for different models."""
        router = ModelRouter()
        time_8b = router._estimate_inference_time(ModelSize.SIZE_8B, False)
        time_4b = router._estimate_inference_time(ModelSize.SIZE_4B, False)
        time_2b = router._estimate_inference_time(ModelSize.SIZE_2B, False)

        # 8B should take longest, 2B shortest
        assert time_8b > time_4b > time_2b
        # All should be positive
        assert all(t > 0 for t in [time_8b, time_4b, time_2b])

    @patch('src.agent.model_router.time.time')
    @pytest.mark.asyncio
    async def test_prefill_deadline_exceeded(self, mock_time):
        """Test prefill deadline exceeded handling."""
        mock_time.return_value = 100.0  # Fixed time

        router = ModelRouter()
        pipeline = router.two_stage_pipeline

        # Initialize caches first
        pipeline.force_flush()

        request = PrefillRequest(prompt="test", deadline_s=99.0)  # Already expired

        future = pipeline.submit_prefill(request)

        # Should raise DeadlineExceededError
        with pytest.raises(DeadlineExceededError):
            await future

    @patch('src.agent.model_router.time.time')
    @pytest.mark.asyncio
    async def test_decode_deadline_exceeded(self, mock_time):
        """Test decode deadline exceeded handling."""
        mock_time.return_value = 100.0  # Fixed time

        router = ModelRouter()
        pipeline = router.two_stage_pipeline

        # Initialize caches first
        pipeline.force_flush()

        prefill_result = PrefillResult(tokenized_input="test")
        request = DecodeRequest(prefill_result=prefill_result, deadline_s=99.0)  # Already expired

        future = pipeline.submit_decode(request)

        # Should raise DeadlineExceededError
        with pytest.raises(DeadlineExceededError):
            await future

    @pytest.mark.asyncio
    async def test_prefill_no_deadline(self):
        """Test prefill processing without deadline works normally."""
        router = ModelRouter()
        pipeline = router.two_stage_pipeline

        request = PrefillRequest(prompt="test prompt")

        future = pipeline.submit_prefill(request)

        # Should not raise exception immediately
        assert future is not None
        assert not future.done()

    @pytest.mark.asyncio
    async def test_decode_no_deadline(self):
        """Test decode processing without deadline works normally."""
        router = ModelRouter()
        pipeline = router.two_stage_pipeline

        prefill_result = PrefillResult(tokenized_input="test")
        request = DecodeRequest(prefill_result=prefill_result)

        future = pipeline.submit_decode(request)

        # Should not raise exception immediately
        assert future is not None
        assert not future.done()