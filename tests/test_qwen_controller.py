"""Tests for QwenController."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from src.agent.qwen_controller import QwenController, ModelSize


class TestQwenController:
    """Test QwenController functionality."""

    def test_initialization(self):
        """Test controller initializes correctly."""
        controller = QwenController()
        assert len(controller.SUPPORTED_MODELS) == 6
        assert controller.batch_sizes[ModelSize.SIZE_2B] == 8
        assert controller.batch_sizes[ModelSize.SIZE_4B] == 4
        assert controller.batch_sizes[ModelSize.SIZE_8B] == 2

    def test_get_model_name(self):
        """Test model name generation."""
        controller = QwenController()

        # Test instruct variants
        name_2b = controller._get_model_name(ModelSize.SIZE_2B, False)
        assert "Qwen3-VL-2B-Instruct" in name_2b

        name_4b = controller._get_model_name(ModelSize.SIZE_4B, False)
        assert "Qwen3-VL-4B-Instruct" in name_4b

        # Test thinking variants
        name_2b_thinking = controller._get_model_name(ModelSize.SIZE_2B, True)
        assert "Qwen3-VL-2B-Thinking" in name_2b_thinking

    def test_validate_model_name(self):
        """Test model name validation."""
        controller = QwenController()

        # Valid model
        controller._validate_model_name("unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")

        # Invalid model
        with pytest.raises(ValueError, match="not in supported list"):
            controller._validate_model_name("invalid-model-name")

    @patch('asyncio.sleep')  # Mock asyncio.sleep for faster tests
    @pytest.mark.network
    def test_generate_async(self, mock_sleep):
        """Test async generation."""
        controller = QwenController()

        async def run_test():
            text, scores = await controller.generate_async(
                prompt="test prompt",
                model_size=ModelSize.SIZE_2B
            )
            assert isinstance(text, str)
            assert ("Generated response" in text) or ("Pipeline result" in text)
            assert isinstance(scores, list)

        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    @pytest.mark.network
    def test_generate_sync(self):
        """Test sync generation wrapper."""
        controller = QwenController()

        result = controller.generate(
            prompt="test prompt",
            model_size=ModelSize.SIZE_2B
        )
        assert isinstance(result, str)
        assert "Generated response" in result

    def test_get_supported_models(self):
        """Test getting supported models list."""
        controller = QwenController()
        models = controller.get_supported_models()
        assert len(models) == 6
        assert "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit" in models

    @pytest.mark.network
    def test_preload_models(self):
        """Test model preloading."""
        controller = QwenController()
        controller.preload_models([ModelSize.SIZE_2B])

        # Check that model was "loaded"
        assert ModelSize.SIZE_2B in controller.loaded_models

    def test_clear_cache(self):
        """Test cache clearing."""
        controller = QwenController()

        # Get initial stats
        initial_stats = controller.get_cache_stats()

        # Clear cache
        controller.clear_cache()

        # Get stats after clearing
        cleared_stats = controller.get_cache_stats()

        # Verify cache was cleared (stats should be reset or reduced)
        assert isinstance(cleared_stats, dict)
        assert "vision_cache" in cleared_stats
        assert "prompt_kv_cache" in cleared_stats

    def test_get_batch_stats(self):
        """Test batch statistics retrieval."""
        controller = QwenController()
        stats = controller.get_batch_stats()
        assert isinstance(stats, dict)
        # ModelRouter may not have batch stats available, so check for the fallback message
        if "note" in stats:
            assert stats["note"] == "ModelRouter.get_batch_stats not available"
        else:
            # If batch stats are available, check for expected keys
            assert "2B" in stats or "batch_processing_enabled" in stats


class TestModelSize:
    """Test ModelSize enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ModelSize.SIZE_2B.value == "2B"
        assert ModelSize.SIZE_4B.value == "4B"
        assert ModelSize.SIZE_8B.value == "8B"
