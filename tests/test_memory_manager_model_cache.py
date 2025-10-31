"""Test smart paired loading in memory_manager.py with Qwen3-VL models."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.memory_manager import MemoryManager, ModelCache, ModelPair


class TestModelCache:
    """Test ModelCache functionality."""

    def test_vram_probing(self):
        """Test VRAM usage probing."""
        # Mock torch.cuda.mem_get_info to return (free, total) in bytes
        with patch('torch.cuda.mem_get_info', return_value=(4 * 1024**3, 8 * 1024**3)):  # 4GB free, 8GB total
            cache = ModelCache()
            free_gb = cache.probe_vram_free_gb()
            assert free_gb == 4.0

    def test_model_pair_creation(self):
        """Test ModelPair dataclass."""
        pair = ModelPair("2B", "instruct", "thinking")
        assert pair.size == "2B"
        assert pair.instruct_name == "instruct"
        assert pair.thinking_name == "thinking"

    def test_cache_eviction_lru(self):
        """Test LRU eviction when VRAM is full."""
        cache = ModelCache(max_vram_gb=2.0)

        # Add first model
        cache._cached_models["2B"] = {"model": MagicMock(), "last_used": 1.0, "vram_gb": 1.5}
        cache._vram_usage_gb = 1.5

        # Try to add second model that exceeds limit
        with patch.object(cache, 'probe_vram_free_gb', return_value=0.5):  # Only 0.5GB free
            evicted = cache._evict_if_needed(3.0)
            assert evicted == ["2B"]  # Should evict 2B model

    def test_tokenizer_reuse(self):
        """Test tokenizer/processor sharing across models."""
        cache = ModelCache()

        # Mock tokenizer and processor
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()

        # Cache shared components
        cache._shared_tokenizers["Qwen/Qwen3-VL-2B-Instruct"] = mock_tokenizer
        cache._shared_processors["Qwen/Qwen3-VL-2B-Instruct"] = mock_processor

        # Verify reuse
        assert cache.get_shared_tokenizer("Qwen/Qwen3-VL-2B-Instruct") == mock_tokenizer
        assert cache.get_shared_processor("Qwen/Qwen3-VL-2B-Instruct") == mock_processor


class TestMemoryManagerIntegration:
    """Test MemoryManager integration with ModelCache."""

    def test_model_cache_initialization(self):
        """Test ModelCache is properly initialized in MemoryManager."""
        manager = MemoryManager()
        assert hasattr(manager, 'model_cache')
        assert isinstance(manager.model_cache, ModelCache)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_load_model_with_cache(self, mock_processor, mock_tokenizer):
        """Test model loading with caching and local_files_only."""
        manager = MemoryManager()

        # Mock the model loading components
        mock_tokenizer.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        # Test loading a model
        with patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock_model:
            mock_model.return_value = MagicMock()
            mock_model.return_value.get_memory_footprint.return_value = 2 * 1024**3

            # Mock HF_HOME environment variable
            with patch.dict(os.environ, {'HF_HOME': 'E:\\transformer_models'}):
                model = manager.model_cache.load_model("Qwen/Qwen3-VL-2B-Instruct", local_files_only=True)

                # Verify local_files_only was passed
                mock_model.assert_called_with(
                    "Qwen/Qwen3-VL-2B-Instruct",
                    trust_remote_code=True,
                    local_files_only=True,
                    cache_dir='E:\\transformer_models'
                )

    def test_paired_loading_preference(self):
        """Test that pairs of same size are kept resident when possible."""
        manager = MemoryManager()

        # Mock VRAM to allow keeping pairs
        with patch.object(manager.model_cache, 'probe_vram_free_gb', return_value=8.0):
            # This would test the logic for keeping instruct/thinking pairs
            # when VRAM permits
            pass  # Implementation detail test