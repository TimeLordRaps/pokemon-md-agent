"""Tests for prompt cache with LRU per model (2-5 entries) RAM + optional disk spill."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, mock_open
from src.agent.prompt_cache import PromptCache, PromptCacheEntry


class TestPromptCacheEntry:
    """Test PromptCacheEntry functionality."""

    def test_entry_creation(self):
        """Test creating a cache entry."""
        entry = PromptCacheEntry(
            prompt_sha="abc123",
            model_name="test-model",
            tokenized_data="tokenized_data",
            kv_cache="kv_data",
            vision_features="vision_data"
        )

        assert entry.prompt_sha == "abc123"
        assert entry.model_name == "test-model"
        assert entry.tokenized_data == "tokenized_data"
        assert entry.kv_cache == "kv_data"
        assert entry.vision_features == "vision_data"
        assert entry.access_count == 0
        assert isinstance(entry.timestamp, float)

    def test_entry_touch(self):
        """Test touching an entry updates timestamp."""
        entry = PromptCacheEntry("test", "model", "data")
        original_timestamp = entry.timestamp

        time.sleep(0.001)  # Small delay
        entry.touch()

        assert entry.timestamp > original_timestamp
        assert entry.access_count == 1


class TestPromptCache:
    """Test PromptCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with default settings."""
        cache = PromptCache()

        assert cache.max_entries_per_model == 5
        assert not cache.enable_disk
        assert cache.cache_dir.name == "pmd_prompt_cache"
        assert len(cache.model_caches) == 0

    def test_cache_initialization_custom(self):
        """Test cache initialization with custom settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = PromptCache(
                max_entries_per_model=3,
                enable_disk=True,
                cache_dir=Path(temp_dir)
            )

            assert cache.max_entries_per_model == 3
            assert cache.enable_disk
            assert cache.cache_dir == Path(temp_dir)

    def test_key_generation(self):
        """Test cache key generation."""
        cache = PromptCache()

        # Test basic key
        key1 = cache._make_key("hello world")
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 truncated

        # Test with images hash
        key2 = cache._make_key("hello world", "img123")
        assert key2 != key1

        # Test with tool schema hash
        key3 = cache._make_key("hello world", None, "tool456")
        assert key3 != key1
        assert key3 != key2

    def test_get_nonexistent_entry(self):
        """Test getting a non-existent entry."""
        cache = PromptCache()

        entry = cache.get("test-model", "hello world")
        assert entry is None

    def test_put_and_get_entry(self):
        """Test putting and getting an entry."""
        cache = PromptCache()

        # Put entry
        cache.put(
            model_name="test-model",
            prompt="hello world",
            tokenized_data="tokenized",
            kv_cache="kv_data"
        )

        # Get entry
        entry = cache.get("test-model", "hello world")
        assert entry is not None
        assert entry.tokenized_data == "tokenized"
        assert entry.kv_cache == "kv_data"
        assert entry.access_count == 1

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = PromptCache(max_entries_per_model=2)

        # Fill cache
        cache.put("model", "prompt1", "data1")
        cache.put("model", "prompt2", "data2")
        cache.put("model", "prompt3", "data3")  # Should evict prompt1

        # Check eviction
        assert cache.get("model", "prompt1") is None
        assert cache.get("model", "prompt2") is not None
        assert cache.get("model", "prompt3") is not None

    def test_lru_access_order(self):
        """Test LRU maintains access order."""
        cache = PromptCache(max_entries_per_model=3)

        # Add entries
        cache.put("model", "prompt1", "data1")
        cache.put("model", "prompt2", "data2")
        cache.put("model", "prompt3", "data3")

        # Access prompt1 (moves to end)
        cache.get("model", "prompt1")

        # Add prompt4 (should evict prompt2)
        cache.put("model", "prompt4", "data4")

        assert cache.get("model", "prompt1") is not None  # Most recently accessed
        assert cache.get("model", "prompt2") is None     # Evicted
        assert cache.get("model", "prompt3") is not None
        assert cache.get("model", "prompt4") is not None

    def test_per_model_caches(self):
        """Test that caches are separate per model."""
        cache = PromptCache(max_entries_per_model=1)

        # Add to different models
        cache.put("model1", "prompt", "data1")
        cache.put("model2", "prompt", "data2")

        # Both should exist
        assert cache.get("model1", "prompt") is not None
        assert cache.get("model2", "prompt") is not None

    def test_clear_model(self):
        """Test clearing cache for a specific model."""
        cache = PromptCache()

        cache.put("model1", "prompt1", "data1")
        cache.put("model1", "prompt2", "data2")
        cache.put("model2", "prompt3", "data3")

        cache.clear_model("model1")

        assert cache.get("model1", "prompt1") is None
        assert cache.get("model1", "prompt2") is None
        assert cache.get("model2", "prompt3") is not None

    def test_clear_all(self):
        """Test clearing all caches."""
        cache = PromptCache()

        cache.put("model1", "prompt1", "data1")
        cache.put("model2", "prompt2", "data2")

        cache.clear_all()

        assert len(cache.model_caches) == 0
        assert cache.get("model1", "prompt1") is None
        assert cache.get("model2", "prompt2") is None

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = PromptCache()

        cache.put("model1", "prompt1", "data1")
        cache.put("model1", "prompt2", "data2")
        cache.put("model2", "prompt3", "data3")

        # Access some entries
        cache.get("model1", "prompt1")
        cache.get("model1", "prompt1")  # Access again

        stats = cache.get_stats()

        assert stats["_total"]["entries"] == 3
        assert "model1" in stats
        assert "model2" in stats
        assert stats["model1"]["entries"] == 2
        assert stats["model1"]["total_accesses"] == 2  # prompt1 accessed twice
        assert stats["model2"]["entries"] == 1


class TestPromptCacheDisk:
    """Test disk spill functionality."""

    def test_disk_disabled_by_default(self):
        """Test that disk is disabled by default."""
        cache = PromptCache()
        assert not cache.enable_disk

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    @patch('os.path.exists', return_value=True)
    @patch('pickle.load')
    def test_disk_spill_and_load(self, mock_pickle_load, mock_exists, mock_pickle_dump, mock_file):
        """Test disk spill and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache = PromptCache(enable_disk=True, cache_dir=cache_dir)

            # Mock the loaded entry
            mock_entry = PromptCacheEntry("test_sha", "test_model", "loaded_data")
            mock_pickle_load.return_value = mock_entry

            # Put entry (should spill to disk)
            cache.put("test_model", "test prompt", "tokenized_data")

            # Simulate cache miss and disk load
            cache.model_caches["test_model"].clear()  # Clear RAM cache
            entry = cache.get("test_model", "test prompt")

            assert entry is not None
            assert entry.tokenized_data == "loaded_data"
            assert mock_pickle_dump.called
            assert mock_pickle_load.called

    def test_disk_directory_creation(self):
        """Test that cache directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache = PromptCache(enable_disk=True, cache_dir=cache_dir)

            assert cache_dir.exists()
            assert (cache_dir / "test_model").exists()

    def test_preload_from_disk(self):
        """Test preloading cache from disk."""
        with patch('builtins.open', new_callable=mock_open) as mock_file:
            with patch('pickle.load') as mock_pickle_load:
                with patch('os.path.exists', return_value=True):
                    with patch('pathlib.Path.glob') as mock_glob:
                        # Mock glob to return some files
                        mock_glob.return_value = [Path("dummy_file.pkl")]

                        # Mock loaded entry
                        mock_entry = PromptCacheEntry("sha", "model", "data")
                        mock_pickle_load.return_value = mock_entry

                        cache = PromptCache(enable_disk=True)
                        loaded = cache.preload_from_disk("test_model")

                        assert loaded == 1
                        assert len(cache.model_caches["test_model"]) == 1


class TestPromptCacheIntegration:
    """Integration tests for prompt cache."""

    def test_repeated_prompt_caching(self):
        """Test that repeated identical prompts are cached."""
        cache = PromptCache()

        # First put
        cache.put("model", "hello world", "tokenized_v1")

        # Second put with same prompt should update
        cache.put("model", "hello world", "tokenized_v2")

        entry = cache.get("model", "hello world")
        assert entry is not None
        assert entry.tokenized_data == "tokenized_v2"

    def test_different_prompts_separate_entries(self):
        """Test that different prompts create separate entries."""
        cache = PromptCache(max_entries_per_model=5)

        cache.put("model", "prompt1", "data1")
        cache.put("model", "prompt2", "data2")

        entry1 = cache.get("model", "prompt1")
        entry2 = cache.get("model", "prompt2")

        assert entry1.tokenized_data == "data1"
        assert entry2.tokenized_data == "data2"
        assert entry1 != entry2

    def test_vision_hash_in_key(self):
        """Test that vision hash affects cache key."""
        cache = PromptCache()

        # Same prompt, different vision
        cache.put("model", "describe image", "data1", images_hash="hash1")
        cache.put("model", "describe image", "data2", images_hash="hash2")

        entry1 = cache.get("model", "describe image", images_hash="hash1")
        entry2 = cache.get("model", "describe image", images_hash="hash2")

        assert entry1.tokenized_data == "data1"
        assert entry2.tokenized_data == "data2"

    def test_tool_schema_hash_in_key(self):
        """Test that tool schema hash affects cache key."""
        cache = PromptCache()

        # Same prompt, different tool schema
        cache.put("model", "use tool", "data1", tool_schema_hash="schema1")
        cache.put("model", "use tool", "data2", tool_schema_hash="schema2")

        entry1 = cache.get("model", "use tool", tool_schema_hash="schema1")
        entry2 = cache.get("model", "use tool", tool_schema_hash="schema2")

        assert entry1.tokenized_data == "data1"
        assert entry2.tokenized_data == "data2"


if __name__ == "__main__":
    pytest.main([__file__])