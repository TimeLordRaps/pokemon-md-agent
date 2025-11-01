"""Prompt cache with LRU per model (2-5 entries) RAM + optional disk spill.

Implements KV cache for Qwen3-VL with SHA256-normalized keys.
Thread-safe with proper exception handling and memory management.
"""

import hashlib
import pickle
import time
import threading
import weakref
from typing import Optional, Any, Dict
from pathlib import Path
from collections import OrderedDict
import logging
import os

logger = logging.getLogger(__name__)


class PromptCacheEntry:
    """Entry in prompt cache."""

    def __init__(self, prompt_sha: str, model_name: str, tokenized_data: Any,
                 kv_cache: Optional[Any] = None, vision_features: Optional[Any] = None):
        self.prompt_sha = prompt_sha
        self.model_name = model_name
        self.tokenized_data = tokenized_data
        self.kv_cache = kv_cache
        self.vision_features = vision_features
        self.timestamp = time.time()
        self.access_count = 0

    def touch(self) -> None:
        """Update access time."""
        self.timestamp = time.time()
        self.access_count += 1


class PromptCache:
    """LRU prompt cache per model with disk spill. Thread-safe."""

    def __init__(self, max_entries_per_model: int = 5, enable_disk: bool = False,
                  cache_dir: Optional[Path] = None):
        """Initialize prompt cache.

        Args:
            max_entries_per_model: Max entries per model (2-5 recommended)
            enable_disk: Enable disk spill to .cache/prompt_cache/
            cache_dir: Cache directory (auto-created)
        """
        if max_entries_per_model < 2 or max_entries_per_model > 5:
            raise ValueError("max_entries_per_model must be between 2 and 5")

        self.max_entries_per_model = max_entries_per_model
        self.enable_disk = enable_disk

        env_cache_dir = os.environ.get("PROMPT_CACHE_DIR")
        resolved_cache_dir: Optional[Path]
        if env_cache_dir:
            sanitized = env_cache_dir.strip().strip('"').strip("'")
            resolved_cache_dir = Path(sanitized).expanduser()
        elif cache_dir is not None:
            resolved_cache_dir = Path(cache_dir)
        else:
            resolved_cache_dir = None

        self.cache_dir = resolved_cache_dir or Path.home() / ".cache" / "pmd_prompt_cache"
        self.model_caches: Dict[str, OrderedDict[str, PromptCacheEntry]] = {}
        self._lock = threading.RLock()  # Allow recursive locking for nested operations

        if self.enable_disk:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to create cache directory {self.cache_dir}: {e}")
                self.enable_disk = False

        logger.info(f"Initialized PromptCache with {max_entries_per_model} entries per model, disk={enable_disk}")

    def _get_model_cache(self, model_name: str) -> OrderedDict[str, PromptCacheEntry]:
        """Get or create model-specific LRU cache."""
        if model_name not in self.model_caches:
            self.model_caches[model_name] = OrderedDict()
        return self.model_caches[model_name]

    def _make_key(self, prompt: str, images_hash: Optional[str] = None,
                   tool_schema_hash: Optional[str] = None) -> str:
        """Generate SHA256 cache key from prompt components."""
        # Normalize prompt for consistent hashing
        normalized_prompt = prompt.strip().lower()

        # Include vision and tool hashes if present
        components = [normalized_prompt]
        if images_hash:
            components.append(images_hash)
        if tool_schema_hash:
            components.append(tool_schema_hash)

        combined = "|".join(components)
        # Return first 16 chars for consistent key length
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, model_name: str, prompt: str, images_hash: Optional[str] = None,
            tool_schema_hash: Optional[str] = None) -> Optional[PromptCacheEntry]:
        """Get cached entry, checking RAM then disk. Thread-safe."""
        with self._lock:
            key = self._make_key(prompt, images_hash, tool_schema_hash)
            cache = self._get_model_cache(model_name)

            # Check RAM cache
            entry = cache.get(key)
            if entry:
                entry.touch()
                cache.move_to_end(key)  # LRU
                logger.debug(f"Prompt cache hit (RAM): {model_name}/{key}")
                return entry

            # Check disk if enabled
            if self.enable_disk:
                entry = self._load_from_disk(model_name, key)
                if entry:
                    # Promote to RAM cache
                    self._put_in_cache(model_name, key, entry)
                    logger.debug(f"Prompt cache hit (disk): {model_name}/{key}")
                    return entry

            logger.debug(f"Prompt cache miss: {model_name}/{key}")
            return None

    def put(self, model_name: str, prompt: str, tokenized_data: Any,
            kv_cache: Optional[Any] = None, vision_features: Optional[Any] = None,
            images_hash: Optional[str] = None, tool_schema_hash: Optional[str] = None) -> None:
        """Cache entry in RAM and optionally disk. Thread-safe."""
        with self._lock:
            key = self._make_key(prompt, images_hash, tool_schema_hash)

            entry = PromptCacheEntry(
                prompt_sha=key,
                model_name=model_name,
                tokenized_data=tokenized_data,
                kv_cache=kv_cache,
                vision_features=vision_features
            )

            self._put_in_cache(model_name, key, entry)

            # Spill to disk if enabled
            if self.enable_disk:
                self._save_to_disk(model_name, key, entry)

    def _put_in_cache(self, model_name: str, key: str, entry: PromptCacheEntry) -> None:
        """Put entry in RAM cache with LRU eviction."""
        cache = self._get_model_cache(model_name)
        cache[key] = entry
        cache.move_to_end(key)  # Most recently used

        # Evict if over limit
        while len(cache) > self.max_entries_per_model:
            evicted_key, _ = cache.popitem(last=False)  # LRU
            logger.debug(f"Evicted prompt cache entry: {model_name}/{evicted_key[:8]}")

    def _load_from_disk(self, model_name: str, key: str) -> Optional[PromptCacheEntry]:
        """Load entry from disk cache with robust exception handling."""
        if not self.enable_disk:
            return None

        model_dir = self.cache_dir / model_name.replace('/', '_')
        cache_file = model_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            # Validate loaded data
            if not isinstance(data, PromptCacheEntry):
                logger.warning(f"Invalid cache file {cache_file}: not a PromptCacheEntry")
                cache_file.unlink(missing_ok=True)  # Remove corrupted file
                return None
            logger.debug(f"Loaded prompt cache from disk: {cache_file}")
            return data
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logger.warning(f"Failed to load prompt cache {cache_file}: {e}")
            # Remove corrupted file
            cache_file.unlink(missing_ok=True)
            return None
        except (OSError, PermissionError) as e:
            logger.error(f"File system error loading cache {cache_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading cache {cache_file}: {e}")
            return None

    def _save_to_disk(self, model_name: str, key: str, entry: PromptCacheEntry) -> None:
        """Save entry to disk cache with robust exception handling."""
        if not self.enable_disk:
            return

        model_dir = self.cache_dir / model_name.replace('/', '_')
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create cache directory {model_dir}: {e}")
            return

        cache_file = model_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved prompt cache to disk: {cache_file}")
        except (pickle.PicklingError, OSError, PermissionError) as e:
            logger.error(f"Failed to save prompt cache {cache_file}: {e}")
            # Clean up partial file if it exists
            cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Unexpected error saving cache {cache_file}: {e}")
            cache_file.unlink(missing_ok=True)

    def clear_model(self, model_name: str) -> None:
        """Clear all entries for a model. Thread-safe."""
        with self._lock:
            if model_name in self.model_caches:
                count = len(self.model_caches[model_name])
                self.model_caches[model_name].clear()
                logger.info(f"Cleared {count} prompt cache entries for {model_name}")

    def clear_all(self) -> None:
        """Clear all cache entries. Thread-safe."""
        with self._lock:
            total_entries = sum(len(cache) for cache in self.model_caches.values())
            self.model_caches.clear()
            logger.info(f"Cleared {total_entries} prompt cache entries from all models")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics. Thread-safe."""
        with self._lock:
            stats = {}
            total_entries = 0

            for model_name, cache in self.model_caches.items():
                if cache:  # Only compute stats if cache has entries
                    timestamps = [entry.timestamp for entry in cache.values()]
                    access_counts = [entry.access_count for entry in cache.values()]
                    stats[model_name] = {
                        "entries": len(cache),
                        "total_accesses": sum(access_counts),
                        "oldest_timestamp": min(timestamps) if timestamps else None,
                        "newest_timestamp": max(timestamps) if timestamps else None,
                    }
                else:
                    stats[model_name] = {
                        "entries": 0,
                        "total_accesses": 0,
                        "oldest_timestamp": None,
                        "newest_timestamp": None,
                    }
                total_entries += len(cache)

            stats["_total"] = {"entries": total_entries}
            return stats

    def preload_from_disk(self, model_name: str) -> int:
        """Preload cache entries from disk for model. Thread-safe."""
        with self._lock:
            if not self.enable_disk:
                return 0

            model_dir = self.cache_dir / model_name.replace('/', '_')
            if not model_dir.exists():
                return 0

            loaded = 0
            cache = self._get_model_cache(model_name)

            try:
                for cache_file in model_dir.glob("*.pkl"):
                    key = cache_file.stem
                    if key not in cache:  # Don't overwrite RAM entries
                        entry = self._load_from_disk(model_name, key)
                        if entry:
                            cache[key] = entry
                            cache.move_to_end(key)
                            loaded += 1

                            # Respect RAM limit - evict LRU if over limit
                            while len(cache) > self.max_entries_per_model:
                                evicted_key, _ = cache.popitem(last=False)
                                logger.debug(f"Evicted during preload: {model_name}/{evicted_key}")

            except Exception as e:
                logger.warning(f"Error preloading cache for {model_name}: {e}")

            logger.info(f"Preloaded {loaded} prompt cache entries for {model_name}")
            return loaded
