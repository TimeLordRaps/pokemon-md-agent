"""Cache classes for PMD-Red agent caching functionality."""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging
import hashlib
import pickle
import mmap
import time
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheTelemetry:
    """Telemetry for cache operations."""
    hits: int = 0
    misses: int = 0
    latency_deltas: List[float] = None  # type: ignore

    def __post_init__(self):
        if self.latency_deltas is None:
            self.latency_deltas = []

    def reset(self) -> None:
        """Reset telemetry counters."""
        self.hits = 0
        self.misses = 0
        self.latency_deltas.clear()


class VisionCache:
    """LRU cache for pre-encoded image tensors keyed by SHA256."""

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.ram_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.telemetry = CacheTelemetry()

    def get_encoded_image(self, image_sha: str) -> Optional[Any]:
        """Get encoded image tensor from cache."""
        start_time = time.time()
        cached = self.ram_cache.get(image_sha)
        latency = time.time() - start_time
        self.telemetry.latency_deltas.append(latency)

        if cached is not None:
            self.ram_cache.move_to_end(image_sha)
            self.telemetry.hits += 1
            logger.debug("Vision cache hit for %s", image_sha)
            return cached

        self.telemetry.misses += 1
        logger.debug("Vision cache miss for %s", image_sha)
        return None

    def cache_encoded_image(self, image_sha: str, encoded_tensor: Any) -> None:
        """Cache encoded image tensor."""
        self.ram_cache[image_sha] = encoded_tensor
        self.ram_cache.move_to_end(image_sha)

        # LRU eviction
        while len(self.ram_cache) > self.max_entries:
            evicted_key, _ = self.ram_cache.popitem(last=False)
            logger.debug("Evicted vision encoding %s from RAM cache", evicted_key)


class PromptKVCache:
    """LRU cache for prompt KV states with disk spill to .cache/prompt_kv/."""

    def __init__(self, cache_dir: Path, max_ram_entries: int = 5):
        self.cache_dir = cache_dir / "prompt_kv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_ram_entries = max_ram_entries
        self.ram_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.telemetry = CacheTelemetry()

    def _make_cache_key(self, model_name: str, prompt_sha: str, image_sha: Optional[str] = None) -> str:
        """Generate cache key from components."""
        safe_model = model_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        image_part = f"_{image_sha}" if image_sha else ""
        return f"{safe_model}_{prompt_sha}{image_part}"

    def get_kv_state(self, cache_key: str) -> Optional[Any]:
        """Get KV state from RAM or disk."""
        start_time = time.time()

        # Check RAM first
        cached = self.ram_cache.get(cache_key)
        if cached is not None:
            self.ram_cache.move_to_end(cache_key)
            latency = time.time() - start_time
            self.telemetry.latency_deltas.append(latency)
            self.telemetry.hits += 1
            logger.debug("KV cache hit for %s", cache_key)
            return cached

        # Check disk
        cache_file = self.cache_dir / f"{cache_key}.mm"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                    size = int.from_bytes(mm[:8], "little")
                    payload = mm[8 : 8 + size]
                    kv_state = pickle.loads(payload)
                    mm.close()

                self._insert_ram(cache_key, kv_state)
                latency = time.time() - start_time
                self.telemetry.latency_deltas.append(latency)
                self.telemetry.hits += 1
                logger.debug("KV cache hit from disk for %s", cache_key)
                return kv_state
            except Exception as exc:
                logger.warning("Failed to load KV cache %s: %s", cache_key, exc)

        latency = time.time() - start_time
        self.telemetry.latency_deltas.append(latency)
        self.telemetry.misses += 1
        logger.debug("KV cache miss for %s", cache_key)
        return None

    def cache_kv_state(self, cache_key: str, kv_state: Any) -> None:
        """Cache KV state to RAM and disk."""
        self._insert_ram(cache_key, kv_state)

        # Write to disk
        cache_file = self.cache_dir / f"{cache_key}.mm"
        try:
            data = pickle.dumps(kv_state, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cache_file, "wb") as f:
                f.write(len(data).to_bytes(8, "little"))
                f.write(data)
            logger.debug("Cached KV state to disk: %s", cache_file)
        except Exception as exc:
            logger.warning("Failed to save KV cache %s: %s", cache_key, exc)

    def _insert_ram(self, key: str, value: Any) -> None:
        """Insert value into RAM cache with LRU eviction."""
        self.ram_cache[key] = value
        self.ram_cache.move_to_end(key)
        while len(self.ram_cache) > self.max_ram_entries:
            evicted_key, _ = self.ram_cache.popitem(last=False)
            logger.debug("Evicted KV state %s from RAM cache", evicted_key)


class PromptCache:
    """Pre-tokenized prefix cache with RAM LRU and disk memmap."""

    def __init__(self, cache_dir: Path, max_ram_entries: int = 1000):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_ram_entries = max_ram_entries
        self.ram_cache: "OrderedDict[str, Any]" = OrderedDict()

    def get_tokenized_prefix(self, prompt_sha: str, model_name: str) -> Optional[Any]:
        """Get tokenized prefix from RAM cache or disk."""
        cached = self.ram_cache.get(prompt_sha)
        if cached is not None:
            self.ram_cache.move_to_end(prompt_sha)
            return cached

        cache_file = self.cache_dir / f"{model_name}_{prompt_sha}.mm"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                size = int.from_bytes(mm[:8], "little")
                payload = mm[8 : 8 + size]
                data = pickle.loads(payload)
                mm.close()
        except Exception as exc:
            logger.warning("Failed to load cached prefix %s: %s", prompt_sha, exc)
            return None

        self._insert_ram(prompt_sha, data)
        return data

    def cache_tokenized_prefix(self, prompt_sha: str, model_name: str, tokenized: Any) -> None:
        """Cache tokenized prefix to RAM and disk."""
        self._insert_ram(prompt_sha, tokenized)

        # Sanitize model_name for filename (replace slashes and spaces)
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        cache_file = self.cache_dir / f"{safe_model_name}_{prompt_sha}.mm"
        try:
            data = pickle.dumps(tokenized, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cache_file, "wb") as f:
                f.write(len(data).to_bytes(8, "little"))
                f.write(data)
        except Exception as exc:
            logger.warning("Failed to save cached prefix %s: %s", prompt_sha, exc)

    def _insert_ram(self, key: str, value: Any) -> None:
        """Insert value into RAM cache with LRU eviction."""
        self.ram_cache[key] = value
        self.ram_cache.move_to_end(key)
        while len(self.ram_cache) > self.max_ram_entries:
            evicted_key, _ = self.ram_cache.popitem(last=False)
            logger.debug("Evicted prefix %s from RAM cache", evicted_key)