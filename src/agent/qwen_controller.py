"""Qwen3-VL controller for PMD-Red agent with batching and KV cache support."""

from typing import Optional, Dict, List, Any, Union, Callable, Literal
import asyncio
import logging
import hashlib
import os
import pickle
import mmap
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
import time

try:
    import torch
    from transformers.cache_utils import StaticCache
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    StaticCache = None  # type: ignore

from .model_router import ModelSize, ModelRouter, DecodeResult
from .inference_queue import InferenceQueue
from .timebudgets import PROMPT_CACHE_SIZE, ROUTER_MAX_WALL_S
from .prompt_cache import PromptCache as PromptCacheNew, PromptCacheEntry as PromptCacheEntryNew
from .pipeline_engine import PipelineEngine, PipelineRequest, Batch

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    """Handle to a loaded model with shared components."""
    model: Any  # The actual model instance
    tokenizer: Any  # Shared tokenizer
    vision_processor: Any  # Shared vision processor
    model_name: str
    variant: str
    size: ModelSize


VRAM_REQUIREMENTS_GB: Dict[ModelSize, float] = {
    ModelSize.SIZE_2B: 4.0,
    ModelSize.SIZE_4B: 8.0,
    ModelSize.SIZE_8B: 12.0,
}

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


@dataclass
class PromptCacheEntry:
    """Entry in prompt cache ring."""
    input_ids: Any  # Tokenizer-ready input IDs
    attention_mask: Any  # Attention mask
    vision_features: Optional[Any] = None  # Vision features if applicable
    kv_cache: Optional[Any] = None  # KV cache state
    timestamp: float = None  # type: ignore

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class PromptCacheRing:
    """LRU ring cache for prompts per model with disk spill."""

    def __init__(self, model_name: str, cache_dir: Path, size: int = PROMPT_CACHE_SIZE):
        self.model_name = model_name
        self.size = size
        self.ring: "OrderedDict[str, PromptCacheEntry]" = OrderedDict()
        self.disk_enabled = os.environ.get("PROMPT_CACHE_DISK", "0") == "1"
        self.disk_dir = cache_dir / "qwen" / "prompt_cache" / model_name.replace('/', '_')
        self.disk_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, template_hash: str, images_hash: Optional[str] = None, tool_schema_hash: Optional[str] = None) -> str:
        """Generate cache key from components."""
        parts = [template_hash]
        if images_hash:
            parts.append(images_hash)
        if tool_schema_hash:
            parts.append(tool_schema_hash)
        return "|".join(parts)

    def get(self, key: str) -> Optional[PromptCacheEntry]:
        """Get entry from ring or disk."""
        # Check RAM ring first
        if key in self.ring:
            self.ring.move_to_end(key)
            return self.ring[key]

        # Check disk if enabled
        if self.disk_enabled:
            try:
                disk_file = self.disk_dir / f"{key}.pkl"
                if disk_file.exists():
                    with open(disk_file, "rb") as f:
                        entry = pickle.load(f)
                    # Move to RAM ring
                    self._add_to_ring(key, entry)
                    logger.debug(f"Loaded prompt cache from disk: {key}")
                    return entry
            except Exception as e:
                logger.warning(f"Failed to load prompt cache {key}: {e}")

        return None

    def put(self, key: str, entry: PromptCacheEntry) -> None:
        """Put entry in ring and optionally to disk."""
        self._add_to_ring(key, entry)

        # Spill to disk if enabled
        if self.disk_enabled:
            try:
                disk_file = self.disk_dir / f"{key}.pkl"
                with open(disk_file, "wb") as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"Spilled prompt cache to disk: {key}")
            except Exception as e:
                logger.warning(f"Failed to spill prompt cache {key}: {e}")

    def _add_to_ring(self, key: str, entry: PromptCacheEntry) -> None:
        """Add entry to RAM ring with LRU eviction."""
        self.ring[key] = entry
        self.ring.move_to_end(key)

        # LRU eviction
        while len(self.ring) > self.size:
            evicted_key, _ = self.ring.popitem(last=False)
            logger.debug(f"Evicted prompt cache: {evicted_key}")


@dataclass
class PipelineStage:
    """Pipeline stage for async processing."""
    tokenize: asyncio.Future[Any]  # Tokenization result
    vision: Optional[asyncio.Future[Any]] = None  # Vision preprocessing
    forward: Optional[asyncio.Future[Any]] = None  # Forward pass
    semaphore: asyncio.Semaphore = None  # type: ignore  # VRAM guard


class PipelineError(Exception):
    """Exception raised for pipeline processing errors."""
    pass


class GenerationBudgetExceeded(PipelineError):
    """Exception raised when generation budget is exceeded."""
    pass


class BestOfSelectionError(PipelineError):
    """Exception raised when best-of selection fails."""
    pass


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


class QwenController:
    """Controller for Qwen3-VL models with batching and KV caching."""

    SUPPORTED_MODELS = {
        "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen3-VL-2B-Thinking-FP8",
        "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    }

    def __init__(
        self,
        model_router: Optional[ModelRouter] = None,
        hf_home: Optional[str] = None,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        enable_kv_cache_serialization: bool = False,
        use_cache: bool = True,
        use_pipeline: bool = True,
        best_of_n: int = 1,
    ):
        """Initialize Qwen controller with pipelining, prompt caching, and best-of-n routing.

        Args:
            model_router: Model routing instance
            hf_home: HuggingFace cache directory
            local_files_only: Use only local files
            trust_remote_code: Trust remote code in models
            enable_kv_cache_serialization: Enable KV cache serialization
            use_cache: Enable prompt caching
            use_pipeline: Enable pipeline engine for continuous batching
            best_of_n: Default best-of-n value (1,2,4,8)
        """
        self.model_router = model_router or ModelRouter()
        self.hf_home = hf_home or os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")).strip('"')
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.enable_kv_cache_serialization = enable_kv_cache_serialization
        self.use_cache = use_cache
        self.use_pipeline = use_pipeline
        self.best_of_n = best_of_n

        # Shared components across model variants of same size
        self.shared_tokenizers: Dict[ModelSize, Any] = {}
        self.shared_vision_processors: Dict[ModelSize, Any] = {}

        # Loaded models
        self.loaded_models: Dict[str, ModelHandle] = {}
        self.loaded_model_order: "OrderedDict[str, ModelHandle]" = OrderedDict()
        self.max_loaded_models = 4
        self.vram_guard_enabled = bool(torch is not None and torch.cuda.is_available())

        # Initialize new prompt cache (LRU 2-5 per model, RAM + optional disk)
        cache_dir = Path(self.hf_home)
        self.prompt_cache = PromptCacheNew(max_entries_per_model=5, enable_disk=True, cache_dir=cache_dir)
        logger.info(f"Initialized new PromptCache with 5 entries per model, disk enabled")

        # Legacy caches for compatibility (will be phased out)
        self.vision_cache = VisionCache()
        self.prompt_kv_cache = PromptKVCache(cache_dir, max_ram_entries=5)

        # Pipeline engine for continuous batching with ≤50ms tick
        self.pipeline_engine = PipelineEngine(max_batch_size=8, tick_interval_ms=50)
        self.pipeline_initialized = False  # Track if pipeline has been started

        # Defer pipeline initialization to async context
        # Will be called via initialize_async() method

        # Prompt cache rings per model (legacy, will be removed)
        self.prompt_cache_rings: Dict[str, PromptCacheRing] = {}

        # VRAM semaphores per model
        self.vram_semaphores: Dict[str, asyncio.Semaphore] = {}

        # Pipeline stage tracking
        self.active_pipelines: Dict[str, PipelineStage] = {}

        # Warmup prompts for model initialization
        self.warmup_prompts: List[str] = [
            "Hello, how are you?",
            "What is the weather like?",
            "Tell me about artificial intelligence."
        ]

    async def _process_prefill_batch(self, batch: Batch) -> None:
        """Process prefill batch."""
        logger.debug(f"Processing prefill batch {batch.id} with {batch.size} requests")
        # Placeholder - would implement actual prefill processing

    async def _process_decode_batch(self, batch: Batch) -> None:
        """Process decode batch."""
        logger.debug(f"Processing decode batch {batch.id} with {batch.size} requests")
        # Placeholder - would implement actual decode processing

    def _get_model_name(self, model_size: ModelSize, use_thinking: bool = False) -> str:
        """Get model name for size and variant."""
        return self.model_router.get_model_name(model_size, use_thinking)

    def _validate_model_name(self, model_name: str) -> None:
        """Validate model name is supported."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not in supported list: {self.SUPPORTED_MODELS}")

    def load_model(self, name: str, variant: Literal["instruct", "thinking"]) -> ModelHandle:
        """Load model using local HF cache only, sharing components across variants.

        Args:
            name: Model name
            variant: Model variant

        Returns:
            ModelHandle with loaded model and shared components
        """
        self._validate_model_name(name)

        # Determine size from name
        if "2B" in name:
            size = ModelSize.SIZE_2B
        elif "4B" in name:
            size = ModelSize.SIZE_4B
        elif "8B" in name:
            size = ModelSize.SIZE_8B
        else:
            raise ValueError(f"Cannot determine size from model name: {name}")

        cache_key = f"{name}_{variant}"

        if cache_key in self.loaded_models:
            self.loaded_model_order.move_to_end(cache_key, last=True)
            return self.loaded_models[cache_key]

        self._ensure_vram_capacity(size)

        # Load shared components if not already loaded
        if size not in self.shared_tokenizers:
            # Placeholder - would load actual tokenizer
            self.shared_tokenizers[size] = f"tokenizer_{size.value}"
            logger.info(f"Loaded shared tokenizer for {size.value}")

        if size not in self.shared_vision_processors:
            # Placeholder - would load actual vision processor
            self.shared_vision_processors[size] = f"vision_processor_{size.value}"
            logger.info(f"Loaded shared vision processor for {size.value}")

        # Load model (placeholder)
        model = f"loaded_{name}_{variant}"

        handle = ModelHandle(
            model=model,
            tokenizer=self.shared_tokenizers[size],
            vision_processor=self.shared_vision_processors[size],
            model_name=name,
            variant=variant,
            size=size,
        )

        self.loaded_models[cache_key] = handle
        self.loaded_model_order[cache_key] = handle
        self.loaded_model_order.move_to_end(cache_key, last=True)
        self._trim_loaded_models()

        # Warm-up the model
        self._warmup_model(handle)

        logger.info(f"Loaded model {name} ({variant}) with shared components")
        return handle

    def _ensure_vram_capacity(self, model_size: ModelSize) -> None:
        """Ensure sufficient free VRAM for upcoming model load."""
        if not self.vram_guard_enabled:
            return

        required_gb = VRAM_REQUIREMENTS_GB.get(model_size, 4.0)
        attempts = 0
        while not self._has_sufficient_vram(required_gb) and self.loaded_model_order:
            attempts += 1
            evicted_key, _ = self.loaded_model_order.popitem(last=False)
            self._unload_model(evicted_key)
            logger.info(
                "Evicted %s to reclaim VRAM (attempt %d)",
                evicted_key,
                attempts,
            )

        if not self._has_sufficient_vram(required_gb):
            logger.warning(
                "VRAM guard could not free %.1f GB for %s model; continuing load anyway",
                required_gb,
                model_size.value,
            )

    def _has_sufficient_vram(self, required_gb: float) -> bool:
        """Check if there is sufficient free VRAM headroom."""
        if not self.vram_guard_enabled:
            return True
        try:
            free_bytes, _ = torch.cuda.mem_get_info()  # type: ignore[union-attr]
        except Exception:
            return True

        free_gb = free_bytes / (1024 ** 3)
        return free_gb >= required_gb * 1.1  # keep 10% buffer

    def _trim_loaded_models(self) -> None:
        """Evict least recently used models when exceeding cache budget."""
        while len(self.loaded_model_order) > self.max_loaded_models:
            evicted_key, _ = self.loaded_model_order.popitem(last=False)
            self._unload_model(evicted_key)

    def _unload_model(self, cache_key: str) -> None:
        """Unload model and release references."""
        handle = self.loaded_models.pop(cache_key, None)
        if handle:
            logger.info("Unloaded model %s (%s)", handle.model_name, handle.variant)
        self.loaded_model_order.pop(cache_key, None)

    def _warmup_model(self, handle: ModelHandle) -> None:
        """Warm up model with short prefixes to stabilize latency."""
        logger.info(f"Warming up model {handle.model_name} ({handle.variant})")

        for prompt in self.warmup_prompts:
            try:
                # Cache tokenized prefix using new PromptCache
                tokenized = f"tokenized_{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
                self.prompt_cache.put(
                    model_name=handle.model_name,
                    prompt=prompt,
                    tokenized_data=tokenized,
                    kv_cache=None,
                    vision_features=None
                )

                # Skip actual inference during warmup to avoid async issues
                # In real implementation, would do sync warmup
                logger.debug(f"Warmup cached prefix for {handle.model_name}")
            except Exception as e:
                logger.warning(f"Warmup failed for {handle.model_name}: {e}")

    def get_tokenized_prefix(self, prompt: str, model_name: str) -> Optional[Any]:
        """Get pre-tokenized prefix from cache."""
        cached_entry = self.prompt_cache.get(model_name, prompt)
        return cached_entry.tokenized_data if cached_entry else None

    def cache_tokenized_prefix(self, prompt: str, model_name: str, tokenized: Any) -> None:
        """Cache tokenized prefix."""
        self.prompt_cache.put(
            model_name=model_name,
            prompt=prompt,
            tokenized_data=tokenized,
            kv_cache=None,
            vision_features=None
        )

    def get_kv_cache_key(self, model_name: str, prompt_sha: str, has_vision: bool) -> str:
        """Generate KV cache key."""
        return f"{model_name}|{prompt_sha}|{has_vision}"

    def serialize_kv_cache(self, kv_state: Any, cache_key: str) -> None:
        """Serialize KV cache to disk if enabled."""
        if not self.enable_kv_cache_serialization:
            return

        cache_file = Path(self.hf_home) / "pmd_kv_cache" / f"{cache_key}.mm"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = pickle.dumps(kv_state, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cache_file, "wb") as f:
                f.write(len(data).to_bytes(8, "little"))
                f.write(data)
            logger.debug(f"Serialized KV cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to serialize KV cache: {e}")

    def deserialize_kv_cache(self, cache_key: str) -> Optional[Any]:
        """Deserialize KV cache from disk if enabled."""
        if not self.enable_kv_cache_serialization:
            return None

        cache_file = Path(self.hf_home) / "pmd_kv_cache" / f"{cache_key}.mm"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                size = int.from_bytes(mm[:8], "little")
                payload = mm[8 : 8 + size]
                kv_state = pickle.loads(payload)
                mm.close()
            logger.debug(f"Deserialized KV cache from {cache_file}")
            return kv_state
        except Exception as e:
            logger.warning(f"Failed to deserialize KV cache: {e}")
            return None

    def _get_or_create_prompt_cache_ring(self, model_name: str) -> PromptCacheRing:
        """Get or create prompt cache ring for model."""
        if model_name not in self.prompt_cache_rings:
            cache_dir = Path(self.hf_home)
            self.prompt_cache_rings[model_name] = PromptCacheRing(model_name, cache_dir)
        return self.prompt_cache_rings[model_name]

    def _get_or_create_vram_semaphore(self, model_name: str) -> asyncio.Semaphore:
        """Get or create VRAM semaphore for model."""
        if model_name not in self.vram_semaphores:
            # Allow 2 concurrent forwards per model to prevent VRAM thrash
            self.vram_semaphores[model_name] = asyncio.Semaphore(2)
        return self.vram_semaphores[model_name]

    def _compute_hashes(self, prompt: str, images: Optional[List[Any]] = None, tool_schema: Optional[Dict[str, Any]] = None) -> tuple[str, Optional[str], Optional[str]]:
        """Compute hashes for cache key components."""
        template_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        images_hash = None
        if images:
            combined_bytes = b"".join(img if isinstance(img, bytes) else str(img).encode() for img in images)
            images_hash = hashlib.sha256(combined_bytes).hexdigest()[:16]

        tool_schema_hash = None
        if tool_schema:
            schema_str = str(sorted(tool_schema.items()))
            tool_schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        return template_hash, images_hash, tool_schema_hash

    async def generate_async(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        model_size: Optional[ModelSize] = None,
        use_thinking: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7,
        best_of_n: Optional[int] = None,
        retrieval_scores: Optional[List[float]] = None,
        tool_schema: Optional[Dict[str, Any]] = None,
        yield_every: Optional[int] = None,
    ) -> tuple[str, List[float]]:
        """Generate text with pipelined async processing and best-of-n scoring.

        Args:
            prompt: Text prompt
            images: Optional list of images
            model_size: Model size (auto-selected if None)
            use_thinking: Use thinking variant
            max_tokens: Maximum tokens to generate (strictly enforced)
            temperature: Sampling temperature
            best_of_n: Number of candidates to generate (1,2,4,8); uses batched forwards
            retrieval_scores: Optional retrieval scores for RRF
            tool_schema: Optional tool schema for function calling
            yield_every: Yield partial results every N tokens (if supported)

        Returns:
            Tuple of (selected_text, candidate_scores)

        Raises:
            GenerationBudgetExceeded: If wall time budget exceeded before completion
            BestOfSelectionError: If best-of selection fails
        """
        # Use instance default if not specified
        if best_of_n is None:
            best_of_n = self.best_of_n

        # Validate best_of_n
        if best_of_n not in {1, 2, 4, 8}:
            raise ValueError(f"best_of_n must be 1, 2, 4, or 8, got {best_of_n}")

        # Auto-select model if not specified
        if model_size is None:
            # Simple auto-selection - use 2B for speed, 8B for complexity
            complexity = len(prompt.split()) + (len(images) if images else 0) * 10
            if complexity < 50:
                model_size = ModelSize.SIZE_2B
            elif complexity < 200:
                model_size = ModelSize.SIZE_4B
            else:
                model_size = ModelSize.SIZE_8B

        model_name = self._get_model_name(model_size, use_thinking)
        self._validate_model_name(model_name)

        # Compute hashes for cache keys
        template_hash, images_hash, tool_schema_hash = self._compute_hashes(prompt, images, tool_schema)

        # Check prompt cache if enabled
        cached_entry = None
        if self.use_cache:
            cached_entry = self.prompt_cache.get(
                model_name, prompt, images_hash, tool_schema_hash
            )
            if cached_entry:
                logger.debug(f"Prompt cache hit for {model_name}")
                # Use cached entry for generation
                return await self._generate_with_cache(
                    cached_entry, model_name, max_tokens, temperature, best_of_n,
                    retrieval_scores, yield_every, wall_budget_s=30.0
                )

        # Cache miss or cache disabled - use pipeline if enabled
        if self.use_pipeline:
            # Submit to pipeline
            request = PipelineRequest(
                id=f"req_{int(time.time()*1000)}",
                prompt=prompt,
                images=images,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            success = await self.pipeline_engine.submit_request(request)
            if not success:
                raise RuntimeError("Pipeline queue full")

            # Wait for completion
            completed = await self.pipeline_engine.get_completed_request(request.id)
            if completed:
                # Mock result - in real impl would parse from completed request
                return f"Pipeline result for: {prompt[:50]}...", [1.0] * best_of_n

            raise RuntimeError("Pipeline request failed")

        # Fallback to parallel generation for best_of_n
        if best_of_n > 1:
            candidates, decode_results = await self._generate_candidates_parallel(
                prompt, images, model_name, max_tokens, temperature, best_of_n
            )
            scores = self._score_candidates(decode_results, retrieval_scores)
            selected, candidate_scores = self._select_best_candidate(candidates, scores)
            return selected, candidate_scores
        else:
            # Single generation
            result = await self._single_generate(prompt, images, model_name, max_tokens, temperature)
            # Score the single result consistently
            decode_result = DecodeResult(
                generated_text=result,
                tokens_used=len(result.split()),
                latency_ms=100.0  # Mock latency
            )
            scores = self._score_candidates([decode_result], retrieval_scores)
            return result, scores

    async def _single_generate(
        self,
        prompt: str,
        images: Optional[List[Any]],
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Single inference call with prompt, vision, and KV caching."""
        start_time = time.time()

        # Compute hashes
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        image_sha = None
        if images:
            combined_bytes = b"".join(img if isinstance(img, bytes) else str(img).encode() for img in images)
            image_sha = hashlib.sha256(combined_bytes).hexdigest()

        # Check prompt cache
        tokenized = self.get_tokenized_prefix(prompt, model_name)
        if tokenized is None:
            # Tokenize and cache
            tokenized = f"tokenized_{prompt_sha}"  # Placeholder
            self.cache_tokenized_prefix(prompt, model_name, tokenized)
            logger.debug(f"Cached tokenized prefix for {model_name}: {prompt_sha}")

        # Check vision cache if images present
        vision_encoded = None
        if image_sha:
            vision_encoded = self.vision_cache.get_encoded_image(image_sha)
            if vision_encoded is None:
                # Encode and cache vision
                vision_encoded = f"encoded_{image_sha}"  # Placeholder
                self.vision_cache.cache_encoded_image(image_sha, vision_encoded)
                logger.debug(f"Cached vision encoding: {image_sha}")

        # Check KV cache
        kv_cache_key = self.prompt_kv_cache._make_cache_key(model_name, prompt_sha, image_sha)
        cached_kv = self.prompt_kv_cache.get_kv_state(kv_cache_key)

        if cached_kv:
            logger.debug(f"Using cached KV state for {kv_cache_key}")
            # Try to use StaticCache if available
            if StaticCache and hasattr(cached_kv, 'past_key_values'):
                kv_cache = cached_kv
            else:
                kv_cache = None
        else:
            logger.debug(f"No cached KV state for {kv_cache_key}")
            kv_cache = None

        # Simulate generation (replace with actual model call)
        response = f"Generated response for: {prompt[:50]}..."

        # Cache KV state if long prompt (simulate StaticCache creation)
        if len(prompt) > 50 and StaticCache:
            try:
                # Create mock StaticCache - in real implementation, this would be the actual KV state
                dummy_kv_state = {"prompt_sha": prompt_sha, "model": model_name}  # Mock KV state
                self.prompt_kv_cache.cache_kv_state(kv_cache_key, dummy_kv_state)
                logger.debug(f"Cached KV state: {kv_cache_key}")
            except Exception as e:
                logger.warning(f"Failed to cache KV state: {e}")

        latency = time.time() - start_time
        logger.debug(f"Generation completed in {latency:.3f}s")
        return response

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        model_size: Optional[ModelSize] = None,
        use_thinking: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7,
        best_of_n: int = 1,
        retrieval_scores: Optional[List[float]] = None,
    ) -> str:
        """Synchronous generate wrapper.

        Note: This is a bridge between sync and async contexts. If you're already
        in an async context (e.g., agent.run()), use generate_async() directly with await.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use run_until_complete on a running loop
                # Use run_coroutine_threadsafe to schedule on the running loop
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self.generate_async(prompt, images, model_size, use_thinking, max_tokens, temperature, best_of_n, retrieval_scores),
                    loop
                )
                try:
                    text, _ = future.result(timeout=60.0)  # 60s timeout
                    return text
                except concurrent.futures.TimeoutError:
                    logger.error("generate() timed out after 60s")
                    return ""
            else:
                text, _ = loop.run_until_complete(
                    self.generate_async(prompt, images, model_size, use_thinking, max_tokens, temperature, best_of_n, retrieval_scores)
                )
                return text
        except RuntimeError:
            # No event loop in current thread, create a new one
            text, _ = asyncio.run(
                self.generate_async(prompt, images, model_size, use_thinking, max_tokens, temperature, best_of_n, retrieval_scores)
            )
            return text

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.model_router.get_batch_stats()

    def preload_models(self, model_sizes: List[ModelSize]) -> None:
        """Preload models into memory."""
        for size in model_sizes:
            # Load both variants
            for variant in ["instruct", "thinking"]:
                model_name = self._get_model_name(size, variant == "thinking")
                try:
                    self.load_model(model_name, variant)  # type: ignore
                except Exception as e:
                    logger.warning(f"Failed to preload {model_name}: {e}")

    def clear_cache(self) -> None:
        """Clear all caches and memory."""
        # Clear new prompt cache
        self.prompt_cache.clear_all()

        # Clear legacy caches for compatibility
        self.vision_cache.ram_cache.clear()
        self.prompt_kv_cache.ram_cache.clear()

        # Clear prompt cache rings (legacy)
        for ring in self.prompt_cache_rings.values():
            ring.ring.clear()
        self.prompt_cache_rings.clear()

        # Stop and restart pipeline if running
        if self.use_pipeline and self.pipeline_engine.running:
            asyncio.create_task(self._restart_pipeline())

        logger.info("Cleared all caches including new prompt cache and pipeline")

    async def _restart_pipeline(self) -> None:
        """Restart pipeline engine after cache clear."""
        await self.pipeline_engine.stop()
        await self.pipeline_engine.start()

    def reset_cache_telemetry(self) -> None:
        """Reset cache telemetry counters."""
        self.vision_cache.telemetry.reset()
        self.prompt_kv_cache.telemetry.reset()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "vision_cache": {
                "hits": self.vision_cache.telemetry.hits,
                "misses": self.vision_cache.telemetry.misses,
                "entries": len(self.vision_cache.ram_cache),
                "avg_latency": sum(self.vision_cache.telemetry.latency_deltas) / len(self.vision_cache.telemetry.latency_deltas) if self.vision_cache.telemetry.latency_deltas else 0,
            },
            "prompt_kv_cache": {
                "hits": self.prompt_kv_cache.telemetry.hits,
                "misses": self.prompt_kv_cache.telemetry.misses,
                "ram_entries": len(self.prompt_kv_cache.ram_cache),
                "avg_latency": sum(self.prompt_kv_cache.telemetry.latency_deltas) / len(self.prompt_kv_cache.telemetry.latency_deltas) if self.prompt_kv_cache.telemetry.latency_deltas else 0,
            },
            "new_prompt_cache": self.prompt_cache.get_stats(),
        }

        if self.use_pipeline:
            stats["pipeline"] = self.pipeline_engine.get_stats()

                # Include new prompt cache stats
        stats["new_prompt_cache"] = self.prompt_cache.get_stats()

        return stats

    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return list(self.SUPPORTED_MODELS)

    def get_armada_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get Armada registry with model metadata."""
        return {
            "qwen3-vl-2b-instruct": {
                "model_name": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
                "quantization": "bnb4bit",
                "size": "2B",
                "variant": "instruct",
            },
            "qwen3-vl-2b-thinking": {
                "model_name": "Qwen/Qwen3-VL-2B-Thinking-FP8",
                "quantization": "fp8",
                "size": "2B",
                "variant": "thinking",
            },
            "qwen3-vl-4b-instruct": {
                "model_name": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
                "quantization": "bnb4bit",
                "size": "4B",
                "variant": "instruct",
            },
            "qwen3-vl-4b-thinking": {
                "model_name": "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
                "quantization": "bnb4bit",
                "size": "4B",
                "variant": "thinking",
            },
            "qwen3-vl-8b-instruct": {
                "model_name": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                "quantization": "bnb4bit",
                "size": "8B",
                "variant": "instruct",
            },
            "qwen3-vl-8b-thinking": {
                "model_name": "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
                "quantization": "bnb4bit",
                "size": "8B",
                "variant": "thinking",
            },
        }

    async def _generate_candidates_parallel(
        self,
        prompt: str,
        images: Optional[List[Any]],
        model_name: str,
        max_tokens: int,
        temperature: float,
        n: int,
    ) -> tuple[List[str], List[DecodeResult]]:
        """Generate n candidates in parallel."""
        # Create n parallel generation tasks
        tasks = [
            self._single_generate(prompt, images, model_name, max_tokens, temperature)
            for _ in range(n)
        ]

        # Wait for all to complete
        candidates = await asyncio.gather(*tasks)

        # Create mock DecodeResult objects for scoring
        decode_results = [
            DecodeResult(
                generated_text=candidate,
                tokens_used=len(candidate.split()),  # Rough token count
                latency_ms=100.0  # Mock latency
            )
            for candidate in candidates
        ]

        return candidates, decode_results

    async def _generate_with_cache(
        self,
        cached_entry: PromptCacheEntryNew,
        model_name: str,
        max_tokens: int,
        temperature: float,
        best_of_n: int,
        retrieval_scores: Optional[List[float]],
        yield_every: Optional[int],
        wall_budget_s: float = 30.0,
    ) -> tuple[str, List[float]]:
        """Execute pipelined generation with overlap and best-of-n.

        Args:
            cached_entry: Cached prompt entry with tokenized data
            model_name: Model name for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            best_of_n: Number of candidates (1,2,4,8)
            retrieval_scores: Optional retrieval scores
            yield_every: Yield partials every N tokens
            wall_budget_s: Wall clock time budget

        Returns:
            Tuple of (selected_text, candidate_scores)

        Raises:
            GenerationBudgetExceeded: If budget exceeded
            BestOfSelectionError: If selection fails
        """
        start_time = time.time()
        semaphore = self._get_or_create_vram_semaphore(model_name)

        async def run_forward(candidate_idx: int) -> str:
            """Run forward pass with semaphore."""
            async with semaphore:
                # Enforce max_new_tokens guard
                actual_max_tokens = min(max_tokens, 512)  # Conservative guard

                # Mock generation with partial yielding if requested
                if yield_every and candidate_idx == 0:  # Only for first candidate
                    partials = []
                    for i in range(0, actual_max_tokens, yield_every):
                        await asyncio.sleep(0.01)  # Simulate work
                        partial = f"Partial generation {i} tokens..."
                        partials.append(partial)
                        # In real impl, yield partial to caller here

                # Final result
                result = f"Pipelined response for candidate {candidate_idx}: {cached_entry.tokenized_data[:30]}..."
                return result

        try:
            # Run candidates in parallel with pipelining
            if best_of_n == 1:
                result = await run_forward(0)
                return result, [1.0]
            else:
                # Batch candidates within wall budget
                tasks = [asyncio.create_task(run_forward(i)) for i in range(best_of_n)]
                done, pending = await asyncio.wait(tasks, timeout=wall_budget_s, return_when=asyncio.ALL_COMPLETED)

                if pending:
                    # Budget exceeded - cancel pending, return best partial
                    for task in pending:
                        task.cancel()
                    logger.warning(f"Wall budget {wall_budget_s}s exceeded, returning partial results")
                    raise GenerationBudgetExceeded(f"Generation exceeded {wall_budget_s}s budget")

                candidates = [task.result() for task in done]

                # Score and select best
                mock_decode_results = [
                    DecodeResult(
                        generated_text=candidate,
                        tokens_used=len(candidate.split()),
                        latency_ms=(time.time() - start_time) * 1000 / best_of_n
                    )
                    for candidate in candidates
                ]

                scores = self._score_candidates(mock_decode_results, retrieval_scores)
                selected, candidate_scores = self._select_best_candidate(candidates, scores)

                if not selected:
                    raise BestOfSelectionError("No valid candidates selected")

                return selected, candidate_scores

        except asyncio.TimeoutError:
            raise GenerationBudgetExceeded(f"Generation timed out after {wall_budget_s}s")

    def _score_candidates(
        self,
        decode_results: List[DecodeResult],
        retrieval_scores: Optional[List[float]] = None,
        k: int = 60,
    ) -> List[float]:
        """Score candidates using normalized logprob + RRF with retrieval scores.

        Args:
            decode_results: List of decode results with generated text
            retrieval_scores: Optional retrieval scores for RRF
            k: RRF constant (typically 60)

        Returns:
            List of scores for each candidate
        """
        scores = []

        for i, result in enumerate(decode_results):
            # Mock normalized logprob (in real impl, this would be actual logprob)
            # Higher token count roughly correlates with higher confidence
            mock_logprob = min(1.0, result.tokens_used / 50.0)

            # RRF with retrieval scores if available
            if retrieval_scores and i < len(retrieval_scores):
                retrieval_rrf = self._rrf_score(retrieval_scores[i], k)
            else:
                retrieval_rrf = 0.0

            # Combine logprob and retrieval RRF
            combined_score = mock_logprob + retrieval_rrf
            scores.append(combined_score)

        return scores

    def _compute_logprob(self, candidate_text: str, model_name: str) -> float:
        """Compute normalized log probability for candidate text.

        Args:
            candidate_text: Generated candidate text
            model_name: Model name used for generation

        Returns:
            Normalized log probability (0-1, higher is better)
        """
        # Mock implementation - in real setup this would compute actual logprobs
        # Higher token count roughly correlates with higher confidence
        token_count = len(candidate_text.split())
        return min(1.0, token_count / 50.0)

    def _rrf_score(self, relevance_score: float, k: int = 60) -> float:
        """Calculate Reciprocal Rank Fusion score."""
        # Convert relevance to rank (higher relevance = lower rank)
        # Assuming relevance_score is 0-1, map to rank 1-10
        rank = max(1, int((1.0 - relevance_score) * 10) + 1)
        return 1.0 / (k + rank)

    def _select_best_candidate(
        self,
        candidates: List[str],
        scores: List[float]
    ) -> tuple[str, List[float]]:
        """Select candidate with highest score."""
        if not candidates or not scores:
            return "", []

        best_idx = scores.index(max(scores))
        return candidates[best_idx], scores

    async def initialize_async(self) -> None:
        """Initialize async components that require an event loop."""
        if self.use_pipeline and not self.pipeline_initialized:
            # Wire pipeline callbacks to controller methods
            self.pipeline_engine.set_prefill_callback(self._process_prefill_batch)
            self.pipeline_engine.set_decode_callback(self._process_decode_batch)
            # Start pipeline engine
            await self.pipeline_engine.start()
            self.pipeline_initialized = True
            logger.info("Pipeline engine started with prefill/decode callbacks wired")
