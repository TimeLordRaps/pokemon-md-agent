"""Memory management for agent context allocation and scratchpad with smart Qwen3-VL model caching."""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import logging
import time
import os
from collections import OrderedDict

try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoProcessor = None
    AutoModelForVision2Seq = None

logger = logging.getLogger(__name__)


class MemoryAllocation:
    """Configuration for memory allocation across temporal ranges."""
    
    def __init__(
        self,
        last_5_minutes: float = 0.75,
        last_30_minutes: float = 0.15,
        active_missions: float = 0.10,
    ):
        """Initialize memory allocation.
        
        Args:
            last_5_minutes: Percentage of context for last 5 minutes (0.0-1.0)
            last_30_minutes: Percentage for last 30 minutes (0.0-1.0)
            active_missions: Percentage for current mission context (0.0-1.0)
        """
        total = last_5_minutes + last_30_minutes + active_missions
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Memory allocation must sum to 1.0, got {total}")
        
        self.last_5_minutes = last_5_minutes
        self.last_30_minutes = last_30_minutes
        self.active_missions = active_missions


@dataclass
class ModelPair:
    """Represents a pair of instruct/thinking models of the same size."""
    size: str  # "2B", "4B", or "8B"
    instruct_name: str
    thinking_name: str


@dataclass
class ScratchpadEntry:
    """Entry in the agent's scratchpad."""
    content: str
    timestamp: float
    priority: int = 0  # Higher priority entries are kept longer


class Scratchpad:
    """Persistent scratchpad for agent to leave notes across interactions."""
    
    def __init__(self, max_entries: int = 100):
        """Initialize scratchpad.
        
        Args:
            max_entries: Maximum number of entries to store
        """
        self.max_entries = max_entries
        self.entries: list[ScratchpadEntry] = []
        self._current_time = 0.0
        
    def write(self, content: str, priority: int = 0) -> None:
        """Write a new entry to the scratchpad.
        
        Args:
            content: Content to write
            priority: Priority level (0=normal, 1=important, 2=critical)
        """
        entry = ScratchpadEntry(
            content=content,
            timestamp=self._current_time,
            priority=priority
        )
        self.entries.append(entry)
        
        # Trim if over capacity
        if len(self.entries) > self.max_entries:
            # Keep higher priority entries
            self.entries.sort(key=lambda e: (e.priority, e.timestamp), reverse=True)
            self.entries = self.entries[:self.max_entries]
        
        # Truncate content for logging
        content_preview = content[:50] + "..." if len(content) > 50 else content
        logger.debug("Added scratchpad entry: %s", content_preview)
        
    def read(self, limit: Optional[int] = None) -> list[str]:
        """Read all scratchpad entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of content strings, most recent first
        """
        entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
        
        if limit is not None:
            entries = entries[:limit]
        
        return [entry.content for entry in entries]
    
    def read_with_metadata(self, limit: Optional[int] = None) -> list[ScratchpadEntry]:
        """Read all scratchpad entries with metadata.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of ScratchpadEntry objects, most recent first
        """
        entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
        
        if limit is not None:
            entries = entries[:limit]
        
        return entries
    
    def clear(self) -> None:
        """Clear all scratchpad entries."""
        self.entries.clear()
        logger.debug("Cleared all scratchpad entries")
    
    def update_time(self, current_time: float) -> None:
        """Update the current time for timestamp calculations.

        Args:
            current_time: Current time in seconds
        """
        self._current_time = current_time


class ModelCache:
    """Smart cache for Qwen3-VL models with VRAM-aware LRU eviction and tokenizer reuse.

    Features:
    - Loads models with local_files_only=True for offline operation
    - Shares tokenizers/processors across models of same architecture
    - LRU eviction based on VRAM usage when memory is tight
    - Prefers keeping instruct/thinking pairs resident when possible
    """

    def __init__(self, max_vram_gb: float = 12.0):
        """Initialize model cache.

        Args:
            max_vram_gb: Maximum VRAM to use before eviction (default 12GB for high-end GPUs)
        """
        self.max_vram_gb = max_vram_gb
        self._cached_models: Dict[str, Dict[str, Any]] = {}
        self._vram_usage_gb: float = 0.0
        self._shared_tokenizers: Dict[str, Any] = {}
        self._shared_processors: Dict[str, Any] = {}
        self._pairs_preference: Dict[str, List[str]] = {}  # size -> [instruct, thinking] model keys

        # Model name mappings for the six specified Qwen3-VL models
        self._model_pairs = {
            "2B": ModelPair("2B", "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit", "Qwen/Qwen3-VL-2B-Thinking-FP8"),
            "4B": ModelPair("4B", "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit", "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit"),
            "8B": ModelPair("8B", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"),
        }

        logger.info(f"Initialized ModelCache with max_vram_gb={max_vram_gb}")

    def probe_vram_free_gb(self) -> float:
        """Probe available VRAM in GB.

        Returns:
            Free VRAM in GB
        """
        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, assuming 8GB free VRAM")
            return 8.0

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            logger.debug(".2f")
            return free_gb
        except Exception as e:
            logger.warning(f"Failed to probe VRAM: {e}, assuming 8GB free")
            return 8.0

    def get_shared_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get cached tokenizer for model, loading if needed.

        Args:
            model_name: HuggingFace model name

        Returns:
            Cached tokenizer or None if loading failed
        """
        if model_name in self._shared_tokenizers:
            return self._shared_tokenizers[model_name]

        if AutoTokenizer is None:
            logger.error("AutoTokenizer not available")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self._shared_tokenizers[model_name] = tokenizer
            logger.info(f"Cached shared tokenizer for {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            return None

    def get_shared_processor(self, model_name: str) -> Optional[Any]:
        """Get cached processor for model, loading if needed.

        Args:
            model_name: HuggingFace model name

        Returns:
            Cached processor or None if loading failed
        """
        if model_name in self._shared_processors:
            return self._shared_processors[model_name]

        if AutoProcessor is None:
            logger.error("AutoProcessor not available")
            return None

        try:
            from .utils import get_hf_cache_dir
            cache_dir = get_hf_cache_dir()
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
            self._shared_processors[model_name] = processor
            logger.info(f"Cached shared processor for {model_name}")
            return processor
        except Exception as e:
            logger.error(f"Failed to load processor for {model_name}: {e}")
            return None

    def load_model(self, model_name: str, local_files_only: bool = True) -> Optional[Any]:
        """Load model with caching and VRAM management.

        Args:
            model_name: HuggingFace model name
            local_files_only: Use local files only (required for offline operation)

        Returns:
            Loaded model or None if loading failed
        """
        if model_name in self._cached_models:
            # Update LRU timestamp
            self._cached_models[model_name]["last_used"] = time.time()
            logger.debug(f"Retrieved cached model: {model_name}")
            return self._cached_models[model_name]["model"]

        if AutoModelForVision2Seq is None:
            logger.error("AutoModelForVision2Seq not available")
            return None

        # Check if we need to evict models
        try:
            # Estimate model size (rough heuristic)
            if "2B" in model_name:
                estimated_gb = 2.0
            elif "4B" in model_name:
                estimated_gb = 4.0
            elif "8B" in model_name:
                estimated_gb = 8.0
            else:
                estimated_gb = 4.0  # default

            evicted_keys = self._evict_if_needed(estimated_gb)
            if evicted_keys:
                logger.info(f"Evicted models to make room: {evicted_keys}")

            # Load model
            from .utils import get_hf_cache_dir
            cache_dir = get_hf_cache_dir()
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
                cache_dir=cache_dir
            )

            # Cache the model
            self._cached_models[model_name] = {
                "model": model,
                "last_used": time.time(),
                "vram_gb": estimated_gb
            }
            self._vram_usage_gb += estimated_gb

            logger.info(f"Loaded and cached model: {model_name} ({estimated_gb}GB)")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def _evict_if_needed(self, required_gb: float) -> List[str]:
        """Evict models if needed to make room for new model.

        Args:
            required_gb: VRAM needed for new model

        Returns:
            List of evicted model keys
        """
        free_gb = self.probe_vram_free_gb()
        available_gb = free_gb + (self.max_vram_gb - self._vram_usage_gb)

        if available_gb >= required_gb:
            return []  # No eviction needed

        # Need to evict - sort by LRU (oldest first)
        sorted_models = sorted(
            self._cached_models.items(),
            key=lambda x: x[1]["last_used"]
        )

        evicted = []
        freed_gb = 0.0

        for model_key, model_info in sorted_models:
            if freed_gb >= required_gb:
                break

            # Prefer not to evict pairs if possible
            size = model_key.split("-")[-2] if "B-" in model_key else None
            if size and size in self._pairs_preference:
                pair_keys = self._pairs_preference[size]
                if len([k for k in pair_keys if k in self._cached_models]) == 2:
                    # Both models of pair are loaded, try to evict single models first
                    continue

            evicted.append(model_key)
            freed_gb += model_info["vram_gb"]
            del self._cached_models[model_key]
            self._vram_usage_gb -= model_info["vram_gb"]

        return evicted

    def get_model_pair(self, size: str) -> Optional[ModelPair]:
        """Get model pair for given size.

        Args:
            size: Model size ("2B", "4B", or "8B")

        Returns:
            ModelPair or None if size not supported
        """
        return self._model_pairs.get(size)

    def preload_pair_if_space(self, size: str) -> bool:
        """Preload both instruct and thinking models of same size if VRAM permits.

        Args:
            size: Model size ("2B", "4B", or "8B")

        Returns:
            True if both models were loaded successfully
        """
        pair = self.get_model_pair(size)
        if not pair:
            return False

        # Check if we have space for both
        free_gb = self.probe_vram_free_gb()
        if free_gb < 8.0:  # Conservative: need at least 8GB free for pair
            logger.debug(f"Insufficient VRAM for {size} pair preload")
            return False

        # Load both models
        instruct_loaded = self.load_model(pair.instruct_name) is not None
        thinking_loaded = self.load_model(pair.thinking_name) is not None

        if instruct_loaded and thinking_loaded:
            self._pairs_preference[size] = [pair.instruct_name, pair.thinking_name]
            logger.info(f"Preloaded {size} model pair")
            return True

        return False


class MemoryManager:
    """Manages agent memory allocation across temporal ranges and smart Qwen3-VL model caching.

    Integrates context allocation with ModelCache for efficient model loading and VRAM management.
    """

    def __init__(
        self,
        total_context_budget: int = 256_000,
        allocation: Optional[MemoryAllocation] = None,
        model_cache_max_vram_gb: float = 12.0,
    ):
        """Initialize memory manager with integrated model caching.

        Args:
            total_context_budget: Total tokens available for context
            allocation: Memory allocation configuration
            model_cache_max_vram_gb: Maximum VRAM for model cache
        """
        self.total_context_budget = total_context_budget
        self.allocation = allocation or MemoryAllocation()
        self.scratchpad = Scratchpad()
        self.model_cache = ModelCache(max_vram_gb=model_cache_max_vram_gb)
        
    def allocate(self, allocation: Optional[MemoryAllocation] = None) -> Dict[str, int]:
        """Calculate token allocation across temporal ranges.
        
        Args:
            allocation: Optional override allocation configuration
            
        Returns:
            Dictionary mapping memory range to token count
        """
        alloc = allocation or self.allocation
        
        return {
            "last_5_minutes": int(self.total_context_budget * alloc.last_5_minutes),
            "last_30_minutes": int(self.total_context_budget * alloc.last_30_minutes),
            "active_missions": int(self.total_context_budget * alloc.active_missions),
        }
    
    def get_memory_budget(self, memory_type: str) -> int:
        """Get token budget for a specific memory type.
        
        Args:
            memory_type: Type of memory ("last_5_minutes", "last_30_minutes", "active_missions")
            
        Returns:
            Token budget for the memory type
        """
        budgets = self.allocate()
        return budgets.get(memory_type, 0)
    
    def update_allocation(
        self,
        last_5_minutes: Optional[float] = None,
        last_30_minutes: Optional[float] = None,
        active_missions: Optional[float] = None,
    ) -> None:
        """Update memory allocation configuration.
        
        Args:
            last_5_minutes: New percentage for last 5 minutes
            last_30_minutes: New percentage for last 30 minutes
            active_missions: New percentage for active missions
            
        Raises:
            ValueError: If percentages don't sum to 1.0
        """
        new_allocation = MemoryAllocation(
            last_5_minutes=last_5_minutes or self.allocation.last_5_minutes,
            last_30_minutes=last_30_minutes or self.allocation.last_30_minutes,
            active_missions=active_missions or self.allocation.active_missions,
        )
        self.allocation = new_allocation
        
        logger.info(
            "Updated memory allocation: 5min=%.1f%%, 30min=%.1f%%, missions=%.1f%%",
            new_allocation.last_5_minutes * 100,
            new_allocation.last_30_minutes * 100,
            new_allocation.active_missions * 100
        )
