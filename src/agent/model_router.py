"""Model routing logic for selecting between 2B, 4B, and 8B Qwen3-VL models."""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import os
import asyncio
import mmap
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from .inference_queue import InferenceQueue
from .timebudgets import TOKENIZE_BUDGET_S, FORWARD_BUDGET_S, DECODE_BUDGET_S

logger = logging.getLogger(__name__)


class DeadlineExceededError(Exception):
    """Raised when a request exceeds its deadline budget."""
    pass


class ModelSize(Enum):
    """Available model sizes."""
    SIZE_2B = "2B"
    SIZE_4B = "4B"
    SIZE_8B = "8B"


class TriggerType(Enum):
    """Types of routing triggers."""
    PRIMARY = "primary"  # Main confidence-based routing
    SECONDARY = "secondary"  # Additional triggers (performance, stuckness, etc.)
    HYSTERESIS = "hysteresis"  # Prevent rapid switching


# Model name mappings for Unsloth quantized models
# Note: 2B Thinking uses FP8 from Qwen (no unsloth-bnb-4bit version available)
MODEL_NAMES = {
    ModelSize.SIZE_2B: {
        "instruct": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "thinking": "Qwen/Qwen3-VL-2B-Thinking-FP8",
    },
    ModelSize.SIZE_4B: {
        "instruct": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "thinking": "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
    },
    ModelSize.SIZE_8B: {
        "instruct": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        "thinking": "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    },
}


MICRO_BATCH_TOKEN_LIMITS = {
    ModelSize.SIZE_2B: 8192,
    ModelSize.SIZE_4B: 4096,
    ModelSize.SIZE_8B: 2048,
}


@dataclass
class RoutingDecision:
    """Result of a model routing decision."""
    selected_model: ModelSize
    use_thinking: bool
    confidence_threshold_met: bool
    stuck_counter: int
    reasoning: str
    trigger_type: TriggerType = TriggerType.PRIMARY
    hysteresis_active: bool = False
    secondary_triggers: List[str] = field(default_factory=list)


@dataclass
class HysteresisState:
    """State for hysteresis logic to prevent rapid model switching."""
    current_model: ModelSize
    last_switch_time: float
    switch_cooldown_seconds: float = 10.0  # Minimum time between switches
    confidence_margin: float = 0.1  # Margin to prevent oscillation


@dataclass
class PrefillRequest:
    """Request for PREFILL stage processing."""
    prompt: str
    images: Optional[List[Any]] = None
    model_size: ModelSize = ModelSize.SIZE_4B
    use_thinking: bool = False
    max_tokens: int = 256
    group_key: Optional[str] = None
    best_of_n: int = 1
    deadline_s: Optional[float] = None


@dataclass
class PrefillResult:
    """Result of PREFILL stage processing."""
    tokenized_input: Any
    vision_encoded: Optional[Any] = None
    kv_state: Optional[Any] = None
    prompt_sha: str = ""
    image_sha: Optional[str] = None
    cache_hit: bool = False


@dataclass
class DecodeRequest:
    """Request for DECODE stage processing."""
    prefill_result: PrefillResult
    temperature: float = 0.7
    group_key: Optional[str] = None
    deadline_s: Optional[float] = None


@dataclass
class DecodeResult:
    """Result of DECODE stage processing."""
    generated_text: str
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class GroupKey:
    """Micro-batch grouping key."""
    model_id: str
    mode: str  # "instruct" or "thinking"
    max_seq: int
    vision_shape: Optional[tuple] = None

    def __hash__(self):
        return hash((self.model_id, self.mode, self.max_seq, self.vision_shape))

    def __eq__(self, other):
        return (self.model_id, self.mode, self.max_seq, self.vision_shape) == \
               (other.model_id, other.mode, other.max_seq, other.vision_shape)


class ModelRouter:
    """Routes inference requests to appropriate models with batching and caching."""

    def __init__(self, hf_home: Optional[str] = None):
        self.hf_home = hf_home or os.getenv('HF_HOME', '~/.cache/huggingface')
        self.two_stage_pipeline = TwoStagePipeline(self)

        # Wire in caches (will be populated later)
        self.prompt_cache = None
        self.vision_cache = None
        self.prompt_kv_cache = None

        # Pipeline integration
        self.pipeline_engine = None  # Will be set by qwen_controller
        self.use_pipeline = True  # Feature flag

        # Batch size configuration (can be overridden for benchmarking)
        self.batch_size_2b = 8
        self.batch_size_4b = 4
        self.batch_size_8b = 2

    def get_model_name(self, model_size: ModelSize, use_thinking: bool = False) -> str:
        """Get model name for given size and thinking mode."""
        return MODEL_NAMES[model_size]["thinking" if use_thinking else "instruct"]

    def infer_async(self, query: str, model_size: ModelSize) -> asyncio.Future[str]:
        """Async inference with pipeline integration."""
        if self.use_pipeline and self.pipeline_engine:
            # Use pipeline for better batching
            request = PrefillRequest(
                prompt=query,
                model_size=model_size,
                max_tokens=256
            )
            prefill_future = self.two_stage_pipeline.submit_prefill(request)

            async def process_result():
                prefill_result = await prefill_future
                decode_request = DecodeRequest(prefill_result=prefill_result)
                decode_future = self.two_stage_pipeline.submit_decode(decode_request)
                decode_result = await decode_future
                return decode_result.generated_text

            future = asyncio.Future()
            asyncio.create_task(self._resolve_future(future, process_result()))
            return future
        else:
            # Fallback to sync
            future = asyncio.Future()
            future.set_result(f"result for {query}")
            return future

    async def _resolve_future(self, future: asyncio.Future, coro):
        """Helper to resolve future from coroutine."""
        try:
            result = await coro
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    def select_model(self, remaining_budget_s: float, preferred_model: Optional[ModelSize] = None,
                     use_thinking: bool = False) -> ModelSize:
        """Select appropriate model based on remaining time budget.

        Args:
            remaining_budget_s: Time remaining in seconds for the request.
            preferred_model: Preferred model size, or None for auto-selection.
            use_thinking: Whether thinking mode is required.

        Returns:
            Selected ModelSize that fits within budget.

        Raises:
            DeadlineExceededError: If no model can fit within remaining budget.
        """
        if preferred_model:
            # Check if preferred model fits budget
            if self._estimate_inference_time(preferred_model, use_thinking) <= remaining_budget_s:
                return preferred_model
            else:
                logger.warning(f"Preferred model {preferred_model} exceeds budget {remaining_budget_s}s, falling back")

        # Auto-select based on budget (prefer larger models when possible)
        for model in [ModelSize.SIZE_8B, ModelSize.SIZE_4B, ModelSize.SIZE_2B]:
            if self._estimate_inference_time(model, use_thinking) <= remaining_budget_s:
                logger.info(f"Selected model {model} for budget {remaining_budget_s}s")
                return model

        raise DeadlineExceededError(f"No model fits within remaining budget {remaining_budget_s}s")

    def _estimate_inference_time(self, model_size: ModelSize, use_thinking: bool) -> float:
        """Estimate total inference time for a model including all stages."""
        # Rough estimates based on model size (tokenize + forward + decode)
        base_times = {
            ModelSize.SIZE_2B: TOKENIZE_BUDGET_S + FORWARD_BUDGET_S * 0.3 + DECODE_BUDGET_S * 0.3,
            ModelSize.SIZE_4B: TOKENIZE_BUDGET_S + FORWARD_BUDGET_S * 0.6 + DECODE_BUDGET_S * 0.6,
            ModelSize.SIZE_8B: TOKENIZE_BUDGET_S + FORWARD_BUDGET_S + DECODE_BUDGET_S,
        }
        return base_times[model_size]

    def auto_batch_size(self, model_size: ModelSize, gpu_utilization: float = 0.5,
                       vram_used_gb: float = 16.0) -> int:
        """Calculate optimal batch size based on model and resources."""
        base_sizes = {
            ModelSize.SIZE_2B: 2,
            ModelSize.SIZE_4B: 2,
            ModelSize.SIZE_8B: 1,
        }
        base = base_sizes[model_size]

        # Scale down with GPU utilization
        scale = max(0.1, 1.0 - gpu_utilization)
        batch_size = int(base * scale)

        # Ensure minimum 1
        return max(1, batch_size)


class TwoStagePipeline:
    """Two-stage pipelining with PREFILL and DECODE phases for efficient inference."""

    def __init__(self, model_router: 'ModelRouter', flush_tick_ms: int = 50):
        self.model_router = model_router
        self.flush_tick_ms = flush_tick_ms
        self.prefill_queues: Dict[GroupKey, List[PrefillRequest]] = {}
        self.decode_queues: Dict[GroupKey, List[DecodeRequest]] = {}
        self.last_flush_time = time.time()
        self._flush_task: Optional[asyncio.Task] = None

    def submit_prefill(self, request: PrefillRequest, deadline_s: Optional[float] = None) -> asyncio.Future[PrefillResult]:
        """Submit request to PREFILL stage with optional deadline.

        Args:
            request: PrefillRequest containing prompt and model parameters.
            deadline_s: Optional deadline in seconds from now.

        Returns:
            Future for PrefillResult.
        """
        if deadline_s is not None:
            # Set deadline on request for batch processing
            request.deadline_s = deadline_s

        group_key = self._make_group_key(request)
        if group_key not in self.prefill_queues:
            self.prefill_queues[group_key] = []
        self.prefill_queues[group_key].append(request)

        future = asyncio.Future()
        # Store future for resolution
        request._future = future  # type: ignore

        self._check_flush()
        return future

    def submit_decode(self, request: DecodeRequest, deadline_s: Optional[float] = None) -> asyncio.Future[DecodeResult]:
        """Submit request to DECODE stage with optional deadline.

        Args:
            request: DecodeRequest containing prefill result and decode parameters.
            deadline_s: Optional deadline in seconds from now.

        Returns:
            Future for DecodeResult.
        """
        if deadline_s is not None:
            # Set deadline on request for batch processing
            request.deadline_s = deadline_s

        group_key = self._make_group_key_from_decode(request)
        if group_key not in self.decode_queues:
            self.decode_queues[group_key] = []
        self.decode_queues[group_key].append(request)

        future = asyncio.Future()
        request._future = future  # type: ignore

        self._check_flush()
        return future

    def _make_group_key(self, request: PrefillRequest) -> GroupKey:
        """Create group key for micro-batching."""
        if request.group_key:
            # Parse existing group key
            parts = request.group_key.split('|')
            model_id = parts[0]
            mode = "thinking" if request.use_thinking else "instruct"
            max_seq = request.max_tokens
            vision_shape = tuple(parts[1:]) if len(parts) > 1 else None
        else:
            model_name = self.model_router.get_model_name(request.model_size, request.use_thinking)
            mode = "thinking" if request.use_thinking else "instruct"
            max_seq = request.max_tokens
            vision_shape = None
            if request.images:
                # Simple shape representation
                vision_shape = (len(request.images), "image")

        return GroupKey(
            model_id=self.model_router.get_model_name(request.model_size, request.use_thinking),
            mode=mode,
            max_seq=max_seq,
            vision_shape=vision_shape
        )

    def submit_prefill_best_of_n(self, request: PrefillRequest) -> asyncio.Future[tuple[PrefillResult, List[PrefillResult]]]:
        """Submit request for best-of-n PREFILL with parallel candidates."""
        if request.best_of_n <= 1:
            # Fallback to single generation
            future = self.submit_prefill(request)
            return asyncio.Future()

        # Create n parallel requests
        futures = []
        for _ in range(request.best_of_n):
            future = self.submit_prefill(request)
            futures.append(future)

        # Combine results
        async def combine_results():
            results = await asyncio.gather(*futures)
            # Return first result and all results for scoring
            return results[0], results

        combined_future = asyncio.Future()
        asyncio.create_task(self._resolve_combined_future(combined_future, combine_results()))
        return combined_future

    async def _resolve_combined_future(self, future: asyncio.Future, coro):
        """Helper to resolve combined future."""
        try:
            result = await coro
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    def _make_group_key_from_decode(self, request: DecodeRequest) -> GroupKey:
        """Create group key from decode request."""
        # Extract from prefill result metadata
        model_id = getattr(request.prefill_result, 'model_id', 'default')
        mode = getattr(request.prefill_result, 'mode', 'instruct')
        max_seq = getattr(request.prefill_result, 'max_tokens', 256)
        vision_shape = getattr(request.prefill_result, 'vision_shape', None)

        return GroupKey(
            model_id=model_id,
            mode=mode,
            max_seq=max_seq,
            vision_shape=vision_shape
        )

    def _check_flush(self):
        """Check if flush tick should trigger batch processing."""
        current_time = time.time()
        if (current_time - self.last_flush_time) * 1000 >= self.flush_tick_ms:
            self._flush_all_batches()

    def _flush_all_batches(self):
        """Flush all accumulated batches."""
        for group_key, requests in list(self.prefill_queues.items()):
            if requests:
                self._process_prefill_batch(group_key, requests)
                self.prefill_queues[group_key] = []

        for group_key, requests in list(self.decode_queues.items()):
            if requests:
                self._process_decode_batch(group_key, requests)
                self.decode_queues[group_key] = []

        self.last_flush_time = time.time()

    def _process_prefill_batch(self, group_key: GroupKey, requests: List[PrefillRequest]):
        """Process a batch of PREFILL requests with deadline checking."""
        try:
            # Check deadlines and truncate if necessary
            valid_requests = []
            for request in requests:
                if hasattr(request, 'deadline_s') and request.deadline_s is not None:
                    current_time = time.time()
                    remaining_time = request.deadline_s - current_time
                    if remaining_time <= 0:
                        logger.warning(f"Prefill request deadline exceeded: {remaining_time}s remaining")
                        if hasattr(request, '_future'):
                            request._future.set_exception(DeadlineExceededError("Prefill deadline exceeded"))
                        continue
                    # Select appropriate model if deadline is tight
                    selected_model = self.model_router.select_model(remaining_time, request.model_size, request.use_thinking)
                    if selected_model != request.model_size:
                        logger.info(f"Selected smaller model {selected_model} for deadline: {remaining_time}s")
                        request.model_size = selected_model
                valid_requests.append(request)

            if not valid_requests:
                return

            # Re-check deadlines before batch processing starts
            batch_start_time = time.time()
            final_requests = []
            for request in valid_requests:
                if hasattr(request, 'deadline_s') and request.deadline_s is not None:
                    remaining_time = request.deadline_s - batch_start_time
                    # Estimate batch processing time (tokenize + vision for this request)
                    estimated_time = TOKENIZE_BUDGET_S
                    if request.images:
                        estimated_time += len(request.images) * 0.1  # Rough vision encoding time
                    if remaining_time < estimated_time:
                        logger.warning(f"Batch processing would exceed deadline: {remaining_time}s remaining, need {estimated_time}s")
                        if hasattr(request, '_future'):
                            request._future.set_exception(DeadlineExceededError("Batch deadline exceeded"))
                        continue
                final_requests.append(request)

            if not final_requests:
                return

            # Adjust batch size based on strictest deadline
            batch_size_limit = len(final_requests)
            if any(hasattr(r, 'deadline_s') and r.deadline_s is not None for r in final_requests):
                strictest_deadline = min(r.deadline_s for r in final_requests if hasattr(r, 'deadline_s') and r.deadline_s is not None)
                remaining_batch_time = strictest_deadline - batch_start_time
                # Estimate time per request in batch and limit batch size
                time_per_request = TOKENIZE_BUDGET_S + 0.05  # Base estimate
                max_batch_size = max(1, int(remaining_batch_time / time_per_request))
                batch_size_limit = min(batch_size_limit, max_batch_size)

            # Apply batch size limit
            if len(final_requests) > batch_size_limit:
                logger.info(f"Limiting batch size from {len(final_requests)} to {batch_size_limit} due to deadlines")
                # Process in smaller batches - keep first batch_size_limit, queue rest for next batch
                next_batch = final_requests[batch_size_limit:]
                final_requests = final_requests[:batch_size_limit]
                # Re-queue the remaining requests
                for req in next_batch:
                    self.prefill_queues[group_key].append(req)

            # Group by common processing needs
            prompts = [r.prompt for r in final_requests]
            images_list = [r.images for r in final_requests]
            model_sizes = [r.model_size for r in final_requests]
            use_thinking_flags = [r.use_thinking for r in final_requests]

            # Batch process tokenization and vision encoding
            results = self._batch_prefill_processing(
                prompts, images_list, model_sizes[0], use_thinking_flags[0]
            )

            # Resolve futures
            for request, result in zip(final_requests, results):
                if hasattr(request, '_future'):
                    request._future.set_result(result)

        except Exception as e:
            logger.error(f"Prefill batch processing failed: {e}")
            for request in requests:
                if hasattr(request, '_future'):
                    request._future.set_exception(e)

    def _process_decode_batch(self, group_key: GroupKey, requests: List[DecodeRequest]):
        """Process a batch of DECODE requests with deadline checking."""
        try:
            # Check deadlines and filter valid requests
            valid_requests = []
            for request in requests:
                if hasattr(request, 'deadline_s') and request.deadline_s is not None:
                    current_time = time.time()
                    remaining_time = request.deadline_s - current_time
                    if remaining_time <= 0:
                        logger.warning(f"Decode request deadline exceeded: {remaining_time}s remaining")
                        if hasattr(request, '_future'):
                            request._future.set_exception(DeadlineExceededError("Decode deadline exceeded"))
                        continue
                valid_requests.append(request)

            if not valid_requests:
                return

            # Re-check deadlines before batch processing starts
            batch_start_time = time.time()
            final_requests = []
            for request in valid_requests:
                if hasattr(request, 'deadline_s') and request.deadline_s is not None:
                    remaining_time = request.deadline_s - batch_start_time
                    # Estimate batch decoding time (forward + decode for this request)
                    estimated_time = FORWARD_BUDGET_S + DECODE_BUDGET_S
                    if remaining_time < estimated_time:
                        logger.warning(f"Batch decoding would exceed deadline: {remaining_time}s remaining, need {estimated_time}s")
                        if hasattr(request, '_future'):
                            request._future.set_exception(DeadlineExceededError("Batch decode deadline exceeded"))
                        continue
                final_requests.append(request)

            if not final_requests:
                return

            # Adjust batch size based on strictest deadline
            batch_size_limit = len(final_requests)
            if any(hasattr(r, 'deadline_s') and r.deadline_s is not None for r in final_requests):
                strictest_deadline = min(r.deadline_s for r in final_requests if hasattr(r, 'deadline_s') and r.deadline_s is not None)
                remaining_batch_time = strictest_deadline - batch_start_time
                # Estimate time per request in decode batch
                time_per_request = (FORWARD_BUDGET_S + DECODE_BUDGET_S) / 2  # Conservative estimate
                max_batch_size = max(1, int(remaining_batch_time / time_per_request))
                batch_size_limit = min(batch_size_limit, max_batch_size)

            # Apply batch size limit
            if len(final_requests) > batch_size_limit:
                logger.info(f"Limiting decode batch size from {len(final_requests)} to {batch_size_limit} due to deadlines")
                # Process in smaller batches - keep first batch_size_limit, queue rest for next batch
                next_batch = final_requests[batch_size_limit:]
                final_requests = final_requests[:batch_size_limit]
                # Re-queue the remaining requests
                for req in next_batch:
                    self.decode_queues[group_key].append(req)

            # Extract prefill results and temperatures
            prefill_results = [r.prefill_result for r in final_requests]
            temperatures = [r.temperature for r in final_requests]

            # Batch decode
            results = self._batch_decode_processing(prefill_results, temperatures)

            # Resolve futures
            for request, result in zip(final_requests, results):
                if hasattr(request, '_future'):
                    request._future.set_result(result)

        except Exception as e:
            logger.error(f"Decode batch processing failed: {e}")
            for request in requests:
                if hasattr(request, '_future'):
                    request._future.set_exception(e)

    def _batch_prefill_processing(self, prompts: List[str], images_list: List[Optional[List[Any]]],
                                model_size: ModelSize, use_thinking: bool) -> List[PrefillResult]:
        """Batch process PREFILL stage: tokenize, encode vision, consult caches."""
        results = []

        for prompt, images in zip(prompts, images_list):
            # Check prompt cache
            prompt_sha = self._compute_sha(prompt)
            model_name = self.model_router.get_model_name(model_size, use_thinking)
            tokenized = self.model_router.prompt_cache.get_tokenized_prefix(prompt_sha, model_name)

            cache_hit = tokenized is not None
            if not cache_hit:
                # Tokenize and cache
                tokenized = f"tokenized_{prompt_sha}"  # Placeholder
                self.model_router.prompt_cache.cache_tokenized_prefix(prompt_sha, model_name, tokenized)

            # Vision encoding
            vision_encoded = None
            image_sha = None
            if images:
                image_sha = self._compute_sha(str(images))
                vision_encoded = self.model_router.vision_cache.get_encoded_image(image_sha)
                if vision_encoded is None:
                    vision_encoded = f"encoded_{image_sha}"  # Placeholder
                    self.model_router.vision_cache.cache_encoded_image(image_sha, vision_encoded)

            # KV cache check
            kv_cache_key = self.model_router.prompt_kv_cache._make_cache_key(model_name, prompt_sha, image_sha)
            kv_state = self.model_router.prompt_kv_cache.get_kv_state(kv_cache_key)

            result = PrefillResult(
                tokenized_input=tokenized,
                vision_encoded=vision_encoded,
                kv_state=kv_state,
                prompt_sha=prompt_sha,
                image_sha=image_sha,
                cache_hit=cache_hit
            )
            results.append(result)

        return results

    def _batch_decode_processing(self, prefill_results: List[PrefillResult],
                               temperatures: List[float]) -> List[DecodeResult]:
        """Batch process DECODE stage: generate tokens."""
        results = []

        for prefill_result, temperature in zip(prefill_results, temperatures):
            # Use cached KV or generate
            start_time = time.time()
            generated_text = f"Generated for {prefill_result.prompt_sha}"
            latency_ms = (time.time() - start_time) * 1000

            result = DecodeResult(
                generated_text=generated_text,
                tokens_used=50,  # Placeholder
                latency_ms=latency_ms
            )
            results.append(result)

        return results

    def _compute_sha(self, content: str) -> str:
        """Compute SHA hash for content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def force_flush(self):
        """Force immediate flush of all batches."""
        self._flush_all_batches()

        # Wire in caches from qwen_controller
        try:
            from .qwen_controller import PromptCache, VisionCache, PromptKVCache
            from pathlib import Path
            cache_dir = Path(self.model_router.hf_home)
            self.prompt_cache = PromptCache(cache_dir / "pmd_prompt_cache")
            self.vision_cache = VisionCache()
            self.prompt_kv_cache = PromptKVCache(cache_dir, max_ram_entries=5)
            logger.info("Wired in caches from qwen_controller")
        except ImportError:
            logger.warning("Could not import caches from qwen_controller")

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        stats = {"prefill_queue_depth": len(self.prefill_queues),
                 "decode_queue_depth": len(self.decode_queues),
                 "last_flush_time": self.last_flush_time}

        # Add pipeline stats if available
        if self.use_pipeline and self.pipeline_engine:
            stats.update({
                "pipeline_queues": self.pipeline_engine.get_queue_depths(),
                "pipeline_stats": self.pipeline_engine.get_stats(),
            })

        return stats


class SecondaryTrigger:
    """Secondary routing trigger configuration."""
    name: str
    condition_func: Callable[[Dict[str, Any]], bool]
    target_model: ModelSize
    priority: int  # Higher priority triggers override lower ones
    cooldown_seconds: float = 0.0
    last_triggered: float = 0.0