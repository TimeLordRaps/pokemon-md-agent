"""Utilities for managing context limits on Qwen3-VL models."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol
import logging

logger = logging.getLogger(__name__)

DEFAULT_SAFETY_BUFFER = 128


class HasContextLength(Protocol):
    """Protocol for registry entries exposing a context_length attribute."""

    context_length: int


def resolve_context_cap(
    registry: Mapping[str, HasContextLength] | Mapping[str, Mapping[str, Any]],
    model_key: str,
    fallback: Optional[int] = None,
) -> int:
    """Resolve the context window cap for a given registry model.

    Args:
        registry: Mapping of model keys to registry entries.
        model_key: Registry key such as ``qwen3-vl-4b-instruct``.
        fallback: Optional fallback value if registry lacks context length.

    Returns:
        Integer context cap in tokens.

    Raises:
        ValueError: If no cap is available and no fallback provided.
    """
    entry = registry.get(model_key)
    if entry is None:
        if fallback is not None:
            logger.debug("Model %s missing; using fallback context cap %d", model_key, fallback)
            return fallback
        raise ValueError(f"Model {model_key} not present in registry; cannot resolve context cap")

    if isinstance(entry, Mapping):
        cap = entry.get("context_length")
    else:
        cap = getattr(entry, "context_length", None)

    if cap is not None:
        return int(cap)

    if fallback is not None:
        logger.debug("Model %s lacks cap; using fallback %d", model_key, fallback)
        return fallback

    raise ValueError(f"Context cap unavailable for {model_key}")


def clamp_generation_length(
    input_tokens: int,
    requested_new_tokens: int,
    context_cap: int,
    safety_buffer: int = DEFAULT_SAFETY_BUFFER,
) -> int:
    """Clamp generation length to respect the model's context window.

    Args:
        input_tokens: Number of tokens already consumed by the prompt.
        requested_new_tokens: Desired number of new tokens.
        context_cap: Maximum context length supported by the model.
        safety_buffer: Reserved tokens to avoid boundary edge cases.

    Returns:
        Allowed number of new tokens (>= 0).
    """
    usable_cap = max(context_cap - safety_buffer, 0)
    remaining = max(usable_cap - input_tokens, 0)
    allowed = max(min(requested_new_tokens, remaining), 0)

    if allowed < requested_new_tokens:
        logger.debug(
            "Clamped generation from %d→%d tokens (input=%d, cap=%d, buffer=%d)",
            requested_new_tokens,
            allowed,
            input_tokens,
            context_cap,
            safety_buffer,
        )
    return allowed


def should_skip_length(
    sequence_length: int,
    context_cap: int,
    safety_buffer: int = DEFAULT_SAFETY_BUFFER,
) -> bool:
    """Determine if a benchmark sequence length exceeds the allowed context window.

    Args:
        sequence_length: Total tokens to evaluate (prompt + expected output).
        context_cap: Maximum context length for the model.
        safety_buffer: Reserved tokens to leave unused.

    Returns:
        True if the sequence should be skipped because it would exceed the cap.
    """
    usable_cap = max(context_cap - safety_buffer, 0)
    skip = sequence_length > usable_cap
    if skip:
        logger.info(
            "Skipping sequence length %d; usable cap %d (buffer=%d)",
            sequence_length,
            usable_cap,
            safety_buffer,
        )
    return skip
