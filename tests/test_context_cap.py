"""Tests for context auto-cap utilities."""

import logging
from types import SimpleNamespace

import pytest

from src.agent.context_cap import (
    clamp_generation_length,
    resolve_context_cap,
    should_skip_length,
    DEFAULT_SAFETY_BUFFER,
)


class TestResolveContextCap:
    """Tests for resolving context limits."""

    def test_resolve_from_namespace_entry(self) -> None:
        """Resolve cap when registry entry exposes attribute."""
        registry = {
            "qwen3": SimpleNamespace(context_length=8192),
        }

        assert resolve_context_cap(registry, "qwen3") == 8192

    def test_resolve_from_mapping_entry(self) -> None:
        """Resolve cap when registry entry is mapping."""
        registry = {
            "qwen3": {"context_length": 4096},
        }

        assert resolve_context_cap(registry, "qwen3", fallback=1024) == 4096

    def test_missing_entry_uses_fallback(self) -> None:
        """Fallback used if entry missing."""
        registry = {}
        assert resolve_context_cap(registry, "missing", fallback=2048) == 2048

    def test_missing_entry_without_fallback_raises(self) -> None:
        """Missing entry without fallback raises ValueError."""
        registry = {}
        with pytest.raises(ValueError):
            resolve_context_cap(registry, "missing")


class TestClampGenerationLength:
    """Tests for clamping generation tokens."""

    def test_clamp_within_cap(self) -> None:
        """Return requested tokens when within cap."""
        allowed = clamp_generation_length(
            input_tokens=1000,
            requested_new_tokens=512,
            context_cap=4096,
            safety_buffer=128,
        )
        assert allowed == 512

    def test_clamp_exceeds_cap(self, caplog: pytest.LogCaptureFixture) -> None:
        """Clamp when request exceeds remaining space."""
        with caplog.at_level(logging.DEBUG):
            allowed = clamp_generation_length(
                input_tokens=4000,
                requested_new_tokens=512,
                context_cap=4096,
                safety_buffer=64,
            )
        assert allowed == 32
        assert "Clamped generation" in caplog.text


class TestShouldSkipLength:
    """Tests for benchmark skip helper."""

    def test_skip_when_length_exceeds_cap(self, caplog: pytest.LogCaptureFixture) -> None:
        """Skip when sequence above usable cap."""
        with caplog.at_level(logging.INFO):
            should_skip = should_skip_length(9000, 8000, safety_buffer=1000)
        assert should_skip is True
        assert "Skipping sequence length" in caplog.text

    def test_do_not_skip_within_cap(self) -> None:
        """Do not skip when within usable cap."""
        should_skip = should_skip_length(1000, 8000, safety_buffer=DEFAULT_SAFETY_BUFFER)
        assert should_skip is False
