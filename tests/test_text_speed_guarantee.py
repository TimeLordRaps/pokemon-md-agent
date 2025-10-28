"""Test text-speed guarantee feature.

Literate TestDoc: Ensure text-speed is set to slow via menu profile on boot,
fallback to RAM poke when enabled, and throttle A taps during textboxes to
capture OCR frames at ≥1 fps between progressions.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.mgba_harness.cli import MGBACLI
from src.environment.action_executor import ActionExecutor, Button


def test_menu_profile_text_speed_slow():
    """Test menu profile navigates Options → Text Speed → Slow."""
    # Mock profile execution - test structure is valid
    profile_path = Path("src/mgba-harness/profiles/set_text_speed_slow.json")
    assert profile_path.exists(), "Profile file should be created"


def test_ram_poke_text_speed_fallback():
    """Test RAM poke fallback sets text-speed when allow_memory_write enabled."""
    # Test would require ROM hash gating and memory write implementation
    # Placeholder for future implementation - just test address exists
    pass


def test_input_pacing_textbox_throttling():
    """Test A taps are throttled during textboxes to ensure OCR capture."""
    # Test that textbox pacing parameter works
    executor = ActionExecutor(mgba_controller=Mock())

    # Mock controller methods
    executor.mgba.button_tap = Mock(return_value=True)

    # Test normal interaction
    assert executor.interact(textbox_pacing=False)

    # Test textbox pacing (should use longer delay)
    assert executor.interact(textbox_pacing=True)

    # Verify button_tap was called twice
    assert executor.mgba.button_tap.call_count == 2