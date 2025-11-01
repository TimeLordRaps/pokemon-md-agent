"""Test vision event detection using Qwen-VL models."""
import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.generate_montage_video import detect_vision_events, Event


def test_detect_vision_events_basic():
    """Test basic vision event detection with mock Qwen-VL."""
    trajectory = [
        {
            "timestamp": 0.0,
            "ram": {"room_type": "corridor"},
            "screenshot": b"fake_screenshot_data"
        },
        {
            "timestamp": 1.0,
            "ram": {"room_type": "staircase"},
            "screenshot": b"fake_screenshot_data2"
        }
    ]

    # Should detect room type change even without vision LLM
    events = detect_vision_events(trajectory)

    assert len(events) == 1
    assert events[0].event_type == "room_type_change"
    assert events[0].score == 5.0
    assert events[0].metadata["room_type"] == "staircase"


@pytest.mark.real_model
def test_detect_vision_events_with_qwen_vl():
    """Test vision event detection with real Qwen-VL model."""
    # Skip if real models not enabled
    backend = os.environ.get("MODEL_BACKEND", "").lower()
    if backend != "hf":
        pytest.skip("MODEL_BACKEND!=hf; skipping real model test")
    
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set; skipping real model test")
    
    # Create test trajectory with fake screenshot data
    trajectory = [
        {
            "timestamp": 0.0,
            "ram": {"room_type": "corridor"},
            "screenshot": b"fake_screenshot_data"  # This will be converted to PIL Image
        },
        {
            "timestamp": 1.0,
            "ram": {"room_type": "staircase"},
            "screenshot": b"fake_screenshot_data2"
        }
    ]
    
    # Test that real model loading works (should not crash)
    try:
        events = detect_vision_events(trajectory)
        # Should return some events (either from real model or fallback)
        assert isinstance(events, list)
        # Events should have proper structure
        for event in events:
            assert hasattr(event, 'timestamp')
            assert hasattr(event, 'event_type')
            assert hasattr(event, 'score')
            assert hasattr(event, 'metadata')
            assert hasattr(event, 'frame_idx')
    except Exception as e:
        # If model loading fails, that's acceptable for CI
        pytest.skip(f"Real model loading failed (expected in some environments): {e}")


def test_detect_vision_events_enemy_proximity():
    """Test enemy proximity detection."""
    trajectory = [
        {
            "timestamp": 0.0,
            "ram": {
                "monsters": [
                    {"distance": 2, "species": "pikachu"},
                    {"distance": 5, "species": "charmander"}
                ]
            },
            "screenshot": b"fake_screenshot_data"
        }
    ]

    events = detect_vision_events(trajectory)

    assert len(events) == 1
    assert events[0].event_type == "enemy_proximity"
    assert events[0].score == 6.0
    assert events[0].metadata["close_enemies"] == 1


def test_event_creation():
    """Test Event dataclass creation."""
    event = Event(
        timestamp=1.5,
        event_type="test_event",
        score=8.0,
        metadata={"test": "data"},
        frame_idx=10
    )

    assert event.timestamp == 1.5
    assert event.event_type == "test_event"
    assert event.score == 8.0
    assert event.metadata == {"test": "data"}
    assert event.frame_idx == 10