"""Test vision event detection using Qwen-VL models."""
import pytest
from unittest.mock import Mock, patch
from pokemon_md_agent.scripts.generate_montage_video import detect_vision_events, Event


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
    # This test would only run when MODEL_BACKEND=hf and real models are available
    pytest.skip("Real model tests require HF_TOKEN and MODEL_BACKEND=hf environment variables")


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