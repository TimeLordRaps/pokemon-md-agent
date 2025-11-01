"""
Unit tests for vision packaging functionality.

Tests model presets, budgets, and town scene grid overlay suppression.
"""

import pytest
from src.vision.packaging import (
    ModelPreset,
    AgentConfig,
    ImagePackager,
    MODEL_PRESETS,
    get_model_preset,
    create_agent_config,
)


def test_model_presets_structure():
    """Test that all three presets (2B, 4B, 8B) have correct structure and budgets."""
    # Check all presets exist
    assert "qwen3-vl-2b" in MODEL_PRESETS
    assert "qwen3-vl-4b" in MODEL_PRESETS
    assert "qwen3-vl-8b" in MODEL_PRESETS

    # Test 2B preset
    preset_2b = MODEL_PRESETS["qwen3-vl-2b"]
    assert isinstance(preset_2b, ModelPreset)
    assert preset_2b.vtokens_budget_per_msg == 4000
    assert preset_2b.max_images_per_msg == 3
    assert preset_2b.retrieved_traj_len == 5
    assert preset_2b.suppress_grid_in_town is True

    # Test 4B preset
    preset_4b = MODEL_PRESETS["qwen3-vl-4b"]
    assert isinstance(preset_4b, ModelPreset)
    assert preset_4b.vtokens_budget_per_msg == 12000
    assert preset_4b.max_images_per_msg == 4
    assert preset_4b.retrieved_traj_len == 8
    assert preset_4b.suppress_grid_in_town is True

    # Test 8B preset
    preset_8b = MODEL_PRESETS["qwen3-vl-8b"]
    assert isinstance(preset_8b, ModelPreset)
    assert preset_8b.vtokens_budget_per_msg == 16000
    assert preset_8b.max_images_per_msg == 6
    assert preset_8b.retrieved_traj_len == 12
    assert preset_8b.suppress_grid_in_town is True


def test_model_preset_sizes():
    """Test that presets have increasing budgets and capacities."""
    preset_2b = MODEL_PRESETS["qwen3-vl-2b"]
    preset_4b = MODEL_PRESETS["qwen3-vl-4b"]
    preset_8b = MODEL_PRESETS["qwen3-vl-8b"]

    # Budgets should increase
    assert preset_2b.vtokens_budget_per_msg < preset_4b.vtokens_budget_per_msg
    assert preset_4b.vtokens_budget_per_msg < preset_8b.vtokens_budget_per_msg

    # Max images should increase
    assert preset_2b.max_images_per_msg < preset_4b.max_images_per_msg
    assert preset_4b.max_images_per_msg < preset_8b.max_images_per_msg

    # Trajectory lengths should increase
    assert preset_2b.retrieved_traj_len < preset_4b.retrieved_traj_len
    assert preset_4b.retrieved_traj_len < preset_8b.retrieved_traj_len


def test_agent_config_properties():
    """Test AgentConfig exposes preset properties correctly."""
    config = AgentConfig(model_name="qwen3-vl-4b")

    assert config.vtokens_budget_per_msg == 12000
    assert config.max_images_per_msg == 4
    assert config.retrieved_traj_len == 8
    assert config.suppress_grid_in_town is True


def test_get_model_preset():
    """Test get_model_preset utility function."""
    preset = get_model_preset("qwen3-vl-2b")
    assert isinstance(preset, ModelPreset)
    assert preset.vtokens_budget_per_msg == 4000

    # Test fallback
    fallback = get_model_preset("unknown-model")
    assert isinstance(fallback, ModelPreset)
    assert fallback.vtokens_budget_per_msg == 12000  # 4B default


def test_create_agent_config():
    """Test create_agent_config utility function."""
    config = create_agent_config("qwen3-vl-8b")
    assert isinstance(config, AgentConfig)
    assert config.model_name == "qwen3-vl-8b"
    assert config.vtokens_budget_per_msg == 16000


def test_image_packager_initialization():
    """Test ImagePackager initializes correctly."""
    config = AgentConfig(model_name="qwen3-vl-4b")
    packager = ImagePackager(config)

    assert packager.config is config
    assert packager.preset is config.preset


def test_package_images_basic():
    """Test basic image packaging functionality."""
    config = AgentConfig(model_name="qwen3-vl-4b")
    packager = ImagePackager(config)

    images = [
        {"path": "env.png", "timestamp": 1.0, "metadata": {}},
        {"path": "grid.png", "timestamp": 2.0, "metadata": {"type": "grid_overlay"}},
    ]

    result = packager.package_images(images, context="test context")

    assert "text" in result
    assert "images" in result
    assert "metadata" in result
    assert result["text"] == "test context"
    assert len(result["images"]) <= 4  # max_images_per_msg for 4B


def test_town_scene_grid_suppression():
    """Test that grid overlays are suppressed in town scenes."""
    config = AgentConfig(model_name="qwen3-vl-4b")
    packager = ImagePackager(config)

    # Create images with grid overlay
    images = [
        {"path": "env.png", "timestamp": 1.0, "metadata": {}},
        {"path": "grid_overlay.png", "timestamp": 2.0, "metadata": {"type": "grid_overlay"}},
        {"path": "map.png", "timestamp": 3.0, "metadata": {}},
    ]

    # Test non-town scene (no suppression)
    result_dungeon = packager.package_images(images, is_town_scene=False)
    assert len(result_dungeon["images"]) == 3  # All images included

    # Test town scene (suppression enabled)
    result_town = packager.package_images(images, is_town_scene=True)
    assert len(result_town["images"]) == 2  # Grid overlay filtered out
    # Check that env.png and map.png are included, but not grid_overlay.png
    paths = [img["path"] for img in result_town["images"]]
    assert "env.png" in paths
    assert "map.png" in paths
    assert "grid_overlay.png" not in paths


def test_town_scene_no_suppression_override():
    """Test town scene suppression can be overridden via custom preset."""
    # Create custom preset with suppression disabled
    custom_preset = ModelPreset(
        name="custom",
        vtokens_budget_per_msg=10000,
        max_images_per_msg=3,
        retrieved_traj_len=6,
        thumb_scale=0.8,
        image_quality="medium",
        max_image_size=(400, 300),
        compression_level=5,
        suppress_grid_in_town=False,  # Explicitly disable suppression
    )

    config = AgentConfig(custom_preset=custom_preset)
    packager = ImagePackager(config)

    images = [
        {"path": "env.png", "timestamp": 1.0, "metadata": {}},
        {"path": "grid_overlay.png", "timestamp": 2.0, "metadata": {"type": "grid_overlay"}},
    ]

    # Even in town scene, grid should not be suppressed
    result = packager.package_images(images, is_town_scene=True)
    assert len(result["images"]) == 2  # Both images included
    paths = [img["path"] for img in result["images"]]
    assert "grid_overlay.png" in paths


def test_grid_overlay_detection():
    """Test grid overlay detection logic."""
    config = AgentConfig(model_name="qwen3-vl-4b")
    packager = ImagePackager(config)

    # Test various grid overlay indicators
    test_cases = [
        ({"path": "grid.png"}, True),
        ({"path": "env_grid.png"}, True),
        ({"metadata": {"type": "grid_overlay"}}, True),
        ({"metadata": {"is_grid": True}}, True),
        ({"path": "env.png"}, False),
        ({"metadata": {"type": "environment"}}, False),
    ]

    for image_data, expected in test_cases:
        assert packager._is_grid_overlay(image_data) == expected


def test_budget_validation():
    """Test token budget validation."""
    config = AgentConfig(model_name="qwen3-vl-2b")  # 4000 budget
    packager = ImagePackager(config)

    # Create a message that fits within budget
    message = {
        "text": "Short message",
        "images": [{"path": "test.png"}] * 3,  # 3 images
    }

    assert packager.validate_budget(message) is True

    # Create oversized message
    oversized_message = {
        "text": "Very long message " * 1000,  # Long text
        "images": [{"path": "test.png"}] * 10,  # Many images
    }

    assert packager.validate_budget(oversized_message) is False


def test_image_processing_error_handling():
    """Test that image processing handles errors gracefully."""
    config = AgentConfig(model_name="qwen3-vl-4b")
    packager = ImagePackager(config)

    # Image with invalid data
    invalid_images = [
        {"path": "env.png", "timestamp": 1.0},  # Missing metadata
        {"invalid": "data"},  # Completely invalid
    ]

    result = packager.package_images(invalid_images)
    # Should still produce valid result, possibly with empty or filtered images
    assert "images" in result
    assert isinstance(result["images"], list)