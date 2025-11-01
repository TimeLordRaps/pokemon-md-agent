"""Test config loader utility."""
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config.loader import ConfigLoader, ConfigLoadError, FPSSettings, ModelRoutingSettings, MemoryManagementSettings, RetrievalSettings, RuntimeParams


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_config_loader_initialization():
    """Test ConfigLoader initializes correctly."""
    loader = ConfigLoader()
    assert loader.config_path is not None
    assert isinstance(loader.config_path, Path)


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_load_valid_config():
    """Test loading valid config file."""
    config_yaml = """
fps_settings:
  target_fps: 30.0
  fps_multipliers:
    combat: 1.5
    menu: 0.5
    exploration: 1.0
  fps_thresholds:
    low_threshold: 0.8
    high_threshold: 1.2

model_routing:
  time_budgets:
    total_loop: 0.1
    vision_processing: 0.03
    retrieval_processing: 0.04
    inference: 0.02
    action_execution: 0.01
  model_preferences:
    size_order: ["small", "medium", "large"]
    max_context_tokens: 4096
    quality_preference: 0.7
  routing_thresholds:
    confidence_threshold: 0.8
    max_routing_retries: 3
    model_switch_timeout: 5.0

memory_management:
  vram_budgets:
    screenshot_buffer: 50
    sprite_cache: 25
    grid_data: 10
    total_limit: 100
  context_limits:
    max_keyframes: 1000
    max_ann_entries: 5000
    max_sprite_cache: 500
    max_retrieval_results: 50
  eviction_policies:
    keyframe_eviction: "lru"
    sprite_eviction: "lfu"
    ann_eviction: "similarity"
    retrieval_eviction: "time_based"

retrieval_settings:
  buffer_config:
    window_minutes: 60
    keyframe_triggers:
      ssim_threshold: 0.1
      floor_change: true
      combat_events: true
      inventory_changes: true
      new_species: true
  ann_config:
    algorithm: "annoy"
    dimensions: 512
    trees: 10
    max_distance: 0.5
  gatekeeper_config:
    min_shallow_hits: 3
    max_web_queries: 10
    content_api_budget: 1000
    batch_size: 50
    web_timeout: 30.0
  stuckness_config:
    max_stuck_turns: 50
    stuck_similarity_threshold: 0.95
    stuck_check_interval: 10.0

runtime_params:
  rate_limits:
    screenshots_per_second: 30.0
    memory_polls_per_second: 10.0
    memory_polls_combat: 20.0
  resilience:
    socket_timeout: 10.0
    max_reconnect_attempts: 5
    backoff_multiplier: 2.0
    max_backoff_delay: 60.0
  sprite_detection:
    confidence_threshold: 0.85
    hash_similarity_threshold: 0.9
    max_sprites_per_frame: 20
  snapshot_policies:
    floor_transitions: true
    turns_interval: 10
    room_changes: true
    combat_events: true
"""

    with patch("builtins.open", mock_open(read_data=config_yaml)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = ConfigLoader()
            config = loader.load_config()

            assert config is not None
            assert isinstance(config.fps_settings, FPSSettings)
            assert isinstance(config.model_routing, ModelRoutingSettings)
            assert isinstance(config.memory_management, MemoryManagementSettings)
            assert isinstance(config.retrieval_settings, RetrievalSettings)
            assert isinstance(config.runtime_params, RuntimeParams)

            # Check some default values
            assert config.fps_settings.target_fps == 30.0
            assert config.model_routing.routing_thresholds["confidence_threshold"] == 0.8
            assert config.memory_management.vram_budgets["total_limit"] == 100
            assert config.retrieval_settings.gatekeeper_config["min_shallow_hits"] == 3
            assert config.runtime_params.rate_limits["screenshots_per_second"] == 30.0


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_config_loader_defaults():
    """Test default fallback values when config sections are missing."""
    config_yaml = """
fps_settings:
  target_fps: 25.0
"""

    with patch("builtins.open", mock_open(read_data=config_yaml)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = ConfigLoader()
            config = loader.load_config()

            # Check that missing sections have defaults
            assert config.fps_settings.target_fps == 25.0
            # Other sections should have default values
            assert config.model_routing.time_budgets["total_loop"] == 0.1
            assert config.memory_management.vram_budgets["total_limit"] == 100
            assert config.retrieval_settings.gatekeeper_config["min_shallow_hits"] == 3
            assert config.runtime_params.rate_limits["screenshots_per_second"] == 30.0


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_config_loader_missing_file():
    """Test error handling when config file is missing."""
    with patch("pathlib.Path.exists", return_value=False):
        loader = ConfigLoader()
        # Should not raise - defaults are used
        config = loader.load_config()
        assert config is not None
        assert hasattr(config, 'fps_settings')  # Basic check


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_config_loader_invalid_yaml():
    """Test error handling for invalid YAML."""
    invalid_yaml = "invalid: yaml: content: ["

    with patch("builtins.open", mock_open(read_data=invalid_yaml)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = ConfigLoader()
            with pytest.raises(ConfigLoadError):
                loader.load_config()


@pytest.mark.skip(reason="ConfigLoader functionality not yet implemented")
def test_config_loader_validation_error():
    """Test validation error for invalid config values."""
    invalid_config = """
fps_settings:
  target_fps: -1  # Invalid negative FPS
"""

    with patch("builtins.open", mock_open(read_data=invalid_config)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = ConfigLoader()
            with pytest.raises(ConfigLoadError):
                loader.load_config()