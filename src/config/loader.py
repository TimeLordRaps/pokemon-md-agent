"""Config loader utility for agent configuration.

Provides type-safe access to agent configuration with validation,
default fallbacks, and Windows-friendly path handling.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class FPSSettings(BaseModel):
    """FPS adjustment settings."""

    target_fps: float = Field(default=30.0, ge=1.0, le=60.0)
    fps_multipliers: Dict[str, float] = Field(default_factory=lambda: {
        "combat": 1.5, "menu": 0.5, "exploration": 1.0
    })
    fps_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "low_threshold": 0.8, "high_threshold": 1.2
    })

    @field_validator("fps_multipliers")
    @classmethod
    def validate_multipliers(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate FPS multipliers are reasonable."""
        for key, value in v.items():
            if not 0.1 <= value <= 3.0:
                raise ValueError(f"FPS multiplier for {key} must be between 0.1 and 3.0")
        return v


class ModelRoutingSettings(BaseModel):
    """Model routing parameters."""

    time_budgets: Dict[str, float] = Field(default_factory=lambda: {
        "total_loop": 0.1, "vision_processing": 0.03,
        "retrieval_processing": 0.04, "inference": 0.02, "action_execution": 0.01
    })
    model_preferences: Dict[str, Any] = Field(default_factory=lambda: {
        "size_order": ["small", "medium", "large"],
        "max_context_tokens": 4096, "quality_preference": 0.7
    })
    routing_thresholds: Dict[str, Any] = Field(default_factory=lambda: {
        "confidence_threshold": 0.8, "max_routing_retries": 3,
        "model_switch_timeout": 5.0
    })

    @field_validator("time_budgets")
    @classmethod
    def validate_time_budgets(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate time budgets are reasonable."""
        for key, value in v.items():
            if value < 0.001:  # Minimum 1ms
                raise ValueError(f"Time budget for {key} too small: {value}")
        return v


class MemoryManagementSettings(BaseModel):
    """Memory management settings."""

    vram_budgets: Dict[str, int] = Field(default_factory=lambda: {
        "screenshot_buffer": 50, "sprite_cache": 25,
        "grid_data": 10, "total_limit": 100
    })
    context_limits: Dict[str, int] = Field(default_factory=lambda: {
        "max_keyframes": 1000, "max_ann_entries": 5000,
        "max_sprite_cache": 500, "max_retrieval_results": 50
    })
    eviction_policies: Dict[str, str] = Field(default_factory=lambda: {
        "keyframe_eviction": "lru", "sprite_eviction": "lfu",
        "ann_eviction": "similarity", "retrieval_eviction": "time_based"
    })

    @field_validator("vram_budgets", "context_limits")
    @classmethod
    def validate_positive_values(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate that all values are positive."""
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"{key} must be non-negative, got {value}")
        return v


class RetrievalSettings(BaseModel):
    """Retrieval and gatekeeper settings."""

    buffer_config: Dict[str, Any] = Field(default_factory=lambda: {
        "window_minutes": 60,
        "keyframe_triggers": {
            "ssim_threshold": 0.1, "floor_change": True,
            "combat_events": True, "inventory_changes": True, "new_species": True
        }
    })
    ann_config: Dict[str, Any] = Field(default_factory=lambda: {
        "algorithm": "annoy", "dimensions": 512, "trees": 10, "max_distance": 0.5
    })
    gatekeeper_config: Dict[str, Any] = Field(default_factory=lambda: {
        "min_shallow_hits": 3, "max_web_queries": 10,
        "content_api_budget": 1000, "batch_size": 50, "web_timeout": 30.0
    })
    stuckness_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_stuck_turns": 50, "stuck_similarity_threshold": 0.95,
        "stuck_check_interval": 10.0
    })


class RuntimeParams(BaseModel):
    """Runtime parameters."""

    rate_limits: Dict[str, float] = Field(default_factory=lambda: {
        "screenshots_per_second": 30.0, "memory_polls_per_second": 10.0,
        "memory_polls_combat": 20.0
    })
    resilience: Dict[str, Any] = Field(default_factory=lambda: {
        "socket_timeout": 10.0, "max_reconnect_attempts": 5,
        "backoff_multiplier": 2.0, "max_backoff_delay": 60.0
    })
    sprite_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "confidence_threshold": 0.85, "hash_similarity_threshold": 0.9,
        "max_sprites_per_frame": 20
    })
    snapshot_policies: Dict[str, Any] = Field(default_factory=lambda: {
        "floor_transitions": True, "turns_interval": 10,
        "room_changes": True, "combat_events": True
    })


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    fps_settings: FPSSettings = Field(default_factory=FPSSettings)
    model_routing: ModelRoutingSettings = Field(default_factory=ModelRoutingSettings)
    memory_management: MemoryManagementSettings = Field(default_factory=MemoryManagementSettings)
    retrieval_settings: RetrievalSettings = Field(default_factory=RetrievalSettings)
    runtime_params: RuntimeParams = Field(default_factory=RuntimeParams)


class ConfigLoader:
    """Config loader with validation and default fallbacks."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config loader.

        Args:
            config_path: Path to config file. Defaults to config/agent_config.yaml
        """
        if config_path is None:
            # Use relative path for no-absolute-paths constraint
            self.config_path = Path("config") / "agent_config.yaml"
        else:
            self.config_path = config_path

        # Normalize path separators for Windows compatibility
        self.config_path = Path(str(self.config_path).replace("\\", "/"))

        logger.debug(f"Config loader initialized with path: {self.config_path}")

    def load_config(self) -> AgentConfig:
        """Load and validate configuration from file.

        Returns:
            Validated AgentConfig instance

        Raises:
            ConfigLoadError: If loading or validation fails
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return AgentConfig()

            logger.info(f"Loading config from: {self.config_path}")

            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                logger.warning("Config file is empty, using defaults")
                return AgentConfig()

            # Validate and create config
            config = AgentConfig(**data)
            logger.info("Config loaded and validated successfully")
            return config

        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in config file: {e}"
            logger.error(error_msg)
            raise ConfigLoadError(error_msg) from e
        except ValidationError as e:
            error_msg = f"Config validation failed: {e}"
            logger.error(error_msg)
            raise ConfigLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading config: {e}"
            logger.error(error_msg)
            raise ConfigLoadError(error_msg) from e

    def get_section(self, section_name: str) -> Any:
        """Get a specific config section by name.

        Args:
            section_name: Name of the section to retrieve

        Returns:
            The requested config section

        Raises:
            ConfigLoadError: If section doesn't exist
        """
        config = self.load_config()

        # Map section names to attributes
        section_map = {
            "fps_settings": config.fps_settings,
            "model_routing": config.model_routing,
            "memory_management": config.memory_management,
            "retrieval_settings": config.retrieval_settings,
            "runtime_params": config.runtime_params
        }

        if section_name not in section_map:
            available = ", ".join(section_map.keys())
            raise ConfigLoadError(f"Unknown config section '{section_name}'. Available: {available}")

        return section_map[section_name]

    def reload_config(self) -> AgentConfig:
        """Reload configuration from file.

        Returns:
            Fresh AgentConfig instance
        """
        logger.info("Reloading configuration")
        return self.load_config()