"""Image packaging and message formatting for vision models.

Provides per-model presets for efficient image packaging with token budget management.
Supports Qwen3-VL variants (2B, 4B, 8B) with optimized image processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelPreset:
    """Configuration preset for a specific vision model."""
    name: str
    vtokens_budget_per_msg: int  # Total visual tokens per message
    max_images_per_msg: int      # Maximum images per message
    retrieved_traj_len: int      # Trajectory length for context
    thumb_scale: float          # Thumbnail scale factor (0.0-1.0)
    image_quality: str          # JPEG quality or format
    max_image_size: tuple[int, int]  # Max width, height in pixels
    compression_level: int      # Compression level (0-9 for PNG, 0-100 for JPEG)
    suppress_grid_in_town: bool  # Suppress grid overlays in town scenes


# Per-model presets optimized for Qwen3-VL variants
MODEL_PRESETS = {
    "qwen3-vl-2b": ModelPreset(
        name="qwen3-vl-2b",
        vtokens_budget_per_msg=4000,  # Conservative for 2B model
        max_images_per_msg=3,
        retrieved_traj_len=5,
        thumb_scale=0.75,
        image_quality="high",
        max_image_size=(480, 320),  # Will be overridden by video config
        compression_level=6,
        suppress_grid_in_town=True,
    ),
    "qwen3-vl-4b": ModelPreset(
        name="qwen3-vl-4b",
        vtokens_budget_per_msg=12000,  # Balanced for 4B model
        max_images_per_msg=4,
        retrieved_traj_len=8,
        thumb_scale=0.85,
        image_quality="high",
        max_image_size=(480, 320),  # Will be overridden by video config
        compression_level=6,
        suppress_grid_in_town=True,
    ),
    "qwen3-vl-8b": ModelPreset(
        name="qwen3-vl-8b",
        vtokens_budget_per_msg=16000,  # Aggressive for 8B model
        max_images_per_msg=6,
        retrieved_traj_len=12,
        thumb_scale=0.95,
        image_quality="high",
        max_image_size=(480, 320),  # Will be overridden by video config
        compression_level=6,
        suppress_grid_in_town=True,
    ),
}


class AgentConfig:
    """Configuration for agent behavior and model selection."""

    def __init__(
        self,
        model_name: str = "qwen3-vl-4b",
        enable_vision: bool = True,
        vision_model_override: Optional[str] = None,
        custom_preset: Optional[ModelPreset] = None,
    ):
        """Initialize agent configuration.

        Args:
            model_name: Primary model name (used for preset lookup)
            enable_vision: Whether vision processing is enabled
            vision_model_override: Override vision model (if different from primary)
            custom_preset: Custom model preset (overrides defaults)
        """
        self.model_name = model_name
        self.enable_vision = enable_vision
        self.vision_model_override = vision_model_override
        self.custom_preset = custom_preset

        # Get effective vision model name
        self.vision_model = vision_model_override or model_name

        # Get preset for vision model - guaranteed to be non-None
        if custom_preset:
            self.preset = custom_preset
        else:
            self.preset = MODEL_PRESETS.get(self.vision_model, MODEL_PRESETS["qwen3-vl-4b"])
            if not self.preset:
                logger.warning("No preset found for model %s, using qwen3-vl-4b defaults", self.vision_model)
                self.preset = MODEL_PRESETS["qwen3-vl-4b"]

    @property
    def vtokens_budget_per_msg(self) -> int:
        """Get visual tokens budget per message."""
        return self.preset.vtokens_budget_per_msg

    @property
    def max_images_per_msg(self) -> int:
        """Get maximum images per message."""
        return self.preset.max_images_per_msg

    @property
    def retrieved_traj_len(self) -> int:
        """Get trajectory length for context."""
        return self.preset.retrieved_traj_len

    @property
    def thumb_scale(self) -> float:
        """Get thumbnail scale factor."""
        return self.preset.thumb_scale

    @property
    def suppress_grid_in_town(self) -> bool:
        """Get whether to suppress grid overlays in town scenes."""
        return self.preset.suppress_grid_in_town


class ImagePackager:
    """Handles image packaging and message formatting for vision models."""

    def __init__(self, config: AgentConfig, video_config=None):
        """Initialize image packager.

        Args:
            config: Agent configuration with model presets
            video_config: Video configuration for dynamic resolution
        """
        self.config = config
        self.preset = config.preset
        self.video_config = video_config

    def package_images(
        self,
        images: List[Dict[str, Any]],
        context: Optional[str] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        is_town_scene: bool = False,
    ) -> Dict[str, Any]:
        """Package images for model consumption.

        Args:
            images: List of image data dicts with 'path', 'timestamp', 'metadata'
            context: Optional text context
            trajectory: Optional trajectory data
            is_town_scene: Whether this is a town scene (affects grid overlay suppression)

        Returns:
            Packaged message dict ready for model input
        """
        if not self.config.enable_vision:
            return {"text": context or "", "images": []}

        # Filter out grid overlays in town scenes if suppression is enabled
        filtered_images = self._filter_images_for_scene(images, is_town_scene)

        # Limit images per message
        limited_images = filtered_images[: self.preset.max_images_per_msg]

        # Process images (resize, compress, etc.)
        processed_images = []
        for img_data in limited_images:
            processed = self._process_image(img_data)
            if processed:
                processed_images.append(processed)

        # Build message
        message = {
            "text": context or "",
            "images": processed_images,
            "metadata": {
                "model": self.config.vision_model,
                "vtokens_budget": self.preset.vtokens_budget_per_msg,
                "image_count": len(processed_images),
                "thumb_scale": self.preset.thumb_scale,
            }
        }

        # Add trajectory context if provided
        if trajectory:
            traj_context = trajectory[-self.preset.retrieved_traj_len :]
            message["trajectory"] = traj_context

        return message

    def _filter_images_for_scene(self, images: List[Dict[str, Any]], is_town_scene: bool) -> List[Dict[str, Any]]:
        """Filter images based on scene type and suppression settings.

        Args:
            images: List of image data dicts
            is_town_scene: Whether this is a town scene

        Returns:
            Filtered list of images
        """
        if not is_town_scene or not self.preset.suppress_grid_in_town:
            return images

        # In town scenes with suppression enabled, filter out grid overlays
        filtered = []
        for img in images:
            # Assume grid overlays can be identified by metadata or path containing 'grid'
            # In a real implementation, this would check image metadata or content
            if not self._is_grid_overlay(img):
                filtered.append(img)
        return filtered

    def _is_grid_overlay(self, image_data: Dict[str, Any]) -> bool:
        """Check if an image is a grid overlay.

        Args:
            image_data: Image data dict

        Returns:
            True if this is a grid overlay image
        """
        # Simple heuristic: check if path or metadata indicates grid overlay
        path = image_data.get("path", "").lower()
        metadata = image_data.get("metadata", {})

        return (
            "grid" in path or
            metadata.get("type") == "grid_overlay" or
            metadata.get("is_grid") == True
        )

    def _process_image(self, image_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single image for packaging.

        Args:
            image_data: Image data dict

        Returns:
            Processed image dict or None if failed
        """
        try:
            # Basic processing - in real implementation would resize/compress
            # Use video config size if available, otherwise fall back to preset
            image_size = self.video_config
            if self.video_config:
                max_size = (self.video_config.width, self.video_config.height)
            else:
                max_size = self.preset.max_image_size
            
            processed = {
                "path": image_data.get("path"),
                "timestamp": image_data.get("timestamp"),
                "size": image_data.get("size", max_size),
                "quality": self.preset.image_quality,
                "scale": self.preset.thumb_scale,
            }

            # Add metadata if available
            if "metadata" in image_data:
                processed["metadata"] = image_data["metadata"]

            return processed

        except (OSError, ValueError, KeyError) as e:
            logger.error("Failed to process image: %s", e)
            return None

    def estimate_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate token count for a packaged message.

        Args:
            message: Packaged message dict

        Returns:
            Estimated token count
        """
        # Rough estimation - would need model-specific tokenizer in real implementation
        text_tokens = len(message.get("text", "").split()) * 1.3  # Rough word to token ratio
        image_tokens = len(message.get("images", [])) * 85  # Rough per-image token estimate
        return int(text_tokens + image_tokens)

    def validate_budget(self, message: Dict[str, Any]) -> bool:
        """Validate that message fits within token budget.

        Args:
            message: Packaged message dict

        Returns:
            True if within budget
        """
        estimated_tokens = self.estimate_tokens(message)
        return estimated_tokens <= self.preset.vtokens_budget_per_msg


# Convenience functions
def get_model_preset(model_name: str) -> ModelPreset:
    """Get preset for a model name.

    Args:
        model_name: Model name

    Returns:
        Model preset
    """
    return MODEL_PRESETS.get(model_name, MODEL_PRESETS["qwen3-vl-4b"])


def create_agent_config(
    model_name: str = "qwen3-vl-4b",
    **kwargs
) -> AgentConfig:
    """Create agent configuration.

    Args:
        model_name: Model name
        **kwargs: Additional config options

    Returns:
        Agent configuration
    """
    return AgentConfig(model_name=model_name, **kwargs)


# Frame Packaging Functions
def env_only(env_path: Path, video_config=None) -> Dict[str, Any]:
    """Package environment screenshot only.

    Args:
        env_path: Path to environment screenshot
        video_config: Video configuration for resolution info

    Returns:
        Packaged frame data
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment image not found: {env_path}")

    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(env_path) as img:
            width, height = img.size
    except Exception as e:
        logger.warning("Could not read image dimensions: %s", e)
        width, height = video_config.width if video_config else 240, video_config.height if video_config else 160

    return {
        "type": "env_only",
        "images": [{
            "path": str(env_path),
            "role": "environment",
            "width": width,
            "height": height,
            "metadata": {
                "type": "environment",
                "description": "Current game environment view"
            }
        }],
        "metadata": {
            "frame_type": "env_only",
            "timestamp": None,  # Would be set by caller
            "resolution": f"{width}x{height}"
        }
    }


def env_plus_grid(env_path: Path, grid_path: Path, video_config=None) -> Dict[str, Any]:
    """Package environment screenshot with grid overlay.

    Args:
        env_path: Path to environment screenshot
        grid_path: Path to grid overlay image
        video_config: Video configuration for resolution info

    Returns:
        Packaged frame data
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment image not found: {env_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid image not found: {grid_path}")

    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(env_path) as img:
            width, height = img.size
    except Exception as e:
        logger.warning("Could not read image dimensions: %s", e)
        width, height = video_config.width if video_config else 240, video_config.height if video_config else 160

    return {
        "type": "env_plus_grid",
        "images": [
            {
                "path": str(env_path),
                "role": "environment",
                "width": width,
                "height": height,
                "metadata": {
                    "type": "environment",
                    "description": "Current game environment view"
                }
            },
            {
                "path": str(grid_path),
                "role": "grid_overlay",
                "width": width,
                "height": height,
                "metadata": {
                    "type": "grid_overlay",
                    "description": "Grid coordinate overlay for spatial reasoning"
                }
            }
        ],
        "metadata": {
            "frame_type": "env_plus_grid",
            "timestamp": None,  # Would be set by caller
            "resolution": f"{width}x{height}",
            "has_grid": True
        }
    }


def env_plus_grid_plus_meta(
    env_path: Path,
    grid_path: Path,
    metadata: Dict[str, Any],
    video_config=None
) -> Dict[str, Any]:
    """Package environment screenshot with grid overlay and metadata.

    Args:
        env_path: Path to environment screenshot
        grid_path: Path to grid overlay image
        metadata: Additional metadata dict
        video_config: Video configuration for resolution info

    Returns:
        Packaged frame data
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment image not found: {env_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid image not found: {grid_path}")

    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(env_path) as img:
            width, height = img.size
    except Exception as e:
        logger.warning("Could not read image dimensions: %s", e)
        width, height = video_config.width if video_config else 240, video_config.height if video_config else 160

    # Merge metadata
    frame_metadata = {
        "frame_type": "env_plus_grid_plus_meta",
        "timestamp": None,  # Would be set by caller
        "resolution": f"{width}x{height}",
        "has_grid": True,
        **metadata  # Merge in additional metadata
    }

    return {
        "type": "env_plus_grid_plus_meta",
        "images": [
            {
                "path": str(env_path),
                "role": "environment",
                "width": width,
                "height": height,
                "metadata": {
                    "type": "environment",
                    "description": "Current game environment view",
                    **metadata.get("env_metadata", {})
                }
            },
            {
                "path": str(grid_path),
                "role": "grid_overlay",
                "width": width,
                "height": height,
                "metadata": {
                    "type": "grid_overlay",
                    "description": "Grid coordinate overlay for spatial reasoning",
                    **metadata.get("grid_metadata", {})
                }
            }
        ],
        "metadata": frame_metadata
    }