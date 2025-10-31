"""Configuration classes for environment components."""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ResolutionProfile:
    """A supported resolution profile for video capture.

    Attributes:
        width: Output width in pixels
        height: Output height in pixels
        scale: Scale factor from base resolution (240x160)
        name: Human-readable name for the profile
    """
    width: int
    height: int
    scale: int
    name: str

    @property
    def size(self) -> Tuple[int, int]:
        """Get the resolution as a (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class VideoConfig:
    """Configuration for video capture resolution and scaling.

    Attributes:
        width: Base width of the game screen in pixels (typically 240)
        height: Base height of the game screen in pixels (typically 160)
        scale: Upscaling factor for capture (typically 2 for 480x320 output)
        supported_profiles: Dict of named resolution profiles
    """
    width: int = 240
    height: int = 160
    scale: int = 2
    supported_profiles: Optional[Dict[str, ResolutionProfile]] = None

    def __post_init__(self):
        """Initialize supported resolution profiles."""
        if self.supported_profiles is None:
            self.supported_profiles = {
                "1x": ResolutionProfile(width=240, height=160, scale=1, name="1x Base"),
                "2x": ResolutionProfile(width=480, height=320, scale=2, name="2x Standard"),
                "4x": ResolutionProfile(width=960, height=640, scale=4, name="4x High-res"),
            }

    @property
    def scaled_width(self) -> int:
        """Get the scaled width."""
        return self.width * self.scale

    @property
    def scaled_height(self) -> int:
        """Get the scaled height."""
        return self.height * self.scale

    @property
    def current_profile(self) -> ResolutionProfile:
        """Get the current resolution profile based on scale."""
        assert self.supported_profiles is not None
        for profile in self.supported_profiles.values():
            if profile.scale == self.scale:
                return profile
        # Fallback to closest match
        return min(
            self.supported_profiles.values(),
            key=lambda p: abs(p.scale - self.scale)
        )

    def get_supported_sizes(self) -> set[Tuple[int, int]]:
        """Get all supported resolution sizes."""
        assert self.supported_profiles is not None
        return {profile.size for profile in self.supported_profiles.values()}

    def infer_profile_from_size(self, size: Tuple[int, int]) -> Optional[ResolutionProfile]:
        """Infer the resolution profile from an image size.

        Args:
            size: (width, height) tuple

        Returns:
            Matching ResolutionProfile or None if no match
        """
        assert self.supported_profiles is not None
        for profile in self.supported_profiles.values():
            if profile.size == size:
                return profile
        return None

    def find_nearest_profile(self, size: Tuple[int, int]) -> ResolutionProfile:
        """Find the nearest supported resolution profile for a given size.

        Args:
            size: (width, height) tuple

        Returns:
            Nearest ResolutionProfile
        """
        assert self.supported_profiles is not None
        width, height = size

        # Calculate aspect ratio difference and size difference
        def profile_distance(profile: ResolutionProfile) -> float:
            # Aspect ratio difference (0-1, lower is better)
            expected_ratio = profile.width / profile.height
            actual_ratio = width / height
            ratio_diff = abs(expected_ratio - actual_ratio)

            # Size difference (normalized)
            size_diff = abs(profile.width - width) + abs(profile.height - height)
            size_diff_norm = size_diff / max(width, height, profile.width, profile.height)

            # Weighted combination (prioritize aspect ratio)
            return ratio_diff * 0.7 + size_diff_norm * 0.3

        return min(self.supported_profiles.values(), key=profile_distance)


@dataclass
class MGBAConfig:
    """Configuration for mGBA emulator connection.
    
    Attributes:
        port: Port for mGBA Lua socket server (default: 8888)
        host: Host for mGBA Lua socket server (default: localhost)
        timeout: Connection timeout in seconds (default: 10.0)
    """
    port: int = 8888
    host: str = "localhost"
    timeout: float = 10.0
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Read MGBA_PORT from environment with fallback to default
        env_port = os.environ.get('MGBA_PORT')
        if env_port:
            try:
                self.port = int(env_port)
            except ValueError:
                pass  # Keep default if invalid
        
        # Could add other env vars here if needed in future
        # host from env, timeout from env, etc.
        
        # Validate port range
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid MGBA_PORT: {self.port} (must be 1-65535)")