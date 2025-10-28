"""Dynamic FPS and frame multiplier adjustment for temporal resolution control."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FPSLevel(Enum):
    """FPS adjustment levels."""
    FPS_30 = 30
    FPS_10 = 10
    FPS_5 = 5
    FPS_3 = 3
    FPS_1 = 1


@dataclass
class FPSConfig:
    """Configuration for FPS and frame multiplier settings."""
    base_fps: int
    current_fps: int
    frame_multiplier: int
    allowed_fps_levels: List[int]


class FPSAdjuster:
    """Manages dynamic FPS and frame multiplier adjustment."""
    
    def __init__(
        self,
        base_fps: int = 30,
        allowed_fps: Optional[List[int]] = None,
        initial_multiplier: int = 4,
    ):
        """Initialize FPS adjuster.
        
        Args:
            base_fps: Base framerate (default 30fps)
            allowed_fps: List of allowed FPS levels
            initial_multiplier: Initial frame multiplier
        """
        self.base_fps = base_fps
        self.allowed_fps = allowed_fps or [30, 10, 5, 3, 1]
        self.frame_multiplier = initial_multiplier
        
        # Track adjustment history for analysis
        self.adjustment_history: List[Dict] = []
        self._current_fps = self.base_fps // self.frame_multiplier
        
        logger.info(
            "Initialized FPSAdjuster: base=%dfps, multiplier=%dx, allowed=%s",
            base_fps,
            initial_multiplier,
            self.allowed_fps
        )
    
    def set_fps(self, target_fps: int) -> bool:
        """Set target FPS.
        
        Args:
            target_fps: Target framerate (must be in allowed_fps)
            
        Returns:
            True if change succeeded
        """
        if target_fps not in self.allowed_fps:
            logger.warning(
                "FPS %d not in allowed levels: %s",
                target_fps,
                self.allowed_fps
            )
            return False
        
        if target_fps > self.base_fps:
            logger.warning(
                "Target FPS %d exceeds base FPS %d",
                target_fps,
                self.base_fps
            )
            return False
        
        old_fps = self.get_current_fps()
        self._set_current_fps(target_fps)
        
        self._record_adjustment(
            "fps_change",
            old_fps=old_fps,
            new_fps=target_fps,
            frame_multiplier=self.frame_multiplier
        )
        
        logger.info("FPS adjusted: %d -> %d", old_fps, target_fps)
        return True
    
    def get_current_fps(self) -> int:
        """Get current effective FPS.
        
        Returns:
            Current effective framerate
        """
        # Effective FPS = base_fps / frame_multiplier
        effective_fps = self.base_fps // self.frame_multiplier
        
        # Clamp to allowed values
        closest_allowed = min(
            self.allowed_fps,
            key=lambda x: abs(x - effective_fps)
        )
        
        return closest_allowed
    
    def _set_current_fps(self, fps: int) -> None:
        """Internal method to set current FPS.
        
        Args:
            fps: New FPS level
        """
        # Calculate new frame multiplier
        self.frame_multiplier = max(1, self.base_fps // fps)
        
        # Record the change
        self._current_fps = fps
    
    def set_multiplier(self, multiplier: int) -> bool:
        """Set frame multiplier.
        
        Args:
            multiplier: Frame multiplier (1, 2, 4, 8, 16, 32, 64)
            
        Returns:
            True if change succeeded
        """
        if multiplier not in [1, 2, 4, 8, 16, 32, 64]:
            logger.warning(
                "Invalid frame multiplier %d (must be power of 2: 1,2,4,8,16,32,64)",
                multiplier
            )
            return False
        
        old_multiplier = self.frame_multiplier
        self.frame_multiplier = multiplier
        
        # Update effective FPS
        new_effective_fps = self.get_current_fps()
        
        self._record_adjustment(
            "multiplier_change",
            old_fps=self.get_effective_fps(old_multiplier),
            new_fps=new_effective_fps,
            frame_multiplier=multiplier
        )
        
        logger.info("Frame multiplier adjusted: %dx -> %dx", old_multiplier, multiplier)
        return True
    
    def get_effective_fps(self, multiplier: Optional[int] = None) -> int:
        """Get effective FPS for a given multiplier.
        
        Args:
            multiplier: Frame multiplier (uses current if None)
            
        Returns:
            Effective framerate
        """
        m = multiplier or self.frame_multiplier
        return max(1, self.base_fps // m)
    
    def zoom_out_temporally(self) -> bool:
        """Zoom out (lower FPS, see longer time span).
        
        Returns:
            True if adjustment succeeded
        """
        current_fps = self.get_current_fps()
        
        # Find next lower FPS level
        lower_fps_levels = [fps for fps in self.allowed_fps if fps < current_fps]
        
        if lower_fps_levels:
            target_fps = max(lower_fps_levels)
            return self.set_fps(target_fps)
        else:
            # Already at lowest FPS, try increasing multiplier
            if self.frame_multiplier < 64:
                new_multiplier = self.frame_multiplier * 2
                logger.info("Zooming out temporally: using 2x frame multiplier")
                return self.set_multiplier(new_multiplier)
            else:
                logger.info("Already at maximum temporal zoom")
                return False
    
    def zoom_in_temporally(self) -> bool:
        """Zoom in (higher FPS, see recent moments with more detail).
        
        Returns:
            True if adjustment succeeded
        """
        current_fps = self.get_current_fps()
        
        # Try increasing FPS first
        higher_fps_levels = [fps for fps in self.allowed_fps if fps > current_fps]
        
        if higher_fps_levels:
            target_fps = min(higher_fps_levels)
            return self.set_fps(target_fps)
        else:
            # Try decreasing multiplier
            if self.frame_multiplier > 1:
                new_multiplier = self.frame_multiplier // 2
                logger.info("Zooming in temporally: using 1/2x frame multiplier")
                return self.set_multiplier(new_multiplier)
            else:
                logger.info("Already at maximum temporal detail")
                return False
    
    def get_temporal_span_info(self) -> Dict[str, Any]:
        """Get information about current temporal span.
        
        Returns:
            Dictionary with temporal span information
        """
        effective_fps = self.get_current_fps()
        
        return {
            "effective_fps": effective_fps,
            "frame_interval_seconds": 1.0 / effective_fps,
            "seconds_per_frame": 1.0 / effective_fps,
            "frames_in_5_seconds": effective_fps * 5,
            "frames_in_1_minute": effective_fps * 60,
            "temporal_resolution": "high" if effective_fps >= 10 else "low",
        }
    
    def _record_adjustment(
        self,
        adjustment_type: str,
        **kwargs
    ) -> None:
        """Record adjustment in history.
        
        Args:
            adjustment_type: Type of adjustment
            **kwargs: Additional adjustment data
        """
        import time
        
        entry = {
            "timestamp": time.time(),
            "type": adjustment_type,
            "base_fps": self.base_fps,
            "frame_multiplier": self.frame_multiplier,
            "effective_fps": self.get_current_fps(),
            **kwargs
        }
        
        self.adjustment_history.append(entry)
        
        # Keep only last 100 adjustments
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of FPS adjustments.
        
        Returns:
            Dictionary with adjustment statistics
        """
        if not self.adjustment_history:
            return {"total_adjustments": 0}
        
        return {
            "total_adjustments": len(self.adjustment_history),
            "current_effective_fps": self.get_current_fps(),
            "current_frame_multiplier": self.frame_multiplier,
            "last_adjustment": self.adjustment_history[-1] if self.adjustment_history else None,
            "adjustment_types": list(set(adj["type"] for adj in self.adjustment_history)),
        }
    
    def reset_to_default(self) -> None:
        """Reset FPS and multiplier to default values."""
        old_fps = self.get_current_fps()
        old_multiplier = self.frame_multiplier
        
        self.frame_multiplier = 4  # Default multiplier
        self._current_fps = self.base_fps // self.frame_multiplier
        
        self._record_adjustment(
            "reset_to_default",
            old_fps=old_fps,
            new_fps=self._current_fps,
            frame_multiplier=self.frame_multiplier,
            old_multiplier=old_multiplier
        )
        
        logger.info("Reset FPS to default: %dfps @ %dx multiplier", self._current_fps, self.frame_multiplier)
