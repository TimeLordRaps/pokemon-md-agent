"""FPS adjustment and performance monitoring for vision pipeline.

Provides timing utilities and performance hooks for vision processing components.
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for vision operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self) -> None:
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "operation": self.operation_name,
            "start_time": self.start_time,
            "metadata": self.metadata
        }
        if self.end_time is not None:
            result["end_time"] = self.end_time
            result["duration_ms"] = self.duration_ms
        return result


@contextmanager
def timed(operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[PerformanceMetrics, None, None]:
    """Context manager for timing operations with metadata emission.

    Args:
        operation_name: Name of the operation being timed
        metadata: Optional metadata to include with timing data

    Yields:
        PerformanceMetrics object for the operation

    Example:
        with timed("sprite_detection", {"image_size": "240x160"}) as metrics:
            # Do work here
            detections = detector.detect(image_path)
            metrics.metadata["detection_count"] = len(detections)
        # metrics.duration_ms is now available
    """
    metrics = PerformanceMetrics(
        operation_name=operation_name,
        start_time=time.time(),
        metadata=metadata or {}
    )

    try:
        yield metrics
    finally:
        metrics.complete()

        # Log performance data
        logger.debug(
            "Operation '%s' completed in %.2f ms",
            operation_name,
            metrics.duration_ms
        )

        # Emit metadata for monitoring systems
        _emit_performance_metadata(metrics)


def _emit_performance_metadata(metrics: PerformanceMetrics) -> None:
    """Emit performance metadata for monitoring.

    Args:
        metrics: Completed performance metrics
    """
    # In a real implementation, this might send to a monitoring system
    # For now, just log at info level for operations over 100ms
    if metrics.duration_ms and metrics.duration_ms > 100:
        logger.info(
            "Slow operation detected: '%s' took %.2f ms",
            metrics.operation_name,
            metrics.duration_ms
        )


class FPSAdjuster:
    """Adjusts processing based on performance metrics and target FPS."""

    def __init__(self, target_fps: float = 30.0, adaptation_window: int = 10):
        """Initialize FPS adjuster.

        Args:
            target_fps: Target frames per second
            adaptation_window: Number of frames to average for adaptation
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.adaptation_window = adaptation_window

        # Performance tracking
        self.recent_frame_times: list[float] = []
        self.last_frame_time = time.time()

        # Adaptation state
        self.skip_next_frame = False
        self.processing_scale = 1.0  # Scale factor for processing intensity

    def start_frame(self) -> float:
        """Mark the start of a frame and return target processing time.

        Returns:
            Target processing time in seconds for this frame
        """
        current_time = time.time()
        self.last_frame_time = current_time

        # Calculate adaptive target based on recent performance
        if len(self.recent_frame_times) >= self.adaptation_window:
            avg_frame_time = sum(self.recent_frame_times[-self.adaptation_window:]) / self.adaptation_window
            adaptive_target = max(self.target_frame_time, avg_frame_time * 0.9)  # 90% of recent average
        else:
            adaptive_target = self.target_frame_time

        return adaptive_target * self.processing_scale

    def end_frame(self, actual_processing_time: float) -> None:
        """Mark the end of a frame with actual processing time.

        Args:
            actual_processing_time: Time spent processing this frame in seconds
        """
        self.recent_frame_times.append(actual_processing_time)

        # Keep only recent frames
        if len(self.recent_frame_times) > self.adaptation_window * 2:
            self.recent_frame_times = self.recent_frame_times[-self.adaptation_window:]

        # Adapt processing scale based on performance
        if len(self.recent_frame_times) >= self.adaptation_window:
            avg_time = sum(self.recent_frame_times[-self.adaptation_window:]) / self.adaptation_window

            if avg_time > self.target_frame_time * 1.2:  # 20% over target
                # Reduce processing intensity
                self.processing_scale = max(0.5, self.processing_scale * 0.95)
                logger.debug("Reduced processing scale to %.2f due to slow performance", self.processing_scale)
            elif avg_time < self.target_frame_time * 0.8:  # 20% under target
                # Increase processing intensity
                self.processing_scale = min(2.0, self.processing_scale * 1.05)
                logger.debug("Increased processing scale to %.2f due to fast performance", self.processing_scale)

    def should_skip_frame(self) -> bool:
        """Check if the next frame should be skipped for FPS control.

        Returns:
            True if frame should be skipped
        """
        if self.skip_next_frame:
            self.skip_next_frame = False
            return True

        # Check if we're falling behind
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time

        if time_since_last > self.target_frame_time * 2:
            # We're falling behind, skip a frame
            self.skip_next_frame = True
            logger.debug("Skipping frame to catch up with target FPS")
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics.

        Returns:
            Dictionary with performance stats
        """
        if not self.recent_frame_times:
            return {"frames_processed": 0, "avg_frame_time": 0, "current_fps": 0}

        avg_frame_time = sum(self.recent_frame_times) / len(self.recent_frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        return {
            "frames_processed": len(self.recent_frame_times),
            "avg_frame_time_ms": avg_frame_time * 1000,
            "current_fps": current_fps,
            "target_fps": self.target_fps,
            "processing_scale": self.processing_scale,
            "adaptation_window": self.adaptation_window
        }


# Global performance monitoring
_performance_monitor: Optional[FPSAdjuster] = None

def get_performance_monitor() -> FPSAdjuster:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = FPSAdjuster()
    return _performance_monitor


def reset_performance_monitor(target_fps: float = 30.0) -> None:
    """Reset the global performance monitor with new target FPS.

    Args:
        target_fps: New target frames per second
    """
    global _performance_monitor
    _performance_monitor = FPSAdjuster(target_fps=target_fps)