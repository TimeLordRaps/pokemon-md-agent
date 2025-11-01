"""Analyze screenshots and vision pipeline outputs to understand what models see.

This module provides tools to inspect:
- Sprite detection results and confidence scores
- ASCII grid parsing accuracy
- Image preprocessing quality
- Model input format and composition
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import hashlib

from src.utils.logging_setup import get_logger


@dataclass
class SpriteDetectionResult:
    """Result of sprite detection analysis."""

    timestamp: str
    screenshot_hash: str
    sprites_detected: int
    confidence_scores: List[float] = field(default_factory=list)
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "screenshot_hash": self.screenshot_hash,
            "sprites_detected": self.sprites_detected,
            "confidence_scores": self.confidence_scores,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
        }


@dataclass
class ASCIIGridResult:
    """Result of ASCII grid parsing analysis."""

    timestamp: str
    grid_width: int
    grid_height: int
    grid_density: float  # Ratio of non-empty cells
    parsed_entities: Dict[str, int]  # Entity type -> count
    grid_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "grid_density": self.grid_density,
            "parsed_entities": self.parsed_entities,
            "grid_hash": self.grid_hash,
        }


@dataclass
class ScreenshotAnalysisFrame:
    """Complete analysis of a single screenshot frame."""

    frame_number: int
    timestamp: str
    screenshot_path: Optional[str]
    sprite_detection: Optional[SpriteDetectionResult] = None
    ascii_grid: Optional[ASCIIGridResult] = None
    preprocessing_quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "screenshot_path": self.screenshot_path,
            "sprite_detection": self.sprite_detection.to_dict() if self.sprite_detection else None,
            "ascii_grid": self.ascii_grid.to_dict() if self.ascii_grid else None,
            "preprocessing_quality": self.preprocessing_quality,
        }


class ScreenshotAnalyzer:
    """Analyze screenshots to understand vision pipeline outputs.

    This analyzer examines:
    - Individual sprite detection results
    - ASCII grid parsing accuracy
    - Image preprocessing pipeline quality
    - Cross-frame consistency
    - Input format composition (4-panel layout)
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize screenshot analyzer.

        Args:
            log_dir: Optional directory to save analysis results
        """
        self.logger = get_logger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.frames: List[ScreenshotAnalysisFrame] = []
        self.sprite_detection_stats: Dict[str, Any] = {}
        self.ascii_grid_stats: Dict[str, Any] = {}

    def add_sprite_detection(
        self,
        frame_number: int,
        screenshot_hash: str,
        sprites_detected: int,
        confidence_scores: List[float],
    ) -> None:
        """Record sprite detection results for a frame.

        Args:
            frame_number: Sequential frame number
            screenshot_hash: Hash of the screenshot image
            sprites_detected: Number of sprites found
            confidence_scores: Confidence score for each detected sprite
        """
        timestamp = datetime.now().isoformat()

        mean_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        min_conf = min(confidence_scores) if confidence_scores else 0.0
        max_conf = max(confidence_scores) if confidence_scores else 0.0

        result = SpriteDetectionResult(
            timestamp=timestamp,
            screenshot_hash=screenshot_hash,
            sprites_detected=sprites_detected,
            confidence_scores=confidence_scores,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
        )

        # Create or update frame analysis
        frame_key = frame_number
        existing_frame = None
        for frame in self.frames:
            if frame.frame_number == frame_key:
                existing_frame = frame
                break

        if existing_frame:
            existing_frame.sprite_detection = result
        else:
            frame = ScreenshotAnalysisFrame(
                frame_number=frame_key,
                timestamp=timestamp,
                screenshot_path=None,
                sprite_detection=result,
            )
            self.frames.append(frame)

        self.logger.info(
            f"Sprite detection: frame={frame_number}, detected={sprites_detected}, "
            f"mean_confidence={mean_conf:.3f}",
            extra={
                "frame_number": frame_number,
                "sprites_detected": sprites_detected,
                "confidence_scores": confidence_scores,
                "mean_confidence": mean_conf,
            }
        )

    def add_ascii_grid(
        self,
        frame_number: int,
        grid_width: int,
        grid_height: int,
        grid_density: float,
        parsed_entities: Dict[str, int],
    ) -> None:
        """Record ASCII grid parsing results for a frame.

        Args:
            frame_number: Sequential frame number
            grid_width: Width of parsed grid
            grid_height: Height of parsed grid
            grid_density: Ratio of non-empty cells (0.0-1.0)
            parsed_entities: Dictionary of entity types and counts
        """
        timestamp = datetime.now().isoformat()

        # Create hash of grid content
        grid_hash = hashlib.md5(
            json.dumps(parsed_entities, sort_keys=True).encode()
        ).hexdigest()

        result = ASCIIGridResult(
            timestamp=timestamp,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_density=grid_density,
            parsed_entities=parsed_entities,
            grid_hash=grid_hash,
        )

        # Create or update frame analysis
        frame_key = frame_number
        existing_frame = None
        for frame in self.frames:
            if frame.frame_number == frame_key:
                existing_frame = frame
                break

        if existing_frame:
            existing_frame.ascii_grid = result
        else:
            frame = ScreenshotAnalysisFrame(
                frame_number=frame_key,
                timestamp=timestamp,
                screenshot_path=None,
                ascii_grid=result,
            )
            self.frames.append(frame)

        self.logger.info(
            f"ASCII grid: frame={frame_number}, grid={grid_width}x{grid_height}, "
            f"density={grid_density:.2f}, entities={parsed_entities}",
            extra={
                "frame_number": frame_number,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "grid_density": grid_density,
                "parsed_entities": parsed_entities,
            }
        )

    def add_preprocessing_quality(
        self,
        frame_number: int,
        quality_metrics: Dict[str, Any],
    ) -> None:
        """Record image preprocessing quality metrics.

        Args:
            frame_number: Sequential frame number
            quality_metrics: Dictionary of quality measurements (contrast, brightness, etc.)
        """
        # Find or create frame
        existing_frame = None
        for frame in self.frames:
            if frame.frame_number == frame_number:
                existing_frame = frame
                break

        if existing_frame:
            existing_frame.preprocessing_quality = quality_metrics
        else:
            frame = ScreenshotAnalysisFrame(
                frame_number=frame_number,
                timestamp=datetime.now().isoformat(),
                screenshot_path=None,
                preprocessing_quality=quality_metrics,
            )
            self.frames.append(frame)

        self.logger.info(
            f"Preprocessing quality: frame={frame_number}, metrics={quality_metrics}",
            extra={
                "frame_number": frame_number,
                **quality_metrics,
            }
        )

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all analyzed frames.

        Returns:
            Dictionary containing sprite detection and ASCII grid statistics
        """
        sprite_stats = self._compute_sprite_stats()
        grid_stats = self._compute_grid_stats()

        stats = {
            "total_frames_analyzed": len(self.frames),
            "sprite_detection": sprite_stats,
            "ascii_grid": grid_stats,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Analysis statistics: {len(self.frames)} frames analyzed",
            extra=stats,
        )

        return stats

    def _compute_sprite_stats(self) -> Dict[str, Any]:
        """Compute sprite detection statistics."""
        sprite_frames = [f for f in self.frames if f.sprite_detection]

        if not sprite_frames:
            return {"total_frames": 0}

        total_sprites = sum(f.sprite_detection.sprites_detected for f in sprite_frames)
        mean_sprites = total_sprites / len(sprite_frames)

        all_confidences = []
        for frame in sprite_frames:
            if frame.sprite_detection.confidence_scores:
                all_confidences.extend(frame.sprite_detection.confidence_scores)

        mean_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return {
            "total_frames": len(sprite_frames),
            "total_sprites_detected": total_sprites,
            "mean_sprites_per_frame": mean_sprites,
            "mean_confidence": mean_confidence,
            "confidence_scores_count": len(all_confidences),
        }

    def _compute_grid_stats(self) -> Dict[str, Any]:
        """Compute ASCII grid statistics."""
        grid_frames = [f for f in self.frames if f.ascii_grid]

        if not grid_frames:
            return {"total_frames": 0}

        mean_density = sum(f.ascii_grid.grid_density for f in grid_frames) / len(grid_frames)

        # Aggregate entity counts
        entity_totals = {}
        for frame in grid_frames:
            for entity_type, count in frame.ascii_grid.parsed_entities.items():
                entity_totals[entity_type] = entity_totals.get(entity_type, 0) + count

        return {
            "total_frames": len(grid_frames),
            "mean_grid_density": mean_density,
            "aggregated_entities": entity_totals,
        }

    def save_analysis(self, filename: str = "screenshot_analysis.jsonl") -> Path:
        """Save analysis results to file.

        Args:
            filename: Output filename (in log_dir)

        Returns:
            Path to saved analysis file
        """
        output_path = self.log_dir / filename

        with open(output_path, 'w') as f:
            for frame in self.frames:
                f.write(json.dumps(frame.to_dict()) + '\n')

        # Also save summary statistics
        stats = self.compute_statistics()
        stats_path = self.log_dir / "screenshot_analysis_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            f"Saved screenshot analysis: {output_path}",
            extra={"output_path": str(output_path), "frames": len(self.frames)},
        )

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analyzed screenshots.

        Returns:
            Dictionary with key metrics and statistics
        """
        return {
            "total_frames_analyzed": len(self.frames),
            "sprite_detection_stats": self._compute_sprite_stats(),
            "ascii_grid_stats": self._compute_grid_stats(),
            "timestamp": datetime.now().isoformat(),
        }
