"""Analyze performance metrics and run comparison.

This module provides tools to understand:
- Game performance metrics (progress, rewards, deaths)
- Run-to-run comparison
- Performance trends across iterations
- Bottleneck identification
- Optimization opportunities
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from collections import defaultdict

from src.utils.logging_setup import get_logger


@dataclass
class GameMetrics:
    """Game performance metrics for a frame or run."""

    timestamp: str
    frame_number: Optional[int] = None
    run_id: Optional[str] = None
    dungeon_floor: int = 0
    player_hp: int = 0
    max_hp: int = 0
    player_level: int = 0
    experience_points: int = 0
    inventory_items: int = 0
    money: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    status_effects: List[str] = field(default_factory=list)
    enemies_defeated: int = 0
    traps_avoided: int = 0
    moves_made: int = 0
    survival_duration: float = 0.0  # Frames survived

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "run_id": self.run_id,
            "dungeon_floor": self.dungeon_floor,
            "player_hp": self.player_hp,
            "max_hp": self.max_hp,
            "player_level": self.player_level,
            "experience_points": self.experience_points,
            "inventory_items": self.inventory_items,
            "money": self.money,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "status_effects": self.status_effects,
            "enemies_defeated": self.enemies_defeated,
            "traps_avoided": self.traps_avoided,
            "moves_made": self.moves_made,
            "survival_duration": self.survival_duration,
        }


@dataclass
class RunSummary:
    """Summary statistics for an entire run."""

    run_id: str
    timestamp: str
    duration_frames: int
    max_floor_reached: int
    final_level: int
    total_experience: int
    enemies_defeated: int
    traps_avoided: int
    total_damage_dealt: int
    total_damage_taken: int
    final_status: str  # "success", "death", "timeout", etc.
    failure_reason: Optional[str] = None
    average_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "duration_frames": self.duration_frames,
            "max_floor_reached": self.max_floor_reached,
            "final_level": self.final_level,
            "total_experience": self.total_experience,
            "enemies_defeated": self.enemies_defeated,
            "traps_avoided": self.traps_avoided,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "final_status": self.final_status,
            "failure_reason": self.failure_reason,
            "average_fps": self.average_fps,
        }


class PerformanceMetricsAnalyzer:
    """Analyze game performance metrics and run comparison.

    Tracks:
    - Per-frame game state metrics
    - Run summaries and comparisons
    - Performance trends
    - Failure pattern analysis
    - Progress metrics
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize metrics analyzer.

        Args:
            log_dir: Optional directory to save analysis results
        """
        self.logger = get_logger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.frame_metrics: Dict[int, GameMetrics] = {}
        self.run_summaries: Dict[str, RunSummary] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)

    def add_frame_metrics(
        self,
        frame_number: int,
        dungeon_floor: int,
        player_hp: int,
        max_hp: int,
        player_level: int,
        experience_points: int,
        inventory_items: int,
        money: int,
        total_damage_dealt: int,
        total_damage_taken: int,
        status_effects: Optional[List[str]] = None,
        enemies_defeated: int = 0,
        traps_avoided: int = 0,
        moves_made: int = 0,
    ) -> None:
        """Record game metrics for a frame.

        Args:
            frame_number: Sequential frame number
            dungeon_floor: Current dungeon floor
            player_hp: Current player HP
            max_hp: Maximum player HP
            player_level: Current player level
            experience_points: Total experience accumulated
            inventory_items: Number of items in inventory
            money: Current money/gold
            total_damage_dealt: Total damage dealt (cumulative)
            total_damage_taken: Total damage taken (cumulative)
            status_effects: List of active status effects
            enemies_defeated: Number of enemies defeated
            traps_avoided: Number of traps avoided
            moves_made: Number of moves made
        """
        timestamp = datetime.now().isoformat()

        metrics = GameMetrics(
            timestamp=timestamp,
            frame_number=frame_number,
            dungeon_floor=dungeon_floor,
            player_hp=player_hp,
            max_hp=max_hp,
            player_level=player_level,
            experience_points=experience_points,
            inventory_items=inventory_items,
            money=money,
            total_damage_dealt=total_damage_dealt,
            total_damage_taken=total_damage_taken,
            status_effects=status_effects or [],
            enemies_defeated=enemies_defeated,
            traps_avoided=traps_avoided,
            moves_made=moves_made,
        )

        self.frame_metrics[frame_number] = metrics

        self.logger.info(
            f"Frame metrics: frame={frame_number}, floor={dungeon_floor}, "
            f"hp={player_hp}/{max_hp}, level={player_level}",
            extra={
                "frame_number": frame_number,
                "dungeon_floor": dungeon_floor,
                "player_hp": player_hp,
                "max_hp": max_hp,
                "player_level": player_level,
                "experience_points": experience_points,
            }
        )

    def add_run_summary(
        self,
        run_id: str,
        duration_frames: int,
        max_floor_reached: int,
        final_level: int,
        total_experience: int,
        enemies_defeated: int,
        traps_avoided: int,
        total_damage_dealt: int,
        total_damage_taken: int,
        final_status: str,
        failure_reason: Optional[str] = None,
        average_fps: float = 0.0,
    ) -> None:
        """Record summary for an entire run.

        Args:
            run_id: Unique run identifier
            duration_frames: Total frames in run
            max_floor_reached: Highest floor reached
            final_level: Final player level
            total_experience: Total experience gained
            enemies_defeated: Total enemies defeated
            traps_avoided: Total traps avoided
            total_damage_dealt: Total damage dealt
            total_damage_taken: Total damage taken
            final_status: Status at end of run
            failure_reason: Reason for failure (if applicable)
            average_fps: Average frames per second
        """
        timestamp = datetime.now().isoformat()

        summary = RunSummary(
            run_id=run_id,
            timestamp=timestamp,
            duration_frames=duration_frames,
            max_floor_reached=max_floor_reached,
            final_level=final_level,
            total_experience=total_experience,
            enemies_defeated=enemies_defeated,
            traps_avoided=traps_avoided,
            total_damage_dealt=total_damage_dealt,
            total_damage_taken=total_damage_taken,
            final_status=final_status,
            failure_reason=failure_reason,
            average_fps=average_fps,
        )

        self.run_summaries[run_id] = summary

        if failure_reason:
            self.failure_counts[failure_reason] += 1

        self.logger.info(
            f"Run summary: {run_id}, status={final_status}, "
            f"duration={duration_frames} frames, floor={max_floor_reached}",
            extra={
                "run_id": run_id,
                "final_status": final_status,
                "duration_frames": duration_frames,
                "max_floor_reached": max_floor_reached,
                "failure_reason": failure_reason,
            }
        )

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all runs.

        Returns:
            Dictionary containing performance statistics
        """
        if not self.run_summaries:
            return {"total_runs": 0}

        run_summaries = list(self.run_summaries.values())

        # Aggregate metrics
        total_runs = len(run_summaries)
        avg_duration = sum(r.duration_frames for r in run_summaries) / total_runs
        max_floor_reached = max(r.max_floor_reached for r in run_summaries)
        avg_level = sum(r.final_level for r in run_summaries) / total_runs
        avg_enemies_defeated = sum(r.enemies_defeated for r in run_summaries) / total_runs

        # Success rate
        successful_runs = sum(1 for r in run_summaries if r.final_status == "success")
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0

        # Failure analysis
        failure_counts = dict(self.failure_counts)
        top_failures = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        stats = {
            "total_runs": total_runs,
            "average_duration_frames": avg_duration,
            "max_floor_reached": max_floor_reached,
            "average_final_level": avg_level,
            "average_enemies_defeated": avg_enemies_defeated,
            "success_rate": success_rate,
            "successful_runs": successful_runs,
            "top_failure_reasons": dict(top_failures),
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Performance statistics: {total_runs} runs, "
            f"success_rate={success_rate:.2f}, avg_duration={avg_duration:.0f}",
            extra=stats,
        )

        return stats

    def compute_run_comparison(
        self,
        run_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare performance across specific runs.

        Args:
            run_ids: List of run IDs to compare (or all if None)

        Returns:
            Dictionary containing comparison data
        """
        if run_ids is None:
            summaries = list(self.run_summaries.values())
        else:
            summaries = [
                self.run_summaries[rid] for rid in run_ids
                if rid in self.run_summaries
            ]

        if not summaries:
            return {"error": "No runs to compare"}

        comparison = {
            "comparison_runs": len(summaries),
            "runs": {},
        }

        for summary in summaries:
            comparison["runs"][summary.run_id] = {
                "final_status": summary.final_status,
                "duration_frames": summary.duration_frames,
                "max_floor": summary.max_floor_reached,
                "final_level": summary.final_level,
                "enemies_defeated": summary.enemies_defeated,
                "damage_dealt": summary.total_damage_dealt,
                "damage_taken": summary.total_damage_taken,
            }

        # Compute deltas
        if len(summaries) > 1:
            sorted_by_duration = sorted(summaries, key=lambda r: r.duration_frames)
            comparison["best_run"] = sorted_by_duration[-1].run_id
            comparison["worst_run"] = sorted_by_duration[0].run_id

        return comparison

    def save_analysis(self, filename: str = "metrics_analysis.jsonl") -> Path:
        """Save analysis results to file.

        Args:
            filename: Output filename (in log_dir)

        Returns:
            Path to saved analysis file
        """
        output_path = self.log_dir / filename

        # Save frame metrics
        with open(output_path, 'w') as f:
            for frame_number in sorted(self.frame_metrics.keys()):
                metrics = self.frame_metrics[frame_number]
                f.write(json.dumps(metrics.to_dict()) + '\n')

        # Save run summaries
        if self.run_summaries:
            summaries_path = self.log_dir / "run_summaries.jsonl"
            with open(summaries_path, 'w') as f:
                for run_id in sorted(self.run_summaries.keys()):
                    summary = self.run_summaries[run_id]
                    f.write(json.dumps(summary.to_dict()) + '\n')

        # Save summary statistics
        stats = self.compute_statistics()
        stats_path = self.log_dir / "metrics_analysis_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            f"Saved metrics analysis: {output_path}",
            extra={
                "output_path": str(output_path),
                "frames": len(self.frame_metrics),
                "runs": len(self.run_summaries),
            }
        )

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics analysis.

        Returns:
            Dictionary with key metrics and statistics
        """
        return {
            "total_frame_records": len(self.frame_metrics),
            "total_runs": len(self.run_summaries),
            "statistics": self.compute_statistics(),
            "timestamp": datetime.now().isoformat(),
        }
