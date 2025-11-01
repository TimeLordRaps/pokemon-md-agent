"""Enhanced logging infrastructure for feedback loop instrumentation.

Provides:
- Structured logging with context propagation
- Performance monitoring instrumentation
- Automatic feedback analyzer integration
- Run session tracking
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import time

from src.utils.logging_setup import get_logger
from src.feedback.screenshot_analyzer import ScreenshotAnalyzer
from src.feedback.ensemble_analyzer import EnsembleConsensusAnalyzer
from src.feedback.context_analyzer import ContextWindowAnalyzer
from src.feedback.skill_analyzer import SkillFormationAnalyzer
from src.feedback.metrics_analyzer import PerformanceMetricsAnalyzer


class EnhancedLogger:
    """Enhanced logger with automatic feedback analyzer integration."""

    def __init__(
        self,
        logger_name: str,
        log_dir: Optional[Path] = None,
        enable_feedback_logging: bool = True,
    ):
        """Initialize enhanced logger.

        Args:
            logger_name: Name for the logger
            log_dir: Directory for log files
            enable_feedback_logging: Whether to integrate feedback analyzers
        """
        self.logger = get_logger(logger_name)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_feedback_logging = enable_feedback_logging

        # Initialize feedback analyzers
        self.screenshot_analyzer = ScreenshotAnalyzer(self.log_dir) if enable_feedback_logging else None
        self.ensemble_analyzer = EnsembleConsensusAnalyzer(self.log_dir) if enable_feedback_logging else None
        self.context_analyzer = ContextWindowAnalyzer(self.log_dir) if enable_feedback_logging else None
        self.skill_analyzer = SkillFormationAnalyzer(self.log_dir) if enable_feedback_logging else None
        self.metrics_analyzer = PerformanceMetricsAnalyzer(self.log_dir) if enable_feedback_logging else None

        # Session tracking
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.frame_count = 0
        self.event_count = 0

    def log_sprite_detection(
        self,
        frame_number: int,
        screenshot_hash: str,
        sprites_detected: int,
        confidence_scores: List[float],
    ) -> None:
        """Log sprite detection event."""
        if self.screenshot_analyzer:
            self.screenshot_analyzer.add_sprite_detection(
                frame_number=frame_number,
                screenshot_hash=screenshot_hash,
                sprites_detected=sprites_detected,
                confidence_scores=confidence_scores,
            )
        self.event_count += 1

    def log_ascii_grid(
        self,
        frame_number: int,
        grid_width: int,
        grid_height: int,
        grid_density: float,
        parsed_entities: Dict[str, int],
    ) -> None:
        """Log ASCII grid parsing event."""
        if self.screenshot_analyzer:
            self.screenshot_analyzer.add_ascii_grid(
                frame_number=frame_number,
                grid_width=grid_width,
                grid_height=grid_height,
                grid_density=grid_density,
                parsed_entities=parsed_entities,
            )
        self.event_count += 1

    def log_model_output(
        self,
        frame_number: int,
        model_name: str,
        model_variant: str,
        prediction: str,
        confidence: float,
        reasoning: Optional[str] = None,
        attention_distribution: Optional[Dict[str, float]] = None,
        tokens_used: int = 0,
    ) -> None:
        """Log model output event."""
        if self.ensemble_analyzer:
            self.ensemble_analyzer.add_model_output(
                frame_number=frame_number,
                model_name=model_name,
                model_variant=model_variant,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                attention_distribution=attention_distribution,
                tokens_used=tokens_used,
            )
        self.event_count += 1

    def log_consensus_decision(
        self,
        frame_number: int,
        winning_action: str,
    ) -> None:
        """Log ensemble consensus decision."""
        if self.ensemble_analyzer:
            self.ensemble_analyzer.compute_consensus(
                frame_number=frame_number,
                winning_action=winning_action,
            )
        self.event_count += 1

    def log_context_snapshot(
        self,
        frame_number: int,
        context_length: int,
        memory_silos_used: int,
        silo_distribution: Dict[str, int],
        game_state_tokens: int,
        retrieval_tokens: int,
        instruction_tokens: int,
        entities_in_context: Optional[List[str]] = None,
    ) -> None:
        """Log context window snapshot."""
        if self.context_analyzer:
            self.context_analyzer.add_context_snapshot(
                frame_number=frame_number,
                context_length=context_length,
                memory_silos_used=memory_silos_used,
                silo_distribution=silo_distribution,
                game_state_tokens=game_state_tokens,
                retrieval_tokens=retrieval_tokens,
                instruction_tokens=instruction_tokens,
                entities_in_context=entities_in_context,
            )
        self.event_count += 1

    def log_skill_discovery(
        self,
        skill_id: str,
        skill_name: str,
        actions_sequence: List[str],
        frames_range: tuple,
        context_tags: Optional[List[str]] = None,
        related_entities: Optional[List[str]] = None,
    ) -> None:
        """Log skill discovery event."""
        if self.skill_analyzer:
            self.skill_analyzer.add_skill_discovery(
                skill_id=skill_id,
                skill_name=skill_name,
                actions_sequence=actions_sequence,
                frames_range=frames_range,
                context_tags=context_tags,
                related_entities=related_entities,
            )
        self.event_count += 1

    def log_skill_execution(
        self,
        frame_number: int,
        skill_id: str,
        skill_name: str,
        success: bool,
        predicted_outcome: str,
        actual_outcome: str,
        reward: Optional[float] = None,
    ) -> None:
        """Log skill execution event."""
        if self.skill_analyzer:
            self.skill_analyzer.add_skill_execution(
                frame_number=frame_number,
                skill_id=skill_id,
                skill_name=skill_name,
                success=success,
                predicted_outcome=predicted_outcome,
                actual_outcome=actual_outcome,
                reward=reward,
            )
        self.event_count += 1

    def log_frame_metrics(
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
        """Log game frame metrics."""
        if self.metrics_analyzer:
            self.metrics_analyzer.add_frame_metrics(
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
                status_effects=status_effects,
                enemies_defeated=enemies_defeated,
                traps_avoided=traps_avoided,
                moves_made=moves_made,
            )
        self.frame_count += 1
        self.event_count += 1

    def log_run_summary(
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
        """Log run summary."""
        if self.metrics_analyzer:
            self.metrics_analyzer.add_run_summary(
                run_id=run_id,
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

        self.logger.info(
            f"Run summary: {run_id}, status={final_status}, "
            f"duration={duration_frames}, floor={max_floor_reached}",
            extra={
                "run_id": run_id,
                "final_status": final_status,
                "duration_frames": duration_frames,
                "max_floor_reached": max_floor_reached,
                "failure_reason": failure_reason,
            }
        )

        self.event_count += 1

    @contextmanager
    def log_performance(self, operation_name: str):
        """Context manager for logging operation performance.

        Usage:
            with logger.log_performance("model_inference"):
                # do work
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        self.logger.info(f"Starting: {operation_name}")

        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory

            self.logger.info(
                f"Completed: {operation_name}",
                extra={
                    "operation": operation_name,
                    "elapsed_time_seconds": elapsed_time,
                    "memory_delta_mb": memory_delta,
                }
            )

    def save_all_analyses(self) -> Dict[str, Path]:
        """Save all feedback analyses to disk.

        Returns:
            Dictionary mapping analyzer name to output path
        """
        output_paths = {}

        if self.screenshot_analyzer:
            output_paths["screenshot"] = self.screenshot_analyzer.save_analysis()

        if self.ensemble_analyzer:
            output_paths["ensemble"] = self.ensemble_analyzer.save_analysis()

        if self.context_analyzer:
            output_paths["context"] = self.context_analyzer.save_analysis()

        if self.skill_analyzer:
            output_paths["skill"] = self.skill_analyzer.save_analysis()

        if self.metrics_analyzer:
            output_paths["metrics"] = self.metrics_analyzer.save_analysis()

        self.logger.info(
            f"Saved all analyses: {list(output_paths.keys())}",
            extra={
                "analyses": list(output_paths.keys()),
                "paths": {k: str(v) for k, v in output_paths.items()},
            }
        )

        return output_paths

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.

        Returns:
            Dictionary with session metrics
        """
        elapsed = (datetime.now() - self.session_start).total_seconds()

        summary = {
            "session_id": self.session_id,
            "elapsed_seconds": elapsed,
            "frame_count": self.frame_count,
            "event_count": self.event_count,
            "screenshot_frames": len(self.screenshot_analyzer.frames) if self.screenshot_analyzer else 0,
            "consensus_decisions": len(self.ensemble_analyzer.consensus_decisions) if self.ensemble_analyzer else 0,
            "context_snapshots": len(self.context_analyzer.snapshots) if self.context_analyzer else 0,
            "skills_discovered": len(self.skill_analyzer.skill_discoveries) if self.skill_analyzer else 0,
            "run_summaries": len(self.metrics_analyzer.run_summaries) if self.metrics_analyzer else 0,
        }

        self.logger.info(
            f"Session summary: {self.session_id}",
            extra=summary,
        )

        return summary

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


# Global enhanced logger instance
_enhanced_logger: Optional[EnhancedLogger] = None


def get_enhanced_logger(
    logger_name: str = "feedback_loop",
    log_dir: Optional[Path] = None,
    enable_feedback_logging: bool = True,
) -> EnhancedLogger:
    """Get or create the global enhanced logger.

    Args:
        logger_name: Name for the logger
        log_dir: Directory for log files
        enable_feedback_logging: Whether to integrate feedback analyzers

    Returns:
        Enhanced logger instance
    """
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(
            logger_name=logger_name,
            log_dir=log_dir,
            enable_feedback_logging=enable_feedback_logging,
        )
    return _enhanced_logger


def reset_enhanced_logger() -> None:
    """Reset the global enhanced logger."""
    global _enhanced_logger
    _enhanced_logger = None
