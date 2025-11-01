"""Analyze skill discovery and formation across runs.

This module provides tools to understand:
- Skill formation patterns
- Action sequences that compose skills
- Skill reusability across runs
- Skill effectiveness metrics
- Skill inheritance and evolution
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
from datetime import datetime
from collections import defaultdict, Counter

from src.utils.logging_setup import get_logger


@dataclass
class SkillDiscovery:
    """Record of a skill being discovered/formed."""

    timestamp: str
    skill_id: str
    skill_name: str
    actions_sequence: List[str]  # Sequence of actions that form the skill
    success_rate: float  # Fraction of successful uses
    usage_count: int
    total_attempts: int
    frames_range: tuple  # (start_frame, end_frame)
    context_tags: List[str]  # Game state context where skill was useful
    related_entities: List[str]  # Game entities involved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "actions_sequence": self.actions_sequence,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "total_attempts": self.total_attempts,
            "frames_range": list(self.frames_range),
            "context_tags": self.context_tags,
            "related_entities": self.related_entities,
        }


@dataclass
class SkillExecution:
    """Record of skill execution."""

    timestamp: str
    frame_number: int
    skill_id: str
    skill_name: str
    success: bool
    predicted_outcome: str
    actual_outcome: str
    predicted_vs_actual_match: bool
    reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "success": self.success,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "predicted_vs_actual_match": self.predicted_vs_actual_match,
            "reward": self.reward,
        }


class SkillFormationAnalyzer:
    """Analyze skill discovery and formation.

    Tracks:
    - New skills discovered per run
    - Skill effectiveness and success rates
    - Action patterns that form skills
    - Skill reusability across runs
    - Skill-context relationships
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize skill analyzer.

        Args:
            log_dir: Optional directory to save analysis results
        """
        self.logger = get_logger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.skill_discoveries: Dict[str, SkillDiscovery] = {}
        self.skill_executions: List[SkillExecution] = []
        self.skill_usage_count: Dict[str, int] = defaultdict(int)
        self.skill_success_count: Dict[str, int] = defaultdict(int)
        self.action_patterns: Dict[str, int] = defaultdict(int)  # Action sequence -> frequency

    def add_skill_discovery(
        self,
        skill_id: str,
        skill_name: str,
        actions_sequence: List[str],
        frames_range: tuple,
        context_tags: Optional[List[str]] = None,
        related_entities: Optional[List[str]] = None,
    ) -> None:
        """Record discovery of a new skill.

        Args:
            skill_id: Unique skill identifier
            skill_name: Human-readable skill name
            actions_sequence: Sequence of actions forming the skill
            frames_range: (start_frame, end_frame) tuple
            context_tags: Game state contexts where skill is useful
            related_entities: Entities involved in skill execution
        """
        timestamp = datetime.now().isoformat()

        discovery = SkillDiscovery(
            timestamp=timestamp,
            skill_id=skill_id,
            skill_name=skill_name,
            actions_sequence=actions_sequence,
            success_rate=0.0,  # Will be updated as skill executes
            usage_count=0,
            total_attempts=0,
            frames_range=frames_range,
            context_tags=context_tags or [],
            related_entities=related_entities or [],
        )

        self.skill_discoveries[skill_id] = discovery

        # Track action pattern
        pattern_key = " → ".join(actions_sequence)
        self.action_patterns[pattern_key] += 1

        self.logger.info(
            f"Skill discovery: {skill_id}={skill_name}, "
            f"actions={len(actions_sequence)}, frames={frames_range}",
            extra={
                "skill_id": skill_id,
                "skill_name": skill_name,
                "action_count": len(actions_sequence),
                "frames_range": frames_range,
                "context_tags": context_tags or [],
            }
        )

    def add_skill_execution(
        self,
        frame_number: int,
        skill_id: str,
        skill_name: str,
        success: bool,
        predicted_outcome: str,
        actual_outcome: str,
        reward: Optional[float] = None,
    ) -> None:
        """Record execution of a skill.

        Args:
            frame_number: Sequential frame number
            skill_id: Skill identifier
            skill_name: Skill name
            success: Whether execution was successful
            predicted_outcome: What the model predicted would happen
            actual_outcome: What actually happened
            reward: Optional reward value
        """
        timestamp = datetime.now().isoformat()

        predicted_vs_actual_match = predicted_outcome.lower() == actual_outcome.lower()

        execution = SkillExecution(
            timestamp=timestamp,
            frame_number=frame_number,
            skill_id=skill_id,
            skill_name=skill_name,
            success=success,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            predicted_vs_actual_match=predicted_vs_actual_match,
            reward=reward,
        )

        self.skill_executions.append(execution)

        # Update usage statistics
        self.skill_usage_count[skill_id] += 1
        if success:
            self.skill_success_count[skill_id] += 1

        # Update skill discovery record
        if skill_id in self.skill_discoveries:
            discovery = self.skill_discoveries[skill_id]
            discovery.usage_count = self.skill_usage_count[skill_id]
            discovery.total_attempts += 1
            if discovery.total_attempts > 0:
                discovery.success_rate = self.skill_success_count[skill_id] / discovery.total_attempts

        self.logger.info(
            f"Skill execution: {skill_id}={skill_name}, success={success}, "
            f"prediction_match={predicted_vs_actual_match}",
            extra={
                "frame_number": frame_number,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "success": success,
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "prediction_match": predicted_vs_actual_match,
                "reward": reward,
            }
        )

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all skills.

        Returns:
            Dictionary containing skill formation statistics
        """
        total_skills = len(self.skill_discoveries)
        total_executions = len(self.skill_executions)

        if not self.skill_executions:
            return {
                "total_skills_discovered": total_skills,
                "total_executions": 0,
            }

        # Compute success rate
        successful_executions = sum(1 for e in self.skill_executions if e.success)
        overall_success_rate = successful_executions / total_executions if total_executions > 0 else 0.0

        # Compute prediction accuracy
        correct_predictions = sum(1 for e in self.skill_executions if e.predicted_vs_actual_match)
        prediction_accuracy = correct_predictions / total_executions if total_executions > 0 else 0.0

        # Per-skill statistics
        skill_stats = {}
        for skill_id, discovery in self.skill_discoveries.items():
            usage = self.skill_usage_count.get(skill_id, 0)
            success = self.skill_success_count.get(skill_id, 0)
            success_rate = success / usage if usage > 0 else 0.0

            skill_stats[skill_id] = {
                "name": discovery.skill_name,
                "action_count": len(discovery.actions_sequence),
                "usage_count": usage,
                "success_count": success,
                "success_rate": success_rate,
                "context_tags": discovery.context_tags,
            }

        # Top action patterns
        top_patterns = sorted(
            self.action_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        stats = {
            "total_skills_discovered": total_skills,
            "total_executions": total_executions,
            "overall_success_rate": overall_success_rate,
            "prediction_accuracy": prediction_accuracy,
            "skill_statistics": skill_stats,
            "top_action_patterns": dict(top_patterns),
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Skill statistics: {total_skills} skills, {total_executions} executions, "
            f"success_rate={overall_success_rate:.2f}, prediction_accuracy={prediction_accuracy:.2f}",
            extra=stats,
        )

        return stats

    def save_analysis(self, filename: str = "skill_analysis.jsonl") -> Path:
        """Save analysis results to file.

        Args:
            filename: Output filename (in log_dir)

        Returns:
            Path to saved analysis file
        """
        output_path = self.log_dir / filename

        # Save skill discoveries
        with open(output_path, 'w') as f:
            for skill_id in sorted(self.skill_discoveries.keys()):
                discovery = self.skill_discoveries[skill_id]
                f.write(json.dumps(discovery.to_dict()) + '\n')

        # Save executions
        if self.skill_executions:
            executions_path = self.log_dir / "skill_executions.jsonl"
            with open(executions_path, 'w') as f:
                for execution in self.skill_executions:
                    f.write(json.dumps(execution.to_dict()) + '\n')

        # Save summary statistics
        stats = self.compute_statistics()
        stats_path = self.log_dir / "skill_analysis_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            f"Saved skill analysis: {output_path}",
            extra={
                "output_path": str(output_path),
                "skills": len(self.skill_discoveries),
                "executions": len(self.skill_executions),
            }
        )

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of skill analysis.

        Returns:
            Dictionary with key metrics and statistics
        """
        return {
            "total_skills_discovered": len(self.skill_discoveries),
            "total_executions": len(self.skill_executions),
            "statistics": self.compute_statistics(),
            "timestamp": datetime.now().isoformat(),
        }
