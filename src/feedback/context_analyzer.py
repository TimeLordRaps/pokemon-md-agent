"""Analyze context window evolution and memory management.

This module provides tools to understand:
- Context window size and composition changes
- Memory silo usage patterns
- Token consumption trends
- Context engineering effectiveness
- Information retention across frames
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
from datetime import datetime
from collections import defaultdict

from src.utils.logging_setup import get_logger


@dataclass
class ContextSnapshot:
    """Snapshot of context state at a specific frame."""

    timestamp: str
    frame_number: int
    context_length: int  # Total tokens in context
    memory_silos_used: int  # Number of temporal silos in use
    silo_distribution: Dict[str, int]  # Silo name -> token count
    game_state_tokens: int
    retrieval_tokens: int  # Tokens from RAG retrieval
    instruction_tokens: int  # Instruction tokens
    entities_in_context: Set[str]  # Entity IDs mentioned

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "context_length": self.context_length,
            "memory_silos_used": self.memory_silos_used,
            "silo_distribution": self.silo_distribution,
            "game_state_tokens": self.game_state_tokens,
            "retrieval_tokens": self.retrieval_tokens,
            "instruction_tokens": self.instruction_tokens,
            "entities_in_context": list(self.entities_in_context),
        }


@dataclass
class ContextTransition:
    """Record of context changes between frames."""

    timestamp: str
    from_frame: int
    to_frame: int
    context_length_change: int  # Positive = growth, negative = shrinking
    tokens_added: int
    tokens_removed: int
    entities_added: Set[str]
    entities_removed: Set[str]
    silos_added: List[str]
    silos_removed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "context_length_change": self.context_length_change,
            "tokens_added": self.tokens_added,
            "tokens_removed": self.tokens_removed,
            "entities_added": list(self.entities_added),
            "entities_removed": list(self.entities_removed),
            "silos_added": self.silos_added,
            "silos_removed": self.silos_removed,
        }


class ContextWindowAnalyzer:
    """Analyze context window evolution and memory management.

    Tracks:
    - Context window size over time
    - Memory silo usage patterns
    - Token consumption trends
    - Entity prevalence in context
    - Context engineering effectiveness
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize context analyzer.

        Args:
            log_dir: Optional directory to save analysis results
        """
        self.logger = get_logger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: Dict[int, ContextSnapshot] = {}
        self.transitions: List[ContextTransition] = []
        self.entity_frequency: Dict[str, int] = defaultdict(int)
        self.silo_usage: Dict[str, List[int]] = defaultdict(list)  # Silo -> token counts

    def add_context_snapshot(
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
        """Record context state at a frame.

        Args:
            frame_number: Sequential frame number
            context_length: Total tokens in context window
            memory_silos_used: Number of temporal silos in use
            silo_distribution: Map of silo names to token counts
            game_state_tokens: Tokens used for game state representation
            retrieval_tokens: Tokens from RAG retrieval
            instruction_tokens: Instruction/prompt tokens
            entities_in_context: List of entity IDs in context
        """
        timestamp = datetime.now().isoformat()
        entities = set(entities_in_context or [])

        snapshot = ContextSnapshot(
            timestamp=timestamp,
            frame_number=frame_number,
            context_length=context_length,
            memory_silos_used=memory_silos_used,
            silo_distribution=silo_distribution,
            game_state_tokens=game_state_tokens,
            retrieval_tokens=retrieval_tokens,
            instruction_tokens=instruction_tokens,
            entities_in_context=entities,
        )

        self.snapshots[frame_number] = snapshot

        # Update entity frequency
        for entity in entities:
            self.entity_frequency[entity] += 1

        # Update silo usage
        for silo, token_count in silo_distribution.items():
            self.silo_usage[silo].append(token_count)

        self.logger.info(
            f"Context snapshot: frame={frame_number}, length={context_length}, "
            f"silos={memory_silos_used}, entities={len(entities)}",
            extra={
                "frame_number": frame_number,
                "context_length": context_length,
                "memory_silos_used": memory_silos_used,
                "game_state_tokens": game_state_tokens,
                "retrieval_tokens": retrieval_tokens,
                "instruction_tokens": instruction_tokens,
                "entity_count": len(entities),
            }
        )

        # Compute transition from previous frame
        if len(self.snapshots) > 1:
            prev_frame = max(f for f in self.snapshots.keys() if f < frame_number)
            self._compute_transition(prev_frame, frame_number)

    def _compute_transition(self, from_frame: int, to_frame: int) -> None:
        """Compute and record context transition between frames."""
        from_snapshot = self.snapshots[from_frame]
        to_snapshot = self.snapshots[to_frame]

        context_length_change = to_snapshot.context_length - from_snapshot.context_length

        # Compute silo transitions
        from_silos = set(from_snapshot.silo_distribution.keys())
        to_silos = set(to_snapshot.silo_distribution.keys())
        silos_added = list(to_silos - from_silos)
        silos_removed = list(from_silos - to_silos)

        # Compute entity transitions
        entities_added = to_snapshot.entities_in_context - from_snapshot.entities_in_context
        entities_removed = from_snapshot.entities_in_context - to_snapshot.entities_in_context

        # Compute token transitions (simplified: assume added/removed based on change)
        tokens_added = max(0, context_length_change)
        tokens_removed = max(0, -context_length_change)

        transition = ContextTransition(
            timestamp=datetime.now().isoformat(),
            from_frame=from_frame,
            to_frame=to_frame,
            context_length_change=context_length_change,
            tokens_added=tokens_added,
            tokens_removed=tokens_removed,
            entities_added=entities_added,
            entities_removed=entities_removed,
            silos_added=silos_added,
            silos_removed=silos_removed,
        )

        self.transitions.append(transition)

        self.logger.info(
            f"Context transition: {from_frame} → {to_frame}, "
            f"change={context_length_change:+d}, silos_added={silos_added}, "
            f"entities_added={len(entities_added)}",
            extra={
                "from_frame": from_frame,
                "to_frame": to_frame,
                "context_length_change": context_length_change,
                "silos_added": silos_added,
                "silos_removed": silos_removed,
                "entities_added_count": len(entities_added),
                "entities_removed_count": len(entities_removed),
            }
        )

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all context snapshots.

        Returns:
            Dictionary containing context evolution statistics
        """
        if not self.snapshots:
            return {"total_snapshots": 0}

        context_lengths = [s.context_length for s in self.snapshots.values()]
        silo_usage_counts = [s.memory_silos_used for s in self.snapshots.values()]

        mean_context_length = sum(context_lengths) / len(context_lengths)
        max_context_length = max(context_lengths)
        min_context_length = min(context_lengths)

        mean_silos_used = sum(silo_usage_counts) / len(silo_usage_counts)

        # Find most frequent entities
        top_entities = sorted(
            self.entity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Silo usage statistics
        silo_stats = {}
        for silo, counts in self.silo_usage.items():
            silo_stats[silo] = {
                "mean_tokens": sum(counts) / len(counts) if counts else 0,
                "max_tokens": max(counts) if counts else 0,
                "min_tokens": min(counts) if counts else 0,
                "usage_count": len(counts),
            }

        stats = {
            "total_snapshots": len(self.snapshots),
            "total_transitions": len(self.transitions),
            "context_length": {
                "mean": mean_context_length,
                "max": max_context_length,
                "min": min_context_length,
            },
            "memory_silos": {
                "mean_used": mean_silos_used,
                "silo_statistics": silo_stats,
            },
            "top_entities": dict(top_entities),
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Context statistics: {len(self.snapshots)} snapshots, "
            f"mean_context={mean_context_length:.0f}, "
            f"mean_silos={mean_silos_used:.1f}",
            extra=stats,
        )

        return stats

    def save_analysis(self, filename: str = "context_analysis.jsonl") -> Path:
        """Save analysis results to file.

        Args:
            filename: Output filename (in log_dir)

        Returns:
            Path to saved analysis file
        """
        output_path = self.log_dir / filename

        with open(output_path, 'w') as f:
            for frame_number in sorted(self.snapshots.keys()):
                snapshot = self.snapshots[frame_number]
                f.write(json.dumps(snapshot.to_dict()) + '\n')

        # Save transitions
        if self.transitions:
            transitions_path = self.log_dir / "context_transitions.jsonl"
            with open(transitions_path, 'w') as f:
                for transition in self.transitions:
                    f.write(json.dumps(transition.to_dict()) + '\n')

        # Save summary statistics
        stats = self.compute_statistics()
        stats_path = self.log_dir / "context_analysis_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            f"Saved context analysis: {output_path}",
            extra={
                "output_path": str(output_path),
                "snapshots": len(self.snapshots),
                "transitions": len(self.transitions),
            }
        )

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of context analysis.

        Returns:
            Dictionary with key metrics and statistics
        """
        return {
            "total_snapshots": len(self.snapshots),
            "total_transitions": len(self.transitions),
            "statistics": self.compute_statistics(),
            "timestamp": datetime.now().isoformat(),
        }
