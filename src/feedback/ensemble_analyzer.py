"""Analyze ensemble consensus and model voting patterns.

This module provides tools to understand:
- Individual model outputs and predictions
- Model agreement/disagreement patterns
- Voting consensus computation
- Attention distribution across models
- Model-specific failure modes
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from collections import defaultdict, Counter

from src.utils.logging_setup import get_logger


@dataclass
class ModelOutput:
    """Output from a single model in the ensemble."""

    model_name: str
    model_variant: str  # e.g., "qwen3-vl-2b-instruct"
    prediction: str  # The model's predicted action
    confidence: float  # Model's confidence score (0-1)
    reasoning: Optional[str] = None  # Model's reasoning text
    attention_distribution: Optional[Dict[str, float]] = None  # Attention weights
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_variant": self.model_variant,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "attention_distribution": self.attention_distribution,
            "tokens_used": self.tokens_used,
        }


@dataclass
class ConsensusDecision:
    """Result of ensemble voting for a decision."""

    timestamp: str
    frame_number: int
    winning_action: str
    voting_scores: Dict[str, int]  # Action -> vote count
    confidence_scores: Dict[str, float]  # Action -> mean confidence
    total_models: int
    consensus_strength: float  # Ratio of votes for winning action
    unanimous: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "winning_action": self.winning_action,
            "voting_scores": self.voting_scores,
            "confidence_scores": self.confidence_scores,
            "total_models": self.total_models,
            "consensus_strength": self.consensus_strength,
            "unanimous": self.unanimous,
        }


@dataclass
class ModelDisagreement:
    """Record of models disagreeing on a prediction."""

    timestamp: str
    frame_number: int
    model_predictions: Dict[str, str]  # Model name -> prediction
    dominant_action: str
    dissenting_models: List[str]
    dissent_ratio: float  # Ratio of models disagreeing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "model_predictions": self.model_predictions,
            "dominant_action": self.dominant_action,
            "dissenting_models": self.dissenting_models,
            "dissent_ratio": self.dissent_ratio,
        }


class EnsembleConsensusAnalyzer:
    """Analyze ensemble consensus and model voting patterns.

    Tracks:
    - Per-frame voting results
    - Model agreement patterns
    - Consensus strength over time
    - Model-specific disagreement patterns
    - Individual model confidence vs. correctness
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize ensemble analyzer.

        Args:
            log_dir: Optional directory to save analysis results
        """
        self.logger = get_logger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs/feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_outputs: Dict[int, List[ModelOutput]] = defaultdict(list)
        self.consensus_decisions: Dict[int, ConsensusDecision] = {}
        self.disagreements: List[ModelDisagreement] = []
        self.model_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_predictions": 0,
            "total_confidence": 0.0,
            "predictions_count": Counter(),
        })

    def add_model_output(
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
        """Record output from a single model.

        Args:
            frame_number: Sequential frame number
            model_name: Name of the model (e.g., "model_1")
            model_variant: Model variant identifier (e.g., "qwen3-vl-2b-instruct")
            prediction: The action predicted by the model
            confidence: Confidence score (0-1)
            reasoning: Optional reasoning text
            attention_distribution: Optional attention weights
            tokens_used: Number of tokens used by the model
        """
        output = ModelOutput(
            model_name=model_name,
            model_variant=model_variant,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            attention_distribution=attention_distribution,
            tokens_used=tokens_used,
        )

        self.model_outputs[frame_number].append(output)

        # Update model statistics
        stats = self.model_stats[model_name]
        stats["total_predictions"] += 1
        stats["total_confidence"] += confidence
        stats["predictions_count"][prediction] += 1

        self.logger.info(
            f"Model output: frame={frame_number}, model={model_name}, "
            f"prediction={prediction}, confidence={confidence:.3f}",
            extra={
                "frame_number": frame_number,
                "model_name": model_name,
                "model_variant": model_variant,
                "prediction": prediction,
                "confidence": confidence,
                "tokens_used": tokens_used,
            }
        )

    def compute_consensus(
        self,
        frame_number: int,
        winning_action: str,
    ) -> ConsensusDecision:
        """Compute ensemble consensus for a frame.

        Args:
            frame_number: Sequential frame number
            winning_action: The action selected by consensus

        Returns:
            ConsensusDecision with voting statistics
        """
        outputs = self.model_outputs.get(frame_number, [])

        if not outputs:
            self.logger.warning(f"No model outputs for frame {frame_number}")
            return None

        # Count votes for each action
        voting_scores = Counter(o.prediction for o in outputs)

        # Compute mean confidence per action
        confidence_by_action = defaultdict(list)
        for output in outputs:
            confidence_by_action[output.prediction].append(output.confidence)

        confidence_scores = {
            action: sum(confs) / len(confs)
            for action, confs in confidence_by_action.items()
        }

        # Compute consensus metrics
        total_models = len(outputs)
        winning_votes = voting_scores.get(winning_action, 0)
        consensus_strength = winning_votes / total_models if total_models > 0 else 0.0
        unanimous = (winning_votes == total_models)

        decision = ConsensusDecision(
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            winning_action=winning_action,
            voting_scores=dict(voting_scores),
            confidence_scores=confidence_scores,
            total_models=total_models,
            consensus_strength=consensus_strength,
            unanimous=unanimous,
        )

        self.consensus_decisions[frame_number] = decision

        # Check for disagreement
        if not unanimous:
            self._record_disagreement(frame_number, outputs, winning_action, voting_scores)

        self.logger.info(
            f"Consensus: frame={frame_number}, action={winning_action}, "
            f"strength={consensus_strength:.2f}, unanimous={unanimous}",
            extra={
                "frame_number": frame_number,
                "winning_action": winning_action,
                "voting_scores": dict(voting_scores),
                "consensus_strength": consensus_strength,
                "unanimous": unanimous,
            }
        )

        return decision

    def _record_disagreement(
        self,
        frame_number: int,
        outputs: List[ModelOutput],
        winning_action: str,
        voting_scores: Counter,
    ) -> None:
        """Record model disagreement."""
        model_predictions = {o.model_name: o.prediction for o in outputs}
        dissenting_models = [
            o.model_name for o in outputs
            if o.prediction != winning_action
        ]
        dissent_ratio = len(dissenting_models) / len(outputs) if outputs else 0.0

        disagreement = ModelDisagreement(
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            model_predictions=model_predictions,
            dominant_action=winning_action,
            dissenting_models=dissenting_models,
            dissent_ratio=dissent_ratio,
        )

        self.disagreements.append(disagreement)

        self.logger.info(
            f"Model disagreement: frame={frame_number}, "
            f"dissent_ratio={dissent_ratio:.2f}, dissenting={dissenting_models}",
            extra={
                "frame_number": frame_number,
                "model_predictions": model_predictions,
                "dissent_ratio": dissent_ratio,
                "dissenting_models": dissenting_models,
            }
        )

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all decisions.

        Returns:
            Dictionary containing consensus and disagreement statistics
        """
        total_frames = len(self.consensus_decisions)
        total_disagreements = len(self.disagreements)

        # Compute average consensus strength
        consensus_strengths = [
            d.consensus_strength for d in self.consensus_decisions.values()
        ]
        mean_consensus_strength = (
            sum(consensus_strengths) / len(consensus_strengths)
            if consensus_strengths else 0.0
        )

        # Compute model agreement statistics
        model_names = list(self.model_stats.keys())
        model_confidences = {
            name: stats["total_confidence"] / stats["total_predictions"]
            if stats["total_predictions"] > 0 else 0.0
            for name, stats in self.model_stats.items()
        }

        stats = {
            "total_frames": total_frames,
            "total_disagreements": total_disagreements,
            "disagreement_ratio": (
                total_disagreements / total_frames if total_frames > 0 else 0.0
            ),
            "mean_consensus_strength": mean_consensus_strength,
            "model_statistics": {
                name: {
                    "total_predictions": self.model_stats[name]["total_predictions"],
                    "mean_confidence": model_confidences[name],
                    "unique_predictions": len(self.model_stats[name]["predictions_count"]),
                }
                for name in model_names
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Ensemble statistics: {total_frames} frames, "
            f"{total_disagreements} disagreements, "
            f"mean_consensus={mean_consensus_strength:.2f}",
            extra=stats,
        )

        return stats

    def save_analysis(self, filename: str = "ensemble_analysis.jsonl") -> Path:
        """Save analysis results to file.

        Args:
            filename: Output filename (in log_dir)

        Returns:
            Path to saved analysis file
        """
        output_path = self.log_dir / filename

        with open(output_path, 'w') as f:
            for frame_number in sorted(self.consensus_decisions.keys()):
                decision = self.consensus_decisions[frame_number]
                f.write(json.dumps(decision.to_dict()) + '\n')

        # Save disagreement records
        if self.disagreements:
            disagree_path = self.log_dir / "disagreements.jsonl"
            with open(disagree_path, 'w') as f:
                for disagreement in self.disagreements:
                    f.write(json.dumps(disagreement.to_dict()) + '\n')

        # Save summary statistics
        stats = self.compute_statistics()
        stats_path = self.log_dir / "ensemble_analysis_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(
            f"Saved ensemble analysis: {output_path}",
            extra={
                "output_path": str(output_path),
                "frames": len(self.consensus_decisions),
                "disagreements": len(self.disagreements),
            }
        )

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble consensus analysis.

        Returns:
            Dictionary with key metrics and statistics
        """
        return {
            "total_frames": len(self.consensus_decisions),
            "total_disagreements": len(self.disagreements),
            "statistics": self.compute_statistics(),
            "timestamp": datetime.now().isoformat(),
        }
