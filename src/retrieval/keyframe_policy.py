"""Keyframe policy for temporal importance scoring and adaptive sampling."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Sampling strategies for keyframe selection."""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    IMPORTANCE = "importance"
    TEMPORAL_DENSITY = "temporal_density"


@dataclass
class KeyframeCandidate:
    """Candidate for keyframe selection."""
    timestamp: float
    embedding: np.ndarray
    metadata: Dict[str, Any]
    importance_score: float = 0.0
    temporal_density: float = 0.0
    ssim_score: Optional[float] = None
    frame_image: Optional[Image.Image] = None  # Image for SSIM calculation
    floor_changed: bool = False
    room_changed: bool = False
    combat_active: bool = False
    inventory_changed: bool = False
    new_species_seen: bool = False


@dataclass
class KeyframeResult:
    """Result of keyframe selection."""
    selected_keyframes: List[KeyframeCandidate]
    sampling_rate: float
    strategy_used: SamplingStrategy
    coverage_score: float
    total_candidates: int


class KeyframePolicy:
    """Policy for selecting keyframes based on temporal importance."""

    def __init__(
        self,
        base_sampling_rate: float = 0.1,  # Sample 10% of frames
        adaptive_threshold: float = 0.7,
        min_keyframes: int = 5,
        max_keyframes: int = 100,
        temporal_window_seconds: float = 60.0,
        ssim_threshold: float = 0.85,  # SSIM threshold for dropping similar frames
    ):
        """Initialize keyframe policy.

        Args:
            base_sampling_rate: Base sampling rate (0.0-1.0)
            adaptive_threshold: Threshold for switching to adaptive mode
            min_keyframes: Minimum keyframes to maintain
            max_keyframes: Maximum keyframes to store
            temporal_window_seconds: Window for temporal analysis
            ssim_threshold: SSIM threshold for dropping similar frames (0.0-1.0)
        """
        self.base_sampling_rate = base_sampling_rate
        self.adaptive_threshold = adaptive_threshold
        self.min_keyframes = min_keyframes
        self.max_keyframes = max_keyframes
        self.temporal_window_seconds = temporal_window_seconds
        self.ssim_threshold = ssim_threshold

        # State tracking
        self.recent_candidates: List[KeyframeCandidate] = []
        self.selected_keyframes: List[KeyframeCandidate] = []
        self.strategy_history: List[SamplingStrategy] = []
        self.last_keyframe_image: Optional[Image.Image] = None  # Store previous keyframe image for SSIM

        logger.info(
            "Initialized KeyframePolicy: rate=%.2f, min=%d, max=%d",
            base_sampling_rate, min_keyframes, max_keyframes
        )

    def select_keyframes(
        self,
        candidates: List[KeyframeCandidate],
        current_stuckness_score: Optional[float] = None,
        force_adaptive: bool = False,
    ) -> KeyframeResult:
        """Select keyframes from candidates based on policy.

        Args:
            candidates: Candidate frames to evaluate
            current_stuckness_score: Current stuckness score (0.0-1.0)
            force_adaptive: Force adaptive sampling

        Returns:
            KeyframeResult with selected keyframes
        """
        if not candidates:
            return KeyframeResult(
                selected_keyframes=[],
                sampling_rate=0.0,
                strategy_used=SamplingStrategy.UNIFORM,
                coverage_score=0.0,
                total_candidates=0,
            )

        # Determine sampling strategy
        strategy = self._determine_strategy(current_stuckness_score, force_adaptive)

        # Calculate SSIM scores for candidates against previous keyframe
        candidates_with_ssim = self._calculate_ssim_scores(candidates)

        # Score candidates based on strategy
        scored_candidates = self._score_candidates(candidates_with_ssim, strategy)

        # Apply specific triggers for keyframe promotion
        promoted_candidates = self._apply_keyframe_triggers(scored_candidates)

        # Select keyframes
        selected = self._select_from_scored(promoted_candidates, strategy)

        # Update state - store the last selected keyframe image for future SSIM comparisons
        self.selected_keyframes.extend(selected)
        if selected and selected[-1].frame_image is not None:
            self.last_keyframe_image = selected[-1].frame_image
        self._maintain_keyframe_limits()
        self.strategy_history.append(strategy)

        # Calculate metrics
        sampling_rate = len(selected) / len(candidates)
        coverage_score = self._calculate_coverage_score(selected, candidates)

        result = KeyframeResult(
            selected_keyframes=selected,
            sampling_rate=sampling_rate,
            strategy_used=strategy,
            coverage_score=coverage_score,
            total_candidates=len(candidates),
        )

        logger.debug(
            "Selected %d/%d keyframes using %s strategy (rate=%.3f, coverage=%.3f)",
            len(selected), len(candidates), strategy.value, sampling_rate, coverage_score
        )

        return result

    def _determine_strategy(
        self,
        stuckness_score: Optional[float],
        force_adaptive: bool,
    ) -> SamplingStrategy:
        """Determine which sampling strategy to use."""
        if force_adaptive:
            return SamplingStrategy.ADAPTIVE

        if stuckness_score is not None and stuckness_score > self.adaptive_threshold:
            # High stuckness -> use importance-based sampling
            return SamplingStrategy.IMPORTANCE

        if len(self.strategy_history) >= 5:
            # Check recent strategy diversity
            recent_strategies = self.strategy_history[-5:]
            if len(set(recent_strategies)) <= 2:
                # Low diversity -> switch to adaptive
                return SamplingStrategy.ADAPTIVE

        # Default to uniform sampling
        return SamplingStrategy.UNIFORM

    def _score_candidates(
        self,
        candidates: List[KeyframeCandidate],
        strategy: SamplingStrategy,
    ) -> List[KeyframeCandidate]:
        """Score candidates based on sampling strategy."""
        scored = []

        for candidate in candidates:
            if strategy == SamplingStrategy.UNIFORM:
                # Uniform: equal weight
                candidate.importance_score = 1.0

            elif strategy == SamplingStrategy.ADAPTIVE:
                # Adaptive: based on temporal density and recency
                candidate.temporal_density = self._calculate_temporal_density(candidate, candidates)
                candidate.importance_score = self._calculate_adaptive_score(candidate)

            elif strategy == SamplingStrategy.IMPORTANCE:
                # Importance: based on embedding variance and metadata
                candidate.importance_score = self._calculate_importance_score(candidate, candidates)

            elif strategy == SamplingStrategy.TEMPORAL_DENSITY:
                # Temporal density: focus on dense activity periods
                candidate.temporal_density = self._calculate_temporal_density(candidate, candidates)
                candidate.importance_score = candidate.temporal_density

            scored.append(candidate)

        return scored

    def _select_from_scored(
        self,
        candidates: List[KeyframeCandidate],
        strategy: SamplingStrategy,
    ) -> List[KeyframeCandidate]:
        """Select keyframes from scored candidates."""
        if not candidates:
            return []

        # Sort by importance score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.importance_score, reverse=True)

        # Apply strategy-specific selection
        if strategy == SamplingStrategy.UNIFORM:
            # Take top N based on base sampling rate
            target_count = max(self.min_keyframes, int(len(candidates) * self.base_sampling_rate))
            selected = sorted_candidates[:target_count]

        elif strategy in [SamplingStrategy.ADAPTIVE, SamplingStrategy.IMPORTANCE]:
            # Take top candidates above threshold
            threshold = self._calculate_dynamic_threshold(sorted_candidates)
            selected = [c for c in sorted_candidates if c.importance_score >= threshold]
            selected = selected[:self.max_keyframes]  # Cap at max

        else:
            # Default: take top min_keyframes
            selected = sorted_candidates[:self.min_keyframes]

        return selected

    def _calculate_adaptive_score(self, candidate: KeyframeCandidate) -> float:
        """Calculate adaptive score based on temporal density and recency."""
        current_time = time.time()

        # Recency factor (newer = higher score)
        time_diff = current_time - candidate.timestamp
        recency_score = max(0.0, 1.0 - (time_diff / self.temporal_window_seconds))

        # Temporal density factor
        density_score = min(1.0, candidate.temporal_density / 10.0)  # Normalize

        # Combine factors
        return 0.7 * recency_score + 0.3 * density_score

    def _calculate_importance_score(
        self,
        candidate: KeyframeCandidate,
        all_candidates: List[KeyframeCandidate],
    ) -> float:
        """Calculate importance score based on embedding variance."""
        if len(all_candidates) < 2:
            return 1.0

        # Calculate distance to nearest neighbors
        min_distance = float('inf')

        for other in all_candidates:
            if other is not candidate:
                distance = np.linalg.norm(candidate.embedding - other.embedding)
                min_distance = min(min_distance, distance)

        # Higher score for more unique embeddings (lower min_distance = more similar = lower score)
        uniqueness_score = min(1.0, min_distance / 0.5)  # Normalize assuming typical distance ~0.5

        # Boost score for frames with action metadata
        action_boost = 1.2 if candidate.metadata.get('has_action') else 1.0

        return uniqueness_score * action_boost

    def _calculate_temporal_density(
        self,
        candidate: KeyframeCandidate,
        all_candidates: List[KeyframeCandidate],
    ) -> float:
        """Calculate temporal density around candidate."""
        window_start = candidate.timestamp - (self.temporal_window_seconds / 2)
        window_end = candidate.timestamp + (self.temporal_window_seconds / 2)

        nearby_count = sum(
            1 for c in all_candidates
            if window_start <= c.timestamp <= window_end
        )

        # Density = events per second in window
        window_duration = self.temporal_window_seconds
        return nearby_count / window_duration if window_duration > 0 else 0

    def _calculate_dynamic_threshold(self, sorted_candidates: List[KeyframeCandidate]) -> float:
        """Calculate dynamic threshold for selection."""
        if not sorted_candidates:
            return 0.0

        # Use percentile-based threshold
        scores = [c.importance_score for c in sorted_candidates]
        threshold_percentile = 75  # Top 25% of candidates

        threshold = np.percentile(scores, threshold_percentile)

        # Ensure minimum threshold
        return max(threshold, 0.5)

    def _calculate_coverage_score(
        self,
        selected: List[KeyframeCandidate],
        candidates: List[KeyframeCandidate],
    ) -> float:
        """Calculate how well selected keyframes cover the temporal space."""
        if not selected or not candidates:
            return 0.0

        # Coverage based on time span coverage
        candidate_times = [c.timestamp for c in candidates]
        selected_times = [c.timestamp for c in selected]

        min_time = min(candidate_times)
        max_time = max(candidate_times)
        total_span = max_time - min_time

        if total_span == 0:
            return 1.0  # Single point

        # Calculate covered time span
        selected_min = min(selected_times)
        selected_max = max(selected_times)
        covered_span = selected_max - selected_min

        return covered_span / total_span

    def _apply_keyframe_triggers(self, candidates: List[KeyframeCandidate]) -> List[KeyframeCandidate]:
        """Apply specific keyframe promotion triggers."""
        for candidate in candidates:
            # SSIM drop (significant visual change)
            if candidate.ssim_score is not None and candidate.ssim_score < self.ssim_threshold:
                candidate.importance_score = max(candidate.importance_score, 2.0)

            # Floor/room change
            if candidate.floor_changed or candidate.room_changed:
                candidate.importance_score = max(candidate.importance_score, 3.0)

            # Combat event
            if candidate.combat_active:
                candidate.importance_score = max(candidate.importance_score, 2.5)

            # Inventory change
            if candidate.inventory_changed:
                candidate.importance_score = max(candidate.importance_score, 1.8)

            # New species seen
            if candidate.new_species_seen:
                candidate.importance_score = max(candidate.importance_score, 2.2)

        return candidates

    def _maintain_keyframe_limits(self) -> None:
        """Maintain keyframe limits by evicting old keyframes."""
        if len(self.selected_keyframes) > self.max_keyframes:
            # Keep most recent keyframes
            self.selected_keyframes.sort(key=lambda k: k.timestamp, reverse=True)
            self.selected_keyframes = self.selected_keyframes[:self.max_keyframes]

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about keyframe policy performance."""
        if not self.strategy_history:
            return {"status": "no_selections_yet"}

        strategy_counts = {}
        for strategy in self.strategy_history:
            strategy_counts[strategy.value] = strategy_counts.get(strategy.value, 0) + 1

        avg_sampling_rate = np.mean([
            result.sampling_rate for result in []  # Would need to store results
        ]) if self.strategy_history else 0.0

        return {
            "total_selections": len(self.strategy_history),
            "current_keyframes": len(self.selected_keyframes),
            "strategy_distribution": strategy_counts,
            "avg_sampling_rate": avg_sampling_rate,
            "temporal_window_seconds": self.temporal_window_seconds,
        }

    def _calculate_ssim_scores(self, candidates: List[KeyframeCandidate]) -> List[KeyframeCandidate]:
        """Calculate SSIM scores between candidates and the last keyframe image."""
        if not self.last_keyframe_image:
            # No previous keyframe, all candidates get None SSIM score
            return candidates

        updated_candidates = []
        for candidate in candidates:
            if candidate.frame_image is not None:
                try:
                    # Convert PIL images to numpy arrays for SSIM calculation
                    prev_array = np.array(self.last_keyframe_image.convert('L'))  # Convert to grayscale
                    curr_array = np.array(candidate.frame_image.convert('L'))

                    # Calculate SSIM
                    ssim_score = ssim(prev_array, curr_array, data_range=curr_array.max() - curr_array.min())
                    candidate.ssim_score = ssim_score
                    logger.debug(f"SSIM score calculated: {ssim_score:.3f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate SSIM for candidate: {e}")
                    candidate.ssim_score = None
            else:
                candidate.ssim_score = None
            updated_candidates.append(candidate)

        return updated_candidates

    def clear_history(self) -> None:
        """Clear keyframe selection history."""
        self.recent_candidates.clear()
        self.selected_keyframes.clear()
        self.strategy_history.clear()
        self.last_keyframe_image = None
        logger.info("Cleared keyframe policy history")