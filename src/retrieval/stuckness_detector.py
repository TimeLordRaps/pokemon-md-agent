"""Stuckness detection for Pokemon MD agent using cross-temporal divergence."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class StucknessStatus(Enum):
    """Stuckness status levels."""
    NOT_STUCK = "not_stuck"
    POTENTIALLY_STUCK = "potentially_stuck"
    STUCK = "stuck"
    VERY_STUCK = "very_stuck"


@dataclass
class StucknessAnalysis:
    """Result of stuckness analysis."""
    status: StucknessStatus
    short_term_similarity: float
    long_term_similarity: float
    divergence_score: float
    confidence: float
    reasons: List[str]
    suggested_actions: List[str]


@dataclass
class TemporalSnapshot:
    """Snapshot of agent state at a point in time."""
    timestamp: float
    embedding: np.ndarray
    position: Optional[tuple[int, int]] = None
    action: Optional[str] = None
    floor: Optional[int] = None
    mission: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StucknessDetector:
    """Detects when agent is stuck in loops using temporal analysis."""
    
    def __init__(
        self,
        divergence_threshold: float = 0.4,
        short_term_window: int = 4,  # Last 4 seconds
        long_term_window: int = 120,  # Last 2 minutes
        min_samples: int = 3,
        similarity_threshold: float = 0.85,
    ):
        """Initialize stuckness detector.
        
        Args:
            divergence_threshold: Threshold for considering agent stuck
            short_term_window: Window for short-term similarity (seconds)
            long_term_window: Window for long-term similarity (seconds)
            min_samples: Minimum samples needed for analysis
            similarity_threshold: Threshold for considering states similar
        """
        self.divergence_threshold = divergence_threshold
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
        
        # History of temporal snapshots
        self.snapshots: List[TemporalSnapshot] = []
        
        # Track stuckness patterns
        self.stuck_patterns: Dict[str, int] = {}
        self.last_stuck_time: Optional[float] = None
        
        logger.info(
            "Initialized StucknessDetector: threshold=%.2f, windows=(%ds, %ds)",
            divergence_threshold,
            short_term_window,
            long_term_window
        )
    
    def add_snapshot(self, snapshot: TemporalSnapshot) -> None:
        """Add a new temporal snapshot with enhanced logging.

        Args:
            snapshot: Temporal snapshot to add
        """
        self.snapshots.append(snapshot)

        # Keep only recent snapshots (last 10 minutes)
        cutoff_time = snapshot.timestamp - 600
        self.snapshots = [
            s for s in self.snapshots if s.timestamp >= cutoff_time
        ]

        logger.info(
            "Stuckness state transition: added snapshot at t=%.1f, position=(%s,%s), action=%s, floor=%s, total_snapshots=%d",
            snapshot.timestamp,
            snapshot.position[0] if snapshot.position else 'None',
            snapshot.position[1] if snapshot.position else 'None',
            snapshot.action or 'None',
            snapshot.floor or 'None',
            len(self.snapshots)
        )
    
    def analyze(
        self,
        current_embedding: np.ndarray,
        current_position: Optional[tuple[int, int]] = None,
        current_action: Optional[str] = None,
        current_time: Optional[float] = None,
        on_device_buffer: Optional[Any] = None,  # OnDeviceBufferManager
    ) -> StucknessAnalysis:
        """Analyze if agent is stuck.
        
        Args:
            current_embedding: Current agent embedding
            current_position: Current agent position
            current_action: Current or last action
            current_time: Current time (uses time.time() if None)
            
        Returns:
            StucknessAnalysis with status and reasoning
        """
        if current_time is None:
            import time
            current_time = time.time()
        
        # Create current snapshot
        current_snapshot = TemporalSnapshot(
            timestamp=current_time,
            embedding=current_embedding,
            position=current_position,
            action=current_action,
        )
        
        self.add_snapshot(current_snapshot)

        # Feed keyframe policy if on-device buffer available
        if on_device_buffer is not None:
            try:
                # Get current stuckness score for keyframe policy
                recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
                if len(recent_snapshots) >= 3:
                    # Calculate simple stuckness score
                    recent_embeddings = [s.embedding for s in recent_snapshots]
                    similarities = []
                    for i in range(1, len(recent_embeddings)):
                        sim = self._cosine_similarity(recent_embeddings[i], recent_embeddings[i-1])
                        similarities.append(sim)

                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    stuckness_score = 1.0 - avg_similarity  # Higher similarity = lower stuckness

                    # Process keyframes
                    import asyncio
                    asyncio.create_task(
                        on_device_buffer.process_keyframes(current_stuckness=stuckness_score)
                    )
            except Exception as e:
                logger.warning("Keyframe policy feed failed: %s", e)

        # Need enough samples for analysis
        if len(self.snapshots) < self.min_samples:
            logger.debug("Stuckness state: insufficient data (%d snapshots < %d required)",
                        len(self.snapshots), self.min_samples)
            return StucknessAnalysis(
                status=StucknessStatus.NOT_STUCK,
                short_term_similarity=0.0,
                long_term_similarity=0.0,
                divergence_score=0.0,
                confidence=0.0,
                reasons=["Insufficient data for analysis"],
                suggested_actions=["Continue normal operation"],
            )

        return self._perform_analysis(current_snapshot)
    
    def _perform_analysis(self, current_snapshot: TemporalSnapshot) -> StucknessAnalysis:
        """Perform the actual stuckness analysis.
        
        Args:
            current_snapshot: Current temporal snapshot
            
        Returns:
            StucknessAnalysis with detailed results
        """
        # Get short-term and long-term windows
        recent_snapshots = self._get_snapshots_in_window(
            current_snapshot.timestamp - self.short_term_window,
            current_snapshot.timestamp
        )
        
        older_snapshots = self._get_snapshots_in_window(
            current_snapshot.timestamp - self.long_term_window,
            current_snapshot.timestamp - self.short_term_window
        )
        
        # Calculate similarities
        short_term_similarity = self._calculate_window_similarity(
            current_snapshot, recent_snapshots
        )
        
        long_term_similarity = self._calculate_window_similarity(
            current_snapshot, older_snapshots
        )
        
        # Calculate divergence score
        divergence_score = self._calculate_divergence(
            short_term_similarity,
            long_term_similarity
        )
        
        # Determine stuckness status
        status, confidence, reasons, actions = self._determine_stuckness(
            divergence_score,
            short_term_similarity,
            long_term_similarity,
            len(recent_snapshots),
            len(older_snapshots)
        )
        
        logger.info(
            "Stuckness state analysis: status=%s (confidence=%.2f), divergence=%.3f, short_term_sim=%.3f, long_term_sim=%.3f, reasons=%s",
            status.value,
            confidence,
            divergence_score,
            short_term_similarity,
            long_term_similarity,
            "; ".join(reasons)
        )
        
        return StucknessAnalysis(
            status=status,
            short_term_similarity=short_term_similarity,
            long_term_similarity=long_term_similarity,
            divergence_score=divergence_score,
            confidence=confidence,
            reasons=reasons,
            suggested_actions=actions,
        )
    
    def _get_snapshots_in_window(
        self,
        start_time: float,
        end_time: float,
    ) -> List[TemporalSnapshot]:
        """Get snapshots within time window.
        
        Args:
            start_time: Start of window
            end_time: End of window
            
        Returns:
            List of snapshots in window
        """
        return [
            snapshot for snapshot in self.snapshots
            if start_time <= snapshot.timestamp <= end_time
        ]
    
    def _calculate_window_similarity(
        self,
        current_snapshot: TemporalSnapshot,
        window_snapshots: List[TemporalSnapshot],
    ) -> float:
        """Calculate average similarity with snapshots in window.
        
        Args:
            current_snapshot: Current snapshot
            window_snapshots: Snapshots in window
            
        Returns:
            Average cosine similarity
        """
        if not window_snapshots:
            return 0.0
        
        similarities = []
        
        for snapshot in window_snapshots:
            similarity = self._cosine_similarity(
                current_snapshot.embedding,
                snapshot.embedding
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def _calculate_divergence(
        self,
        short_term_similarity: float,
        long_term_similarity: float,
    ) -> float:
        """Calculate divergence between short and long term similarities.
        
        Args:
            short_term_similarity: Short-term window similarity
            long_term_similarity: Long-term window similarity
            
        Returns:
            Divergence score (higher = more stuck)
        """
        # Divergence is high when short-term is similar but long-term is different
        # This indicates repetitive behavior without progress
        
        if long_term_similarity < 0.1:
            # Not enough variation in long-term to make判断
            return 0.0
        
        divergence = (short_term_similarity - long_term_similarity) / long_term_similarity
        
        # Clip to [0, 1] range
        return max(0.0, min(1.0, divergence))
    
    def _determine_stuckness(
        self,
        divergence_score: float,
        short_term_similarity: float,
        long_term_similarity: float,
        num_recent: int,
        num_older: int,
    ) -> tuple[StucknessStatus, float, List[str], List[str]]:
        """Determine stuckness status and generate recommendations.
        
        Args:
            divergence_score: Calculated divergence score
            short_term_similarity: Short-term similarity
            long_term_similarity: Long-term similarity
            num_recent: Number of recent samples
            num_older: Number of older samples
            
        Returns:
            Tuple of (status, confidence, reasons, suggested_actions)
        """
        reasons = []
        actions = []
        confidence = 0.0
        
        # Not enough data
        if num_recent < 2:
            return (
                StucknessStatus.NOT_STUCK,
                0.5,
                ["Insufficient recent data"],
                ["Continue normal operation"]
            )
        
        if num_older < 2:
            return (
                StucknessStatus.NOT_STUCK,
                0.3,
                ["Insufficient historical data"],
                ["Continue normal operation, gather more data"]
            )
        
        # High divergence indicates being stuck
        if divergence_score >= self.divergence_threshold:
            if divergence_score >= 0.8:
                status = StucknessStatus.VERY_STUCK
                confidence = 0.9
                reasons.append("Very high behavioral divergence detected")
                actions.extend([
                    "Escalate to 8B model",
                    "Fetch guidance from dashboard",
                    "Consider environment reset or different strategy"
                ])
            else:
                status = StucknessStatus.STUCK
                confidence = 0.7
                reasons.append("High behavioral divergence detected")
                actions.extend([
                    "Escalate to 4B model",
                    "Increase temporal zoom (lower FPS)",
                    "Change movement patterns"
                ])
        # Medium divergence
        elif divergence_score >= self.divergence_threshold * 0.6:
            status = StucknessStatus.POTENTIALLY_STUCK
            confidence = 0.5
            reasons.append("Moderate behavioral divergence")
            actions.extend([
                "Monitor closely",
                "Consider alternative routes",
                "Increase exploration radius"
            ])
        # Low divergence
        else:
            status = StucknessStatus.NOT_STUCK
            confidence = 0.8
            reasons.append("Behavioral patterns show normal variation")
            actions.append("Continue current strategy")
        
        # Additional checks
        if short_term_similarity > self.similarity_threshold:
            reasons.append(f"Very similar recent behavior (similarity: {short_term_similarity:.2f})")
            if status == StucknessStatus.NOT_STUCK:
                actions[0] = "Monitor for repetitive patterns"
        
        if long_term_similarity < 0.3:
            reasons.append("Low long-term similarity suggests exploration")
        
        return status, confidence, reasons, actions
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get analysis of stuckness patterns.
        
        Returns:
            Dictionary with pattern analysis
        """
        if len(self.snapshots) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze action patterns
        action_counts = {}
        position_changes = []
        
        for i in range(1, len(self.snapshots)):
            prev_snap = self.snapshots[i-1]
            curr_snap = self.snapshots[i]
            
            # Count actions
            action = curr_snap.action or "unknown"
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Track position changes
            if prev_snap.position and curr_snap.position:
                pos_change = np.sqrt(
                    (prev_snap.position[0] - curr_snap.position[0]) ** 2 +
                    (prev_snap.position[1] - curr_snap.position[1]) ** 2
                )
                position_changes.append(pos_change)
        
        # Calculate statistics
        avg_position_change = np.mean(position_changes) if position_changes else 0
        
        most_common_action = max(action_counts.items(), key=lambda x: x[1]) if action_counts else None
        
        return {
            "total_snapshots": len(self.snapshots),
            "time_span_minutes": (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) / 60,
            "most_common_action": most_common_action,
            "action_distribution": action_counts,
            "avg_position_change_per_step": avg_position_change,
            "estimated_stuckness_events": len([s for s in self.snapshots if s.metadata and s.metadata.get('stuck')]),
        }
    
    def clear_history(self) -> None:
        """Clear all stuckness detection history."""
        self.snapshots.clear()
        self.stuck_patterns.clear()
        self.last_stuck_time = None
        logger.info("Cleared stuckness detection history")
    
    def export_analysis(self, filename: str) -> None:
        """Export stuckness analysis to file.
        
        Args:
            filename: Output filename
        """
        analysis = self.get_pattern_analysis()
        
        with open(filename, 'w') as f:
            f.write("Stuckness Detection Analysis\n")
            f.write("=" * 30 + "\n\n")
            
            for key, value in analysis.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nLast {min(10, len(self.snapshots))} snapshots:\n")
            for snapshot in self.snapshots[-10:]:
                f.write(f"  t={snapshot.timestamp:.1f}, action={snapshot.action}\n")
        
        logger.info("Exported stuckness analysis to %s", filename)
