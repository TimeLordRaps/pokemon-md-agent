"""Automatic trajectory retrieval for Pokemon MD agent."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time
import numpy as np

from ..embeddings.temporal_silo import TemporalSiloManager, SiloEntry
from ..embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedTrajectory:
    """A retrieved trajectory from the RAG system."""
    trajectory_id: str
    similarity_score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    silo_id: str
    action_sequence: List[str]
    outcome: Optional[str] = None


@dataclass
class RetrievalQuery:
    """Query for trajectory retrieval."""
    current_embedding: np.ndarray
    current_position: Optional[tuple[int, int]] = None
    current_mission: Optional[str] = None
    current_floor: Optional[int] = None
    max_distance: float = 50.0  # Maximum distance for position-based filtering
    time_window_seconds: float = 60.0  # Only consider recent trajectories


class AutoRetriever:
    """Automatically retrieves relevant trajectories from temporal silos."""
    
    def __init__(
        self,
        silo_manager: TemporalSiloManager,
        vector_store: VectorStore,
        auto_retrieval_count: int = 3,
        similarity_threshold: float = 0.7,
    ):
        """Initialize auto retriever.
        
        Args:
            silo_manager: Temporal silo manager
            vector_store: Vector store for similarity search
            auto_retrieval_count: Number of trajectories to retrieve automatically
            similarity_threshold: Minimum similarity threshold
        """
        self.silo_manager = silo_manager
        self.vector_store = vector_store
        self.auto_retrieval_count = auto_retrieval_count
        self.similarity_threshold = similarity_threshold
        
        # Track retrieval patterns
        self.retrieval_history: List[Dict[str, Any]] = []
        self.successful_patterns: Dict[str, int] = {}
        
        logger.info(
            "Initialized AutoRetriever: count=%d, threshold=%.2f",
            auto_retrieval_count,
            similarity_threshold
        )
    
    def retrieve_similar_trajectories(
        self,
        query: RetrievalQuery,
        silo_filter: Optional[List[str]] = None,
        on_device_buffer: Optional[Any] = None,  # OnDeviceBufferManager
    ) -> List[RetrievedTrajectory]:
        """Retrieve trajectories similar to current situation.

        Args:
            query: Retrieval query with current state
            silo_filter: Only search in these silos
            on_device_buffer: Optional on-device buffer for additional search

        Returns:
            List of retrieved trajectories
        """
        logger.debug("Retrieving similar trajectories")

        # Cross-silo search
        silo_results = self.silo_manager.cross_silo_search(
            query_embedding=query.current_embedding,
            silo_ids=silo_filter,
            top_k=self.auto_retrieval_count * 2  # Get more, then filter
        )

        # On-device ANN search if available
        ann_results = []
        if on_device_buffer is not None:
            try:
                # Run synchronous search for simplicity
                ann_results = on_device_buffer.search_similar(
                    query_embedding=query.current_embedding,
                    top_k=self.auto_retrieval_count,
                    search_timeout_ms=100,  # Fast search
                )
            except Exception as e:
                logger.warning("On-device ANN search failed: %s", e)

        # Convert to RetrievedTrajectory objects
        retrieved_trajectories = []

        # Add silo results
        for silo_id, matches in silo_results.items():
            for entry, similarity in matches:
                if similarity >= self.similarity_threshold:
                    # Apply additional filters
                    if self._passes_filters(entry, query):
                        trajectory = RetrievedTrajectory(
                            trajectory_id=entry.trajectory_id,
                            similarity_score=similarity,
                            embedding=entry.embedding,
                            metadata=entry.metadata,
                            timestamp=entry.timestamp,
                            silo_id=silo_id,
                            action_sequence=entry.metadata.get("action_sequence", []),
                            outcome=entry.metadata.get("outcome"),
                        )
                        retrieved_trajectories.append(trajectory)

        # Add ANN results (avoid duplicates)
        existing_ids = {t.trajectory_id for t in retrieved_trajectories}
        for ann_result in ann_results:
            if ann_result.entry_id not in existing_ids:
                trajectory = RetrievedTrajectory(
                    trajectory_id=ann_result.entry_id,
                    similarity_score=ann_result.score,
                    embedding=None,  # Not available from ANN search
                    metadata=ann_result.metadata,
                    timestamp=ann_result.metadata.get("timestamp", 0.0),
                    silo_id="on_device_ann",
                    action_sequence=ann_result.metadata.get("action_sequence", []),
                    outcome=ann_result.metadata.get("outcome"),
                )
                retrieved_trajectories.append(trajectory)

        # Sort by similarity and return top results
        retrieved_trajectories.sort(key=lambda t: t.similarity_score, reverse=True)
        final_results = retrieved_trajectories[:self.auto_retrieval_count]

        # Log retrieval
        self._log_retrieval(query, final_results)

        logger.info(
            "Retrieved %d trajectories (searched %d silos, %d ANN results)",
            len(final_results),
            len(silo_results),
            len(ann_results)
        )

        return final_results
    
    def _passes_filters(
        self,
        entry: SiloEntry,
        query: RetrievalQuery,
    ) -> bool:
        """Check if entry passes additional filters.
        
        Args:
            entry: Silo entry to check
            query: Retrieval query
            
        Returns:
            True if entry passes all filters
        """
        metadata = entry.metadata
        
        # Time window filter
        time_diff = time.time() - entry.timestamp
        if time_diff > query.time_window_seconds:
            return False
        
        # Position-based filter (if position data available)
        if query.current_position and "position" in metadata:
            entry_position = metadata["position"]
            if isinstance(entry_position, (list, tuple)) and len(entry_position) == 2:
                distance = np.sqrt(
                    (query.current_position[0] - entry_position[0]) ** 2 +
                    (query.current_position[1] - entry_position[1]) ** 2
                )
                if distance > query.max_distance:
                    return False
        
        # Mission filter
        if query.current_mission:
            entry_mission = metadata.get("mission")
            if entry_mission and entry_mission != query.current_mission:
                return False
        
        # Floor filter
        if query.current_floor:
            entry_floor = metadata.get("floor")
            if entry_floor and entry_floor != query.current_floor:
                return False
        
        return True
    
    def _log_retrieval(
        self,
        query: RetrievalQuery,
        trajectories: List[RetrievedTrajectory],
    ) -> None:
        """Log retrieval event for analysis.
        
        Args:
            query: Retrieval query
            trajectories: Retrieved trajectories
        """
        retrieval_record = {
            "timestamp": time.time(),
            "num_retrieved": len(trajectories),
            "avg_similarity": np.mean([t.similarity_score for t in trajectories]) if trajectories else 0.0,
            "silo_distribution": {},
            "query_metadata": {
                "has_position": query.current_position is not None,
                "has_mission": query.current_mission is not None,
                "has_floor": query.current_floor is not None,
            }
        }
        
        # Count silo distribution
        for trajectory in trajectories:
            silo_id = trajectory.silo_id
            retrieval_record["silo_distribution"][silo_id] = \
                retrieval_record["silo_distribution"].get(silo_id, 0) + 1
        
        self.retrieval_history.append(retrieval_record)
        
        # Keep only recent history
        if len(self.retrieval_history) > 1000:
            self.retrieval_history = self.retrieval_history[-1000:]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        if not self.retrieval_history:
            return {"status": "no_retrievals_yet"}
        
        recent_history = self.retrieval_history[-100:]  # Last 100 retrievals
        
        avg_retrieved = np.mean([r["num_retrieved"] for r in recent_history])
        avg_similarity = np.mean([r["avg_similarity"] for r in recent_history if r["avg_similarity"] > 0])
        
        # Most common silos
        all_silos = {}
        for record in recent_history:
            for silo_id, count in record["silo_distribution"].items():
                all_silos[silo_id] = all_silos.get(silo_id, 0) + count
        
        most_common_silo = max(all_silos.items(), key=lambda x: x[1]) if all_silos else None
        
        return {
            "total_retrievals": len(self.retrieval_history),
            "recent_avg_retrieved": avg_retrieved,
            "recent_avg_similarity": avg_similarity,
            "most_common_silo": most_common_silo,
            "silo_usage_distribution": all_silos,
            "retrieval_rate": len(recent_history) / max(1, len(self.retrieval_history)),
        }
    
    def find_patterns(
        self,
        successful_outcomes: List[str],
        min_occurrences: int = 3,
    ) -> Dict[str, Any]:
        """Find patterns in successful retrievals.
        
        Args:
            successful_outcomes: List of outcomes considered successful
            min_occurrences: Minimum occurrences for pattern recognition
            
        Returns:
            Dictionary with pattern analysis
        """
        pattern_analysis = {
            "successful_trajectories": [],
            "common_action_sequences": {},
            "successful_silo_patterns": {},
            "time_based_patterns": {},
        }
        
        # Analyze successful trajectories
        for record in self.retrieval_history[-500:]:  # Last 500 retrievals
            # This would need to correlate with actual outcomes
            # For now, just track silo usage patterns
            
            silo_dist = record["silo_distribution"]
            for silo_id, count in silo_dist.items():
                if silo_id not in pattern_analysis["successful_silo_patterns"]:
                    pattern_analysis["successful_silo_patterns"][silo_id] = 0
                pattern_analysis["successful_silo_patterns"][silo_id] += count
        
        return pattern_analysis
    
    def clear_history(self) -> None:
        """Clear retrieval history."""
        self.retrieval_history.clear()
        self.successful_patterns.clear()
        logger.info("Cleared auto-retriever history")
