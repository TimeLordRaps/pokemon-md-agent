"""Automatic trajectory retrieval for Pokemon MD agent.

Analysis of retrieval logic, dependencies, and integration hooks:

**Retrieval Logic:**
- Top-k=3 retrieval with deduplication by trajectory_id and episode
- Recency bias with exponential decay (rate=0.001/s)
- Cross-floor gating with diversity preservation (same-floor + ≥1 other-floor)
- RRF merge for parallel multi-head searches (vision/memory/action heads)
- Filtering by time window, position, mission, and floor constraints
- Fallback to on-device ANN search when available

**Dependencies:**
- TemporalSiloManager: Cross-silo search across temporal silos
- VectorStore: Similarity search for embeddings
- Deduplicator: Content deduplication (optional)
- numpy: Vector operations and statistics

**Integration Hooks:**
- RAG pipeline entry point for trajectory retrieval
- Works with StucknessDetector for loop prevention
- Provides retrieval stats for ModelRouter decision making
- Logs retrieval history for pattern analysis
- Gatekeeper integration via shallow hit thresholds

Changed lines & context scanned: top-k=3, dedup, recency bias, cross-floor gating, diversity preservation."""

from typing import List, Dict, Any, Optional, Tuple, Iterable, Set
from dataclasses import dataclass, replace
import logging
import time
import asyncio
import concurrent.futures
from collections import defaultdict
import numpy as np

from ..embeddings.temporal_silo import TemporalSiloManager, SiloEntry
from ..embeddings.vector_store import VectorStore
from .deduplicator import Deduplicator

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Exception raised for retrieval system errors."""
    pass


@dataclass
class RetrievedTrajectory:
    """A retrieved trajectory from the RAG system."""
    trajectory_id: str
    similarity_score: float
    embedding: Optional[np.ndarray]  # Allow None for ANN results
    metadata: Dict[str, Any]
    timestamp: float
    silo_id: str
    action_sequence: List[str]
    outcome: Optional[str] = None
    raw_similarity: float = 0.0
    recency_weight: float = 1.0
    episode_id: Optional[int] = None


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
    """Automatically retrieves relevant trajectories from temporal silos.

    Provides intelligent retrieval with top-k=3, deduplication, recency bias,
    and cross-floor gating capabilities for the PMD-Red Agent RAG pipeline.
    """

    def __init__(
        self,
        silo_manager: TemporalSiloManager,
        vector_store: VectorStore,
        deduplicator: Optional[Deduplicator] = None,
        auto_retrieval_count: int = 3,
        similarity_threshold: float = 0.7,
        rrf_k: int = 60,  # RRF constant
        recency_decay_rate: float = 0.001,  # Exponential decay per second
        distance_threshold: float = 0.5,  # Cosine distance for conflicts
        cross_floor_gating: bool = True,  # Allow retrieval across different floors
        on_device_buffer: Optional[Any] = None,  # OnDeviceBuffer instance for query buffering
    ):
        """Initialize auto retriever.

        Args:
            silo_manager: Temporal silo manager
            vector_store: Vector store for similarity search
            deduplicator: Deduplicator instance for content deduplication
            auto_retrieval_count: Number of trajectories to retrieve automatically
            similarity_threshold: Minimum similarity threshold
            rrf_k: RRF constant (higher = less aggressive fusion)
            recency_decay_rate: Exponential decay rate for recency bias
            distance_threshold: Cosine distance threshold for trajectory conflicts
            cross_floor_gating: If True, allow retrieval across different dungeon floors
            on_device_buffer: OnDeviceBuffer instance for buffering recent queries
        """
        self.silo_manager = silo_manager
        self.vector_store = vector_store
        self.deduplicator = deduplicator or Deduplicator()
        self.auto_retrieval_count = auto_retrieval_count
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        self.recency_decay_rate = recency_decay_rate
        self.distance_threshold = distance_threshold
        self.cross_floor_gating = cross_floor_gating
        self.on_device_buffer = on_device_buffer

        # Track retrieval patterns
        self.retrieval_history: List[Dict[str, Any]] = []
        self.successful_patterns: Dict[str, int] = {}

        logger.info(
            "Initialized AutoRetriever: count=%d, threshold=%.2f, cross_floor_gating=%s",
            auto_retrieval_count,
            similarity_threshold,
            cross_floor_gating
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        cross_floor_gating: Optional[bool] = None,
    ) -> List[RetrievedTrajectory]:
        """Retrieve top-3 relevant trajectories with deduplication and recency bias.

        Args:
            query: Retrieval query with current state
            cross_floor_gating: Override class-level cross_floor_gating setting

        Returns:
            Exactly 3 retrieved trajectories (or fewer if insufficient matches)
        """
        # Buffer query if on-device buffer is available
        if self.on_device_buffer is not None:
            try:
                self.on_device_buffer.store(
                    embedding=query.current_embedding,
                    metadata={
                        "timestamp": time.time(),
                        "query_type": "retrieval",
                        "current_position": query.current_position,
                        "current_mission": query.current_mission,
                        "current_floor": query.current_floor,
                        "time_window_seconds": query.time_window_seconds,
                    }
                )
                logger.debug("Buffered query in on-device buffer")
            except Exception as e:
                logger.warning("Failed to buffer query: %s", e)

        # Use parameter override or class default
        allow_cross_floor = cross_floor_gating if cross_floor_gating is not None else self.cross_floor_gating

        # Episode-aware search with recency weighting
        current_time = time.time()
        episode_results = self.silo_manager.search_across_episodes(
            query_embedding=query.current_embedding,
            top_k_per_episode=max(self.auto_retrieval_count * 3, 9),
            max_episodes=3,
            current_time=current_time,
        )

        candidates: List[RetrievedTrajectory] = []
        episodes_seen: set[int] = set()

        for result in episode_results:
            entry = result.entry
            raw_similarity = result.raw_similarity or 0.0
            adjusted_similarity = result.score
            recency_weight = result.recency_weight

            episodes_seen.add(result.episode_id)

            # Enforce similarity threshold using raw similarity as baseline
            if raw_similarity < self.similarity_threshold and adjusted_similarity < self.similarity_threshold:
                continue

            if not self._passes_filters(entry, query, allow_cross_floor):
                continue

            trajectory = self._build_retrieved_trajectory(
                entry=entry,
                similarity=adjusted_similarity,
                raw_similarity=raw_similarity,
                recency_weight=recency_weight,
                episode_id=result.episode_id,
            )
            candidates.append(trajectory)

        # Deduplicate by trajectory_id (keep highest similarity)
        deduped = self._deduplicate_by_trajectory_id(candidates)

        # Apply recency bias (respects pre-weighted trajectories)
        final_candidates = self._apply_recency_bias(deduped)

        # Ensure cross-floor diversity when gating is enabled
        if allow_cross_floor and query.current_floor is not None:
            final_candidates = self._ensure_cross_floor_diversity(final_candidates, query.current_floor)

        # Return top-3 results
        results = final_candidates[:3]

        # Log retrieval with floor mix
        floor_mix = self._compute_floor_mix(results, query.current_floor)
        logger.info(
            "Retrieved %d trajectories (cross_floor=%s, candidates=%d, deduped=%d, floor_mix=%s, episodes=%d)",
            len(results),
            allow_cross_floor,
            len(candidates),
            len(deduped),
            floor_mix,
            len(episodes_seen),
        )

        self._log_retrieval(query, results, episodes_seen=episodes_seen)

        return results

    def _deduplicate_by_trajectory_id(
        self,
        trajectories: List[RetrievedTrajectory],
    ) -> List[RetrievedTrajectory]:
        """Deduplicate trajectories by trajectory_id, keeping highest similarity.

        Args:
            trajectories: List of trajectories to deduplicate

        Returns:
            Deduplicated list
        """
        trajectory_map = {}

        for trajectory in trajectories:
            tid = trajectory.trajectory_id
            if tid not in trajectory_map or trajectory.similarity_score > trajectory_map[tid].similarity_score:
                trajectory_map[tid] = trajectory

        return list(trajectory_map.values())

    def _build_retrieved_trajectory(
        self,
        entry: SiloEntry,
        similarity: float,
        raw_similarity: float,
        recency_weight: float,
        episode_id: Optional[int],
    ) -> RetrievedTrajectory:
        """Convert a SiloEntry into a RetrievedTrajectory with episode metadata."""
        metadata = dict(entry.metadata)
        metadata.setdefault("floor", entry.floor)
        metadata.setdefault("timestamp", entry.timestamp)
        effective_episode = episode_id if episode_id is not None else metadata.get("episode_id", entry.episode_id)
        metadata.setdefault("episode_id", effective_episode)
        if effective_episode is not None and "episode" not in metadata:
            metadata["episode"] = str(effective_episode)

        return RetrievedTrajectory(
            trajectory_id=entry.trajectory_id,
            similarity_score=similarity,
            raw_similarity=raw_similarity,
            recency_weight=recency_weight,
            episode_id=effective_episode,
            embedding=entry.embedding,
            metadata=metadata,
            timestamp=entry.timestamp,
            silo_id=entry.silo,
            action_sequence=metadata.get("action_sequence", []),
            outcome=metadata.get("outcome"),
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

        # Buffer query if on-device buffer is available (use class instance or parameter)
        buffer_to_use = on_device_buffer if on_device_buffer is not None else self.on_device_buffer
        if buffer_to_use is not None:
            try:
                buffer_to_use.store(
                    embedding=query.current_embedding,
                    metadata={
                        "timestamp": time.time(),
                        "query_type": "similar_trajectories",
                        "current_position": query.current_position,
                        "current_mission": query.current_mission,
                        "current_floor": query.current_floor,
                        "time_window_seconds": query.time_window_seconds,
                    }
                )
                logger.debug("Buffered query in on-device buffer")
            except Exception as e:
                logger.warning("Failed to buffer query: %s", e)

        # Episode-aware search (optional silo filter)
        current_time = time.time()
        episode_results = self.silo_manager.search_across_episodes(
            query_embedding=query.current_embedding,
            top_k_per_episode=max(self.auto_retrieval_count * 2, 6),
            max_episodes=3,
            silo_ids=silo_filter,
            current_time=current_time,
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

        # Add silo results from episode-aware search
        for result in episode_results:
            entry = result.entry
            raw_similarity = result.raw_similarity or 0.0
            adjusted_similarity = result.score

            if raw_similarity < self.similarity_threshold and adjusted_similarity < self.similarity_threshold:
                continue

            if not self._passes_filters(entry, query, self.cross_floor_gating):
                continue

            trajectory = self._build_retrieved_trajectory(
                entry=entry,
                similarity=adjusted_similarity,
                raw_similarity=raw_similarity,
                recency_weight=result.recency_weight,
                episode_id=result.episode_id,
            )
            retrieved_trajectories.append(trajectory)

        # Add ANN results (avoid duplicates)
        existing_ids = {t.trajectory_id for t in retrieved_trajectories}
        for ann_result in ann_results:
            if ann_result.entry_id not in existing_ids:
                metadata = dict(ann_result.metadata or {})
                trajectory = RetrievedTrajectory(
                    trajectory_id=ann_result.entry_id,
                    similarity_score=ann_result.score,
                    raw_similarity=ann_result.score,
                    recency_weight=1.0,
                    embedding=None,  # Not available from ANN search
                    metadata=metadata,
                    timestamp=metadata.get("timestamp", 0.0),
                    silo_id="on_device_ann",
                    action_sequence=metadata.get("action_sequence", []),
                    outcome=metadata.get("outcome"),
                )
                retrieved_trajectories.append(trajectory)
                existing_ids.add(trajectory.trajectory_id)

        # On-device buffer search if available
        if on_device_buffer is not None and buffer_to_use is not None:
            try:
                buffer_results = buffer_to_use.search_similar(
                    query_embedding=query.current_embedding,
                    top_k=self.auto_retrieval_count,
                )
                # Convert to RetrievedTrajectory format
                for result in buffer_results:
                    if result.entry_id not in existing_ids:
                        metadata = dict(result.metadata or {})
                        trajectory = RetrievedTrajectory(
                            trajectory_id=result.entry_id or f"buffer_{len(retrieved_trajectories)}",
                            similarity_score=result.score,
                            raw_similarity=result.score,
                            recency_weight=1.0,
                            embedding=None,  # Available in result.embedding
                            metadata=metadata,
                            timestamp=metadata.get("timestamp", time.time()),
                            silo_id="on_device_buffer",
                            action_sequence=metadata.get("action_sequence", []),
                            outcome=metadata.get("outcome"),
                        )
                        retrieved_trajectories.append(trajectory)
                        existing_ids.add(trajectory.trajectory_id)
            except Exception as e:
                logger.warning("On-device buffer search failed: %s", e)

        # Sort by similarity and return top results
        retrieved_trajectories.sort(key=lambda t: t.similarity_score, reverse=True)
        final_results = retrieved_trajectories[:self.auto_retrieval_count]

        # Log retrieval with episode spread
        episodes_seen = {t.episode_id for t in retrieved_trajectories if t.episode_id is not None}
        self._log_retrieval(query, final_results, episodes_seen=episodes_seen)

        logger.info(
            "Retrieved %d trajectories (episodes=%d, ann=%d)",
            len(final_results),
            len(episodes_seen),
            len(ann_results),
        )

        return final_results

    async def retrieve_parallel_rrf(
        self,
        query: RetrievalQuery,
        model_heads: Optional[List[str]] = None,
    ) -> List[RetrievedTrajectory]:
        """Retrieve trajectories using parallel queries with RRF merge.

        Args:
            query: Retrieval query with current state
            model_heads: List of model head identifiers for parallel search

        Returns:
            List of retrieved trajectories after RRF merge and deduplication

        Raises:
            RetrievalError: If parallel retrieval fails
        """
        try:
            logger.debug("Starting parallel RRF retrieval")

            # Buffer query if on-device buffer is available
            if self.on_device_buffer is not None:
                try:
                    self.on_device_buffer.store(
                        embedding=query.current_embedding,
                        metadata={
                            "timestamp": time.time(),
                            "query_type": "parallel_rrf",
                            "current_position": query.current_position,
                            "current_mission": query.current_mission,
                            "current_floor": query.current_floor,
                            "time_window_seconds": query.time_window_seconds,
                        }
                    )
                    logger.debug("Buffered query in on-device buffer")
                except Exception as e:
                    logger.warning("Failed to buffer query: %s", e)

            # Default to global + per-model heads if not specified
            if model_heads is None:
                model_heads = ["global", "vision", "memory", "action"]

            # Parallel search across heads
            search_tasks = []
            for head in model_heads:
                task = asyncio.create_task(
                    self._search_single_head(query, head)
                )
                search_tasks.append(task)

            # Wait for all searches to complete
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning("Head %s search failed: %s", model_heads[i], result)
                    continue
                valid_results.append(result)

            if not valid_results:
                logger.warning("All parallel searches failed")
                return []

            # RRF merge
            merged_trajectories = self._rrf_merge(valid_results, self.rrf_k)

            # Episode deduplication
            deduped_trajectories = self._deduplicate_by_episode(merged_trajectories)

            # Apply recency bias
            final_trajectories = self._apply_recency_bias(deduped_trajectories)

            # Limit to auto_retrieval_count
            final_results = final_trajectories[:self.auto_retrieval_count]

            # Log retrieval with stats
            retrieval_stats = self._compute_retrieval_stats(final_results, self.distance_threshold)
            logger.info(
                "Parallel RRF retrieval: %d heads, %d raw, %d merged, %d final",
                len(valid_results),
                sum(len(r) for r in valid_results),
                len(merged_trajectories),
                len(final_results)
            )

            episodes_seen = {t.episode_id for t in final_results if t.episode_id is not None}
            self._log_retrieval(query, final_results, episodes_seen=episodes_seen)
            logged_stats = getattr(self, "_last_retrieval_stats", {})
            self._last_retrieval_stats = {**logged_stats, **retrieval_stats}

            return final_results

        except Exception as e:
            logger.error("Parallel RRF retrieval failed: %s", e)
            raise RetrievalError(f"Parallel retrieval failed: {e}") from e

    async def _search_single_head(
        self,
        query: RetrievalQuery,
        head: str,
    ) -> List[Tuple[str, float]]:
        """Search a single model head and return ranked (id, score) pairs.

        Args:
            query: Retrieval query
            head: Model head identifier

        Returns:
            List of RetrievedTrajectory objects sorted by similarity
        """
        try:
            silo_filter = self._get_silo_filter_for_head(head)

            loop = asyncio.get_running_loop()
            silo_results = await loop.run_in_executor(
                None,
                lambda: self.silo_manager.cross_silo_search(
                    query.current_embedding,
                    silo_filter,
                    self.auto_retrieval_count * 2,  # gather extras for ranking
                ),
            )

            ranked_results: List[RetrievedTrajectory] = []
            for silo_id, matches in (silo_results or {}).items():
                for entry, similarity in matches:
                    if similarity < self.similarity_threshold:
                        continue

                    raw_similarity = getattr(entry, "raw_similarity", similarity) or similarity
                    recency_weight = getattr(entry, "recency_weight", 1.0) or 1.0
                    episode_id = getattr(entry, "episode_id", None)

                    try:
                        trajectory = self._build_retrieved_trajectory(
                            entry=entry,
                            similarity=similarity,
                            raw_similarity=raw_similarity,
                            recency_weight=recency_weight,
                            episode_id=episode_id,
                        )
                    except Exception as build_error:
                        logger.debug(
                            "Failed to convert entry %s for head %s: %s",
                            getattr(entry, "trajectory_id", "unknown"),
                            head,
                            build_error,
                        )
                        continue

                    trajectory.silo_id = getattr(entry, "silo", "") or silo_id
                    trajectory.metadata["head"] = head
                    trajectory.metadata.setdefault("source_silo", silo_id)
                    if trajectory.episode_id is not None:
                        trajectory.metadata.setdefault("episode_id", trajectory.episode_id)
                        trajectory.metadata.setdefault("episode", str(trajectory.episode_id))
                    ranked_results.append(trajectory)

            ranked_results.sort(key=lambda t: t.similarity_score, reverse=True)
            return ranked_results[: self.auto_retrieval_count]

        except Exception as e:
            logger.warning("Search for head %s failed: %s", head, e)
            return []

    def _get_silo_filter_for_head(self, head: str) -> Optional[List[str]]:
        """Map model head to appropriate silo filter.

        Args:
            head: Model head identifier

        Returns:
            List of silo IDs to search, or None for all
        """
        head_silo_mapping = {
            "global": None,  # Search all silos
            "vision": ["temporal_1frame", "temporal_2frame"],
            "memory": ["temporal_4frame", "temporal_8frame"],
            "action": ["temporal_16frame", "temporal_32frame"],
        }
        return head_silo_mapping.get(head)

    def _rrf_merge(
        self,
        ranked_lists: List[List[RetrievedTrajectory]],
        k: int = 60,
    ) -> List[RetrievedTrajectory]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion while preserving metadata.

        Args:
            ranked_lists: Lists of trajectories from different sources/head searches
            k: RRF constant

        Returns:
            Merged list of RetrievedTrajectory objects with episode metadata retained
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        aggregated: Dict[str, RetrievedTrajectory] = {}
        head_sources: Dict[str, Set[str]] = defaultdict(set)

        for ranked_list in ranked_lists:
            for rank, trajectory in enumerate(ranked_list, 1):
                rrf_scores[trajectory.trajectory_id] += 1.0 / (k + rank)

                head = trajectory.metadata.get("head")
                if head:
                    head_sources[trajectory.trajectory_id].add(head)

                if trajectory.episode_id is not None:
                    trajectory.metadata.setdefault("episode_id", trajectory.episode_id)
                    trajectory.metadata.setdefault("episode", str(trajectory.episode_id))
                elif "episode_id" in trajectory.metadata:
                    try:
                        trajectory.episode_id = int(trajectory.metadata["episode_id"])
                    except (TypeError, ValueError):
                        trajectory.episode_id = None

                existing = aggregated.get(trajectory.trajectory_id)
                if existing is None or trajectory.similarity_score > existing.similarity_score:
                    aggregated[trajectory.trajectory_id] = trajectory

        merged_trajectories: List[RetrievedTrajectory] = []
        for trajectory_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            base = aggregated.get(trajectory_id)
            if base is None:
                continue

            merged = replace(base, similarity_score=rrf_score)
            merged.metadata = dict(base.metadata)

            if head_sources[trajectory_id]:
                merged.metadata["heads"] = sorted(head_sources[trajectory_id])
                merged.metadata.setdefault("head", merged.metadata["heads"][0])

            if merged.episode_id is not None:
                merged.metadata.setdefault("episode_id", merged.episode_id)
                merged.metadata.setdefault("episode", str(merged.episode_id))

            merged_trajectories.append(merged)

        return merged_trajectories

    def _deduplicate_by_episode(
        self,
        trajectories: List[RetrievedTrajectory],
    ) -> List[RetrievedTrajectory]:
        """Deduplicate trajectories by episode, keeping highest score.

        Args:
            trajectories: List of trajectories

        Returns:
            Deduplicated list
        """
        episode_map = {}

        for trajectory in trajectories:
            episode_value = trajectory.metadata.get("episode")
            if episode_value is None:
                episode_value = trajectory.metadata.get("episode_id", trajectory.episode_id)
            if episode_value is None:
                episode_value = trajectory.trajectory_id

            if episode_value not in episode_map or trajectory.similarity_score > episode_map[episode_value].similarity_score:
                episode_map[episode_value] = trajectory

        return list(episode_map.values())

    def _apply_recency_bias(
        self,
        trajectories: List[RetrievedTrajectory],
        now: Optional[float] = None,
    ) -> List[RetrievedTrajectory]:
        """Apply recency bias with exponential decay.

        Args:
            trajectories: List of trajectories
            now: Current timestamp (default: time.time())

        Returns:
            Trajectories with recency-adjusted scores
        """
        if now is None:
            now = time.time()

        adjusted: List[RetrievedTrajectory] = []

        for trajectory in trajectories:
            if trajectory.raw_similarity > 0:
                # Already weighted via episode search
                adjusted.append(trajectory)
                continue

            age_seconds = now - trajectory.timestamp
            recency_weight = np.exp(-self.recency_decay_rate * age_seconds)
            trajectory.recency_weight = recency_weight
            trajectory.similarity_score *= recency_weight
            adjusted.append(trajectory)

        # Re-sort after recency adjustment
        adjusted.sort(key=lambda t: t.similarity_score, reverse=True)

        return adjusted

    def _compute_retrieval_stats(
        self,
        trajectories: List[RetrievedTrajectory],
        distance_threshold: float,
    ) -> Dict[str, Any]:
        """Compute retrieval statistics for router decision making.

        Args:
            trajectories: Retrieved trajectories
            distance_threshold: Threshold for detecting conflicts

        Returns:
            Statistics dictionary
        """
        if not trajectories:
            return {"status": "no_trajectories"}

        # Average distance between trajectories
        distances = []
        conflicts = 0

        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                emb_i = trajectories[i].embedding
                emb_j = trajectories[j].embedding
                if (emb_i is not None and emb_j is not None and
                    hasattr(emb_i, 'shape') and hasattr(emb_j, 'shape') and
                    emb_i.shape == emb_j.shape):
                    distance = 1.0 - self._cosine_similarity(emb_i, emb_j)
                    distances.append(distance)
                    if distance < distance_threshold:
                        conflicts += 1

        avg_distance = np.mean(distances) if distances else 0.0

        # Episode coverage
        episodes = set()
        for t in trajectories:
            episode = t.metadata.get("episode")
            if not episode:
                episode = t.metadata.get("episode_id", t.episode_id)
            if episode is not None:
                episodes.add(str(episode))

        return {
            "avg_distance": avg_distance,
            "conflicts_detected": conflicts,
            "episodes_covered": len(episodes),
            "num_trajectories": len(trajectories),
            "avg_similarity": np.mean([t.similarity_score for t in trajectories]),
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_retrieval_stats_for_router(self) -> Optional[Dict[str, Any]]:
        """Get last retrieval stats for router decision making.

        Returns:
            Statistics from last retrieval, or None if no retrieval done
        """
        return getattr(self, '_last_retrieval_stats', None)

    def _passes_filters(
        self,
        entry: SiloEntry,
        query: RetrievalQuery,
        allow_cross_floor: bool,
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
        if query.current_floor and not allow_cross_floor:
            entry_floor = metadata.get("floor")
            if entry_floor and entry_floor != query.current_floor:
                return False
        
        return True
    
    def _log_retrieval(
        self,
        query: RetrievalQuery,
        trajectories: List[RetrievedTrajectory],
        episodes_seen: Optional[Iterable[Optional[int]]] = None,
    ) -> None:
        """Log retrieval event for analysis.
        
        Args:
            query: Retrieval query
            trajectories: Retrieved trajectories
        """
        episode_counts: Dict[str, int] = {}
        recency_weights: List[float] = []
        recency_lifts: List[float] = []

        for trajectory in trajectories:
            episode_value = trajectory.metadata.get("episode_id", trajectory.episode_id)
            if episode_value is not None:
                key = str(episode_value)
                episode_counts[key] = episode_counts.get(key, 0) + 1

            if trajectory.recency_weight:
                recency_weights.append(trajectory.recency_weight)

            if trajectory.raw_similarity > 0:
                lift = (trajectory.similarity_score - trajectory.raw_similarity) / max(trajectory.raw_similarity, 1e-6)
                recency_lifts.append(lift)

        avg_recency_weight = float(np.mean(recency_weights)) if recency_weights else 1.0
        avg_recency_lift = float(np.mean(recency_lifts)) if recency_lifts else 0.0
        episodes_considered = len(set(episodes_seen)) if episodes_seen is not None else len(episode_counts)

        retrieval_record = {
            "timestamp": time.time(),
            "num_retrieved": len(trajectories),
            "avg_similarity": np.mean([t.similarity_score for t in trajectories]) if trajectories else 0.0,
            "silo_distribution": {},
            "query_metadata": {
                "has_position": query.current_position is not None,
                "has_mission": query.current_mission is not None,
                "has_floor": query.current_floor is not None,
            },
            "episode_distribution": episode_counts,
            "episodes_considered": episodes_considered,
            "avg_recency_weight": avg_recency_weight,
            "avg_recency_lift": avg_recency_lift,
            "recency_lift_target_met": avg_recency_lift >= 0.2 if recency_lifts else None,
        }
        
        # Count silo distribution
        for trajectory in trajectories:
            silo_id = trajectory.silo_id
            retrieval_record["silo_distribution"][silo_id] = \
                retrieval_record["silo_distribution"].get(silo_id, 0) + 1
        
        # Persist stats for router consumption
        self._last_retrieval_stats = {
            "episodes_considered": episodes_considered,
            "avg_recency_lift": avg_recency_lift,
            "avg_recency_weight": avg_recency_weight,
            "num_candidates": len(trajectories),
        }
        
        self.retrieval_history.append(retrieval_record)
        
        # Keep only recent history
        if len(self.retrieval_history) > 1000:
            self.retrieval_history = self.retrieval_history[-1000:]
    
    def _ensure_cross_floor_diversity(
        self,
        trajectories: List[RetrievedTrajectory],
        current_floor: int,
    ) -> List[RetrievedTrajectory]:
        """Include same-floor trajectories plus ≥1 other-floor when available, preserving ranking.

        Args:
            trajectories: Sorted list of trajectories (best first)
            current_floor: Current floor number

        Returns:
            Same-floor + ≥1 other-floor trajectories when available, sorted by similarity
        """
        if len(trajectories) <= 1:
            return trajectories

        # Separate same-floor and different-floor trajectories
        same_floor = []
        other_floors = []

        for trajectory in trajectories:
            floor = trajectory.metadata.get("floor")
            if floor == current_floor:
                same_floor.append(trajectory)
            else:
                other_floors.append(trajectory)

        # Include same-floor + ≥1 other-floor when available
        if other_floors:
            # Leave room for at least 1 other-floor
            max_same = 2 if len(same_floor) > 2 else len(same_floor)
            result = same_floor[:max_same] + [other_floors[0]]
        else:
            # No other-floor available, take up to 3 same-floor
            result = same_floor[:3]

        # Preserve ranking by sorting result
        result.sort(key=lambda t: t.similarity_score, reverse=True)

        return result

    def _compute_floor_mix(
        self,
        trajectories: List[RetrievedTrajectory],
        current_floor: Optional[int],
    ) -> str:
        """Compute floor mix summary for logging.

        Args:
            trajectories: Retrieved trajectories
            current_floor: Current floor number

        Returns:
            String summary of floor distribution
        """
        if not trajectories or current_floor is None:
            return "unknown"

        floor_counts = {}
        for trajectory in trajectories:
            floor = trajectory.metadata.get("floor", "unknown")
            floor_counts[floor] = floor_counts.get(floor, 0) + 1

        same_count = floor_counts.get(current_floor, 0)
        other_count = sum(count for floor, count in floor_counts.items() if floor != current_floor)

        if other_count > 0:
            return f"same:{same_count},other:{other_count}"
        else:
            return f"same:{same_count}"

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
        avg_recency_lift = np.mean([
            r.get("avg_recency_lift", 0.0) for r in recent_history if r.get("avg_recency_lift") is not None
        ]) if recent_history else 0.0
        avg_recency_weight = np.mean([
            r.get("avg_recency_weight", 1.0) for r in recent_history if r.get("avg_recency_weight") is not None
        ]) if recent_history else 1.0
        episode_counts = np.mean([
            r.get("episodes_considered", 0) for r in recent_history if r.get("episodes_considered") is not None
        ]) if recent_history else 0.0

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
            "recent_avg_recency_lift": avg_recency_lift,
            "recent_avg_recency_weight": avg_recency_weight,
            "recent_avg_episodes_considered": episode_counts,
            "recency_lift_target_met": avg_recency_lift >= 0.2,
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
