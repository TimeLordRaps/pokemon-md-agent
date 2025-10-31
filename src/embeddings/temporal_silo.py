"""7 temporal resolution silos for hierarchical RAG system.
Changed lines & context scanned: composite index (floor, silo, ts), 7 silos, cross-floor search."""

from typing import List, Dict, Optional, Any, Tuple, Iterable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    from faiss import METRIC_INNER_PRODUCT, IO_FLAG_MMAP  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    METRIC_INNER_PRODUCT = None
    IO_FLAG_MMAP = None

DEFAULT_DECAY_FACTOR_PER_HOUR = 0.001


class SiloType(Enum):
    """Types of temporal silos."""
    TEMPORAL_1FRAME = "temporal_1frame"
    TEMPORAL_2FRAME = "temporal_2frame"
    TEMPORAL_4FRAME = "temporal_4frame"
    TEMPORAL_8FRAME = "temporal_8frame"
    TEMPORAL_16FRAME = "temporal_16frame"
    TEMPORAL_32FRAME = "temporal_32frame"
    TEMPORAL_64FRAME = "temporal_64frame"


@dataclass
class SiloConfig:
    """Configuration for a temporal silo."""
    silo_id: str
    sample_rate: int
    time_span_seconds: float
    max_entries: int = 1000
    description: str = ""


@dataclass
class SiloEntry:
    """Entry stored in a temporal silo."""
    embedding: np.ndarray
    timestamp: float
    metadata: Dict[str, Any]
    trajectory_id: str
    floor: int = 0  # Dungeon floor number
    silo: str = ""  # Silo type identifier
    similarity_score: Optional[float] = None
    episode_id: int = 0
    recency_weight: float = 1.0
    raw_similarity: Optional[float] = None

    @property
    def composite_index(self) -> Tuple[int, str, float]:
        """Composite index (floor, silo, ts) for efficient retrieval."""
        return (self.floor, self.silo, self.timestamp)


@dataclass
class EpisodeIndex:
    """Container for per-episode FAISS index state."""
    episode_id: int
    index: Any
    entries: List[SiloEntry] = field(default_factory=list)


@dataclass
class EpisodeRetrieval:
    """Result container for cross-episode retrieval."""
    entry: SiloEntry
    score: float
    episode_id: int
    context: str
    raw_similarity: float = 0.0
    recency_weight: float = 1.0


class TemporalSilo:
    """Individual temporal resolution silo."""
    
    def __init__(self, config: SiloConfig):
        """Initialize temporal silo.
        
        Args:
            config: Silo configuration
        """
        self.config = config
        self.entries: List[SiloEntry] = []
        self.last_sample_time = 0.0
        self.episode_entries: Dict[int, List[SiloEntry]] = defaultdict(list)
        self._episode_indexes: Dict[int, EpisodeIndex] = {}
        self._embedding_dim: Optional[int] = None
        self._faiss_available = faiss is not None
        
        logger.debug(
            "Created silo %s: %dms sample rate, %.1fs span, %d max entries",
            config.silo_id,
            config.sample_rate,
            config.time_span_seconds,
            config.max_entries
        )
    
    def should_sample(self, current_time: float) -> bool:
        """Check if this silo should sample at current time.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            True if silo should sample now
        """
        time_since_last = current_time - self.last_sample_time
        sample_interval = self.config.sample_rate / 1000.0  # Convert ms to seconds
        
        return time_since_last >= sample_interval
    
    def store(
        self,
        embedding: np.ndarray,
        current_time: float,
        trajectory_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        floor: int = 0,
        episode_id: int = 0,
    ) -> None:
        """Store embedding in this silo.

        Args:
            embedding: Vector embedding to store
            current_time: Current time in seconds
            trajectory_id: ID of trajectory this belongs to
            metadata: Additional metadata to store
            floor: Current dungeon floor number
            episode_id: Episode identifier for boundary-aware retrieval
        """
        if not self.should_sample(current_time):
            return

        normalized_embedding = self._prepare_embedding(embedding)

        if self._embedding_dim is None:
            self._embedding_dim = int(normalized_embedding.shape[0])

        entry_metadata = dict(metadata or {})
        entry_metadata.setdefault("floor", floor)
        entry_metadata.setdefault("episode_id", episode_id)

        entry = SiloEntry(
            embedding=normalized_embedding,
            timestamp=current_time,
            metadata=entry_metadata,
            trajectory_id=trajectory_id,
            floor=floor,
            silo=self.config.silo_id,
            episode_id=episode_id,
        )

        self.entries.append(entry)
        self.episode_entries[episode_id].append(entry)
        self._add_to_episode_index(entry)
        self.last_sample_time = current_time

        # Trim if over capacity (keep most recent)
        if len(self.entries) > self.config.max_entries:
            self._trim_to_capacity()

        logger.debug(
            "Stored entry in silo %s floor %d (total entries: %d, composite_index: %s)",
            self.config.silo_id,
            floor,
            len(self.entries),
            entry.composite_index
        )
    
    def retrieve_recent(
        self,
        time_window_seconds: float,
        current_time: float,
        limit: Optional[int] = None,
    ) -> List[SiloEntry]:
        """Retrieve entries from recent time window.
        
        Args:
            time_window_seconds: Time window to retrieve from
            current_time: Current time in seconds
            limit: Maximum number of entries to return
            
        Returns:
            List of entries within time window
        """
        cutoff_time = current_time - time_window_seconds
        
        recent_entries = [
            entry for entry in self.entries
            if entry.timestamp >= cutoff_time
        ]
        
        # Sort by timestamp (most recent first)
        recent_entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit is not None:
            recent_entries = recent_entries[:limit]
        
        return recent_entries
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        episode_ids: Optional[Iterable[int]] = None,
    ) -> List[Tuple[SiloEntry, float]]:
        """Search for similar embeddings in this silo.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            episode_ids: Optional iterable of episode IDs to restrict search
            
        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        normalized_query = self._prepare_embedding(query_embedding)
        target_episode_ids: List[int]

        if episode_ids is None:
            target_episode_ids = list(self.episode_entries.keys()) or [0]
        else:
            target_episode_ids = list(episode_ids)

        similarities: List[Tuple[SiloEntry, float]] = []

        for episode_id in target_episode_ids:
            episode_results = self._search_episode(
                episode_id=episode_id,
                query_embedding=normalized_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            similarities.extend(episode_results)

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity semantics."""
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _search_episode(
        self,
        episode_id: int,
        query_embedding: np.ndarray,
        top_k: int,
        similarity_threshold: float,
    ) -> List[Tuple[SiloEntry, float]]:
        """Search a specific episode index if available."""
        if self._faiss_available and episode_id in self._episode_indexes:
            state = self._episode_indexes[episode_id]

            if not state.entries:
                return []

            query = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = state.index.search(query, top_k)

            results: List[Tuple[SiloEntry, float]] = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                similarity = float(distance)
                if similarity < similarity_threshold:
                    continue
                entry = state.entries[idx]
                results.append((entry, similarity))
            return results

        # Fallback to numpy search
        if episode_id in self.episode_entries:
            entries = self.episode_entries[episode_id]
        elif episode_id == 0:
            entries = self.entries
        else:
            return []
        results = []
        for entry in entries:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= similarity_threshold:
                results.append((entry, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _add_to_episode_index(self, entry: SiloEntry) -> None:
        """Add entry to per-episode FAISS index if available."""
        if not self._faiss_available or self._embedding_dim is None:
            return

        state = self._get_or_create_episode_index(entry.episode_id)
        if state is None:
            return

        vector = entry.embedding.reshape(1, -1).astype(np.float32)
        state.index.add(vector)
        state.entries.append(entry)

    def _get_or_create_episode_index(self, episode_id: int) -> Optional[EpisodeIndex]:
        """Retrieve or initialize FAISS index for an episode."""
        if not self._faiss_available or self._embedding_dim is None:
            return None

        if episode_id not in self._episode_indexes:
            index = faiss.IndexFlatIP(self._embedding_dim)  # type: ignore
            self._episode_indexes[episode_id] = EpisodeIndex(
                episode_id=episode_id,
                index=index,
                entries=[],
            )

        return self._episode_indexes[episode_id]

    def _trim_to_capacity(self) -> None:
        """Trim stored entries to configured capacity with index rebuild."""
        overflow = len(self.entries) - self.config.max_entries
        if overflow <= 0:
            return

        removed = self.entries[:overflow]
        removal_ids = {id(entry) for entry in removed}
        self.entries = self.entries[-self.config.max_entries:]

        for episode_id in list(self.episode_entries.keys()):
            updated = [entry for entry in self.episode_entries[episode_id] if id(entry) not in removal_ids]
            if updated:
                self.episode_entries[episode_id] = updated
            else:
                del self.episode_entries[episode_id]
                self._episode_indexes.pop(episode_id, None)

        if self._faiss_available:
            for episode_id in list(self._episode_indexes.keys()):
                self._rebuild_episode_index(episode_id)

    def _rebuild_episode_index(self, episode_id: int) -> None:
        """Rebuild FAISS index for an episode after removals."""
        if not self._faiss_available or self._embedding_dim is None:
            return

        entries = self.episode_entries.get(episode_id)
        if not entries:
            self._episode_indexes.pop(episode_id, None)
            return

        index = faiss.IndexFlatIP(self._embedding_dim)  # type: ignore
        vectors = np.stack([entry.embedding.astype(np.float32) for entry in entries])
        index.add(vectors)  # type: ignore
        self._episode_indexes[episode_id] = EpisodeIndex(
            episode_id=episode_id,
            index=index,
            entries=list(entries),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this silo.
        
        Returns:
            Dictionary with silo statistics
        """
        if not self.entries:
            time_span = 0.0
        else:
            time_span = max(entry.timestamp for entry in self.entries) - \
                       min(entry.timestamp for entry in self.entries)
        
        return {
            "silo_id": self.config.silo_id,
            "total_entries": len(self.entries),
            "max_capacity": self.config.max_entries,
            "actual_time_span": time_span,
            "configured_time_span": self.config.time_span_seconds,
            "sample_rate_ms": self.config.sample_rate,
            "utilization": len(self.entries) / self.config.max_entries,
        }
    
    def clear(self) -> None:
        """Clear all entries from this silo."""
        self.entries.clear()
        self.last_sample_time = 0.0
        logger.debug("Cleared silo %s", self.config.silo_id)


class TemporalSiloManager:
    """Manager for 7 temporal resolution silos."""
    
    def __init__(
        self,
        base_fps: int = 30,
        silos: Optional[List[int]] = None,
        decay_factor_per_hour: float = DEFAULT_DECAY_FACTOR_PER_HOUR,
    ):
        """Initialize temporal silo manager.
        
        Args:
            base_fps: Base framerate for timing calculations
            silos: List of frame intervals for silos
            decay_factor_per_hour: Recency decay applied during similarity scoring
        """
        if decay_factor_per_hour < 0:
            raise ValueError(
                f"decay_factor_per_hour must be non-negative, got {decay_factor_per_hour}"
            )

        self.base_fps = base_fps
        self.decay_factor_per_hour = decay_factor_per_hour
        
        # Default 7 temporal silos
        if silos is None:
            silos = [1, 2, 4, 8, 16, 32, 64]
        
        self.silos: Dict[str, TemporalSilo] = {}
        self._create_silos(silos)
        self._episode_counter = 0
        self._current_episode_id: Optional[int] = None
        self._episode_start_times: Dict[int, float] = {}
        self._episode_last_activity: Dict[int, float] = {}
        self._episode_order: List[int] = []
        self._last_floor: Optional[int] = None
        self._last_event: Optional[str] = None
        
        logger.info(
            "Created %d temporal silos: %s",
            len(self.silos),
            list(self.silos.keys())
        )
    
    def _create_silos(self, frame_intervals: List[int]) -> None:
        """Create silos with specified frame intervals.
        
        Args:
            frame_intervals: List of frame intervals (1, 2, 4, 8, 16, 32, 64)
        """
        # Define silo configurations
        silo_configs = [
            SiloConfig(
                silo_id="temporal_1frame",
                sample_rate=1000 // self.base_fps,  # Every frame
                time_span_seconds=4.0,
                description="Immediate (0-4 sec)"
            ),
            SiloConfig(
                silo_id="temporal_2frame", 
                sample_rate=2000 // self.base_fps,  # Every 2nd frame
                time_span_seconds=8.0,
                description="Combat (0-8 sec)"
            ),
            SiloConfig(
                silo_id="temporal_4frame",
                sample_rate=4000 // self.base_fps,  # Every 4th frame
                time_span_seconds=16.0,
                description="Navigation (0-16 sec)"
            ),
            SiloConfig(
                silo_id="temporal_8frame",
                sample_rate=8000 // self.base_fps,  # Every 8th frame
                time_span_seconds=32.0,
                description="Room explore (0-32 sec)"
            ),
            SiloConfig(
                silo_id="temporal_16frame",
                sample_rate=16000 // self.base_fps,  # Every 16th frame
                time_span_seconds=64.0,
                description="Floor strategy (0-64 sec)"
            ),
            SiloConfig(
                silo_id="temporal_32frame",
                sample_rate=32000 // self.base_fps,  # Every 32nd frame
                time_span_seconds=128.0,
                description="Long planning (0-128 sec)"
            ),
            SiloConfig(
                silo_id="temporal_64frame",
                sample_rate=64000 // self.base_fps,  # Every 64th frame
                time_span_seconds=256.0,
                description="Cross-floor (2+ min)"
            ),
        ]
        
        # Create silos for specified intervals
        for interval in frame_intervals:
            config = silo_configs[interval.bit_length() - 1]  # 1->0, 2->1, 4->2, etc.
            
            # Adjust sample rate for custom intervals
            if interval not in [1, 2, 4, 8, 16, 32, 64]:
                config.sample_rate = (interval * 1000) // self.base_fps
            
            self.silos[config.silo_id] = TemporalSilo(config)
    
    def _resolve_episode(
        self,
        requested_episode: Optional[int],
        metadata: Dict[str, Any],
        floor: int,
        current_time: float,
    ) -> int:
        """Resolve the effective episode identifier for storage."""
        if requested_episode is not None:
            self._register_episode_start_if_needed(requested_episode, current_time)
            self._current_episode_id = requested_episode
            return requested_episode

        metadata_episode = metadata.get("episode_id")
        if isinstance(metadata_episode, int):
            self._register_episode_start_if_needed(metadata_episode, current_time)
            self._current_episode_id = metadata_episode
            return metadata_episode

        return self._ensure_episode_started(current_time, floor_hint=floor)

    def _register_episode_start_if_needed(self, episode_id: int, current_time: float) -> None:
        """Register episode metadata if encountering it for the first time."""
        if episode_id not in self._episode_start_times:
            self._episode_start_times[episode_id] = current_time
            self._episode_last_activity[episode_id] = current_time
            if episode_id not in self._episode_order:
                self._episode_order.append(episode_id)

    def _start_new_episode(
        self,
        reason: str,
        current_time: float,
        floor_hint: Optional[int] = None,
    ) -> int:
        """Begin a new episode and record tracking metadata."""
        self._episode_counter += 1
        episode_id = self._episode_counter
        self._current_episode_id = episode_id
        self._register_episode_start_if_needed(episode_id, current_time)
        if floor_hint is not None:
            self._last_floor = floor_hint
        logger.info("Started new episode %d (reason=%s)", episode_id, reason)
        return episode_id

    def _ensure_episode_started(
        self,
        current_time: float,
        floor_hint: Optional[int] = None,
    ) -> int:
        """Ensure there is an active episode ID."""
        if self._current_episode_id is None:
            return self._start_new_episode("bootstrap", current_time, floor_hint=floor_hint)

        self._register_episode_start_if_needed(self._current_episode_id, current_time)
        if floor_hint is not None:
            self._last_floor = floor_hint
        return self._current_episode_id

    def _record_episode_activity(self, episode_id: int, current_time: float) -> None:
        """Record latest activity timestamp for an episode."""
        self._register_episode_start_if_needed(episode_id, current_time)
        self._episode_last_activity[episode_id] = current_time

    def _handle_floor_change_episode(
        self,
        floor: int,
        metadata: Dict[str, Any],
        current_time: float,
    ) -> int:
        """Determine whether a floor change should trigger a new episode."""
        boundary_flag = bool(metadata.get("episode_boundary"))
        transition_hint = metadata.get("floor_transition_type")

        if boundary_flag or transition_hint == "episode_start":
            return self._start_new_episode("floor_flagged", current_time, floor_hint=floor)

        previous_floor = self._last_floor
        self._last_floor = floor

        if previous_floor is None:
            return self._ensure_episode_started(current_time, floor_hint=floor)

        if floor <= 1 and previous_floor > floor:
            return self._start_new_episode("floor_reset", current_time, floor_hint=floor)

        if floor < previous_floor:
            return self._start_new_episode("floor_regression", current_time, floor_hint=floor)

        return self._ensure_episode_started(current_time, floor_hint=floor)

    def add_with_episode_boundary(
        self,
        embedding: np.ndarray,
        trajectory_id: str,
        silo_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
        floor: Optional[int] = None,
        on_device_buffer: Optional[Any] = None,
    ) -> int:
        """Store embedding with automatic episode boundary detection.

        Args:
            embedding: Embedding vector to store
            trajectory_id: Unique trajectory identifier
            silo_id: Optional silo restriction
            metadata: Additional metadata describing the frame
            current_time: Timestamp override (defaults to `time.time()`)
            floor: Explicit floor number (fallback to metadata)
            on_device_buffer: Optional on-device buffer for dual writes

        Returns:
            Episode identifier used for storage.
        """
        if current_time is None:
            current_time = time.time()

        payload_metadata = dict(metadata or {})
        event = payload_metadata.get("event") or payload_metadata.get("trigger")
        resolved_floor = floor if floor is not None else int(payload_metadata.get("floor", 0))

        savestate_flag = bool(
            payload_metadata.get("savestate_loaded")
            or payload_metadata.get("restored_from_savestate")
            or event == "savestate_loaded"
        )

        if savestate_flag:
            episode_id = self._start_new_episode("savestate_load", current_time, floor_hint=resolved_floor)
        elif event == "on_floor_change":
            episode_id = self._handle_floor_change_episode(resolved_floor, payload_metadata, current_time)
        elif payload_metadata.get("episode_boundary"):
            episode_id = self._start_new_episode("metadata_boundary", current_time, floor_hint=resolved_floor)
        else:
            episode_id = self._ensure_episode_started(current_time, floor_hint=resolved_floor)

        payload_metadata["event"] = event
        self._last_event = event

        self.store(
            embedding=embedding,
            trajectory_id=trajectory_id,
            silo_id=silo_id,
            metadata=payload_metadata,
            current_time=current_time,
            floor=resolved_floor,
            episode_id=episode_id,
            on_device_buffer=on_device_buffer,
        )

        return episode_id
    
    def store(
        self,
        embedding: np.ndarray,
        trajectory_id: str,
        silo_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
        floor: Optional[int] = None,
        episode_id: Optional[int] = None,
        on_device_buffer: Optional[Any] = None,  # OnDeviceBufferManager
    ) -> None:
        """Store embedding in appropriate silo(s) with floor tracking.

        Args:
            embedding: Vector embedding to store
            trajectory_id: ID of trajectory this belongs to
            silo_id: Specific silo ID (auto-select if None)
            metadata: Additional metadata
            current_time: Current time (uses time.time() if None)
            floor: Current dungeon floor number
            episode_id: Override episode ID (auto-detected when None)
            on_device_buffer: Optional on-device buffer manager for dual storage
        """
        if current_time is None:
            current_time = time.time()

        resolved_metadata = dict(metadata or {})
        resolved_floor = floor if floor is not None else int(resolved_metadata.get("floor", 0))
        resolved_episode_id = self._resolve_episode(
            requested_episode=episode_id,
            metadata=resolved_metadata,
            floor=resolved_floor,
            current_time=current_time,
        )
        resolved_metadata["floor"] = resolved_floor
        resolved_metadata["episode_id"] = resolved_episode_id

        if silo_id is not None:
            # Store in specific silo
            if silo_id in self.silos:
                self.silos[silo_id].store(
                    embedding,
                    current_time,
                    trajectory_id,
                    resolved_metadata,
                    resolved_floor,
                    episode_id=resolved_episode_id,
                )
            else:
                logger.warning("Unknown silo ID: %s", silo_id)
        else:
            # Auto-select silos based on temporal resolution needs
            # Store in multiple relevant silos for hierarchical retrieval
            for silo in self.silos.values():
                if silo.should_sample(current_time):
                    silo.store(
                        embedding,
                        current_time,
                        trajectory_id,
                        resolved_metadata,
                        resolved_floor,
                        episode_id=resolved_episode_id,
                    )

        self._record_episode_activity(resolved_episode_id, current_time)
        if resolved_floor:
            self._last_floor = resolved_floor

        # Store in on-device buffer if available
        if on_device_buffer is not None:
            import asyncio
            # Run async operation in background with composite index metadata
            asyncio.create_task(
                on_device_buffer.store_embedding(
                    embedding=embedding,
                    metadata={
                        **resolved_metadata,
                        "trajectory_id": trajectory_id,
                        "silo_stored": silo_id,
                        "floor": resolved_floor,
                        "episode_id": resolved_episode_id,
                        "composite_index": (resolved_floor, silo_id, current_time),
                    }
                )
            )
    
    def cross_silo_search(
        self,
        query_embedding: np.ndarray,
        silo_ids: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, List[Tuple[SiloEntry, float]]]:
        """Search across multiple silos for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            silo_ids: List of silos to search (searches all if None)
            top_k: Number of results per silo
            
        Returns:
            Dictionary mapping silo_id to list of (entry, similarity) tuples
        """
        silos_to_search = silo_ids or list(self.silos.keys())
        results = {}
        
        for silo_id in silos_to_search:
            if silo_id in self.silos:
                similar_entries = self.silos[silo_id].search_similar(
                    query_embedding,
                    top_k=top_k
                )
                results[silo_id] = similar_entries
        
        logger.debug(
            "Cross-silo search across %d silos, found %d total matches",
            len(silos_to_search),
            sum(len(matches) for matches in results.values())
        )
        
        return results

    def search_with_decay(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        silo_ids: Optional[List[str]] = None,
        decay_factor: Optional[float] = None,
        current_time: Optional[float] = None,
        episode_ids: Optional[Iterable[int]] = None,
    ) -> List[SiloEntry]:
        """Search silos with recency-aware scoring.

        Args:
            query_embedding: Embedding used as query vector
            top_k: Maximum results to return
            silo_ids: Optional silo whitelist
            decay_factor: Overrides default per-hour decay factor
            current_time: Timestamp override (defaults to now)
            episode_ids: Optional filter restricting results to specific episodes

        Returns:
            List of SiloEntry objects with updated similarity_score values.
        """
        if current_time is None:
            current_time = time.time()

        effective_decay = decay_factor if decay_factor is not None else self.decay_factor_per_hour
        if effective_decay < 0:
            raise ValueError(f"decay_factor must be non-negative, got {effective_decay}")
        silos_to_search = silo_ids or list(self.silos.keys())
        scored_entries: Dict[Tuple[str, str, float], SiloEntry] = {}

        for silo_id in silos_to_search:
            silo = self.silos.get(silo_id)
            if silo is None:
                continue

            matches = silo.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                episode_ids=episode_ids,
            )

            for entry, similarity in matches:
                hours_delta = (entry.timestamp - current_time) / 3600.0
                recency_weight = max(0.0, 1.0 + (effective_decay * hours_delta))
                adjusted_score = similarity * recency_weight

                entry.recency_weight = recency_weight
                entry.raw_similarity = similarity
                entry.similarity_score = adjusted_score

                key = (entry.trajectory_id, entry.silo, entry.timestamp)
                stored_entry = scored_entries.get(key)

                if stored_entry is None or (stored_entry.similarity_score or 0.0) < adjusted_score:
                    scored_entries[key] = entry

        ranked_entries = sorted(
            scored_entries.values(),
            key=lambda item: item.similarity_score or 0.0,
            reverse=True,
        )

        return ranked_entries[:top_k]

    def search_across_episodes(
        self,
        query_embedding: np.ndarray,
        top_k_per_episode: int = 3,
        max_episodes: int = 3,
        silo_ids: Optional[List[str]] = None,
        decay_factor: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> List[EpisodeRetrieval]:
        """Search recent episodes and re-rank results globally.

        Args:
            query_embedding: Query embedding vector
            top_k_per_episode: Results to retain per episode before re-ranking
            max_episodes: Maximum number of episodes to consider
            silo_ids: Optional silo whitelist
            decay_factor: Optional override for decay factor
            current_time: Timestamp override for recency calculations

        Returns:
            List of EpisodeRetrieval objects sorted by decayed similarity.
        """
        if current_time is None:
            current_time = time.time()

        if not self._episode_last_activity:
            return []

        recent_episode_pairs = sorted(
            self._episode_last_activity.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        selected_episode_ids = [episode_id for episode_id, _ in recent_episode_pairs[:max_episodes]]

        aggregated_results: List[EpisodeRetrieval] = []

        for episode_id in selected_episode_ids:
            episode_entries = self.search_with_decay(
                query_embedding=query_embedding,
                top_k=top_k_per_episode,
                silo_ids=silo_ids,
                decay_factor=decay_factor,
                current_time=current_time,
                episode_ids=[episode_id],
            )

            for entry in episode_entries:
                score = entry.similarity_score or 0.0
                aggregated_results.append(
                    EpisodeRetrieval(
                        entry=entry,
                        score=score,
                        episode_id=episode_id,
                        context=f"From episode {episode_id}",
                        raw_similarity=entry.raw_similarity or score,
                        recency_weight=entry.recency_weight,
                    )
                )

        aggregated_results.sort(key=lambda item: item.score, reverse=True)
        return aggregated_results
    
    def get_recent_trajectories(
        self,
        time_window_seconds: float = 30.0,
        silo_ids: Optional[List[str]] = None,
        limit_per_silo: int = 5,
        floor: Optional[int] = None,
        top_k: int = 3,
    ) -> Dict[str, List[SiloEntry]]:
        """Get recent trajectories with deduplication and recency bias.

        Args:
            time_window_seconds: Time window to retrieve from
            silo_ids: Silos to search (searches all if None)
            limit_per_silo: Maximum entries per silo
            floor: Optional floor filter (returns all floors if None)
            top_k: Final number of trajectories to return after dedup and recency bias

        Returns:
            Dictionary mapping silo_id to list of entries (top_k total after processing)
        """
        silos_to_search = silo_ids or list(self.silos.keys())
        all_entries = []
        current_time = time.time()

        # Collect entries from all silos
        for silo_id in silos_to_search:
            if silo_id in self.silos:
                recent_entries = self.silos[silo_id].retrieve_recent(
                    time_window_seconds,
                    current_time,
                    limit=limit_per_silo * 2  # Get more for dedup
                )

                # Filter by floor if specified
                if floor is not None:
                    recent_entries = [entry for entry in recent_entries if entry.floor == floor]

                all_entries.extend(recent_entries)

        # Deduplicate by trajectory_id (keep highest similarity score)
        trajectory_map = {}
        for entry in all_entries:
            tid = entry.trajectory_id
            if tid not in trajectory_map or entry.similarity_score > trajectory_map[tid].similarity_score:
                trajectory_map[tid] = entry

        deduped_entries = list(trajectory_map.values())

        # Apply recency bias (exponential decay based on age)
        recency_decay_rate = 0.001  # Configurable decay rate
        for entry in deduped_entries:
            age_seconds = current_time - entry.timestamp
            recency_weight = np.exp(-recency_decay_rate * age_seconds)
            # Store recency-adjusted score
            entry.similarity_score = (entry.similarity_score or 0.0) * recency_weight

        # Sort by recency-adjusted score and return top_k
        deduped_entries.sort(key=lambda e: e.similarity_score or 0.0, reverse=True)
        final_entries = deduped_entries[:top_k]

        # Group by silo for return format
        results = {}
        for entry in final_entries:
            silo_id = entry.silo
            if silo_id not in results:
                results[silo_id] = []
            results[silo_id].append(entry)

        return results

    def search_by_composite_index(
        self,
        floor: Optional[int] = None,
        silo: Optional[str] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
        limit: int = 100,
    ) -> List[SiloEntry]:
        """Search entries using composite index (floor, silo, ts).

        Args:
            floor: Optional floor filter
            silo: Optional silo filter
            min_timestamp: Optional minimum timestamp
            max_timestamp: Optional maximum timestamp
            limit: Maximum results to return

        Returns:
            List of matching entries sorted by composite index
        """
        all_matching_entries = []

        for silo_obj in self.silos.values():
            for entry in silo_obj.entries:
                # Apply filters
                if floor is not None and entry.floor != floor:
                    continue
                if silo is not None and entry.silo != silo:
                    continue
                if min_timestamp is not None and entry.timestamp < min_timestamp:
                    continue
                if max_timestamp is not None and entry.timestamp > max_timestamp:
                    continue

                all_matching_entries.append(entry)

        # Sort by composite index (floor, silo, timestamp)
        all_matching_entries.sort(key=lambda e: e.composite_index)

        return all_matching_entries[:limit]
    
    def get_silo_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all silos.
        
        Returns:
            Dictionary mapping silo_id to statistics dict
        """
        return {
            silo_id: silo.get_stats()
            for silo_id, silo in self.silos.items()
        }
    
    def clear_all_silos(self) -> None:
        """Clear all data from all silos."""
        for silo in self.silos.values():
            silo.clear()
        
        logger.info("Cleared all temporal silos")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        total_entries = sum(len(silo.entries) for silo in self.silos.values())
        total_capacity = sum(silo.config.max_entries for silo in self.silos.values())
        
        return {
            "total_entries": total_entries,
            "total_capacity": total_capacity,
            "overall_utilization": total_entries / total_capacity if total_capacity > 0 else 0.0,
            "per_silo_usage": {
                silo_id: len(silo.entries)
                for silo_id, silo in self.silos.items()
            }
        }
