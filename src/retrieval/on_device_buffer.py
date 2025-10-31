"""Simple on-device buffer with TTL-based eviction and stuckness detection.

OnDeviceBuffer provides a minimal interface for on-device retrieval with circular buffer storage,
cosine similarity search, TTL/capacity-based pruning, and micro stuckness detection.
The buffer maintains a ~60-minute window with automatic eviction and tracks recent queries
to detect when the agent may be stuck in repetitive behavior patterns.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import threading
import numpy as np
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a similarity search operation."""
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    entry_id: Optional[str] = None


@dataclass
class BufferEntry:
    """Entry stored in the on-device buffer."""
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    id: Optional[str] = None


class OnDeviceBuffer:
    """Simple on-device buffer with TTL, search, and stuckness detection.

    Provides a minimal interface coordinating ring buffer slots with ~60-min TTL window,
    cosine similarity search, capacity/TTL-based pruning, and micro stuckness detection
    based on recent query patterns.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_minutes: int = 60,
        stuckness_threshold: float = 0.8,
        stuckness_window: int = 3,
    ):
        """Initialize on-device buffer.

        Args:
            max_entries: Maximum number of entries to store
            ttl_minutes: TTL in minutes for entries
            stuckness_threshold: Similarity threshold for stuckness detection
            stuckness_window: Number of recent queries to consider for stuckness
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_minutes * 60
        self.stuckness_threshold = stuckness_threshold
        self.stuckness_window = stuckness_window

        # Thread-safe storage
        self._buffer: deque[BufferEntry] = deque(maxlen=max_entries)
        self._lock = threading.RLock()

        # Stuckness tracking
        self._recent_queries: deque[np.ndarray] = deque(maxlen=stuckness_window * 2)
        self._stuckness_score = 0.0

        logger.info(
            "Initialized OnDeviceBuffer: max_entries=%d, ttl=%dmin, stuckness_threshold=%.2f",
            max_entries, ttl_minutes, stuckness_threshold
        )

    def store(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store embedding with metadata in buffer.

        Args:
            embedding: Embedding vector to store
            metadata: Associated metadata dictionary

        Returns:
            True if stored successfully, False otherwise

        Raises:
            ValueError: If embedding or metadata is invalid
        """
        if not self._validate_embedding(embedding):
            raise ValueError("Invalid embedding: must be numpy array with finite float32 values")

        if not isinstance(metadata, dict):
            raise ValueError("Invalid metadata: must be dictionary")

        # Ensure metadata is serializable
        try:
            # Basic serialization check - try to JSON serialize
            import json
            json.dumps(metadata)
        except (TypeError, ValueError):
            raise ValueError("Invalid metadata: must contain serializable values")

        entry = BufferEntry(
            embedding=embedding.astype(np.float32),
            metadata=metadata.copy(),
            timestamp=metadata.get("timestamp", time.time()),
        )

        with self._lock:
            self._buffer.append(entry)
            logger.debug("Stored entry with metadata keys: %s", list(metadata.keys()))

        return True

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results to return

        Returns:
            List of search results ordered by similarity (descending)
        """
        if not self._validate_embedding(query_embedding):
            logger.warning("Invalid query embedding, returning empty results")
            return []

        # Track query for stuckness detection
        with self._lock:
            self._recent_queries.append(query_embedding.copy())
            self._update_stuckness_score()

        # Perform search
        results = []
        with self._lock:
            for entry in self._buffer:
                # Skip expired entries
                if time.time() - entry.timestamp > self.ttl_seconds:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity > 0:  # Only include positive similarities
                    results.append(SearchResult(
                        score=float(similarity),
                        metadata=entry.metadata.copy(),
                        embedding=entry.embedding.copy(),
                        entry_id=getattr(entry, 'id', None),
                    ))

        # Sort by similarity descending and return top-k
        results.sort(key=lambda r: r.score, reverse=True)
        final_results = results[:top_k]

        # Log cross-silo delegation stub
        logger.debug("Cross-silo delegation stub logged for potential ANN search")
        logger.debug("Search completed: %d results from %d total entries", len(final_results), len(self._buffer))

        return final_results

    def prune(self, by_time: bool = True, by_capacity: bool = False, max_entries: Optional[int] = None) -> int:
        """Prune entries based on TTL and/or capacity constraints.

        Args:
            by_time: If True, remove entries older than TTL
            by_capacity: If True, reduce to capacity limit (removes oldest)
            max_entries: Override max_entries for this prune operation

        Returns:
            Number of entries removed
        """
        removed_count = 0
        current_time = time.time()
        capacity_limit = max_entries if max_entries is not None else self.max_entries
        # Auto-enable capacity pruning if max_entries is specified
        enable_capacity_prune = by_capacity or (max_entries is not None)

        with self._lock:
            if by_time:
                # Remove expired entries
                original_len = len(self._buffer)
                self._buffer = deque(
                    [entry for entry in self._buffer if current_time - entry.timestamp <= self.ttl_seconds],
                    maxlen=self.max_entries
                )
                removed_count += original_len - len(self._buffer)

            if enable_capacity_prune and len(self._buffer) > capacity_limit:
                # Remove oldest entries to fit capacity
                excess = len(self._buffer) - capacity_limit
                for _ in range(excess):
                    self._buffer.popleft()
                removed_count += excess
                # Reconstruct deque with proper maxlen after manual removal
                self._buffer = deque(self._buffer, maxlen=self.max_entries)

        logger.debug("Pruned %d entries (time=%s, capacity=%s)", removed_count, by_time, enable_capacity_prune)
        return removed_count

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics including stuckness flag.

        Returns:
            Dictionary with buffer metrics and stuckness information
        """
        current_time = time.time()

        with self._lock:
            # Basic metrics
            total_entries = len(self._buffer)
            total_size_bytes = sum(entry.embedding.nbytes + len(str(entry.metadata).encode('utf-8')) for entry in self._buffer)

            # Age statistics
            if total_entries > 0:
                ages = [current_time - entry.timestamp for entry in self._buffer]
                avg_entry_age_seconds = sum(ages) / len(ages)
            else:
                avg_entry_age_seconds = 0.0

            # Capacity utilization
            capacity_utilization = total_entries / self.max_entries if self.max_entries > 0 else 0.0

            # Stuckness metrics
            is_stuck = self._stuckness_score >= self.stuckness_threshold

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size_bytes,
            "avg_entry_age_seconds": avg_entry_age_seconds,
            "capacity_utilization": capacity_utilization,
            "stuckness_score": self._stuckness_score,
            "is_stuck": is_stuck,
            "max_entries": self.max_entries,
            "ttl_minutes": self.ttl_seconds / 60,
            "stuckness_threshold": self.stuckness_threshold,
            "stuckness_window": self.stuckness_window,
        }

    def is_stuck(self) -> bool:
        """Check if buffer detects stuckness based on recent query patterns.

        Returns:
            True if stuckness score exceeds threshold
        """
        return self._stuckness_score >= self.stuckness_threshold

    def _update_stuckness_score(self) -> None:
        """Update stuckness score based on recent query similarity patterns."""
        if len(self._recent_queries) < self.stuckness_window:
            self._stuckness_score = 0.0
            return

        # Calculate average similarity between recent queries
        similarities = []
        recent_queries = list(self._recent_queries)[-self.stuckness_window:]

        for i in range(len(recent_queries)):
            for j in range(i + 1, len(recent_queries)):
                sim = self._cosine_similarity(recent_queries[i], recent_queries[j])
                similarities.append(sim)

        if similarities:
            self._stuckness_score = sum(similarities) / len(similarities)
        else:
            self._stuckness_score = 0.0

        logger.debug("Updated stuckness score: %.3f (threshold: %.3f)", self._stuckness_score, self.stuckness_threshold)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _validate_embedding(self, embedding: Any) -> bool:
        """Validate embedding array."""
        if not isinstance(embedding, np.ndarray):
            return False

        if embedding.dtype != np.float32:
            return False

        if not np.all(np.isfinite(embedding)):
            return False

        if embedding.ndim != 1:
            return False

        return True