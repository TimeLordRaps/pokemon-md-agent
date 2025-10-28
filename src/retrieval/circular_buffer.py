"""Circular buffer for on-device memory management."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import threading
import time
import logging
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """Entry in the circular buffer."""
    id: str
    data: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    priority: float = 1.0


class CircularBuffer:
    """Thread-safe circular buffer with fixed memory size."""

    def __init__(
        self,
        max_size_mb: float = 50.0,
        entry_size_estimate: int = 1024,  # bytes per entry estimate
        max_entries: Optional[int] = None,
        enable_async: bool = True,
    ):
        """Initialize circular buffer.

        Args:
            max_size_mb: Maximum memory usage in MB
            entry_size_estimate: Estimated bytes per entry
            max_entries: Override max entries (None = calculate from size)
            enable_async: Enable async operations
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)

        if max_entries is None:
            self.max_entries = max(100, self.max_size_bytes // entry_size_estimate)
        else:
            self.max_entries = max_entries

        self.buffer: deque[BufferEntry] = deque(maxlen=self.max_entries)
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        self._enable_async = enable_async

        # Memory monitoring
        self._memory_stats = {
            'peak_usage_bytes': 0,
            'total_added': 0,
            'total_evicted': 0,
            'avg_entry_size': 0,
        }

        logger.info(
            "Initialized CircularBuffer: max_size=%.1fMB (%d entries), async=%s",
            max_size_mb, self.max_entries, enable_async
        )

    def add_entry(self, entry: BufferEntry) -> bool:
        """Add entry to buffer, evicting old entries if necessary.

        Args:
            entry: Entry to add

        Returns:
            True if added successfully
        """
        with self._lock:
            try:
                entry_size = self._estimate_entry_size(entry)

                # Evict entries if needed
                while self._current_size_bytes + entry_size > self.max_size_bytes and self.buffer:
                    evicted = self.buffer.popleft()
                    evicted_size = self._estimate_entry_size(evicted)
                    self._current_size_bytes -= evicted_size
                    self._memory_stats['total_evicted'] += 1

                # Add new entry
                if self._current_size_bytes + entry_size <= self.max_size_bytes:
                    self.buffer.append(entry)
                    self._current_size_bytes += entry_size
                    self._memory_stats['total_added'] += 1

                    # Update stats
                    self._memory_stats['peak_usage_bytes'] = max(
                        self._memory_stats['peak_usage_bytes'],
                        self._current_size_bytes
                    )

                    return True
                else:
                    logger.warning("Entry too large for buffer: %d bytes", entry_size)
                    return False

            except Exception as e:
                logger.error("Failed to add entry: %s", e)
                return False

    async def add_entry_async(self, entry: BufferEntry) -> bool:
        """Async version of add_entry."""
        if not self._enable_async:
            return self.add_entry(entry)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.add_entry, entry)

    def get_entries(
        self,
        limit: Optional[int] = None,
        min_priority: float = 0.0,
        time_window: Optional[float] = None,
    ) -> List[BufferEntry]:
        """Get entries from buffer with optional filtering.

        Args:
            limit: Maximum number of entries to return
            min_priority: Minimum priority threshold
            time_window: Only entries from last N seconds

        Returns:
            List of matching entries
        """
        with self._lock:
            try:
                current_time = time.time()
                entries = []

                for entry in self.buffer:
                    if entry.priority < min_priority:
                        continue

                    if time_window and (current_time - entry.timestamp) > time_window:
                        continue

                    entries.append(entry)

                    if limit and len(entries) >= limit:
                        break

                return entries

            except Exception as e:
                logger.error("Failed to get entries: %s", e)
                return []

    async def get_entries_async(
        self,
        limit: Optional[int] = None,
        min_priority: float = 0.0,
        time_window: Optional[float] = None,
    ) -> List[BufferEntry]:
        """Async version of get_entries."""
        if not self._enable_async:
            return self.get_entries(limit, min_priority, time_window)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_entries, limit, min_priority, time_window
        )

    def search_similar(
        self,
        query_data: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[BufferEntry, float]]:
        """Search for similar entries using cosine similarity.

        Args:
            query_data: Query data vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of (entry, similarity_score) tuples
        """
        with self._lock:
            try:
                results = []

                for entry in self.buffer:
                    similarity = self._cosine_similarity(query_data, entry.data)

                    if similarity >= similarity_threshold:
                        results.append((entry, similarity))

                # Sort by similarity (descending) and return top_k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]

            except Exception as e:
                logger.error("Failed to search similar: %s", e)
                return []

    async def search_similar_async(
        self,
        query_data: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[BufferEntry, float]]:
        """Async version of search_similar."""
        if not self._enable_async:
            return self.search_similar(query_data, top_k, similarity_threshold)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.search_similar, query_data, top_k, similarity_threshold
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            return {
                'current_usage_bytes': self._current_size_bytes,
                'current_usage_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_bytes': self.max_size_bytes,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'current_entries': len(self.buffer),
                'max_entries': self.max_entries,
                'utilization_percent': (self._current_size_bytes / self.max_size_bytes) * 100,
                **self._memory_stats,
            }

    def clear(self) -> None:
        """Clear all entries from buffer."""
        with self._lock:
            self.buffer.clear()
            self._current_size_bytes = 0
            self._memory_stats = {
                'peak_usage_bytes': 0,
                'total_added': 0,
                'total_evicted': 0,
                'avg_entry_size': 0,
            }
            logger.info("Cleared circular buffer")

    def _estimate_entry_size(self, entry: BufferEntry) -> int:
        """Estimate memory size of an entry in bytes."""
        try:
            # Data size
            data_size = entry.data.nbytes if hasattr(entry.data, 'nbytes') else len(entry.data) * 8

            # Metadata size (rough estimate)
            metadata_size = len(str(entry.metadata).encode('utf-8'))

            # Overhead
            overhead = 256  # Python object overhead

            total = data_size + metadata_size + overhead
            return total

        except Exception:
            # Fallback estimate
            return 1024

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0