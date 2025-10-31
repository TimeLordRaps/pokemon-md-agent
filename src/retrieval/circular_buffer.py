"""Circular buffer for on-device memory management."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import threading
import time
import logging
import json
import os
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
    is_keyframe: bool = False


class CircularBuffer:
    """Thread-safe circular buffer with 60-minute rolling window."""

    def __init__(
        self,
        window_seconds: float = 3600.0,  # 60 minutes
        max_entries: Optional[int] = None,
        enable_async: bool = True,
        keyframe_window_multiplier: float = 3.0,  # Keep keyframes 3x longer
    ):
        """Initialize circular buffer with time-based rolling window.

        Args:
            window_seconds: Rolling window duration in seconds (default 3600 = 60 minutes)
            max_entries: Maximum number of entries (None = 108000 for 30 FPS * 60 min)
            enable_async: Enable async operations
            keyframe_window_multiplier: Multiplier for keyframe retention window
        """
        self.window_seconds = window_seconds
        self.keyframe_window_multiplier = keyframe_window_multiplier

        if max_entries is None:
            # Assume ~30 FPS: 30 * 60 * 60 = 108,000 frames per hour
            self.max_entries = int(30 * 60 * (window_seconds / 60))
        else:
            self.max_entries = max_entries

        self.buffer: deque[BufferEntry] = deque(maxlen=self.max_entries)
        self._lock = threading.RLock()
        self._enable_async = enable_async

        # Keyframe tracking
        self._last_floor: Optional[int] = None
        self._last_combat_state: Optional[bool] = None
        self._last_inventory: Optional[Dict[str, int]] = None

        # Stats (adapted for time-based)
        self._memory_stats = {
            'total_added': 0,
            'total_evicted': 0,
            'keyframes_added': 0,
        }

        logger.info(
            "Initialized CircularBuffer: window=%.1fs (%d max entries), keyframe_mult=%.1f, async=%s",
            window_seconds, self.max_entries, keyframe_window_multiplier, enable_async
        )

    def add_entry(self, entry: BufferEntry) -> bool:
        """Add entry to buffer, evicting old entries if necessary to maintain time window.

        Args:
            entry: Entry to add

        Returns:
            True if added successfully
        """
        with self._lock:
            try:
                current_time = time.time()

                # Evict entries older than the rolling window, but preserve keyframes longer
                while self.buffer:
                    age = current_time - self.buffer[0].timestamp
                    max_age = self.keyframe_window_multiplier * self.window_seconds if self.buffer[0].is_keyframe else self.window_seconds
                    if age > max_age:
                        evicted = self.buffer.popleft()
                        self._memory_stats['total_evicted'] += 1
                    else:
                        break

                # Add new entry if within window
                if len(self.buffer) < self.max_entries:
                    self.buffer.append(entry)
                    self._memory_stats['total_added'] += 1
                    if entry.is_keyframe:
                        self._memory_stats['keyframes_added'] += 1
                    return True
                else:
                    logger.warning("Buffer full, could not add entry")
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

    def add_frame(self, frame_data: np.ndarray, timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None, is_keyframe: bool = False) -> bool:
        """Add a frame to the buffer with automatic timestamp.

        Args:
            frame_data: Frame data (numpy array)
            timestamp: Frame timestamp (current time if None)
            metadata: Additional metadata for the frame
            is_keyframe: Whether this frame is a keyframe

        Returns:
            True if added successfully
        """
        if timestamp is None:
            timestamp = time.time()

        if metadata is None:
            metadata = {}

        entry = BufferEntry(
            id=f"frame_{timestamp}",
            data=frame_data,
            metadata=metadata,
            timestamp=timestamp,
            priority=2.0 if is_keyframe else 1.0,  # Higher priority for keyframes
            is_keyframe=is_keyframe
        )

        success = self.add_entry(entry)
        if success and is_keyframe:
            logger.info("Added keyframe: %s", entry.id)
        return success

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get current buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            return {
                'current_entries': len(self.buffer),
                'max_entries': self.max_entries,
                'window_seconds': self.window_seconds,
                'oldest_timestamp': self.buffer[0].timestamp if self.buffer else None,
                'newest_timestamp': self.buffer[-1].timestamp if self.buffer else None,
                **self._memory_stats,
            }

    def check_floor_keyframe(self, current_floor: int) -> bool:
        """Check if floor change should trigger a keyframe.

        Args:
            current_floor: Current floor number

        Returns:
            True if this is a keyframe event
        """
        if self._last_floor is None or current_floor != self._last_floor:
            self._last_floor = current_floor
            return True
        return False

    def check_combat_keyframe(self, in_combat: bool) -> bool:
        """Check if combat state change should trigger a keyframe.

        Args:
            in_combat: Whether currently in combat

        Returns:
            True if this is a keyframe event
        """
        if self._last_combat_state is None or in_combat != self._last_combat_state:
            self._last_combat_state = in_combat
            return True
        return False

    def check_inventory_keyframe(self, inventory: Dict[str, int]) -> bool:
        """Check if inventory changes should trigger a keyframe.

        Args:
            inventory: Current inventory state

        Returns:
            True if this is a keyframe event
        """
        if self._last_inventory is None or inventory != self._last_inventory:
            self._last_inventory = inventory.copy()
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from buffer."""
        with self._lock:
            self.buffer.clear()
            self._current_size_bytes = 0
            self._last_floor = None
            self._last_combat_state = None
            self._last_inventory = None
            self._memory_stats = {
                'peak_usage_bytes': 0,
                'total_added': 0,
                'total_evicted': 0,
                'avg_entry_size': 0,
                'keyframes_added': 0,
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

    def save_to_json(self, file_path: str) -> None:
        """Save the circular buffer state to a JSON file.

        Args:
            file_path: Path to save the JSON file

        Raises:
            IOError: If file cannot be written
            ValueError: If serialization fails
        """
        try:
            # Serialize buffer entries
            entries_data = []
            for entry in self.buffer:
                entry_dict = {
                    'id': entry.id,
                    'data': entry.data.tolist() if hasattr(entry.data, 'tolist') else entry.data,
                    'metadata': entry.metadata,
                    'timestamp': entry.timestamp,
                    'priority': entry.priority,
                    'is_keyframe': entry.is_keyframe
                }
                entries_data.append(entry_dict)

            # Serialize buffer state
            buffer_state = {
                'window_seconds': self.window_seconds,
                'keyframe_window_multiplier': self.keyframe_window_multiplier,
                'max_entries': self.max_entries,
                'enable_async': self._enable_async,
                'entries': entries_data,
                'last_floor': self._last_floor,
                'last_combat_state': self._last_combat_state,
                'last_inventory': self._last_inventory,
                'memory_stats': self._memory_stats
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(buffer_state, f, indent=2, ensure_ascii=False)

            logger.info("Successfully saved CircularBuffer state to %s", file_path)

        except Exception as e:
            logger.error("Failed to save CircularBuffer to JSON: %s", e)
            raise IOError(f"Failed to save buffer to {file_path}: {e}") from e

    @classmethod
    def load_from_json(cls, file_path: str) -> 'CircularBuffer':
        """Load a CircularBuffer instance from a JSON file.

        Args:
            file_path: Path to the JSON file to load

        Returns:
            Loaded CircularBuffer instance

        Raises:
            IOError: If file cannot be read
            ValueError: If deserialization fails
        """
        try:
            # Read from file
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer_state = json.load(f)

            # Validate required fields
            required_fields = ['window_seconds', 'keyframe_window_multiplier', 'max_entries', 'enable_async', 'entries']
            for field in required_fields:
                if field not in buffer_state:
                    raise ValueError(f"Missing required field '{field}' in JSON data")

            # Create buffer instance
            buffer = cls(
                window_seconds=buffer_state['window_seconds'],
                max_entries=buffer_state['max_entries'],
                enable_async=buffer_state['enable_async'],
                keyframe_window_multiplier=buffer_state['keyframe_window_multiplier']
            )

            # Restore entries
            for entry_data in buffer_state['entries']:
                # Convert data back to numpy array if it was serialized as list
                data = entry_data['data']
                if isinstance(data, list):
                    data = np.array(data)

                entry = BufferEntry(
                    id=entry_data['id'],
                    data=data,
                    metadata=entry_data['metadata'],
                    timestamp=entry_data['timestamp'],
                    priority=entry_data.get('priority', 1.0),
                    is_keyframe=entry_data.get('is_keyframe', False)
                )
                buffer.buffer.append(entry)

            # Restore internal state
            buffer._last_floor = buffer_state.get('last_floor')
            buffer._last_combat_state = buffer_state.get('last_combat_state')
            buffer._last_inventory = buffer_state.get('last_inventory')
            buffer._memory_stats = buffer_state.get('memory_stats', {
                'total_added': 0,
                'total_evicted': 0,
                'keyframes_added': 0,
            })

            logger.info("Successfully loaded CircularBuffer from %s with %d entries", file_path, len(buffer.buffer))
            return buffer

        except FileNotFoundError:
            raise IOError(f"Buffer file not found: {file_path}") from None
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}") from e
        except Exception as e:
            logger.error("Failed to load CircularBuffer from JSON: %s", e)
            raise IOError(f"Failed to load buffer from {file_path}: {e}") from e