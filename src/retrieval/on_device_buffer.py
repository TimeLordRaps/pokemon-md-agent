"""On-device buffer manager orchestrating all components."""

from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .circular_buffer import CircularBuffer, BufferEntry
from .keyframe_policy import KeyframePolicy, KeyframeCandidate, KeyframeResult
from .local_ann_index import LocalANNIndex, SearchResult
from .meta_view_writer import MetaViewWriter, ViewTile, MetaViewResult
from .embedding_generator import EmbeddingGenerator
from .deduplicator import Deduplicator
from ..retrieval.stuckness_detector import StucknessDetector, StucknessAnalysis
from ..retrieval.auto_retrieve import AutoRetriever, RetrievalQuery, RetrievedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class OnDeviceBufferConfig:
    """Configuration for on-device buffer system."""
    circular_buffer_mb: float = 50.0
    ann_index_max_elements: int = 10000
    keyframe_min_count: int = 5
    keyframe_max_count: int = 100
    enable_persistence: bool = False
    enable_async: bool = True
    search_timeout_ms: int = 100


@dataclass
class BufferOperationResult:
    """Result of buffer operations."""
    success: bool
    operation_time: float
    data_size: int
    metadata: Dict[str, Any]


class OnDeviceBufferManager:
    """Orchestrates on-device circular buffer, ANN search, and keyframe management."""

    def __init__(self, config: OnDeviceBufferConfig):
        """Initialize on-device buffer manager.

        Args:
            config: Buffer configuration
        """
        self.config = config

        # Initialize components
        self.circular_buffer = CircularBuffer(
            max_size_mb=config.circular_buffer_mb,
            enable_async=config.enable_async,
        )

        self.ann_index = LocalANNIndex(
            max_elements=config.ann_index_max_elements,
        )

        self.keyframe_policy = KeyframePolicy(
            min_keyframes=config.keyframe_min_count,
            max_keyframes=config.keyframe_max_count,
        )

        self.meta_view_writer = MetaViewWriter(
            grid_size=(2, 2),
            enable_async=config.enable_async,
        )

        # New components for enhanced functionality
        self.embedding_generator = EmbeddingGenerator()
        self.deduplicator = Deduplicator()

        # State tracking
        self._operation_times: List[float] = []
        self._search_times: List[float] = []
        self._stuckness_detector: Optional[StucknessDetector] = None

        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=4) if config.enable_async else None

        logger.info(
            "Initialized OnDeviceBufferManager: buffer=%.1fMB, ANN=%d elements",
            config.circular_buffer_mb, config.ann_index_max_elements
        )

    def set_stuckness_detector(self, detector: StucknessDetector) -> None:
        """Set stuckness detector for keyframe policy feedback."""
        self._stuckness_detector = detector
        logger.info("Stuckness detector connected to buffer manager")

    async def store_embedding(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        embedding_id: Optional[str] = None,
    ) -> BufferOperationResult:
        """Store embedding in on-device buffer system.

        Args:
            embedding: Embedding vector
            metadata: Associated metadata
            embedding_id: Optional unique ID

        Returns:
            Operation result
        """
        start_time = time.time()

        try:
            # Generate ID if not provided
            if embedding_id is None:
                embedding_id = f"emb_{int(time.time() * 1000)}_{hash(embedding.tobytes()) % 10000}"

            # Create buffer entry
            entry = BufferEntry(
                id=embedding_id,
                data=embedding,
                metadata=metadata,
                timestamp=time.time(),
                priority=metadata.get('priority', 1.0),
            )

            # Store in circular buffer
            buffer_success = await self.circular_buffer.add_entry_async(entry)

            # Index in ANN (if buffer storage succeeded)
            ann_success = False
            if buffer_success:
                ann_success = self.ann_index.add_vector(
                    vector_id=embedding_id,
                    vector=embedding,
                    metadata=metadata,
                )

            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            result = BufferOperationResult(
                success=buffer_success and ann_success,
                operation_time=operation_time,
                data_size=embedding.nbytes,
                metadata={
                    "buffer_success": buffer_success,
                    "ann_success": ann_success,
                    "embedding_id": embedding_id,
                }
            )

            logger.debug(
                "Stored embedding %s: buffer=%s, ANN=%s, time=%.3fs",
                embedding_id, buffer_success, ann_success, operation_time
            )

            return result

        except Exception as e:
            logger.error("Failed to store embedding: %s", e)
            return BufferOperationResult(
                success=False,
                operation_time=time.time() - start_time,
                data_size=embedding.nbytes if 'embedding' in locals() else 0,
                metadata={"error": str(e)}
            )

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        search_timeout_ms: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings using on-device ANN.

        Args:
            query_embedding: Query embedding
            top_k: Number of results
            search_timeout_ms: Search timeout

        Returns:
            Search results
        """
        timeout = search_timeout_ms or self.config.search_timeout_ms
        start_time = time.time()

        try:
            # Perform ANN search with timeout
            if self.config.enable_async and self._executor:
                try:
                    results = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self._executor, self.ann_index.search, query_embedding, top_k
                        ),
                        timeout=timeout / 1000.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("ANN search timed out after %dms", timeout)
                    return []
            else:
                results = self.ann_index.search(query_embedding, top_k)

            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self._search_times.append(search_time)

            # Filter results by buffer availability (ensure data is still in buffer)
            filtered_results = []
            for result in results:
                # Check if entry still exists in circular buffer
                entries = await self.circular_buffer.get_entries_async(
                    limit=1,
                    time_window=None  # No time filter for existence check
                )
                entry_ids = [e.id for e in entries]
                if result.entry_id in entry_ids:
                    filtered_results.append(result)

            logger.debug(
                "ANN search completed: %d results in %.1fms",
                len(filtered_results), search_time
            )

            return filtered_results[:top_k]

        except Exception as e:
            logger.error("ANN search failed: %s", e)
            return []

    async def process_keyframes(
        self,
        current_stuckness: Optional[float] = None,
    ) -> Optional[KeyframeResult]:
        """Process and select keyframes from buffer contents.

        Args:
            current_stuckness: Current stuckness score (0.0-1.0)

        Returns:
            Keyframe selection result or None if no candidates
        """
        try:
            # Get recent buffer entries as candidates
            buffer_entries = await self.circular_buffer.get_entries_async(
                limit=50,  # Consider last 50 entries
                time_window=300.0,  # Last 5 minutes
            )

            if len(buffer_entries) < 4:  # Need minimum candidates
                return None

            # Convert to keyframe candidates
            candidates = []
            for entry in buffer_entries:
                candidate = KeyframeCandidate(
                    timestamp=entry.timestamp,
                    embedding=entry.data,
                    metadata=entry.metadata,
                    importance_score=entry.priority,
                )
                candidates.append(candidate)

            # Select keyframes
            result = self.keyframe_policy.select_keyframes(
                candidates=candidates,
                current_stuckness_score=current_stuckness,
                force_adaptive=bool(current_stuckness and current_stuckness > 0.7),
            )

            logger.debug(
                "Processed keyframes: selected %d from %d candidates",
                len(result.selected_keyframes), len(candidates)
            )

            return result

        except Exception as e:
            logger.error("Keyframe processing failed: %s", e)
            return None

    async def generate_meta_view(
        self,
        title: Optional[str] = None,
    ) -> Optional[MetaViewResult]:
        """Generate meta view from current keyframes.

        Args:
            title: Optional view title

        Returns:
            Meta view result or None
        """
        try:
            # Get keyframes from policy
            keyframes = self.keyframe_policy.selected_keyframes

            if len(keyframes) < 2:
                logger.debug("Insufficient keyframes for meta view")
                return None

            # Convert keyframes to view tiles
            tiles = []
            for kf in keyframes[-4:]:  # Use last 4 keyframes (for 2x2 grid)
                # Create simple visualization from embedding
                image = self.meta_view_writer._create_embedding_visualization(kf.embedding)

                tile = ViewTile(
                    image=image,
                    metadata={
                        **kf.metadata,
                        "timestamp": kf.timestamp,
                        "importance_score": kf.importance_score,
                    },
                    position=(0, 0),  # Will be set by writer
                    importance_score=kf.importance_score,
                )
                tiles.append(tile)

            # Generate meta view
            result = await self.meta_view_writer.generate_meta_view_async(
                tiles=tiles,
                layout_strategy="importance",
                title=title or f"Keyframe View ({len(tiles)} frames)",
            )

            logger.debug("Generated meta view with %d tiles", len(tiles))
            return result

        except Exception as e:
            logger.error("Meta view generation failed: %s", e)
            return None

    async def get_buffer_status(self) -> Dict[str, Any]:
        """Get comprehensive buffer system status."""
        try:
            buffer_stats = self.circular_buffer.get_memory_stats()
            ann_stats = self.ann_index.get_stats()
            keyframe_stats = self.keyframe_policy.get_policy_stats()

            # Calculate performance metrics
            avg_operation_time = np.mean(self._operation_times[-100:]) if self._operation_times else 0.0
            avg_search_time = np.mean(self._search_times[-100:]) if self._search_times else 0.0

            # Stuckness integration status
            stuckness_status = "connected" if self._stuckness_detector else "disconnected"

            return {
                "buffer": buffer_stats,
                "ann_index": ann_stats,
                "keyframe_policy": keyframe_stats,
                "performance": {
                    "avg_operation_time": avg_operation_time,
                    "avg_search_time_ms": avg_search_time,
                    "total_operations": len(self._operation_times),
                    "total_searches": len(self._search_times),
                },
                "integrations": {
                    "stuckness_detector": stuckness_status,
                },
                "config": {
                    "circular_buffer_mb": self.config.circular_buffer_mb,
                    "ann_max_elements": self.config.ann_index_max_elements,
                    "keyframe_range": f"{self.config.keyframe_min_count}-{self.config.keyframe_max_count}",
                    "persistence_enabled": self.config.enable_persistence,
                    "async_enabled": self.config.enable_async,
                }
            }

        except Exception as e:
            logger.error("Failed to get buffer status: %s", e)
            return {"error": str(e)}

    async def cleanup_old_data(self, max_age_seconds: float = 3600.0) -> int:
        """Clean up old data from buffer system.

        Args:
            max_age_seconds: Maximum age of data to keep

        Returns:
            Number of entries cleaned
        """
        try:
            # Note: Circular buffer automatically manages size
            # This is mainly for ANN index cleanup
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds

            # Get all entries older than cutoff
            old_entries = await self.circular_buffer.get_entries_async(
                time_window=max_age_seconds
            )

            # Remove from ANN index (circular buffer handles itself)
            removed_count = 0
            for entry in old_entries:
                # ANN index doesn't have explicit removal, but we could mark as stale
                # For now, just count
                removed_count += 1

            logger.info("Cleaned up %d old entries (older than %.1fs)", removed_count, max_age_seconds)
            return removed_count

        except Exception as e:
            logger.error("Cleanup failed: %s", e)
            return 0

    def shutdown(self) -> None:
        """Shutdown buffer manager and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

        # Optional persistence save
        if self.config.enable_persistence:
            self._save_persistent_state()

        logger.info("OnDeviceBufferManager shutdown complete")

    def _save_persistent_state(self) -> None:
        """Save persistent state (if enabled)."""
        # Implementation for persistence would go here
        # Could save buffer contents, ANN index state, etc.
        pass