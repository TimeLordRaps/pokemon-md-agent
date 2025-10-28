"""7 temporal resolution silos for hierarchical RAG system."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


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
    similarity_score: Optional[float] = None


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
    ) -> None:
        """Store embedding in this silo.
        
        Args:
            embedding: Vector embedding to store
            current_time: Current time in seconds
            trajectory_id: ID of trajectory this belongs to
            metadata: Additional metadata to store
        """
        if not self.should_sample(current_time):
            return
        
        entry = SiloEntry(
            embedding=embedding,
            timestamp=current_time,
            metadata=metadata or {},
            trajectory_id=trajectory_id,
        )
        
        self.entries.append(entry)
        self.last_sample_time = current_time
        
        # Trim if over capacity
        if len(self.entries) > self.config.max_entries:
            self.entries = self.entries[-self.config.max_entries:]
        
        logger.debug(
            "Stored entry in silo %s (total entries: %d)",
            self.config.silo_id,
            len(self.entries)
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
    ) -> List[Tuple[SiloEntry, float]]:
        """Search for similar embeddings in this silo.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        similarities = []
        
        for entry in self.entries:
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            
            if similarity >= similarity_threshold:
                similarities.append((entry, similarity))
        
        # Sort by similarity (highest first)
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
    ):
        """Initialize temporal silo manager.
        
        Args:
            base_fps: Base framerate for timing calculations
            silos: List of frame intervals for silos
        """
        self.base_fps = base_fps
        
        # Default 7 temporal silos
        if silos is None:
            silos = [1, 2, 4, 8, 16, 32, 64]
        
        self.silos: Dict[str, TemporalSilo] = {}
        self._create_silos(silos)
        
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
    
    def store(
        self,
        embedding: np.ndarray,
        trajectory_id: str,
        silo_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
        on_device_buffer: Optional[Any] = None,  # OnDeviceBufferManager
    ) -> None:
        """Store embedding in appropriate silo(s).

        Args:
            embedding: Vector embedding to store
            trajectory_id: ID of trajectory this belongs to
            silo_id: Specific silo ID (auto-select if None)
            metadata: Additional metadata
            current_time: Current time (uses time.time() if None)
            on_device_buffer: Optional on-device buffer manager for dual storage
        """
        if current_time is None:
            current_time = time.time()

        if silo_id is not None:
            # Store in specific silo
            if silo_id in self.silos:
                self.silos[silo_id].store(embedding, current_time, trajectory_id, metadata)
            else:
                logger.warning("Unknown silo ID: %s", silo_id)
        else:
            # Auto-select silos based on embedding characteristics
            # Store in multiple relevant silos
            for silo in self.silos.values():
                if silo.should_sample(current_time):
                    silo.store(embedding, current_time, trajectory_id, metadata)

        # Store in on-device buffer if available
        if on_device_buffer is not None:
            import asyncio
            # Run async operation in background
            asyncio.create_task(
                on_device_buffer.store_embedding(
                    embedding=embedding,
                    metadata={
                        **(metadata or {}),
                        "trajectory_id": trajectory_id,
                        "silo_stored": silo_id,
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
    
    def get_recent_trajectories(
        self,
        time_window_seconds: float = 30.0,
        silo_ids: Optional[List[str]] = None,
        limit_per_silo: int = 5,
    ) -> Dict[str, List[SiloEntry]]:
        """Get recent trajectories from specified time window.
        
        Args:
            time_window_seconds: Time window to retrieve from
            silo_ids: Silos to search (searches all if None)
            limit_per_silo: Maximum entries per silo
            
        Returns:
            Dictionary mapping silo_id to list of entries
        """
        silos_to_search = silo_ids or list(self.silos.keys())
        results = {}
        current_time = time.time()
        
        for silo_id in silos_to_search:
            if silo_id in self.silos:
                recent_entries = self.silos[silo_id].retrieve_recent(
                    time_window_seconds,
                    current_time,
                    limit=limit_per_silo
                )
                results[silo_id] = recent_entries
        
        return results
    
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
