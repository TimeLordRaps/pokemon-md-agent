"""Lightweight SQLite-based ANN index for on-device KNN search."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import sqlite3
import numpy as np
from dataclasses import dataclass
import os
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ANNEntry:
    """Entry in the ANN index."""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class SearchResult:
    """Result from ANN search."""
    entry_id: str
    score: float
    metadata: Dict[str, Any]


class LocalANNIndex:
    """SQLite-based ANN index for on-device KNN search."""

    def __init__(
        self,
        db_path: str = ":memory:",
        max_elements: int = 10000,
        vector_dim: int = 1024,
        normalize_vectors: bool = True,
    ):
        """Initialize SQLite ANN index.

        Args:
            db_path: Path to SQLite database file
            max_elements: Maximum number of elements
            vector_dim: Dimension of vectors
            normalize_vectors: Whether to normalize input vectors
        """
        self.db_path = db_path
        self.max_elements = max_elements
        self.vector_dim = vector_dim
        self.normalize_vectors = normalize_vectors

        # Initialize database
        self._init_db()

        # Performance tracking
        self.search_times: List[float] = []
        self.insert_times: List[float] = []

        logger.info(
            f"Initialized LocalANNIndex: db={db_path}, max_elements={max_elements}, "
            f"vector_dim={vector_dim}"
        )

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                metadata BLOB NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON vectors(timestamp)")
        self.conn.commit()

    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add vector to index.

        Args:
            vector_id: Unique identifier for vector
            vector: Vector to add
            metadata: Optional metadata

        Returns:
            True if added successfully
        """
        start_time = time.time()

        try:
            # Normalize vector if required
            if self.normalize_vectors:
                vector = vector / np.linalg.norm(vector)

            # Serialize data
            vector_blob = pickle.dumps(vector.astype(np.float32))
            metadata_blob = pickle.dumps(metadata or {})

            # Check if exists and count
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            count = cursor.fetchone()[0]

            if count >= self.max_elements:
                logger.warning("Index at max capacity")
                return False

            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO vectors (id, vector, metadata, timestamp)
                VALUES (?, ?, ?, ?)
            """, (vector_id, vector_blob, metadata_blob, time.time()))

            self.conn.commit()

            insert_time = time.time() - start_time
            self.insert_times.append(insert_time)

            logger.debug(f"Added vector {vector_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            return False

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[SearchResult]:
        """Search for k nearest neighbors using brute force.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            List of search results
        """
        start_time = time.time()

        try:
            # Normalize query if required
            if self.normalize_vectors:
                query_vector = query_vector / np.linalg.norm(query_vector)

            # Get all vectors from database
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, vector, metadata FROM vectors")

            results = []
            for row in cursor.fetchall():
                vector_id, vector_blob, metadata_blob = row
                vector = pickle.loads(vector_blob)
                metadata = pickle.loads(metadata_blob)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, vector)
                results.append((vector_id, similarity, metadata))

            # Sort by similarity (descending) and return top k
            results.sort(key=lambda x: x[1], reverse=True)

            search_results = []
            for vector_id, score, metadata in results[:k]:
                search_results.append(SearchResult(
                    entry_id=vector_id,
                    score=score,
                    metadata=metadata,
                ))

            search_time = time.time() - start_time
            self.search_times.append(search_time)

            logger.debug(f"Search completed in {search_time:.4f}s, found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        total_entries = cursor.fetchone()[0]

        if not self.search_times:
            avg_search_time = 0.0
        else:
            avg_search_time = np.mean(self.search_times)

        if not self.insert_times:
            avg_insert_time = 0.0
        else:
            avg_insert_time = np.mean(self.insert_times)

        # Get database file size
        db_size = os.path.getsize(self.db_path) if self.db_path != ":memory:" and os.path.exists(self.db_path) else 0

        return {
            'total_entries': total_entries,
            'max_elements': self.max_elements,
            'avg_search_time_ms': avg_search_time * 1000,
            'avg_insert_time_ms': avg_insert_time * 1000,
            'db_size_bytes': db_size,
            'db_size_mb': db_size / (1024 * 1024) if db_size > 0 else 0,
            'vector_dim': self.vector_dim,
            'normalize_vectors': self.normalize_vectors,
        }

    def clear(self) -> None:
        """Clear all entries from index."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM vectors")
        self.conn.commit()
        self.search_times.clear()
        self.insert_times.clear()
        logger.info("Cleared ANN index")

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()