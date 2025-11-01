"""
FAISS-based approximate nearest neighbor search for vector retrieval.

This module provides a VectorSearch class that loads a pre-built FAISS index
and performs efficient similarity search operations.
"""

import logging
from typing import List, Dict, Any
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class VectorSearch:
    """
    FAISS-based vector search for approximate nearest neighbor queries.

    Loads a pre-built FAISS index from disk and provides search functionality
    for finding similar vectors by ID and similarity score.
    """

    def __init__(self, index_path: str) -> None:
        """
        Initialize the vector search with a pre-built FAISS index.

        Args:
            index_path: Path to the FAISS index file (.faiss extension recommended)

        Raises:
            FileNotFoundError: If the index file does not exist
            RuntimeError: If the index cannot be loaded or is invalid
        """
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_path}: {e}")
            raise RuntimeError(f"Could not load FAISS index: {e}") from e

    async def search_async(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Async version of search for non-blocking ANN queries.

        Args:
            query_vector: Input vector(s) as numpy array
            k: Number of nearest neighbors to return

        Returns:
            List of search results with id and score
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.search, query_vector, k)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the k nearest neighbors of the query vector.

        Args:
            query_vector: Input vector(s) as numpy array. Can be 1D (single vector)
                         or 2D (batch of vectors) with shape (n_queries, dimension)
            k: Number of nearest neighbors to return (default: 5)

        Returns:
            List of dictionaries containing 'id' (vector index) and 'score' (similarity)
            for each result. Returns empty list if no results found.

        Raises:
            ValueError: If query_vector has invalid shape or k is invalid
            RuntimeError: If search operation fails
        """
        if not isinstance(query_vector, np.ndarray):
            raise ValueError("query_vector must be a numpy array")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim != 2:
            raise ValueError("query_vector must be 1D or 2D")

        if k <= 0:
            raise ValueError("k must be positive")

        if query_vector.shape[0] == 0:
            return []

        try:
            distances, indices = self.index.search(query_vector.astype(np.float32), k)
            results = []
            for i in range(query_vector.shape[0]):
                for j in range(min(k, len(indices[i]))):
                    idx = indices[i][j]
                    if idx != -1:  # FAISS returns -1 for unfound neighbors
                        results.append({
                            "id": int(idx),
                            "score": float(distances[i][j])
                        })
            return results
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e