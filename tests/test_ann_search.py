"""
Test FAISS-based vector search functionality with dummy data.

This test verifies that VectorSearch can load an index and perform
basic nearest neighbor search on randomly generated vectors.
"""

import tempfile
import numpy as np
import pytest
try:
    import faiss
    from faiss import IndexFlatL2
except ImportError:
    pytest.skip("FAISS not installed", allow_module_level=True)
from src.retrieval.ann_search import VectorSearch


def test_basic_search():
    """Test basic ANN search with dummy vectors and flat index."""
    # Create dummy vectors (10 vectors of dimension 128)
    dimension = 128
    n_vectors = 10
    np.random.seed(42)  # For reproducible results
    vectors = np.random.random((n_vectors, dimension)).astype(np.float32)

    # Build simple FAISS index
    index = IndexFlatL2(dimension)
    index.add(vectors)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
        faiss.write_index(index, tmp.name)

        # Test VectorSearch
        search = VectorSearch(tmp.name)

        # Search for nearest neighbors of first vector
        query = vectors[0:1]  # Single vector
        results = search.search(query, k=3)

        # Verify results structure and basic properties
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all('id' in r and 'score' in r for r in results)
        assert results[0]['id'] == 0  # Nearest should be itself
        assert results[0]['score'] == pytest.approx(0.0, abs=1e-6)  # Exact match
        assert all(r['score'] >= 0 for r in results)  # L2 distances are non-negative