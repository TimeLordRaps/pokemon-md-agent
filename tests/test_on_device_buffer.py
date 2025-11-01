"""Test OnDeviceBuffer interface with Literate TestDoc.

OnDeviceBuffer provides a simple interface for on-device retrieval with TTL-based eviction,
top-k search with cross-silo delegation stubs, capacity/time-based pruning, and micro stuckness
detection. The buffer maintains a ~60-minute window by evicting oldest entries on
overflow, supports top-k retrieval with ordering by relevance, and signals stuckness when
N recent queries return near-duplicates above threshold. All operations are deterministic
and thread-safe with proper error handling.

Key behaviors: store() adds entries with metadata, search() returns top-k by score,
prune() removes by age/capacity, stats() reports metrics including stuckness flag.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.on_device_buffer import OnDeviceBuffer


class TestOnDeviceBuffer:
    """Test OnDeviceBuffer interface functionality."""

    def setup_method(self):
        """Set up test fixtures with deterministic seed."""
        np.random.seed(42)  # For reproducible test results
        self.buffer = OnDeviceBuffer(
            max_entries=10,
            ttl_minutes=60,
            stuckness_threshold=0.8,
            stuckness_window=3
        )

    def test_store_basic(self):
        """Store operation adds entries with metadata and returns success."""
        embedding = np.random.rand(128).astype(np.float32)
        metadata = {"type": "test", "timestamp": time.time()}

        result = self.buffer.store(embedding, metadata)
        assert result is True

        # Verify entry was stored
        stats = self.buffer.stats()
        assert stats["total_entries"] == 1
        assert stats["total_size_bytes"] > 0

    def test_store_overflow_evicts_oldest(self):
        """Store overflow evicts oldest entries to maintain window size."""
        # Fill buffer beyond capacity
        embeddings = [np.random.rand(128).astype(np.float32) for _ in range(12)]

        for i, emb in enumerate(embeddings):
            metadata = {"index": i, "timestamp": time.time() + i}
            self.buffer.store(emb, metadata)

        # Should maintain max_entries
        stats = self.buffer.stats()
        assert stats["total_entries"] == 10

        # Oldest entries should be evicted (indices 0, 1)
        results = self.buffer.search(embeddings[10], top_k=10)
        stored_indices = [r.metadata["index"] for r in results]
        assert 0 not in stored_indices
        assert 1 not in stored_indices
        assert 10 in stored_indices  # Most recent should remain

    def test_search_top_k_ordering(self):
        """Search returns top-k results ordered by relevance score."""
        # Store embeddings with known similarities
        base_emb = np.ones(128).astype(np.float32)
        similar_emb = base_emb * 0.9  # High similarity
        dissimilar_emb = np.zeros(128).astype(np.float32)  # Low similarity

        embeddings = [base_emb, similar_emb, dissimilar_emb]
        for i, emb in enumerate(embeddings):
            metadata = {"id": i}
            self.buffer.store(emb, metadata)

        # Search with base embedding
        results = self.buffer.search(base_emb, top_k=2)

        assert len(results) == 2
        assert results[0].score >= results[1].score  # Ordered by score descending
        assert results[0].metadata["id"] == 0  # Most similar first
        assert results[1].metadata["id"] == 1  # Second most similar

    def test_search_cross_silo_stub(self):
        """Search includes cross-silo delegation stub without actual calls."""
        embedding = np.random.rand(128).astype(np.float32)
        self.buffer.store(embedding, {"test": True})

        with patch('src.retrieval.on_device_buffer.logger') as mock_logger:
            results = self.buffer.search(embedding, top_k=5)

            # Should log delegation attempt but not make real calls
            mock_logger.debug.assert_called()
            # Check that cross-silo delegation was logged
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("Cross-silo" in call for call in debug_calls)

    def test_prune_by_time(self):
        """Prune removes entries older than TTL."""
        # Store entries with different timestamps
        old_time = time.time() - (70 * 60)  # 70 minutes ago
        recent_time = time.time() - (30 * 60)  # 30 minutes ago

        # Old entry
        old_emb = np.random.rand(128).astype(np.float32)
        self.buffer.store(old_emb, {"timestamp": old_time})

        # Recent entries
        for i in range(3):
            emb = np.random.rand(128).astype(np.float32)
            self.buffer.store(emb, {"timestamp": recent_time + i})

        removed = self.buffer.prune()
        assert removed == 1  # Only old entry removed

        stats = self.buffer.stats()
        assert stats["total_entries"] == 3

    def test_prune_by_capacity(self):
        """Prune reduces entries when exceeding capacity threshold."""
        # Fill buffer
        for i in range(15):
            emb = np.random.rand(128).astype(np.float32)
            self.buffer.store(emb, {"index": i})

        # Prune should reduce to reasonable size
        removed = self.buffer.prune(max_entries=8)
        assert removed > 0

        stats = self.buffer.stats()
        assert stats["total_entries"] <= 8

    def test_micro_stuckness_detection(self):
        """Micro stuckness signals when N recent queries return near-duplicates."""
        # Store diverse embeddings
        embeddings = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        for emb in embeddings:
            self.buffer.store(emb, {"type": "diverse"})

        # First few searches on different embeddings - not stuck
        for emb in embeddings[:2]:
            results = self.buffer.search(emb, top_k=3)
            assert not self.buffer.is_stuck()

        # Search same embedding multiple times - should trigger stuckness
        repeated_emb = embeddings[0]
        for _ in range(3):
            results = self.buffer.search(repeated_emb, top_k=3)
            # Should have near-duplicate results

        assert self.buffer.is_stuck()

    def test_stats_comprehensive(self):
        """Stats reports comprehensive metrics including stuckness."""
        # Add some data
        for i in range(5):
            emb = np.random.rand(128).astype(np.float32)
            self.buffer.store(emb, {"index": i})

        stats = self.buffer.stats()

        required_keys = ["total_entries", "total_size_bytes", "avg_entry_age_seconds",
                        "stuckness_score", "is_stuck", "capacity_utilization"]
        for key in required_keys:
            assert key in stats

        assert stats["total_entries"] == 5
        assert isinstance(stats["is_stuck"], bool)
        assert 0.0 <= stats["capacity_utilization"] <= 1.0

    def test_concurrent_access(self):
        """Buffer handles concurrent store/search operations safely."""
        import threading
        import concurrent.futures

        results = []

        def store_worker(worker_id):
            for i in range(10):
                emb = np.random.rand(128).astype(np.float32)
                metadata = {"worker": worker_id, "index": i}
                self.buffer.store(emb, metadata)

        def search_worker():
            query = np.random.rand(128).astype(np.float32)
            result = self.buffer.search(query, top_k=5)
            results.append(len(result))

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Start store operations
            store_futures = [executor.submit(store_worker, i) for i in range(3)]
            # Start search operations
            search_futures = [executor.submit(search_worker) for _ in range(5)]

            # Wait for completion
            concurrent.futures.wait(store_futures + search_futures)

        # Verify buffer integrity
        stats = self.buffer.stats()
        assert stats["total_entries"] > 0
        assert all(r > 0 for r in results)

    def test_empty_buffer_search(self):
        """Search on empty buffer returns empty results without error."""
        query = np.random.rand(128).astype(np.float32)
        results = self.buffer.search(query, top_k=5)

        assert results == []
        assert not self.buffer.is_stuck()

    def test_error_handling_invalid_embedding(self):
        """Store rejects invalid embeddings with appropriate exceptions."""
        # Test None embedding
        with pytest.raises(ValueError, match="Invalid embedding"):
            self.buffer.store(None, {"test": True})

        # Test wrong shape
        invalid_emb = np.random.rand(64)  # Wrong dimension
        with pytest.raises(ValueError, match="Invalid embedding"):
            self.buffer.store(invalid_emb, {"test": True})

        # Test non-numeric
        invalid_emb = "not_an_array"
        with pytest.raises(ValueError, match="Invalid embedding"):
            self.buffer.store(invalid_emb, {"test": True})

    def test_metadata_validation(self):
        """Metadata must be dictionary with serializable values."""
        embedding = np.random.rand(128).astype(np.float32)

        # Valid metadata
        result = self.buffer.store(embedding, {"string": "value", "number": 42})
        assert result is True

        # Invalid metadata types
        with pytest.raises(ValueError, match="Invalid metadata"):
            self.buffer.store(embedding, "not_a_dict")

        with pytest.raises(ValueError, match="Invalid metadata"):
            self.buffer.store(embedding, {"unserializable": lambda x: x})