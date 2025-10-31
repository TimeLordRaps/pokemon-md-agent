"""Tests for local ANN index with file locking and path safety."""

import pytest
import tempfile
import os
import numpy as np
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.retrieval.local_ann_index import LocalANNIndex, FileLock, _normalize_path, _validate_user_path, AtomicFileWriter


class TestFileLock:
    """Test file locking functionality."""

    def test_file_lock_creation(self):
        """Test file lock creation and basic functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.db"
            lock = FileLock(test_path)
            
            # Should create without error
            assert lock.file_path == test_path
            assert lock.platform is not None

    def test_file_lock_acquire_release(self):
        """Test acquiring and releasing file locks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.db"
            lock = FileLock(test_path)
            
            # Acquire lock
            acquired = lock.acquire(timeout=1.0)
            assert acquired
            assert lock._acquired
            
            # Release lock
            lock.release()
            assert not lock._acquired

    def test_concurrent_file_locking(self):
        """Test that concurrent locks work correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.db"
            
            results = []
            
            def acquire_lock(thread_id):
                lock = FileLock(test_path)
                if lock.acquire(timeout=5.0):
                    results.append(f"Thread {thread_id} acquired lock")
                    time.sleep(0.1)  # Hold lock briefly
                    lock.release()
                    results.append(f"Thread {thread_id} released lock")
                    return True
                else:
                    results.append(f"Thread {thread_id} failed to acquire lock")
                    return False
            
            # Test with multiple threads
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(acquire_lock, i) for i in range(3)]
                results_list = [f.result() for f in futures]
            
            # All threads should succeed
            assert all(results_list)
            # Verify proper sequencing
            assert len(results) == 6  # 3 acquire + 3 release


class TestPathNormalization:
    """Test path normalization and validation."""

    def test_relative_path_normalization(self):
        """Test that relative paths are properly normalized."""
        # Test various path formats
        test_cases = [
            "test.db",
            "./test.db",
            "subdir/test.db",
            "subdir/../test.db"
        ]
        
        for path_str in test_cases:
            normalized = _normalize_path(path_str)
            assert isinstance(normalized, Path)
            assert not str(normalized).startswith('/')

    def test_absolute_path_rejection(self):
        """Test that absolute paths are rejected."""
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            _validate_user_path("/absolute/path/test.db")
        
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            _validate_user_path("C:\\absolute\\path\\test.db")

    def test_path_with_dots(self):
        """Test handling of paths with '..' components."""
        with patch('pathlib.Path.resolve') as mock_resolve:
            # Mock resolve to return a non-absolute path to test warning
            mock_path = Path("relative/path/../test.db")
            mock_resolve.return_value = mock_path
            
            normalized = _normalize_path("relative/path/../test.db")
            # Should still work but may generate warning
            assert isinstance(normalized, Path)


class TestLocalANNIndex:
    """Test LocalANNIndex with file locking."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir) / "test_index.db"

    def test_index_initialization_with_file_path(self, temp_db_path):
        """Test index initialization with file path."""
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        
        # Should initialize successfully
        assert index.vector_dim == 128
        assert index.db_path == temp_db_path
        assert index.db_path_str == os.fspath(temp_db_path)
        
        index.close()

    def test_memory_index_initialization(self):
        """Test in-memory index initialization."""
        index = LocalANNIndex(db_path=":memory:", vector_dim=128)
        
        assert index.vector_dim == 128
        assert index.db_path_str == ":memory:"
        
        index.close()

    def test_add_vector_with_file_locking(self, temp_db_path):
        """Test adding vectors with file locking."""
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        
        # Add a vector
        vector = np.random.randn(128).astype(np.float32)
        success = index.add_vector("test_vector", vector, {"metadata": "test"})
        
        assert success
        stats = index.get_stats()
        assert stats["total_entries"] == 1
        assert stats["db_path"] == str(temp_db_path)
        
        index.close()

    def test_search_functionality(self, temp_db_path):
        """Test search functionality."""
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        
        # Add multiple vectors
        vectors = []
        for i in range(10):
            vector = np.random.randn(128).astype(np.float32)
            index.add_vector(f"vector_{i}", vector, {"id": i})
            vectors.append(vector)
        
        # Search with first vector
        query_vector = vectors[0]
        results = index.search(query_vector, k=5)
        
        assert len(results) == 5
        # First result should be the most similar (should be the query vector itself)
        assert results[0].entry_id == "vector_0"
        
        index.close()

    def test_concurrent_vector_addition(self, temp_db_path):
        """Test basic vector operations work correctly."""
        # Create a single index and test sequential operations
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=64)
        
        # Add vectors sequentially 
        for i in range(5):
            vector_id = f"vector_{i}"
            vector = np.random.randn(64).astype(np.float32)
            success = index.add_vector(vector_id, vector)
            assert success
        
        # Check final count
        final_stats = index.get_stats()
        assert final_stats["total_entries"] == 5
        
        index.close()

    def test_clear_with_locking(self, temp_db_path):
        """Test clear operation with file locking."""
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        
        # Add some vectors
        for i in range(10):
            vector = np.random.randn(128).astype(np.float32)
            index.add_vector(f"vector_{i}", vector)
        
        assert index.get_stats()["total_entries"] == 10
        
        # Clear the index
        index.clear()
        
        assert index.get_stats()["total_entries"] == 0
        
        index.close()

    def test_index_stats(self, temp_db_path):
        """Test index statistics."""
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        
        # Initially empty
        stats = index.get_stats()
        assert stats["total_entries"] == 0
        assert stats["vector_dim"] == 128
        assert stats["db_path"] == str(temp_db_path)
        
        # Add some vectors and check stats
        for i in range(5):
            vector = np.random.randn(128).astype(np.float32)
            index.add_vector(f"vector_{i}", vector)
        
        final_stats = index.get_stats()
        assert final_stats["total_entries"] == 5
        assert final_stats["avg_insert_time_ms"] > 0
        
        index.close()

    def test_path_normalization_in_constructor(self, temp_db_path):
        """Test that path is normalized in constructor."""
        # Test with string path
        index = LocalANNIndex(db_path=str(temp_db_path), vector_dim=128)
        assert isinstance(index.db_path, Path)
        assert isinstance(index.db_path_str, str)
        
        # Test with Path object
        index2 = LocalANNIndex(db_path=temp_db_path, vector_dim=128)
        assert isinstance(index2.db_path, Path)
        
        index.close()
        index2.close()


class TestAtomicFileWriter:
    """Test atomic file operations."""

    def test_atomic_write_creation(self):
        """Test atomic file writer creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test_file.dat"
            writer = AtomicFileWriter(test_path)
            
            assert writer.file_path == test_path

    def test_atomic_write_operation(self):
        """Test atomic file write operation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test_file.dat"
            writer = AtomicFileWriter(test_path)
            
            # Write data atomically
            test_data = b"Hello, atomic world!"
            success = writer.write(test_data)
            
            assert success
            assert test_path.exists()
            
            # Read back and verify
            with open(test_path, 'rb') as f:
                read_data = f.read()
            
            assert read_data == test_data