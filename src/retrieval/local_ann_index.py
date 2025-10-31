"""Lightweight SQLite-based ANN index for on-device KNN search."""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import sqlite3
import numpy as np
from dataclasses import dataclass
import os
import pickle
import platform
import sys
from pathlib import Path
import shutil
import tempfile

logger = logging.getLogger(__name__)

# Conditional imports for file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class FileLock:
    """Cross-platform file locking utility."""
    
    def __init__(self, file_path: Path):
        """Initialize file lock for the given path.
        
        Args:
            file_path: Path to the file to lock
        """
        self.file_path = file_path
        self.lock_file = None
        self.platform = platform.system()
        self._acquired = False
    
    def acquire(self, exclusive: bool = True, timeout: float = 10.0) -> bool:
        """Acquire file lock.
        
        Args:
            exclusive: Whether to acquire exclusive lock (vs shared)
            timeout: Timeout in seconds
            
        Returns:
            True if lock acquired successfully
        """
        try:
            # Create lock file in same directory as target file
            lock_dir = self.file_path.parent
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_name = f"{self.file_path.name}.lock"
            lock_path = lock_dir / lock_name
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Try to open/create the lock file
                    try:
                        if self.lock_file:
                            self.lock_file.close()
                        self.lock_file = open(lock_path, 'w')
                    except (IOError, OSError):
                        # Can't create lock file, wait and retry
                        time.sleep(0.1)
                        continue
                    
                    if self.platform == 'Windows':
                        # Windows: use msvcrt for advisory locking (optional)
                        try:
                            import msvcrt
                            # Try to lock (0=exclusive, 1=shared)
                            # Note: msvcrt.locking only works on file descriptors, not always available
                            mode = 0 if exclusive else 1
                            msvcrt.locking(self.lock_file.fileno(), mode, 1)
                            logger.debug(f"Successfully acquired msvcrt lock for {self.file_path}")
                        except (ImportError, OSError):
                            # Advisory locking not available - use file-based lock as fallback
                            logger.debug(f"msvcrt locking not available for {self.file_path}, using file-based lock")
                            self.lock_file.write(f"locked_by_pid_{os.getpid()}")
                            self.lock_file.flush()
                    else:
                        # Unix/Linux: try fcntl
                        try:
                            import fcntl
                            # Write lock info to file first
                            self.lock_file.write(f"locked_by_pid_{os.getpid()}")
                            self.lock_file.flush()
                            self.lock_file.seek(0)
                            
                            # Try to acquire lock using fcntl
                            try:
                                # Try with getattr to safely access fcntl constants
                                lock_ex = getattr(fcntl, 'LOCK_EX', 2)
                                lock_sh = getattr(fcntl, 'LOCK_SH', 1)
                                lock_nb = getattr(fcntl, 'LOCK_NB', 4)
                                
                                if exclusive:
                                    fcntl.flock(self.lock_file.fileno(), lock_ex | lock_nb)
                                else:
                                    fcntl.flock(self.lock_file.fileno(), lock_sh | lock_nb)
                                logger.debug(f"Successfully acquired fcntl lock for {self.file_path}")
                            except (OSError, AttributeError):
                                # If fcntl.flock fails, fall back to file-based locking
                                logger.debug(f"fcntl flock not available for {self.file_path}, using file-based lock")
                                pass
                        except (ImportError, OSError):
                            # fcntl not available - continue without it
                            logger.debug(f"fcntl not available for {self.file_path}")
                            pass
                    
                    # If we get here, we have either acquired a lock or are using file-based approach
                    break
                        
                except (IOError, OSError) as e:
                    # Lock not available, wait and retry
                    logger.debug(f"Lock attempt failed for {self.file_path}: {e}")
                    time.sleep(0.1)
                    continue
            
            self._acquired = True
            logger.debug(f"Lock acquired for {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire lock for {self.file_path}: {e}")
            if self.lock_file:
                try:
                    self.lock_file.close()
                except:
                    pass
                self.lock_file = None
            return False
    
    def release(self) -> None:
        """Release file lock."""
        try:
            if self.lock_file and self._acquired:
                if self.platform == 'Windows':
                    try:
                        import msvcrt
                        msvcrt.locking(self.lock_file.fileno(), 2, 1)  # Unlock
                    except (ImportError, OSError):
                        pass
                else:
                    try:
                        import fcntl
                        # Try to unlock with fcntl
                        # If fcntl.flock doesn't exist, try LOCK_UN constant
                        try:
                            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                        except AttributeError:
                            # Use numeric constant LOCK_UN = 8
                            fcntl.flock(self.lock_file.fileno(), 8)
                    except (ImportError, OSError, AttributeError):
                        pass
                
                self.lock_file.close()
                self.lock_file = None
                self._acquired = False
                
                # Clean up lock file
                lock_dir = self.file_path.parent
                lock_path = lock_dir / f"{self.file_path.name}.lock"
                try:
                    lock_path.unlink(missing_ok=True)
                except:
                    pass
                    
                logger.debug(f"Released lock for {self.file_path}")
                
        except Exception as e:
            logger.error(f"Failed to release lock for {self.file_path}: {e}")


class AtomicFileWriter:
    """Atomic file writer using temporary files."""
    
    def __init__(self, file_path: Path):
        """Initialize atomic writer.
        
        Args:
            file_path: Path to the final file
        """
        self.file_path = file_path
        self.temp_path = None
    
    def write(self, data: bytes) -> bool:
        """Write data atomically.
        
        Args:
            data: Data to write
            
        Returns:
            True if write successful
        """
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp',
                prefix='',
                dir=str(self.file_path.parent)
            )
            self.temp_path = Path(temp_path)
            
            try:
                # Write data to temporary file
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk
                
                # Atomic rename
                os.replace(self.temp_path, self.file_path)
                self.temp_path = None
                
                logger.debug(f"Atomic write completed for {self.file_path}")
                return True
                
            except Exception:
                # Clean up temp file if rename failed
                if self.temp_path and self.temp_path.exists():
                    try:
                        self.temp_path.unlink()
                    except:
                        pass
                raise
                
        except Exception as e:
            logger.error(f"Atomic write failed for {self.file_path}: {e}")
            return False


def _normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for safe file operations.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized Path object
    """
    # Convert to Path for processing
    if isinstance(path, str):
        path = Path(path)
    
    # Normalize the path - use as_posix for consistent separator handling
    normalized_str = str(path).replace('\\', '/')
    normalized_path = Path(normalized_str)
    
    # Check for potentially dangerous paths
    if '..' in str(normalized_path).split('/'):
        logger.warning(f"Path contains '..': {path}")
    
    return normalized_path


def _validate_user_path(path: Union[str, Path]) -> Path:
    """Validate user-provided path for security (reject absolute paths).
    
    Args:
        path: Path to validate
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If path is absolute or contains unsafe patterns
    """
    # Convert to string early to check for absolute paths before any Path conversion
    path_str = str(path)
    
    # Check for absolute paths in a cross-platform way
    # Windows absolute path: starts with drive letter (C:\, D:\, etc.)
    import re
    if re.match(r'^[A-Za-z]:[/\\]', path_str):
        raise ValueError(f"Absolute paths not allowed in public API: {path}")
    
    # Unix absolute path: starts with /
    if path_str.startswith('/'):
        raise ValueError(f"Absolute paths not allowed in public API: {path}")
    
    # Now convert to Path for further processing
    if isinstance(path, str):
        path = Path(path)
    
    # Check if Path.is_absolute() also returns True (for platform-specific cases)
    if path.is_absolute():
        raise ValueError(f"Absolute paths not allowed in public API: {path}")
    
    # Use _normalize_path for further processing
    return _normalize_path(path)


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
        # Normalize and validate path
        self.db_path = _normalize_path(db_path)
        self.max_elements = max_elements
        self.vector_dim = vector_dim
        self.normalize_vectors = normalize_vectors

        # Convert Path to string for SQLite compatibility
        self.db_path_str = os.fspath(self.db_path)

        # Initialize database
        self._init_db()

        # Performance tracking
        self.search_times: List[float] = []
        self.insert_times: List[float] = []

        logger.info(
            f"Initialized LocalANNIndex: db={self.db_path}, max_elements={max_elements}, "
            f"vector_dim={vector_dim}"
        )

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path_str)
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

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add vector to index with file locking.

        Args:
            vector_id: Unique identifier for vector
            vector: Vector to add
            metadata: Optional metadata

        Returns:
            True if added successfully
        """
        start_time = time.time()

        try:
            # Acquire exclusive lock for database modifications
            lock = FileLock(self.db_path)
            if not lock.acquire(exclusive=True, timeout=30.0):
                logger.error(f"Failed to acquire lock for {self.db_path}")
                return False

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

            finally:
                # Always release the lock
                lock.release()

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

        # Get database file size using pathlib
        db_size = 0
        if self.db_path_str != ":memory:" and os.path.exists(self.db_path_str):
            db_size = os.path.getsize(self.db_path_str)

        return {
            'total_entries': total_entries,
            'max_elements': self.max_elements,
            'avg_search_time_ms': avg_search_time * 1000,
            'avg_insert_time_ms': avg_insert_time * 1000,
            'db_size_bytes': db_size,
            'db_size_mb': db_size / (1024 * 1024) if db_size > 0 else 0,
            'vector_dim': self.vector_dim,
            'normalize_vectors': self.normalize_vectors,
            'db_path': str(self.db_path),  # Return normalized path
        }

    def clear(self) -> None:
        """Clear all entries from index."""
        # Acquire exclusive lock for database modifications
        lock = FileLock(self.db_path)
        if not lock.acquire(exclusive=True, timeout=30.0):
            logger.error(f"Failed to acquire lock for {self.db_path}")
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM vectors")
            self.conn.commit()
            self.search_times.clear()
            self.insert_times.clear()
            logger.info("Cleared ANN index")
        finally:
            lock.release()

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
