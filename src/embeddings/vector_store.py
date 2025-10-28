"""Vector store wrapper for ChromaDB or FAISS with temporal silo support."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """Entry in the vector store."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    silo_id: str


class VectorStore:
    """Vector store interface with ChromaDB/FAISS backend."""
    
    def __init__(
        self,
        backend: str = "memory",  # "memory", "chromadb", "faiss"
        collection_name: str = "pokemon_md_embeddings",
        embedding_dimension: int = 1024,
    ):
        """Initialize vector store.
        
        Args:
            backend: Storage backend ("memory", "chromadb", "faiss")
            collection_name: Name of collection/table
            embedding_dimension: Dimension of embedding vectors
        """
        self.backend = backend
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize backend-specific storage
        if backend == "memory":
            self._init_memory_backend()
        elif backend == "chromadb":
            self._init_chromadb_backend()
        elif backend == "faiss":
            self._init_faiss_backend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        logger.info("Initialized vector store: backend=%s, collection=%s", backend, collection_name)
    
    def _init_memory_backend(self) -> None:
        """Initialize in-memory storage backend."""
        self._entries: Dict[str, VectorEntry] = {}
        self._embeddings: np.ndarray = np.array([])  # Will reshape when adding entries
        self._faiss = None  # FAISS reference for FAISS backend
        
    def _init_chromadb_backend(self) -> None:
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ))
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Pokemon MD agent embeddings"}
            )
            
            self._chromadb = chromadb  # Store reference
            self._chromadb_settings = Settings
            
            logger.info("ChromaDB backend initialized")
            
        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            raise
    
    def _init_faiss_backend(self) -> None:
        """Initialize FAISS backend."""
        try:
            import faiss
            
            # Create FAISS index
            self._index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
            
            # Create metadata storage
            self._entries: Dict[str, VectorEntry] = {}
            self._id_mapping: Dict[int, str] = {}  # FAISS index -> entry ID
            self._faiss_lib = faiss  # Store reference
            
            logger.info("FAISS backend initialized")
            
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
    
    def _get_faiss(self):
        """Get FAISS library reference."""
        return getattr(self, '_faiss_lib', None)
    
    def add_entry(
        self,
        entry_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        silo_id: str,
    ) -> bool:
        """Add a single entry to the vector store.
        
        Args:
            entry_id: Unique entry ID
            embedding: Vector embedding
            metadata: Associated metadata
            silo_id: Temporal silo this belongs to
            
        Returns:
            True if added successfully
        """
        timestamp = time.time()
        
        entry = VectorEntry(
            id=entry_id,
            embedding=embedding,
            metadata=metadata,
            timestamp=timestamp,
            silo_id=silo_id,
        )
        
        try:
            if self.backend == "memory":
                self._add_to_memory(entry)
            elif self.backend == "chromadb":
                self._add_to_chromadb(entry)
            elif self.backend == "faiss":
                self._add_to_faiss(entry)
            
            logger.debug("Added entry %s to silo %s", entry_id, silo_id)
            return True
            
        except Exception as e:
            logger.error("Failed to add entry %s: %s", entry_id, e)
            return False
    
    def _add_to_memory(self, entry: VectorEntry) -> None:
        """Add entry to memory backend."""
        self._entries[entry.id] = entry
        
        # Update embedding matrix
        if len(self._embeddings) == 0:
            self._embeddings = entry.embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, entry.embedding])
    
    def _add_to_chromadb(self, entry: VectorEntry) -> None:
        """Add entry to ChromaDB backend."""
        # Convert embedding to list for ChromaDB
        embedding_list = entry.embedding.tolist()
        
        # Add to collection
        self._collection.add(
            ids=[entry.id],
            embeddings=[embedding_list],
            metadatas=[{
                **entry.metadata,
                "silo_id": entry.silo_id,
                "timestamp": entry.timestamp,
            }],
            documents=[entry.metadata.get("document", "")]
        )
    
    def _add_to_faiss(self, entry: VectorEntry) -> None:
        """Add entry to FAISS backend."""
        # Normalize embedding for cosine similarity
        normalized_embedding = entry.embedding / np.linalg.norm(entry.embedding)
        
        # Add to FAISS index
        self._index.add(normalized_embedding.reshape(1, -1))
        
        # Store metadata
        self._entries[entry.id] = entry
        self._id_mapping[self._index.ntotal - 1] = entry.id
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        silo_filter: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        on_device_backend: Optional[Any] = None,  # OnDeviceBufferManager
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            silo_filter: Only search in these silos
            metadata_filter: Filter by metadata
            on_device_backend: Optional on-device ANN backend

        Returns:
            List of (entry_id, similarity_score, metadata) tuples
        """
        results = []

        try:
            # Primary search in vector store
            if self.backend == "memory":
                results = self._search_memory(query_embedding, top_k, silo_filter, metadata_filter)
            elif self.backend == "chromadb":
                results = self._search_chromadb(query_embedding, top_k, silo_filter, metadata_filter)
            elif self.backend == "faiss":
                results = self._search_faiss(query_embedding, top_k, silo_filter, metadata_filter)

            # Supplement with on-device ANN if available
            if on_device_backend is not None:
                try:
                    ann_results = on_device_backend.search_similar(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        search_timeout_ms=50,  # Fast fallback
                    )

                    # Convert and merge results
                    ann_converted = []
                    for ann_result in ann_results:
                        ann_converted.append((
                            ann_result.entry_id,
                            ann_result.score,
                            ann_result.metadata,
                        ))

                    # Merge and deduplicate
                    existing_ids = {r[0] for r in results}
                    for ann_result in ann_converted:
                        if ann_result[0] not in existing_ids:
                            results.append(ann_result)

                    # Re-sort by score and limit
                    results.sort(key=lambda x: x[1], reverse=True)
                    results = results[:top_k]

                except Exception as e:
                    logger.warning("On-device ANN search failed: %s", e)

        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        return results
    
    def _search_memory(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        silo_filter: Optional[List[str]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in memory backend."""
        if len(self._embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = []
        
        for i, (entry_id, entry) in enumerate(self._entries.items()):
            # Apply filters
            if silo_filter and entry.silo_id not in silo_filter:
                continue
            
            if metadata_filter:
                if not self._metadata_matches(entry.metadata, metadata_filter):
                    continue
            
            # Compute cosine similarity
            embedding = self._embeddings[i]
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((entry_id, similarity, entry.metadata))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _search_chromadb(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        silo_filter: Optional[List[str]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in ChromaDB backend."""
        # Prepare query
        query_embedding_list = query_embedding.tolist()
        
        # Build where clause for filtering
        where_clause = {}
        
        if silo_filter:
            where_clause["silo_id"] = {"$in": silo_filter}
        
        if metadata_filter:
            where_clause.update(metadata_filter)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            where=where_clause if where_clause else None,
        )
        
        # Format results
        formatted_results = []
        
        # Handle None results
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        for i, entry_id in enumerate(results["ids"][0]):
            similarity = 1.0 - results["distances"][0][i]  # ChromaDB returns distances
            metadata = results["metadatas"][0][i]
            formatted_results.append((entry_id, similarity, metadata))
        
        return formatted_results
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        silo_filter: Optional[List[str]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in FAISS backend."""
        if self._index.ntotal == 0:
            return []
        
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        k = min(top_k, self._index.ntotal)
        similarities, indices = self._index.search(
            normalized_query.reshape(1, -1),
            k
        )
        
        # Format results
        results = []
        
        for similarity, faiss_idx in zip(similarities[0], indices[0]):
            if faiss_idx >= 0:  # Valid index
                entry_id = self._id_mapping.get(faiss_idx)
                if entry_id:
                    entry = self._entries[entry_id]
                    
                    # Apply filters
                    if silo_filter and entry.silo_id not in silo_filter:
                        continue
                    
                    if metadata_filter:
                        if not self._metadata_matches(entry.metadata, metadata_filter):
                            continue
                    
                    results.append((entry_id, float(similarity), entry.metadata))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _metadata_matches(self, entry_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if entry metadata matches filter.
        
        Args:
            entry_metadata: Entry metadata to check
            filter_metadata: Filter criteria
            
        Returns:
            True if metadata matches filter
        """
        for key, value in filter_metadata.items():
            if key not in entry_metadata:
                return False
            
            if isinstance(value, dict):
                # Handle complex queries (e.g., {"$gte": 0.8})
                for operator, filter_value in value.items():
                    if operator == "$gte":
                        if entry_metadata[key] < filter_value:
                            return False
                    elif operator == "$lte":
                        if entry_metadata[key] > filter_value:
                            return False
                    elif operator == "$in":
                        if entry_metadata[key] not in filter_value:
                            return False
                    # Add more operators as needed
            else:
                if entry_metadata[key] != value:
                    return False
        
        return True
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        if self.backend == "memory":
            return {
                "backend": self.backend,
                "total_entries": len(self._entries),
                "embedding_dimension": self.embedding_dimension,
                "collection_name": self.collection_name,
            }
        
        elif self.backend == "chromadb":
            count = self._collection.count()
            return {
                "backend": self.backend,
                "total_entries": count,
                "embedding_dimension": self.embedding_dimension,
                "collection_name": self.collection_name,
            }
        
        elif self.backend == "faiss":
            return {
                "backend": self.backend,
                "total_entries": self._index.ntotal,
                "embedding_dimension": self.embedding_dimension,
                "index_type": "IndexFlatIP",
            }
        
        return {"backend": self.backend, "status": "unknown"}
    
    def clear(self) -> None:
        """Clear all entries from the vector store."""
        if self.backend == "memory":
            self._entries.clear()
            self._embeddings = np.array([])
        
        elif self.backend == "chromadb":
            self._collection.delete()
        
        elif self.backend == "faiss":
            self._index = self._faiss.IndexFlatIP(self.embedding_dimension)
            self._entries.clear()
            self._id_mapping.clear()
        
        logger.info("Cleared vector store")
    
    def export_entries(
        self,
        silo_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Export all entries (for backup/migration).
        
        Args:
            silo_filter: Only export entries from these silos
            
        Returns:
            List of entry dictionaries
        """
        exported = []
        
        if self.backend == "memory":
            for entry in self._entries.values():
                if silo_filter and entry.silo_id not in silo_filter:
                    continue
                
                exported.append({
                    "id": entry.id,
                    "embedding": entry.embedding.tolist(),
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp,
                    "silo_id": entry.silo_id,
                })
        
        # Add other backends as needed
        
        logger.info("Exported %d entries", len(exported))
        return exported
