"""RAG retrieval system with ANN search and RRF reranking."""

from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import logging
import time
import math
from pathlib import Path
import json

from .schema import TrajectoryEntry, EmbeddingEntry, RetrievalResult, QueryContext

logger = logging.getLogger(__name__)


class ANNIndex:
    """Approximate Nearest Neighbor index for embeddings."""
    
    def __init__(self, dimension: int = 768):
        """Initialize ANN index.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.entries: List[TrajectoryEntry] = []
        self.id_to_idx: Dict[str, int] = {}
        
        # Simple in-memory index (replace with FAISS/Annoy for production)
        logger.info("Initialized in-memory ANN index (dim=%d)", dimension)
    
    def add_entry(self, entry: TrajectoryEntry) -> None:
        """Add entry to index."""
        if entry.id in self.id_to_idx:
            # Update existing
            idx = self.id_to_idx[entry.id]
            self.entries[idx] = entry
        else:
            # Add new
            self.entries.append(entry)
            self.id_to_idx[entry.id] = len(self.entries) - 1
    
    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            
        Returns:
            List of (entry_id, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        # Simple cosine similarity (replace with proper ANN for production)
        similarities = []
        for entry in self.entries:
            sim = self._cosine_similarity(query_vector, entry.emb_vector)
            similarities.append((entry.id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        data = {
            "dimension": self.dimension,
            "entries": [entry.to_dict() for entry in self.entries]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Saved ANN index with %d entries to %s", len(self.entries), path)
    
    def load(self, path: Path) -> None:
        """Load index from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.dimension = data["dimension"]
        self.entries = [TrajectoryEntry.from_dict(entry_data) for entry_data in data["entries"]]
        
        # Rebuild id_to_idx mapping
        self.id_to_idx = {entry.id: idx for idx, entry in enumerate(self.entries)}
        
        logger.info("Loaded ANN index with %d entries from %s", len(self.entries), path)


class RRFCombiner:
    """Reciprocal Rank Fusion combiner for multiple retrieval sources."""
    
    def __init__(self, k: float = 60.0):
        """Initialize RRF combiner.
        
        Args:
            k: RRF parameter (higher = less aggressive reranking)
        """
        self.k = k
    
    def combine(
        self,
        result_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Combine multiple ranked lists using RRF.
        
        Args:
            result_lists: List of (id, score) tuples from different sources
            weights: Optional weights for each source
            
        Returns:
            Combined and reranked results
        """
        if not result_lists:
            return []
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Collect all unique IDs and their ranks per source
        id_ranks: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        for source_idx, result_list in enumerate(result_lists):
            weight = weights[source_idx]
            for rank, (entry_id, score) in enumerate(result_list):
                # RRF score = weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                id_ranks[entry_id].append((rank, rrf_score))
        
        # Calculate final scores
        final_scores = []
        for entry_id, rank_scores in id_ranks.items():
            # Sum RRF scores across sources
            total_score = sum(score for _, score in rank_scores)
            final_scores.append((entry_id, total_score))
        
        # Sort by final score (descending)
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores


class RAGRetrieval:
    """RAG retrieval system with ANN search and RRF reranking."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """Initialize RAG retrieval system.
        
        Args:
            index_path: Path to save/load ANN index
        """
        self.ann_index = ANNIndex()
        self.rrf_combiner = RRFCombiner()
        self.index_path = index_path or Path("data/rag_index.json")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        if self.index_path.exists():
            try:
                self.ann_index.load(self.index_path)
            except Exception as e:
                logger.warning("Failed to load RAG index: %s", e)
        
        logger.info("Initialized RAG retrieval system")
    
    def add_trajectory(self, entry: TrajectoryEntry) -> None:
        """Add trajectory entry to index."""
        self.ann_index.add_entry(entry)
        logger.debug("Added trajectory entry %s", entry.id)
    
    def retrieve(
        self,
        context: QueryContext,
        use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve relevant trajectories.
        
        Args:
            context: Query context
            use_rrf: Whether to use RRF for reranking
            
        Returns:
            List of retrieval results
        """
        start_time = time.time()
        
        # ANN search
        ann_results = self.ann_index.search(context.query_embedding, k=context.max_results * 2)
        
        if not ann_results:
            logger.debug("No ANN results found")
            return []
        
        # Apply filters
        filtered_results = self._apply_filters(ann_results, context)
        
        # Deduplication
        if context.dedup_by_episode:
            filtered_results = self._dedup_by_episode(filtered_results)
        
        # Apply recency bias
        if context.recency_bias > 0:
            filtered_results = self._apply_recency_bias(filtered_results, context.recency_bias)
        
        # Convert to RetrievalResult objects
        results = []
        for rank, (entry_id, score) in enumerate(filtered_results[:context.max_results]):
            entry = self._get_entry_by_id(entry_id)
            if entry:
                result = RetrievalResult(
                    entry=entry,
                    score=score,
                    rank=rank + 1,
                    source="ann"
                )
                results.append(result)
        
        elapsed = time.time() - start_time
        logger.debug("Retrieved %d results in %.3fs", len(results), elapsed)
        
        return results
    
    def _apply_filters(
        self,
        results: List[Tuple[str, float]],
        context: QueryContext
    ) -> List[Tuple[str, float]]:
        """Apply floor and silo filters."""
        filtered = []
        
        for entry_id, score in results:
            entry = self._get_entry_by_id(entry_id)
            if not entry:
                continue
            
            # Floor filter
            if context.current_floor is not None and entry.floor != context.current_floor:
                continue
            
            # Silo filter (prefer same silo, but allow others)
            if context.current_silo is not None and entry.silo != context.current_silo:
                # Reduce score for different silos
                score *= 0.5
            
            filtered.append((entry_id, score))
        
        return filtered
    
    def _dedup_by_episode(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Deduplicate results by episode/silo."""
        seen_silos: Set[str] = set()
        deduped = []
        
        for entry_id, score in results:
            entry = self._get_entry_by_id(entry_id)
            if entry and entry.silo not in seen_silos:
                deduped.append((entry_id, score))
                seen_silos.add(entry.silo)
        
        return deduped
    
    def _apply_recency_bias(
        self,
        results: List[Tuple[str, float]],
        bias_factor: float
    ) -> List[Tuple[str, float]]:
        """Apply recency bias to results."""
        current_time = time.time()
        
        biased_results = []
        for entry_id, score in results:
            entry = self._get_entry_by_id(entry_id)
            if entry:
                # Calculate recency score (newer = higher)
                age_hours = (current_time - entry.timestamp) / 3600
                recency_score = 1.0 / (1.0 + age_hours)  # Decay over time
                
                # Combine original score with recency
                combined_score = score * (1.0 + bias_factor * recency_score)
                biased_results.append((entry_id, combined_score))
        
        # Re-sort by combined score
        biased_results.sort(key=lambda x: x[1], reverse=True)
        return biased_results
    
    def _get_entry_by_id(self, entry_id: str) -> Optional[TrajectoryEntry]:
        """Get trajectory entry by ID."""
        if entry_id in self.ann_index.id_to_idx:
            idx = self.ann_index.id_to_idx[entry_id]
            return self.ann_index.entries[idx]
        return None
    
    def save_index(self) -> None:
        """Save the ANN index to disk."""
        self.ann_index.save(self.index_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "total_entries": len(self.ann_index.entries),
            "index_path": str(self.index_path),
            "dimension": self.ann_index.dimension,
        }