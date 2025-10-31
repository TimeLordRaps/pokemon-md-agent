"""Cross-silo search functionality for temporal resolution retrieval."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from ..embeddings.temporal_silo import TemporalSiloManager, SiloEntry
from .deduplicator import Deduplicator

logger = logging.getLogger(__name__)


@dataclass
class CrossSiloResult:
    """Result from cross-silo search."""
    silo_id: str
    entries: List[Tuple[SiloEntry, float]]  # (entry, similarity)
    aggregated_score: float
    diversity_score: float


@dataclass
class SearchConfig:
    """Configuration for cross-silo search."""
    top_k_per_silo: int = 3
    similarity_threshold: float = 0.7
    diversity_weight: float = 0.3  # Weight for diversity vs similarity
    silo_weights: Optional[Dict[str, float]] = None
    require_multiple_silos: bool = False


class CrossSiloRetriever:
    """Retrieve and aggregate results across multiple temporal silos."""
    
    def __init__(
        self,
        silo_manager: TemporalSiloManager,
        deduplicator: Optional[Deduplicator] = None,
        default_config: Optional[SearchConfig] = None,
    ):
        """Initialize cross-silo retriever.

        Args:
            silo_manager: Temporal silo manager
            deduplicator: Deduplicator instance for content deduplication
            default_config: Default search configuration
        """
        self.silo_manager = silo_manager
        self.deduplicator = deduplicator or Deduplicator()
        self.default_config = default_config or SearchConfig()

        # Default silo weights (favor more recent, higher resolution)
        self.default_silo_weights = {
            "temporal_1frame": 1.0,
            "temporal_2frame": 0.9,
            "temporal_4frame": 0.8,
            "temporal_8frame": 0.7,
            "temporal_16frame": 0.6,
            "temporal_32frame": 0.5,
            "temporal_64frame": 0.4,
        }

        logger.info("Initialized CrossSiloRetriever")
    
    def search(
        self,
        query_embedding: np.ndarray,
        config: Optional[SearchConfig] = None,
        silo_filter: Optional[List[str]] = None,
    ) -> List[CrossSiloResult]:
        """Search across silos with configurable parameters.
        
        Args:
            query_embedding: Query embedding vector
            config: Search configuration
            silo_filter: Only search in these silos
            
        Returns:
            List of CrossSiloResult objects
        """
        search_config = config or self.default_config
        silo_weights = search_config.silo_weights or self.default_silo_weights
        
        # Get silo IDs to search
        all_silo_ids = list(self.silo_manager.silos.keys())
        search_silo_ids = silo_filter or all_silo_ids
        
        if not search_silo_ids:
            logger.warning("No silos specified for search")
            return []
        
        # Search each silo
        silo_results = {}
        
        for silo_id in search_silo_ids:
            if silo_id not in self.silo_manager.silos:
                logger.warning("Silo %s not found", silo_id)
                continue
            
            silo = self.silo_manager.silos[silo_id]
            
            # Search in this silo
            matches = silo.search_similar(
                query_embedding=query_embedding,
                top_k=search_config.top_k_per_silo,
                similarity_threshold=search_config.similarity_threshold,
            )
            
            if matches:
                # Calculate aggregated score for this silo
                silo_weight = silo_weights.get(silo_id, 0.5)
                avg_similarity = np.mean([sim for _, sim in matches])
                diversity_score = self._calculate_diversity([entry for entry, _ in matches])
                
                aggregated_score = float(
                    avg_similarity * (1.0 - search_config.diversity_weight) +
                    diversity_score * search_config.diversity_weight * silo_weight
                )
                
                silo_results[silo_id] = CrossSiloResult(
                    silo_id=silo_id,
                    entries=matches,
                    aggregated_score=aggregated_score,
                    diversity_score=diversity_score,
                )
        
        # Sort by aggregated score
        sorted_results = sorted(
            silo_results.values(),
            key=lambda x: x.aggregated_score,
            reverse=True
        )
        
        # Filter results based on requirements
        final_results = self._filter_results(
            sorted_results,
            search_config,
            search_silo_ids
        )
        
        logger.info(
            "Cross-silo search: %d silos, %d results",
            len(search_silo_ids),
            len(final_results)
        )
        
        return final_results
    
    def search_aggregated(
        self,
        query_embedding: np.ndarray,
        max_results: int = 10,
        config: Optional[SearchConfig] = None,
    ) -> List[Tuple[SiloEntry, float, str]]:
        """Search and aggregate all results into single ranked list.
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
            config: Search configuration
            
        Returns:
            List of (entry, similarity, silo_id) tuples
        """
        silo_results = self.search(query_embedding, config)
        
        # Aggregate all entries
        all_entries = []
        
        for result in silo_results:
            for entry, similarity in result.entries:
                # Weight by silo result score
                weighted_similarity = similarity * result.aggregated_score
                all_entries.append((entry, weighted_similarity, result.silo_id))
        
        # Sort by weighted similarity
        all_entries.sort(key=lambda x: x[1], reverse=True)
        
        return all_entries[:max_results]
    
    def find_complementary_patterns(
        self,
        query_embedding: np.ndarray,
        primary_silo: str,
        config: Optional[SearchConfig] = None,
    ) -> Dict[str, List[Tuple[SiloEntry, float]]]:
        """Find complementary patterns across different temporal resolutions.
        
        Args:
            query_embedding: Query embedding vector
            primary_silo: Primary silo to focus on
            config: Search configuration
            
        Returns:
            Dictionary mapping silo_id to complementary entries
        """
        search_config = config or self.default_config
        search_config = SearchConfig(
            top_k_per_silo=5,  # Get more for pattern analysis
            similarity_threshold=search_config.similarity_threshold * 0.8,  # Lower threshold
            diversity_weight=search_config.diversity_weight,
            silo_weights=search_config.silo_weights,
        )
        
        # Search in primary silo
        primary_results = self.silo_manager.silos[primary_silo].search_similar(
            query_embedding=query_embedding,
            top_k=search_config.top_k_per_silo,
            similarity_threshold=search_config.similarity_threshold,
        )
        
        if not primary_results:
            return {}
        
        # Get other silos
        other_silos = [
            silo_id for silo_id in self.silo_manager.silos.keys()
            if silo_id != primary_silo
        ]
        
        # Find complementary patterns
        complementary = {}
        
        for other_silo in other_silos:
            silo = self.silo_manager.silos[other_silo]
            
            # Find entries that are similar to primary entries
            complementary_entries = []
            
            for primary_entry, primary_similarity in primary_results:
                matches = silo.search_similar(
                    query_embedding=primary_entry.embedding,
                    top_k=2,  # Top 2 complementary matches
                    similarity_threshold=0.6,  # Lower threshold for complementarity
                )
                
                for entry, similarity in matches:
                    # Avoid duplicates
                    if not any(e[0].trajectory_id == entry.trajectory_id 
                              for e in complementary_entries):
                        complementary_entries.append((entry, similarity))
            
            if complementary_entries:
                complementary[other_silo] = complementary_entries
        
        logger.info(
            "Found complementary patterns in %d silos for %s",
            len(complementary),
            primary_silo
        )
        
        return complementary
    
    def analyze_silo_relationships(
        self,
        query_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze relationships between silos for a given query.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Dictionary with silo relationship analysis
        """
        silo_stats = {}
        
        # Search each silo individually
        for silo_id, silo in self.silo_manager.silos.items():
            matches = silo.search_similar(
                query_embedding=query_embedding,
                top_k=5,
                similarity_threshold=0.0,  # No threshold for analysis
            )
            
            if matches:
                similarities = [sim for _, sim in matches]
                silo_stats[silo_id] = {
                    "num_matches": len(matches),
                    "max_similarity": max(similarities),
                    "avg_similarity": np.mean(similarities),
                    "min_similarity": min(similarities),
                    "similarity_std": np.std(similarities),
                }
        
        # Analyze correlations between silos
        correlations = {}
        silo_ids = list(silo_stats.keys())
        
        for i, silo1 in enumerate(silo_ids):
            for silo2 in silo_ids[i+1:]:
                # Calculate correlation based on similarity patterns
                # This is a simplified correlation measure
                stats1 = silo_stats[silo1]
                stats2 = silo_stats[silo2]
                
                correlation = self._calculate_silo_correlation(
                    stats1, stats2, query_embedding
                )
                
                correlations[f"{silo1}_vs_{silo2}"] = correlation
        
        return {
            "silo_stats": silo_stats,
            "correlations": correlations,
            "recommended_silos": self._recommend_silos(silo_stats),
        }
    
    def _calculate_diversity(self, entries: List[SiloEntry]) -> float:
        """Calculate diversity score for a list of entries.
        
        Args:
            entries: List of silo entries
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(entries) < 2:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                distance = 1.0 - self._cosine_similarity(
                    entries[i].embedding, entries[j].embedding
                )
                distances.append(distance)
        
        return float(np.mean(distances))
    
    def _filter_results(
        self,
        results: List[CrossSiloResult],
        config: SearchConfig,
        search_silo_ids: List[str],
    ) -> List[CrossSiloResult]:
        """Filter results based on configuration requirements.
        
        Args:
            results: Raw search results
            config: Search configuration
            search_silo_ids: Silos that were searched
            
        Returns:
            Filtered results
        """
        filtered = results
        
        # Require results from multiple silos
        if config.require_multiple_silos and len(filtered) < 2:
            logger.debug("Filtering: require multiple silos but only found %d", len(filtered))
            return []
        
        # Apply minimum number of silos requirement
        min_silos = 2 if config.require_multiple_silos else 1
        
        if len(filtered) < min_silos:
            logger.debug("Filtering: need at least %d silos, got %d", min_silos, len(filtered))
            return []
        
        return filtered
    
    def _calculate_silo_correlation(self, stats1: Dict, stats2: Dict, query: np.ndarray) -> float:
        """Calculate correlation between two silos.
        
        Args:
            stats1: Statistics for first silo
            stats2: Statistics for second silo
            query: Query embedding
            
        Returns:
            Correlation score
        """
        # Simplified correlation based on similarity patterns
        # In practice, this would be more sophisticated
        
        similarity_correlation = abs(stats1["avg_similarity"] - stats2["avg_similarity"])
        return 1.0 - similarity_correlation  # Higher correlation = more similar patterns
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _recommend_silos(self, silo_stats: Dict[str, Dict]) -> List[str]:
        """Recommend which silos to use based on query.
        
        Args:
            silo_stats: Statistics for each silo
            
        Returns:
            List of recommended silo IDs
        """
        if not silo_stats:
            return []
        
        # Sort by avg similarity
        sorted_silos = sorted(
            silo_stats.items(),
            key=lambda x: x[1]["avg_similarity"],
            reverse=True
        )
        
        # Return top 3 silos
        return [silo_id for silo_id, _ in sorted_silos[:3]]
