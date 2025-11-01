"""Test parallel RRF retrieval with deduplication and recency bias."""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, AsyncMock
from src.retrieval.auto_retrieve import AutoRetriever, RetrievedTrajectory, RetrievalQuery
from src.retrieval.cross_silo_search import CrossSiloRetriever, SearchConfig
from src.embeddings.temporal_silo import TemporalSiloManager, SiloEntry
from src.embeddings.vector_store import VectorStore


@pytest.fixture
def mock_silo_manager():
    """Mock silo manager with multiple silos."""
    manager = Mock(spec=TemporalSiloManager)
    manager.cross_silo_search.return_value = {
        "temporal_1frame": [(Mock(trajectory_id="traj1", embedding=np.random.rand(128), metadata={"episode": "ep1", "timestamp": 1000.0}), 0.9)],
        "temporal_4frame": [(Mock(trajectory_id="traj2", embedding=np.random.rand(128), metadata={"episode": "ep2", "timestamp": 2000.0}), 0.8)],
    }
    manager.silos = {"temporal_1frame": Mock(), "temporal_4frame": Mock()}
    return manager


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    return Mock(spec=VectorStore)


@pytest.fixture
def auto_retriever(mock_silo_manager, mock_vector_store):
    """AutoRetriever instance with mocks."""
    return AutoRetriever(
        silo_manager=mock_silo_manager,
        vector_store=mock_vector_store,
        auto_retrieval_count=3,
        similarity_threshold=0.7
    )


class TestParallelRRFRetrieval:
    """Test parallel RRF retrieval coordination."""

    def test_parallel_rrf_merge_basic(self, auto_retriever, mock_silo_manager):
        """RRF merge combines rankings from multiple sources with reciprocal ranks."""
        query = RetrievalQuery(current_embedding=np.random.rand(128))

        # Mock parallel search results with proper attributes
        mock1 = Mock(
            trajectory_id="g1",
            embedding=np.random.rand(128),
            metadata={"episode": "ep1", "timestamp": 1000.0},
            raw_similarity=0.9,
            similarity_score=0.9,
            timestamp=1000.0,
            recency_weight=1.0
        )
        mock2 = Mock(
            trajectory_id="m1",
            embedding=np.random.rand(128),
            metadata={"episode": "ep2", "timestamp": 2000.0},
            raw_similarity=0.8,
            similarity_score=0.8,
            timestamp=2000.0,
            recency_weight=1.0
        )
        
        # Mock cross_silo_search to return consistent results for parallel calls
        def mock_cross_silo_search(query_embedding, silo_filter=None, top_k=None):
            if silo_filter and "global" in str(silo_filter):
                return {"global": [(mock1, 0.9)]}
            elif silo_filter and any("temporal" in s for s in silo_filter):
                return {"model1": [(mock2, 0.8)]}
            else:
                # Default fallback for any other calls
                return {"fallback": [(mock1, 0.9)]}
        
        mock_silo_manager.cross_silo_search.side_effect = mock_cross_silo_search

        results = asyncio.run(auto_retriever.retrieve_parallel_rrf(query))

        # Assert RRF fusion applied
        assert len(results) <= 3
        # Check deduplication by episode
        episodes = [r.metadata.get("episode") for r in results]
        assert len(set(episodes)) == len(episodes)  # No duplicates

        # Check recency bias (higher weight for recent)
        recent_scores = [r.similarity_score for r in results if r.metadata.get("timestamp", 0) > 1500]
        older_scores = [r.similarity_score for r in results if r.metadata.get("timestamp", 0) <= 1500]
        if recent_scores and older_scores:
            assert max(recent_scores) >= max(older_scores)  # Recency bias

    def test_rrf_fusion_weights_by_rank(self, auto_retriever):
        """RRF assigns higher scores to higher-ranked items across sources."""
        # Create actual RetrievedTrajectory instances for RRF testing
        traj_a = RetrievedTrajectory(
            trajectory_id="A",
            similarity_score=0.9,
            embedding=np.random.rand(128),
            metadata={},
            timestamp=time.time(),
            silo_id="test",
            action_sequence=[]
        )
        traj_b = RetrievedTrajectory(
            trajectory_id="B",
            similarity_score=0.8,
            embedding=np.random.rand(128),
            metadata={},
            timestamp=time.time(),
            silo_id="test",
            action_sequence=[]
        )
        traj_c = RetrievedTrajectory(
            trajectory_id="C",
            similarity_score=0.7,
            embedding=np.random.rand(128),
            metadata={},
            timestamp=time.time(),
            silo_id="test",
            action_sequence=[]
        )
        traj_d = RetrievedTrajectory(
            trajectory_id="D",
            similarity_score=0.5,
            embedding=np.random.rand(128),
            metadata={},
            timestamp=time.time(),
            silo_id="test",
            action_sequence=[]
        )

        # Simulate ranks: item A rank 1 in source1, rank 3 in source2
        # Item B rank 2 in source1, rank 1 in source2
        results = auto_retriever._rrf_merge([
            [traj_a, traj_b, traj_c],  # Source 1: A(1), B(2), C(3)
            [traj_b, traj_a, traj_d]   # Source 2: B(1), A(2), D(3)
        ], k=60)

        # A should have higher RRF score than C due to better average rank
        a_score = next(r.similarity_score for r in results if r.trajectory_id == "A")
        c_score = next(r.similarity_score for r in results if r.trajectory_id == "C")
        assert a_score > c_score

    def test_episode_deduplication(self, auto_retriever):
        """Deduplicate trajectories from same episode, keeping highest score."""
        trajectories = [
            Mock(trajectory_id="t1", similarity_score=0.8, metadata={"episode": "ep1"}),
            Mock(trajectory_id="t2", similarity_score=0.9, metadata={"episode": "ep1"}),
            Mock(trajectory_id="t3", similarity_score=0.7, metadata={"episode": "ep2"}),
        ]

        deduped = auto_retriever._deduplicate_by_episode(trajectories)
        assert len(deduped) == 2  # Two episodes
        # ep1 should keep t2 (higher score)
        ep1_traj = next(t for t in deduped if t.metadata["episode"] == "ep1")
        assert ep1_traj.trajectory_id == "t2"

    def test_recency_bias_application(self, auto_retriever):
        """Apply recency bias with exponential decay."""
        now = 3000.0
        trajectories = [
            RetrievedTrajectory(
                trajectory_id="recent",
                similarity_score=0.8,
                embedding=np.random.rand(128),
                metadata={},
                timestamp=2500.0,  # Recent
                silo_id="test",
                action_sequence=[]
            ),
            RetrievedTrajectory(
                trajectory_id="old",
                similarity_score=0.8,
                embedding=np.random.rand(128),
                metadata={},
                timestamp=1000.0,  # Old
                silo_id="test",
                action_sequence=[]
            ),
        ]

        biased = auto_retriever._apply_recency_bias(trajectories, now=now)
        # Recent should have higher score
        assert biased[0].similarity_score > biased[1].similarity_score

    def test_retrieval_stats_for_router(self, auto_retriever):
        """Generate stats for router: distance > τ, trajectory conflicts."""
        trajectories = [
            Mock(similarity_score=0.9, embedding=np.array([1, 0]), metadata={"episode": "ep1"}),
            Mock(similarity_score=0.8, embedding=np.array([0, 1]), metadata={"episode": "ep2"}),
            Mock(similarity_score=0.7, embedding=np.array([1, 0]), metadata={"episode": "ep1"}),  # Duplicate episode
        ]

        stats = auto_retriever._compute_retrieval_stats(trajectories, distance_threshold=0.5)

        assert "avg_distance" in stats
        assert "conflicts_detected" in stats
        assert stats["conflicts_detected"] > 0  # Same embeddings = conflict
        assert stats["episodes_covered"] == 2  # After dedup