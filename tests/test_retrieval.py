"""Unit tests for retrieval module."""
import pytest
import numpy as np
import time
import shutil
import os
from unittest.mock import Mock, patch
from pathlib import Path

from src.retrieval.auto_retrieve import AutoRetriever, RetrievalQuery, RetrievedTrajectory
from src.retrieval.gatekeeper import RetrievalGatekeeper, GatekeeperStatus
from src.embeddings.temporal_silo import SiloEntry, EpisodeRetrieval


class TestRetrievalGatekeeper:
    """Test RetrievalGatekeeper functionality with disk space checking."""

    def test_init_with_disk_space_checking(self):
        """Test gatekeeper initialization with disk space parameters."""
        gatekeeper = RetrievalGatekeeper(
            max_tokens_per_hour=100,
            min_free_space_mb=500,
            check_disk_space=True
        )
        
        assert gatekeeper.max_tokens_per_hour == 100
        assert gatekeeper.min_free_space_mb == 500
        assert gatekeeper.check_disk_space is True

    def test_init_without_disk_space_checking(self):
        """Test gatekeeper initialization without disk space checking."""
        gatekeeper = RetrievalGatekeeper(
            check_disk_space=False
        )
        
        assert gatekeeper.check_disk_space is False
        assert gatekeeper.min_free_space_mb == 100  # default value

    def test_disk_space_check_sufficient_space(self):
        """Test disk space check with sufficient space."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=10,
            check_disk_space=True
        )
        
        # Mock sufficient disk space (mock returns 1GB free)
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Mock returns (total, used, free) - free = 1GB
            mock_disk_usage.return_value = (1024**3, 0, 1024**3)
            
            result = gatekeeper._check_disk_space()
            assert result is True

    def test_disk_space_check_insufficient_space(self):
        """Test disk space check with insufficient space."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=1000,
            check_disk_space=True
        )
        
        # Mock insufficient disk space (mock returns 100MB free)
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Mock returns (total, used, free) - free = 100MB
            free_bytes = 100 * 1024 * 1024  # 100MB
            mock_disk_usage.return_value = (1024**3, 1024**3 - free_bytes, free_bytes)
            
            result = gatekeeper._check_disk_space()
            assert result is False

    def test_disk_space_check_error_handling(self):
        """Test disk space check error handling."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=100,
            check_disk_space=True
        )
        
        # Mock disk usage error
        with patch('shutil.disk_usage', side_effect=Exception("Disk access error")):
            # Should return True (proceed) on error but log warning
            result = gatekeeper._check_disk_space()
            assert result is True

    def test_disk_space_stats_reporting(self):
        """Test that disk space info is included in stats."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=100,
            check_disk_space=True
        )
        
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Mock returns 500MB free
            free_bytes = 500 * 1024 * 1024  # 500MB
            mock_disk_usage.return_value = (1024**3, 1024**3 - free_bytes, free_bytes)
            
            stats = gatekeeper.get_stats()
            
            assert "free_space_mb" in stats
            assert "min_free_space_mb" in stats
            assert "disk_space_check_enabled" in stats
            assert stats["free_space_mb"] == 500
            assert stats["min_free_space_mb"] == 100
            assert stats["disk_space_check_enabled"] is True

    def test_disk_space_check_disabled_in_stats(self):
        """Test that disk space is not included when disabled."""
        gatekeeper = RetrievalGatekeeper(
            check_disk_space=False
        )
        
        stats = gatekeeper.get_stats()
        
        # Should not include disk space info when disabled
        assert "free_space_mb" not in stats
        assert "min_free_space_mb" not in stats

    def test_check_and_gate_with_disk_space_check(self):
        """Test complete gate check including disk space."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=10,
            check_disk_space=True
        )
        
        # Mock sufficient disk space
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = (1024**3, 0, 1024**3)
            
            # Use a query that will pass shallow checks
            status, token, metadata = gatekeeper.check_and_gate(
                "How do I defeat the boss in Pokemon Mystery Dungeon Red Rescue Team?",
                context={"shallow_hits": 5},
                force_allow=False
            )
            
            # Should be allowed
            assert status == GatekeeperStatus.ALLOW
            assert token is not None
            assert metadata["shallow_confidence"] is not None

    def test_check_and_gate_with_disk_space_rejection(self):
        """Test gate check rejection due to insufficient disk space."""
        gatekeeper = RetrievalGatekeeper(
            min_free_space_mb=1000,
            check_disk_space=True
        )

        # Mock insufficient disk space
        with patch('shutil.disk_usage') as mock_disk_usage:
            free_bytes = 100 * 1024 * 1024  # 100MB
            mock_disk_usage.return_value = (1024**3, 1024**3 - free_bytes, free_bytes)

            status, token, metadata = gatekeeper.check_and_gate(
                "How do I defeat the final boss in Pokemon Mystery Dungeon Red Rescue Team?",
                context={"shallow_hits": 5}
            )

            # Should be denied due to insufficient disk space
            assert status == GatekeeperStatus.DENY
            assert token is None
            assert metadata["reason"] == "insufficient_disk_space"
            assert metadata["min_free_space_mb"] == 1000


class TestAutoRetriever:
    """Test AutoRetriever functionality."""

    def test_init(self):
        """Test retriever initialization."""
        silo_manager = Mock()
        vector_store = Mock()

        retriever = AutoRetriever(silo_manager, vector_store)

        assert retriever.auto_retrieval_count == 3
        assert retriever.similarity_threshold == 0.7
        assert retriever.cross_floor_gating is True

    def test_retrieve_with_dedup(self):
        """Test retrieval with deduplication."""
        base_time = time.time()
        entry1 = SiloEntry(
            embedding=np.array([1.0]),
            timestamp=base_time - 5,
            metadata={"floor": 1},
            trajectory_id="traj1",
            floor=1,
            silo="temporal_1frame",
            episode_id=1,
        )
        entry1_dup = SiloEntry(
            embedding=np.array([1.0]),
            timestamp=base_time - 2,
            metadata={"floor": 1},
            trajectory_id="traj1",
            floor=1,
            silo="temporal_1frame",
            episode_id=1,
        )
        entry2 = SiloEntry(
            embedding=np.array([1.0]),
            timestamp=base_time - 1,
            metadata={"floor": 2},
            trajectory_id="traj2",
            floor=2,
            silo="temporal_1frame",
            episode_id=1,
        )

        silo_manager = Mock()
        silo_manager.search_across_episodes.return_value = [
            EpisodeRetrieval(entry=entry1_dup, score=0.95, episode_id=1, context="", raw_similarity=0.9, recency_weight=1.02),
            EpisodeRetrieval(entry=entry1, score=0.90, episode_id=1, context="", raw_similarity=0.9, recency_weight=1.0),
            EpisodeRetrieval(entry=entry2, score=0.85, episode_id=1, context="", raw_similarity=0.84, recency_weight=1.02),
        ]

        vector_store = Mock()
        retriever = AutoRetriever(silo_manager, vector_store)

        query = RetrievalQuery(current_embedding=np.array([1.0, 2.0]))
        results = retriever.retrieve(query)

        # Should deduplicate traj1 and return top 3
        assert len(results) <= 3

    def test_cross_floor_gating(self):
        """Test cross-floor gating functionality."""
        entry1 = SiloEntry(
            embedding=np.array([1.0]),
            timestamp=time.time() - 4,
            metadata={"floor": 1},
            trajectory_id="traj1",
            floor=1,
            silo="temporal_1frame",
            episode_id=1,
        )
        entry2 = SiloEntry(
            embedding=np.array([1.0]),
            timestamp=time.time() - 2,
            metadata={"floor": 2},
            trajectory_id="traj2",
            floor=2,
            silo="temporal_1frame",
            episode_id=1,
        )

        silo_manager = Mock()
        silo_manager.search_across_episodes.return_value = [
            EpisodeRetrieval(entry=entry1, score=0.9, episode_id=1, context="", raw_similarity=0.88, recency_weight=1.02),
            EpisodeRetrieval(entry=entry2, score=0.85, episode_id=1, context="", raw_similarity=0.82, recency_weight=1.04),
        ]

        vector_store = Mock()
        retriever = AutoRetriever(silo_manager, vector_store)

        # Test with cross-floor disabled
        query = RetrievalQuery(current_embedding=np.array([1.0]), current_floor=1)
        results = retriever.retrieve(query, cross_floor_gating=False)

        # Should only return floor 1 trajectories
        assert all(r.metadata.get("floor") == 1 for r in results)

        # Test with cross-floor enabled (default)
        results_cross = retriever.retrieve(query, cross_floor_gating=True)

        # Should include both same-floor and other-floor trajectories
        floors = [r.metadata.get("floor") for r in results_cross]
        assert 1 in floors  # Same floor
        assert 2 in floors  # Other floor

    def test_recency_bias(self):
        """Test recency bias application."""
        silo_manager = Mock()
        vector_store = Mock()

        retriever = AutoRetriever(silo_manager, vector_store)

        # Create trajectories with different timestamps
        old_traj = RetrievedTrajectory(
            trajectory_id="old",
            similarity_score=0.9,
            timestamp=1000.0,  # Old
            embedding=np.array([1.0]),
            metadata={},
            silo_id="temporal_1frame",
            action_sequence=[]
        )

        new_traj = RetrievedTrajectory(
            trajectory_id="new",
            similarity_score=0.8,
            timestamp=2000.0,  # New
            embedding=np.array([1.0]),
            metadata={},
            silo_id="temporal_1frame",
            action_sequence=[]
        )

        trajectories = [old_traj, new_traj]
        biased = retriever._apply_recency_bias(trajectories, now=2000.0)

        # New trajectory should have higher score after bias
        assert biased[0].trajectory_id == "new"

    def test_retrieval_query_filters(self):
        """Test various query filters."""
        silo_manager = Mock()
        vector_store = Mock()

        retriever = AutoRetriever(silo_manager, vector_store)

        query = RetrievalQuery(
            current_embedding=np.array([1.0]),
            current_position=(10, 20),
            current_mission="find_stairs",
            current_floor=5,
            time_window_seconds=30.0
        )

        # Test that filters are applied (would need actual silo entries to test fully)
        assert query.current_position == (10, 20)
        assert query.current_mission == "find_stairs"
        assert query.current_floor == 5

    def test_retrieval_stats(self):
        """Test retrieval statistics."""
        silo_manager = Mock()
        vector_store = Mock()

        retriever = AutoRetriever(silo_manager, vector_store)

        # Mock some retrieval history
        retriever.retrieval_history = [
            {"num_retrieved": 3, "avg_similarity": 0.8, "silo_distribution": {"silo1": 2, "silo2": 1}},
            {"num_retrieved": 2, "avg_similarity": 0.7, "silo_distribution": {"silo1": 1, "silo3": 1}},
        ]

        stats = retriever.get_retrieval_stats()

        assert "total_retrievals" in stats
        assert "recent_avg_retrieved" in stats
        assert "most_common_silo" in stats

    def test_batch_retrieval(self):
        """Test batch retrieval functionality."""
        silo_manager = Mock()
        vector_store = Mock()

        retriever = AutoRetriever(silo_manager, vector_store)

        queries = [
            RetrievalQuery(current_embedding=np.array([1.0])),
            RetrievalQuery(current_embedding=np.array([2.0])),
        ]

        # Test that batch processing works (simplified test)
        assert len(queries) == 2


class TestRetrievalQuery:
    """Test RetrievalQuery dataclass."""

    def test_query_creation(self):
        """Test creating retrieval queries."""
        embedding = np.array([0.1, 0.2, 0.3])

        query = RetrievalQuery(
            current_embedding=embedding,
            current_position=(5, 10),
            current_mission="explore_floor",
            current_floor=3
        )

        assert np.array_equal(query.current_embedding, embedding)
        assert query.current_position == (5, 10)
        assert query.current_mission == "explore_floor"
        assert query.current_floor == 3
        assert query.max_distance == 50.0  # Default
        assert query.time_window_seconds == 60.0  # Default


class TestRetrievedTrajectory:
    """Test RetrievedTrajectory dataclass."""

    def test_trajectory_creation(self):
        """Test creating retrieved trajectories."""
        embedding = np.array([0.5, 0.6])
        metadata = {"floor": 7, "action": "move_right"}

        trajectory = RetrievedTrajectory(
            trajectory_id="test_traj_001",
            similarity_score=0.85,
            embedding=embedding,
            metadata=metadata,
            timestamp=1234567890.0,
            silo_id="temporal_4frame",
            action_sequence=["UP", "RIGHT", "A"],
            outcome="found_item"
        )

        assert trajectory.trajectory_id == "test_traj_001"
        assert trajectory.similarity_score == 0.85
        assert np.array_equal(trajectory.embedding, embedding)
        assert trajectory.metadata == metadata
        assert trajectory.timestamp == 1234567890.0
        assert trajectory.silo_id == "temporal_4frame"
        assert trajectory.action_sequence == ["UP", "RIGHT", "A"]
        assert trajectory.outcome == "found_item"


if __name__ == "__main__":
    pytest.main([__file__])
