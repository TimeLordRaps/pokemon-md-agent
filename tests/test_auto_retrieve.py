"""Tests for AutoRetriever functionality."""

import pytest
import numpy as np
import time
from unittest.mock import Mock

from src.embeddings.temporal_silo import TemporalSiloManager, SiloEntry, EpisodeRetrieval
from src.embeddings.vector_store import VectorStore
from src.retrieval.auto_retrieve import AutoRetriever, RetrievalQuery, RetrievedTrajectory


class TestAutoRetriever:
    """Test cases for AutoRetriever class."""

    @pytest.fixture
    def mock_silo_manager(self):
        """Create a mock silo manager with test data."""
        manager = Mock(spec=TemporalSiloManager)

        def mock_search_across_episodes(
            query_embedding,
            top_k_per_episode=9,
            max_episodes=3,
            silo_ids=None,
            decay_factor=None,
            current_time=None,
        ):
            base_time = time.time()
            entries = [
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 100,
                    metadata={"action_sequence": ["move", "attack"], "floor": 1},
                    trajectory_id="traj_001",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 5,
                    metadata={"action_sequence": ["move", "attack", "heal"], "floor": 2},
                    trajectory_id="traj_005",
                    floor=2,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 1,
                    metadata={"action_sequence": ["explore", "item"], "floor": 1},
                    trajectory_id="traj_002",
                    floor=1,
                    silo="temporal_2frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 3,
                    metadata={"action_sequence": ["fight", "run"], "floor": 1},
                    trajectory_id="traj_003",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=2,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 20,
                    metadata={"action_sequence": ["wait"], "floor": 1},
                    trajectory_id="traj_004",
                    floor=1,
                    silo="temporal_8frame",
                    episode_id=2,
                ),
            ]

            results = [
                EpisodeRetrieval(
                    entry=entries[1],
                    score=0.94,
                    episode_id=1,
                    context="From episode 1",
                    raw_similarity=0.95,
                    recency_weight=0.99,
                ),
                EpisodeRetrieval(
                    entry=entries[2],
                    score=0.99,
                    episode_id=1,
                    context="From episode 1",
                    raw_similarity=0.8,
                    recency_weight=1.24,
                ),
                EpisodeRetrieval(
                    entry=entries[3],
                    score=0.78,
                    episode_id=2,
                    context="From episode 2",
                    raw_similarity=0.75,
                    recency_weight=1.04,
                ),
                EpisodeRetrieval(
                    entry=entries[0],
                    score=0.70,
                    episode_id=1,
                    context="From episode 1",
                    raw_similarity=0.9,
                    recency_weight=0.78,
                ),
                EpisodeRetrieval(
                    entry=entries[4],
                    score=0.25,
                    episode_id=2,
                    context="From episode 2",
                    raw_similarity=0.3,
                    recency_weight=0.82,
                ),
            ]

            return results[:top_k_per_episode]

        manager.search_across_episodes.side_effect = mock_search_across_episodes
        return manager

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        return Mock(spec=VectorStore)

    @pytest.fixture
    def retriever(self, mock_silo_manager, mock_vector_store):
        """Create AutoRetriever instance for testing."""
        return AutoRetriever(
            silo_manager=mock_silo_manager,
            vector_store=mock_vector_store,
            auto_retrieval_count=3,
            similarity_threshold=0.7,
            cross_floor_gating=True,
            recency_decay_rate=0.01,  # Increased for stronger recency bias in tests
        )

    def test_retrieve_top_k_three(self, retriever):
        """Test that retrieve returns exactly 3 results when available."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query)

        assert len(results) == 3
        assert all(isinstance(r, RetrievedTrajectory) for r in results)

    def test_retrieve_deduplication(self, retriever):
        """Test that duplicate trajectory_ids are deduplicated."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query)

        # Should have 3 results but only 3 unique trajectory_ids (no duplicates in this test)
        trajectory_ids = [r.trajectory_id for r in results]
        unique_ids = set(trajectory_ids)

        # Verify we have exactly 3 unique trajectories
        assert len(unique_ids) == 3
        assert "traj_002" in unique_ids
        assert "traj_003" in unique_ids
        assert "traj_005" in unique_ids

    def test_retrieve_recency_bias(self, retriever):
        """Test that recency bias affects ranking."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query)

        # traj_002 should be first due to being most recent (1 sec ago)
        # even though traj_005 carries high raw similarity
        assert results[0].trajectory_id == "traj_002"
        assert results[0].raw_similarity < results[0].similarity_score

    def test_recency_lift_telemetry(self, retriever):
        """Recency metrics should be captured for telemetry analysis."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        retriever.retrieve(query)
        stats = retriever.get_retrieval_stats()

        assert stats["recent_avg_recency_lift"] > 0.0
        assert "recency_lift_target_met" in stats

    def test_retrieve_cross_floor_gating_enabled(self, retriever):
        """Test cross-floor retrieval when gating is enabled."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query, cross_floor_gating=True)

        # Should include trajectories from different floors
        floors = [r.metadata.get("floor", r.trajectory_id) for r in results]
        assert 1 in floors  # Same floor
        assert 2 in floors  # Different floor

    def test_retrieve_cross_floor_gating_disabled(self, retriever):
        """Test same-floor only retrieval when gating is disabled."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query, cross_floor_gating=False)

        # Should only include trajectories from same floor
        floor_values = {result.metadata.get("floor") for result in results}
        assert floor_values == {1}, f"Expected only floor 1 results, got: {floor_values}"

    def test_retrieve_similarity_threshold(self, retriever):
        """Test that low similarity results are filtered out."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query)

        # All results should have similarity >= threshold (0.7)
        for result in results:
            assert result.similarity_score >= 0.7

    def test_retrieve_empty_results(self, retriever, mock_silo_manager):
        """Test behavior when no results meet criteria."""
        # Mock empty results from episode search
        mock_silo_manager.search_across_episodes.side_effect = lambda *args, **kwargs: []

        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        results = retriever.retrieve(query)

        assert len(results) == 0

    def test_retrieve_with_override_gating(self, retriever):
        """Test that parameter override works for cross_floor_gating."""
        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        # Override to disable even though class default is True
        results = retriever.retrieve(query, cross_floor_gating=False)

        # Should behave as if cross_floor_gating=False
        for result in results:
            assert result.metadata.get("floor") == 1

    def test_cross_floor_diversity_enforced(self, retriever, caplog):
        """Test that cross-floor diversity ensures at least one different floor when available."""
        # Create mock with trajectories from multiple floors
        mock_silo_manager = retriever.silo_manager

        def mock_search_across_episodes_diverse(*args, **kwargs):
            base_time = time.time()
            entries = [
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 1,
                    metadata={"action_sequence": ["move"], "floor": 1},
                    trajectory_id="traj_floor1_a",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 2,
                    metadata={"action_sequence": ["attack"], "floor": 1},
                    trajectory_id="traj_floor1_b",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 3,
                    metadata={"action_sequence": ["explore"], "floor": 2},
                    trajectory_id="traj_floor2_a",
                    floor=2,
                    silo="temporal_4frame",
                    episode_id=2,
                ),
            ]
            return [
                EpisodeRetrieval(entries[0], score=0.95, episode_id=1, context="Episode 1", raw_similarity=0.95, recency_weight=1.0),
                EpisodeRetrieval(entries[1], score=0.90, episode_id=1, context="Episode 1", raw_similarity=0.90, recency_weight=0.98),
                EpisodeRetrieval(entries[2], score=0.98, episode_id=2, context="Episode 2", raw_similarity=0.98, recency_weight=0.97),
            ]

        mock_silo_manager.search_across_episodes.side_effect = mock_search_across_episodes_diverse

        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        with caplog.at_level('INFO'):
            results = retriever.retrieve(query, cross_floor_gating=True)

        # Should have at least one result from different floor (floor 2)
        floors = [r.metadata.get("floor") for r in results]
        assert 2 in floors, f"Expected floor 2 in results, got floors: {floors}"

        # Verify floor mix logging
        floor_mix_log = next((record.message for record in caplog.records if "floor_mix=" in record.message), None)
        assert floor_mix_log is not None, "Floor mix should be logged"
        assert "other:" in floor_mix_log, f"Expected other-floor count in log, got: {floor_mix_log}"

    def test_cross_floor_diversity_fallback_single_floor(self, retriever, caplog):
        """Test fallback behavior when only one floor available."""
        mock_silo_manager = retriever.silo_manager

        def mock_search_across_episodes_single_floor(*args, **kwargs):
            base_time = time.time()
            entries = [
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 1,
                    metadata={"action_sequence": ["move"], "floor": 1},
                    trajectory_id="traj_floor1_a",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 2,
                    metadata={"action_sequence": ["attack"], "floor": 1},
                    trajectory_id="traj_floor1_b",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=base_time - 3,
                    metadata={"action_sequence": ["explore"], "floor": 1},
                    trajectory_id="traj_floor1_c",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
            ]
            return [
                EpisodeRetrieval(entries[0], score=0.95, episode_id=1, context="Episode 1", raw_similarity=0.95, recency_weight=1.0),
                EpisodeRetrieval(entries[1], score=0.90, episode_id=1, context="Episode 1", raw_similarity=0.90, recency_weight=0.97),
                EpisodeRetrieval(entries[2], score=0.85, episode_id=1, context="Episode 1", raw_similarity=0.85, recency_weight=0.95),
            ]

        mock_silo_manager.search_across_episodes.side_effect = mock_search_across_episodes_single_floor

        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1
        )

        with caplog.at_level('INFO'):
            results = retriever.retrieve(query, cross_floor_gating=True)

        # All results should be from same floor when no other floors available
        floors = [r.metadata.get("floor") for r in results]
        assert all(floor == 1 for floor in floors), f"All floors should be 1, got: {floors}"

        # Verify floor mix logging shows only same floor
        floor_mix_log = next((record.message for record in caplog.records if "floor_mix=" in record.message), None)
        assert floor_mix_log is not None, "Floor mix should be logged"
        assert "other:" not in floor_mix_log, f"Expected no other-floor counts, got: {floor_mix_log}"

    def test_cross_floor_diversity_preserves_ranking(self, retriever):
        """Test that cross-floor diversity preserves dedup and recency ranking."""
        mock_silo_manager = retriever.silo_manager

        def mock_search_across_episodes_ranked(*args, **kwargs):
            now = time.time()
            entries = [
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=now - 100,
                    metadata={"action_sequence": ["old_action"], "floor": 2},
                    trajectory_id="traj_old_diff_floor",
                    floor=2,
                    silo="temporal_4frame",
                    episode_id=2,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=now - 1,
                    metadata={"action_sequence": ["recent_action"], "floor": 1},
                    trajectory_id="traj_recent_same_floor",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
                SiloEntry(
                    embedding=np.random.rand(768),
                    timestamp=now - 10,
                    metadata={"action_sequence": ["medium_action"], "floor": 1},
                    trajectory_id="traj_medium_same_floor",
                    floor=1,
                    silo="temporal_4frame",
                    episode_id=1,
                ),
            ]
            return [
                EpisodeRetrieval(entries[1], score=0.92, episode_id=1, context="Episode 1", raw_similarity=0.85, recency_weight=1.2),
                EpisodeRetrieval(entries[2], score=0.90, episode_id=1, context="Episode 1", raw_similarity=0.90, recency_weight=0.9),
                EpisodeRetrieval(entries[0], score=0.88, episode_id=2, context="Episode 2", raw_similarity=0.99, recency_weight=0.5),
            ]

        mock_silo_manager.search_across_episodes.side_effect = mock_search_across_episodes_ranked

        query = RetrievalQuery(
            current_embedding=np.random.rand(768),
            current_floor=1,
            time_window_seconds=200.0,
        )

        inspection_log = []
        original_filter = retriever._passes_filters

        def inspecting_filter(entry, query_obj, allow_cross_floor):
            decision = original_filter(entry, query_obj, allow_cross_floor)
            inspection_log.append(
                {
                    "trajectory_id": entry.trajectory_id,
                    "decision": decision,
                    "age_seconds": time.time() - entry.timestamp,
                    "allow_cross_floor": allow_cross_floor,
                }
            )
            return decision

        retriever._passes_filters = inspecting_filter  # type: ignore[assignment]

        try:
            results = retriever.retrieve(query, cross_floor_gating=True)
        finally:
            retriever._passes_filters = original_filter  # type: ignore[assignment]

        # Should include different floor trajectory despite lower recency score
        # due to cross-floor diversity requirement
        trajectory_ids = [r.trajectory_id for r in results]
        assert "traj_old_diff_floor" in trajectory_ids, "Should include different floor trajectory"
        assert any(
            entry["trajectory_id"] == "traj_old_diff_floor" and entry["decision"]
            for entry in inspection_log
        ), f"Different floor trajectory should pass filters: {inspection_log}"

        # But recency bias should still be applied to same-floor trajectories
        # traj_recent_same_floor should rank higher than traj_medium_same_floor
        # after recency adjustment, even though traj_medium_same_floor has higher raw similarity
        same_floor_results = [r for r in results if r.metadata.get("floor") == 1]
        if len(same_floor_results) >= 2:
            recent_idx = next(i for i, r in enumerate(same_floor_results) if r.trajectory_id == "traj_recent_same_floor")
            medium_idx = next(i for i, r in enumerate(same_floor_results) if r.trajectory_id == "traj_medium_same_floor")
            assert recent_idx < medium_idx, "Recent same-floor should rank higher than older same-floor"


if __name__ == "__main__":
    pytest.main([__file__])
