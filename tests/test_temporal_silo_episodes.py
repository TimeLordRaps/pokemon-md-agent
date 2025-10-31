"""Tests for temporal silo episode-aware retrieval and decay weighting."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.temporal_silo import (
    TemporalSiloManager,
    DEFAULT_DECAY_FACTOR_PER_HOUR,
)


def _unit_vector(dim: int, seed: int) -> np.ndarray:
    """Create a reproducible unit vector for test embeddings."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(0, 1, dim).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def test_episode_boundary_detection_with_floor_and_savestate() -> None:
    """Episode ID increments on savestate and floor resets while skipping normal flow."""
    manager = TemporalSiloManager(silos=[1])
    base_time = 1_700_000_000.0
    embedding = _unit_vector(8, seed=42)

    # Initial floor change should bootstrap episode 1.
    first_episode = manager.add_with_episode_boundary(
        embedding=embedding,
        trajectory_id="traj_0",
        metadata={"event": "on_floor_change", "floor": 1},
        current_time=base_time,
        floor=1,
    )
    assert first_episode == 1

    # Progressing to higher floor stays in same episode.
    same_episode = manager.add_with_episode_boundary(
        embedding=embedding,
        trajectory_id="traj_1",
        metadata={"event": "on_floor_change", "floor": 2},
        current_time=base_time + 60,
        floor=2,
    )
    assert same_episode == first_episode

    # Savestate load forces a new episode.
    savestate_episode = manager.add_with_episode_boundary(
        embedding=embedding,
        trajectory_id="traj_2",
        metadata={"event": "savestate_loaded", "savestate_loaded": True, "floor": 2},
        current_time=base_time + 120,
        floor=2,
    )
    assert savestate_episode == first_episode + 1

    # Floor regression back to 1 also triggers a new episode.
    regression_episode = manager.add_with_episode_boundary(
        embedding=embedding,
        trajectory_id="traj_3",
        metadata={"event": "on_floor_change", "floor": 1},
        current_time=base_time + 180,
        floor=1,
    )
    assert regression_episode == savestate_episode + 1


def test_search_with_decay_prefers_recent_entries() -> None:
    """Recency weighting prioritises newer memories with identical embeddings."""
    manager = TemporalSiloManager(silos=[1])
    base_time = 1_700_000_000.0
    embedding = _unit_vector(16, seed=7)

    # Store an older entry (10 hours old).
    manager.store(
        embedding=embedding,
        trajectory_id="old_memory",
        current_time=base_time - (10 * 3600),
        floor=3,
    )

    # Store a very recent entry (1 minute old).
    manager.store(
        embedding=embedding,
        trajectory_id="fresh_memory",
        current_time=base_time - 60,
        floor=3,
    )

    results = manager.search_with_decay(
        query_embedding=embedding,
        top_k=2,
        current_time=base_time,
        decay_factor=DEFAULT_DECAY_FACTOR_PER_HOUR,
    )

    assert len(results) == 2
    # The fresher memory should score higher after decay bias.
    assert results[0].trajectory_id == "fresh_memory"
    assert (results[0].similarity_score or 0.0) > (results[1].similarity_score or 0.0)
    assert results[0].recency_weight > results[1].recency_weight


def test_search_with_decay_rejects_negative_decay() -> None:
    """Negative decay factors are rejected to avoid runaway weighting."""
    manager = TemporalSiloManager(silos=[1])
    embedding = _unit_vector(4, seed=3)
    manager.store(
        embedding=embedding,
        trajectory_id="baseline",
        current_time=1_700_000_000.0,
        floor=1,
    )

    with pytest.raises(ValueError):
        manager.search_with_decay(
            query_embedding=embedding,
            top_k=1,
            decay_factor=-0.5,
            current_time=1_700_000_001.0,
        )


def test_cross_episode_search_reranks_and_meets_latency_budget() -> None:
    """Cross-episode retrieval adds episode context and executes under 100ms for 1000 entries."""
    manager = TemporalSiloManager(silos=[1])
    base_time = 1_700_000_000.0
    query_embedding = _unit_vector(12, seed=11)
    episodes_to_generate = 4
    entries_per_episode = 250

    # Seed distinct episodes and populate them with embeddings.
    for episode_idx in range(episodes_to_generate):
        timestamp = base_time + (episode_idx * 500.0)
        if episode_idx == 0:
            manager.add_with_episode_boundary(
                embedding=query_embedding,
                trajectory_id=f"episode_bootstrap_{episode_idx}",
                metadata={"event": "on_floor_change", "floor": 1},
                current_time=timestamp,
                floor=1,
            )
        else:
            manager.add_with_episode_boundary(
                embedding=query_embedding,
                trajectory_id=f"episode_seed_{episode_idx}",
                metadata={"event": "savestate_loaded", "savestate_loaded": True, "floor": 1},
                current_time=timestamp,
                floor=1,
            )

        for entry_idx in range(entries_per_episode):
            embedding = _unit_vector(12, seed=(episode_idx * 1000) + entry_idx)
            manager.store(
                embedding=embedding,
                trajectory_id=f"ep{episode_idx}_traj_{entry_idx}",
                current_time=timestamp + entry_idx + 1,
                floor=episode_idx + 1,
                episode_id=episode_idx + 1,
            )

    # Ensure we generated at least 1000 entries across silos.
    total_entries: List[int] = [
        len(silo.entries) for silo in manager.silos.values()
    ]
    assert sum(total_entries) >= episodes_to_generate * entries_per_episode

    start = time.perf_counter()
    results = manager.search_across_episodes(
        query_embedding=query_embedding,
        top_k_per_episode=5,
        max_episodes=3,
        current_time=base_time + 10_000,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert duration_ms < 100.0
    assert results, "Expected cross-episode search to return results"

    # Validate ordering and context annotations.
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert result.context.startswith("From episode ")
