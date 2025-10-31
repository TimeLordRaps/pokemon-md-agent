"""Tests for temporal silo maintenance daemon."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from src.retrieval.maint.daemon import TemporalSiloMaintenanceDaemon
from src.retrieval.maint.policies import MaintenancePolicy


class DummyEntry:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding


class DummySilo:
    def __init__(self, entries=None):
        self.entries = entries or []
        self.compacted: list[int] = []
        self.expired: list[float] = []

    def compact(self, window: int) -> int:
        self.compacted.append(window)
        return 1

    def expire_older_than(self, cutoff: float) -> int:
        self.expired.append(cutoff)
        return 2


def _make_target_with_methods():
    target = MagicMock()
    target.compact.return_value = 1
    target.expire_older_than.return_value = 2
    target.get_silo_stats.return_value = {
        "temporal_1frame": {"total_entries": 42, "approx_bytes": 1024},
        "temporal_2frame": {"total_entries": 7, "approx_bytes": 256},
    }
    return target


def test_daemon_orders_compact_then_expire():
    policies = [
        MaintenancePolicy("temporal_1frame", compaction_window_seconds=4, retention_seconds=300),
        MaintenancePolicy("temporal_2frame", compaction_window_seconds=8, retention_seconds=900),
    ]
    target = _make_target_with_methods()
    daemon = TemporalSiloMaintenanceDaemon(target, policies=policies, cadence_seconds=0)

    metrics = daemon.run(force=True)

    expected_calls = [
        call("temporal_1frame", 4),
        call("temporal_2frame", 8),
    ]
    target.compact.assert_has_calls(expected_calls)
    target.expire_older_than.assert_has_calls([call(300), call(900)])
    assert metrics.total_removed_compaction == {"temporal_1frame": 1, "temporal_2frame": 1}
    assert metrics.total_removed_retention == {"temporal_1frame": 2, "temporal_2frame": 2}
    assert metrics.per_silo_counts["temporal_1frame"] == 42
    assert metrics.per_silo_bytes["temporal_2frame"] == 256


def test_daemon_step_cadence_by_steps(monkeypatch):
    policies = [MaintenancePolicy("temporal_1frame", 2, 60)]
    target = _make_target_with_methods()
    daemon = TemporalSiloMaintenanceDaemon(target, policies=policies, cadence_steps=2, cadence_seconds=0)

    assert daemon.step() is None  # step 1
    metrics = daemon.step()       # step 2 triggers
    assert metrics.total_removed_compaction == {"temporal_1frame": 1}
    assert target.compact.call_count == 1


def test_daemon_adapter_for_silo_objects(monkeypatch):
    silo_a = DummySilo(entries=[DummyEntry(np.zeros(4)) for _ in range(3)])
    silo_b = DummySilo(entries=[DummyEntry(np.zeros(8)) for _ in range(2)])
    target = MagicMock()
    target.silos = {
        "temporal_1frame": silo_a,
        "temporal_2frame": silo_b,
    }
    target.compact = None
    target.expire_older_than = None

    policies = [
        MaintenancePolicy("temporal_1frame", 5, 120),
        MaintenancePolicy("temporal_2frame", 10, 240),
    ]
    daemon = TemporalSiloMaintenanceDaemon(target, policies=policies, cadence_seconds=0)

    metrics = daemon.run(force=True)

    assert silo_a.compacted == [5]
    assert len(silo_a.expired) == 1
    assert silo_b.compacted == [10]
    assert len(silo_b.expired) == 1
    assert metrics.per_silo_counts == {"temporal_1frame": 3, "temporal_2frame": 2}
    # bytes should approximate numpy nbytes
    assert metrics.per_silo_bytes["temporal_1frame"] == 3 * np.zeros(4).nbytes


def test_daemon_respects_retention_zero():
    policies = [MaintenancePolicy("temporal_1frame", 5, 0)]
    target = _make_target_with_methods()
    daemon = TemporalSiloMaintenanceDaemon(target, policies=policies, cadence_seconds=0)

    metrics = daemon.run(force=True)

    target.expire_older_than.assert_not_called()
    assert metrics.total_removed_retention == {}


def test_daemon_handles_metric_fallback(monkeypatch):
    target = MagicMock()
    target.compact = MagicMock(return_value=0)
    target.expire_older_than = MagicMock(return_value=0)
    target.get_silo_stats.side_effect = RuntimeError("boom")
    entry = DummyEntry(np.zeros(3))
    target.silos = {"temporal_1frame": DummySilo(entries=[entry])}

    policies = [MaintenancePolicy("temporal_1frame", 2, 60)]
    daemon = TemporalSiloMaintenanceDaemon(target, policies=policies, cadence_seconds=0)

    metrics = daemon.run(force=True)

    assert metrics.per_silo_counts == {"temporal_1frame": 1}
    assert metrics.per_silo_bytes["temporal_1frame"] == entry.embedding.nbytes
