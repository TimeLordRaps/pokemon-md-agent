"""Maintenance daemon orchestrating temporal silo compaction and retention."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

from .policies import MaintenancePolicy, build_policy_map, iter_policies

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceMetrics:
    """Snapshot of maintenance side-effects."""

    per_silo_counts: Dict[str, int] = field(default_factory=dict)
    per_silo_bytes: Dict[str, int] = field(default_factory=dict)
    total_removed_compaction: Dict[str, int] = field(default_factory=dict)
    total_removed_retention: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


class TemporalSiloMaintenanceDaemon:
    """Schedules compact/expire passes for temporal silo managers."""

    def __init__(
        self,
        target: object,
        policies: Optional[Iterable[MaintenancePolicy]] = None,
        cadence_seconds: float = 60.0,
        cadence_steps: Optional[int] = None,
        logger_override: Optional[logging.Logger] = None,
    ) -> None:
        self._target = target
        self._policy_map = build_policy_map(policies)
        self._cadence_seconds = max(0.0, cadence_seconds)
        self._cadence_steps = cadence_steps if cadence_steps is None else max(1, cadence_steps)
        self._last_run_time: float = 0.0
        self._step_counter: int = 0
        self._logger = logger_override or logger

    def step(self, force: bool = False) -> Optional[MaintenanceMetrics]:
        """Advance the daemon; run maintenance when cadence triggers."""
        self._step_counter += 1
        if not force and not self._should_run():
            return None
        return self.run(force=force)

    def run(self, force: bool = False) -> MaintenanceMetrics:
        """Execute maintenance immediately, bypassing cadence when forced."""
        now = time.time()
        if not force:
            if self._cadence_seconds > 0.0 and (now - self._last_run_time) < self._cadence_seconds:
                return MaintenanceMetrics()
            if self._cadence_steps is not None and self._step_counter % self._cadence_steps != 0:
                return MaintenanceMetrics()

        start_time = time.time()
        compaction_totals: Dict[str, int] = {}
        retention_totals: Dict[str, int] = {}

        for policy in iter_policies(self._policy_map):
            compact_removed = self._invoke_compact(policy)
            expire_removed = self._invoke_retention(policy)
            if compact_removed:
                compaction_totals[policy.silo_id] = compaction_totals.get(policy.silo_id, 0) + compact_removed
            if expire_removed:
                retention_totals[policy.silo_id] = retention_totals.get(policy.silo_id, 0) + expire_removed

        metrics = self._collect_metrics()
        metrics.total_removed_compaction = compaction_totals
        metrics.total_removed_retention = retention_totals
        metrics.duration_seconds = time.time() - start_time

        self._last_run_time = now
        self._logger.debug(
            "Maintenance completed in %.3fs (compact=%s, expire=%s)",
            metrics.duration_seconds,
            compaction_totals,
            retention_totals,
        )
        return metrics

    def _should_run(self) -> bool:
        if self._cadence_steps is not None and (self._step_counter % self._cadence_steps) == 0:
            return True
        if self._cadence_seconds <= 0.0:
            return False
        now = time.time()
        return (now - self._last_run_time) >= self._cadence_seconds

    def _invoke_compact(self, policy: MaintenancePolicy) -> int:
        window = policy.compact_window()
        if window <= 0:
            return 0

        compact_fn = getattr(self._target, "compact", None)
        if callable(compact_fn):
            try:
                removed = compact_fn(policy.silo_id, window)
                return int(removed or 0)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Compaction failed for %s: %s", policy.silo_id, exc)
                return 0

        # Fallback to per-silo adapters
        silo = self._get_silo(policy.silo_id)
        if silo and hasattr(silo, "compact"):
            try:
                removed = silo.compact(window)
                return int(removed or 0)
            except Exception as exc:  # pragma: no cover
                self._logger.warning("Per-silo compaction failed for %s: %s", policy.silo_id, exc)
        return 0

    def _invoke_retention(self, policy: MaintenancePolicy) -> int:
        horizon = policy.retention_horizon()
        if horizon <= 0:
            return 0

        expire_fn = getattr(self._target, "expire_older_than", None)
        if callable(expire_fn):
            try:
                removed = expire_fn(horizon)
                return int(removed or 0)
            except Exception as exc:  # pragma: no cover
                self._logger.warning("Retention failed for %s horizon=%s: %s", policy.silo_id, horizon, exc)
                return 0

        silo = self._get_silo(policy.silo_id)
        if silo and hasattr(silo, "expire_older_than"):
            try:
                now = time.time()
                cutoff = now - float(horizon)
                removed = silo.expire_older_than(cutoff)
                return int(removed or 0)
            except Exception as exc:  # pragma: no cover
                self._logger.warning("Per-silo retention failed for %s: %s", policy.silo_id, exc)
        return 0

    def _get_silo(self, silo_id: str):
        silos = getattr(self._target, "silos", None)
        if isinstance(silos, Mapping):
            return silos.get(silo_id)
        return None

    def _collect_metrics(self) -> MaintenanceMetrics:
        metrics = MaintenanceMetrics()
        try:
            stats_fn = getattr(self._target, "get_silo_stats", None)
            if callable(stats_fn):
                stats = stats_fn()
                if isinstance(stats, Mapping):
                    per_silo_counts: Dict[str, int] = {}
                    per_silo_bytes: Dict[str, int] = {}
                    for silo_id, data in stats.items():
                        total_entries = int(data.get("total_entries", 0))
                        per_silo_counts[silo_id] = total_entries
                        approx_bytes = data.get("approx_bytes")
                        if approx_bytes is None and "average_embedding_dim" in data:
                            approx_bytes = total_entries * data["average_embedding_dim"] * 4
                        if approx_bytes is not None:
                            per_silo_bytes[silo_id] = int(approx_bytes)
                    metrics.per_silo_counts = per_silo_counts
                    metrics.per_silo_bytes = per_silo_bytes
                    return metrics
        except Exception as exc:  # pragma: no cover
            self._logger.debug("Metric collection via get_silo_stats failed: %s", exc)

        # Fallback: inspect silos directly
        silos = getattr(self._target, "silos", None)
        if isinstance(silos, Mapping):
            counts: Dict[str, int] = {}
            approx_bytes: Dict[str, int] = {}
            for silo_id, silo in silos.items():
                entries = getattr(silo, "entries", None)
                if entries is None:
                    continue
                counts[silo_id] = len(entries)
                bytes_total = 0
                for entry in entries:
                    embedding = getattr(entry, "embedding", None)
                    if embedding is not None and hasattr(embedding, "nbytes"):
                        bytes_total += int(getattr(embedding, "nbytes"))
                if bytes_total:
                    approx_bytes[silo_id] = bytes_total
            metrics.per_silo_counts = counts
            metrics.per_silo_bytes = approx_bytes
        return metrics
