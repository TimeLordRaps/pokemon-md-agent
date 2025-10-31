"""Temporal silo maintenance policy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional


@dataclass(frozen=True)
class MaintenancePolicy:
    """Policy describing compaction + retention thresholds for a silo."""

    silo_id: str
    compaction_window_seconds: int
    retention_seconds: int

    def compact_window(self) -> int:
        """Return the compaction window in seconds (non-negative)."""
        return max(0, self.compaction_window_seconds)

    def retention_horizon(self) -> int:
        """Return the retention horizon in seconds (non-negative)."""
        return max(0, self.retention_seconds)


def default_policies() -> List[MaintenancePolicy]:
    """Return sensible defaults aligned with seven-scale temporal design."""
    return [
        MaintenancePolicy("temporal_1frame", 2, 60 * 60),          # keep ~1 hour of fine frames
        MaintenancePolicy("temporal_2frame", 4, 2 * 60 * 60),
        MaintenancePolicy("temporal_4frame", 8, 6 * 60 * 60),
        MaintenancePolicy("temporal_8frame", 16, 12 * 60 * 60),
        MaintenancePolicy("temporal_16frame", 32, 24 * 60 * 60),
        MaintenancePolicy("temporal_32frame", 64, 3 * 24 * 60 * 60),
        MaintenancePolicy("temporal_64frame", 128, 7 * 24 * 60 * 60),  # aggressively prune coarse history
    ]


def build_policy_map(
    policies: Optional[Iterable[MaintenancePolicy]],
) -> Dict[str, MaintenancePolicy]:
    """Normalise iterable of policies into a mapping by silo id."""
    mapping: Dict[str, MaintenancePolicy] = {}
    for policy in policies or default_policies():
        mapping[policy.silo_id] = policy
    return mapping


def iter_policies(
    policies: Optional[Mapping[str, MaintenancePolicy]] = None,
) -> Iterator[MaintenancePolicy]:
    """Yield policies in deterministic order for predictable maintenance."""
    policy_map = policies or build_policy_map(None)
    for silo_id in sorted(policy_map.keys()):
        yield policy_map[silo_id]
