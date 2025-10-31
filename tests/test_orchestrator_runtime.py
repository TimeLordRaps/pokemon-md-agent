"""Tests for orchestrator runtime builder hooking maintenance."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.orchestrator.runtime import build_router_runtime
from src.orchestrator.router_glue import RouterGlue
from src.retrieval.maint.daemon import TemporalSiloMaintenanceDaemon


def test_build_router_runtime_creates_daemon():
    """Builder should create and attach a maintenance daemon when absent."""
    silo_manager = MagicMock()

    router, daemon = build_router_runtime(silo_manager=silo_manager, cadence_seconds=0, cadence_steps=1)

    assert isinstance(router, RouterGlue)
    assert isinstance(daemon, TemporalSiloMaintenanceDaemon)
    assert router.maintenance_daemon is daemon


def test_build_router_runtime_reuses_existing_daemon():
    """Builder should reuse supplied maintenance daemon."""
    silo_manager = MagicMock()
    existing_daemon = TemporalSiloMaintenanceDaemon(target=silo_manager, cadence_seconds=0, cadence_steps=1)

    router, daemon = build_router_runtime(
        silo_manager=silo_manager,
        maintenance_daemon=existing_daemon,
        cadence_seconds=0,
        cadence_steps=1,
    )

    assert daemon is existing_daemon
    assert router.maintenance_daemon is existing_daemon
