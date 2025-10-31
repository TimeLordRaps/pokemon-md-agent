"""Factory helpers for constructing RouterGlue with maintenance wiring."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple
import logging

from .router_glue import RouterGlue
from src.router.policy_v2 import PolicyV2, ModelSize
from src.retrieval.maint.daemon import TemporalSiloMaintenanceDaemon, MaintenancePolicy
from src.retrieval.maint.policies import default_policies

logger = logging.getLogger(__name__)


def build_router_runtime(
    *,
    silo_manager: Any,
    policy: Optional[PolicyV2] = None,
    maintenance_daemon: Optional[TemporalSiloMaintenanceDaemon] = None,
    maintenance_policies: Optional[Iterable[MaintenancePolicy]] = None,
    cadence_seconds: float = 60.0,
    cadence_steps: Optional[int] = 25,
    prefetch_callback: Optional[Callable[[ModelSize], None]] = None,
    hotswap_callback: Optional[Callable[[ModelSize], None]] = None,
    router_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[RouterGlue, TemporalSiloMaintenanceDaemon]:
    """Create RouterGlue pre-wired with temporal silo maintenance.

    Args:
        silo_manager: Temporal silo manager to maintain.
        policy: Optional PolicyV2 instance (created if omitted).
        maintenance_daemon: Existing maintenance daemon to reuse.
        maintenance_policies: Optional iterable of MaintenancePolicy overrides.
        cadence_seconds: Wall-clock cadence between maintenance passes.
        cadence_steps: Optional step cadence gating maintenance execution.
        prefetch_callback: Optional prefetch callback for RouterGlue.
        hotswap_callback: Optional hotswap callback for RouterGlue.
        router_kwargs: Additional keyword arguments forwarded to RouterGlue.

    Returns:
        Tuple of (RouterGlue instance, TemporalSiloMaintenanceDaemon).
    """
    router_kwargs = dict(router_kwargs or {})
    policy_v2 = policy or PolicyV2()

    if maintenance_daemon is None:
        policies = list(maintenance_policies) if maintenance_policies is not None else default_policies()
        maintenance_daemon = TemporalSiloMaintenanceDaemon(
            target=silo_manager,
            policies=policies,
            cadence_seconds=cadence_seconds,
            cadence_steps=cadence_steps,
        )
        logger.info(
            "Created TemporalSiloMaintenanceDaemon: cadence=%ss steps=%s",
            cadence_seconds,
            cadence_steps,
        )

    router = RouterGlue(
        policy_v2=policy_v2,
        prefetch_callback=prefetch_callback,
        hotswap_callback=hotswap_callback,
        maintenance_daemon=maintenance_daemon,
        **router_kwargs,
    )

    return router, maintenance_daemon
