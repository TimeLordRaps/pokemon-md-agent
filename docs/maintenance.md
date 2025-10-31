# Temporal Silo Maintenance Daemon

This module provides a lightweight background task that keeps the seven temporal silos tidy without touching the retrieval or vision stacks.

## Overview

`TemporalSiloMaintenanceDaemon` (in `src/retrieval/maint/daemon.py`) calls public maintenance hooks (`compact`, `expire_older_than`) on the temporal silo manager. Runs can be scheduled by wall-clock cadence, step cadence, or invoked manually.

Policies live in `src/retrieval/maint/policies.py`. The defaults mirror the seven-scale temporal layout:

- Fine-grained silos (`temporal_1frame`, `temporal_2frame`) compact within a few seconds and retain roughly an hour of data.
- Coarser silos use larger compaction windows and shorter retentions to control storage cost.

No retrieval path changes are required; the daemon only interacts with write-side maintenance APIs.

## Wiring into the Orchestrator Loop

1. Import the daemon and policies:
   ```python
   from src.retrieval.maint.daemon import TemporalSiloMaintenanceDaemon
   from src.retrieval.maint.policies import default_policies
   ```

2. Instantiate the daemon alongside the existing temporal silo manager. A common pattern is to run maintenance every 60 seconds or every N inference steps:
   ```python
    maintenance = TemporalSiloMaintenanceDaemon(
        target=temporal_manager,
        policies=default_policies(),
        cadence_seconds=60,
        cadence_steps=50,  # optional, trigger every 50 loop iterations
    )
   ```

3. Inside the orchestrator loop, call `maintenance.step()`:
   ```python
    metrics = maintenance.step()
    if metrics:
        telemetry.emit("temporal_silo_maintenance", {
            "duration": metrics.duration_seconds,
            "per_silo_counts": metrics.per_silo_counts,
            "per_silo_bytes": metrics.per_silo_bytes,
        })
   ```

4. To force maintenance during shutdown or long blocking operations, use `maintenance.run(force=True)`.

### Router Glue Integration

`RouterGlue` accepts an optional `maintenance_daemon` parameter. When attached, the daemon is stepped automatically at the end of each `execute_turn_loop` run. The runtime helper in `src/orchestrator/runtime.py` wires this up and returns both the glue instance and the shared daemon:

```python
from src.orchestrator.runtime import build_router_runtime

router, maintenance = build_router_runtime(
    silo_manager=temporal_manager,
    cadence_seconds=60,
    cadence_steps=25,
)
```

The loop can also swap daemons later via `router.attach_maintenance_daemon(maintenance)` if needed.

## Metrics

Every run returns a `MaintenanceMetrics` object with:

- `per_silo_counts`: entries per silo after maintenance
- `per_silo_bytes`: approximate memory footprint per silo
- `total_removed_compaction` / `total_removed_retention`: counts removed per silo in this run
- `duration_seconds`: wall time for the pass

Hook these fields into existing logging or telemetry systems for capacity monitoring.

## Testing

`tests/test_maint_temporal_silos.py` covers execution order, cadence triggers, adapter fallbacks, and metric export. No changes are required in existing test suites; running the standard pytest target will include the new coverage.
