"""Telemetry logging for agent orchestrator.

Provides JSONL per-step logging with fields: model, vt_total, tokens, latency_ms,
fps, router_decision, rag_dists, skill_names. Includes exporter stub for future
Prom-style metrics export. Windows-friendly file handling, no absolute paths.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouterTelemetryRecord:
    """Router Policy v2 telemetry record."""
    model: str
    tokens: int
    latency: float
    fps_delta: float
    outcome: str

    def __post_init__(self):
        """Validate router telemetry data."""
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")
        if self.latency < 0:
            raise ValueError("latency cannot be negative")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.outcome.strip():
            raise ValueError("outcome cannot be empty")


class RouterTelemetryLogger:
    """JSONL router telemetry logger."""

    def __init__(self, log_file: Optional[str] = None):
        """Initialize logger with optional file path."""
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_router_decision(self, record: RouterTelemetryRecord) -> None:
        """Log router telemetry record as JSONL line."""
        if not isinstance(record, RouterTelemetryRecord):
            raise TelemetryError("Invalid record type")

        data = {
            "model": record.model,
            "tokens": record.tokens,
            "latency": record.latency,
            "fps_delta": record.fps_delta,
            "outcome": record.outcome,
            "timestamp": time.time()
        }

        line = json.dumps(data, separators=(',', ':')) + '\n'

        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(line)
                logger.debug(f"Logged router telemetry to {self.log_file}")
            except OSError as e:
                logger.error(f"Failed to write router telemetry log: {e}")
                raise TelemetryError(f"Log write failed: {e}") from e
        else:
            # No file specified, just log to console for debugging
            logger.info(f"Router Telemetry: {line.strip()}")


class RouterTelemetryExporter:
    """Stub for future Prom-style router telemetry exporter."""

    def export_batch(self, records: List[RouterTelemetryRecord]) -> None:
        """Export batch of router telemetry records (stub implementation)."""
        logger.warning("Prom-style router telemetry exporter not implemented yet")
        # Future: integrate with Prometheus client, push to monitoring system


class TelemetryError(Exception):
    """Specific exception for telemetry operations."""
    pass