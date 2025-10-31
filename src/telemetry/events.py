"""Telemetry events stub for JSONL per step logging."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Step-level telemetry event for JSONL logging."""
    model: str
    vt_total: int
    tokens: int
    latency_ms: float
    fps: float
    router_decision: str
    rag_dists: List[float]
    skill_names: List[str]

    def __post_init__(self):
        """Basic validation for telemetry data."""
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.fps < 0:
            raise ValueError("fps must be non-negative")


class TelemetryEvents:
    """Stub implementation for telemetry events logging."""

    def __init__(self, log_file: Optional[str] = None):
        """Initialize with optional log file path."""
        self.log_file = Path(log_file) if log_file else None

    def log_event(self, event: TelemetryEvent) -> None:
        """Log a telemetry event as JSONL line."""
        if not isinstance(event, TelemetryEvent):
            raise ValueError("Invalid event type")

        data = {
            "model": event.model,
            "vt_total": event.vt_total,
            "tokens": event.tokens,
            "latency_ms": event.latency_ms,
            "fps": event.fps,
            "router_decision": event.router_decision,
            "rag_dists": event.rag_dists,
            "skill_names": event.skill_names,
            "timestamp": None  # Could add actual timestamp
        }

        line = json.dumps(data, separators=(',', ':')) + '\n'

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(line)
                logger.debug(f"Logged telemetry event to {self.log_file}")
            except OSError as e:
                logger.error(f"Failed to write telemetry log: {e}")
                raise
        else:
            # No file specified, just log to console
            logger.info(f"Telemetry Event: {line.strip()}")

    def export_events(self, events: List[TelemetryEvent]) -> None:
        """Export batch of events (stub implementation)."""
        logger.warning("Telemetry events export not implemented yet")