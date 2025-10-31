"""Tests for telemetry JSONL logging per step.

Telemetry logs step-level metrics: model, vt_total, tokens, latency_ms, fps,
router_decision, rag_dists, skill_names. Verifies JSONL output, exporter stub
integration, and error handling for invalid data or file write failures.
"""

import json
import tempfile
import pytest
from unittest.mock import patch, mock_open

from src.telemetry.events import TelemetryEvent as TelemetryRecord, TelemetryEvents as TelemetryLogger


class TestTelemetryLogger:
    """Test telemetry logging functionality."""

    def test_log_step_metrics(self):
        """Log step metrics and verify JSONL format."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            logger = TelemetryLogger(log_file=f.name)

            record = TelemetryRecord(
                model="qwen-vl",
                vt_total=1500,
                tokens=250,
                latency_ms=120.5,
                fps=8.3,
                router_decision="vision",
                rag_dists=[0.1, 0.2, 0.3],
                skill_names=["explore", "attack"]
            )

            logger.log_event(record)

            f.seek(0)
            lines = f.readlines()
            assert len(lines) == 1

            parsed = json.loads(lines[0].strip())
            assert parsed["model"] == "qwen-vl"
            assert parsed["vt_total"] == 1500
            assert parsed["tokens"] == 250
            assert parsed["latency_ms"] == 120.5
            assert parsed["fps"] == 8.3
            assert parsed["router_decision"] == "vision"
            assert parsed["rag_dists"] == [0.1, 0.2, 0.3]
            assert parsed["skill_names"] == ["explore", "attack"]

    def test_invalid_data_raises_exception(self):
        """Raise specific exception on invalid telemetry data."""
        logger = TelemetryLogger()

        # Invalid: negative latency
        with pytest.raises(ValueError):
            TelemetryRecord(
                model="qwen-vl",
                vt_total=1500,
                tokens=250,
                latency_ms=-10.0,  # Invalid
                fps=8.3,
                router_decision="vision",
                rag_dists=[0.1, 0.2, 0.3],
                skill_names=["explore", "attack"]
            )

    def test_export_events_stub(self):
        """Verify export_events stub accepts events without error."""
        logger = TelemetryLogger()
        
        record = TelemetryRecord(
            model="qwen-vl",
            vt_total=1500,
            tokens=250,
            latency_ms=120.5,
            fps=8.3,
            router_decision="vision",
            rag_dists=[0.1, 0.2, 0.3],
            skill_names=["explore", "attack"]
        )

        # Stub should not raise, but log unimplemented
        with patch('src.telemetry.events.logger') as mock_logger:
            logger.export_events([record])
            mock_logger.warning.assert_called_with("Telemetry events export not implemented yet")