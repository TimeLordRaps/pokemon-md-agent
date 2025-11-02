"""Tests for autosave_system.py"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.utils.autosave_system import (
    AutosaveMode,
    AutosaveMetrics,
    AutosaveEntry,
    AutosaveSystem
)


class TestAutosaveMode:
    """Test AutosaveMode enum."""

    def test_mode_values(self):
        """Test autosave mode values."""
        assert AutosaveMode.SIMPLE.value == "simple"
        assert AutosaveMode.BEST_METRICS.value == "best-metrics"
        assert AutosaveMode.BALANCED.value == "balanced"


class TestAutosaveMetrics:
    """Test AutosaveMetrics dataclass."""

    def test_creation(self):
        """Test creating metrics."""
        metrics = AutosaveMetrics(
            frame=1000,
            timestamp=time.time(),
            score=0.85,
            progress=0.5
        )

        assert metrics.frame == 1000
        assert metrics.score == 0.85
        assert metrics.progress == 0.5


class TestAutosaveEntry:
    """Test AutosaveEntry dataclass."""

    def test_creation(self):
        """Test creating autosave entry."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())
        entry = AutosaveEntry(
            slot=0,
            file_path="/path/to/save.ss0",
            created_at="2025-11-01T00:00:00Z",
            frame=1000,
            mode="simple",
            metrics=metrics,
            reason="interval_300s"
        )

        assert entry.slot == 0
        assert entry.file_path == "/path/to/save.ss0"
        assert entry.mode == "simple"


class TestAutosaveSystem:
    """Test AutosaveSystem class."""

    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_controller(self):
        """Create mock MGBAController."""
        controller = MagicMock()
        controller.current_frame.return_value = 1000
        controller.save_state_to_file.return_value = True
        return controller

    @pytest.fixture
    def autosave_system(self, temp_run_dir, mock_controller):
        """Create AutosaveSystem instance."""
        return AutosaveSystem(
            run_dir=temp_run_dir,
            controller=mock_controller,
            mode=AutosaveMode.SIMPLE,
            interval=5.0  # Short interval for testing
        )

    def test_initialization(self, autosave_system, temp_run_dir):
        """Test autosave system initialization."""
        assert autosave_system.run_dir == temp_run_dir
        assert autosave_system.autosave_dir == temp_run_dir / "autosaves"
        assert autosave_system.autosave_dir.exists()
        assert autosave_system.next_slot == 0

    def test_simple_mode_time_trigger(self, autosave_system):
        """Test simple mode triggers on time interval."""
        # Should not trigger immediately
        should_save, reason = autosave_system.should_save()
        assert should_save is False

        # Advance time
        autosave_system.last_save_time = time.time() - 10.0

        # Should trigger now
        should_save, reason = autosave_system.should_save()
        assert should_save is True
        assert "interval" in reason

    def test_best_metrics_mode_score_trigger(self, autosave_system, mock_controller):
        """Test best_metrics mode triggers on score improvement."""
        autosave_system.mode = AutosaveMode.BEST_METRICS

        metrics1 = AutosaveMetrics(frame=1000, timestamp=time.time(), score=0.5)

        # First score should trigger (no baseline yet)
        should_save, reason = autosave_system.should_save(metrics1)
        assert should_save is True
        assert "best_score" in reason

        # Create a save to establish baseline
        entry = autosave_system.create_autosave(metrics1)
        assert entry is not None

        # Now try with same score - should not trigger
        metrics_same = AutosaveMetrics(frame=1250, timestamp=time.time(), score=0.5)
        should_save, reason = autosave_system.should_save(metrics_same)
        assert should_save is False

        # Now try with improved score
        metrics2 = AutosaveMetrics(frame=1500, timestamp=time.time(), score=0.6)
        should_save, reason = autosave_system.should_save(metrics2)
        assert should_save is True
        assert "best_score" in reason

    def test_balanced_mode_time_and_metrics(self, autosave_system):
        """Test balanced mode uses both time and metrics."""
        autosave_system.mode = AutosaveMode.BALANCED

        # Time-based trigger
        autosave_system.last_save_time = time.time() - 10.0
        should_save, reason = autosave_system.should_save()
        assert should_save is True

        # Reset time
        autosave_system.last_save_time = time.time()

        # Metrics-based trigger (significant improvement)
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time(), score=0.5)
        entry = autosave_system.create_autosave(metrics)

        # 20% improvement should trigger
        new_metrics = AutosaveMetrics(frame=1500, timestamp=time.time(), score=0.6)
        should_save, reason = autosave_system.should_save(new_metrics)
        assert should_save is True
        assert "improvement" in reason

    def test_create_autosave(self, autosave_system, mock_controller):
        """Test creating an autosave."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time(), score=0.8)

        entry = autosave_system.create_autosave(metrics, reason="test_save")

        assert entry is not None
        assert entry.slot == 0
        assert entry.frame == 1000
        assert entry.metrics.score == 0.8
        assert entry.reason == "test_save"

        # Verify file was saved
        mock_controller.save_state_file.assert_called_once()

    def test_autosave_slot_increment(self, autosave_system, mock_controller):
        """Test that slot numbers increment."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())

        entry1 = autosave_system.create_autosave(metrics)
        entry2 = autosave_system.create_autosave(metrics)
        entry3 = autosave_system.create_autosave(metrics)

        assert entry1.slot == 0
        assert entry2.slot == 1
        assert entry3.slot == 2

    def test_get_latest(self, autosave_system, mock_controller):
        """Test getting latest autosave."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())

        autosave_system.create_autosave(metrics)
        autosave_system.create_autosave(metrics)
        entry = autosave_system.create_autosave(metrics)

        latest = autosave_system.get_latest()
        assert latest is not None
        assert latest.slot == 2

    def test_get_best(self, autosave_system, mock_controller):
        """Test getting best autosave by score."""
        scores = [0.5, 0.8, 0.6]

        for score in scores:
            metrics = AutosaveMetrics(frame=1000, timestamp=time.time(), score=score)
            autosave_system.create_autosave(metrics)

        best = autosave_system.get_best()
        assert best is not None
        assert best.metrics.score == 0.8

    def test_get_by_slot(self, autosave_system, mock_controller):
        """Test getting autosave by slot number."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())

        autosave_system.create_autosave(metrics)
        autosave_system.create_autosave(metrics)
        autosave_system.create_autosave(metrics)

        entry = autosave_system.get_by_slot(1)
        assert entry is not None
        assert entry.slot == 1

    def test_list_autosaves(self, autosave_system, mock_controller):
        """Test listing all autosaves."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())

        autosave_system.create_autosave(metrics)
        autosave_system.create_autosave(metrics)
        autosave_system.create_autosave(metrics)

        entries = autosave_system.list_autosaves()
        assert len(entries) == 3

    def test_cleanup_old_autosaves(self, autosave_system, mock_controller):
        """Test cleanup of old autosaves."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())

        # Create 5 autosaves
        for _ in range(5):
            autosave_system.create_autosave(metrics)

        assert len(autosave_system.list_autosaves()) == 5

        # Clean up, keeping only 3
        deleted = autosave_system.cleanup_old(keep_count=3)

        assert deleted == 2
        assert len(autosave_system.list_autosaves()) == 3

    def test_manifest_persistence(self, temp_run_dir, mock_controller):
        """Test autosave manifest persists to disk."""
        system1 = AutosaveSystem(
            run_dir=temp_run_dir,
            controller=mock_controller,
            mode=AutosaveMode.SIMPLE
        )

        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())
        system1.create_autosave(metrics)

        # Create new system reading same directory
        system2 = AutosaveSystem(
            run_dir=temp_run_dir,
            controller=mock_controller,
            mode=AutosaveMode.SIMPLE
        )

        # Should load persisted manifest
        entries = system2.list_autosaves()
        assert len(entries) == 1
        assert entries[0].slot == 0

    def test_manifest_format(self, autosave_system, mock_controller):
        """Test manifest JSON format."""
        metrics = AutosaveMetrics(frame=1000, timestamp=time.time(), score=0.85)
        autosave_system.create_autosave(metrics)

        # Read manifest
        with open(autosave_system.manifest_file, 'r') as f:
            data = json.load(f)

        assert 'mode' in data
        assert 'interval' in data
        assert 'next_slot' in data
        assert 'best_score' in data
        assert 'entries' in data
        assert len(data['entries']) == 1

    @patch('src.utils.autosave_system.MGBAController')
    def test_autosave_handles_save_failure(self, mock_controller_class, temp_run_dir):
        """Test autosave handles save failures gracefully."""
        mock_controller = MagicMock()
        mock_controller.current_frame.return_value = 1000
        mock_controller.save_state_file.return_value = False  # Simulated failure

        system = AutosaveSystem(
            run_dir=temp_run_dir,
            controller=mock_controller,
            mode=AutosaveMode.SIMPLE
        )

        metrics = AutosaveMetrics(frame=1000, timestamp=time.time())
        entry = system.create_autosave(metrics)

        # Should return None on failure
        assert entry is None
