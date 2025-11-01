"""Tests for SaveManager integration with checkpoints.

Tests verify that SaveStateCheckpointPrimitive and LoadStateCheckpointPrimitive
properly integrate with the SaveManager for persistent game state management.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.skills.python_runtime import (
    PythonSkillRuntime,
    AbortSignal,
)
from src.skills.spec import (
    SaveStateCheckpointPrimitive,
    LoadStateCheckpointPrimitive,
    CheckpointPrimitive,
)
from src.skills.checkpoint_state import CheckpointState
from src.environment.save_manager import SaveManager


class TestSaveStateCheckpointWithSaveManager:
    """Test SaveStateCheckpointPrimitive with SaveManager."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        controller.save_state_file = Mock(return_value=True)
        controller.current_frame = Mock(return_value=0)
        controller.is_connected = Mock(return_value=True)
        return controller

    @pytest.fixture
    def save_dir(self):
        """Create a temporary save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def save_manager(self, mock_controller, save_dir):
        """Create a SaveManager instance."""
        return SaveManager(mock_controller, save_dir)

    @pytest.fixture
    def runtime(self, mock_controller, save_manager):
        """Create a PythonSkillRuntime with SaveManager."""
        return PythonSkillRuntime(mock_controller, save_manager=save_manager)

    def test_save_checkpoint_without_save_manager_raises_error(self, mock_controller):
        """Test that save checkpoint raises error when SaveManager not configured."""
        runtime = PythonSkillRuntime(mock_controller)  # No SaveManager
        primitive = SaveStateCheckpointPrimitive(slot=3, label="test_save")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        with pytest.raises(AbortSignal, match="SaveManager not configured"):
            runtime._handle_save_checkpoint(primitive, ctx)

    def test_save_checkpoint_success(self, runtime, save_manager):
        """Test successful checkpoint save."""
        primitive = SaveStateCheckpointPrimitive(slot=5, label="success_save")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_save_checkpoint(primitive, ctx)

        # Verify the save was logged
        assert any("Saved checkpoint slot 5" in note for note in ctx["notes"])

    def test_save_checkpoint_failed_save_raises_error(self, mock_controller, save_manager):
        """Test that failed save raises AbortSignal."""
        # Make save_state_file return False
        mock_controller.save_state_file = Mock(return_value=False)

        runtime = PythonSkillRuntime(mock_controller, save_manager=save_manager)
        primitive = SaveStateCheckpointPrimitive(slot=2, label="fail_save")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        with pytest.raises(AbortSignal, match="Failed to save checkpoint slot"):
            runtime._handle_save_checkpoint(primitive, ctx)

    def test_save_checkpoint_with_label_recorded(self, runtime):
        """Test that checkpoint label is recorded in save description."""
        label = "critical_save_point"
        primitive = SaveStateCheckpointPrimitive(slot=7, label=label)
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_save_checkpoint(primitive, ctx)

        assert any(label in note for note in ctx["notes"])


class TestLoadStateCheckpointWithSaveManager:
    """Test LoadStateCheckpointPrimitive with SaveManager."""

    @pytest.fixture
    def mock_controller(self, save_dir):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        controller.current_frame = Mock(return_value=0)
        controller.is_connected = Mock(return_value=True)

        # Track saved slots for load operations
        saved_slots = set()

        def save_side_effect(path, slot):
            # Create the actual file
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("mock save data")
            saved_slots.add(slot)
            return True

        def load_side_effect(path, slot):
            # Check if file exists (was previously saved)
            return Path(path).exists()

        controller.save_state_file = Mock(side_effect=save_side_effect)
        controller.load_state_file = Mock(side_effect=load_side_effect)
        return controller

    @pytest.fixture
    def save_dir(self):
        """Create a temporary save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def save_manager(self, mock_controller, save_dir):
        """Create a SaveManager instance."""
        return SaveManager(mock_controller, save_dir)

    @pytest.fixture
    def runtime(self, mock_controller, save_manager):
        """Create a PythonSkillRuntime with SaveManager."""
        return PythonSkillRuntime(mock_controller, save_manager=save_manager)

    def test_load_checkpoint_without_save_manager_raises_error(self, mock_controller):
        """Test that load checkpoint raises error when SaveManager not configured."""
        runtime = PythonSkillRuntime(mock_controller)  # No SaveManager
        primitive = LoadStateCheckpointPrimitive(slot=3)
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        with pytest.raises(AbortSignal, match="SaveManager not configured"):
            runtime._handle_load_checkpoint(primitive, ctx)

    def test_load_checkpoint_success(self, runtime):
        """Test successful checkpoint load."""
        # First, save a checkpoint slot
        save_primitive = SaveStateCheckpointPrimitive(slot=5, label="test_save")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }
        runtime._handle_save_checkpoint(save_primitive, ctx)

        # Now load it
        load_primitive = LoadStateCheckpointPrimitive(slot=5)
        ctx_load = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }
        runtime._handle_load_checkpoint(load_primitive, ctx_load)

        # Verify the load was logged
        assert any("Loaded checkpoint slot 5" in note for note in ctx_load["notes"])

    def test_load_checkpoint_missing_slot_raises_error(self, mock_controller, save_manager):
        """Test that loading missing slot raises AbortSignal."""
        # Make load_state_file return False (slot doesn't exist)
        mock_controller.load_state_file = Mock(return_value=False)

        runtime = PythonSkillRuntime(mock_controller, save_manager=save_manager)
        primitive = LoadStateCheckpointPrimitive(slot=15)  # Valid slot number but not saved
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        with pytest.raises(AbortSignal, match="Failed to load checkpoint slot"):
            runtime._handle_load_checkpoint(primitive, ctx)


class TestCheckpointDiskPersistence:
    """Test saving and loading checkpoints from disk."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 50, "items": ["potion"]})
        return controller

    @pytest.fixture
    def save_dir(self):
        """Create a temporary save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime."""
        return PythonSkillRuntime(mock_controller)

    def test_save_checkpoint_to_disk(self, runtime, save_dir):
        """Test saving checkpoint to disk."""
        # Create a checkpoint
        ctx = {
            "params": {"difficulty": "hard"},
            "notes": ["started exploration"],
            "frames": [],
            "snapshots": [{"location": "start"}],
            "status": "indeterminate",
        }
        runtime._handle_checkpoint(CheckpointPrimitive(label="test_cp"), ctx)

        # Save to disk
        checkpoint_path = save_dir / "test_checkpoint.json"
        success = runtime.save_checkpoint_to_disk("test_cp", checkpoint_path)

        assert success
        assert checkpoint_path.exists()

        # Verify content
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert data["checkpoint_id"] == "test_cp"
        assert data["parameters"]["difficulty"] == "hard"

    def test_load_checkpoint_from_disk(self, runtime, save_dir):
        """Test loading checkpoint from disk."""
        # Create a checkpoint file manually
        checkpoint_file = save_dir / "loaded_cp.json"
        checkpoint_data = {
            "checkpoint_id": "loaded_cp",
            "timestamp": 1234567890.0,
            "skill_name": "test_skill",
            "execution_context": {"state": {"hp": 100}},
            "parameters": {},
            "notes": ["manual create"],
            "frames_captured": 3,
            "description": "Manually created checkpoint",
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        checkpoint = runtime.load_checkpoint_from_disk(checkpoint_file)

        assert checkpoint.checkpoint_id == "loaded_cp"
        assert checkpoint.skill_name == "test_skill"
        assert checkpoint.frames_captured == 3
        assert len(checkpoint.notes) == 1

    def test_save_load_roundtrip(self, runtime, save_dir):
        """Test save and load roundtrip."""
        # Create checkpoint
        ctx = {
            "params": {"mode": "exploration"},
            "notes": ["note1", "note2"],
            "frames": ["f1", "f2", "f3"],
            "snapshots": [{"hp": 50}, {"hp": 75}],
            "status": "indeterminate",
        }
        runtime._handle_checkpoint(
            CheckpointPrimitive(label="roundtrip", description="Test roundtrip"),
            ctx
        )

        # Save to disk
        path = save_dir / "roundtrip.json"
        runtime.save_checkpoint_to_disk("roundtrip", path)

        # Load from disk
        loaded = runtime.load_checkpoint_from_disk(path)

        # Verify all fields match
        original = runtime.get_checkpoint("roundtrip")
        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.skill_name == original.skill_name
        assert loaded.frames_captured == original.frames_captured
        assert loaded.execution_context == original.execution_context
        assert loaded.notes == original.notes

    def test_save_checkpoint_invalid_id_raises_error(self, runtime, save_dir):
        """Test that saving non-existent checkpoint raises ValueError."""
        path = save_dir / "nonexistent.json"

        with pytest.raises(ValueError, match="Checkpoint not found"):
            runtime.save_checkpoint_to_disk("does_not_exist", path)

    def test_load_checkpoint_missing_file_raises_error(self, runtime, save_dir):
        """Test that loading missing file raises FileNotFoundError."""
        path = save_dir / "missing.json"

        with pytest.raises(FileNotFoundError):
            runtime.load_checkpoint_from_disk(path)

    def test_load_checkpoint_invalid_json_raises_error(self, runtime, save_dir):
        """Test that loading invalid JSON raises ValueError."""
        path = save_dir / "invalid.json"

        with open(path, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ValueError, match="Failed to load checkpoint"):
            runtime.load_checkpoint_from_disk(path)

    def test_load_checkpoint_invalid_data_raises_error(self, runtime, save_dir):
        """Test that loading checkpoint with invalid data raises ValueError."""
        path = save_dir / "invalid_data.json"

        invalid_data = {
            "checkpoint_id": "",  # Empty ID is invalid
            "timestamp": -1.0,    # Negative timestamp is invalid
            "skill_name": "test",
        }

        with open(path, "w") as f:
            json.dump(invalid_data, f)

        with pytest.raises(ValueError, match="invalid"):
            runtime.load_checkpoint_from_disk(path)
