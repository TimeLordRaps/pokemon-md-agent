"""Tests for checkpoint primitive handlers in PythonSkillRuntime.

Tests verify that CheckpointPrimitive, ResumePrimitive, SaveStateCheckpointPrimitive,
and LoadStateCheckpointPrimitive are properly handled during skill execution.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from src.skills.python_runtime import (
    PythonSkillRuntime,
    PrimitiveExecutor,
    AbortSignal,
)
from src.skills.spec import (
    SkillSpec,
    CheckpointPrimitive,
    ResumePrimitive,
    SaveStateCheckpointPrimitive,
    LoadStateCheckpointPrimitive,
    AnnotatePrimitive,
)
from src.skills.checkpoint_state import CheckpointState


class TestCheckpointPrimitiveHandler:
    """Test CheckpointPrimitive execution."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100, "level": 1})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime with mock controller."""
        return PythonSkillRuntime(mock_controller)

    def test_create_checkpoint_minimal(self, runtime):
        """Test creating a checkpoint with minimal data."""
        primitive = CheckpointPrimitive(label="test_checkpoint")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(primitive, ctx)

        # Verify checkpoint was created
        assert "test_checkpoint" in runtime._checkpoints
        checkpoint = runtime._checkpoints["test_checkpoint"]
        assert checkpoint.checkpoint_id == "test_checkpoint"
        assert checkpoint.description is None
        assert len(ctx["notes"]) == 1
        assert "Checkpoint created: test_checkpoint" in ctx["notes"][0]

    def test_create_checkpoint_with_description(self, runtime):
        """Test creating a checkpoint with description."""
        primitive = CheckpointPrimitive(
            label="boss_fight",
            description="Ready to fight boss"
        )
        ctx = {
            "params": {"difficulty": "hard"},
            "notes": ["started boss area"],
            "frames": ["frame1.png", "frame2.png"],
            "snapshots": [{"hp": 100}],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(primitive, ctx)

        checkpoint = runtime._checkpoints["boss_fight"]
        assert checkpoint.description == "Ready to fight boss"
        assert checkpoint.frames_captured == 2
        assert len(checkpoint.notes) == 1
        assert checkpoint.parameters == {"difficulty": "hard"}

    def test_create_checkpoint_captures_state(self, runtime, mock_controller):
        """Test that checkpoint captures current game state."""
        state = {"hp": 50, "items": ["potion", "key"]}
        mock_controller.semantic_state = Mock(return_value=state)

        primitive = CheckpointPrimitive(label="state_capture")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
            "_current_skill_name": "test_skill",
        }

        runtime._handle_checkpoint(primitive, ctx)

        checkpoint = runtime._checkpoints["state_capture"]
        assert checkpoint.execution_context["state"] == state
        assert checkpoint.skill_name == "test_skill"

    def test_create_checkpoint_validates_state(self, runtime):
        """Test that checkpoint validates its state properly."""
        # Valid checkpoint should work fine
        primitive = CheckpointPrimitive(label="valid_checkpoint")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        # Should not raise
        runtime._handle_checkpoint(primitive, ctx)
        assert "valid_checkpoint" in runtime._checkpoints

    def test_list_checkpoints_empty(self, runtime):
        """Test listing checkpoints when none exist."""
        assert runtime.list_checkpoints() == []

    def test_list_checkpoints_multiple(self, runtime):
        """Test listing multiple checkpoints."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(CheckpointPrimitive(label="cp1"), ctx)
        runtime._handle_checkpoint(CheckpointPrimitive(label="cp2"), ctx)
        runtime._handle_checkpoint(CheckpointPrimitive(label="cp3"), ctx)

        checkpoints = runtime.list_checkpoints()
        assert len(checkpoints) == 3
        assert "cp1" in checkpoints
        assert "cp2" in checkpoints
        assert "cp3" in checkpoints

    def test_get_checkpoint_exists(self, runtime):
        """Test retrieving an existing checkpoint."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(CheckpointPrimitive(label="my_cp"), ctx)
        checkpoint = runtime.get_checkpoint("my_cp")

        assert checkpoint is not None
        assert checkpoint.checkpoint_id == "my_cp"

    def test_get_checkpoint_not_exists(self, runtime):
        """Test retrieving a non-existent checkpoint."""
        checkpoint = runtime.get_checkpoint("nonexistent")
        assert checkpoint is None

    def test_clear_checkpoints(self, runtime):
        """Test clearing all checkpoints."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(CheckpointPrimitive(label="cp1"), ctx)
        runtime._handle_checkpoint(CheckpointPrimitive(label="cp2"), ctx)

        assert len(runtime.list_checkpoints()) == 2
        runtime.clear_checkpoints()
        assert len(runtime.list_checkpoints()) == 0


class TestResumePrimitiveHandler:
    """Test ResumePrimitive execution."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime with mock controller."""
        return PythonSkillRuntime(mock_controller)

    def test_resume_existing_checkpoint(self, runtime):
        """Test resuming from an existing checkpoint."""
        # Create a checkpoint first
        ctx = {
            "params": {},
            "notes": ["note1"],
            "frames": [],
            "snapshots": [{"snap1": "data"}],
            "status": "indeterminate",
        }
        runtime._handle_checkpoint(CheckpointPrimitive(label="cp1"), ctx)

        # Clear context and resume
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }
        primitive = ResumePrimitive(label="cp1")
        runtime._handle_resume(primitive, ctx)

        # Verify context was restored
        assert len(ctx["snapshots"]) == 1
        assert ctx["snapshots"][0] == {"snap1": "data"}
        assert any("Resumed from checkpoint" in note for note in ctx["notes"])

    def test_resume_missing_checkpoint_no_fallback_raises_error(self, runtime):
        """Test resuming from non-existent checkpoint without fallback."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }
        primitive = ResumePrimitive(label="missing")

        with pytest.raises(AbortSignal, match="Checkpoint not found"):
            runtime._handle_resume(primitive, ctx)

    def test_resume_missing_checkpoint_with_fallback(self, runtime):
        """Test resuming from non-existent checkpoint with fallback steps."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        # Create fallback steps (simple annotation)
        fallback_steps = [AnnotatePrimitive(message="Executed fallback")]

        primitive = ResumePrimitive(label="missing", fallback_steps=fallback_steps)
        runtime._handle_resume(primitive, ctx)

        # Verify fallback was executed
        assert any("fallback" in note.lower() for note in ctx["notes"])
        assert any("Executed fallback" in note for note in ctx["notes"])

    def test_resume_restores_snapshots(self, runtime):
        """Test that resume properly restores snapshots."""
        # Create checkpoint with multiple snapshots
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [
                {"hp": 100, "level": 1},
                {"hp": 75, "level": 2},
            ],
            "status": "indeterminate",
        }
        runtime._handle_checkpoint(CheckpointPrimitive(label="multi_snap"), ctx)

        # Clear and resume
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }
        runtime._handle_resume(ResumePrimitive(label="multi_snap"), ctx)

        assert len(ctx["snapshots"]) == 2
        assert ctx["snapshots"][0] == {"hp": 100, "level": 1}
        assert ctx["snapshots"][1] == {"hp": 75, "level": 2}


class TestSaveStateCheckpointHandler:
    """Test SaveStateCheckpointPrimitive execution."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime with mock controller and SaveManager."""
        mock_save_manager = Mock()
        mock_save_manager.save_slot.return_value = True
        return PythonSkillRuntime(mock_controller, save_manager=mock_save_manager)

    def test_save_checkpoint_logs_operation(self, runtime):
        """Test that save checkpoint logs the operation."""
        primitive = SaveStateCheckpointPrimitive(slot=3, label="save_point")
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_save_checkpoint(primitive, ctx)

        # Verify operation was logged in notes
        assert any("Saved checkpoint slot 3" in note for note in ctx["notes"])
        assert any("save_point" in note for note in ctx["notes"])

    def test_save_checkpoint_slot_validation(self, runtime):
        """Test that save checkpoint validates slot numbers."""
        # Slot numbers are validated by Pydantic in the primitive itself
        # This test ensures the handler works with valid slots
        for slot in [0, 5, 10, 15]:
            primitive = SaveStateCheckpointPrimitive(slot=slot, label=f"slot_{slot}")
            ctx = {
                "params": {},
                "notes": [],
                "frames": [],
                "snapshots": [],
                "status": "indeterminate",
            }
            runtime._handle_save_checkpoint(primitive, ctx)


class TestLoadStateCheckpointHandler:
    """Test LoadStateCheckpointPrimitive execution."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime with mock controller and SaveManager."""
        mock_save_manager = Mock()
        mock_save_manager.load_slot.return_value = True
        return PythonSkillRuntime(mock_controller, save_manager=mock_save_manager)

    def test_load_checkpoint_logs_operation(self, runtime):
        """Test that load checkpoint logs the operation."""
        primitive = LoadStateCheckpointPrimitive(slot=2)
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_load_checkpoint(primitive, ctx)

        # Verify operation was logged in notes
        assert any("Loaded checkpoint slot 2" in note for note in ctx["notes"])

    def test_load_checkpoint_multiple_slots(self, runtime):
        """Test loading from different slots."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_load_checkpoint(LoadStateCheckpointPrimitive(slot=0), ctx)
        runtime._handle_load_checkpoint(LoadStateCheckpointPrimitive(slot=5), ctx)
        runtime._handle_load_checkpoint(LoadStateCheckpointPrimitive(slot=15), ctx)

        # All three should be logged
        slot_notes = [note for note in ctx["notes"] if "Loaded checkpoint slot" in note]
        assert len(slot_notes) == 3


class TestCheckpointIntegration:
    """Integration tests for checkpoint workflow."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100, "level": 1})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create a PythonSkillRuntime with mock controller."""
        return PythonSkillRuntime(mock_controller)

    def test_checkpoint_resume_workflow(self, runtime):
        """Test complete checkpoint and resume workflow."""
        # Phase 1: Create checkpoint
        ctx = {
            "params": {"mode": "exploration"},
            "notes": ["started exploration"],
            "frames": ["frame1.png", "frame2.png"],
            "snapshots": [{"location": "dungeon_1f"}],
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(
            CheckpointPrimitive(label="floor_1_ready", description="Ready for floor 1"),
            ctx
        )

        # Phase 2: Simulate more execution
        ctx["notes"].append("progressed deeper")
        ctx["snapshots"].append({"location": "dungeon_2f"})

        # Phase 3: Resume from checkpoint
        ctx_new = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        runtime._handle_resume(ResumePrimitive(label="floor_1_ready"), ctx_new)

        # Verify state restoration
        assert len(ctx_new["snapshots"]) == 1
        assert ctx_new["snapshots"][0] == {"location": "dungeon_1f"}

    def test_multiple_checkpoints_independent(self, runtime):
        """Test that multiple checkpoints are stored independently."""
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        # Create multiple checkpoints at different states
        ctx["snapshots"].append({"state": "early"})
        runtime._handle_checkpoint(CheckpointPrimitive(label="early"), ctx)

        ctx["snapshots"] = [{"state": "mid"}]
        runtime._handle_checkpoint(CheckpointPrimitive(label="mid"), ctx)

        ctx["snapshots"] = [{"state": "late"}]
        runtime._handle_checkpoint(CheckpointPrimitive(label="late"), ctx)

        # Verify each checkpoint has its own state
        assert runtime.get_checkpoint("early").frames_captured == 0
        assert runtime.get_checkpoint("mid").frames_captured == 0
        assert runtime.get_checkpoint("late").frames_captured == 0

        # Check snapshots are independent
        early_snap = runtime.get_checkpoint("early").execution_context["snapshots"]
        mid_snap = runtime.get_checkpoint("mid").execution_context["snapshots"]
        late_snap = runtime.get_checkpoint("late").execution_context["snapshots"]

        assert early_snap[0]["state"] == "early"
        assert mid_snap[0]["state"] == "mid"
        assert late_snap[0]["state"] == "late"

    def test_checkpoint_state_immutability(self, runtime):
        """Test that checkpoint state is properly isolated from execution context."""
        original_snapshots = [{"hp": 100}]
        ctx = {
            "params": {},
            "notes": [],
            "frames": [],
            "snapshots": original_snapshots,
            "status": "indeterminate",
        }

        runtime._handle_checkpoint(CheckpointPrimitive(label="immutable"), ctx)

        # Modify original context
        ctx["snapshots"].append({"hp": 50})

        # Verify checkpoint snapshot is unchanged
        checkpoint = runtime.get_checkpoint("immutable")
        assert len(checkpoint.execution_context["snapshots"]) == 1
        assert checkpoint.execution_context["snapshots"][0] == {"hp": 100}
