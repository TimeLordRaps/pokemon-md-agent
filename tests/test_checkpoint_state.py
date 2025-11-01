"""Tests for checkpoint state management.

Tests verify that CheckpointState can be created, validated, serialized,
and deserialized correctly.
"""

import json
import time
import pytest
from src.skills.checkpoint_state import CheckpointState


class TestCheckpointStateCreation:
    """Test creating CheckpointState instances."""

    def test_create_minimal_checkpoint(self):
        """Test creating a checkpoint with only required fields."""
        cp = CheckpointState(
            checkpoint_id="test_point",
            timestamp=time.time(),
            skill_name="test_skill",
        )

        assert cp.checkpoint_id == "test_point"
        assert cp.skill_name == "test_skill"
        assert cp.execution_context == {}
        assert cp.parameters == {}
        assert cp.notes == []
        assert cp.frames_captured == 0
        assert cp.description is None

    def test_create_checkpoint_with_all_fields(self):
        """Test creating a checkpoint with all fields populated."""
        timestamp = time.time()
        context = {"state": {"hp": 100}, "snapshots": []}
        params = {"difficulty": "hard"}
        notes = ["started exploration", "found item"]

        cp = CheckpointState(
            checkpoint_id="full_checkpoint",
            timestamp=timestamp,
            skill_name="explore_dungeon",
            execution_context=context,
            parameters=params,
            notes=notes,
            frames_captured=5,
            description="Ready to enter boss area",
        )

        assert cp.checkpoint_id == "full_checkpoint"
        assert cp.timestamp == timestamp
        assert cp.skill_name == "explore_dungeon"
        assert cp.execution_context == context
        assert cp.parameters == params
        assert cp.notes == notes
        assert cp.frames_captured == 5
        assert cp.description == "Ready to enter boss area"


class TestCheckpointStateSerialization:
    """Test serialization and deserialization of CheckpointState."""

    def test_to_dict_minimal(self):
        """Test converting minimal checkpoint to dictionary."""
        cp = CheckpointState(
            checkpoint_id="simple",
            timestamp=1234567890.0,
            skill_name="test",
        )

        data = cp.to_dict()

        assert data["checkpoint_id"] == "simple"
        assert data["timestamp"] == 1234567890.0
        assert data["skill_name"] == "test"
        assert data["execution_context"] == {}
        assert data["parameters"] == {}
        assert data["notes"] == []
        assert data["frames_captured"] == 0
        assert data["description"] is None

    def test_to_dict_with_complex_context(self):
        """Test serializing checkpoint with complex execution context."""
        context = {
            "state": {
                "hp": 50,
                "items": ["potion", "key"],
                "floor": 5,
            },
            "snapshots": [
                {"hp": 100, "level": 1},
                {"hp": 75, "level": 2},
            ],
            "params": {"mode": "exploration"},
        }

        cp = CheckpointState(
            checkpoint_id="complex",
            timestamp=1234567890.0,
            skill_name="dungeon_exploration",
            execution_context=context,
            description="After defeating first monster",
        )

        data = cp.to_dict()
        assert data["execution_context"] == context
        assert data["description"] == "After defeating first monster"

    def test_to_dict_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        cp = CheckpointState(
            checkpoint_id="json_test",
            timestamp=1234567890.0,
            skill_name="test",
            notes=["note1", "note2"],
            frames_captured=3,
        )

        data = cp.to_dict()
        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        assert loaded["checkpoint_id"] == "json_test"
        assert loaded["skill_name"] == "test"
        assert loaded["notes"] == ["note1", "note2"]

    def test_from_dict_minimal(self):
        """Test creating checkpoint from minimal dictionary."""
        data = {
            "checkpoint_id": "restored",
            "timestamp": 1234567890.0,
            "skill_name": "restored_skill",
        }

        cp = CheckpointState.from_dict(data)

        assert cp.checkpoint_id == "restored"
        assert cp.timestamp == 1234567890.0
        assert cp.skill_name == "restored_skill"
        assert cp.execution_context == {}
        assert cp.parameters == {}
        assert cp.notes == []

    def test_from_dict_with_all_fields(self):
        """Test creating checkpoint from full dictionary."""
        data = {
            "checkpoint_id": "full_restore",
            "timestamp": 1234567890.0,
            "skill_name": "full_skill",
            "execution_context": {"state": {"hp": 100}},
            "parameters": {"param1": "value1"},
            "notes": ["note1", "note2"],
            "frames_captured": 5,
            "description": "Full restore test",
        }

        cp = CheckpointState.from_dict(data)

        assert cp.checkpoint_id == "full_restore"
        assert cp.execution_context == {"state": {"hp": 100}}
        assert cp.parameters == {"param1": "value1"}
        assert cp.notes == ["note1", "note2"]
        assert cp.frames_captured == 5
        assert cp.description == "Full restore test"

    def test_from_dict_missing_required_field(self):
        """Test that from_dict raises ValueError when required fields are missing."""
        data = {
            "checkpoint_id": "incomplete",
            "timestamp": 1234567890.0,
            # missing skill_name
        }

        with pytest.raises(ValueError, match="Missing required field"):
            CheckpointState.from_dict(data)

    def test_from_dict_wrong_types(self):
        """Test that from_dict raises ValueError for wrong field types."""
        data = {
            "checkpoint_id": "type_error",
            "timestamp": "not_a_number",  # Should be float
            "skill_name": "test",
        }

        with pytest.raises(ValueError, match="timestamp must be numeric"):
            CheckpointState.from_dict(data)

    def test_roundtrip_serialization(self):
        """Test that checkpoint survives to_dict -> from_dict -> to_dict."""
        original = CheckpointState(
            checkpoint_id="roundtrip",
            timestamp=1234567890.0,
            skill_name="test_skill",
            execution_context={"state": {"hp": 50, "items": ["potion"]}},
            parameters={"difficulty": "hard"},
            notes=["started", "progressed"],
            frames_captured=10,
            description="Roundtrip test checkpoint",
        )

        dict1 = original.to_dict()
        restored = CheckpointState.from_dict(dict1)
        dict2 = restored.to_dict()

        assert dict1 == dict2
        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.timestamp == original.timestamp
        assert restored.skill_name == original.skill_name
        assert restored.execution_context == original.execution_context
        assert restored.parameters == original.parameters
        assert restored.notes == original.notes
        assert restored.frames_captured == original.frames_captured
        assert restored.description == original.description


class TestCheckpointStateValidation:
    """Test validation of CheckpointState."""

    def test_validate_valid_checkpoint(self):
        """Test that valid checkpoint passes validation."""
        cp = CheckpointState(
            checkpoint_id="valid_checkpoint",
            timestamp=1234567890.0,
            skill_name="valid_skill",
        )

        errors = cp.validate()
        assert errors == []
        assert cp.is_valid() is True

    def test_validate_invalid_checkpoint_id_empty(self):
        """Test that empty checkpoint_id fails validation."""
        cp = CheckpointState(
            checkpoint_id="",
            timestamp=1234567890.0,
            skill_name="test",
        )

        errors = cp.validate()
        assert any("checkpoint_id" in err for err in errors)
        assert cp.is_valid() is False

    def test_validate_invalid_checkpoint_id_too_long(self):
        """Test that checkpoint_id exceeding 64 chars fails validation."""
        cp = CheckpointState(
            checkpoint_id="a" * 65,
            timestamp=1234567890.0,
            skill_name="test",
        )

        errors = cp.validate()
        assert any("64 characters" in err for err in errors)

    def test_validate_invalid_checkpoint_id_special_chars(self):
        """Test that checkpoint_id with invalid characters fails validation."""
        cp = CheckpointState(
            checkpoint_id="test@#$%",
            timestamp=1234567890.0,
            skill_name="test",
        )

        errors = cp.validate()
        assert any("alphanumeric" in err for err in errors)

    def test_validate_valid_checkpoint_id_with_hyphens_underscores(self):
        """Test that checkpoint_id with hyphens and underscores passes validation."""
        cp = CheckpointState(
            checkpoint_id="valid-check_point-123",
            timestamp=1234567890.0,
            skill_name="test",
        )

        errors = cp.validate()
        assert all("checkpoint_id" not in err for err in errors)

    def test_validate_invalid_timestamp_negative(self):
        """Test that negative timestamp fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=-1.0,
            skill_name="test",
        )

        errors = cp.validate()
        assert any("timestamp" in err and "non-negative" in err for err in errors)

    def test_validate_invalid_timestamp_wrong_type(self):
        """Test that non-numeric timestamp fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp="not_numeric",  # type: ignore
            skill_name="test",
        )

        errors = cp.validate()
        assert any("timestamp" in err for err in errors)

    def test_validate_invalid_skill_name_empty(self):
        """Test that empty skill_name fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="",
        )

        errors = cp.validate()
        assert any("skill_name" in err for err in errors)

    def test_validate_invalid_skill_name_too_long(self):
        """Test that skill_name exceeding 128 chars fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="a" * 129,
        )

        errors = cp.validate()
        assert any("skill_name" in err and "128" in err for err in errors)

    def test_validate_invalid_execution_context_wrong_type(self):
        """Test that non-dict execution_context fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            execution_context="not_a_dict",  # type: ignore
        )

        errors = cp.validate()
        assert any("execution_context" in err and "dictionary" in err for err in errors)

    def test_validate_invalid_parameters_wrong_type(self):
        """Test that non-dict parameters fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            parameters="not_a_dict",  # type: ignore
        )

        errors = cp.validate()
        assert any("parameters" in err and "dictionary" in err for err in errors)

    def test_validate_invalid_notes_wrong_type(self):
        """Test that non-list notes fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            notes="not_a_list",  # type: ignore
        )

        errors = cp.validate()
        assert any("notes" in err and "list" in err for err in errors)

    def test_validate_invalid_notes_non_string_element(self):
        """Test that notes with non-string elements fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            notes=["valid", 123, "string"],  # type: ignore
        )

        errors = cp.validate()
        assert any("notes[1]" in err for err in errors)

    def test_validate_invalid_frames_captured_negative(self):
        """Test that negative frames_captured fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            frames_captured=-1,
        )

        errors = cp.validate()
        assert any("frames_captured" in err and "non-negative" in err for err in errors)

    def test_validate_invalid_frames_captured_wrong_type(self):
        """Test that non-int frames_captured fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            frames_captured="not_an_int",  # type: ignore
        )

        errors = cp.validate()
        assert any("frames_captured" in err for err in errors)

    def test_validate_invalid_description_wrong_type(self):
        """Test that non-string description fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            description=123,  # type: ignore
        )

        errors = cp.validate()
        assert any("description" in err for err in errors)

    def test_validate_invalid_description_too_long(self):
        """Test that description exceeding 500 chars fails validation."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            description="a" * 501,
        )

        errors = cp.validate()
        assert any("description" in err and "500" in err for err in errors)

    def test_validate_valid_description_none(self):
        """Test that None description is valid."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            description=None,
        )

        errors = cp.validate()
        assert all("description" not in err for err in errors)


class TestCheckpointStateRepr:
    """Test string representation of CheckpointState."""

    def test_repr_contains_important_info(self):
        """Test that __repr__ includes checkpoint ID, skill, and timestamp."""
        cp = CheckpointState(
            checkpoint_id="repr_test",
            timestamp=1234567890.0,
            skill_name="repr_skill",
            frames_captured=5,
        )

        repr_str = repr(cp)

        assert "repr_test" in repr_str
        assert "repr_skill" in repr_str
        assert "5" in repr_str  # frames_captured

    def test_repr_is_readable(self):
        """Test that __repr__ produces a readable format."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test_skill",
        )

        repr_str = repr(cp)
        assert repr_str.startswith("CheckpointState(")
        assert repr_str.endswith(")")


class TestCheckpointStateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_notes_list_valid(self):
        """Test that empty notes list is valid."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            notes=[],
        )

        assert cp.is_valid()

    def test_large_execution_context_valid(self):
        """Test that large execution context is valid as long as serializable."""
        large_context = {
            "state": {f"field_{i}": f"value_{i}" for i in range(100)},
            "snapshots": [{f"snap_{i}": i for i in range(10)}],
        }

        cp = CheckpointState(
            checkpoint_id="large_test",
            timestamp=1234567890.0,
            skill_name="test",
            execution_context=large_context,
        )

        assert cp.is_valid()
        data = cp.to_dict()
        restored = CheckpointState.from_dict(data)
        assert restored.execution_context == large_context

    def test_max_length_fields_valid(self):
        """Test that fields at maximum allowed length are valid."""
        cp = CheckpointState(
            checkpoint_id="a" * 64,
            timestamp=1234567890.0,
            skill_name="b" * 128,
            description="c" * 500,
        )

        assert cp.is_valid()

    def test_zero_timestamp_valid(self):
        """Test that timestamp of 0 is valid (Unix epoch)."""
        cp = CheckpointState(
            checkpoint_id="epoch",
            timestamp=0.0,
            skill_name="test",
        )

        assert cp.is_valid()

    def test_zero_frames_captured_valid(self):
        """Test that zero frames_captured is valid."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            frames_captured=0,
        )

        errors = cp.validate()
        assert all("frames_captured" not in err for err in errors)

    def test_large_frames_captured_valid(self):
        """Test that large frames_captured value is valid."""
        cp = CheckpointState(
            checkpoint_id="test",
            timestamp=1234567890.0,
            skill_name="test",
            frames_captured=1000000,
        )

        assert cp.is_valid()

    def test_unicode_in_fields(self):
        """Test that Unicode characters in fields are handled correctly."""
        cp = CheckpointState(
            checkpoint_id="test_ユーザー",
            timestamp=1234567890.0,
            skill_name="skill_🎮",
            description="Description with émojis 🏆",
            notes=["Note with ñoño characters"],
        )

        assert cp.is_valid()
        data = cp.to_dict()
        json_str = json.dumps(data, ensure_ascii=False)
        loaded = json.loads(json_str)
        restored = CheckpointState.from_dict(loaded)
        assert restored.checkpoint_id == "test_ユーザー"
        assert "🎮" in restored.skill_name
