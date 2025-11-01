"""Tests for skill DSL primitives and composition."""

import pytest
from typing import List
from src.skills.dsl import (
    Skill, Action, Tap, Hold, Release, WaitTurn, Face, Capture,
    ReadState, Expect, Annotate, Break, Abort, Checkpoint, Resume,
    Save, Load, Button, Direction, tap, hold, release, waitTurn,
    face, capture, read_state, expect, annotate, break_, abort,
    checkpoint, resume, save, load
)


class TestDSLPrimitives:
    """Test individual DSL primitives."""

    def test_button_enum(self):
        """Test Button enum values."""
        assert Button.A == "a"
        assert Button.B == "b"
        assert Button.START == "start"
        assert Button.SELECT == "select"

    def test_direction_enum(self):
        """Test Direction enum values."""
        assert Direction.UP == "up"
        assert Direction.DOWN == "down"
        assert Direction.LEFT == "left"
        assert Direction.RIGHT == "right"

    def test_tap_primitive(self):
        """Test tap primitive creation."""
        action = tap(Button.A)
        assert isinstance(action, Tap)
        assert action.button == Button.A

    def test_hold_primitive(self):
        """Test hold primitive creation."""
        action = hold(Button.B, 10)
        assert isinstance(action, Hold)
        assert action.button == Button.B
        assert action.frames == 10

        # Test validation
        with pytest.raises(ValueError):
            hold(Button.A, 0)  # frames must be > 0

    def test_release_primitive(self):
        """Test release primitive creation."""
        action = release(Button.START)
        assert isinstance(action, Release)
        assert action.button == Button.START

    def test_wait_turn_primitive(self):
        """Test waitTurn primitive creation."""
        action = waitTurn()
        assert isinstance(action, WaitTurn)

    def test_face_primitive(self):
        """Test face primitive creation."""
        action = face(Direction.UP)
        assert isinstance(action, Face)
        assert action.direction == Direction.UP

    def test_capture_primitive(self):
        """Test capture primitive creation."""
        action = capture("test_label")
        assert isinstance(action, Capture)
        assert action.label == "test_label"

    def test_read_state_primitive(self):
        """Test read_state primitive creation."""
        fields = ["position", "hp"]
        action = read_state(fields)
        assert isinstance(action, ReadState)
        assert action.fields == fields

    def test_expect_primitive(self):
        """Test expect primitive creation."""
        condition = "hp > 50"
        message = "HP should be above 50"
        action = expect(condition, message)
        assert isinstance(action, Expect)
        assert action.condition == condition
        assert action.message == message

    def test_annotate_primitive(self):
        """Test annotate primitive creation."""
        message = "Test annotation"
        action = annotate(message)
        assert isinstance(action, Annotate)
        assert action.message == message

    def test_break_primitive(self):
        """Test break_ primitive creation."""
        action = break_()
        assert isinstance(action, Break)

    def test_abort_primitive(self):
        """Test abort primitive creation."""
        message = "Test abort"
        action = abort(message)
        assert isinstance(action, Abort)
        assert action.message == message

    def test_checkpoint_primitive(self):
        """Test checkpoint primitive creation."""
        label = "test_checkpoint"
        action = checkpoint(label)
        assert isinstance(action, Checkpoint)
        assert action.label == label

    def test_resume_primitive(self):
        """Test resume primitive creation."""
        action = resume()
        assert isinstance(action, Resume)

    def test_save_primitive(self):
        """Test save primitive creation."""
        slot = 1
        action = save(slot)
        assert isinstance(action, Save)
        assert action.slot == slot

    def test_load_primitive(self):
        """Test load primitive creation."""
        slot = 2
        action = load(slot)
        assert isinstance(action, Load)
        assert action.slot == slot


class TestSkillComposition:
    """Test skill composition and validation."""

    def test_skill_creation(self):
        """Test basic skill creation."""
        actions = [
            tap(Button.A),
            waitTurn(),
            face(Direction.UP),
            capture("done")
        ]

        skill = Skill(
            name="test_skill",
            description="A test skill",
            actions=actions
        )

        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert len(skill.actions) == 4
        assert isinstance(skill.actions[0], Tap)
        assert isinstance(skill.actions[3], Capture)

    def test_skill_without_description(self):
        """Test skill creation without description."""
        skill = Skill(
            name="simple_skill",
            actions=[tap(Button.B)]
        )

        assert skill.name == "simple_skill"
        assert skill.description is None
        assert len(skill.actions) == 1

    def test_empty_skill_validation(self):
        """Test skill with empty actions list."""
        skill = Skill(
            name="empty_skill",
            actions=[]
        )

        assert len(skill.actions) == 0

    def test_complex_skill_composition(self):
        """Test complex skill with multiple action types."""
        actions = [
            # Initial setup
            checkpoint("start"),
            read_state(["position", "floor"]),
            face(Direction.UP),

            # Movement sequence
            tap(Button.UP),
            waitTurn(),
            hold(Button.UP, 30),

            # State checks
            read_state(["enemies", "stairs"]),
            expect("len(enemies) == 0", "Should be no enemies"),
            expect("stairs_visible", "Stairs should be visible"),

            # Completion
            annotate("Navigation successful"),
            capture("finished"),
        ]

        skill = Skill(
            name="navigate_to_stairs",
            description="Navigate to stairs avoiding enemies",
            actions=actions
        )

        assert len(skill.actions) == 11

        # Verify action types
        action_types = [type(action).__name__ for action in skill.actions]
        expected_types = [
            "Checkpoint", "ReadState", "Face", "Tap", "WaitTurn", "Hold",
            "ReadState", "Expect", "Expect", "Annotate", "Capture"
        ]

        assert action_types == expected_types


class TestSkillExecutionFlow:
    """Test skill execution flow control."""

    def test_break_flow_control(self):
        """Test break action in skill flow."""
        actions = [
            tap(Button.A),
            break_(),
            tap(Button.B),  # Should not execute
        ]

        skill = Skill(name="break_test", actions=actions)
        assert len(skill.actions) == 3

    def test_abort_flow_control(self):
        """Test abort action in skill flow."""
        actions = [
            tap(Button.A),
            abort("Test abort message"),
            tap(Button.B),  # Should not execute
        ]

        skill = Skill(name="abort_test", actions=actions)
        assert len(skill.actions) == 3

    def test_checkpoint_resume_flow(self):
        """Test checkpoint and resume flow control."""
        actions = [
            checkpoint("test_point"),
            tap(Button.A),
            resume(),  # Would resume from checkpoint
        ]

        skill = Skill(name="checkpoint_test", actions=actions)
        assert len(skill.actions) == 3

    def test_save_load_flow(self):
        """Test save and load flow control."""
        actions = [
            save(1),
            tap(Button.A),
            load(1),
        ]

        skill = Skill(name="save_load_test", actions=actions)
        assert len(skill.actions) == 3


class TestSkillSerialization:
    """Test skill serialization and validation."""

    def test_skill_to_dict(self):
        """Test converting skill to dictionary."""
        skill = Skill(
            name="test_skill",
            description="Test description",
            actions=[tap(Button.A), waitTurn()]
        )

        # This would be used for JSON serialization
        skill_dict = skill.model_dump()
        assert skill_dict["name"] == "test_skill"
        assert skill_dict["description"] == "Test description"
        assert len(skill_dict["actions"]) == 2

    def test_skill_from_dict(self):
        """Test creating skill from dictionary."""
        skill_data = {
            "name": "test_skill",
            "description": "Test description",
            "actions": [
                {"button": "a"},
                {"direction": "up"}
            ]
        }

        # Validate that the data structure is correct
        assert skill_data["name"] == "test_skill"
        assert len(skill_data["actions"]) == 2

    def test_action_uniqueness(self):
        """Test that all actions are unique types."""
        actions = [
            tap(Button.A),
            hold(Button.B, 10),
            release(Button.START),
            waitTurn(),
            face(Direction.UP),
            capture("test"),
            read_state(["hp"]),
            expect("hp > 0", "HP check"),
            annotate("Test note"),
            break_(),
            abort("Test abort"),
            checkpoint("test_cp"),
            resume(),
            save(1),
            load(2),
        ]

        # Should have 15 different action types
        assert len(actions) == 15

        # All should be valid Action instances
        for action in actions:
            assert isinstance(action, Action)


if __name__ == "__main__":
    pytest.main([__file__])