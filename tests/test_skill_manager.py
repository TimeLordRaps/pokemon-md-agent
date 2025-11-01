"""Comprehensive tests for skill management system.

Tests SkillStep, Skill, and SkillManager classes for persistence,
retrieval, filtering, and success rate tracking.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from src.skills.skill_manager import Skill, SkillManager, SkillStep


class TestSkillStep:
    """Test SkillStep dataclass."""

    def test_skill_step_creation(self) -> None:
        """Test creating a SkillStep with required fields."""
        step = SkillStep(
            action="move_down",
            confidence=0.95,
        )
        assert step.action == "move_down"
        assert step.confidence == 0.95
        assert step.observation is None
        assert step.reasoning is None

    def test_skill_step_with_optional_fields(self) -> None:
        """Test SkillStep with all fields populated."""
        obs = {"position": (10, 20), "hp": 100}
        step = SkillStep(
            action="confirm",
            confidence=0.85,
            observation=obs,
            reasoning="Player at entrance"
        )
        assert step.action == "confirm"
        assert step.confidence == 0.85
        assert step.observation == obs
        assert step.reasoning == "Player at entrance"

    def test_skill_step_confidence_bounds(self) -> None:
        """Test that confidence values are preserved (validation optional)."""
        step_low = SkillStep(action="test", confidence=0.0)
        step_high = SkillStep(action="test", confidence=1.0)
        assert step_low.confidence == 0.0
        assert step_high.confidence == 1.0

    def test_skill_step_dict_roundtrip(self) -> None:
        """Test SkillStep serialization and deserialization."""
        original = SkillStep(
            action="move_left",
            confidence=0.75,
            observation={"level": 1},
            reasoning="Exploration"
        )

        # Simulate dict conversion
        step_dict = {
            "action": original.action,
            "confidence": original.confidence,
            "observation": original.observation,
            "reasoning": original.reasoning
        }

        recreated = SkillStep(**step_dict)
        assert recreated.action == original.action
        assert recreated.confidence == original.confidence
        assert recreated.observation == original.observation
        assert recreated.reasoning == original.reasoning


class TestSkill:
    """Test Skill dataclass."""

    def test_skill_creation_minimal(self) -> None:
        """Test creating a Skill with minimal required fields."""
        skill = Skill(
            name="navigate_down",
            description="Move down to next room"
        )
        assert skill.name == "navigate_down"
        assert skill.description == "Move down to next room"
        assert skill.steps == []
        assert skill.precondition is None
        assert skill.postcondition is None
        assert skill.success_rate == 0.0
        assert skill.times_used == 0
        assert skill.tags == []

    def test_skill_creation_with_steps(self) -> None:
        """Test creating a Skill with step sequences."""
        steps = [
            SkillStep(action="move_down", confidence=0.9),
            SkillStep(action="move_down", confidence=0.9),
            SkillStep(action="confirm", confidence=0.85),
        ]
        skill = Skill(
            name="traverse_hall",
            description="Traverse a hallway",
            steps=steps,
            precondition="in_hallway",
            postcondition="reached_stairs",
        )
        assert len(skill.steps) == 3
        assert skill.precondition == "in_hallway"
        assert skill.postcondition == "reached_stairs"

    def test_skill_success_rate_tracking(self) -> None:
        """Test success rate field."""
        skill = Skill(
            name="test_skill",
            description="Test"
        )
        assert skill.success_rate == 0.0

        # Simulate updating success rate
        skill.success_rate = 0.8
        assert skill.success_rate == 0.8

    def test_skill_usage_tracking(self) -> None:
        """Test usage counting."""
        skill = Skill(name="test", description="Test")
        assert skill.times_used == 0

        skill.times_used += 1
        assert skill.times_used == 1

    def test_skill_timestamps(self) -> None:
        """Test skill timestamp fields."""
        skill = Skill(name="test", description="Test")

        # created_at should be set automatically
        assert skill.created_at is not None
        assert "T" in skill.created_at  # ISO format check

        # last_used_at initially None
        assert skill.last_used_at is None

        # Update last used
        now_iso = datetime.now().isoformat()
        skill.last_used_at = now_iso
        assert skill.last_used_at == now_iso

    def test_skill_tags(self) -> None:
        """Test skill tagging."""
        skill = Skill(
            name="test",
            description="Test",
            tags=["exploration", "combat"]
        )
        assert "exploration" in skill.tags
        assert "combat" in skill.tags

    def test_skill_to_dict(self) -> None:
        """Test Skill serialization to dict."""
        steps = [SkillStep(action="move", confidence=0.9)]
        skill = Skill(
            name="test_skill",
            description="A test skill",
            steps=steps,
            precondition="start",
            postcondition="end",
            success_rate=0.75,
            times_used=5,
            tags=["test"]
        )

        skill_dict = skill.to_dict()
        assert skill_dict["name"] == "test_skill"
        assert skill_dict["description"] == "A test skill"
        assert len(skill_dict["steps"]) == 1
        assert skill_dict["steps"][0]["action"] == "move"
        assert skill_dict["precondition"] == "start"
        assert skill_dict["postcondition"] == "end"
        assert skill_dict["success_rate"] == 0.75
        assert skill_dict["times_used"] == 5
        assert skill_dict["tags"] == ["test"]

    def test_skill_from_dict(self) -> None:
        """Test Skill deserialization from dict."""
        skill_dict = {
            "name": "test_skill",
            "description": "Test",
            "steps": [
                {"action": "move", "confidence": 0.9, "observation": None, "reasoning": None}
            ],
            "precondition": "start",
            "postcondition": "end",
            "success_rate": 0.8,
            "times_used": 3,
            "created_at": datetime.now().isoformat(),
            "last_used_at": None,
            "tags": ["test"]
        }

        skill = Skill.from_dict(skill_dict)
        assert skill.name == "test_skill"
        assert skill.description == "Test"
        assert len(skill.steps) == 1
        assert skill.steps[0].action == "move"
        assert skill.success_rate == 0.8
        assert skill.times_used == 3

    def test_skill_roundtrip(self) -> None:
        """Test Skill serialization and deserialization roundtrip."""
        original = Skill(
            name="roundtrip_test",
            description="Test roundtrip",
            steps=[SkillStep(action="test", confidence=0.5)],
            precondition="p1",
            postcondition="p2",
            success_rate=0.7,
            times_used=2,
            tags=["tag1", "tag2"]
        )

        # Serialize
        skill_dict = original.to_dict()

        # Deserialize
        recreated = Skill.from_dict(skill_dict)

        # Compare
        assert recreated.name == original.name
        assert recreated.description == original.description
        assert len(recreated.steps) == len(original.steps)
        assert recreated.steps[0].action == original.steps[0].action
        assert recreated.precondition == original.precondition
        assert recreated.postcondition == original.postcondition
        assert recreated.success_rate == original.success_rate
        assert recreated.times_used == original.times_used
        assert recreated.tags == original.tags


class TestSkillManager:
    """Test SkillManager class."""

    @pytest.fixture
    def temp_skills_dir(self) -> Path:
        """Create a temporary directory for skill storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_skills_dir: Path) -> SkillManager:
        """Create a SkillManager with temporary directory."""
        return SkillManager(skills_dir=temp_skills_dir)

    def test_manager_initialization(self, manager: SkillManager) -> None:
        """Test SkillManager initialization creates directory."""
        assert manager.skills_dir.exists()
        assert manager.skills_dir.is_dir()
        assert isinstance(manager.skills, dict)
        assert len(manager.skills) == 0

    def test_save_skill(self, manager: SkillManager) -> None:
        """Test saving a skill to disk."""
        skill = Skill(
            name="test_skill",
            description="A test skill",
            steps=[SkillStep(action="move", confidence=0.9)]
        )

        result = manager.save_skill(skill)
        assert result is True
        assert skill.name in manager.skills
        assert manager.skills[skill.name] == skill

        # Check file was created
        skill_file = manager._get_skill_path(skill.name)
        assert skill_file.exists()

    def test_save_skill_with_special_characters(self, manager: SkillManager) -> None:
        """Test saving skill with special characters in name."""
        skill = Skill(
            name="move_down/left!@#$%",
            description="Test",
            steps=[]
        )

        result = manager.save_skill(skill)
        assert result is True

        # Check that special chars are sanitized
        skill_file = manager._get_skill_path(skill.name)
        assert skill_file.exists()
        assert "/" not in skill_file.name
        assert "@" not in skill_file.name

    def test_load_skill(self, manager: SkillManager) -> None:
        """Test loading a skill from disk."""
        original = Skill(
            name="load_test",
            description="Test loading",
            steps=[SkillStep(action="test", confidence=0.8)],
            precondition="start"
        )

        manager.save_skill(original)

        loaded = manager.load_skill("load_test")
        assert loaded is not None
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert len(loaded.steps) == 1
        assert loaded.precondition == original.precondition

    def test_load_nonexistent_skill(self, manager: SkillManager) -> None:
        """Test loading a nonexistent skill returns None."""
        result = manager.load_skill("nonexistent")
        assert result is None

    def test_delete_skill(self, manager: SkillManager) -> None:
        """Test deleting a skill."""
        skill = Skill(name="delete_test", description="Test")
        manager.save_skill(skill)

        assert "delete_test" in manager.skills
        skill_file = manager._get_skill_path("delete_test")
        assert skill_file.exists()

        result = manager.delete_skill("delete_test")
        assert result is True
        assert "delete_test" not in manager.skills
        assert not skill_file.exists()

    def test_delete_nonexistent_skill(self, manager: SkillManager) -> None:
        """Test deleting nonexistent skill succeeds gracefully."""
        result = manager.delete_skill("nonexistent")
        assert result is True  # Graceful failure

    def test_list_skills(self, manager: SkillManager) -> None:
        """Test listing all skills."""
        skills = [
            Skill(name="skill_a", description="A"),
            Skill(name="skill_b", description="B"),
            Skill(name="skill_c", description="C"),
        ]

        for skill in skills:
            manager.save_skill(skill)

        listed = manager.list_skills()
        assert len(listed) == 3

        # Should be sorted by name
        names = [s.name for s in listed]
        assert names == sorted(names)

    def test_list_skills_with_tag_filter(self, manager: SkillManager) -> None:
        """Test listing skills filtered by tags."""
        skills = [
            Skill(name="s1", description="", tags=["combat", "quick"]),
            Skill(name="s2", description="", tags=["exploration"]),
            Skill(name="s3", description="", tags=["combat", "slow"]),
        ]

        for skill in skills:
            manager.save_skill(skill)

        # Filter by combat
        combat_skills = manager.list_skills(tags=["combat"])
        assert len(combat_skills) == 2
        assert all("combat" in s.tags for s in combat_skills)

        # Filter by exploration
        exploration_skills = manager.list_skills(tags=["exploration"])
        assert len(exploration_skills) == 1
        assert exploration_skills[0].name == "s2"

    def test_list_skills_empty(self, manager: SkillManager) -> None:
        """Test listing skills when none exist."""
        listed = manager.list_skills()
        assert listed == []

    def test_create_skill_from_trajectory(self, manager: SkillManager) -> None:
        """Test creating a skill from action trajectory."""
        actions = ["move_down", "move_down", "confirm"]
        confidences = [0.9, 0.9, 0.85]

        skill = manager.create_skill_from_trajectory(
            name="dungeon_enter",
            actions=actions,
            confidences=confidences,
            description="Enter dungeon entrance",
            precondition="at_entrance",
            postcondition="in_dungeon",
            tags=["exploration"]
        )

        assert skill is not None
        assert skill.name == "dungeon_enter"
        assert len(skill.steps) == 3
        assert skill.steps[0].action == "move_down"
        assert skill.steps[0].confidence == 0.9
        assert skill.precondition == "at_entrance"
        assert skill.postcondition == "in_dungeon"
        assert "exploration" in skill.tags

    def test_create_skill_from_trajectory_mismatch(self, manager: SkillManager) -> None:
        """Test that mismatched actions and confidences returns None."""
        skill = manager.create_skill_from_trajectory(
            name="bad_skill",
            actions=["move", "move"],
            confidences=[0.9],  # Wrong length
        )

        assert skill is None

    def test_update_skill_success_increment(self, manager: SkillManager) -> None:
        """Test updating skill with successful execution."""
        skill = Skill(
            name="success_test",
            description="Test",
            success_rate=0.5,
            times_used=10
        )
        manager.save_skill(skill)

        result = manager.update_skill_success("success_test", succeeded=True)
        assert result is True

        updated = manager.load_skill("success_test")
        assert updated.times_used == 11
        # EMA: 0.9 * 0.5 + 0.1 * 1.0 = 0.45 + 0.1 = 0.55
        assert abs(updated.success_rate - 0.55) < 0.01

    def test_update_skill_failure(self, manager: SkillManager) -> None:
        """Test updating skill with failed execution."""
        skill = Skill(
            name="fail_test",
            description="Test",
            success_rate=0.8,
            times_used=5
        )
        manager.save_skill(skill)

        result = manager.update_skill_success("fail_test", succeeded=False)
        assert result is True

        updated = manager.load_skill("fail_test")
        assert updated.times_used == 6
        # EMA: 0.9 * 0.8 + 0.1 * 0.0 = 0.72
        assert abs(updated.success_rate - 0.72) < 0.01

    def test_update_nonexistent_skill(self, manager: SkillManager) -> None:
        """Test updating nonexistent skill returns False."""
        result = manager.update_skill_success("nonexistent", succeeded=True)
        assert result is False

    def test_find_similar_skills(self, manager: SkillManager) -> None:
        """Test finding skills by precondition and postcondition."""
        skills = [
            Skill(
                name="s1",
                description="",
                precondition="room_a",
                postcondition="room_b",
                success_rate=0.9
            ),
            Skill(
                name="s2",
                description="",
                precondition="room_a",
                postcondition="room_c",
                success_rate=0.8
            ),
            Skill(
                name="s3",
                description="",
                precondition="room_a",
                postcondition="room_b",
                success_rate=0.95
            ),
            Skill(
                name="s4",
                description="",
                precondition="room_x",
                postcondition="room_b",
                success_rate=0.85
            ),
        ]

        for skill in skills:
            manager.save_skill(skill)

        # Find by precondition only
        matches = manager.find_similar_skills("room_a", postcondition="room_b")
        assert len(matches) == 2
        # Should be sorted by success rate
        assert matches[0].success_rate >= matches[1].success_rate
        assert matches[0].name == "s3"  # 0.95 > 0.9

    def test_find_similar_skills_max_results(self, manager: SkillManager) -> None:
        """Test max_results parameter."""
        for i in range(10):
            skill = Skill(
                name=f"skill_{i}",
                description="",
                precondition="test",
                success_rate=float(i) / 10
            )
            manager.save_skill(skill)

        matches = manager.find_similar_skills("test", max_results=3)
        assert len(matches) <= 3

    def test_persistence_across_instances(self, temp_skills_dir: Path) -> None:
        """Test that skills persisted by one manager are loaded by another."""
        # Create and save with first manager
        manager1 = SkillManager(skills_dir=temp_skills_dir)
        skill = Skill(
            name="persistent",
            description="Should persist",
            steps=[SkillStep(action="test", confidence=0.9)]
        )
        manager1.save_skill(skill)

        # Create new manager with same directory
        manager2 = SkillManager(skills_dir=temp_skills_dir)

        # Should load the saved skill
        loaded = manager2.load_skill("persistent")
        assert loaded is not None
        assert loaded.name == "persistent"
        assert loaded.description == "Should persist"
        assert len(loaded.steps) == 1

    def test_load_all_skills_on_init(self, temp_skills_dir: Path) -> None:
        """Test that manager loads all existing skills on initialization."""
        # Create and save multiple skills
        manager1 = SkillManager(skills_dir=temp_skills_dir)
        for i in range(3):
            skill = Skill(name=f"skill_{i}", description=f"Skill {i}")
            manager1.save_skill(skill)

        # Create new manager - should auto-load all skills
        manager2 = SkillManager(skills_dir=temp_skills_dir)
        assert len(manager2.skills) == 3
        assert all(f"skill_{i}" in manager2.skills for i in range(3))

    def test_get_skill_path_sanitization(self, manager: SkillManager) -> None:
        """Test that skill paths are properly sanitized."""
        unsafe_names = [
            "skill/with/slashes",
            "skill..with..dots",
            "skill with spaces",
            "skill:with:colons",
            "skill|with|pipes",
        ]

        for name in unsafe_names:
            path = manager._get_skill_path(name)
            # Path should only contain alphanumeric, underscore, hyphen
            filename = path.name
            assert all(c.isalnum() or c in "_-." for c in filename), f"Unsafe chars in {filename}"


class TestSkillManagerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def manager(self) -> SkillManager:
        """Create manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SkillManager(skills_dir=Path(tmpdir))

    def test_empty_skill_name(self, manager: SkillManager) -> None:
        """Test handling of empty skill name."""
        skill = Skill(name="", description="Empty name")
        # Should still work (sanitization handles empty)
        result = manager.save_skill(skill)
        assert isinstance(result, bool)

    def test_large_step_sequence(self, manager: SkillManager) -> None:
        """Test skill with large number of steps."""
        steps = [SkillStep(action=f"action_{i}", confidence=0.9) for i in range(1000)]
        skill = Skill(name="large_skill", description="Large", steps=steps)

        manager.save_skill(skill)
        loaded = manager.load_skill("large_skill")

        assert loaded is not None
        assert len(loaded.steps) == 1000

    def test_unicode_in_skill_metadata(self, manager: SkillManager) -> None:
        """Test handling of unicode characters."""
        skill = Skill(
            name="unicode_test",
            description="Skill with émojis 🎮 and spéciål çhårs",
            tags=["日本語", "中文"]
        )

        manager.save_skill(skill)
        loaded = manager.load_skill("unicode_test")

        assert loaded is not None
        assert "émojis 🎮" in loaded.description
        assert "日本語" in loaded.tags

    def test_concurrent_access_same_skill(self, manager: SkillManager) -> None:
        """Test that updating same skill multiple times works correctly."""
        skill = Skill(name="concurrent_test", description="Test", success_rate=0.5)
        manager.save_skill(skill)

        # Simulate concurrent updates
        for i in range(5):
            manager.update_skill_success("concurrent_test", succeeded=True)

        final = manager.load_skill("concurrent_test")
        assert final.times_used == 5

    def test_skill_with_none_optional_fields(self, manager: SkillManager) -> None:
        """Test skill roundtrip with all None optional fields."""
        skill = Skill(
            name="minimal",
            description="Minimal skill",
            precondition=None,
            postcondition=None,
            tags=[]
        )

        manager.save_skill(skill)
        loaded = manager.load_skill("minimal")

        assert loaded.precondition is None
        assert loaded.postcondition is None
        assert loaded.tags == []
