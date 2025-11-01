"""Tests for bootstrap state management system.

Tests saving and loading checkpoint state for cross-run continuity,
bootstrap mode, and history tracking.
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from src.bootstrap.state_manager import BootstrapCheckpoint, BootstrapStateManager


class TestBootstrapCheckpoint:
    """Test BootstrapCheckpoint dataclass."""

    def test_checkpoint_creation(self) -> None:
        """Test creating a checkpoint."""
        checkpoint = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=datetime.now().isoformat()
        )
        assert checkpoint.run_id == "run_001"
        assert checkpoint.is_bootstrap is False
        assert checkpoint.parent_run_id is None

    def test_checkpoint_with_learned_skills(self) -> None:
        """Test checkpoint with learned skills."""
        skills = [
            {"name": "skill_1", "success": 0.9},
            {"name": "skill_2", "success": 0.8},
        ]
        checkpoint = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=datetime.now().isoformat(),
            learned_skills=skills
        )
        assert len(checkpoint.learned_skills) == 2
        assert checkpoint.learned_skills[0]["name"] == "skill_1"

    def test_checkpoint_with_game_state(self) -> None:
        """Test checkpoint with game state metadata."""
        checkpoint = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=datetime.now().isoformat(),
            last_known_position={"x": 10, "y": 20},
            last_known_hp=100,
            dungeon_level=3
        )
        assert checkpoint.last_known_position == {"x": 10, "y": 20}
        assert checkpoint.last_known_hp == 100
        assert checkpoint.dungeon_level == 3

    def test_checkpoint_bootstrap_flag(self) -> None:
        """Test bootstrap flag and parent tracking."""
        checkpoint = BootstrapCheckpoint(
            run_id="run_002",
            timestamp=datetime.now().isoformat(),
            is_bootstrap=True,
            parent_run_id="run_001"
        )
        assert checkpoint.is_bootstrap is True
        assert checkpoint.parent_run_id == "run_001"

    def test_checkpoint_statistics(self) -> None:
        """Test checkpoint statistics."""
        checkpoint = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=datetime.now().isoformat(),
            total_steps=250,
            success_rate=0.75,
            skills_discovered=5
        )
        assert checkpoint.total_steps == 250
        assert checkpoint.success_rate == 0.75
        assert checkpoint.skills_discovered == 5

    def test_checkpoint_to_dict(self) -> None:
        """Test checkpoint serialization."""
        now = datetime.now().isoformat()
        checkpoint = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=now,
            learned_skills=[{"name": "skill_1"}],
            total_steps=100,
            is_bootstrap=True,
            parent_run_id="run_000"
        )

        checkpoint_dict = checkpoint.to_dict()
        assert checkpoint_dict["run_id"] == "run_001"
        assert checkpoint_dict["timestamp"] == now
        assert len(checkpoint_dict["learned_skills"]) == 1
        assert checkpoint_dict["total_steps"] == 100
        assert checkpoint_dict["is_bootstrap"] is True
        assert checkpoint_dict["parent_run_id"] == "run_000"

    def test_checkpoint_from_dict(self) -> None:
        """Test checkpoint deserialization."""
        now = datetime.now().isoformat()
        data = {
            "run_id": "run_001",
            "timestamp": now,
            "learned_skills": [{"name": "skill_1"}],
            "memory_buffer": {"key": "value"},
            "trajectory_embeddings": {},
            "last_known_position": None,
            "last_known_hp": None,
            "dungeon_level": None,
            "is_bootstrap": False,
            "parent_run_id": None,
            "total_steps": 100,
            "success_rate": 0.8,
            "skills_discovered": 3
        }

        checkpoint = BootstrapCheckpoint.from_dict(data)
        assert checkpoint.run_id == "run_001"
        assert checkpoint.total_steps == 100
        assert len(checkpoint.learned_skills) == 1

    def test_checkpoint_roundtrip(self) -> None:
        """Test serialization and deserialization roundtrip."""
        original = BootstrapCheckpoint(
            run_id="run_001",
            timestamp=datetime.now().isoformat(),
            learned_skills=[{"name": "skill_1", "success": 0.9}],
            memory_buffer={"buffer": [1, 2, 3]},
            trajectory_embeddings={"traj_1": [0.1, 0.2, 0.3]},
            total_steps=150,
            is_bootstrap=True,
            parent_run_id="run_000"
        )

        # Serialize and deserialize
        data = original.to_dict()
        recreated = BootstrapCheckpoint.from_dict(data)

        assert recreated.run_id == original.run_id
        assert recreated.total_steps == original.total_steps
        assert recreated.learned_skills == original.learned_skills
        assert recreated.is_bootstrap == original.is_bootstrap
        assert recreated.parent_run_id == original.parent_run_id


class TestBootstrapStateManager:
    """Test BootstrapStateManager class."""

    @pytest.fixture
    def temp_bootstrap_dir(self) -> Path:
        """Create temporary directory for bootstrap data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_bootstrap_dir: Path) -> BootstrapStateManager:
        """Create BootstrapStateManager with temp directory."""
        return BootstrapStateManager(bootstrap_dir=temp_bootstrap_dir)

    def test_manager_initialization(self, manager: BootstrapStateManager) -> None:
        """Test manager initialization."""
        assert manager.bootstrap_dir.exists()
        assert manager.enable_bootstrap is True
        assert manager.current_checkpoint is None
        assert len(manager.checkpoint_history) == 0

    def test_create_checkpoint(self, manager: BootstrapStateManager) -> None:
        """Test creating a new checkpoint."""
        checkpoint = manager.create_checkpoint("run_001")

        assert checkpoint.run_id == "run_001"
        assert manager.current_checkpoint == checkpoint
        assert checkpoint.is_bootstrap is False

    def test_save_checkpoint(self, manager: BootstrapStateManager) -> None:
        """Test saving checkpoint to disk."""
        skills = [{"name": "skill_1", "success": 0.9}]
        memory = {"buffer": [1, 2, 3]}
        embeddings = {"traj_1": [0.1, 0.2, 0.3]}

        result = manager.save_checkpoint(
            run_id="run_001",
            learned_skills=skills,
            memory_buffer=memory,
            trajectory_embeddings=embeddings
        )

        assert result is True

        # Check file was created
        checkpoint_path = manager._get_checkpoint_path("run_001")
        assert checkpoint_path.exists()

        # Check content
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            assert data["run_id"] == "run_001"
            assert len(data["learned_skills"]) == 1

    def test_save_checkpoint_with_game_state(self, manager: BootstrapStateManager) -> None:
        """Test saving checkpoint with game state."""
        game_state = {
            "position": {"x": 10, "y": 20},
            "hp": 100,
            "dungeon_level": 3
        }
        stats = {
            "total_steps": 250,
            "success_rate": 0.8,
            "skills_discovered": 5
        }

        result = manager.save_checkpoint(
            run_id="run_001",
            learned_skills=[],
            memory_buffer={},
            game_state=game_state,
            stats=stats
        )

        assert result is True

        # Load and verify
        checkpoint = manager.load_checkpoint("run_001")
        assert checkpoint.last_known_position == {"x": 10, "y": 20}
        assert checkpoint.last_known_hp == 100
        assert checkpoint.total_steps == 250

    def test_load_latest_checkpoint(self, manager: BootstrapStateManager) -> None:
        """Test loading the most recent checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            manager.save_checkpoint(
                run_id=f"run_{i:03d}",
                learned_skills=[],
                memory_buffer={}
            )
            time.sleep(0.01)

        # Load latest
        latest = manager.load_latest_checkpoint()
        assert latest is not None
        assert latest.run_id == "run_002"

    def test_load_specific_checkpoint(self, manager: BootstrapStateManager) -> None:
        """Test loading a specific checkpoint."""
        manager.save_checkpoint(
            run_id="run_001",
            learned_skills=[{"name": "skill_1"}],
            memory_buffer={"key": "value"}
        )

        checkpoint = manager.load_checkpoint("run_001")
        assert checkpoint is not None
        assert checkpoint.run_id == "run_001"
        assert len(checkpoint.learned_skills) == 1

    def test_load_nonexistent_checkpoint(self, manager: BootstrapStateManager) -> None:
        """Test loading nonexistent checkpoint returns None."""
        result = manager.load_checkpoint("nonexistent")
        assert result is None

    def test_list_checkpoints(self, manager: BootstrapStateManager) -> None:
        """Test listing checkpoints."""
        for i in range(5):
            manager.save_checkpoint(
                run_id=f"run_{i:03d}",
                learned_skills=[],
                memory_buffer={}
            )
            time.sleep(0.01)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 5

        # Should be sorted by recency (most recent first)
        run_ids = [c.run_id for c in checkpoints]
        expected = sorted(run_ids, reverse=True)
        assert run_ids == expected

    def test_list_checkpoints_with_limit(self, manager: BootstrapStateManager) -> None:
        """Test limiting checkpoint list."""
        for i in range(10):
            manager.save_checkpoint(
                run_id=f"run_{i:03d}",
                learned_skills=[],
                memory_buffer={}
            )
            time.sleep(0.01)

        limited = manager.list_checkpoints(limit=3)
        assert len(limited) == 3
        # Should be the 3 most recent (most recent first)
        assert limited[0].run_id == "run_009"
        assert limited[1].run_id == "run_008"
        assert limited[2].run_id == "run_007"

    def test_create_bootstrap_run(self, manager: BootstrapStateManager) -> None:
        """Test creating a run that bootstraps from previous."""
        # Save parent run
        manager.save_checkpoint(
            run_id="run_001",
            learned_skills=[{"name": "skill_1"}],
            memory_buffer={"memory": [1, 2, 3]},
            trajectory_embeddings={"traj": [0.1, 0.2]},
            stats={"total_steps": 100, "skills_discovered": 1}
        )

        # Create bootstrap run
        bootstrap_checkpoint = manager.create_bootstrap_run(
            new_run_id="run_002",
            parent_run_id="run_001"
        )

        assert bootstrap_checkpoint.run_id == "run_002"
        assert bootstrap_checkpoint.is_bootstrap is True
        assert bootstrap_checkpoint.parent_run_id == "run_001"

        # Should inherit parent's data
        assert len(bootstrap_checkpoint.learned_skills) == 1
        assert bootstrap_checkpoint.memory_buffer == {"memory": [1, 2, 3]}

    def test_bootstrap_run_is_independent_copy(self, manager: BootstrapStateManager) -> None:
        """Test that bootstrap run data is independent copy, not reference."""
        # Save parent
        parent_skills = [{"name": "skill_1"}]
        parent_memory = {"key": "value"}
        manager.save_checkpoint(
            run_id="run_001",
            learned_skills=parent_skills,
            memory_buffer=parent_memory
        )

        # Create bootstrap
        bootstrap = manager.create_bootstrap_run("run_002", "run_001")

        # Modify bootstrap data
        bootstrap.learned_skills.append({"name": "skill_2"})

        # Load original - should be unchanged
        parent = manager.load_checkpoint("run_001")
        assert len(parent.learned_skills) == 1

    def test_bootstrap_from_nonexistent_parent(self, manager: BootstrapStateManager) -> None:
        """Test bootstrap from nonexistent parent gracefully fails."""
        bootstrap = manager.create_bootstrap_run("run_001", "nonexistent")

        # Should create a normal checkpoint
        assert bootstrap.run_id == "run_001"
        assert bootstrap.is_bootstrap is False

    def test_get_bootstrap_status(self, manager: BootstrapStateManager) -> None:
        """Test getting bootstrap status information."""
        for i in range(3):
            manager.save_checkpoint(
                run_id=f"run_{i:03d}",
                learned_skills=[],
                memory_buffer={},
                stats={"total_steps": 100 * (i+1)}
            )
            time.sleep(0.01)

        status = manager.get_bootstrap_status()

        assert status["enabled"] is True
        assert status["total_checkpoints"] == 3
        assert status["current_checkpoint"] == "run_002"
        assert len(status["recent_checkpoints"]) == 3

    def test_cleanup_old_checkpoints(self, manager: BootstrapStateManager) -> None:
        """Test cleaning up old checkpoints."""
        # Create 15 checkpoints
        for i in range(15):
            manager.save_checkpoint(
                run_id=f"run_{i:03d}",
                learned_skills=[],
                memory_buffer={}
            )
            time.sleep(0.01)

        # Cleanup, keeping only 10
        deleted = manager.cleanup_old_checkpoints(keep_count=10)

        assert deleted == 5
        assert len(manager.checkpoint_history) == 10

        # Should keep the most recent
        remaining = manager.list_checkpoints()
        assert remaining[0].run_id == "run_014"
        assert remaining[-1].run_id == "run_005"

    def test_bootstrap_disabled(self, temp_bootstrap_dir: Path) -> None:
        """Test behavior when bootstrap is disabled."""
        manager = BootstrapStateManager(
            bootstrap_dir=temp_bootstrap_dir,
            enable_bootstrap=False
        )

        result = manager.save_checkpoint(
            run_id="run_001",
            learned_skills=[],
            memory_buffer={}
        )

        assert result is False

    def test_checkpoint_path_sanitization(self, manager: BootstrapStateManager) -> None:
        """Test that run IDs are sanitized in file paths."""
        unsafe_ids = [
            "run/with/slashes",
            "run:with:colons",
            "run|with|pipes",
            "run with spaces",
        ]

        for unsafe_id in unsafe_ids:
            path = manager._get_checkpoint_path(unsafe_id)
            # Path should only contain safe characters
            filename = path.name
            assert all(c.isalnum() or c in "_-." for c in filename)

    def test_persistence_across_instances(self, temp_bootstrap_dir: Path) -> None:
        """Test that checkpoints persist across manager instances."""
        # Save with first manager
        manager1 = BootstrapStateManager(bootstrap_dir=temp_bootstrap_dir)
        manager1.save_checkpoint(
            run_id="persistent",
            learned_skills=[{"name": "persistent_skill"}],
            memory_buffer={"persistent": True}
        )

        # Load with second manager
        manager2 = BootstrapStateManager(bootstrap_dir=temp_bootstrap_dir)
        checkpoint = manager2.load_checkpoint("persistent")

        assert checkpoint is not None
        assert checkpoint.run_id == "persistent"
        assert len(checkpoint.learned_skills) == 1

    def test_load_checkpoint_history_on_init(self, temp_bootstrap_dir: Path) -> None:
        """Test that manager loads all existing checkpoints on initialization."""
        # Create checkpoints with first manager
        manager1 = BootstrapStateManager(bootstrap_dir=temp_bootstrap_dir)
        for i in range(3):
            manager1.save_checkpoint(
                run_id=f"run_{i}",
                learned_skills=[],
                memory_buffer={}
            )

        # Second manager should auto-load history
        manager2 = BootstrapStateManager(bootstrap_dir=temp_bootstrap_dir)
        assert len(manager2.checkpoint_history) == 3

    def test_large_skill_list(self, manager: BootstrapStateManager) -> None:
        """Test handling large number of learned skills."""
        large_skill_list = [
            {"name": f"skill_{i}", "success": 0.5 + (i % 50) / 100}
            for i in range(1000)
        ]

        result = manager.save_checkpoint(
            run_id="large_run",
            learned_skills=large_skill_list,
            memory_buffer={}
        )

        assert result is True

        loaded = manager.load_checkpoint("large_run")
        assert len(loaded.learned_skills) == 1000

    def test_large_memory_buffer(self, manager: BootstrapStateManager) -> None:
        """Test handling large memory buffers."""
        large_memory = {
            f"frame_{i}": {
                "observation": [0.1] * 100,
                "action": "move",
                "reward": 1.0
            }
            for i in range(500)
        }

        result = manager.save_checkpoint(
            run_id="large_memory",
            learned_skills=[],
            memory_buffer=large_memory
        )

        assert result is True

        loaded = manager.load_checkpoint("large_memory")
        assert len(loaded.memory_buffer) == 500

    def test_unicode_in_run_id(self, manager: BootstrapStateManager) -> None:
        """Test handling unicode in run IDs."""
        run_id = "run_日本語_🎮"
        result = manager.save_checkpoint(
            run_id=run_id,
            learned_skills=[],
            memory_buffer={}
        )

        assert result is True

        # Path should be safe despite unicode in ID
        path = manager._get_checkpoint_path(run_id)
        assert path.exists()
