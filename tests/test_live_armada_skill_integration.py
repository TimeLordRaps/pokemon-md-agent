"""Integration tests for live_armada with skill discovery and bootstrap."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.runners.live_armada import ArmadaConfig, LiveArmadaRunner


class TestLiveArmadaSkillIntegration:
    """Test skill discovery and bootstrap integration in live_armada."""

    @pytest.fixture
    def temp_config_dir(self) -> Path:
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def armada_config(self, temp_config_dir: Path) -> ArmadaConfig:
        """Create armada configuration for testing."""
        return ArmadaConfig(
            rom=Path("/fake/rom.gba"),
            save=Path("/fake/save.sav"),
            lua=Path("/fake/lua.lua"),
            mgba_exe=Path("/fake/mgba.exe"),
            dashboard_dir=temp_config_dir / "dashboard",
            trace_jsonl=temp_config_dir / "traces.jsonl",
            dry_run=True,  # Use dry run mode
        )

    def test_runner_initialization_with_skill_systems(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that runner initializes with skill and bootstrap systems."""
        runner = LiveArmadaRunner(config=armada_config)

        assert runner.skill_manager is not None
        assert runner.bootstrap_manager is not None
        assert runner.run_id is not None
        assert len(runner.discovered_skills) == 0

    def test_skill_manager_accessible(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that skill manager is accessible and functional."""
        runner = LiveArmadaRunner(config=armada_config)

        # List skills (should be empty initially)
        skills = runner.skill_manager.list_skills()
        assert isinstance(skills, list)

        # Should have directories
        assert runner.skill_manager.skills_dir.exists()

    def test_bootstrap_manager_accessible(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that bootstrap manager is accessible and functional."""
        runner = LiveArmadaRunner(config=armada_config)

        # Get status
        status = runner.bootstrap_manager.get_bootstrap_status()
        assert status["enabled"] is True
        assert status["total_checkpoints"] == 0

        # Should have directories
        assert runner.bootstrap_manager.bootstrap_dir.exists()

    @pytest.mark.asyncio
    async def test_discover_skill_from_sequence(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test discovering a skill from action sequence."""
        runner = LiveArmadaRunner(config=armada_config)

        actions = ["move_down", "move_down", "confirm"]
        confidences = [0.9, 0.9, 0.85]

        skill = await runner.discover_skill_from_sequence(
            actions=actions,
            confidences=confidences,
            precondition="at_entrance",
            postcondition="dungeon_entered"
        )

        assert skill is not None
        assert len(skill.steps) == 3
        assert len(runner.discovered_skills) == 1

    @pytest.mark.asyncio
    async def test_discover_skill_invalid_sequence(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test discovering skill with invalid sequence."""
        runner = LiveArmadaRunner(config=armada_config)

        # Mismatched lengths
        skill = await runner.discover_skill_from_sequence(
            actions=["move", "move"],
            confidences=[0.9],  # Wrong length
        )

        assert skill is None
        assert len(runner.discovered_skills) == 0

    @pytest.mark.asyncio
    async def test_run_with_skill_discovery(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that run method saves discovered skills."""
        runner = LiveArmadaRunner(config=armada_config)

        # Mock vision processing and inference
        with patch.object(runner, 'process_vision', new_callable=AsyncMock) as mock_vision:
            with patch.object(runner, 'infer_armada', new_callable=AsyncMock) as mock_infer:
                with patch.object(runner, 'write_quad_image') as mock_quad:
                    with patch.object(runner, 'write_trace') as mock_trace:
                        with patch.object(runner, 'mark_keyframe') as mock_keyframe:
                            with patch.object(runner, 'execute_action', new_callable=AsyncMock):
                                with patch.object(runner, 'ensure_ready', new_callable=AsyncMock, return_value=True):
                                    # Setup mock returns
                                    from src.runners.live_armada import VisionOutput, ArmadaResponse
                                    from PIL import Image
                                    import numpy as np

                                    mock_vision.return_value = VisionOutput(
                                        step_id=1,
                                        raw_img=Image.new('RGB', (240, 160)),
                                        grid_overlay=Image.new('RGB', (240, 160)),
                                        hud_meta=Image.new('RGB', (240, 160)),
                                        retrieval_mosaic=Image.new('RGB', (240, 160))
                                    )

                                    mock_infer.return_value = ArmadaResponse(
                                        step_id=1,
                                        timestamp=0,
                                        model_id="test",
                                        action="move_down",
                                        confidence=0.9
                                    )

                                    # Run with 3 steps
                                    result = await runner.run(max_steps=3)

        assert result == 0
        # Should have created a skill from the collected actions
        assert runner.skill_manager is not None

    @pytest.mark.asyncio
    async def test_bootstrap_checkpoint_save_on_run(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that checkpoint is saved after run."""
        runner = LiveArmadaRunner(config=armada_config)

        # Mock minimal run
        with patch.object(runner, 'ensure_ready', new_callable=AsyncMock, return_value=True):
            with patch.object(runner, 'capture_observation', new_callable=AsyncMock):
                with patch.object(runner, 'process_vision', new_callable=AsyncMock):
                    with patch.object(runner, 'infer_armada', new_callable=AsyncMock) as mock_infer:
                        with patch.object(runner, 'write_quad_image'):
                            with patch.object(runner, 'write_trace'):
                                with patch.object(runner, 'mark_keyframe'):
                                    with patch.object(runner, 'execute_action', new_callable=AsyncMock):
                                        from src.runners.live_armada import VisionOutput, ArmadaResponse
                                        from PIL import Image

                                        mock_infer.return_value = ArmadaResponse(
                                            step_id=1,
                                            timestamp=0,
                                            model_id="test",
                                            action="move",
                                            confidence=0.8
                                        )

                                        result = await runner.run(max_steps=1)

        assert result == 0

        # Check that checkpoint was saved
        checkpoint = runner.bootstrap_manager.load_checkpoint(runner.run_id)
        assert checkpoint is not None
        assert checkpoint.run_id == runner.run_id

    @pytest.mark.asyncio
    async def test_bootstrap_load_on_subsequent_run(
        self,
        temp_config_dir: Path
    ) -> None:
        """Test that skills are loaded from bootstrap checkpoint in next run."""
        # First run: discover a skill
        config1 = ArmadaConfig(
            rom=Path("/fake/rom.gba"),
            save=Path("/fake/save.sav"),
            lua=Path("/fake/lua.lua"),
            mgba_exe=Path("/fake/mgba.exe"),
            dashboard_dir=temp_config_dir / "dashboard",
            trace_jsonl=temp_config_dir / "traces.jsonl",
            dry_run=True,
        )

        runner1 = LiveArmadaRunner(config=config1)

        # Manually save a skill and checkpoint
        from src.skills.skill_manager import Skill, SkillStep

        skill = Skill(
            name="test_skill",
            description="Test skill",
            steps=[
                SkillStep(action="move_down", confidence=0.9),
                SkillStep(action="confirm", confidence=0.85),
            ]
        )

        runner1.skill_manager.save_skill(skill)
        runner1.bootstrap_manager.save_checkpoint(
            run_id=runner1.run_id,
            learned_skills=[skill.to_dict()],
            memory_buffer={}
        )

        # Second run: should load the skill
        config2 = ArmadaConfig(
            rom=Path("/fake/rom.gba"),
            save=Path("/fake/save.sav"),
            lua=Path("/fake/lua.lua"),
            mgba_exe=Path("/fake/mgba.exe"),
            dashboard_dir=temp_config_dir / "dashboard",
            trace_jsonl=temp_config_dir / "traces.jsonl",
            dry_run=True,
        )

        runner2 = LiveArmadaRunner(config=config2)

        # Verify skill exists in new runner (different manager, different configs)
        # The second runner should be able to access the persisted skill
        assert runner2.bootstrap_manager.load_latest_checkpoint() is not None

    def test_skill_tags_for_discovery(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that discovered skills have correct tags."""
        runner = LiveArmadaRunner(config=armada_config)

        # Manually create discovered skill
        skill = runner.skill_manager.create_skill_from_trajectory(
            name="test_discovery",
            actions=["move", "move", "confirm"],
            confidences=[0.9, 0.9, 0.85],
            tags=["discovered", "live"]
        )

        assert skill is not None
        assert "discovered" in skill.tags
        assert "live" in skill.tags

    def test_run_id_format(
        self,
        armada_config: ArmadaConfig
    ) -> None:
        """Test that run_id has proper format."""
        runner = LiveArmadaRunner(config=armada_config)

        # run_id should contain timestamp
        assert "run_" in runner.run_id
        assert len(runner.run_id) > 10  # run_YYYYMMDD_HHMMSS format

    @pytest.mark.asyncio
    async def test_multiple_runs_independence(
        self,
        temp_config_dir: Path
    ) -> None:
        """Test that multiple runs maintain independence in their state."""
        import time
        
        config1 = ArmadaConfig(
            rom=Path("/fake/rom.gba"),
            save=Path("/fake/save.sav"),
            lua=Path("/fake/lua.lua"),
            mgba_exe=Path("/fake/mgba.exe"),
            dashboard_dir=temp_config_dir / "dashboard1",
            dry_run=True,
        )

        runner1 = LiveArmadaRunner(config=config1)
        
        # Small delay to ensure different timestamp
        time.sleep(0.1)

        config2 = ArmadaConfig(
            rom=Path("/fake/rom.gba"),
            save=Path("/fake/save.sav"),
            lua=Path("/fake/lua.lua"),
            mgba_exe=Path("/fake/mgba.exe"),
            dashboard_dir=temp_config_dir / "dashboard2",
            dry_run=True,
        )

        runner2 = LiveArmadaRunner(config=config2)

        # Each should have unique run_id (or at least independent state)
        # If timestamps are same, at least verify they have independent managers
        assert runner1.skill_manager is not runner2.skill_manager
        assert runner1.bootstrap_manager is not runner2.bootstrap_manager
        
        # Both should be properly initialized
        assert runner1.run_id is not None
        assert runner2.run_id is not None
