"""Tests for skill trigger integration in agent core.

Skill triggers activate when belly < 30% or HP < 25%, coordinating runtime skill execution.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.agent.agent_core import PokemonMDAgent, AgentConfig
from src.environment.ram_decoders import PMDRedDecoder


class TestSkillTriggers:
    """Test skill trigger detection and coordination."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AgentConfig(
            enable_skill_triggers=True,
            skill_belly_threshold=0.3,  # 30%
            skill_hp_threshold=0.25,    # 25%
            skill_backoff_seconds=5.0
        )

        self.mock_controller = MagicMock()
        self.mock_decoder = MagicMock()

        with patch('src.agent.agent_core.MGBAController', return_value=self.mock_controller), \
             patch('src.agent.agent_core.PMDRedDecoder', return_value=self.mock_decoder), \
             patch('src.agent.agent_core.RAMWatcher'), \
             patch('src.agent.agent_core.ModelRouter'), \
             patch('src.agent.agent_core.MemoryManager'), \
             patch('src.agent.agent_core.StucknessDetector'), \
             patch('src.agent.agent_core.WorldModel'), \
             patch('src.agent.agent_core.TrajectoryLogger'), \
             patch('src.agent.agent_core.QuadCapture'), \
             patch('src.agent.agent_core.SaveManager'):

            self.agent = PokemonMDAgent(
                rom_path=Path("test.rom"),
                save_dir=Path("test_saves"),
                config=self.config
            )

    def test_belly_trigger_detection(self):
        # Mock party status with low belly
        party_status = {
            "leader": {
                "hp": 40, "hp_max": 50, "belly": 50, "status": 0  # belly=50, max=200 -> 25%
            },
            "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
        }

        self.mock_decoder.get_party_status.return_value = party_status

        # Should trigger (25% < 30%)
        assert self.agent._check_skill_triggers(party_status) is True

    def test_hp_trigger_detection(self):
        """Test that HP below threshold triggers skill execution."""
        party_status = {
            "leader": {
                "hp": 10, "hp_max": 50, "belly": 150, "status": 0  # hp=10/50=20% < 25%
            },
            "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
        }

        self.mock_decoder.get_party_status.return_value = party_status

        # Should trigger (20% < 25%)
        assert self.agent._check_skill_triggers(party_status) is True

    def test_no_trigger_when_healthy(self):
        """Test that no trigger occurs when health is good."""
        party_status = {
            "leader": {
                "hp": 40, "hp_max": 50, "belly": 150, "status": 0  # hp=80%, belly=75%
            },
            "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
        }

        self.mock_decoder.get_party_status.return_value = party_status

        # Should not trigger
        assert self.agent._check_skill_triggers(party_status) is False

    @pytest.mark.asyncio
    async def test_skill_execution_success(self):
        """Test successful skill execution and logging."""
        # Mock a skill
        mock_skill = MagicMock()
        mock_skill.name = "heal"
        mock_skill.priority = 1
        self.agent.skill_dsl.skills = {"heal": mock_skill}

        with patch.object(self.agent.skill_runtime, 'evaluate_triggers', return_value=True), \
             patch.object(self.agent.skill_runtime, 'evaluate_preconditions', return_value=True), \
             patch.object(self.agent.skill_runtime, 'execute_skill', return_value=True) as mock_execute:

            party_status = {
                "leader": {"hp": 10, "hp_max": 50, "belly": 50, "status": 0},
                "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
            }

            await self.agent._handle_skill_trigger(party_status)

            # Should have called execute_skill
            assert mock_execute.called

    @pytest.mark.asyncio
    async def test_skill_execution_failure_backoff(self):
        """Test failure handling with backoff."""
        # Mock a skill
        mock_skill = MagicMock()
        mock_skill.name = "heal"
        mock_skill.priority = 1
        self.agent.skill_dsl.skills = {"heal": mock_skill}

        with patch.object(self.agent.skill_runtime, 'evaluate_triggers', return_value=True), \
             patch.object(self.agent.skill_runtime, 'evaluate_preconditions', return_value=True), \
             patch.object(self.agent.skill_runtime, 'execute_skill', side_effect=Exception("Skill failed")) as mock_execute:

            party_status = {
                "leader": {"hp": 10, "hp_max": 50, "belly": 50, "status": 0},
                "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
            }

            await self.agent._handle_skill_trigger(party_status)

            # Should have called execute_skill
            assert mock_execute.called
            # Backoff timer should be set
            import time
            assert self.agent.skill_backoff_until > time.time()

    def test_backoff_prevents_trigger(self):
        """Test that backoff prevents repeated triggers."""
        import time
        # Set backoff
        self.agent.skill_backoff_until = time.time() + 10

        party_status = {
            "leader": {"hp": 10, "hp_max": 50, "belly": 50, "status": 0},
            "partner": {"hp": 40, "hp_max": 50, "belly": 100, "status": 0}
        }

        # Should not trigger during backoff
        assert self.agent._check_skill_triggers(party_status) is False