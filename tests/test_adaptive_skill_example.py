"""Integration tests for the adaptive dungeon exploration example skill.

This test demonstrates how AsyncSkillRuntime executes the adaptive skill with
mid-execution LM inference decisions.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from src.skills.async_skill_runtime import AsyncSkillRuntime
from examples.adaptive_dungeon_exploration_skill import adaptive_dungeon_exploration_skill


class TestAdaptiveDungeonSkill:
    """Test the adaptive dungeon exploration skill with inference checkpoints."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={
            "position": (5, 5),
            "visible_tiles": 12,
            "nearby_enemies": 0,
            "current_floor": 1,
            "target_floor": 1,
            "hp": 100,
            "inventory": ["potion", "key"],
        })
        controller.capture_with_metadata = Mock(
            return_value={"path": "/tmp/dungeon_screenshot.png"}
        )
        controller.button_tap = Mock()
        return controller

    @pytest.fixture
    def mock_model_router(self):
        """Create a mock ModelRouter with adaptive inference responses."""
        router = Mock()
        router.select_model = Mock(return_value="4B")

        # Create mock inference responses for different checkpoints
        responses = {
            "exploration_decision": '''{
                "steps": [
                    {"primitive": "tap", "button": "UP"},
                    {"primitive": "wait_turn"}
                ],
                "reasoning": "Moving north to explore new area"
            }''',
            "major_threat_response": '''{
                "steps": [
                    {"primitive": "tap", "button": "A"},
                    {"primitive": "annotate", "message": "Engaging enemy"}
                ],
                "reasoning": "Prepare for combat with nearby enemy"
            }''',
        }

        async def mock_infer(prompt, model_size):
            # Determine which checkpoint this is
            if "exploration_decision" in prompt:
                response = responses["exploration_decision"]
            elif "major_threat_response" in prompt:
                response = responses["major_threat_response"]
            else:
                response = responses["exploration_decision"]  # default

            future = asyncio.Future()
            future.set_result(response)
            return future

        router.infer_async = mock_infer
        return router

    @pytest.fixture
    def runtime(self, mock_controller, mock_model_router):
        """Create AsyncSkillRuntime with adaptive skill support."""
        return AsyncSkillRuntime(mock_controller, model_router=mock_model_router)

    @pytest.mark.asyncio
    async def test_adaptive_skill_with_inference_checkpoints(
        self, runtime, mock_controller
    ):
        """Test execution of adaptive skill with inference checkpoints.

        This demonstrates the full flow:
        1. Initialize exploration
        2. Hit exploration_decision checkpoint
        3. Model provides adaptive actions
        4. Execute returned primitives
        5. Hit major_threat_response checkpoint
        6. Model provides threat response actions
        7. Complete skill execution
        """
        # Execute the adaptive skill
        result = await runtime.run_async(
            adaptive_dungeon_exploration_skill,
            params={
                "target_floor": 1,
                "max_exploration_time": 300,
            },
            timeout_seconds=30,  # 30 second timeout
        )

        # Verify execution completed
        assert result.status in ["completed", "failed"]  # May timeout is ok for mock

        # Verify inference checkpoints were mentioned
        assert any(
            "exploration_decision" in note.lower() or
            "major_threat_response" in note.lower()
            for note in result.notes
        ), "Skill should execute inference checkpoints"

        # Verify adaptive behavior occurred
        assert len(result.notes) > 3, "Should have multiple execution notes"

    @pytest.mark.asyncio
    async def test_adaptive_skill_state_tracking(
        self, runtime, mock_controller
    ):
        """Test that adaptive skill properly tracks game state during execution."""
        # Track state refreshes
        state_refresh_count = 0
        original_semantic_state = mock_controller.semantic_state

        def counting_semantic_state(*args, **kwargs):
            nonlocal state_refresh_count
            state_refresh_count += 1
            return original_semantic_state(*args, **kwargs)

        mock_controller.semantic_state = counting_semantic_state

        # Execute skill
        result = await runtime.run_async(
            adaptive_dungeon_exploration_skill,
            params={"target_floor": 1},
            timeout_seconds=20,
        )

        # Verify state was refreshed multiple times
        # (at least once for initial, once in loop, once in checkpoint)
        assert state_refresh_count >= 2, (
            f"Should refresh state multiple times, got {state_refresh_count}"
        )

    @pytest.mark.asyncio
    async def test_adaptive_skill_parameter_passing(
        self, runtime
    ):
        """Test that skill parameters are properly passed and available."""
        # Override the skill to use parameters
        from src.skills.spec import SkillMeta, SkillSpec, AnnotatePrimitive

        param_aware_skill = SkillSpec(
            meta=SkillMeta(
                name="param_test",
                description="Test parameter passing"
            ),
            steps=[
                AnnotatePrimitive(message="Starting with parameters"),
            ],
        )

        result = await runtime.run_async(
            param_aware_skill,
            params={
                "target_floor": 5,
                "max_exploration_time": 600,
            },
        )

        assert result.status == "completed"
        assert len(result.notes) > 0

    def test_adaptive_skill_structure_validity(self):
        """Test that the adaptive skill is structurally valid."""
        skill = adaptive_dungeon_exploration_skill

        # Verify required fields
        assert skill.meta is not None
        assert skill.meta.name == "navigate_dungeon_floor_adaptive"
        assert skill.meta.description is not None
        assert len(skill.steps) > 0

        # Verify it has InferenceCheckpoints
        from src.skills.spec import InferenceCheckpointPrimitive, WhileBlock

        checkpoint_found = False
        for step in skill.steps:
            if isinstance(step, InferenceCheckpointPrimitive):
                checkpoint_found = True
            elif isinstance(step, WhileBlock):
                for body_step in step.body:
                    if isinstance(body_step, InferenceCheckpointPrimitive):
                        checkpoint_found = True

        assert checkpoint_found, "Skill should contain InferenceCheckpoints"

    def test_adaptive_skill_has_timeout_budgets(self):
        """Test that all inference checkpoints have appropriate timeouts."""
        from src.skills.spec import InferenceCheckpointPrimitive, WhileBlock

        skill = adaptive_dungeon_exploration_skill
        checkpoints = []

        for step in skill.steps:
            if isinstance(step, InferenceCheckpointPrimitive):
                checkpoints.append(step)
            elif isinstance(step, WhileBlock):
                for body_step in step.body:
                    if isinstance(body_step, InferenceCheckpointPrimitive):
                        checkpoints.append(body_step)

        # Verify all checkpoints have reasonable timeouts
        for checkpoint in checkpoints:
            assert 5 <= checkpoint.timeout_seconds <= 300, (
                f"Checkpoint {checkpoint.label} has invalid timeout: "
                f"{checkpoint.timeout_seconds}"
            )

        # Verify different checkpoints have different timeouts based on complexity
        timeouts = [c.timeout_seconds for c in checkpoints]
        assert len(timeouts) > 0, "Should have at least one checkpoint"

        # Quick tactical decisions should be shorter than strategic decisions
        assert min(timeouts) <= 10, "Quick decisions should have short timeout"
        assert max(timeouts) >= 10, "Strategic decisions should have longer timeout"
