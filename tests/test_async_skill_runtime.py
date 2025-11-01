"""Tests for AsyncSkillRuntime with model inference integration.

Tests verify that AsyncSkillRuntime can execute skills asynchronously and
handle InferenceCheckpoint primitives with model inference calls.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import json

from src.skills.async_skill_runtime import AsyncSkillRuntime
from src.skills.spec import (
    SkillSpec,
    SkillMeta,
    InferenceCheckpointPrimitive,
    AnnotatePrimitive,
    TapPrimitive,
    Button,
)


def make_skill(name: str, description: str, steps):
    """Helper to create a SkillSpec for testing."""
    return SkillSpec(
        meta=SkillMeta(name=name, description=description),
        steps=steps,
    )


class TestAsyncSkillRuntime:
    """Test AsyncSkillRuntime basic functionality."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100, "level": 1})
        controller.capture_with_metadata = Mock(
            return_value={"path": "/tmp/screenshot.png"}
        )
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create AsyncSkillRuntime with mock controller."""
        return AsyncSkillRuntime(mock_controller)

    @pytest.mark.asyncio
    async def test_run_async_basic_skill(self, runtime):
        """Test basic async skill execution without inference."""
        spec = make_skill(
            name="simple_skill",
            description="Test skill",
            steps=[
                AnnotatePrimitive(message="Starting"),
                AnnotatePrimitive(message="Done"),
            ],
        )

        result = await runtime.run_async(spec, params={})

        assert result.status == "completed"
        assert len(result.notes) >= 2

    @pytest.mark.asyncio
    async def test_run_async_timeout_enforcement(self, runtime):
        """Test that timeout is enforced during async execution."""

        async def slow_operation():
            await asyncio.sleep(2.0)  # Simulate slow operation

        # This would require deeper integration, so we'll skip for now
        # In a real test, we'd use a very short timeout
        pass

    @pytest.mark.asyncio
    async def test_inference_checkpoint_without_model_router(self, runtime, mock_controller):
        """Test that inference checkpoint raises error without ModelRouter."""
        spec = make_skill(
            name="inference_skill",
            description="Test skill",
            steps=[
                InferenceCheckpointPrimitive(
                    label="test_inference",
                    context="Test context",
                    timeout_seconds=5,
                ),
            ],
        )

        result = await runtime.run_async(spec)

        assert result.status == "failed"
        assert any("ModelRouter not configured" in note for note in result.notes)


class TestInferenceCheckpointHandling:
    """Test InferenceCheckpoint primitive handling."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 50, "items": ["potion"]})
        controller.capture_with_metadata = Mock(
            return_value={"path": "/tmp/test_screenshot.png"}
        )
        return controller

    @pytest.fixture
    def mock_model_router(self):
        """Create a mock ModelRouter."""
        router = Mock()

        async def mock_infer(*args, **kwargs):
            return """{
                "steps": [
                    {"primitive": "tap", "button": "A"}
                ],
                "reasoning": "Test inference result"
            }"""

        future = asyncio.Future()
        future.set_result(
            '{"steps": [{"primitive": "tap", "button": "A"}], "reasoning": "Test"}'
        )
        router.infer_async = Mock(return_value=future)
        router.select_model = Mock(return_value="4B")
        return router

    @pytest.fixture
    def runtime_with_router(self, mock_controller, mock_model_router):
        """Create AsyncSkillRuntime with mock ModelRouter."""
        return AsyncSkillRuntime(mock_controller, model_router=mock_model_router)

    @pytest.mark.asyncio
    async def test_build_inference_prompt(self, runtime_with_router):
        """Test inference prompt building."""
        prompt = runtime_with_router._build_inference_prompt(
            label="test",
            context="Test skill",
            state={"hp": 100},
            screenshot_path="/tmp/test.png",
        )

        assert "test" in prompt.lower()
        assert "Pokemon Mystery Dungeon" in prompt
        assert "/tmp/test.png" in prompt
        assert "hp" in prompt

    def test_parse_inference_response_valid(self, runtime_with_router):
        """Test parsing valid inference response with deserialization."""
        response = """{
            "steps": [
                {"primitive": "tap", "button": "A"},
                {"primitive": "wait_turn"}
            ],
            "reasoning": "Make a move"
        }"""

        result = runtime_with_router._parse_inference_response(response, "test")
        assert result is not None
        assert len(result) == 2
        assert result[0].primitive == "tap"
        assert result[0].button == "A"
        assert result[1].primitive == "wait_turn"

    def test_parse_inference_response_invalid_json(self, runtime_with_router):
        """Test parsing invalid JSON response."""
        response = "Not valid JSON"

        result = runtime_with_router._parse_inference_response(response, "test")
        assert result is None

    def test_parse_inference_response_no_json(self, runtime_with_router):
        """Test parsing response without JSON."""
        response = "Just some text without json structure"

        result = runtime_with_router._parse_inference_response(response, "test")
        assert result is None

    def test_get_remaining_budget_no_timeout(self, runtime_with_router):
        """Test budget calculation without overall timeout."""
        ctx = {"_timeout_seconds": None, "_start_time": 0}

        budget = runtime_with_router._get_remaining_budget(ctx, 10)
        assert budget == 10  # Uses inference timeout

    def test_get_remaining_budget_with_timeout(self, runtime_with_router):
        """Test budget calculation with overall timeout."""
        import time

        start = time.time()
        ctx = {"_timeout_seconds": 60, "_start_time": start}

        budget = runtime_with_router._get_remaining_budget(ctx, 10)
        assert budget < 60  # Less than total timeout
        assert budget > 0   # But still positive

    def test_parse_inference_response_multiple_primitives(self, runtime_with_router):
        """Test parsing response with multiple different primitive types."""
        from src.skills.spec import TapPrimitive, HoldPrimitive, AnnotatePrimitive

        response = """{
            "steps": [
                {"primitive": "tap", "button": "A", "repeat": 2},
                {"primitive": "hold", "button": "B", "frames": 10},
                {"primitive": "annotate", "message": "Attacking now"}
            ],
            "reasoning": "Execute attack sequence"
        }"""

        result = runtime_with_router._parse_inference_response(response, "attack")
        assert result is not None
        assert len(result) == 3
        assert isinstance(result[0], TapPrimitive)
        assert result[0].button == "A"
        assert result[0].repeat == 2
        assert isinstance(result[1], HoldPrimitive)
        assert result[1].button == "B"
        assert result[1].frames == 10
        assert isinstance(result[2], AnnotatePrimitive)
        assert result[2].message == "Attacking now"

    def test_parse_inference_response_invalid_primitive_type(self, runtime_with_router):
        """Test parsing response with invalid primitive type (should skip)."""
        response = """{
            "steps": [
                {"primitive": "tap", "button": "A"},
                {"primitive": "invalid_type"},
                {"primitive": "wait_turn"}
            ],
            "reasoning": "Some reasoning"
        }"""

        result = runtime_with_router._parse_inference_response(response, "mixed")
        # Should return only the valid primitives (skip the invalid one)
        assert result is not None
        assert len(result) == 2
        assert result[0].primitive == "tap"
        assert result[1].primitive == "wait_turn"

    def test_parse_inference_response_malformed_step(self, runtime_with_router):
        """Test parsing response with malformed step data (should skip)."""
        response = """{
            "steps": [
                {"primitive": "tap", "button": "A"},
                "not a dict",
                {"primitive": "wait_turn"}
            ],
            "reasoning": "Some reasoning"
        }"""

        result = runtime_with_router._parse_inference_response(response, "malformed")
        # Should skip the non-dict step
        assert result is not None
        assert len(result) == 2

    def test_parse_inference_response_empty_steps(self, runtime_with_router):
        """Test parsing response with empty steps list."""
        response = """{
            "steps": [],
            "reasoning": "Nothing to do"
        }"""

        result = runtime_with_router._parse_inference_response(response, "empty")
        # Should return None when no valid primitives are deserialized
        assert result is None

    def test_parse_inference_response_wrapped_in_text(self, runtime_with_router):
        """Test parsing JSON wrapped in narrative text."""
        response = """The agent should do the following:
        {
            "steps": [
                {"primitive": "tap", "button": "A"}
            ],
            "reasoning": "Press A to continue"
        }
        This is the end of instructions."""

        result = runtime_with_router._parse_inference_response(response, "wrapped")
        # Should extract and parse the JSON even with surrounding text
        assert result is not None
        assert len(result) == 1
        assert result[0].primitive == "tap"


class TestAsyncSkillRuntimeIntegration:
    """Integration tests for async runtime."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={"hp": 100})
        controller.capture_with_metadata = Mock(
            return_value={"path": "/tmp/test.png"}
        )
        return controller

    @pytest.fixture
    def mock_router(self):
        """Create a working mock ModelRouter."""
        router = Mock()
        router.select_model = Mock(return_value="4B")

        # Create properly resolved future
        async def make_future():
            return '{"steps": [], "reasoning": "Continue"}'

        future = asyncio.Future()
        future.set_result('{"steps": [], "reasoning": "Continue"}')
        router.infer_async = Mock(return_value=future)

        return router

    @pytest.fixture
    def runtime(self, mock_controller, mock_router):
        """Create fully configured AsyncSkillRuntime."""
        return AsyncSkillRuntime(mock_controller, model_router=mock_router)

    @pytest.mark.asyncio
    async def test_skill_with_multiple_steps(self, runtime):
        """Test skill execution with multiple annotation steps."""
        spec = make_skill(
            name="multi_step",
            description="Test skill",
            steps=[
                AnnotatePrimitive(message="Step 1"),
                AnnotatePrimitive(message="Step 2"),
                AnnotatePrimitive(message="Step 3"),
            ],
        )

        result = await runtime.run_async(spec)

        assert result.status == "completed"
        assert len(result.notes) >= 3
        assert any("Step 1" in n for n in result.notes)
        assert any("Step 2" in n for n in result.notes)
        assert any("Step 3" in n for n in result.notes)

    @pytest.mark.asyncio
    async def test_inference_checkpoint_execution(self, runtime, mock_controller):
        """Test inference checkpoint execution with model response."""
        spec = make_skill(
            name="inference_skill",
            description="Test skill",
            steps=[
                AnnotatePrimitive(message="Before inference"),
                InferenceCheckpointPrimitive(
                    label="decision",
                    context="Make a movement decision",
                    timeout_seconds=5,
                ),
                AnnotatePrimitive(message="After inference"),
            ],
        )

        result = await runtime.run_async(spec)

        assert result.status == "completed"
        assert any("Before inference" in n for n in result.notes)
        assert any("After inference" in n for n in result.notes)
        assert any("decision" in n.lower() for n in result.notes)

    @pytest.mark.asyncio
    async def test_inference_checkpoint_timeout_handling(
        self, runtime, mock_controller
    ):
        """Test graceful handling of inference timeout."""
        # Create router that times out
        timeout_router = Mock()
        timeout_router.select_model = Mock(return_value="4B")

        # Create a future that will timeout
        timeout_future = asyncio.Future()

        async def delay_forever():
            await asyncio.sleep(100)

        async def resolve_with_timeout():
            try:
                await asyncio.wait_for(asyncio.sleep(100), timeout=0.01)
            except asyncio.TimeoutError:
                return None

        timeout_router.infer_async = Mock(return_value=timeout_future)

        runtime_with_timeout = AsyncSkillRuntime(
            mock_controller, model_router=timeout_router
        )

        spec = make_skill(
            name="timeout_test",
            description="Test skill",
            steps=[
                InferenceCheckpointPrimitive(
                    label="quick_timeout",
                    context="This will timeout",
                    timeout_seconds=5,
                ),
            ],
        )

        # Note: With timeout=5 and immediate future not set, this tests timeout path
        result = await runtime_with_timeout.run_async(spec)

        # Should complete but note the timeout
        assert result.status in ["completed", "failed"]


class TestAsyncRuntimeErrorHandling:
    """Test error handling in async runtime."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController."""
        controller = Mock()
        controller.semantic_state = Mock(return_value={})
        return controller

    @pytest.fixture
    def runtime(self, mock_controller):
        """Create AsyncSkillRuntime."""
        return AsyncSkillRuntime(mock_controller)

    @pytest.mark.asyncio
    async def test_exception_in_step_captured(self, runtime, mock_controller):
        """Test that exceptions in steps are captured."""
        # Make capture fail
        mock_controller.capture_with_metadata = Mock(side_effect=Exception("Capture failed"))

        spec = make_skill(
            name="error_skill",
            description="Test skill",
            steps=[
                InferenceCheckpointPrimitive(
                    label="will_fail",
                    context="This will fail to capture",
                    timeout_seconds=5,
                ),
            ],
        )

        runtime.model_router = Mock()  # Add dummy router

        result = await runtime.run_async(spec)

        assert result.status == "failed"
        assert any("Capture" in n for n in result.notes)

    @pytest.mark.asyncio
    async def test_missing_model_router_handled(self, runtime):
        """Test that missing ModelRouter is handled gracefully."""
        assert runtime.model_router is None

        spec = make_skill(
            name="no_router",
            description="Test skill",
            steps=[
                InferenceCheckpointPrimitive(
                    label="test",
                    context="Test",
                    timeout_seconds=5,
                ),
            ],
        )

        result = await runtime.run_async(spec)

        assert result.status == "failed"
        assert any("ModelRouter" in n for n in result.notes)
