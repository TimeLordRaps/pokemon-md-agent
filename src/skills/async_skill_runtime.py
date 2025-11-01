"""Async runtime for executing skills with mid-execution model inference.

This module provides AsyncSkillRuntime for supporting InferenceCheckpoint primitives,
which allow skills to pause execution and query an LM for adaptive next steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable

from .spec import (
    SkillSpec,
    InferenceCheckpointPrimitive,
    Primitive,
    TapPrimitive,
    HoldPrimitive,
    ReleasePrimitive,
    WaitTurnPrimitive,
    CapturePrimitive,
    RefreshStatePrimitive,
    AnnotatePrimitive,
    AbortPrimitive,
    SuccessPrimitive,
    CallPrimitive,
)
from .python_runtime import (
    PythonSkillRuntime,
    SkillExecutionResult,
    AbortSignal,
)

logger = logging.getLogger(__name__)

# Type stub for ModelRouter (avoid circular imports)
try:
    from ..agent.model_router import ModelRouter, ModelSize
except ImportError:
    ModelRouter = None
    ModelSize = None


class AsyncSkillRuntime(PythonSkillRuntime):
    """Async runtime for skills with model inference support.

    Extends PythonSkillRuntime to support InferenceCheckpoint primitives.
    These primitives pause skill execution and query a language model for
    adaptive decision-making and next steps.

    Features:
    - Async execution of skills with model inference
    - Mid-skill LM calls for adaptive behavior
    - Time-budgeted inference with deadline awareness
    - Graceful fallback on inference timeouts
    """

    def __init__(
        self,
        controller,
        skill_lookup: Optional[Callable[[str], SkillSpec]] = None,
        save_manager=None,
        model_router: Optional[ModelRouter] = None,
    ):
        """Initialize AsyncSkillRuntime.

        Args:
            controller: MGBAController instance.
            skill_lookup: Function to resolve skill names to SkillSpec.
            save_manager: SaveManager for game state persistence.
            model_router: ModelRouter for LM inference calls.
        """
        super().__init__(controller, skill_lookup, save_manager)
        self.model_router = model_router

    async def run_async(
        self,
        spec: SkillSpec,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> SkillExecutionResult:
        """Execute skill asynchronously with model inference support.

        Args:
            spec: SkillSpec to execute.
            params: Runtime parameters for the skill.
            timeout_seconds: Overall execution timeout.

        Returns:
            SkillExecutionResult with status, notes, frames, and snapshots.
        """
        ctx = {
            "params": params or {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
            "_start_time": time.time(),
            "_timeout_seconds": timeout_seconds,
        }

        try:
            await self._execute_steps_async(spec.steps, ctx)
        except AbortSignal as exc:
            ctx["status"] = "failed"
            ctx["notes"].append(str(exc.reason))
            logger.error("Skill execution failed: %s", exc.reason)
        except asyncio.TimeoutError:
            ctx["status"] = "failed"
            ctx["notes"].append("Skill execution timeout")
            logger.error("Skill execution exceeded timeout")
        except Exception as exc:
            ctx["status"] = "failed"
            ctx["notes"].append(f"Unexpected error: {exc}")
            logger.error("Unexpected error during skill execution: %s", exc)
        else:
            if ctx["status"] == "indeterminate":
                ctx["status"] = "completed"

        return SkillExecutionResult(
            status=ctx["status"],
            notes=list(ctx["notes"]),
            frames=list(ctx["frames"]),
            state_snapshots=list(ctx["snapshots"]),
        )

    async def _execute_steps_async(
        self, steps: List[Primitive], ctx: Dict[str, Any]
    ) -> None:
        """Execute a list of steps asynchronously.

        Args:
            steps: List of primitives/blocks to execute.
            ctx: Execution context.

        Raises:
            AbortSignal: If execution should be aborted.
            asyncio.TimeoutError: If overall timeout exceeded.
        """
        for node in steps:
            # Check timeout
            if ctx.get("_timeout_seconds"):
                elapsed = time.time() - ctx["_start_time"]
                if elapsed > ctx["_timeout_seconds"]:
                    raise asyncio.TimeoutError()

            # Handle inference checkpoint specially
            if isinstance(node, InferenceCheckpointPrimitive):
                await self._handle_inference_checkpoint(node, ctx)
            else:
                # Delegate to sync execution (most primitives don't need async)
                self._execute_primitive(node, ctx)

    async def _handle_inference_checkpoint(
        self, primitive: InferenceCheckpointPrimitive, ctx: Dict[str, Any]
    ) -> None:
        """Handle InferenceCheckpoint primitive with async model call.

        Pauses execution, captures game state, queries the model for
        adaptive next steps, then resumes execution with returned steps.

        Args:
            primitive: InferenceCheckpointPrimitive to execute.
            ctx: Execution context.

        Raises:
            AbortSignal: If ModelRouter not configured or inference fails.
        """
        label = primitive.label
        context = primitive.context
        timeout = primitive.timeout_seconds

        if not self.model_router:
            raise AbortSignal(
                f"Cannot execute inference checkpoint '{label}': "
                "ModelRouter not configured"
            )

        # Capture current game state
        try:
            state = self._exec.refresh_state()
            screenshot_path = self._exec.capture(f"inference_{label}")
        except Exception as exc:
            raise AbortSignal(
                f"Failed to capture game state for inference checkpoint '{label}': {exc}"
            )

        # Build inference prompt with context
        prompt = self._build_inference_prompt(label, context, state, screenshot_path)

        # Call model asynchronously with timeout
        try:
            logger.info(
                "Executing inference checkpoint '%s' with timeout %ds",
                label,
                timeout,
            )

            # Select appropriate model based on remaining time budget
            remaining_budget = self._get_remaining_budget(ctx, timeout)
            try:
                selected_model = self.model_router.select_model(
                    remaining_budget_s=remaining_budget
                )
            except Exception as exc:
                # Fallback to 2B model if selection fails
                logger.warning(
                    "Model selection failed for checkpoint '%s': %s, using 2B fallback",
                    label,
                    exc,
                )
                selected_model = ModelSize.SIZE_2B

            # Call inference async with timeout
            future = self.model_router.infer_async(prompt, selected_model)
            inference_result = await asyncio.wait_for(future, timeout=timeout)

        except asyncio.TimeoutError:
            ctx["notes"].append(
                f"Inference checkpoint '{label}' timeout ({timeout}s), continuing"
            )
            logger.warning("Inference checkpoint '%s' timed out", label)
            return  # Continue without next steps
        except Exception as exc:
            ctx["notes"].append(
                f"Inference checkpoint '{label}' failed: {exc}, continuing"
            )
            logger.error("Inference checkpoint '%s' error: %s", label, exc)
            return  # Continue without next steps

        # Parse inference result for next steps
        ctx["notes"].append(f"Inference checkpoint '{label}' completed")
        try:
            next_steps = self._parse_inference_response(inference_result, label)
            if next_steps:
                ctx["notes"].append(f"Executing {len(next_steps)} steps from inference")
                logger.info(
                    "Executing %d inferred steps from checkpoint '%s'",
                    len(next_steps),
                    label,
                )
                await self._execute_steps_async(next_steps, ctx)
        except Exception as exc:
            logger.warning(
                "Failed to parse/execute inference response for checkpoint '%s': %s",
                label,
                exc,
            )

    def _build_inference_prompt(
        self,
        label: str,
        context: str,
        state: Dict[str, Any],
        screenshot_path: str,
    ) -> str:
        """Build prompt for model inference.

        Args:
            label: Checkpoint label for logging.
            context: Skill intent and checkpoint purpose.
            state: Current semantic game state.
            screenshot_path: Path to captured screenshot.

        Returns:
            Formatted prompt for the model.
        """
        state_str = json.dumps(state, indent=2)
        prompt = f"""You are an AI agent controlling a Pokemon Mystery Dungeon game.

Checkpoint: {label}
Context: {context}

Current Game State:
{state_str}

Screenshot: {screenshot_path}

What are the next 1-3 actions the agent should take?
Respond in JSON format with a "steps" array of primitive objects.
Each step should have "primitive" and relevant fields.

Example format:
{{
  "steps": [
    {{"primitive": "tap", "button": "A", "repeat": 1}},
    {{"primitive": "wait_turn"}}
  ],
  "reasoning": "Brief explanation of why these steps"
}}"""
        return prompt

    def _parse_inference_response(
        self, response: str, checkpoint_label: str
    ) -> Optional[List[Primitive]]:
        """Parse model response into executable primitives.

        Args:
            response: Model's text response.
            checkpoint_label: Checkpoint label for error messages.

        Returns:
            List of Primitive objects to execute, or None if parsing fails.

        Raises:
            ValueError: If response format is invalid.
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                logger.warning(
                    "No JSON found in inference response for checkpoint '%s'",
                    checkpoint_label,
                )
                return None

            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            if not isinstance(parsed, dict):
                logger.warning(
                    "Inference response root not a dict for checkpoint '%s'",
                    checkpoint_label,
                )
                return None

            steps_data = parsed.get("steps", [])
            if not isinstance(steps_data, list):
                logger.warning(
                    "Inference response 'steps' not a list for checkpoint '%s'",
                    checkpoint_label,
                )
                return None

            # Mapping of primitive type names to their classes
            primitive_map = {
                "tap": TapPrimitive,
                "hold": HoldPrimitive,
                "release": ReleasePrimitive,
                "wait_turn": WaitTurnPrimitive,
                "capture": CapturePrimitive,
                "refresh_state": RefreshStatePrimitive,
                "annotate": AnnotatePrimitive,
                "abort": AbortPrimitive,
                "success": SuccessPrimitive,
                "call": CallPrimitive,
            }

            # Deserialize steps
            primitives: List[Primitive] = []
            for i, step_data in enumerate(steps_data):
                if not isinstance(step_data, dict):
                    logger.warning(
                        "Step %d in checkpoint '%s' is not a dict, skipping",
                        i,
                        checkpoint_label,
                    )
                    continue

                primitive_type = step_data.get("primitive")
                if not primitive_type or primitive_type not in primitive_map:
                    logger.warning(
                        "Unknown or missing primitive type '%s' in step %d of checkpoint '%s', skipping",
                        primitive_type,
                        i,
                        checkpoint_label,
                    )
                    continue

                try:
                    # Create primitive instance with the step data
                    PrimitiveClass = primitive_map[primitive_type]
                    primitive = PrimitiveClass(**step_data)
                    primitives.append(primitive)
                    logger.debug(
                        "Deserialized primitive '%s' from checkpoint '%s'",
                        primitive_type,
                        checkpoint_label,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to deserialize primitive '%s' in step %d of checkpoint '%s': %s",
                        primitive_type,
                        i,
                        checkpoint_label,
                        exc,
                    )
                    continue

            logger.info(
                "Parsed %d inferred steps from checkpoint '%s' (deserialized %d primitives)",
                len(steps_data),
                checkpoint_label,
                len(primitives),
            )
            return primitives if primitives else None

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse JSON from inference response for checkpoint '%s': %s",
                checkpoint_label,
                exc,
            )
            return None

    def _get_remaining_budget(
        self, ctx: Dict[str, Any], inference_timeout: int
    ) -> float:
        """Calculate remaining time budget for model selection.

        Args:
            ctx: Execution context with timing info.
            inference_timeout: Timeout for this inference call.

        Returns:
            Remaining time budget in seconds.
        """
        if not ctx.get("_timeout_seconds"):
            return float(inference_timeout)  # Use inference timeout as budget

        elapsed = time.time() - ctx["_start_time"]
        total_budget = ctx["_timeout_seconds"]
        remaining = total_budget - elapsed

        # Leave some buffer (10% of inference timeout)
        buffer = inference_timeout * 0.1
        safe_budget = max(remaining - buffer, 0)

        logger.debug(
            "Time budget: total=%fs, elapsed=%fs, remaining=%fs",
            total_budget,
            elapsed,
            safe_budget,
        )

        return safe_budget
