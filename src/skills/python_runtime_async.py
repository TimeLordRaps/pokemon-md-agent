"""Runtime for executing Python-based SkillSpec objects against the environment."""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from .spec import (
    SkillSpec,
    Step,
    TapPrimitive,
    HoldPrimitive,
    ReleasePrimitive,
    WaitTurnPrimitive,
    CapturePrimitive,
    RefreshStatePrimitive,
    ExpectPrimitive,
    AnnotatePrimitive,
    BreakPrimitive,
    AbortPrimitive,
    SuccessPrimitive,
    CallPrimitive,
    CheckpointPrimitive,
    ResumePrimitive,
    SaveStateCheckpointPrimitive,
    LoadStateCheckpointPrimitive,
    InferenceCheckpointPrimitive,
    IfBlock,
    WhileBlock,
    Primitive,
)
from ..environment.mgba_controller import MGBAController

logger = logging.getLogger(__name__)


@dataclass
class SkillExecutionResult:
    """Return value from a skill execution."""

    status: str
    notes: List[str] = field(default_factory=list)
    frames: List[str] = field(default_factory=list)
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)


class PrimitiveExecutor:
    """Thin adapter between primitives and the mgba-http controller."""

    def __init__(self, controller: MGBAController):
        self._c = controller

    def tap(self, button: str, repeat: int = 1) -> None:
        for _ in range(repeat):
            self._c.button_tap(button)

    def hold(self, button: str, frames: int) -> None:
        duration_ms = int(frames * 1000 / 60)
        self._c.button_hold(button, duration_ms)

    def release(self, button: str) -> None:
        self._c.button_clear_many([button])

    def wait_turn(self) -> None:
        self._c.button_tap("A")
        self._c.button_tap("B")

    def capture(self, label: str) -> str:
        metadata = self._c.capture_with_metadata()
        path = metadata.get("path")
        if not path:
            raise AbortSignal("Failed to capture screenshot")
        return str(path)

    def refresh_state(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._c.semantic_state(fields=fields)

    def save_state_snapshot(self, label: str) -> Dict[str, Any]:
        snapshot = self._c.semantic_state()
        snapshot["_snapshot_label"] = label
        return snapshot


class PythonSkillRuntime:
    """Execute SkillSpec definitions using the PrimitiveExecutor."""

    def __init__(
        self,
        controller: MGBAController,
        skill_lookup: Optional[Callable[[str], SkillSpec]] = None,
    ):
        self._controller = controller
        self._exec = PrimitiveExecutor(controller)
        self._skill_lookup = skill_lookup

    def run(self, spec: SkillSpec, params: Optional[Dict[str, Any]] = None) -> SkillExecutionResult:
        """Execute the skill and return telemetry."""
        ctx = {
            "params": params or {},
            "notes": [],
            "frames": [],
            "snapshots": [],
            "status": "indeterminate",
        }

        try:
            await self._execute_steps(spec.steps, ctx)
        except BreakSignal:
            ctx["notes"].append("skill interrupted by break()")
        except AbortSignal as exc:
            ctx["status"] = "failed"
            ctx["notes"].append(exc.reason)
        else:
            if ctx["status"] == "indeterminate":
            ctx["status"] = "completed"

        return SkillExecutionResult(
            status=ctx["status"],
            notes=list(ctx["notes"]),
            frames=list(ctx["frames"]),
            state_snapshots=list(ctx["snapshots"]),
        )

    def _execute_steps(self, steps: List[Step], ctx: Dict[str, Any]) -> None:
        for node in steps:
            if isinstance(node, IfBlock):
                self._handle_if(node, ctx)
            elif isinstance(node, WhileBlock):
                self._handle_while(node, ctx)
            else:
                self._execute_primitive(node, ctx)

    def _handle_if(self, block: IfBlock, ctx: Dict[str, Any]) -> None:
        state = self._exec.refresh_state()
        if self._evaluate_condition(block.condition, state, ctx):
            self._execute_steps(block.then, ctx)
        elif block.otherwise:
            self._execute_steps(block.otherwise, ctx)

    def _handle_while(self, block: WhileBlock, ctx: Dict[str, Any]) -> None:
        iterations = 0
        while iterations < block.max_iterations:
            state = self._exec.refresh_state()
            if not self._evaluate_condition(block.condition, state, ctx):
                break
            self._execute_steps(block.body, ctx)
            iterations += 1

    def _execute_primitive(self, node: Primitive, ctx: Dict[str, Any]) -> None:
        if isinstance(node, TapPrimitive):
            self._exec.tap(node.button.value, node.repeat)
        elif isinstance(node, HoldPrimitive):
            self._exec.hold(node.button.value, node.frames)
        elif isinstance(node, ReleasePrimitive):
            self._exec.release(node.button.value)
        elif isinstance(node, WaitTurnPrimitive):
            self._exec.wait_turn()
        elif isinstance(node, CapturePrimitive):
            frame_path = self._exec.capture(node.label)
            ctx["frames"].append(frame_path)
        elif isinstance(node, RefreshStatePrimitive):
            snapshot = self._exec.refresh_state(node.fields)
            ctx["snapshots"].append(snapshot)
        elif isinstance(node, ExpectPrimitive):
            state = self._exec.refresh_state()
            ok = self._evaluate_condition(node.expectation, state, ctx)
            if not ok:
                message = f"Expectation failed: {node.expectation}"
                ctx["notes"].append(message)
                if node.severity == "fail":
                    raise AbortSignal(message)
        elif isinstance(node, AnnotatePrimitive):
            ctx["notes"].append(node.message)
        elif isinstance(node, BreakPrimitive):
            raise BreakSignal()
        elif isinstance(node, AbortPrimitive):
            raise AbortSignal(node.reason)
        elif isinstance(node, SuccessPrimitive):
            ctx["status"] = "succeeded"
            ctx["notes"].append(node.summary)
            raise BreakSignal()
        elif isinstance(node, CallPrimitive):
            nested = self._resolve_skill(node.skill)
            self._execute_steps(nested.steps, ctx)
        else:
            logger.warning("Unhandled primitive type: %s", node)

    def _resolve_skill(self, name: str) -> SkillSpec:
        if self._skill_lookup is None:
            raise AbortSignal(f"Unknown skill '{name}' (no lookup configured)")
        spec = self._skill_lookup(name)
        if not isinstance(spec, SkillSpec):
            raise AbortSignal(f"Skill lookup did not return SkillSpec for '{name}'")
        return spec

    def _evaluate_condition(self, expression: str, state: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        """Very small expression evaluator usable by LM generated code."""
        local_vars = {
            "state": state,
            "params": ctx.get("params", {}),
            "notes": ctx.get("notes", []),
        }
        try:
            return bool(eval(expression, {"__builtins__": {}}, local_vars))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Condition eval error for '%s': %s", expression, exc)
            return False


class BreakSignal(RuntimeError):
    """Raised to exit the current block early."""


class AbortSignal(RuntimeError):
    """Raised when a skill decides to abort execution."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
