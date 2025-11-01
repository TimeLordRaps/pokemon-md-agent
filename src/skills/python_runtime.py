"""Runtime for executing Python-based SkillSpec objects against the environment."""

from __future__ import annotations

import logging
import time
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
    IfBlock,
    WhileBlock,
    Primitive,
)
from .checkpoint_state import CheckpointState
from ..environment.mgba_controller import MGBAController
from ..environment.save_manager import SaveManager

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
        save_manager: Optional[SaveManager] = None,
    ):
        self._controller = controller
        self._exec = PrimitiveExecutor(controller)
        self._skill_lookup = skill_lookup
        self._save_manager = save_manager
        self._checkpoints: Dict[str, CheckpointState] = {}

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
            self._execute_steps(spec.steps, ctx)
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
        elif isinstance(node, CheckpointPrimitive):
            self._handle_checkpoint(node, ctx)
        elif isinstance(node, ResumePrimitive):
            self._handle_resume(node, ctx)
        elif isinstance(node, SaveStateCheckpointPrimitive):
            self._handle_save_checkpoint(node, ctx)
        elif isinstance(node, LoadStateCheckpointPrimitive):
            self._handle_load_checkpoint(node, ctx)
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

    def _handle_checkpoint(self, primitive: CheckpointPrimitive, ctx: Dict[str, Any]) -> None:
        """Create a checkpoint of the current execution state.

        Args:
            primitive: CheckpointPrimitive with checkpoint label and optional description.
            ctx: Execution context to capture in the checkpoint.

        Raises:
            AbortSignal: If checkpoint state is invalid.
        """
        checkpoint_id = primitive.label

        # Create checkpoint state with current execution context
        checkpoint = CheckpointState(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            skill_name=ctx.get("_current_skill_name", "unknown"),
            execution_context={
                "state": self._exec.refresh_state(),
                "snapshots": ctx.get("snapshots", []).copy(),
                "params": ctx.get("params", {}),
            },
            parameters=ctx.get("params", {}),
            notes=ctx.get("notes", []).copy(),
            frames_captured=len(ctx.get("frames", [])),
            description=primitive.description,
        )

        # Validate checkpoint
        errors = checkpoint.validate()
        if errors:
            raise AbortSignal(
                f"Failed to create checkpoint '{checkpoint_id}': {'; '.join(errors)}"
            )

        # Store checkpoint in registry
        self._checkpoints[checkpoint_id] = checkpoint
        ctx["notes"].append(f"Checkpoint created: {checkpoint_id}")
        logger.info("Created checkpoint: %s", checkpoint_id)

    def _handle_resume(self, primitive: ResumePrimitive, ctx: Dict[str, Any]) -> None:
        """Resume execution from a previously created checkpoint.

        Args:
            primitive: ResumePrimitive with checkpoint label and optional fallback steps.
            ctx: Execution context to restore.

        Raises:
            AbortSignal: If checkpoint not found and no fallback steps provided.
        """
        checkpoint_id = primitive.label

        if checkpoint_id not in self._checkpoints:
            message = f"Checkpoint not found: {checkpoint_id}"
            logger.warning(message)

            # Try fallback steps if provided
            if primitive.fallback_steps:
                ctx["notes"].append(f"Resume: checkpoint '{checkpoint_id}' not found, executing fallback")
                self._execute_steps(primitive.fallback_steps, ctx)
                return

            # No fallback, abort
            raise AbortSignal(message)

        # Get checkpoint state
        checkpoint = self._checkpoints[checkpoint_id]

        # Restore execution context
        context = checkpoint.execution_context
        if "state" in context:
            # Note: We can't directly restore the emulator state from a checkpoint
            # In a full implementation, you'd use the SaveManager for that
            logger.debug("Restoring execution context from checkpoint: %s", checkpoint_id)

        # Restore snapshots and frames
        ctx["snapshots"] = context.get("snapshots", []).copy()
        ctx["notes"].append(f"Resumed from checkpoint: {checkpoint_id}")
        logger.info("Resumed from checkpoint: %s", checkpoint_id)

    def _handle_save_checkpoint(self, primitive: SaveStateCheckpointPrimitive, ctx: Dict[str, Any]) -> None:
        """Save game state to a checkpoint slot.

        This creates a persistent save point that can be loaded later for recovery
        or trying alternative strategies.

        Args:
            primitive: SaveStateCheckpointPrimitive with slot number and label.
            ctx: Execution context (for notes).

        Raises:
            AbortSignal: If SaveManager is not available or save fails.
        """
        slot = primitive.slot
        label = primitive.label

        if not self._save_manager:
            raise AbortSignal(
                f"Cannot save checkpoint slot {slot}: SaveManager not configured"
            )

        # Save the game state to the slot
        success = self._save_manager.save_slot(slot, description=label)
        if not success:
            raise AbortSignal(
                f"Failed to save checkpoint slot {slot}: {label}"
            )

        ctx["notes"].append(f"Saved checkpoint slot {slot}: {label}")
        logger.info("Saved checkpoint slot %d: %s", slot, label)

    def _handle_load_checkpoint(self, primitive: LoadStateCheckpointPrimitive, ctx: Dict[str, Any]) -> None:
        """Load game state from a checkpoint slot.

        Restores the dungeon to a previously saved state for recovery or
        trying alternative strategies after failures.

        Args:
            primitive: LoadStateCheckpointPrimitive with slot number.
            ctx: Execution context (for notes).

        Raises:
            AbortSignal: If SaveManager is not available, slot doesn't exist, or load fails.
        """
        slot = primitive.slot

        if not self._save_manager:
            raise AbortSignal(
                f"Cannot load checkpoint slot {slot}: SaveManager not configured"
            )

        # Load the game state from the slot
        success = self._save_manager.load_slot(slot)
        if not success:
            raise AbortSignal(
                f"Failed to load checkpoint slot {slot} (slot may not exist)"
            )

        ctx["notes"].append(f"Loaded checkpoint slot {slot}")
        logger.info("Loaded checkpoint slot %d", slot)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointState]:
        """Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint label/ID to retrieve.

        Returns:
            CheckpointState if found, None otherwise.
        """
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs created during execution.

        Returns:
            List of checkpoint IDs in creation order.
        """
        return list(self._checkpoints.keys())

    def clear_checkpoints(self) -> None:
        """Clear all stored checkpoints.

        Useful for cleaning up after execution or starting fresh.
        """
        self._checkpoints.clear()
        logger.info("Cleared all checkpoints")

    def save_checkpoint_to_disk(
        self, checkpoint_id: str, path: str | Any
    ) -> bool:
        """Save a checkpoint to disk as JSON.

        Args:
            checkpoint_id: The checkpoint ID to save.
            path: File path to save checkpoint to.

        Returns:
            True if save successful, False otherwise.

        Raises:
            ValueError: If checkpoint doesn't exist.
        """
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        checkpoint = self._checkpoints[checkpoint_id]

        try:
            import json
            from pathlib import Path

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            logger.info("Saved checkpoint %s to %s", checkpoint_id, path)
            return True
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save checkpoint %s to %s: %s",
                checkpoint_id,
                path,
                e,
            )
            return False

    def load_checkpoint_from_disk(self, path: str | Any) -> CheckpointState:
        """Load a checkpoint from disk JSON.

        Args:
            path: File path to load checkpoint from.

        Returns:
            CheckpointState instance.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint data is invalid.
        """
        try:
            import json
            from pathlib import Path

            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")

            with open(path, "r") as f:
                data = json.load(f)

            checkpoint = CheckpointState.from_dict(data)

            if not checkpoint.is_valid():
                errors = checkpoint.validate()
                raise ValueError(
                    f"Loaded checkpoint is invalid: {'; '.join(errors)}"
                )

            logger.info("Loaded checkpoint from %s", path)
            return checkpoint
        except FileNotFoundError:
            raise
        except (OSError, IOError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to load checkpoint from {path}: {e}"
            ) from e


class BreakSignal(RuntimeError):
    """Raised to exit the current block early."""


class AbortSignal(RuntimeError):
    """Raised when a skill decides to abort execution."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
