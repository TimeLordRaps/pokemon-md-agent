"""Skill runtime - async execution engine with trajectory logging and error handling."""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass, field
import logging
import time

from .dsl import Skill, Action, Tap, Hold, Release, WaitTurn, Face, Capture, ReadState, Expect, Annotate, Break, Abort, Checkpoint, Resume, Save, Load
from ..environment.mgba_controller import MGBAController

logger = logging.getLogger(__name__)


class RAMPredicates:
    """RAM-based predicate evaluation for skill triggers."""

    def __init__(self, controller: MGBAController):
        """Initialize RAM predicates."""
        self.controller = controller


@dataclass
class TrajectoryEntry:
    """Single trajectory log entry."""
    timestamp: float
    action: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    annotation: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for skill execution."""
    controller: MGBAController
    ram_predicates: RAMPredicates
    current_state: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[TrajectoryEntry] = field(default_factory=list)


class SkillRuntime:
    """Async skill execution runtime."""

    def __init__(self, controller: MGBAController):
        """Initialize runtime with MGBA controller."""
        self.controller = controller
        logger.info("Initialized SkillRuntime")

    async def execute_skill(self, skill: Skill) -> bool:
        """Execute skill with full state management.

        Args:
            skill: Skill to execute

        Returns:
            True if execution successful
        """
        logger.info(f"Executing skill: {skill.name}")

        # Initialize execution context
        context = ExecutionContext(
            controller=self.controller,
            ram_predicates=RAMPredicates(self.controller)
        )

        try:
            # Execute actions sequentially with state capture
            for action in skill.actions:
                await self._execute_action(action, context)
                # Log trajectory entry after each action

            logger.info(f"Skill {skill.name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Skill {skill.name} failed: {e}")
            return False

    async def _execute_action(self, action: Action, context: ExecutionContext) -> None:
        """Execute single action with state capture."""
        state_before = context.current_state.copy()

        # Execute action based on type
        if isinstance(action, Tap):
            success = context.controller.press([action.button.value])
            if not success:
                raise RuntimeError(f"Failed to tap {action.button.value}")
        elif isinstance(action, Hold):
            success = context.controller.hold_button(action.button.value, action.frames)
            if not success:
                raise RuntimeError(f"Failed to hold {action.button.value}")
        elif isinstance(action, Release):
            success = context.controller.release_button(action.button.value)
            if not success:
                raise RuntimeError(f"Failed to release {action.button.value}")
        elif isinstance(action, WaitTurn):
            await self._wait_turn(context.controller)
        elif isinstance(action, Face):
            await self._face(action.direction.value, context.controller)
        elif isinstance(action, Capture):
            await self._capture_state(action.label, context)
        elif isinstance(action, ReadState):
            await self._read_state(action.fields, context)
        elif isinstance(action, Expect):
            await self._expect(action.condition, action.message, context)
        elif isinstance(action, Annotate):
            self._annotate(action.message, context)
        elif isinstance(action, Break):
            raise StopIteration("Execution broken")
        elif isinstance(action, Abort):
            raise RuntimeError(f"Aborted: {action.message}")
        elif isinstance(action, Checkpoint):
            self._create_checkpoint(action.label, context)
        elif isinstance(action, Resume):
            await self._resume_from_checkpoint(context)
        elif isinstance(action, Save):
            await self._save(action.slot, context)
        elif isinstance(action, Load):
            await self._load(action.slot, context)

        state_after = context.current_state.copy()

        # Add trajectory entry
        entry = TrajectoryEntry(
            timestamp=time.time(),
            action=action.__class__.__name__,
            state_before=state_before,
            state_after=state_after
        )
        context.trajectory.append(entry)

    async def _wait_turn(self, controller: MGBAController) -> None:
        """Wait one turn (A+B cycle)."""
        controller.press(["a"])
        await asyncio.sleep(0.1)
        controller.press(["b"])
        await controller.await_frames(60)

    async def _face(self, direction: str, controller: MGBAController) -> None:
        """Face direction."""
        controller.press([direction])

    async def _capture_state(self, label: str, context: ExecutionContext) -> None:
        """Capture current state with label."""
        screenshot = context.controller.screenshot()
        ram_data = context.controller.read_ram()  # Assume controller has read_ram method

        context.current_state.update({
            "label": label,
            "screenshot": screenshot,
            "ram": ram_data,
            "timestamp": time.time()
        })

    async def _read_state(self, fields: List[str], context: ExecutionContext) -> None:
        """Read specific state fields."""
        # This would typically use StateMap to get semantic fields
        # For now, placeholder implementation
        for field in fields:
            context.current_state[field] = f"mock_value_for_{field}"

    async def _expect(self, condition: str, message: str, context: ExecutionContext) -> None:
        """Evaluate expectation condition."""
        try:
            result = eval(condition, {"__builtins__": {}}, context.current_state)
            if not result:
                raise AssertionError(f"Expectation failed: {message}")
        except Exception as e:
            raise AssertionError(f"Expectation evaluation failed: {e}")

    def _annotate(self, message: str, context: ExecutionContext) -> None:
        """Add annotation to last trajectory entry."""
        if context.trajectory:
            context.trajectory[-1].annotation = message

    def _create_checkpoint(self, label: str, context: ExecutionContext) -> None:
        """Create execution checkpoint."""
        # Simplified checkpoint - in real implementation would store full state
        logger.info(f"Created checkpoint: {label}")

    async def _resume_from_checkpoint(self, context: ExecutionContext) -> None:
        """Resume from last checkpoint."""
        # Simplified resume - would restore full state
        logger.info("Resumed from checkpoint")

    async def _save(self, slot: int, context: ExecutionContext) -> None:
        """Save game state to slot."""
        # Placeholder - implement actual save logic
        context.current_state["save_slot"] = slot
        logger.info(f"Saved to slot {slot}")

    async def _load(self, slot: int, context: ExecutionContext) -> None:
        """Load game state from slot."""
        # Placeholder - implement actual load logic
        logger.info(f"Loaded from slot {slot}")


class AssertionError(Exception):
    """Raised when skill assertion fails."""
    pass