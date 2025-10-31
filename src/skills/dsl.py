"""Pythonic Skill DSL - Pydantic-guided declarative skills for Pokemon MD gameplay."""

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Button(str, Enum):
    """Game controller buttons."""
    A = "a"
    B = "b"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    START = "start"
    SELECT = "select"


class Direction(str, Enum):
    """Cardinal directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


# Action Types - Pydantic models for skill actions
class Tap(BaseModel):
    """Tap a button."""
    button: Button


class Hold(BaseModel):
    """Hold a button for specified frames."""
    button: Button
    frames: int = Field(gt=0)


class Release(BaseModel):
    """Release a button."""
    button: Button


class WaitTurn(BaseModel):
    """Wait one turn (A+B press cycle)."""
    pass


class Face(BaseModel):
    """Face a direction."""
    direction: Direction


class Capture(BaseModel):
    """Capture current state with label."""
    label: str


class ReadState(BaseModel):
    """Read state fields."""
    fields: List[str]


class Expect(BaseModel):
    """Assert condition with message."""
    condition: str
    message: str


class Annotate(BaseModel):
    """Add annotation to trajectory."""
    message: str


class Break(BaseModel):
    """Break execution."""
    pass


class Abort(BaseModel):
    """Abort execution with message."""
    message: str


class Checkpoint(BaseModel):
    """Create checkpoint with label."""
    label: str


class Resume(BaseModel):
    """Resume from last checkpoint."""
    pass


class Save(BaseModel):
    """Save game state to slot."""
    slot: int


class Load(BaseModel):
    """Load game state from slot."""
    slot: int


# Union of all action types
Action = Union[
    Tap, Hold, Release, WaitTurn, Face, Capture,
    ReadState, Expect, Annotate, Break, Abort,
    Checkpoint, Resume, Save, Load
]


class Trigger(BaseModel):
    """Trigger condition for skill activation."""
    type: str
    condition: str
    description: Optional[str] = None


class Skill(BaseModel):
    """Skill definition with sequenced actions."""
    name: str
    description: Optional[str] = None
    actions: List[Action]


class SkillDSL(BaseModel):
    """Complete skill definition with triggers."""
    skill: Skill
    triggers: List[Trigger] = Field(default_factory=list)


def navigate_to_stairs():
    """Navigate to visible stairs."""
    return Skill(
        name="navigate_to_stairs",
        description="Navigate to stairs by moving and checking for obstacles",
        actions=[
            read_state(["coords", "stairs_visible", "path_to_stairs"]),
            expect("stairs_visible == True", "Stairs must be visible"),
            face(Direction.UP),
            tap(Button.A),
            waitTurn(),
            annotate("Navigation to stairs completed")
        ]
    )


# DSL primitive functions - Pythonic skill building
def tap(btn: Button) -> Tap:
    """Tap a button."""
    return Tap(button=btn)


def hold(btn: Button, frames: int) -> Hold:
    """Hold a button for frames."""
    return Hold(button=btn, frames=frames)


def release(btn: Button) -> Release:
    """Release a button."""
    return Release(button=btn)


def waitTurn() -> WaitTurn:
    """Wait one turn (A+B cycle)."""
    return WaitTurn()


def face(dir: Direction) -> Face:
    """Face a direction."""
    return Face(direction=dir)


def capture(label: str) -> Capture:
    """Capture state with label."""
    return Capture(label=label)


def read_state(fields: List[str]) -> ReadState:
    """Read state fields."""
    return ReadState(fields=fields)


def expect(cond: str, msg: str) -> Expect:
    """Assert condition."""
    return Expect(condition=cond, message=msg)


def annotate(msg: str) -> Annotate:
    """Add annotation."""
    return Annotate(message=msg)


def break_() -> Break:
    """Break execution."""
    return Break()


def abort(msg: str) -> Abort:
    """Abort with message."""
    return Abort(message=msg)


def checkpoint(lbl: str) -> Checkpoint:
    """Create checkpoint."""
    return Checkpoint(label=lbl)


def resume() -> Resume:
    """Resume from checkpoint."""
    return Resume()


def save(slot: int) -> Save:
    """Save to slot."""
    return Save(slot=slot)


def load(slot: int) -> Load:
    """Load from slot."""
    return Load(slot=slot)