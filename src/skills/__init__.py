"""Skills package for Pokemon MD agent."""

# Legacy YAML DSL exports (still used by downstream tooling)
from .dsl import Skill, Action, Trigger, SkillDSL  # noqa: F401
from .runtime import SkillRuntime, RAMPredicates, ExecutionContext  # noqa: F401

# New Python DSL exports
from .dsl import (  # noqa: F401
    # Pydantic models
    Tap, Hold, Release, WaitTurn, Face, Capture, ReadState,
    Expect, Annotate, Break, Abort, Checkpoint, Resume,
    Save, Load, Action, Skill, Button, Direction,
    # DSL functions
    tap, hold, release, waitTurn, face, capture,
    read_state, expect, annotate, break_, abort,
    checkpoint, resume, save, load,
)

__all__ = [
    # Legacy surface
    "Skill",
    "Action",
    "Trigger",
    "SkillDSL",
    "SkillRuntime",
    "RAMPredicates",
    "ExecutionContext",
    # New Python DSL
    "Tap", "Hold", "Release", "WaitTurn", "Face", "Capture", "ReadState",
    "Expect", "Annotate", "Break", "Abort", "Checkpoint", "Resume",
    "Save", "Load", "Button", "Direction",
    "tap", "hold", "release", "waitTurn", "face", "capture",
    "read_state", "expect", "annotate", "break_", "abort",
    "checkpoint", "resume", "save", "load",
]
