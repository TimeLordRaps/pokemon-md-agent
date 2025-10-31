# Skills: Pause & Checkpoint System

## Overview

The skill system now includes comprehensive **pause/checkpoint/breakpoint** primitives for:
- Execution state snapshots (save/resume mid-skill)
- Game state persistence (save/load slots)
- **Mid-skill model inference** for adaptive agent behavior

This enables skills to query the LM during execution when encountering unexpected situations or decision points.

## New Primitives

### 1. CheckpointPrimitive

Save the current execution state (notes, frames, snapshots) without persisting game state.

```json
{
  "primitive": "checkpoint",
  "label": "before_boss_fight",
  "description": "Saved state before combat, can resume if battle fails"
}
```

**Use case**: Recovery from transient failures, attempting alternative paths.

**Key fields**:
- `label`: Unique checkpoint identifier (1-64 chars)
- `description`: Optional metadata (max 200 chars)

### 2. ResumePrimitive

Resume from a previously created checkpoint, restoring execution state.

```json
{
  "primitive": "resume",
  "label": "before_boss_fight",
  "fallback_steps": [
    {"primitive": "annotate", "message": "Checkpoint not found, using fallback strategy"}
  ]
}
```

**Use case**: Recover from failures by jumping back to a known good state.

**Key fields**:
- `label`: Checkpoint to resume from
- `fallback_steps`: Optional steps if checkpoint doesn't exist

### 3. SaveStateCheckpointPrimitive

Persist the current game state to a save slot (0-15).

```json
{
  "primitive": "save_checkpoint",
  "slot": 0,
  "label": "safe_room_with_full_hp"
}
```

**Use case**: Save before risky operations (boss fights, dangerous dungeons).

**Integrates with**: `SaveManager` for persistent state storage.

### 4. LoadStateCheckpointPrimitive

Restore game state from a save slot.

```json
{
  "primitive": "load_checkpoint",
  "slot": 0
}
```

**Use case**: Rollback after failed attempts.

### 5. InferenceCheckpointPrimitive (NEW - Critical for Adaptive Agent)

**Pause skill execution and query the model for next steps.**

```json
{
  "primitive": "inference_checkpoint",
  "label": "boss_fight_decision",
  "context": "At boss battle HP 50%, considering whether to use healing item or continue attacking. Need model decision.",
  "timeout_seconds": 30
}
```

**This is the breakthrough primitive that enables**:
- Adaptive decision-making during skill execution
- Recovery from unexpected situations
- Multi-attempt strategies with model guidance
- Real-time LM integration (not pre-planned)

**Flow**:
1. Skill pauses at checkpoint
2. Runtime captures current game state (screenshot + semantic state)
3. Runtime calls LM async with context and game state
4. LM returns new list of steps (or empty if no changes needed)
5. Returned steps are executed immediately
6. Skill continues or completes

**Key fields**:
- `label`: Checkpoint identifier
- `context`: What the skill is trying to do and why this decision point exists
- `timeout_seconds`: Max wait for model (5-300s, default 30s)

## Architecture

### Execution Flow

```
Skill → _execute_steps() → _execute_primitive() →

  If CheckpointPrimitive:
    _create_checkpoint(label, game_state, execution_state)

  If InferenceCheckpointPrimitive:
    _handle_inference_checkpoint(label, context) →
      await _inference.query_model(label, context, game_state, timeout) →
        asyncio.wait_for(inference_fn(...), timeout) →
      [returned_steps] → await _execute_steps([returned_steps])
```

### State Storage

**ExecutionCheckpoint** dataclass:
```python
@dataclass
class ExecutionCheckpoint:
    label: str
    description: Optional[str]
    game_state: Dict[str, Any]  # From semantic_state()
    execution_state: Dict[str, Any]  # {notes, frames, snapshots}
    timestamp: float
```

### Inference Integration

**InferenceCheckpointHandler**:
```python
async def query_model(
    label: str,
    context: str,
    game_state: Dict[str, Any],
    timeout_seconds: int,
) -> List[Step]:
    """
    Calls inference_fn with:
    - label: Checkpoint identifier
    - data: {"context": str, "game_state": dict}

    Returns: List of primitives to execute next (or empty list)
    """
```

**Usage**:
```python
runtime = PythonSkillRuntime(
    controller=mgba_controller,
    skill_lookup=skill_registry.lookup,
    inference_fn=async_model_inference_fn,  # Your LM integration
)
```

## Implementation Status

| Feature | Status | Details |
|---------|--------|---------|
| Primitive definitions | DONE | All 5 in spec.py |
| Checkpoint creation | IN PROGRESS | Needs handler in _execute_primitive |
| Resume logic | IN PROGRESS | Needs restore implementation |
| Save/Load wiring | PENDING | Needs SaveManager integration |
| Inference checkpoint | PENDING | Needs async/await implementation |
| Tests | PENDING | Need comprehensive test coverage |

## Missing Implementation (TODO)

### 1. python_runtime.py Updates

**Add imports**:
```python
import asyncio
from typing import Coroutine

from .spec import (
    CheckpointPrimitive,
    ResumePrimitive,
    SaveStateCheckpointPrimitive,
    LoadStateCheckpointPrimitive,
    InferenceCheckpointPrimitive,
)
```

**Make runtime async**:
```python
class PythonSkillRuntime:
    def __init__(self, inference_fn: Optional[Callable[..., Coroutine]] = None):
        self._inference = InferenceCheckpointHandler(inference_fn)

    def run(self, spec: SkillSpec, ..., event_loop: Optional[asyncio.AbstractEventLoop] = None):
        if event_loop is None:
            event_loop = asyncio.get_event_loop()
        return event_loop.run_until_complete(self._execute_steps(spec.steps, ctx))

    async def _execute_steps(...):
        # Convert all step execution to async
```

**Add checkpoint handlers**:
```python
async def _execute_primitive(self, node: Primitive, ctx: Dict) -> None:
    ...
    elif isinstance(node, CheckpointPrimitive):
        self._create_checkpoint(node.label, node.description, ctx)
    elif isinstance(node, InferenceCheckpointPrimitive):
        await self._handle_inference_checkpoint(node, ctx)
    ...

def _create_checkpoint(self, label, description, ctx):
    checkpoint = ExecutionCheckpoint(
        label=label,
        description=description,
        game_state=self._exec.refresh_state(),
        execution_state={"notes": ctx["notes"], "frames": ctx["frames"], ...},
    )
    self._checkpoints[label] = checkpoint

async def _handle_inference_checkpoint(self, node, ctx):
    game_state = self._exec.refresh_state()
    next_steps = await self._inference.query_model(
        label=node.label,
        context=node.context,
        game_state=game_state,
        timeout_seconds=node.timeout_seconds,
    )
    if next_steps:
        await self._execute_steps(next_steps, ctx)
```

### 2. SaveManager Integration

Wire `save_state_slot()` and `load_state_slot()` from MGBAController:

```python
def _execute_primitive(self, node, ctx):
    elif isinstance(node, SaveStateCheckpointPrimitive):
        success = self._controller.save_state_slot(node.slot)
        if not success:
            raise AbortSignal(f"Failed to save to slot {node.slot}")
    elif isinstance(node, LoadStateCheckpointPrimitive):
        success = self._controller.load_state_slot(node.slot)
        if not success:
            raise AbortSignal(f"Failed to load from slot {node.slot}")
```

### 3. Comprehensive Tests

Create tests in `tests/test_skill_pause_checkpoint.py`:

```python
def test_checkpoint_create_and_resume():
    """Test checkpoint save/restore cycle."""

def test_checkpoint_not_found_fallback():
    """Test fallback steps when checkpoint doesn't exist."""

def test_save_load_game_state():
    """Test save/load slot integration."""

async def test_inference_checkpoint_basic():
    """Test pause and model query."""

async def test_inference_checkpoint_timeout():
    """Test timeout graceful handling."""

async def test_inference_checkpoint_returns_steps():
    """Test execution of model-returned steps."""

def test_checkpoint_before_risky_operation():
    """Integration: save checkpoint, attempt risky op, recover if failed."""
```

### 4. Skill Examples

Update `src/skills/examples/` to demonstrate pause patterns:

**fight_wild_monster.py** (existing):
```python
# Already has checkpoint calls - now they'll actually work!
checkpoint("before_combat")
# ... battle logic ...
```

**adaptive_dungeon_navigation.py** (new):
```python
{
    "meta": {"name": "adaptive_navigation", ...},
    "steps": [
        {"primitive": "refresh_state"},
        {"primitive": "save_checkpoint", "slot": 0, "label": "safe_room"},

        # Try to navigate to stairs
        {"primitive": "annotate", "message": "Attempting to navigate to stairs"},
        {"primitive": "tap", "button": "UP", "repeat": 3},

        # Check if stuck or in unexpected situation
        {"primitive": "inference_checkpoint",
         "label": "stuck_check",
         "context": "Tried moving UP 3 times. If still in unsafe position or unexpected room, model should suggest recovery.",
         "timeout_seconds": 30},

        # If model returned steps, they execute above
        # Otherwise continue
        {"primitive": "success", "summary": "Navigated safely"}
    ]
}
```

## Usage Pattern for Adaptive Agent

```python
from src.skills.python_runtime import PythonSkillRuntime

async def my_model_inference(label: str, data: dict) -> List[Step]:
    """Your LM integration here."""
    screenshot = data["game_state"].get("screen")
    context = data["context"]

    # Call your LM
    response = await llm.generate(
        prompt=f"You paused at checkpoint {label}.\n{context}\nWhat should we do next?",
        image=screenshot,
        output_schema=SkillSpec,
    )

    # Extract next steps from response
    return response.steps or []

# Create runtime with inference capability
runtime = PythonSkillRuntime(
    controller=mgba_controller,
    skill_lookup=skill_registry.lookup,
    inference_fn=my_model_inference,  # Enable adaptive behavior!
)

# Run skill - it will now pause for model input at inference checkpoints
result = runtime.run(skill_spec, params={})
```

## Benefits

| Capability | Before | After |
|-----------|--------|-------|
| Recovery from failures | Manual skill retry | Checkpoint restore |
| Adaptive decisions | Pre-planned only | Mid-execution model queries |
| Error handling | Abort/restart | Fallback + recovery |
| State persistence | Manual management | Integrated with SaveManager |
| Agent learning | Limited (pre-trained) | Real-time adaptation |

## Performance Considerations

- **Checkpoint overhead**: Minimal (semantic_state capture ~10-50ms)
- **Inference delay**: Model-dependent (typically 5-30s)
- **Concurrency**: Inference is fully async, doesn't block other game actions
- **Memory**: Checkpoints stored in-memory (max ~16 for save slots)

## Next Phase

Once implementation is complete:
1. Integrate with agent's main loop for true adaptive behavior
2. Add checkpoint cleanup/expiry policies
3. Implement checkpoint replay for debugging/analysis
4. Add metrics: checkpoint hit rate, inference success rate, fallback frequency
5. Create skill library with common pause patterns

---

**Status**: Primitive definitions complete, implementation in progress.
**Owner**: Core skills team
**Related**: ModelRouter, SaveManager, SemanticState
