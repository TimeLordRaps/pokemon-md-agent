"""Example skill demonstrating adaptive dungeon exploration using InferenceCheckpoint.

This skill shows how to use InferenceCheckpointPrimitive to enable mid-execution
LM decision-making. The agent pauses at key decision points and queries the model
for adaptive next steps based on current game state and context.

Example scenario:
- Agent enters a dungeon floor
- Explores until reaching a branching path
- Pauses at the branch and queries the model for direction
- Continues based on model's adaptive decision
- Repeat at each major decision point
"""

from src.skills.spec import (
    SkillSpec,
    SkillMeta,
    AnnotatePrimitive,
    TapPrimitive,
    WaitTurnPrimitive,
    RefreshStatePrimitive,
    InferenceCheckpointPrimitive,
    IfBlock,
    WhileBlock,
)

# Define skill metadata
metadata = SkillMeta(
    name="navigate_dungeon_floor_adaptive",
    description="Adaptively navigate a dungeon floor using mid-execution LM decisions",
    version="v1",
    tags=["dungeon", "exploration", "adaptive"],
    expects=["player_moved_forward", "floor_explored"],
    partial_success_notes=[
        "Reached a dead end but explored partially",
        "Encountered enemy and retreated",
        "Found stairs but took inefficient path",
    ],
)

# Define the adaptive dungeon exploration skill
steps = [
    # Phase 1: Initial exploration setup
    AnnotatePrimitive(message="Starting adaptive dungeon floor exploration"),
    RefreshStatePrimitive(fields=["position", "visible_tiles", "nearby_enemies"]),
    AnnotatePrimitive(message="Initial state captured, beginning exploration"),

    # Phase 2: Main exploration loop
    WhileBlock(
        condition="state.get('current_floor', 0) == state.get('target_floor', 1)",
        body=[
            # Capture current state before each decision
            RefreshStatePrimitive(
                fields=["position", "visible_tiles", "nearby_enemies", "inventory"]
            ),

            # Decision checkpoint: Where should we go?
            InferenceCheckpointPrimitive(
                label="exploration_decision",
                context="""You are exploring a dungeon floor. Based on the current game state:
                - Your position
                - Visible tiles
                - Nearby enemies
                - Your inventory

                Decide the next 1-3 actions. Should you:
                - Move towards the stairs (if visible)?
                - Explore new areas?
                - Engage an enemy?
                - Use an item?

                Provide 1-3 primitive actions to take next.""",
                timeout_seconds=10,
            ),

            # Fallback: Simple forward movement if model times out
            AnnotatePrimitive(
                message="Continuing with model-suggested or fallback actions"
            ),
        ],
        max_iterations=50,  # Prevent infinite loops
    ),

    # Phase 3: Tactical checkpoint - Major threat detected
    InferenceCheckpointPrimitive(
        label="major_threat_response",
        context="""A significant threat has been detected ahead. Based on:
        - Your HP and status
        - Nearby enemy power levels
        - Available escape routes
        - Your inventory

        Should you:
        - Prepare for combat (buff, use items)?
        - Try to sneak around?
        - Retreat to a safe area?

        Respond with 1-5 actions.""",
        timeout_seconds=15,
    ),

    # Phase 4: Completion verification
    RefreshStatePrimitive(
        fields=["position", "current_floor", "floor_cleared", "items_collected"]
    ),
    AnnotatePrimitive(
        message="Adaptive dungeon exploration complete"
    ),
]

# Assemble the full skill specification
adaptive_dungeon_exploration_skill = SkillSpec(
    meta=metadata,
    steps=steps,
    parameters={
        "target_floor": {
            "type": "int",
            "description": "Which floor to explore (1-indexed)",
            "default": 1,
        },
        "max_exploration_time": {
            "type": "float",
            "description": "Max time in seconds for exploration phase",
            "default": 300.0,
        },
    },
)


# Example usage documentation
"""
USAGE EXAMPLE:
==============

```python
from src.skills.async_skill_runtime import AsyncSkillRuntime
from examples.adaptive_dungeon_exploration_skill import adaptive_dungeon_exploration_skill

# Initialize runtime with model router for LM inference
runtime = AsyncSkillRuntime(
    controller=mgba_controller,
    model_router=model_router,  # Provides LM inference capability
)

# Execute the adaptive skill
result = await runtime.run_async(
    adaptive_dungeon_exploration_skill,
    params={"target_floor": 1, "max_exploration_time": 300},
    timeout_seconds=600,  # 10 minute overall timeout
)

# Check results
print(f"Status: {result.status}")
print(f"Notes: {result.notes}")
print(f"Frames captured: {len(result.frames)}")
```

KEY FEATURES:
=============

1. **Multiple Decision Points**: The skill has three main InferenceCheckpoint primitives:
   - exploration_decision: Per-turn tactical decisions
   - major_threat_response: Strategic response to threats
   - (implicit in loop) Continuous adaptation

2. **Context-Aware Prompts**: Each checkpoint provides rich context about:
   - Current game state
   - Available options
   - Time/resource constraints

3. **Timeout Handling**: Each checkpoint has a timeout_seconds parameter
   - Graceful fallback if model doesn't respond in time
   - Quick decisions (10-15s) for responsive gameplay

4. **Loop with Iteration Limit**: Main exploration loop has max_iterations=50
   - Prevents infinite loops
   - Allows extended exploration while being safe

5. **State Validation**: RefreshStatePrimitive calls capture game state
   - Ensures LM has current information
   - Tracks progress through dungeon

EXPECTED MODEL RESPONSES:
========================

The model should respond with JSON like:
```json
{
  "steps": [
    {"primitive": "tap", "button": "UP", "repeat": 1},
    {"primitive": "wait_turn"},
    {"primitive": "tap", "button": "A"}
  ],
  "reasoning": "Moving north to unexplored area, then examining it"
}
```

Allowed primitive types in inference responses:
- tap, hold, release: Button inputs
- wait_turn: Advance turn without input
- capture: Take screenshot
- refresh_state: Update state
- annotate: Add notes to trajectory
- abort, success: End execution with status
- call: Invoke other skills
"""
