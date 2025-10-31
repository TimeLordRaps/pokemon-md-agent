"""Example skill: Navigate to stairs using Skill DSL."""

from src.skills.dsl import (
    Skill, tap, hold, waitTurn, face, capture, read_state,
    expect, annotate, checkpoint, Button, Direction
)

# Define the navigate_to_stairs skill
navigate_to_stairs = Skill(
    name="navigate_to_stairs",
    description="Navigate to stairs by moving and checking for obstacles",
    actions=[
        # Initial state capture
        capture("start_navigation"),

        # Read current position
        read_state(["position", "floor"]),

        # Face up initially
        face(Direction.UP),

        # Checkpoint before movement
        checkpoint("before_movement"),

        # Move forward sequence
        tap(Button.UP),
        waitTurn(),

        # Check for stairs
        read_state(["visible_entities"]),
        expect("any(e.get('type') == 'stairs' for e in visible_entities)", "Stairs should be visible"),

        # Annotate success
        annotate("Successfully navigated to stairs"),

        # Final capture
        capture("reached_stairs")
    ]
)

if __name__ == "__main__":
    # Example usage (would normally be executed by runtime)
    print(f"Skill: {navigate_to_stairs.name}")
    print(f"Description: {navigate_to_stairs.description}")
    print(f"Actions: {len(navigate_to_stairs.actions)}")

    # Print action sequence
    for i, action in enumerate(navigate_to_stairs.actions):
        print(f"{i+1}. {action.__class__.__name__}")