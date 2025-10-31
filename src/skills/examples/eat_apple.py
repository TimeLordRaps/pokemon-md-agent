"""Example skill: Eat apple when belly low."""

from src.skills.dsl import (
    Skill, read_state, expect, face, tap, waitTurn, annotate,
    Button, Direction
)

# Define the eat_apple skill
eat_apple = Skill(
    name="eat_apple",
    description="Eat an apple when belly is low to restore hunger",
    actions=[
        # Check current belly status
        read_state(["belly_level", "has_apple", "apple_position"]),

        # Ensure belly is low enough to warrant eating
        expect("belly_level < 30", "Belly must be low to eat apple"),

        # Ensure we have an apple
        expect("has_apple == True", "Must have an apple to eat"),

        # Navigate to apple position if needed
        read_state(["current_position"]),
        expect("current_position == apple_position", "Must be at apple position"),

        # Face down to interact with ground item
        face(Direction.DOWN),

        # Press A to pick up/eat apple
        tap(Button.A),
        waitTurn(),

        # Verify belly improved
        read_state(["belly_level"]),
        expect("belly_level > 50", "Apple should have restored belly"),

        # Annotate success
        annotate("Successfully ate apple, belly restored")
    ]
)

if __name__ == "__main__":
    # Example usage
    print(f"Skill: {eat_apple.name}")
    print(f"Description: {eat_apple.description}")
    print(f"Actions: {len(eat_apple.actions)}")

    # Print action sequence
    for i, action in enumerate(eat_apple.actions):
        print(f"{i+1}. {action.__class__.__name__}")