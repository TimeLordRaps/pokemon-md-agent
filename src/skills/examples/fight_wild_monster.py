"""Example skill: Fight wild monster."""

from src.skills.dsl import (
    Skill, read_state, expect, face, tap, hold, waitTurn, annotate,
    checkpoint, Button, Direction
)

# Define the fight_wild_monster skill
fight_wild_monster = Skill(
    name="fight_wild_monster",
    description="Engage and defeat a wild monster using basic attacks",
    actions=[
        # Initial state assessment
        read_state(["wild_monster_present", "monster_type", "monster_position", "current_hp"]),

        # Ensure there's a monster to fight
        expect("wild_monster_present == True", "No wild monster present to fight"),

        # Ensure we have HP to fight
        expect("current_hp > 20", "HP too low to engage in combat"),

        # Navigate to monster position
        read_state(["current_position"]),
        expect("current_position == monster_position", "Must be adjacent to monster"),

        # Create checkpoint before combat
        checkpoint("before_combat"),

        # Face the monster
        face(Direction.UP),  # Assuming monster is above

        # Initiate battle by pressing A
        tap(Button.A),
        waitTurn(),

        # Basic attack pattern - tap A repeatedly
        tap(Button.A),
        waitTurn(),
        tap(Button.A),
        waitTurn(),
        tap(Button.A),
        waitTurn(),

        # Check if monster defeated
        read_state(["wild_monster_present", "exp_gained"]),
        expect("wild_monster_present == False", "Monster should be defeated"),

        # Annotate victory
        annotate("Successfully defeated wild monster"),

        # Final state capture
        read_state(["current_hp", "current_exp"])
    ]
)

if __name__ == "__main__":
    # Example usage
    print(f"Skill: {fight_wild_monster.name}")
    print(f"Description: {fight_wild_monster.description}")
    print(f"Actions: {len(fight_wild_monster.actions)}")

    # Print action sequence
    for i, action in enumerate(fight_wild_monster.actions):
        print(f"{i+1}. {action.__class__.__name__}")