# Skill Libraries

This directory contains skill libraries for different Pokemon Mystery Dungeon agent configurations. Each subdirectory represents a separate skill library that can be loaded by the agent.

## Directory Structure

```
skill-libraries/
├── basic/                    # Basic survival and navigation skills
│   ├── use_food_when_hungry.yaml
│   ├── heal_when_low_hp.yaml
│   └── take_stairs_when_visible.yaml
├── exploration/              # Advanced exploration skills
├── combat/                   # Battle-related skills
└── custom/                   # User-defined skills
```

## Creating a New Skill Library

1. Create a new directory under `skill-libraries/`
2. Add YAML skill definition files
3. Load the library in your agent code:

```python
from src.skills.dsl import SkillDSL

# Load basic skills
dsl = SkillDSL(library_name="basic")
skills = dsl.load_all_skills()
```

## Skill Definition Format

Each skill is defined in a YAML file with the following structure:

```yaml
name: skill_name
description: Brief description of what the skill does
cooldown: 30.0  # Seconds between executions
triggers:
  - type: ram_condition|vision_condition|time_based|event_based
    condition: condition_expression
preconditions:
  - ram_condition1
  - ram_condition2
actions:
  - type: press_button|use_item|move_to|wait|task
    params:
      # Action-specific parameters
fallback:
  - type: action_type
    params: {}
priority: 10  # Higher numbers = higher priority
max_executions: 5  # Optional limit on executions
```

## RAM Conditions

Common RAM-based conditions:
- `hp < 50` - HP below 50
- `belly < 30` - Belly below 30% of max
- `is_hungry` - Belly < 30% of max
- `is_low_hp` - HP < 50% of max
- `has_food_item` - Has food items in inventory
- `has_recovery_item` - Has healing items in inventory
- `not_in_battle` - Not currently in battle

## Vision Conditions

Vision-based conditions (require sprite detection):
- `stairs_visible` - Stairs are visible on screen
- `enemy_nearby` - Enemy Pokemon nearby
- `item_visible` - Items visible on ground

## Action Types

- `press_button`: Press game buttons
  - `keys`: List of buttons to press
- `use_item`: Use an item from inventory
  - `item_type`: Type of item (food, recovery, etc.)
  - `predicate`: How to select item (first_available, best_food_available)
- `move_to`: Move to coordinates
  - `x`, `y`: Target coordinates
- `wait`: Wait for frames
  - `frames`: Number of frames to wait
- `task`: Execute a complex task
  - `task`: Task identifier