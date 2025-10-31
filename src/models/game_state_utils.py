"""Utilities for game state schema operations.

Provides helpers for:
- Converting schema to LM-friendly JSON
- Parsing and validating model outputs
- Generating few-shot examples
- Error recovery for partial data
"""

import json
import logging
from typing import Optional, Dict, Any, List
from pydantic import ValidationError
from src.models.game_state_schema import GameState, Entity, GameStateEnum, RoomType

logger = logging.getLogger(__name__)


def schema_to_json_template() -> str:
    """Generate JSON schema template for LM prompt guidance.

    Returns:
        JSON-formatted schema example for use in system prompts.
    """
    schema = GameState.model_json_schema()
    return json.dumps(schema, indent=2)


def schema_to_prompt_json() -> str:
    """Generate compact JSON template showing key fields only.

    Returns:
        Minimal JSON structure highlighting required/important fields.
    """
    example = {
        "player_pos": "[x, y]",
        "player_hp": "int (optional)",
        "floor": "int (1-50)",
        "state": "exploring|battle|menu|stairs_found|boss_battle|unknown",
        "enemies": [
            {
                "x": 0,
                "y": 0,
                "type": "enemy",
                "species": "Pokemon name",
                "status_effects": ["poison", "burn"]
            }
        ],
        "items": [
            {
                "x": 0,
                "y": 0,
                "type": "item",
                "name": "Item name"
            }
        ],
        "confidence": 0.0,
        "notes": "Additional observations"
    }
    return json.dumps(example, indent=2)


def parse_model_output(
    output: str,
    partial_ok: bool = True,
    confidence_threshold: float = 0.0
) -> Optional[GameState]:
    """Parse and validate model output as GameState.

    Args:
        output: Model output string (JSON or free-form)
        partial_ok: If True, allow partial/missing fields with defaults
        confidence_threshold: Reject states with confidence below this (0-1)

    Returns:
        GameState object if valid, None if unparseable and partial_ok=False

    Raises:
        ValidationError: If partial_ok=False and validation fails
    """
    try:
        # Try direct JSON parsing
        data = json.loads(output)
        state = GameState.model_validate(data)

        if state.confidence < confidence_threshold:
            logger.warning(
                f"Model confidence {state.confidence:.2f} below threshold {confidence_threshold}"
            )
            if not partial_ok:
                return None

        return state

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        if not partial_ok:
            raise ValidationError(f"Invalid JSON: {e}")
        return None

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        if not partial_ok:
            raise

        # Try to recover with defaults
        try:
            return _recover_from_partial_output(output)
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return None


def _recover_from_partial_output(output: str) -> Optional[GameState]:
    """Attempt to extract GameState from malformed output.

    Args:
        output: Raw model output

    Returns:
        GameState with best-effort extraction, None if unrecoverable
    """
    try:
        # Try to find JSON-like structure
        start = output.find('{')
        end = output.rfind('}') + 1

        if start >= 0 and end > start:
            json_candidate = output[start:end]
            data = json.loads(json_candidate)
            # Attempt validation with loose mode
            return GameState.model_validate(data)
    except Exception:
        pass

    return None


def generate_few_shot_examples(num_examples: int = 3) -> List[Dict[str, Any]]:
    """Generate few-shot examples for in-context learning.

    Args:
        num_examples: Number of examples to generate (1-5)

    Returns:
        List of (screenshot_description, expected_output) pairs
    """
    examples = [
        {
            "description": "Exploring corridor, enemy approaching from right",
            "state": GameState(
                player_pos=(8, 8),
                player_hp=30,
                floor=2,
                dungeon_name="Drenched Bluff",
                room_type=RoomType.CORRIDOR,
                state=GameStateEnum.EXPLORING,
                enemies=[
                    Entity(x=10, y=8, type="enemy", species="Zubat", level=3)
                ],
                items=[],
                threats=["Zubat 2 tiles away"],
                opportunities=["Move left to dodge"],
                confidence=0.95,
                notes="Enemy moving right, clear escape path left"
            )
        },
        {
            "description": "Combat with Bulbasaur, poisoned status",
            "state": GameState(
                player_pos=(9, 7),
                player_hp=22,
                player_status=["poison"],
                floor=5,
                dungeon_name="Mystery Dungeon",
                room_type=RoomType.CHAMBER,
                state=GameStateEnum.BATTLE,
                enemies=[
                    Entity(
                        x=11,
                        y=7,
                        type="enemy",
                        species="Bulbasaur",
                        status_effects=["confusion"],
                        hp=25,
                        level=5
                    )
                ],
                items=[
                    Entity(x=8, y=5, type="item", name="Antidote")
                ],
                threats=["Bulbasaur adjacent", "HP low"],
                opportunities=["Use Antidote", "Attack now (confused)"],
                confidence=0.88,
                notes="Enemy confusion good for attack window"
            )
        },
        {
            "description": "Stairs found, clear path to stairs",
            "state": GameState(
                player_pos=(7, 9),
                player_hp=55,
                floor=3,
                dungeon_name="Mt. Horn",
                room_type=RoomType.CORRIDOR,
                state=GameStateEnum.STAIRS,
                enemies=[],
                items=[],
                special_objects=[
                    Entity(x=9, y=6, type="stairs")
                ],
                significant_change="Found stairs at (9, 6)",
                threats=[],
                opportunities=["Move up to stairs"],
                confidence=0.99,
                notes="Direct path to stairs, no enemies in way"
            )
        },
        {
            "description": "Boss room with Rayquaza",
            "state": GameState(
                player_pos=(8, 8),
                player_hp=50,
                floor=10,
                dungeon_name="Sky Tower",
                room_type=RoomType.BOSS,
                state=GameStateEnum.BOSS,
                enemies=[
                    Entity(
                        x=10,
                        y=8,
                        type="enemy",
                        species="Rayquaza",
                        hp=100,
                        level=30
                    )
                ],
                items=[],
                threats=["Rayquaza boss encounter"],
                opportunities=["Use prepared moves"],
                confidence=0.94,
                notes="Boss battle detected, high difficulty"
            )
        },
        {
            "description": "Shop area with NPC",
            "state": GameState(
                player_pos=(10, 10),
                player_hp=40,
                floor=1,
                dungeon_name="Pokémon Square",
                room_type=RoomType.SHOP,
                state=GameStateEnum.MENU,
                enemies=[],
                items=[
                    Entity(x=12, y=10, type="item", name="Potion"),
                    Entity(x=14, y=10, type="item", name="Oran Berry")
                ],
                threats=[],
                opportunities=["Purchase items", "Talk to shopkeeper"],
                confidence=0.97,
                notes="Safe shop area, multiple items available"
            )
        },
    ]

    return examples[:min(num_examples, len(examples))]


def validate_game_state(state: GameState) -> Dict[str, Any]:
    """Validate and report on GameState quality.

    Args:
        state: GameState to validate

    Returns:
        Dict with validation report: {
            "valid": bool,
            "warnings": List[str],
            "quality_score": float (0-1),
            "issues": List[str]
        }
    """
    warnings = []
    issues = []
    quality_score = 1.0

    # Check confidence
    if state.confidence < 0.7:
        warnings.append(f"Low confidence: {state.confidence:.2f}")
        quality_score *= (state.confidence / 0.7)

    # Check coordinates
    if state.player_pos[0] < 0 or state.player_pos[1] < 0:
        issues.append("Negative player coordinates")

    # Check entity counts
    if len(state.enemies) > 20:
        warnings.append(f"Unusually high enemy count: {len(state.enemies)}")

    # Check state consistency
    if state.state == GameStateEnum.BATTLE and len(state.enemies) == 0:
        warnings.append("Battle state but no enemies detected")

    if state.state == GameStateEnum.STAIRS and len(state.special_objects) == 0:
        warnings.append("Stairs state but no stairs object detected")

    # Check HP validity
    if state.player_hp is not None and state.player_max_hp is not None:
        if state.player_hp > state.player_max_hp:
            issues.append(f"HP {state.player_hp} exceeds max {state.player_max_hp}")

    return {
        "valid": len(issues) == 0,
        "warnings": warnings,
        "issues": issues,
        "quality_score": max(0.0, quality_score)
    }


def format_state_for_decision(state: GameState) -> str:
    """Format GameState as readable text for agent decision-making.

    Args:
        state: GameState to format

    Returns:
        Human-readable description of current state
    """
    lines = [
        f"Game State: {state.state.value}",
        f"Floor: {state.floor}",
        f"Player: {state.player_pos} HP={state.player_hp or '?'}",
        f"Enemies: {len(state.enemies)} | Items: {len(state.items)}",
    ]

    if state.enemies:
        lines.append("  Enemies:")
        for enemy in state.enemies[:3]:  # Show first 3
            lines.append(
                f"    - {enemy.species or '?'} at {(enemy.x, enemy.y)} "
                f"HP={enemy.hp or '?'}"
            )

    if state.threats:
        lines.append(f"  Threats: {', '.join(state.threats[:2])}")

    if state.opportunities:
        lines.append(f"  Options: {', '.join(state.opportunities[:2])}")

    return "\n".join(lines)


__all__ = [
    "schema_to_json_template",
    "schema_to_prompt_json",
    "parse_model_output",
    "generate_few_shot_examples",
    "validate_game_state",
    "format_state_for_decision",
]
