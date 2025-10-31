#!/usr/bin/env python
"""Quick test script for vision schema validation."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run quick validation of game state schema."""
    try:
        from src.models.game_state_schema import GameState, Entity, GameStateEnum
        from src.models.game_state_utils import (
            parse_model_output,
            validate_game_state,
            generate_few_shot_examples,
            schema_to_prompt_json
        )

        print("[INFO] Schema imports successful")

        # Create sample state
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=0.95
        )

        # Validate
        report = validate_game_state(state)
        print("[OK] Schema validation: PASS")
        print(f"  - Player at: {state.player_pos}")
        print(f"  - Floor: {state.floor}")
        print(f"  - Confidence: {state.confidence:.2f}")
        print(f"  - Quality score: {report['quality_score']:.2f}")

        # Test JSON roundtrip
        json_str = state.model_dump_json()
        restored = GameState.model_validate_json(json_str)
        assert restored.player_pos == state.player_pos
        print("[OK] JSON roundtrip: PASS")

        print("\n[OK] All validation checks passed!")
        return 0

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
