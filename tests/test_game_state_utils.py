"""Tests for game state utilities."""

import pytest
import json
from src.models.game_state_schema import GameState, Entity, GameStateEnum, RoomType
from src.models.game_state_utils import (
    schema_to_json_template,
    schema_to_prompt_json,
    parse_model_output,
    generate_few_shot_examples,
    validate_game_state,
    format_state_for_decision,
)


class TestSchemaUtilities:
    """Test schema generation utilities."""

    def test_schema_to_json_template(self):
        """Generate JSON schema template."""
        template = schema_to_json_template()
        schema = json.loads(template)

        assert "properties" in schema
        assert "player_pos" in schema["properties"]
        assert "state" in schema["properties"]

    def test_schema_to_prompt_json(self):
        """Generate compact prompt JSON."""
        prompt = schema_to_prompt_json()
        data = json.loads(prompt)

        assert "player_pos" in data
        assert "state" in data
        assert "enemies" in data

    def test_prompt_json_is_compact(self):
        """Prompt JSON is suitable for LM context."""
        prompt = schema_to_prompt_json()
        # Should be relatively short for token efficiency
        assert len(prompt) < 2000


class TestParseModelOutput:
    """Test model output parsing."""

    def test_parse_valid_json(self):
        """Parse valid JSON output."""
        json_output = '''{
            "player_pos": [12, 8],
            "floor": 3,
            "state": "exploring",
            "confidence": 0.95
        }'''

        state = parse_model_output(json_output)
        assert state is not None
        assert state.player_pos == (12, 8)
        assert state.confidence == 0.95

    def test_parse_with_all_fields(self):
        """Parse complex output with all fields."""
        json_output = '''{
            "player_pos": [12, 8],
            "player_hp": 45,
            "floor": 3,
            "dungeon_name": "Mt. Horn",
            "state": "exploring",
            "enemies": [
                {"x": 14, "y": 8, "type": "enemy", "species": "Geodude"}
            ],
            "items": [
                {"x": 10, "y": 6, "type": "item", "name": "Apple"}
            ],
            "confidence": 0.92,
            "notes": "Enemy 2 tiles away"
        }'''

        state = parse_model_output(json_output)
        assert state is not None
        assert len(state.enemies) == 1
        assert len(state.items) == 1

    def test_parse_invalid_json_partial_ok_true(self):
        """Return None for invalid JSON with partial_ok=True."""
        invalid_output = "This is not JSON"
        state = parse_model_output(invalid_output, partial_ok=True)
        assert state is None

    def test_parse_invalid_json_partial_ok_false(self):
        """Raise on invalid JSON with partial_ok=False."""
        invalid_output = "This is not JSON"
        with pytest.raises(Exception):
            parse_model_output(invalid_output, partial_ok=False)

    def test_parse_confidence_threshold(self):
        """Filter by confidence threshold."""
        json_output = '''{
            "player_pos": [12, 8],
            "floor": 3,
            "state": "exploring",
            "confidence": 0.5
        }'''

        # Below threshold with partial_ok=False returns None
        state = parse_model_output(json_output, confidence_threshold=0.7, partial_ok=False)
        assert state is None

        # Below threshold with partial_ok=True still returns state (logs warning)
        state = parse_model_output(json_output, confidence_threshold=0.7, partial_ok=True)
        assert state is not None
        assert state.confidence == 0.5

        # Above threshold succeeds
        state = parse_model_output(json_output, confidence_threshold=0.4, partial_ok=True)
        assert state is not None


class TestFewShotExamples:
    """Test few-shot example generation."""

    def test_generate_few_shot_examples(self):
        """Generate few-shot examples."""
        examples = generate_few_shot_examples(num_examples=3)

        assert len(examples) == 3
        for example in examples:
            assert "description" in example
            assert "state" in example
            assert isinstance(example["state"], GameState)

    def test_few_shot_examples_valid(self):
        """All generated examples are valid GameStates."""
        examples = generate_few_shot_examples(num_examples=5)

        for example in examples:
            state = example["state"]
            assert state.confidence > 0.8
            assert state.floor >= 1
            assert state.player_pos[0] >= 0

    def test_few_shot_diversity(self):
        """Examples cover different game states."""
        examples = generate_few_shot_examples(num_examples=5)
        states = [ex["state"].state for ex in examples]

        # Should have some diversity
        unique_states = set(states)
        assert len(unique_states) > 1


class TestValidateGameState:
    """Test game state validation."""

    def test_validate_good_state(self):
        """Validate a high-quality state."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=0.95
        )

        report = validate_game_state(state)
        assert report["valid"] is True
        assert len(report["issues"]) == 0
        assert report["quality_score"] >= 0.95

    def test_validate_low_confidence(self):
        """Detect low confidence warning."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=0.5
        )

        report = validate_game_state(state)
        assert len(report["warnings"]) > 0
        assert "confidence" in str(report["warnings"]).lower()

    def test_validate_battle_no_enemies_warning(self):
        """Warn when battle state has no enemies."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.BATTLE,
            enemies=[]
        )

        report = validate_game_state(state)
        assert len(report["warnings"]) > 0

    def test_validate_invalid_hp(self):
        """Detect invalid HP (exceeds max)."""
        state = GameState(
            player_pos=(12, 8),
            player_hp=100,
            player_max_hp=60,
            floor=3,
            state=GameStateEnum.EXPLORING
        )

        report = validate_game_state(state)
        assert report["valid"] is False
        assert len(report["issues"]) > 0


class TestFormatStateForDecision:
    """Test formatting for decision-making."""

    def test_format_simple_state(self):
        """Format simple exploring state."""
        state = GameState(
            player_pos=(12, 8),
            player_hp=45,
            floor=3,
            state=GameStateEnum.EXPLORING
        )

        text = format_state_for_decision(state)
        assert "exploring" in text.lower()
        assert "12, 8" in text or "12" in text
        assert "3" in text  # Floor number

    def test_format_with_enemies(self):
        """Format state with enemies."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.BATTLE,
            enemies=[
                Entity(x=14, y=8, type="enemy", species="Geodude", hp=20)
            ],
            threats=["Geodude 2 tiles away"]
        )

        text = format_state_for_decision(state)
        assert "Geodude" in text
        assert "battle" in text.lower()
        assert "threat" in text.lower()

    def test_format_with_opportunities(self):
        """Format includes opportunities."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            opportunities=["Move up", "Attack"]
        )

        text = format_state_for_decision(state)
        assert "option" in text.lower() or "Move" in text


class TestUtilitiesBenchmark:
    """Performance benchmarks for utilities."""

    def test_parse_model_output_fast(self):
        """Parsing completes quickly."""
        import time

        json_output = '''{
            "player_pos": [12, 8],
            "floor": 3,
            "state": "exploring",
            "confidence": 0.95
        }'''

        start = time.time()
        for _ in range(100):
            parse_model_output(json_output)
        elapsed = time.time() - start

        # 100 parses should be fast
        assert elapsed < 0.5

    def test_validate_state_fast(self):
        """Validation completes quickly."""
        import time

        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING
        )

        start = time.time()
        for _ in range(100):
            validate_game_state(state)
        elapsed = time.time() - start

        assert elapsed < 0.5

    def test_format_state_fast(self):
        """Formatting completes quickly."""
        import time

        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            enemies=[Entity(x=14, y=8, type="enemy", species="Geodude")]
        )

        start = time.time()
        for _ in range(100):
            format_state_for_decision(state)
        elapsed = time.time() - start

        assert elapsed < 0.2
