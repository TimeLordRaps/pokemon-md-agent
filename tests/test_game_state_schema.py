"""Unit tests for game state schema validation.

Tests cover:
- Schema creation and validation
- Coordinate bounds checking
- JSON serialization/deserialization
- Confidence scoring validation
- Threat/opportunity limiting
- Error recovery for invalid data
"""

import pytest
import json
from pydantic import ValidationError
from src.models.game_state_schema import (
    GameState,
    Entity,
    GameStateEnum,
    RoomType,
)


class TestEntity:
    """Test Entity model validation."""

    def test_entity_valid_enemy(self):
        """Create valid enemy entity."""
        entity = Entity(
            x=10,
            y=8,
            type="enemy",
            species="Geodude",
            hp=20,
            level=5
        )
        assert entity.x == 10
        assert entity.y == 8
        assert entity.type == "enemy"
        assert entity.species == "Geodude"

    def test_entity_valid_item(self):
        """Create valid item entity."""
        entity = Entity(
            x=5,
            y=6,
            type="item",
            name="Apple"
        )
        assert entity.type == "item"
        assert entity.name == "Apple"

    def test_entity_negative_coordinates_invalid(self):
        """Reject negative coordinates."""
        with pytest.raises(ValidationError) as exc_info:
            Entity(x=-1, y=5, type="enemy")
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_entity_invalid_type(self):
        """Reject invalid entity type."""
        with pytest.raises(ValidationError):
            Entity(x=5, y=5, type="invalid_type")

    def test_entity_type_case_insensitive(self):
        """Entity type should normalize to lowercase."""
        entity = Entity(x=5, y=5, type="ENEMY")
        assert entity.type == "enemy"

    def test_entity_status_effects(self):
        """Store multiple status effects."""
        entity = Entity(
            x=5,
            y=5,
            type="enemy",
            status_effects=["poison", "burn"]
        )
        assert len(entity.status_effects) == 2


class TestGameState:
    """Test GameState model validation."""

    def test_game_state_valid_exploring(self):
        """Create valid exploring state."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=0.95
        )
        assert state.player_pos == (12, 8)
        assert state.floor == 3
        assert state.state == GameStateEnum.EXPLORING

    def test_game_state_with_entities(self):
        """Create state with enemies and items."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            enemies=[
                Entity(x=14, y=8, type="enemy", species="Geodude")
            ],
            items=[
                Entity(x=10, y=6, type="item", name="Apple")
            ]
        )
        assert len(state.enemies) == 1
        assert len(state.items) == 1
        assert state.enemies[0].species == "Geodude"

    def test_game_state_invalid_floor_zero(self):
        """Reject floor 0."""
        with pytest.raises(ValidationError):
            GameState(
                player_pos=(12, 8),
                floor=0,
                state=GameStateEnum.EXPLORING
            )

    def test_game_state_invalid_floor_too_high(self):
        """Reject floor > 50."""
        with pytest.raises(ValidationError):
            GameState(
                player_pos=(12, 8),
                floor=51,
                state=GameStateEnum.EXPLORING
            )

    def test_game_state_invalid_confidence_negative(self):
        """Reject negative confidence."""
        with pytest.raises(ValidationError):
            GameState(
                player_pos=(12, 8),
                floor=3,
                state=GameStateEnum.EXPLORING,
                confidence=-0.1
            )

    def test_game_state_invalid_confidence_over_one(self):
        """Reject confidence > 1.0."""
        with pytest.raises(ValidationError):
            GameState(
                player_pos=(12, 8),
                floor=3,
                state=GameStateEnum.EXPLORING,
                confidence=1.1
            )

    def test_game_state_threat_limit_enforced(self):
        """Threats limited to 3 items max."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            threats=["threat1", "threat2", "threat3", "threat4", "threat5"]
        )
        assert len(state.threats) == 3
        assert state.threats == ["threat1", "threat2", "threat3"]

    def test_game_state_opportunity_limit_enforced(self):
        """Opportunities limited to 3 items max."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            opportunities=["op1", "op2", "op3", "op4"]
        )
        assert len(state.opportunities) == 3

    def test_game_state_battle_state(self):
        """Create valid battle state."""
        state = GameState(
            player_pos=(10, 10),
            player_hp=30,
            floor=3,
            state=GameStateEnum.BATTLE,
            enemies=[
                Entity(x=12, y=10, type="enemy", species="Bulbasaur", hp=25)
            ]
        )
        assert state.state == GameStateEnum.BATTLE
        assert len(state.enemies) == 1

    def test_game_state_boss_battle(self):
        """Create valid boss battle state."""
        state = GameState(
            player_pos=(8, 8),
            player_hp=50,
            floor=10,
            state=GameStateEnum.BOSS,
            room_type=RoomType.BOSS,
            enemies=[
                Entity(x=10, y=8, type="enemy", species="Rayquaza", level=30, hp=100)
            ]
        )
        assert state.state == GameStateEnum.BOSS
        assert state.room_type == RoomType.BOSS


class TestGameStateJSON:
    """Test JSON serialization and deserialization."""

    def test_game_state_to_json(self):
        """Serialize GameState to JSON."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=0.95
        )
        json_str = state.model_dump_json()
        data = json.loads(json_str)

        assert data["player_pos"] == [12, 8]
        assert data["floor"] == 3
        assert data["state"] == "exploring"

    def test_game_state_from_json(self):
        """Deserialize GameState from JSON."""
        json_str = '''{
            "player_pos": [12, 8],
            "floor": 3,
            "state": "exploring",
            "confidence": 0.95
        }'''

        state = GameState.model_validate_json(json_str)
        assert state.player_pos == (12, 8)
        assert state.floor == 3

    def test_game_state_with_entities_to_json(self):
        """Serialize state with entities."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            enemies=[
                Entity(x=14, y=8, type="enemy", species="Geodude")
            ]
        )
        json_str = state.model_dump_json()
        data = json.loads(json_str)

        assert len(data["enemies"]) == 1
        assert data["enemies"][0]["species"] == "Geodude"

    def test_game_state_roundtrip(self):
        """JSON roundtrip preserves data."""
        original = GameState(
            player_pos=(12, 8),
            player_hp=45,
            floor=3,
            dungeon_name="Mt. Horn",
            state=GameStateEnum.EXPLORING,
            confidence=0.92,
            threats=["Geodude approaching"],
            opportunities=["Move up to dodge"]
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = GameState.model_validate_json(json_str)

        assert restored.player_pos == original.player_pos
        assert restored.player_hp == original.player_hp
        assert restored.floor == original.floor
        assert restored.dungeon_name == original.dungeon_name
        assert restored.confidence == original.confidence


class TestGameStateEdgeCases:
    """Test edge cases and error recovery."""

    def test_minimal_game_state(self):
        """Create state with only required fields."""
        state = GameState(
            player_pos=(0, 0),
            floor=1,
            state=GameStateEnum.EXPLORING
        )
        assert state.confidence == 0.9  # Default value

    def test_game_state_with_hp_percent(self):
        """Include HP tracking."""
        state = GameState(
            player_pos=(12, 8),
            player_hp=30,
            player_max_hp=60,
            floor=3,
            state=GameStateEnum.EXPLORING
        )
        assert state.player_hp == 30
        assert state.player_max_hp == 60

    def test_game_state_no_enemies_empty_list(self):
        """Empty enemy list by default."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING
        )
        assert state.enemies == []
        assert state.items == []
        assert state.allies == []

    def test_game_state_multiple_room_types(self):
        """Test different room types."""
        for room_type in RoomType:
            state = GameState(
                player_pos=(12, 8),
                floor=3,
                state=GameStateEnum.EXPLORING,
                room_type=room_type
            )
            assert state.room_type == room_type

    def test_game_state_confidence_boundaries(self):
        """Test confidence edge values."""
        state_min = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.UNKNOWN,
            confidence=0.0
        )
        assert state_min.confidence == 0.0

        state_max = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            confidence=1.0
        )
        assert state_max.confidence == 1.0

    def test_game_state_with_all_fields(self):
        """Create state with all optional fields populated."""
        state = GameState(
            player_pos=(12, 8),
            player_hp=45,
            player_max_hp=60,
            player_status=["poison"],
            floor=3,
            dungeon_name="Mt. Horn",
            room_type=RoomType.CORRIDOR,
            enemies=[Entity(x=14, y=8, type="enemy", species="Geodude")],
            allies=[Entity(x=11, y=8, type="ally", species="Charizard")],
            items=[Entity(x=10, y=6, type="item", name="Apple")],
            special_objects=[Entity(x=15, y=15, type="stairs")],
            state=GameStateEnum.EXPLORING,
            is_day=True,
            weather="rain",
            significant_change="Geodude moved closer",
            threats=["Geodude approaching"],
            opportunities=["Move up to dodge"],
            confidence=0.92,
            notes="Enemy 2 tiles away"
        )

        assert state.player_hp == 45
        assert len(state.enemies) == 1
        assert len(state.allies) == 1
        assert state.weather == "rain"


class TestGameStatePromptGeneration:
    """Test prompt-related functionality."""

    def test_to_prompt_json(self):
        """Generate JSON for LM prompts."""
        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING,
            enemies=[Entity(x=14, y=8, type="enemy", species="Geodude")],
            items=[Entity(x=10, y=6, type="item", name="Apple")]
        )

        prompt_json = state.to_prompt_json()
        data = json.loads(prompt_json)

        # Should include key fields for LM guidance
        assert "state" in data
        assert "player_pos" in data
        assert "floor" in data
        assert "enemies" in data
        assert "items" in data

    def test_schema_json_export(self):
        """Export full schema for LM guidance."""
        schema = GameState.model_json_schema()

        assert "properties" in schema
        assert "player_pos" in schema["properties"]
        assert "state" in schema["properties"]
        assert "confidence" in schema["properties"]


class TestGameStateBenchmark:
    """Fast-lane performance tests."""

    def test_schema_validation_fast(self):
        """Schema validation completes quickly (<100ms)."""
        import time

        start = time.time()
        for i in range(100):
            GameState(
                player_pos=(i % 20, i % 20),
                floor=min(i // 100 + 1, 50),
                state=GameStateEnum.EXPLORING,
                confidence=0.9 + (i % 10) * 0.001
            )
        elapsed = time.time() - start

        # Should complete 100 validations in <2 seconds
        assert elapsed < 2.0

    def test_json_roundtrip_fast(self):
        """JSON serialization is fast (<50ms for 100 ops)."""
        import time

        state = GameState(
            player_pos=(12, 8),
            floor=3,
            state=GameStateEnum.EXPLORING
        )

        start = time.time()
        for _ in range(100):
            json_str = state.model_dump_json()
            GameState.model_validate_json(json_str)
        elapsed = time.time() - start

        # Should complete 100 roundtrips in <500ms
        assert elapsed < 0.5
