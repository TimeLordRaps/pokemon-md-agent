"""Tests for state mapping functionality."""

import pytest
from unittest.mock import Mock, patch

from src.environment.state_map import StateMap, StateField


class TestStateField:
    """Test StateField dataclass."""

    def test_state_field_creation(self):
        """Test basic StateField creation."""
        field = StateField("test_field", "test_value", 0.95, "test_source")

        assert field.name == "test_field"
        assert field.value == "test_value"
        assert field.confidence == 0.95
        assert field.source == "test_source"

    def test_state_field_immutable(self):
        """Test StateField is immutable (frozen dataclass)."""
        field = StateField("test", "value", 1.0, "source")

        with pytest.raises(AttributeError):
            field.name = "new_name"

        with pytest.raises(AttributeError):
            field.value = "new_value"


class TestStateMap:
    """Test StateMap functionality."""

    @pytest.fixture
    def state_map(self):
        """Create StateMap instance for testing."""
        return StateMap()

    @pytest.fixture
    def sample_ram_data(self):
        """Generate sample RAM data for testing."""
        # Create 64KB of zero data, then set some test values
        data = bytearray(65536)

        # Set player state values (using actual addresses from config)
        # Floor number = 5
        data[33544] = 5
        # Dungeon ID = 10
        data[33546] = 10
        data[33547] = 0  # Little endian high byte
        # Turn counter = 150
        data[33548] = 150
        data[33549] = 0
        # Player tile X = 10, Y = 8
        data[33550] = 10
        data[33551] = 8
        # Partner tile X = 12, Y = 10
        data[33552] = 12
        data[33553] = 10
        # Room flag = 1 (room)
        data[33554] = 1

        # Party status
        # Leader HP = 200/250
        data[33572] = 200
        data[33573] = 0  # Little endian
        data[33574] = 250
        data[33575] = 0
        # Leader belly = 75
        data[33576] = 75
        data[33577] = 0
        # Partner HP = 180/220
        data[33582] = 180
        data[33583] = 0
        data[33584] = 220
        data[33585] = 0
        # Partner belly = 80
        data[33586] = 80
        data[33587] = 0

        # Monsters (1 monster)
        # Monster count = 1
        data[33566] = 1
        # Monster pointer = 40000
        data[33562] = 128  # 40000 in little endian (128 + 156*256)
        data[33563] = 156
        data[33564] = 0
        data[33565] = 0
        # Monster data at offset 40000
        monster_offset = 40000
        # Species ID = 25 (Pikachu)
        data[monster_offset] = 25
        data[monster_offset + 1] = 0
        # Level = 15
        data[monster_offset + 2] = 15
        # HP = 50/50
        data[monster_offset + 4] = 50
        data[monster_offset + 5] = 0
        data[monster_offset + 6] = 50
        data[monster_offset + 7] = 0
        # Tile X=15, Y=12, Direction=2 (down)
        data[monster_offset + 16] = 15
        data[monster_offset + 17] = 12
        data[monster_offset + 18] = 2

        # Items (1 item - apple)
        # Item count = 1
        data[33571] = 1
        # Item pointer = 41000
        data[33567] = 232  # 41000 in little endian
        data[33568] = 160
        data[33569] = 0
        data[33570] = 0
        # Item data at offset 41000
        item_offset = 41000
        # Item ID = 120 (apple), Quantity = 3
        data[item_offset] = 120
        data[item_offset + 1] = 0
        data[item_offset + 6] = 3
        data[item_offset + 7] = 0
        # Tile X=20, Y=15
        data[item_offset + 4] = 20
        data[item_offset + 5] = 15

        # Map data - stairs at (25, 30)
        data[33556] = 25  # stairs_x
        data[33557] = 30  # stairs_y

        return bytes(data)

    def test_initialization(self, state_map):
        """Test StateMap initialization."""
        assert state_map._current_ram is None
        assert len(state_map._cached_fields) == 0
        assert hasattr(state_map, 'decoder')

    def test_update_ram(self, state_map, sample_ram_data):
        """Test RAM data updates."""
        state_map.update_ram(sample_ram_data)

        assert state_map._current_ram == sample_ram_data
        assert len(state_map._cached_fields) == 0  # Cache should be cleared

    def test_get_field_basic(self, state_map, sample_ram_data):
        """Test basic field retrieval."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("floor")
        assert field is not None
        assert field.name == "floor"
        assert field.value == 5
        assert field.confidence == 1.0
        assert field.source == "player_state.floor_number"

    def test_get_field_coords(self, state_map, sample_ram_data):
        """Test coordinate field retrieval."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("coords")
        assert field is not None
        assert field.name == "coords"
        assert field.value == {"x": 10, "y": 8}
        assert field.confidence == 1.0

    def test_get_field_health(self, state_map, sample_ram_data):
        """Test health field computation."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("health")
        assert field is not None
        assert field.name == "health"
        assert field.value["current"] == 200
        assert field.value["max"] == 250
        assert field.value["ratio"] == 0.8

    def test_get_field_inventory_highlights(self, state_map, sample_ram_data):
        """Test inventory highlights computation."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("inventory_highlights")
        assert field is not None
        assert field.name == "inventory_highlights"
        assert len(field.value) == 1  # Should highlight the apple
        assert field.value[0]["item_id"] == 120

    def test_get_field_enemies(self, state_map, sample_ram_data):
        """Test enemy detection."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("enemies_on_screen")
        assert field is not None
        assert field.name == "enemies_on_screen"
        assert len(field.value) == 1  # One enemy monster
        assert field.value[0]["species_id"] == 25

    def test_get_field_stairs_visible(self, state_map, sample_ram_data):
        """Test stairs visibility."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("stairs_visible")
        assert field is not None
        assert field.name == "stairs_visible"
        assert field.value is True  # Stairs at valid position

    def test_get_field_path_to_stairs(self, state_map, sample_ram_data):
        """Test path computation to stairs."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("path_to_stairs")
        assert field is not None
        assert field.name == "path_to_stairs"
        assert isinstance(field.value, list)
        assert field.confidence == 0.8

    def test_get_field_unknown(self, state_map, sample_ram_data):
        """Test unknown field handling."""
        state_map.update_ram(sample_ram_data)

        field = state_map.get_field("nonexistent_field")
        assert field is None

    def test_get_field_no_ram(self, state_map):
        """Test field retrieval without RAM data."""
        field = state_map.get_field("floor")
        assert field is None

    def test_get_multiple_fields(self, state_map, sample_ram_data):
        """Test batch field retrieval."""
        state_map.update_ram(sample_ram_data)

        fields = state_map.get_multiple_fields(["floor", "coords", "health"])

        assert len(fields) == 3
        assert "floor" in fields
        assert "coords" in fields
        assert "health" in fields
        assert fields["floor"].value == 5

    def test_caching(self, state_map, sample_ram_data):
        """Test field caching behavior."""
        state_map.update_ram(sample_ram_data)

        # First access should compute
        field1 = state_map.get_field("floor")
        assert len(state_map._cached_fields) == 1

        # Second access should use cache
        field2 = state_map.get_field("floor")
        assert field1 is field2  # Same object from cache

    def test_cache_invalidation(self, state_map, sample_ram_data):
        """Test cache invalidation on RAM update."""
        state_map.update_ram(sample_ram_data)

        # Populate cache
        state_map.get_field("floor")
        assert len(state_map._cached_fields) == 1

        # Update RAM should clear cache
        state_map.update_ram(sample_ram_data)
        assert len(state_map._cached_fields) == 0

    def test_clear_cache(self, state_map, sample_ram_data):
        """Test manual cache clearing."""
        state_map.update_ram(sample_ram_data)

        # Populate cache
        state_map.get_field("floor")
        assert len(state_map._cached_fields) == 1

        # Clear cache
        state_map.clear_cache()
        assert len(state_map._cached_fields) == 0

    def test_get_all_fields(self, state_map, sample_ram_data):
        """Test retrieving all available fields."""
        state_map.update_ram(sample_ram_data)

        fields = state_map.get_all_fields()

        # Should have multiple fields
        assert len(fields) > 5
        assert "floor" in fields
        assert "coords" in fields
        assert "health" in fields

    def test_bounded_path_computation(self, state_map):
        """Test bounded path computation."""
        path = state_map._compute_bounded_path(0, 0, 3, 3)

        # Should compute a reasonable path
        assert isinstance(path, list)
        assert len(path) <= 50  # Bounded

        # Path should end near target
        if path:
            final_x, final_y = path[-1]
            assert abs(final_x - 3) + abs(final_y - 3) < len(path)  # Reasonable progress

    @patch('src.environment.state_map.logger')
    def test_error_handling(self, mock_logger, state_map):
        """Test error handling in field computation."""
        # Mock decoder to raise exception
        state_map.decoder.decode_all = Mock(side_effect=Exception("Test error"))

        sample_ram = b"x" * 1000
        state_map.update_ram(sample_ram)

        field = state_map.get_field("floor")

        assert field is None
        mock_logger.error.assert_called()

    def test_key_error_handling(self, state_map):
        """Test handling of missing RAM data keys."""
        # Mock decoder to return incomplete data
        state_map.decoder.decode_all = Mock(return_value={"incomplete": "data"})

        sample_ram = b"x" * 1000
        state_map.update_ram(sample_ram)

        field = state_map.get_field("floor")

        assert field is None