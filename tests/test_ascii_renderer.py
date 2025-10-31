"""Tests for ASCII renderer functionality.

Tests verify the four deterministic modalities: full, compact, overlay, legend.
Also tests optional cell indices every 5 tiles.
"""

import pytest
from unittest.mock import Mock

from src.vision.ascii_renderer import ASCIIRenderer, ASCIIRenderOptions
from src.vision.grid_parser import GridFrame, TileType, GridCell
from src.environment.ram_decoders import RAMSnapshot, Entity, Item, MapData, PlayerState, PartyStatus


@pytest.fixture
def mock_grid_frame():
    """Create a mock grid frame for testing."""
    tiles = []
    for y in range(10):
        row = []
        for x in range(20):
            # Create a simple pattern
            if x == 5 and y == 3:
                tile_type = TileType.STAIRS
            elif x == 10 and y == 5:
                tile_type = TileType.ITEM
            elif x == 15 and y == 7:
                tile_type = TileType.MONSTER
            else:
                tile_type = TileType.FLOOR
            row.append(GridCell(tile_type=tile_type, visible=True))
        tiles.append(row)

    return GridFrame(
        width=20,
        height=10,
        tiles=tiles,
        tile_size_px=18,
        camera_tile_origin=(0, 0),
        view_rect_tiles=(0, 0, 20, 10),
        timestamp=1234567890.0
    )


@pytest.fixture
def mock_snapshot():
    """Create a mock RAM snapshot for testing."""
    map_data = MapData(
        camera_origin_x=0,
        camera_origin_y=0,
        weather_state=0,
        turn_phase=0,
        stairs_x=5,
        stairs_y=3
    )

    player_state = PlayerState(
        player_tile_x=2,
        player_tile_y=2,
        partner_tile_x=3,
        partner_tile_y=2,
        floor_number=1,
        dungeon_id=1,
        turn_counter=42
    )

    party_status = PartyStatus(
        leader_hp=50,
        leader_hp_max=100,
        leader_belly=50,
        partner_hp=60,
        partner_hp_max=100,
        partner_belly=60
    )

    entities = [
        Entity(
            species_id=1, 
            level=5, 
            hp_current=50, 
            hp_max=100, 
            status=0, 
            affiliation=1, 
            tile_x=15, 
            tile_y=7, 
            direction=0, 
            visible=True
        ),
    ]

    items = [
        Item(item_id=1, tile_x=10, tile_y=5, quantity=1),
    ]

    return RAMSnapshot(
        entities=entities,
        items=items,
        map_data=map_data,
        player_state=player_state,
        party_status=party_status,
        timestamp=1234567890.0
    )


class TestASCIIRenderer:
    """Test ASCII renderer functionality."""

    def test_render_environment_with_entities(self, mock_grid_frame, mock_snapshot):
        """Test full environment + entities rendering."""
        renderer = ASCIIRenderer()
        result = renderer.render_environment_with_entities(mock_grid_frame, mock_snapshot)

        # Should contain header, map, legend, and meta
        assert "DUNGEON:" in result
        assert "TURN:" in result
        assert "@ = Player" in result
        assert "STATUS:" in result
        assert len(result.split('\n')) > 10

    def test_render_map_only(self, mock_grid_frame):
        """Test map-only rendering."""
        renderer = ASCIIRenderer()
        result = renderer.render_map_only(mock_grid_frame)

        # Should contain header and map, but no entities or meta
        assert "MAP ONLY" in result
        assert "@ = Player" in result  # Legend is included
        assert "STATUS:" not in result

    def test_render_environment_with_grid(self, mock_grid_frame, mock_snapshot):
        """Test environment + grid overlay rendering."""
        renderer = ASCIIRenderer()
        result = renderer.render_environment_with_grid(mock_grid_frame, mock_snapshot)

        # Should show grid indices
        assert "0|" in result or "5|" in result

    def test_render_meta(self, mock_snapshot):
        """Test meta HUD rendering."""
        renderer = ASCIIRenderer()
        result = renderer.render_meta(mock_snapshot)

        # Should contain status information
        assert "HUD METADATA" in result
        assert "Player HP:" in result
        assert "Partner HP:" in result
        assert "Floor:" in result

    def test_deterministic_modalities(self, mock_grid_frame, mock_snapshot):
        """Test that all four modalities produce consistent output."""
        renderer = ASCIIRenderer()

        # Render all modalities
        full = renderer.render_environment_with_entities(mock_grid_frame, mock_snapshot)
        map_only = renderer.render_map_only(mock_grid_frame)
        overlay = renderer.render_environment_with_grid(mock_grid_frame, mock_snapshot)
        meta = renderer.render_meta(mock_snapshot)

        # All should be strings
        assert isinstance(full, str)
        assert isinstance(map_only, str)
        assert isinstance(overlay, str)
        assert isinstance(meta, str)

        # Full and overlay should be similar but overlay has indices
        assert len(full) > len(map_only)
        assert len(overlay) > len(map_only)
        assert len(meta) < len(map_only)

    def test_grid_indices_option(self):
        """Test optional grid indices every 5 tiles."""
        renderer = ASCIIRenderer(ASCIIRenderOptions(show_grid_indices=True))

        # Create larger test grid for grid indices
        tiles = [[GridCell(tile_type=TileType.FLOOR, visible=True) for _ in range(15)] for _ in range(10)]
        grid = GridFrame(
            width=15, height=10, tiles=tiles, tile_size_px=18,
            camera_tile_origin=(0, 0), view_rect_tiles=(0, 0, 15, 10), timestamp=0
        )

        result = renderer.render_map_only(grid)

        # Should contain row indices
        assert "0|" in result
        assert "5|" in result

    def test_species_codes(self, mock_grid_frame, mock_snapshot):
        """Test species code rendering."""
        renderer = ASCIIRenderer()

        # Add a Pokemon with known species code
        mock_snapshot.entities[0].species_id = 1  # Bulbasaur = "Ba"

        result = renderer.render_environment_with_entities(mock_grid_frame, mock_snapshot)

        # Should contain species code
        assert "Ba" in result

    def test_item_symbols(self, mock_grid_frame, mock_snapshot):
        """Test item symbol rendering."""
        renderer = ASCIIRenderer()

        # Set item to known type
        mock_snapshot.items[0].item_id = 1  # Stick = "S"

        result = renderer.render_environment_with_entities(mock_grid_frame, mock_snapshot)

        # Should contain item symbol
        assert "S" in result

    def test_create_multi_view_output(self, mock_grid_frame, mock_snapshot, tmp_path):
        """Test creating all four view variants."""
        renderer = ASCIIRenderer()

        paths = renderer.create_multi_view_output(mock_grid_frame, mock_snapshot, tmp_path)

        # Should create 4 files
        assert len(paths) == 4
        assert "environment" in paths
        assert "map_only" in paths
        assert "env_grid" in paths
        assert "meta" in paths

        # All files should exist
        for path in paths.values():
            assert path.exists()

    def test_options_customization(self):
        """Test renderer options customization."""
        options = ASCIIRenderOptions(
            width=120,
            height=30,
            show_grid_indices=True,
            use_species_codes=False
        )

        renderer = ASCIIRenderer(options)

        assert renderer.options.width == 120
        assert renderer.options.height == 30
        assert renderer.options.show_grid_indices == True
        assert renderer.options.use_species_codes == False

    def test_legend_rendering(self):
        """Test legend section rendering."""
        renderer = ASCIIRenderer()

        # Create minimal grid for legend test
        tiles = [[GridCell(tile_type=TileType.FLOOR, visible=True)]]
        grid = GridFrame(
            width=1, height=1, tiles=tiles, tile_size_px=18,
            camera_tile_origin=(0, 0), view_rect_tiles=(0, 0, 1, 1), timestamp=0
        )

        result = renderer.render_map_only(grid)

        # Should contain legend
        assert "LEGEND:" in result
        assert "@ = Player" in result
        assert "# = Wall" in result
        assert ". = Floor" in result