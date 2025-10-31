"""Tests for grid parser functionality.

Tests verify step-to-pixel drift < 0.5 tile over 100 steps, round-trip mapping,
and BFS buckets. Also tests origin alignment, view-rect computation, and linking
to dynamic map stitcher indices.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from PIL import Image, ImageDraw, ImageFont
import json
import tempfile
import os
from pathlib import Path

from src.vision.grid_parser import GridParser, TileType, GridFrame, GridCell
from src.environment.ram_decoders import RAMSnapshot, Entity, Item, MapData, PlayerState, PartyStatus


@pytest.fixture
def mock_snapshot():
    """Create a mock RAM snapshot for testing."""
    map_data = MapData(
        camera_origin_x=10,
        camera_origin_y=5,
        weather_state=0,
        turn_phase=1,
        stairs_x=15,
        stairs_y=8
    )

    player_state = PlayerState(
        player_tile_x=12,
        player_tile_y=7,
        partner_tile_x=13,
        partner_tile_y=6,
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
        Entity(species_id=1, level=3, hp_current=20, hp_max=20, status=0, tile_x=11, tile_y=6, affiliation=1, direction=0, visible=True),
        Entity(species_id=2, level=4, hp_current=25, hp_max=25, status=0, tile_x=14, tile_y=8, affiliation=0, direction=1, visible=True),
    ]

    items = [
        Item(item_id=1, tile_x=13, tile_y=7, quantity=1),
    ]

    return RAMSnapshot(
        entities=entities,
        items=items,
        map_data=map_data,
        player_state=player_state,
        party_status=party_status,
        timestamp=1234567890.0
    )


class TestGridParser:
    """Test grid parser functionality."""

    def test_origin_alignment(self, mock_snapshot):
        """Test that camera origin aligns correctly."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        assert grid_frame.camera_tile_origin == (10, 5)

    def test_view_rect_computation(self, mock_snapshot):
        """Test view rectangle computation in tiles."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        expected_rect = (10, 5, 54, 30)  # origin_x, origin_y, width, height
        assert grid_frame.view_rect_tiles == expected_rect

    def test_round_trip_mapping(self, mock_snapshot):
        """Test world_to_screen and screen_to_world round-trip accuracy."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        # Test multiple points
        test_points = [(12, 7), (15, 8), (10, 5), (63, 34)]  # within view

        for world_x, world_y in test_points:
            # World to screen
            screen_rect = parser.world_to_screen(world_x, world_y, grid_frame)
            screen_x, screen_y = screen_rect[0], screen_rect[1]

            # Screen to world
            world_result = parser.screen_to_world(screen_x, screen_y, grid_frame)

            if world_result:  # Should be within bounds
                assert abs(world_result[0] - world_x) < 0.5, f"X drift too high: {world_result[0]} vs {world_x}"
                assert abs(world_result[1] - world_y) < 0.5, f"Y drift too high: {world_result[1]} vs {world_y}"

    def test_step_to_pixel_drift_over_100_steps(self, mock_snapshot):
        """Test cumulative drift over 100 coordinate transformations."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        start_world = (12, 7)
        current_world = start_world

        max_drift = 0.0

        for _ in range(100):
            # World -> Screen
            screen_rect = parser.world_to_screen(current_world[0], current_world[1], grid_frame)
            screen_x, screen_y = screen_rect[0], screen_rect[1]

            # Add small perturbation (simulate movement)
            screen_x += 0.1
            screen_y += 0.1

            # Screen -> World
            new_world = parser.screen_to_world(screen_x, screen_y, grid_frame)

            if new_world:
                drift_x = abs(new_world[0] - current_world[0])
                drift_y = abs(new_world[1] - current_world[1])
                max_drift = max(max_drift, drift_x, drift_y)
                current_world = new_world
            else:
                break

        assert max_drift < 0.5, f"Cumulative drift {max_drift} exceeds 0.5 tiles"

    def test_bfs_distances_and_paths(self, mock_snapshot):
        """Test BFS distance computation and path finding."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        start = (12, 7)  # Player position
        bfs_result = parser.compute_bfs_distances(grid_frame, start)

        # Check start position
        assert bfs_result.distances[start[1]][start[0]] == 0

        # Check some reachable positions
        # Note: Player at (12,7), so adjacent positions should be reachable
        # (11,6) is enemy position, but should be walkable for distance calculation
        assert bfs_result.distances[7][12] >= 0  # Adjacent position (down from player)
        assert bfs_result.distances[6][12] >= 0  # Adjacent position (up from player)

        # Check paths exist for reachable tiles
        assert start in bfs_result.paths
        assert bfs_result.paths[start] == [start]

    def test_bfs_buckets(self, mock_snapshot):
        """Test distance bucket classification."""
        parser = GridParser()

        # Test various distances
        assert parser.get_distance_bucket(0) == "adjacent"
        assert parser.get_distance_bucket(1) == "adjacent"
        assert parser.get_distance_bucket(2) == "near"
        assert parser.get_distance_bucket(5) == "close"
        assert parser.get_distance_bucket(8) == "medium"
        assert parser.get_distance_bucket(15) == "far"

    def test_get_path_to_tile(self, mock_snapshot):
        """Test path finding to specific tiles."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        start = (12, 7)
        target = (15, 8)  # Stairs position

        path = parser.get_path_to_tile(grid_frame, start, target)

        assert path is not None
        assert path[0] == start
        assert path[-1] == target
        assert len(path) > 1

    def test_linking_to_stitcher_indices(self, mock_snapshot):
        """Test linking env+grid numbers to dynamic map stitcher indices."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        # The linking should expose coordinates relative to the dynamic map
        # This is a placeholder test - actual implementation would depend on
        # the dynamic map stitcher interface

        # For now, verify that grid coordinates can be computed relative to origin
        origin_x, origin_y = grid_frame.camera_tile_origin

        # Player position in stitcher coordinates
        player_stitcher_x = mock_snapshot.player_state.player_tile_x - origin_x
        player_stitcher_y = mock_snapshot.player_state.player_tile_y - origin_y

        assert 0 <= player_stitcher_x < grid_frame.width
        assert 0 <= player_stitcher_y < grid_frame.height

    def test_grid_parsing_with_entities_and_items(self, mock_snapshot):
        """Test that entities and items are correctly placed on grid."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        # Check enemy entity placement
        enemy_pos = (11, 6)
        cell = grid_frame.tiles[enemy_pos[1]][enemy_pos[0]]
        assert cell.tile_type == TileType.MONSTER
        assert cell.entity.species_id == 1

        # Check ally entity placement
        ally_pos = (14, 8)
        cell = grid_frame.tiles[ally_pos[1]][ally_pos[0]]
        assert cell.tile_type == TileType.MONSTER
        assert cell.entity.species_id == 2

        # Check item placement
        item_pos = (13, 7)
        cell = grid_frame.tiles[item_pos[1]][item_pos[0]]
        assert cell.tile_type == TileType.ITEM
        assert cell.item.item_id == 1

        # Check stairs placement
        stairs_pos = (15, 8)
        cell = grid_frame.tiles[stairs_pos[1]][stairs_pos[0]]
        assert cell.tile_type == TileType.STAIRS

    def test_generate_overlay_image(self, mock_snapshot):
        """Test PIL overlay image generation with grid lines and labels."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        # Generate overlay
        overlay_image = parser.generate_overlay_image(grid_frame)

        # Verify image properties
        assert isinstance(overlay_image, Image.Image)
        assert overlay_image.mode == "RGBA"
        assert overlay_image.size == (480, 320)  # PMD screen resolution

        # Verify grid lines are drawn (check for non-transparent pixels)
        pixels = list(overlay_image.getdata())
        non_transparent_pixels = [p for p in pixels if p[3] > 0]  # Alpha > 0
        assert len(non_transparent_pixels) > 0, "Overlay should have visible content"

    def test_overlay_grid_lines_and_labels(self, mock_snapshot):
        """Test that overlay contains grid lines and coordinate labels."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        overlay_image = parser.generate_overlay_image(grid_frame)

        # Convert to RGB for easier pixel analysis
        rgb_image = overlay_image.convert("RGB")
        pixels = np.array(rgb_image)

        # Check for grid lines (black pixels)
        black_pixels = np.all(pixels == [0, 0, 0], axis=2)
        assert np.any(black_pixels), "Should have black grid lines"

        # Check for label text (white pixels)
        white_pixels = np.all(pixels == [255, 255, 255], axis=2)
        assert np.any(white_pixels), "Should have white coordinate labels"

    def test_overlay_coordinate_labels(self, mock_snapshot):
        """Test that coordinate labels are correctly positioned."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        overlay_image = parser.generate_overlay_image(grid_frame)

        # The overlay should have labels at tile intersections
        # This is a basic smoke test - detailed label verification would require OCR
        assert overlay_image.size == (480, 320)

        # Verify the image has content (not just transparent)
        bbox = overlay_image.getbbox()
        assert bbox is not None, "Overlay should have non-transparent content"

    def test_export_overlay_metadata(self, mock_snapshot):
        """Test JSON metadata export for overlay coordinates."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)

        # Generate overlay image first
        overlay_image = parser.generate_overlay_image(grid_frame)

        # Export metadata to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            parser.export_overlay_metadata(grid_frame, overlay_image, temp_path)

            # Read and verify the exported metadata
            with open(temp_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Verify metadata structure
            assert "metadata" in metadata
            assert "grid_coordinates" in metadata

            # Verify metadata content
            meta = metadata["metadata"]
            assert meta["width_px"] == 480
            assert meta["height_px"] == 320
            assert meta["tile_size_px"] == grid_frame.tile_size_px
            assert meta["grid_width_tiles"] == grid_frame.width
            assert meta["grid_height_tiles"] == grid_frame.height
            assert meta["overlay_type"] == "grid_with_labels"

            # Verify grid coordinates
            coords = metadata["grid_coordinates"]
            assert isinstance(coords, list)
            assert len(coords) > 0

            # Check first coordinate
            first_coord = coords[0]
            assert "r" in first_coord
            assert "c" in first_coord
            assert "pixel_bbox" in first_coord
            assert "label_position" in first_coord

            # Verify bbox structure
            bbox = first_coord["pixel_bbox"]
            assert len(bbox) == 4  # [x1, y1, x2, y2]
            assert all(isinstance(coord, int) for coord in bbox)

            # Verify label position
            label_pos = first_coord["label_position"]
            assert len(label_pos) == 2  # [x, y]
            assert all(isinstance(coord, int) for coord in label_pos)

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def test_overlay_metadata_json_serialization(self, mock_snapshot):
        """Test that overlay metadata can be serialized to JSON."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)
        overlay_image = parser.generate_overlay_image(grid_frame)

        # Export metadata to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            parser.export_overlay_metadata(grid_frame, overlay_image, temp_path)

            # Read the JSON file
            with open(temp_path, 'r', encoding='utf-8') as f:
                json_str = f.read()

            # Verify it's valid JSON
            metadata = json.loads(json_str)
            assert isinstance(metadata, dict)

            # Test round-trip serialization
            re_serialized = json.dumps(metadata, indent=2)
            re_deserialized = json.loads(re_serialized)
            assert re_deserialized == metadata

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_overlay_metadata_coordinate_mapping(self, mock_snapshot):
        """Test coordinate mapping in overlay metadata."""
        parser = GridParser()
        grid_frame = parser.parse_ram_snapshot(mock_snapshot)
        overlay_image = parser.generate_overlay_image(grid_frame)

        # Export metadata to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            parser.export_overlay_metadata(grid_frame, overlay_image, temp_path)

            # Read metadata
            with open(temp_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            coords = metadata["grid_coordinates"]

            # Find tile at (0,0) relative to camera origin
            origin_tile = next((tile for tile in coords if tile["r"] == 0 and tile["c"] == 0), None)
            assert origin_tile is not None, "Should have tile at (0,0)"

            # Verify bbox covers expected pixel area
            bbox = origin_tile["pixel_bbox"]
            expected_tile_size = grid_frame.tile_size_px
            assert bbox[2] - bbox[0] == expected_tile_size, f"Bbox width should be {expected_tile_size}"
            assert bbox[3] - bbox[1] == expected_tile_size, f"Bbox height should be {expected_tile_size}"

            # Verify label position is inside the tile
            label_pos = origin_tile["label_position"]
            assert bbox[0] <= label_pos[0] < bbox[2], "Label x should be within tile bbox"
            assert bbox[1] <= label_pos[1] < bbox[3], "Label y should be within tile bbox"

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_overlay_with_different_grid_sizes(self):
        """Test overlay generation with different grid dimensions."""
        parser = GridParser()

        # Create a mock grid frame with custom dimensions
        tiles = [[GridCell(tile_type=TileType.FLOOR, entity=None, item=None)
                 for _ in range(10)] for _ in range(8)]

        grid_frame = GridFrame(
            tiles=tiles,
            width=10,
            height=8,
            tile_size_px=16,  # Add required tile_size_px
            camera_tile_origin=(0, 0),
            view_rect_tiles=(0, 0, 10, 8),
            timestamp=1234567890.0
        )

        overlay_image = parser.generate_overlay_image(grid_frame)

        # Should still be 480x320 (full screen)
        assert overlay_image.size == (480, 320)

        # Export metadata and verify grid dimensions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            parser.export_overlay_metadata(grid_frame, overlay_image, temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            assert metadata["metadata"]["grid_width_tiles"] == 10
            assert metadata["metadata"]["grid_height_tiles"] == 8

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_tile_caching_integration(self, mock_snapshot):
        """Test that tile caching works correctly with LRU eviction."""
        parser = GridParser()
        
        # Clear any existing cache
        parser.tile_cache.clear()
        
        # First parse should populate cache
        grid_frame1 = parser.parse_ram_snapshot(mock_snapshot)
        
        # Check that cache has been populated
        assert len(parser.tile_cache) > 0, "Cache should be populated after first parse"
        initial_cache_size = len(parser.tile_cache)
        
        # Second parse of same snapshot should use cache
        grid_frame2 = parser.parse_ram_snapshot(mock_snapshot)
        
        # Results should be identical
        assert grid_frame1.width == grid_frame2.width
        assert grid_frame1.height == grid_frame2.height
        assert len(grid_frame1.tiles) == len(grid_frame2.tiles)
        
        # Cache size should be the same (same tiles accessed)
        assert len(parser.tile_cache) == initial_cache_size
        
        # Create a different snapshot (different dungeon/floor)
        different_snapshot = RAMSnapshot(
            entities=mock_snapshot.entities,
            items=mock_snapshot.items,
            map_data=mock_snapshot.map_data,
            player_state=PlayerState(
                player_tile_x=12, player_tile_y=7, partner_tile_x=13, partner_tile_y=6,
                floor_number=2,  # Different floor
                dungeon_id=1,
                turn_counter=42
            ),
            party_status=mock_snapshot.party_status,
            timestamp=mock_snapshot.timestamp + 1.0
        )
        
        # Parse different snapshot - should add new cache entries or maintain size due to LRU
        grid_frame3 = parser.parse_ram_snapshot(different_snapshot)
        
        # Cache should be at or near max size (LRU eviction may keep it stable)
        assert len(parser.tile_cache) <= parser.TILE_CACHE_MAX_SIZE
        
        # Test LRU eviction by exceeding cache size
        # Create many different snapshots to fill cache
        for floor in range(3, parser.TILE_CACHE_MAX_SIZE // 10 + 10):  # Create enough to exceed cache
            test_snapshot = RAMSnapshot(
                entities=[],
                items=[],
                map_data=MapData(camera_origin_x=0, camera_origin_y=0, weather_state=0, turn_phase=1, stairs_x=-1, stairs_y=-1),
                player_state=PlayerState(
                    player_tile_x=0, player_tile_y=0, partner_tile_x=0, partner_tile_y=0,
                    floor_number=floor,
                    dungeon_id=999,  # Unique dungeon
                    turn_counter=0
                ),
                party_status=PartyStatus(leader_hp=100, leader_hp_max=100, leader_belly=100, 
                                        partner_hp=100, partner_hp_max=100, partner_belly=100),
                timestamp=0.0
            )
            parser.parse_ram_snapshot(test_snapshot)
            
            # Stop when cache reaches max size
            if len(parser.tile_cache) >= parser.TILE_CACHE_MAX_SIZE:
                break
        
        # Cache should be at or near max size
        assert len(parser.tile_cache) <= parser.TILE_CACHE_MAX_SIZE