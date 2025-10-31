"""Grid parser for converting minimap/memory to uniform grid representation with (r,c) overlays."""

from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import logging
from pathlib import Path
import json
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont

from ..environment.ram_decoders import Entity, Item, RAMSnapshot

logger = logging.getLogger(__name__)


class TileType(IntEnum):
    """Tile type enumeration."""
    UNKNOWN = 0
    FLOOR = 1
    WALL = 2
    WATER = 3
    LAVA = 4
    STAIRS = 5
    ITEM = 6
    TRAP = 7
    MONSTER = 8
    SHOP = 9


@dataclass
class GridCell:
    """Single grid cell."""
    tile_type: TileType
    entity: Optional[Entity] = None
    item: Optional[Item] = None
    visible: bool = True


@dataclass
class GridFrame:
    """Complete grid representation."""
    width: int
    height: int
    tiles: List[List[GridCell]]
    tile_size_px: int
    camera_tile_origin: Tuple[int, int]
    view_rect_tiles: Tuple[int, int, int, int]  # (x, y, w, h)
    timestamp: float


@dataclass
class BFSResult:
    """Result of BFS pathfinding."""
    distances: np.ndarray  # 2D array of distances
    paths: Dict[Tuple[int, int], List[Tuple[int, int]]]  # Paths from start to each tile
    reachable: set  # Set of reachable tile coordinates


class GridParser:
    """Parses RAM/memory data into grid representation."""
    
    # Base game dimensions (240x160 pixels = 54x30 tiles at 4.44 px/tile)
    BASE_WIDTH_TILES = 54
    BASE_HEIGHT_TILES = 30
    BASE_TILE_SIZE_PX = 4.44  # Approximate pixel per tile
    
    # Tile cache configuration
    TILE_CACHE_MAX_SIZE = 1000
    
    def __init__(self, video_config=None):
        """Initialize grid parser.
        
        Args:
            video_config: Video configuration for dynamic resolution
        """
        self.video_config = video_config
        # Calculate tile size based on video config
        if video_config:
            # For scaled video, tile size scales proportionally
            self.tile_size_px = int(self.BASE_TILE_SIZE_PX * video_config.scale)
            # Grid dimensions scale with video resolution
            self.width_tiles = video_config.width // self.tile_size_px
            self.height_tiles = video_config.height // self.tile_size_px
        else:
            # Default to base game dimensions
            self.tile_size_px = int(self.BASE_TILE_SIZE_PX * 2)  # 8.88 -> 8
            self.width_tiles = self.BASE_WIDTH_TILES
            self.height_tiles = self.BASE_HEIGHT_TILES
        
        # Initialize tile cache with LRU eviction for tile properties
        self.tile_cache = OrderedDict()
        
        logger.info("GridParser initialized with %dx%d tiles, %dpx per tile, cache size %d", 
                   self.width_tiles, self.height_tiles, self.tile_size_px, self.TILE_CACHE_MAX_SIZE)
    
    def _get_cached_tile_props(self, cache_key: str) -> Optional[Tuple[TileType, bool]]:
        """Get tile properties from cache with LRU update.
        
        Args:
            cache_key: Unique key for the tile properties
            
        Returns:
            Cached (tile_type, visible) tuple or None if not found
        """
        if cache_key in self.tile_cache:
            # Move to end (most recently used)
            self.tile_cache.move_to_end(cache_key)
            return self.tile_cache[cache_key]
        return None
    
    def _set_cached_tile_props(self, cache_key: str, tile_props: Tuple[TileType, bool]) -> None:
        """Store tile properties in cache with LRU eviction.
        
        Args:
            cache_key: Unique key for the tile properties
            tile_props: (tile_type, visible) tuple to cache
        """
        if len(self.tile_cache) >= self.TILE_CACHE_MAX_SIZE:
            # Remove least recently used item
            self.tile_cache.popitem(last=False)
        
        self.tile_cache[cache_key] = tile_props
        self.tile_cache.move_to_end(cache_key)  # Mark as most recently used
    
    def parse_ram_snapshot(self, snapshot: RAMSnapshot, tile_map: Optional[np.ndarray] = None) -> GridFrame:
        """Parse RAM snapshot into grid frame.
        
        Args:
            snapshot: RAM snapshot
            tile_map: Optional pre-rendered tile map from memory (height x width array of tile types)
            
        Returns:
            GridFrame representation
        """
        try:
            # Initialize grid with base terrain from tile_map or default floor tiles
            grid = self._initialize_grid(snapshot, tile_map)
            
            # Add entities
            self._add_entities_to_grid(grid, snapshot.entities)
            
            # Add items
            self._add_items_to_grid(grid, snapshot.items)
            
            # Add stairs
            if snapshot.map_data.stairs_x >= 0 and snapshot.map_data.stairs_y >= 0:
                if (snapshot.map_data.stairs_y < len(grid) and 
                    snapshot.map_data.stairs_x < len(grid[0])):
                    grid[snapshot.map_data.stairs_y][snapshot.map_data.stairs_x].tile_type = TileType.STAIRS
            
            # Create GridFrame
            frame = GridFrame(
                width=self.width_tiles,
                height=self.height_tiles,
                tiles=grid,
                tile_size_px=self.tile_size_px,
                camera_tile_origin=(snapshot.map_data.camera_origin_x, snapshot.map_data.camera_origin_y),
                view_rect_tiles=(
                    snapshot.map_data.camera_origin_x,
                    snapshot.map_data.camera_origin_y,
                    self.width_tiles,
                    self.height_tiles
                ),
                timestamp=snapshot.timestamp,
            )
            
            logger.debug("Parsed grid frame: %dx%d tiles", frame.width, frame.height)
            return frame
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to parse RAM snapshot: %s", e)
            # Return minimal grid frame
            return self._create_minimal_grid()
    
    def _initialize_grid(self, snapshot: RAMSnapshot, tile_map: Optional[np.ndarray] = None) -> List[List[GridCell]]:
        """Initialize grid with base terrain from tile_map or default floor tiles, using LRU cache.
        
        Args:
            snapshot: RAM snapshot for cache key generation
            tile_map: Optional 2D array of tile types (height x width)
            
        Returns:
            2D grid of GridCell objects
        """
        # Vectorized grid initialization - create all cells at once
        # Use single list creation and reshape for maximum performance
        total_cells = self.height_tiles * self.width_tiles
        
        # Create all cells in a single operation (most efficient)
        # Use list comprehension to create separate objects
        cells_1d = []
        
        for y in range(self.height_tiles):
            for x in range(self.width_tiles):
                # Generate cache key based on position and snapshot timestamp
                # This allows caching tile properties across frames when terrain doesn't change
                cache_key = f"{snapshot.player_state.dungeon_id}_{snapshot.player_state.floor_number}_{y}_{x}"
                
                # Try to get cached tile properties
                cached_props = self._get_cached_tile_props(cache_key)
                
                if cached_props is not None:
                    # Use cached properties
                    tile_type, visible = cached_props
                else:
                    # Determine tile type from tile_map or default to floor
                    if tile_map is not None and y < tile_map.shape[0] and x < tile_map.shape[1]:
                        # Map tile_map values to TileType enum
                        tile_map_value = int(tile_map[y, x])
                        try:
                            tile_type = TileType(tile_map_value)
                        except ValueError:
                            # Invalid tile type, default to floor
                            tile_type = TileType.FLOOR
                    else:
                        # No tile_map provided, default to floor
                        tile_type = TileType.FLOOR
                    
                    visible = True
                    
                    # Cache the computed properties
                    self._set_cached_tile_props(cache_key, (tile_type, visible))
                
                cells_1d.append(GridCell(tile_type=tile_type, visible=visible))
        
        # Reshape into 2D grid using list slicing (faster than nested comprehensions)
        grid = []
        for y in range(self.height_tiles):
            start_idx = y * self.width_tiles
            end_idx = start_idx + self.width_tiles
            grid.append(cells_1d[start_idx:end_idx])
        
        return grid
    
    def _add_entities_to_grid(self, grid: List[List[GridCell]], entities: List[Entity]) -> None:
        """Add entities to grid.
        
        Args:
            grid: Grid to modify
            entities: List of entities
        """
        for entity in entities:
            if entity.visible and entity.tile_y < len(grid) and entity.tile_x < len(grid[0]):
                # Set tile type based on affiliation
                if entity.affiliation == 0:  # Ally
                    tile_type = TileType.MONSTER  # Use monster type for now
                else:  # Enemy or neutral
                    tile_type = TileType.MONSTER
                
                grid[entity.tile_y][entity.tile_x].tile_type = tile_type
                grid[entity.tile_y][entity.tile_x].entity = entity
    
    def _add_items_to_grid(self, grid: List[List[GridCell]], items: List[Item]) -> None:
        """Add items to grid.
        
        Args:
            grid: Grid to modify
            items: List of items
        """
        for item in items:
            if item.tile_y < len(grid) and item.tile_x < len(grid[0]):
                grid[item.tile_y][item.tile_x].tile_type = TileType.ITEM
                grid[item.tile_y][item.tile_x].item = item
    
    def _create_minimal_grid(self) -> GridFrame:
        """Create minimal grid frame for error cases.
        
        Returns:
            Minimal GridFrame
        """
        # Vectorized minimal grid creation
        total_cells = self.height_tiles * self.width_tiles
        cells_1d = [GridCell(tile_type=TileType.FLOOR, visible=False)] * total_cells
        
        # Reshape into 2D grid
        grid = []
        for y in range(self.height_tiles):
            start_idx = y * self.width_tiles
            end_idx = start_idx + self.width_tiles
            grid.append(cells_1d[start_idx:end_idx])
        
        return GridFrame(
            width=self.width_tiles,
            height=self.height_tiles,
            tiles=grid,
            tile_size_px=self.tile_size_px,
            camera_tile_origin=(0, 0),
            view_rect_tiles=(0, 0, self.width_tiles, self.height_tiles),
            timestamp=0.0,
        )
    
    def world_to_screen(self, tile_x: int, tile_y: int, grid_frame: GridFrame) -> Tuple[int, int, int, int]:
        """Convert world tile coordinates to screen pixel rectangle.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            grid_frame: Grid frame context
            
        Returns:
            Rectangle as (x, y, width, height) in pixels
        """
        # Calculate screen position
        screen_x = int((tile_x - grid_frame.camera_tile_origin[0]) * grid_frame.tile_size_px)
        screen_y = int((tile_y - grid_frame.camera_tile_origin[1]) * grid_frame.tile_size_px)
        
        # Calculate size (in pixels)
        width = int(grid_frame.tile_size_px)
        height = int(grid_frame.tile_size_px)
        
        return (screen_x, screen_y, width, height)
    
    def screen_to_world(self, x: int, y: int, grid_frame: GridFrame) -> Optional[Tuple[int, int]]:
        """Convert screen pixel coordinates to world tile coordinates.
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            grid_frame: Grid frame context
            
        Returns:
            Tile coordinates (x, y) or None if out of bounds
        """
        # Convert screen to tile coordinates
        tile_x = int(x / grid_frame.tile_size_px) + grid_frame.camera_tile_origin[0]
        tile_y = int(y / grid_frame.tile_size_px) + grid_frame.camera_tile_origin[1]
        
        # Check bounds
        if (0 <= tile_x < grid_frame.width and 
            0 <= tile_y < grid_frame.height):
            return (tile_x, tile_y)
        
        return None
    
    def compute_bfs_distances(self, grid_frame: GridFrame, start: Tuple[int, int]) -> BFSResult:
        """Compute BFS distances from start position using advanced NumPy vectorization.
        
        Args:
            grid_frame: Grid frame
            start: Starting tile coordinates (x, y)
            
        Returns:
            BFSResult with distances and paths
        """
        width = grid_frame.width
        height = grid_frame.height
        
        # Initialize distance grid with NumPy
        distances = np.full((height, width), -1, dtype=np.int32)
        paths = {}
        reachable = set()
        
        # Pre-compute walkability mask using vectorized operations
        walkable_mask = np.ones((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                walkable_mask[y, x] = self._is_walkable(grid_frame.tiles[y][x])
        
        # Use NumPy arrays for queue to enable vectorized operations
        queue = np.array([start], dtype=np.int32).reshape(1, 2)
        visited = np.zeros((height, width), dtype=bool)
        
        # Starting position
        start_y, start_x = start[1], start[0]
        distances[start_y, start_x] = 0
        paths[start] = [start]
        reachable.add(start)
        visited[start_y, start_x] = True
        
        # Directions as NumPy array for vectorized neighbor calculation
        directions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)  # up, down, left, right
        
        while len(queue) > 0:
            # Process all nodes at current level (vectorized)
            current_positions = queue.copy()
            queue = np.empty((0, 2), dtype=np.int32)  # Reset queue for next level
            
            for current in current_positions:
                current_x, current_y = int(current[0]), int(current[1])
                current_dist = distances[current_y, current_x]
                
                # Vectorized neighbor calculation
                neighbors = current + directions
                
                # Filter valid bounds using NumPy boolean indexing
                valid_bounds = (
                    (neighbors[:, 0] >= 0) & (neighbors[:, 0] < height) &
                    (neighbors[:, 1] >= 0) & (neighbors[:, 1] < width)
                )
                valid_neighbors = neighbors[valid_bounds]
                
                # Filter unvisited and walkable neighbors using vectorized operations
                for ny, nx in valid_neighbors:
                    if not visited[ny, nx] and walkable_mask[ny, nx]:
                        visited[ny, nx] = True
                        distances[ny, nx] = current_dist + 1
                        new_pos = (nx, ny)
                        paths[new_pos] = paths[(current_x, current_y)] + [new_pos]
                        reachable.add(new_pos)
                        
                        # Add to next level queue
                        queue = np.vstack([queue, np.array([[nx, ny]], dtype=np.int32)])
        
        return BFSResult(distances=distances, paths=paths, reachable=reachable)
    
    def serialize_grid_for_memory(self, grid_frame: GridFrame) -> Dict[str, Any]:
        """Serialize grid frame for memory manager storage.
        
        Args:
            grid_frame: Grid frame to serialize
            
        Returns:
            Serialized grid data as dictionary
        """
        # Create compact representation focusing on non-floor tiles
        tiles_data = []
        
        for r, row in enumerate(grid_frame.tiles):
            for c, cell in enumerate(row):
                # Only serialize non-default tiles to save space
                if (cell.tile_type != TileType.FLOOR or 
                    cell.entity is not None or 
                    cell.item is not None or 
                    not cell.visible):
                    
                    tile_dict = {
                        "r": r,
                        "c": c,
                        "type": int(cell.tile_type),
                        "visible": cell.visible
                    }
                    
                    if cell.entity:
                        tile_dict["entity"] = {
                            "species_id": cell.entity.species_id,
                            "level": cell.entity.level,
                            "hp": cell.entity.hp_current,
                            "max_hp": cell.entity.hp_max,
                            "status": cell.entity.status,
                            "affiliation": cell.entity.affiliation,
                            "direction": cell.entity.direction
                        }
                    
                    if cell.item:
                        tile_dict["item"] = {
                            "id": cell.item.item_id,
                            "quantity": cell.item.quantity
                        }
                    
                    tiles_data.append(tile_dict)
        
        return {
            "version": "1.0",
            "timestamp": grid_frame.timestamp,
            "dimensions": {
                "width": grid_frame.width,
                "height": grid_frame.height,
                "tile_size_px": grid_frame.tile_size_px
            },
            "camera": {
                "origin": grid_frame.camera_tile_origin,
                "view_rect": grid_frame.view_rect_tiles
            },
            "tiles": tiles_data,
            "stats": {
                "total_tiles": grid_frame.width * grid_frame.height,
                "serialized_tiles": len(tiles_data),
                "compression_ratio": len(tiles_data) / (grid_frame.width * grid_frame.height)
            }
        }
    
    def deserialize_grid_from_memory(self, grid_data: Dict[str, Any]) -> GridFrame:
        """Deserialize grid frame from memory manager data.
        
        Args:
            grid_data: Serialized grid data
            
        Returns:
            Reconstructed GridFrame
        """
        # Initialize empty grid
        width = grid_data["dimensions"]["width"]
        height = grid_data["dimensions"]["height"]
        tile_size_px = grid_data["dimensions"]["tile_size_px"]
        
        # Create base grid with floor tiles
        grid = [
            [GridCell(tile_type=TileType.FLOOR, visible=True) 
             for _ in range(width)]
            for _ in range(height)
        ]
        
        # Apply serialized tile data
        for tile_dict in grid_data["tiles"]:
            r, c = tile_dict["r"], tile_dict["c"]
            if 0 <= r < height and 0 <= c < width:
                cell = grid[r][c]
                cell.tile_type = TileType(tile_dict["type"])
                cell.visible = tile_dict["visible"]
                
                if "entity" in tile_dict:
                    entity_data = tile_dict["entity"]
                    cell.entity = Entity(
                        species_id=entity_data["species_id"],
                        level=entity_data["level"],
                        hp_current=entity_data["hp"],
                        hp_max=entity_data["max_hp"],
                        status=entity_data["status"],
                        tile_x=c,
                        tile_y=r,
                        affiliation=entity_data["affiliation"],
                        direction=entity_data["direction"],
                        visible=True
                    )
                
                if "item" in tile_dict:
                    item_data = tile_dict["item"]
                    cell.item = Item(
                        item_id=item_data["id"],
                        tile_x=c,
                        tile_y=r,
                        quantity=item_data["quantity"]
                    )
        
        return GridFrame(
            width=width,
            height=height,
            tiles=grid,
            tile_size_px=tile_size_px,
            camera_tile_origin=tuple(grid_data["camera"]["origin"]),
            view_rect_tiles=tuple(grid_data["camera"]["view_rect"]),
            timestamp=grid_data["timestamp"]
        )
    
    def _is_walkable(self, cell: GridCell) -> bool:
        """Check if a cell is walkable.
        
        Args:
            cell: Grid cell to check
            
        Returns:
            True if walkable
        """
        # Not walkable if wall, water, lava, or occupied by monster
        if cell.tile_type in [TileType.WALL, TileType.WATER, TileType.LAVA]:
            return False
        
        # Check if occupied by monster
        if cell.entity and cell.entity.affiliation != 0:  # Enemy monster
            return False
        
        return True
    
    def get_distance_bucket(self, distance: int) -> str:
        """Get distance bucket for classification.

        Args:
            distance: Distance in tiles

        Returns:
            Distance bucket string
        """
        if distance <= 1:
            return "adjacent"
        elif distance == 2:
            return "near"
        elif distance <= 5:
            return "close"
        elif distance <= 10:
            return "medium"
        else:
            return "far"
    
    def get_path_to_tile(self, grid_frame: GridFrame, start: Tuple[int, int], 
                        target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Get path from start to target.
        
        Args:
            grid_frame: Grid frame
            start: Starting coordinates
            target: Target coordinates
            
        Returns:
            List of coordinates forming the path, or None if no path
        """
        bfs_result = self.compute_bfs_distances(grid_frame, start)
        
        if target in bfs_result.paths:
            return bfs_result.paths[target]
        
        return None
    
    def export_grid_json(self, grid_frame: GridFrame, output_path: Path) -> None:
        """Export grid to JSON file with (r,c) coordinates for overlay rendering.

        Args:
            grid_frame: Grid frame to export
            output_path: Output file path
        """
        grid_data = {
            "metadata": {
                "width": grid_frame.width,
                "height": grid_frame.height,
                "tile_size_px": grid_frame.tile_size_px,
                "camera_tile_origin": grid_frame.camera_tile_origin,
                "view_rect_tiles": grid_frame.view_rect_tiles,
                "timestamp": grid_frame.timestamp,
                "format_version": "1.0",
                "coordinate_system": "row_column",  # (r,c) coordinates for overlays
            },
            "tiles": []
        }

        for r, row in enumerate(grid_frame.tiles):
            for c, cell in enumerate(row):
                # Only export non-empty tiles to keep JSON size reasonable
                if cell.tile_type != TileType.FLOOR or cell.entity or cell.item:
                    tile_data = {
                        "r": r,  # Row coordinate (0-based)
                        "c": c,  # Column coordinate (0-based)
                        "tile_type": int(cell.tile_type),
                        "visible": cell.visible,
                    }

                    if cell.entity:
                        tile_data["entity"] = {
                            "species_id": cell.entity.species_id,
                            "level": cell.entity.level,
                            "hp_current": cell.entity.hp_current,
                            "hp_max": cell.entity.hp_max,
                            "status": cell.entity.status,
                            "affiliation": cell.entity.affiliation,
                            "direction": cell.entity.direction,
                        }

                    if cell.item:
                        tile_data["item"] = {
                            "item_id": cell.item.item_id,
                            "quantity": cell.item.quantity,
                        }

                    grid_data["tiles"].append(tile_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grid_data, f, indent=2)

        logger.info("Exported grid overlay to %s with %d tiles", output_path, len(grid_data["tiles"]))

    def generate_overlay_image(self, grid_frame: GridFrame, base_image: Optional[Image.Image] = None) -> Image.Image:
        """Generate PIL overlay image with grid lines and (r,c) labels.

        Args:
            grid_frame: Grid frame to overlay
            base_image: Optional base image to overlay on (480x320 expected)

        Returns:
            PIL Image with grid overlay
        """
        # Use base image or create blank canvas
        if base_image:
            overlay = base_image.copy()
        else:
            # Create blank 480x320 image
            overlay = Image.new('RGBA', (480, 320), (0, 0, 0, 0))

        draw = ImageDraw.Draw(overlay, 'RGBA')

        # Grid parameters
        tile_size = grid_frame.tile_size_px
        width_tiles = grid_frame.width
        height_tiles = grid_frame.height

        # Colors
        grid_color = (255, 255, 255, 128)  # Semi-transparent white
        label_bg = (0, 0, 0, 160)  # Semi-transparent black
        label_color = (255, 255, 255, 255)  # White text

        # Try to load a small font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 8)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Draw vertical grid lines
        for c in range(width_tiles + 1):
            x = c * tile_size
            draw.line([(x, 0), (x, 320)], fill=grid_color, width=1)

        # Draw horizontal grid lines
        for r in range(height_tiles + 1):
            y = r * tile_size
            draw.line([(0, y), (480, y)], fill=grid_color, width=1)

        # Draw (r,c) labels for each tile
        for r in range(height_tiles):
            for c in range(width_tiles):
                # Label position (top-left of tile)
                label_x = c * tile_size + 2
                label_y = r * tile_size + 2

                # Background rectangle for label
                bbox = draw.textbbox((label_x, label_y), f"({r},{c})", font=font)
                draw.rectangle(bbox, fill=label_bg)

                # Draw label text
                draw.text((label_x, label_y), f"({r},{c})", fill=label_color, font=font)

        logger.info("Generated grid overlay image: %dx%d with %dx%d tiles",
                   overlay.width, overlay.height, width_tiles, height_tiles)
        return overlay

    def export_overlay_metadata(self, grid_frame: GridFrame, overlay_image: Image.Image, output_path: Path) -> None:
        """Export overlay metadata as JSON.

        Args:
            grid_frame: Grid frame
            overlay_image: Generated overlay image
            output_path: Output JSON path
        """
        metadata = {
            "metadata": {
                "width_px": overlay_image.width,
                "height_px": overlay_image.height,
                "tile_size_px": grid_frame.tile_size_px,
                "grid_width_tiles": grid_frame.width,
                "grid_height_tiles": grid_frame.height,
                "camera_tile_origin": grid_frame.camera_tile_origin,
                "view_rect_tiles": grid_frame.view_rect_tiles,
                "timestamp": grid_frame.timestamp,
                "format_version": "1.0",
                "overlay_type": "grid_with_labels",
            },
            "grid_coordinates": []
        }

        # Add coordinate mapping for each tile
        for r in range(grid_frame.height):
            for c in range(grid_frame.width):
                tile_info = {
                    "r": r,
                    "c": c,
                    "pixel_bbox": [
                        c * grid_frame.tile_size_px,
                        r * grid_frame.tile_size_px,
                        (c + 1) * grid_frame.tile_size_px,
                        (r + 1) * grid_frame.tile_size_px
                    ],
                    "label_position": [
                        c * grid_frame.tile_size_px + 2,
                        r * grid_frame.tile_size_px + 2
                    ]
                }
                metadata["grid_coordinates"].append(tile_info)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Exported overlay metadata to %s", output_path)
