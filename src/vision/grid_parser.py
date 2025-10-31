"""Grid parser for converting minimap/memory to uniform grid representation with (r,c) overlays."""

from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import logging
from pathlib import Path
import json
from collections import OrderedDict
import warnings

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
        # Input guards for video_config
        if video_config is not None:
            if not hasattr(video_config, 'scale') or not hasattr(video_config, 'width') or not hasattr(video_config, 'height'):
                logger.warning("Video config missing required attributes (scale, width, height), using defaults")
                video_config = None
            elif video_config.scale <= 0:
                logger.warning("Video config scale must be positive, got %f, using defaults", video_config.scale)
                video_config = None
            elif video_config.width <= 0 or video_config.height <= 0:
                logger.warning("Video config dimensions must be positive, got %dx%d, using defaults",
                             video_config.width, video_config.height)
                video_config = None

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

        # Input guards for dimensions
        if self.width_tiles <= 0:
            raise ValueError(f"Grid width must be positive, got {self.width_tiles}")
        if self.height_tiles <= 0:
            raise ValueError(f"Grid height must be positive, got {self.height_tiles}")
        if self.tile_size_px <= 0:
            raise ValueError(f"Tile size must be positive, got {self.tile_size_px}")

        # Maximum reasonable dimensions to prevent memory issues
        max_dimension = 1000
        if self.width_tiles > max_dimension or self.height_tiles > max_dimension:
            raise ValueError(f"Grid dimensions too large: {self.width_tiles}x{self.height_tiles}, max {max_dimension}")

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
        # Input guards for snapshot validity
        if snapshot is None:
            raise ValueError("RAM snapshot cannot be None")
        if not hasattr(snapshot, 'entities') or not hasattr(snapshot, 'items') or not hasattr(snapshot, 'map_data'):
            raise ValueError("RAM snapshot missing required attributes (entities, items, map_data)")
        if not hasattr(snapshot, 'timestamp') or not isinstance(snapshot.timestamp, (int, float)):
            raise ValueError("RAM snapshot timestamp must be numeric")
        if snapshot.timestamp < 0:
            logger.warning("RAM snapshot timestamp is negative: %f", snapshot.timestamp)

        # Guards for tile_map
        if tile_map is not None:
            if not isinstance(tile_map, np.ndarray):
                raise ValueError("tile_map must be a numpy array")
            if tile_map.ndim != 2:
                raise ValueError(f"tile_map must be 2D, got {tile_map.ndim}D")
            # Allow empty tile_map but log warning
            if tile_map.size == 0:
                logger.warning("tile_map is empty")

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
        # Invariant asserts
        assert self.width_tiles > 0 and self.height_tiles > 0, f"Invalid grid dimensions: {self.width_tiles}x{self.height_tiles}"
        assert self.tile_size_px > 0, f"Invalid tile size: {self.tile_size_px}"

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
                    # Assert cached values are valid
                    assert isinstance(tile_type, TileType), f"Invalid cached tile_type: {tile_type}"
                    assert isinstance(visible, bool), f"Invalid cached visible: {visible}"
                else:
                    # Determine tile type from tile_map or default to floor
                    if tile_map is not None and y < tile_map.shape[0] and x < tile_map.shape[1]:
                        # Map tile_map values to TileType enum
                        tile_map_value = int(tile_map[y, x])
                        try:
                            tile_type = TileType(tile_map_value)
                        except ValueError:
                            # Invalid tile type, default to floor
                            logger.warning("Invalid tile type %d at (%d,%d), defaulting to FLOOR", tile_map_value, x, y)
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
            row = cells_1d[start_idx:end_idx]
            assert len(row) == self.width_tiles, f"Row {y} has incorrect length: {len(row)} != {self.width_tiles}"
            grid.append(row)

        # Final invariant check
        assert len(grid) == self.height_tiles, f"Grid has incorrect height: {len(grid)} != {self.height_tiles}"
        assert all(len(row) == self.width_tiles for row in grid), "All grid rows must have correct width"

        return grid
    
    def _add_entities_to_grid(self, grid: List[List[GridCell]], entities: List[Entity]) -> None:
        """Add entities to grid.

        Args:
            grid: Grid to modify
            entities: List of entities
        """
        # Input guards
        if entities is None:
            entities = []
        if not isinstance(entities, list):
            raise ValueError(f"entities must be a list, got {type(entities)}")

        for entity in entities:
            # Guards for entity validity
            if entity is None:
                logger.warning("Encountered None entity in entities list")
                continue
            if not hasattr(entity, 'tile_y') or not hasattr(entity, 'tile_x') or not hasattr(entity, 'visible'):
                logger.warning("Entity missing required attributes (tile_y, tile_x, visible)")
                continue
            if not isinstance(entity.tile_y, int) or not isinstance(entity.tile_x, int):
                logger.warning("Entity coordinates must be integers: (%s, %s)", entity.tile_y, entity.tile_x)
                continue

            # Invariant asserts
            assert len(grid) > 0 and len(grid[0]) > 0, "Grid must be properly initialized"
            assert 0 <= entity.tile_y < len(grid), f"Entity tile_y {entity.tile_y} out of grid bounds [0, {len(grid)})"
            assert 0 <= entity.tile_x < len(grid[0]), f"Entity tile_x {entity.tile_x} out of grid bounds [0, {len(grid[0])})"

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
        # Input guards
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError(f"items must be a list, got {type(items)}")

        for item in items:
            # Guards for item validity
            if item is None:
                logger.warning("Encountered None item in items list")
                continue
            if not hasattr(item, 'tile_y') or not hasattr(item, 'tile_x'):
                logger.warning("Item missing required attributes (tile_y, tile_x)")
                continue
            if not isinstance(item.tile_y, int) or not isinstance(item.tile_x, int):
                logger.warning("Item coordinates must be integers: (%s, %s)", item.tile_y, item.tile_x)
                continue

            # Invariant asserts
            assert len(grid) > 0 and len(grid[0]) > 0, "Grid must be properly initialized"
            assert 0 <= item.tile_y < len(grid), f"Item tile_y {item.tile_y} out of grid bounds [0, {len(grid)})"
            assert 0 <= item.tile_x < len(grid[0]), f"Item tile_x {item.tile_x} out of grid bounds [0, {len(grid[0])})"

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
        # Input guards
        if grid_frame is None:
            raise ValueError("grid_frame cannot be None")
        if start is None or len(start) != 2:
            raise ValueError(f"start must be a tuple of (x, y), got {start}")
        if not isinstance(start[0], int) or not isinstance(start[1], int):
            raise ValueError(f"start coordinates must be integers, got {start}")

        width = grid_frame.width
        height = grid_frame.height

        # Invariant asserts
        assert width > 0 and height > 0, f"Invalid grid dimensions: {width}x{height}"
        assert 0 <= start[0] < width and 0 <= start[1] < height, f"Start position {start} out of bounds [0,{width})x[0,{height})"
        assert len(grid_frame.tiles) == height, f"Grid height mismatch: {len(grid_frame.tiles)} != {height}"
        assert all(len(row) == width for row in grid_frame.tiles), "Grid rows have inconsistent width"

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
                # Invariant asserts during BFS
                assert len(queue) > 0, "Queue should not be empty during BFS traversal"
                assert current_dist >= 0, f"Distance should be non-negative, got {current_dist}"
                assert (current_x, current_y) in paths, f"Current position {(current_x, current_y)} should have a path"

                for ny, nx in valid_neighbors:
                    if not visited[ny, nx] and walkable_mask[ny, nx]:
                        visited[ny, nx] = True
                        distances[ny, nx] = current_dist + 1
                        new_pos = (nx, ny)
                        paths[new_pos] = paths[(current_x, current_y)] + [new_pos]
                        reachable.add(new_pos)

                        # Add to next level queue
                        queue = np.vstack([queue, np.array([[nx, ny]], dtype=np.int32)])

        # Post-condition asserts
        assert distances[start_y, start_x] == 0, "Start position must have distance 0"
        assert start in paths, "Start position must be in paths"
        assert start in reachable, "Start position must be reachable"
        # Verify distances are non-negative for reachable tiles
        for pos in reachable:
            px, py = pos
            assert distances[py, px] >= 0, f"Reachable position {pos} has negative distance {distances[py, px]}"

        return BFSResult(distances=distances, paths=paths, reachable=reachable)
    
    def serialize_grid_for_memory(self, grid_frame: GridFrame) -> Dict[str, Any]:
        """Serialize grid frame for memory manager storage.

        Args:
            grid_frame: Grid frame to serialize

        Returns:
            Serialized grid data as dictionary
        """
        # Input guards
        if grid_frame is None:
            raise ValueError("grid_frame cannot be None")
        if not hasattr(grid_frame, 'tiles') or not hasattr(grid_frame, 'width') or not hasattr(grid_frame, 'height'):
            raise ValueError("grid_frame missing required attributes (tiles, width, height)")

        # Invariant asserts
        assert grid_frame.width > 0 and grid_frame.height > 0, f"Invalid grid dimensions: {grid_frame.width}x{grid_frame.height}"
        assert len(grid_frame.tiles) == grid_frame.height, f"Grid height mismatch: {len(grid_frame.tiles)} != {grid_frame.height}"
        assert all(len(row) == grid_frame.width for row in grid_frame.tiles), "Grid rows have inconsistent width"

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

        serialized = {
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

        # Post-condition assert
        assert "version" in serialized and "tiles" in serialized, "Serialization must include version and tiles"

        return serialized
    
    def deserialize_grid_from_memory(self, grid_data: Dict[str, Any]) -> GridFrame:
        """Deserialize grid frame from memory manager data.

        Args:
            grid_data: Serialized grid data

        Returns:
            Reconstructed GridFrame
        """
        # Input guards
        if grid_data is None:
            raise ValueError("grid_data cannot be None")
        if not isinstance(grid_data, dict):
            raise ValueError("grid_data must be a dictionary")

        required_keys = ["dimensions", "camera", "tiles", "timestamp"]
        for key in required_keys:
            if key not in grid_data:
                raise ValueError(f"grid_data missing required key: {key}")

        # Extract dimensions with guards
        dimensions = grid_data["dimensions"]
        if not isinstance(dimensions, dict):
            raise ValueError("dimensions must be a dictionary")
        width = dimensions.get("width")
        height = dimensions.get("height")
        tile_size_px = dimensions.get("tile_size_px")

        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be positive integer, got {width}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be positive integer, got {height}")
        if not isinstance(tile_size_px, int) or tile_size_px <= 0:
            raise ValueError(f"tile_size_px must be positive integer, got {tile_size_px}")

        # Invariant asserts
        assert width <= 1000 and height <= 1000, f"Grid dimensions too large: {width}x{height}"

        # Initialize empty grid
        grid = [
            [GridCell(tile_type=TileType.FLOOR, visible=True)
             for _ in range(width)]
            for _ in range(height)
        ]

        # Apply serialized tile data with guards
        tiles_data = grid_data["tiles"]
        if not isinstance(tiles_data, list):
            raise ValueError("tiles must be a list")

        for tile_dict in tiles_data:
            if not isinstance(tile_dict, dict):
                logger.warning("Skipping non-dict tile data")
                continue

            r = tile_dict.get("r")
            c = tile_dict.get("c")
            if r is None or c is None:
                logger.warning("Skipping tile data missing r/c coordinates")
                continue

            if not isinstance(r, int) or not isinstance(c, int):
                logger.warning("Skipping tile data with non-integer coordinates")
                continue

            if not (0 <= r < height and 0 <= c < width):
                logger.warning(f"Skipping tile at out-of-bounds coordinates ({r},{c})")
                continue

            cell = grid[r][c]

            # Apply tile type with guards
            tile_type_val = tile_dict.get("type", 1)  # Default to FLOOR
            try:
                cell.tile_type = TileType(tile_type_val)
            except ValueError:
                logger.warning(f"Invalid tile type {tile_type_val}, defaulting to FLOOR")
                cell.tile_type = TileType.FLOOR

            cell.visible = tile_dict.get("visible", True)

            # Apply entity data with guards
            if "entity" in tile_dict:
                entity_data = tile_dict["entity"]
                if isinstance(entity_data, dict):
                    try:
                        cell.entity = Entity(
                            species_id=entity_data.get("species_id", 0),
                            level=entity_data.get("level", 1),
                            hp_current=entity_data.get("hp", 0),
                            hp_max=entity_data.get("max_hp", 0),
                            status=entity_data.get("status", 0),
                            tile_x=c,
                            tile_y=r,
                            affiliation=entity_data.get("affiliation", 0),
                            direction=entity_data.get("direction", 0),
                            visible=True
                        )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to create entity from data: {e}")

            # Apply item data with guards
            if "item" in tile_dict:
                item_data = tile_dict["item"]
                if isinstance(item_data, dict):
                    try:
                        cell.item = Item(
                            item_id=item_data.get("id", 0),
                            tile_x=c,
                            tile_y=r,
                            quantity=item_data.get("quantity", 1)
                        )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to create item from data: {e}")

        # Create GridFrame with final asserts
        camera_data = grid_data["camera"]
        if not isinstance(camera_data, dict):
            raise ValueError("camera must be a dictionary")

        origin = camera_data.get("origin")
        view_rect = camera_data.get("view_rect")
        if origin is None or view_rect is None:
            raise ValueError("camera missing origin or view_rect")

        result = GridFrame(
            width=width,
            height=height,
            tiles=grid,
            tile_size_px=tile_size_px,
            camera_tile_origin=tuple(origin),
            view_rect_tiles=tuple(view_rect),
            timestamp=grid_data["timestamp"]
        )

        # Post-condition asserts
        assert len(result.tiles) == result.height, f"Deserialized grid height mismatch: {len(result.tiles)} != {result.height}"
        assert all(len(row) == result.width for row in result.tiles), "Deserialized grid rows have inconsistent width"

        return result

    def test_roundtrip_serialization(self, grid_frame: GridFrame) -> bool:
        """Test that serialization/deserialization maintains grid equivalence.

        Args:
            grid_frame: Grid frame to test roundtrip

        Returns:
            True if roundtrip preserves grid state
        """
        try:
            # Serialize
            serialized = self.serialize_grid_for_memory(grid_frame)

            # Deserialize
            deserialized = self.deserialize_grid_from_memory(serialized)

            # Check equivalence
            return self._grids_equivalent(grid_frame, deserialized)
        except Exception as e:
            logger.error(f"Roundtrip test failed: {e}")
            return False

    def _grids_equivalent(self, grid1: GridFrame, grid2: GridFrame) -> bool:
        """Check if two grid frames are equivalent.

        Args:
            grid1: First grid frame
            grid2: Second grid frame

        Returns:
            True if grids are equivalent
        """
        # Basic dimension checks
        if (grid1.width != grid2.width or
            grid1.height != grid2.height or
            grid1.tile_size_px != grid2.tile_size_px or
            grid1.camera_tile_origin != grid2.camera_tile_origin or
            grid1.view_rect_tiles != grid2.view_rect_tiles):
            return False

        # Check each cell
        for r in range(grid1.height):
            for c in range(grid1.width):
                cell1 = grid1.tiles[r][c]
                cell2 = grid2.tiles[r][c]

                # Compare tile types and visibility
                if cell1.tile_type != cell2.tile_type or cell1.visible != cell2.visible:
                    return False

                # Compare entities
                if (cell1.entity is None) != (cell2.entity is None):
                    return False
                if cell1.entity is not None and cell2.entity is not None:
                    e1, e2 = cell1.entity, cell2.entity
                    if (e1.species_id != e2.species_id or
                        e1.level != e2.level or
                        e1.hp_current != e2.hp_current or
                        e1.hp_max != e2.hp_max or
                        e1.status != e2.status or
                        e1.affiliation != e2.affiliation or
                        e1.direction != e2.direction):
                        return False

                # Compare items
                if (cell1.item is None) != (cell2.item is None):
                    return False
                if cell1.item is not None and cell2.item is not None:
                    i1, i2 = cell1.item, cell2.item
                    if i1.item_id != i2.item_id or i1.quantity != i2.quantity:
                        return False

        return True
    
    def _is_walkable(self, cell: GridCell) -> bool:
        """Check if a cell is walkable.

        Args:
            cell: Grid cell to check

        Returns:
            True if walkable
        """
        # Input guards
        if cell is None:
            return False
        if not isinstance(cell, GridCell):
            logger.warning(f"Expected GridCell, got {type(cell)}")
            return False

        # Invariant asserts
        assert isinstance(cell.tile_type, TileType), f"Invalid tile_type: {cell.tile_type}"

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
        # Input guards
        if grid_frame is None:
            raise ValueError("grid_frame cannot be None")
        if start is None or target is None:
            raise ValueError("start and target cannot be None")
        if len(start) != 2 or len(target) != 2:
            raise ValueError("start and target must be (x, y) tuples")
        if not all(isinstance(coord, int) for coord in start + target):
            raise ValueError("All coordinates must be integers")

        bfs_result = self.compute_bfs_distances(grid_frame, start)

        # Invariant assert
        assert bfs_result is not None, "BFS computation should always return a result"

        if target in bfs_result.paths:
            path = bfs_result.paths[target]
            # Additional invariant check
            assert path[0] == start and path[-1] == target, f"Path should start at {start} and end at {target}"
            return path

        return None
    
    def export_grid_json(self, grid_frame: GridFrame, output_path: Path) -> None:
        """Export grid to JSON file with (r,c) coordinates for overlay rendering.

        Args:
            grid_frame: Grid frame to export
            output_path: Output file path
        """
        # Input guards
        if grid_frame is None:
            raise ValueError("grid_frame cannot be None")
        if output_path is None:
            raise ValueError("output_path cannot be None")

        # Invariant asserts
        assert grid_frame.width > 0 and grid_frame.height > 0, f"Invalid grid dimensions: {grid_frame.width}x{grid_frame.height}"
        assert len(grid_frame.tiles) == grid_frame.height, f"Grid height mismatch: {len(grid_frame.tiles)} != {grid_frame.height}"

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

        tile_count = 0
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
                    tile_count += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grid_data, f, indent=2)

        # Post-condition assert
        assert len(grid_data["tiles"]) == tile_count, "Tile count mismatch in exported data"

        logger.info("Exported grid overlay to %s with %d tiles", output_path, tile_count)

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
