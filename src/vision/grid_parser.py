"""Grid parser for converting minimap/memory to uniform grid representation."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import logging
from pathlib import Path

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
    
    # Extended view dimensions (960x640 pixels)
    VIEW_WIDTH_TILES = 54
    VIEW_HEIGHT_TILES = 30
    VIEW_TILE_SIZE_PX = 18  # 4x scaling (rounded)
    
    def __init__(self):
        """Initialize grid parser."""
        logger.info("GridParser initialized")
    
    def parse_ram_snapshot(self, snapshot: RAMSnapshot, _tile_map: Optional[np.ndarray] = None) -> GridFrame:
        """Parse RAM snapshot into grid frame.
        
        Args:
            snapshot: RAM snapshot
            tile_map: Optional pre-rendered tile map from memory
            
        Returns:
            GridFrame representation
        """
        try:
            # Initialize grid
            grid = self._initialize_grid()
            
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
                width=self.VIEW_WIDTH_TILES,
                height=self.VIEW_HEIGHT_TILES,
                tiles=grid,
                tile_size_px=self.VIEW_TILE_SIZE_PX,
                camera_tile_origin=(snapshot.map_data.camera_origin_x, snapshot.map_data.camera_origin_y),
                view_rect_tiles=(
                    snapshot.map_data.camera_origin_x,
                    snapshot.map_data.camera_origin_y,
                    self.VIEW_WIDTH_TILES,
                    self.VIEW_HEIGHT_TILES
                ),
                timestamp=snapshot.timestamp,
            )
            
            logger.debug("Parsed grid frame: %dx%d tiles", frame.width, frame.height)
            return frame
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to parse RAM snapshot: %s", e)
            # Return minimal grid frame
            return self._create_minimal_grid()
    
    def _initialize_grid(self) -> List[List[GridCell]]:
        """Initialize grid with floor tiles.
        
        Returns:
            2D grid of GridCell objects
        """
        grid = []
        for _y in range(self.VIEW_HEIGHT_TILES):
            row = []
            for _x in range(self.VIEW_WIDTH_TILES):
                # Determine tile type based on camera position and room flag
                # For now, assume basic floor layout
                # Future: integrate with actual minimap data
                tile_type = TileType.FLOOR
                
                row.append(GridCell(tile_type=tile_type, visible=True))
            grid.append(row)
        
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
        grid = []
        for _y in range(self.VIEW_HEIGHT_TILES):
            row = []
            for _x in range(self.VIEW_WIDTH_TILES):
                row.append(GridCell(tile_type=TileType.FLOOR, visible=False))
            grid.append(row)
        
        return GridFrame(
            width=self.VIEW_WIDTH_TILES,
            height=self.VIEW_HEIGHT_TILES,
            tiles=grid,
            tile_size_px=self.VIEW_TILE_SIZE_PX,
            camera_tile_origin=(0, 0),
            view_rect_tiles=(0, 0, self.VIEW_WIDTH_TILES, self.VIEW_HEIGHT_TILES),
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
        """Compute BFS distances from start position.
        
        Args:
            grid_frame: Grid frame
            start: Starting tile coordinates (x, y)
            
        Returns:
            BFSResult with distances and paths
        """
        width = grid_frame.width
        height = grid_frame.height
        
        # Initialize distance grid
        distances = np.full((height, width), -1, dtype=np.int32)
        paths = {}
        reachable = set()
        
        # BFS queue
        from collections import deque
        queue = deque([start])
        
        # Starting position
        distances[start[1]][start[0]] = 0
        paths[start] = [start]
        reachable.add(start)
        
        # Directions: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current[1]][current[0]]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Check bounds
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                
                # Check if already visited
                if distances[ny][nx] != -1:
                    continue
                
                # Check if tile is walkable
                if not self._is_walkable(grid_frame.tiles[ny][nx]):
                    continue
                
                # Set distance and path
                distances[ny][nx] = current_dist + 1
                paths[(nx, ny)] = paths[current] + [(nx, ny)]
                reachable.add((nx, ny))
                queue.append((nx, ny))
        
        return BFSResult(distances=distances, paths=paths, reachable=reachable)
    
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
        if distance == 1:
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
        """Export grid to JSON file.
        
        Args:
            grid_frame: Grid frame to export
            output_path: Output file path
        """
        grid_data = {
            "width": grid_frame.width,
            "height": grid_frame.height,
            "tile_size_px": grid_frame.tile_size_px,
            "camera_tile_origin": grid_frame.camera_tile_origin,
            "view_rect_tiles": grid_frame.view_rect_tiles,
            "timestamp": grid_frame.timestamp,
            "tiles": []
        }
        
        for _y, row in enumerate(grid_frame.tiles):
            row_data = []
            for _x, cell in enumerate(row):
                cell_data = {
                    "tile_type": int(cell.tile_type),
                    "visible": cell.visible,
                }
                
                if cell.entity:
                    cell_data["entity"] = {
                        "species_id": cell.entity.species_id,
                        "tile_x": cell.entity.tile_x,
                        "tile_y": cell.entity.tile_y,
                        "affiliation": int(cell.entity.affiliation),
                    }
                
                if cell.item:
                    cell_data["item"] = {
                        "item_id": cell.item.item_id,
                        "tile_x": cell.item.tile_x,
                        "tile_y": cell.item.tile_y,
                    }
                
                row_data.append(cell_data)
            
            grid_data["tiles"].append(row_data)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grid_data, f, indent=2)
        
        logger.info("Exported grid to %s", output_path)
