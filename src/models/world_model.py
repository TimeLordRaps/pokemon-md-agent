"""World model for Pokemon Mystery Dungeon.

Maintains dual world model: dynamic floor model + global hub model.
Tracks entities, items, connections, and exploration state.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import IntEnum

logger = logging.getLogger(__name__)


class TileType(IntEnum):
    """Types of tiles in the dungeon."""
    FLOOR = 0
    WALL = 1
    WATER = 2
    LAVA = 3
    VOID = 4
    STAIRS_UP = 5
    STAIRS_DOWN = 6
    SHOP = 7
    TREASURE = 8
    TRAP = 9


class EntityType(IntEnum):
    """Types of entities."""
    PLAYER = 0
    MONSTER = 1
    ITEM = 2
    NPC = 3


@dataclass
class Position:
    """A position in the world."""
    x: int
    y: int
    floor: int
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position."""
        if self.floor != other.floor:
            return float('inf')
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def __hash__(self):
        return hash((self.x, self.y, self.floor))


@dataclass
class Entity:
    """An entity in the world."""
    id: int
    type: EntityType
    position: Position
    species_id: Optional[int] = None
    level: Optional[int] = None
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    status: Optional[str] = None
    item_id: Optional[int] = None
    is_hostile: bool = False
    last_seen: float = 0.0


@dataclass
class FloorTile:
    """A tile on a dungeon floor."""
    position: Position
    type: TileType
    entities: List[Entity] = field(default_factory=list)
    explored: bool = False
    reachable: bool = False
    last_updated: float = 0.0


@dataclass
class FloorModel:
    """Model of a single dungeon floor."""
    floor_number: int
    dungeon_id: int
    width: int
    height: int
    tiles: Dict[Tuple[int, int], FloorTile] = field(default_factory=dict)
    entities: Dict[int, Entity] = field(default_factory=dict)
    stairs_up: Optional[Position] = None
    stairs_down: Optional[Position] = None
    shops: List[Position] = field(default_factory=list)
    treasures: List[Position] = field(default_factory=list)
    traps: List[Position] = field(default_factory=list)
    explored_ratio: float = 0.0
    last_updated: float = 0.0
    
    def get_tile(self, x: int, y: int) -> Optional[FloorTile]:
        """Get tile at position."""
        return self.tiles.get((x, y))
    
    def set_tile(self, tile: FloorTile) -> None:
        """Set tile at position."""
        self.tiles[(tile.position.x, tile.position.y)] = tile
        self.last_updated = tile.last_updated
    
    def update_explored_ratio(self) -> None:
        """Update the explored ratio."""
        if not self.tiles:
            self.explored_ratio = 0.0
            return
        
        explored_count = sum(1 for tile in self.tiles.values() if tile.explored)
        self.explored_ratio = explored_count / len(self.tiles)
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find a path from start to end using BFS."""
        if start not in self.tiles or end not in self.tiles:
            return None
        
        # Simple BFS for reachable tiles
        from collections import deque
        
        queue = deque([(start, [])])
        visited = set([start])
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == end:
                return path + [(x, y)]
            
            # Check adjacent tiles
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.tiles and (nx, ny) not in visited:
                    tile = self.tiles[(nx, ny)]
                    if tile.type not in [TileType.WALL, TileType.VOID] and tile.reachable:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(x, y)]))
        
        return None


@dataclass
class HubConnection:
    """Connection between hubs."""
    from_hub: str
    to_hub: str
    distance: int  # In some unit (steps, time, etc.)
    requirements: List[str] = field(default_factory=list)  # Items needed, etc.
    last_traversed: float = 0.0


@dataclass
class GlobalHub:
    """A global hub location."""
    name: str
    position: Optional[Position] = None  # May not have coordinates
    connections: List[HubConnection] = field(default_factory=list)
    features: List[str] = field(default_factory=list)  # "shop", "healing", etc.
    visited_count: int = 0
    last_visited: float = 0.0


class WorldModel:
    """Dual world model: dynamic floors + global hubs."""
    
    def __init__(self, save_dir: Path):
        """Initialize world model.
        
        Args:
            save_dir: Directory to save world model state
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic floor models (current dungeon)
        self.floor_models: Dict[int, FloorModel] = {}
        self.current_floor: Optional[int] = None
        self.current_dungeon_id: Optional[int] = None
        
        # Global hub model
        self.hubs: Dict[str, GlobalHub] = {}
        self.current_hub: Optional[str] = None
        
        # Entity tracking
        self.global_entities: Dict[int, Entity] = {}
        
        # Load saved state
        self._load_state()
        
        logger.info("WorldModel initialized with %d floors, %d hubs", 
                   len(self.floor_models), len(self.hubs))
    
    def update_floor(self, floor_model: FloorModel) -> None:
        """Update a floor model.
        
        Args:
            floor_model: Updated floor model
        """
        self.floor_models[floor_model.floor_number] = floor_model
        self.current_floor = floor_model.floor_number
        self.current_dungeon_id = floor_model.dungeon_id
        
        logger.debug("Updated floor %d in dungeon %d", 
                    floor_model.floor_number, floor_model.dungeon_id)
    
    def get_current_floor(self) -> Optional[FloorModel]:
        """Get the current floor model."""
        if self.current_floor is None:
            return None
        return self.floor_models.get(self.current_floor)
    
    def update_entity(self, entity: Entity) -> None:
        """Update an entity in the current floor and global tracking.
        
        Args:
            entity: Entity to update
        """
        # Update in current floor
        floor = self.get_current_floor()
        if floor:
            floor.entities[entity.id] = entity
        
        # Update global tracking
        self.global_entities[entity.id] = entity
        
        logger.debug("Updated entity %d at (%d, %d)", 
                    entity.id, entity.position.x, entity.position.y)
    
    def remove_entity(self, entity_id: int) -> None:
        """Remove an entity.
        
        Args:
            entity_id: ID of entity to remove
        """
        # Remove from current floor
        floor = self.get_current_floor()
        if floor and entity_id in floor.entities:
            del floor.entities[entity_id]
        
        # Remove from global tracking
        if entity_id in self.global_entities:
            del self.global_entities[entity_id]
        
        logger.debug("Removed entity %d", entity_id)
    
    def update_hub(self, hub: GlobalHub) -> None:
        """Update a global hub.
        
        Args:
            hub: Hub to update
        """
        self.hubs[hub.name] = hub
        logger.debug("Updated hub %s", hub.name)
    
    def set_current_hub(self, hub_name: str) -> None:
        """Set the current hub.
        
        Args:
            hub_name: Name of current hub
        """
        if hub_name in self.hubs:
            self.current_hub = hub_name
            self.hubs[hub_name].visited_count += 1
            self.hubs[hub_name].last_visited = time.time()
            logger.debug("Set current hub to %s", hub_name)
    
    def find_nearest_entity(self, position: Position, entity_type: EntityType,
                           max_distance: float = 10.0) -> Optional[Entity]:
        """Find nearest entity of given type.
        
        Args:
            position: Reference position
            entity_type: Type of entity to find
            max_distance: Maximum search distance
            
        Returns:
            Nearest entity or None
        """
        nearest = None
        min_distance = float('inf')
        
        for entity in self.global_entities.values():
            if entity.type == entity_type and entity.position.floor == position.floor:
                distance = position.distance_to(entity.position)
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest = entity
        
        return nearest
    
    def get_explorable_tiles(self) -> List[FloorTile]:
        """Get unexplored but reachable tiles in current floor.
        
        Returns:
            List of explorable tiles
        """
        floor = self.get_current_floor()
        if not floor:
            return []
        
        return [tile for tile in floor.tiles.values() 
                if not tile.explored and tile.reachable]
    
    def get_hostile_entities(self) -> List[Entity]:
        """Get hostile entities in current floor.
        
        Returns:
            List of hostile entities
        """
        floor = self.get_current_floor()
        if not floor:
            return []
        
        return [entity for entity in floor.entities.values() if entity.is_hostile]
    
    def get_items_in_floor(self) -> List[Entity]:
        """Get items in current floor.
        
        Returns:
            List of item entities
        """
        floor = self.get_current_floor()
        if not floor:
            return []
        
        return [entity for entity in floor.entities.values() 
                if entity.type == EntityType.ITEM]
    
    def save_state(self) -> None:
        """Save world model state to disk."""
        state = {
            "floor_models": {k: asdict(v) for k, v in self.floor_models.items()},
            "hubs": {k: asdict(v) for k, v in self.hubs.items()},
            "global_entities": {k: asdict(v) for k, v in self.global_entities.items()},
            "current_floor": self.current_floor,
            "current_dungeon_id": self.current_dungeon_id,
            "current_hub": self.current_hub,
        }
        
        save_path = self.save_dir / "world_model.json"
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("Saved world model to %s", save_path)
        except (OSError, ValueError) as e:
            logger.error("Failed to save world model: %s", e)
    
    def _load_state(self) -> None:
        """Load world model state from disk."""
        save_path = self.save_dir / "world_model.json"
        if not save_path.exists():
            return
        
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Reconstruct objects
            self.floor_models = {}
            for k, v in state.get("floor_models", {}).items():
                # Convert tiles back
                tiles = {}
                for pos_str, tile_dict in v.get("tiles", {}).items():
                    x, y = map(int, pos_str.strip("()").split(", "))
                    tile = FloorTile(**tile_dict)
                    tiles[(x, y)] = tile
                v["tiles"] = tiles
                
                # Convert positions
                for pos_field in ["stairs_up", "stairs_down"]:
                    if v.get(pos_field):
                        pos_dict = v[pos_field]
                        v[pos_field] = Position(**pos_dict)
                
                self.floor_models[int(k)] = FloorModel(**v)
            
            self.hubs = {k: GlobalHub(**v) for k, v in state.get("hubs", {}).items()}
            self.global_entities = {int(k): Entity(**v) for k, v in state.get("global_entities", {}).items()}
            self.current_floor = state.get("current_floor")
            self.current_dungeon_id = state.get("current_dungeon_id")
            self.current_hub = state.get("current_hub")
            
            logger.debug("Loaded world model from %s", save_path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to load world model: %s", e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics.
        
        Returns:
            Dictionary with stats
        """
        floor = self.get_current_floor()
        
        return {
            "current_floor": self.current_floor,
            "current_dungeon": self.current_dungeon_id,
            "current_hub": self.current_hub,
            "floors_modeled": len(self.floor_models),
            "hubs_known": len(self.hubs),
            "entities_tracked": len(self.global_entities),
            "floor_explored_ratio": floor.explored_ratio if floor else 0.0,
            "hostile_entities": len(self.get_hostile_entities()) if floor else 0,
            "items_available": len(self.get_items_in_floor()) if floor else 0,
        }