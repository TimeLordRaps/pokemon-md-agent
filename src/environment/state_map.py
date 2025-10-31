"""State mapping layer - coalesced reads to semantic fields.

Maps low-level RAM decodes to semantic fields that skills can read.
Provides read-only view with bounded path computation.
Ensures models see only semantic representations, never raw memory addresses.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .ram_decoders import create_decoder, PMDRedDecoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateField:
    """Semantic state field with confidence scoring.

    Represents a single piece of game state that has been coalesced from
    raw RAM data into a meaningful semantic field.

    Attributes:
        name: Human-readable field identifier
        value: The semantic value (never raw bytes or addresses)
        confidence: Float between 0.0-1.0 indicating data reliability
        source: Description of where this field was derived from
    """
    name: str
    value: Any
    confidence: float  # 0.0 to 1.0
    source: str  # RAM address or computed


class StateMap:
    """Maps RAM data to semantic state fields with caching and bounded computation.

    This class provides a read-only interface to game state, coalescing multiple
    RAM reads into meaningful semantic fields like position, health, inventory, etc.
    All computations are bounded to prevent excessive resource usage.

    The mapping ensures runtime integrity by providing semantic representations
    without exposing raw memory addresses or write access to the models.
    """

    def __init__(self):
        """Initialize state mapper with decoder."""
        self.decoder = create_decoder()
        self._current_ram: Optional[bytes] = None
        self._cached_fields: Dict[str, StateField] = {}
        logger.info("Initialized StateMap with %s decoder", type(self.decoder).__name__)

    def update_ram(self, ram_data: bytes) -> None:
        """Update RAM data and invalidate cached fields.

        Args:
            ram_data: Raw RAM bytes from the emulator
        """
        self._current_ram = ram_data
        self._cached_fields.clear()
        logger.debug("Updated RAM data (%d bytes), cleared %d cached fields",
                    len(ram_data), len(self._cached_fields))

    def get_field(self, field_name: str) -> Optional[StateField]:
        """Get semantic field by name, computing if needed.

        Uses caching to avoid recomputation on subsequent calls.

        Args:
            field_name: Name of the semantic field to retrieve

        Returns:
            StateField if available, None if field unknown or no RAM data
        """
        if field_name in self._cached_fields:
            logger.debug("Returning cached field: %s", field_name)
            return self._cached_fields[field_name]

        if self._current_ram is None:
            logger.warning("No RAM data available for field: %s", field_name)
            return None

        # Compute field based on name
        field = self._compute_field(field_name)
        if field:
            self._cached_fields[field_name] = field
            logger.debug("Computed and cached field: %s", field_name)
        else:
            logger.warning("Failed to compute field: %s", field_name)
        return field

    def get_multiple_fields(self, field_names: List[str]) -> Dict[str, StateField]:
        """Get multiple fields efficiently with batch processing.

        Args:
            field_names: List of field names to retrieve

        Returns:
            Dictionary mapping field names to StateField objects
        """
        result = {}
        for name in field_names:
            field = self.get_field(name)
            if field:
                result[name] = field
        logger.debug("Retrieved %d/%d requested fields", len(result), len(field_names))
        return result

    def _compute_field(self, field_name: str) -> Optional[StateField]:
        """Compute semantic field from RAM data with bounded computation.

        Args:
            field_name: Name of the field to compute

        Returns:
            StateField if computable, None otherwise
        """
        if not self._current_ram:
            return None

        try:
            decoded = self.decoder.decode_all(self._current_ram)

            # Player position and dungeon state
            if field_name == "floor":
                floor = decoded["player_state"]["floor_number"]
                return StateField("floor", floor, 1.0, "player_state.floor_number")

            elif field_name == "coords":
                x = decoded["player_state"]["player_tile_x"]
                y = decoded["player_state"]["player_tile_y"]
                return StateField("coords", {"x": x, "y": y}, 1.0, "player_state.player_tile_*")

            # Health and party status
            elif field_name == "health":
                leader_hp = decoded["party_status"]["leader"]["hp"]
                leader_max = decoded["party_status"]["leader"]["hp_max"]
                return StateField("health", {
                    "current": leader_hp,
                    "max": leader_max,
                    "ratio": leader_hp / leader_max if leader_max > 0 else 0.0
                }, 1.0, "party_status.leader.hp")

            elif field_name == "belly":
                belly = decoded["party_status"]["leader"]["belly"]
                return StateField("belly", belly, 1.0, "party_status.leader.belly")

            elif field_name == "party_status":
                return StateField("party_status", decoded["party_status"], 1.0, "party_status")

            # Inventory and items
            elif field_name == "inventory":
                items = decoded.get("items", [])
                return StateField("inventory", items, 1.0, "items")

            elif field_name == "inventory_highlights":
                # Items that are highlighted/important (apples, keys, etc.)
                items = decoded.get("items", [])
                highlights = []
                for item in items:
                    item_id = item.get("item_id", 0)
                    # Highlight important items (apples=120-125, keys=50-60, etc.)
                    if 120 <= item_id <= 125 or 50 <= item_id <= 60:
                        highlights.append(item)
                return StateField("inventory_highlights", highlights, 0.9, "computed")

            # Map and navigation
            elif field_name == "tile_flags":
                # Tile properties (walkable, water, etc.) - simplified placeholder
                # In real implementation, would decode tile collision data
                return StateField("tile_flags", {
                    "walkable": True,  # Placeholder
                    "has_water": False,
                    "has_trap": False
                }, 0.8, "computed")

            elif field_name == "stairs_visible":
                stairs_x = decoded["map_data"]["stairs_x"]
                stairs_y = decoded["map_data"]["stairs_y"]
                # Check if stairs are on screen (simplified)
                visible = stairs_x > 0 and stairs_y > 0
                return StateField("stairs_visible", visible, 0.95, "map_data.stairs_*")

            elif field_name == "path_to_stairs":
                # Bounded path computation to stairs
                player_x = decoded["player_state"]["player_tile_x"]
                player_y = decoded["player_state"]["player_tile_y"]
                stairs_x = decoded["map_data"]["stairs_x"]
                stairs_y = decoded["map_data"]["stairs_y"]

                # Simple Manhattan distance path (bounded)
                path = self._compute_bounded_path(player_x, player_y, stairs_x, stairs_y)
                return StateField("path_to_stairs", path, 0.8, "computed")

            # Combat and enemies
            elif field_name == "enemies_on_screen":
                monsters = decoded["monsters"]
                enemies = [m for m in monsters if m["affiliation"] == 1]  # enemy affiliation
                return StateField("enemies_on_screen", enemies, 1.0, "monsters.affiliation")

            elif field_name == "allies_on_screen":
                monsters = decoded["monsters"]
                allies = [m for m in monsters if m["affiliation"] == 0]  # ally affiliation
                return StateField("allies_on_screen", allies, 1.0, "monsters.affiliation")

            # UI and interaction state
            elif field_name == "dialog_active":
                # Check various dialog/menu states - simplified
                # In real implementation, would check multiple RAM locations
                return StateField("dialog_active", False, 0.9, "computed")

            elif field_name == "menu_open":
                # Check if any menu is open
                return StateField("menu_open", False, 0.85, "computed")

            # Turn and timing
            elif field_name == "turn_counter":
                turn = decoded["player_state"]["turn_counter"]
                return StateField("turn_counter", turn, 1.0, "player_state.turn_counter")

            else:
                logger.warning("Unknown field requested: %s", field_name)
                return None

        except KeyError as e:
            logger.error("Missing expected RAM data for field %s: %s", field_name, e)
            return None
        except Exception as e:
            logger.error("Error computing field %s: %s", field_name, e)
            return None

    def _compute_bounded_path(self, start_x: int, start_y: int, end_x: int, end_y: int) -> List[Tuple[int, int]]:
        """Compute bounded Manhattan path between points.

        Uses simple greedy Manhattan distance to prevent excessive computation.
        Limited to 50 steps to maintain bounded computation guarantees.

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Target coordinates

        Returns:
            List of (x, y) coordinate tuples representing the path
        """
        path = []
        current_x, current_y = start_x, start_y

        # Limit path length to prevent excessive computation
        max_steps = 50

        for _ in range(max_steps):
            if current_x == end_x and current_y == end_y:
                break

            # Move towards end using Manhattan distance (prioritize x then y)
            if current_x < end_x:
                current_x += 1
            elif current_x > end_x:
                current_x -= 1
            elif current_y < end_y:
                current_y += 1
            elif current_y > end_y:
                current_y -= 1
            else:
                break  # Shouldn't happen

            path.append((current_x, current_y))

            # Safety check to prevent infinite loops
            if len(path) >= max_steps:
                logger.warning("Path computation reached max steps (%d)", max_steps)
                break

        logger.debug("Computed path from (%d,%d) to (%d,%d): %d steps",
                    start_x, start_y, end_x, end_y, len(path))
        return path

    def get_all_fields(self) -> Dict[str, StateField]:
        """Get all available semantic fields.

        Returns:
            Dictionary of all computable semantic fields
        """
        field_names = [
            "floor", "coords", "health", "belly", "party_status",
            "inventory", "inventory_highlights", "tile_flags",
            "enemies_on_screen", "allies_on_screen",
            "dialog_active", "menu_open", "stairs_visible",
            "path_to_stairs", "turn_counter"
        ]

        return self.get_multiple_fields(field_names)

    def clear_cache(self) -> None:
        """Clear computed field cache.

        Forces recomputation of all fields on next access.
        Useful for debugging or when RAM data changes significantly.
        """
        self._cached_fields.clear()
        logger.debug("Cleared field cache")