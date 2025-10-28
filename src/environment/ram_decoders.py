"""RAM decoders for Pokemon MD Red Rescue Team."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import struct
import logging
from enum import IntFlag

from .mgba_controller import MGBAController

logger = logging.getLogger(__name__)


class StatusCondition(IntFlag):
    """Status condition bit flags."""
    NONE = 0
    SLEEP = 1 << 0
    PARALYSIS = 1 << 1
    BURN = 1 << 2
    POISON = 1 << 3
    CONFUSION = 1 << 4
    CURSE = 1 << 5


class Affiliation(IntFlag):
    """Entity affiliation."""
    ALLY = 0
    ENEMY = 1
    NEUTRAL = 2


@dataclass
class Entity:
    """Entity structure (Pokemon or NPC)."""
    species_id: int
    level: int
    hp_current: int
    hp_max: int
    status: StatusCondition
    affiliation: Affiliation
    tile_x: int
    tile_y: int
    direction: int  # 0=up, 1=right, 2=down, 3=left
    visible: bool


@dataclass
class Item:
    """Item structure."""
    item_id: int
    tile_x: int
    tile_y: int
    quantity: int


@dataclass
class PlayerState:
    """Player state information."""
    floor_number: int
    dungeon_id: int
    turn_counter: int
    player_tile_x: int
    player_tile_y: int
    partner_tile_x: int
    partner_tile_y: int
    room_flag: bool


@dataclass
class PartyStatus:
    """Party status information."""
    leader_hp: int
    leader_hp_max: int
    leader_belly: int
    leader_status: StatusCondition
    partner_hp: int
    partner_hp_max: int
    partner_belly: int
    partner_status: StatusCondition


@dataclass
class MapData:
    """Map data information."""
    stairs_x: int
    stairs_y: int
    camera_origin_x: int
    camera_origin_y: int
    weather_state: int
    turn_phase: int


@dataclass
class RAMSnapshot:
    """Complete RAM snapshot."""
    timestamp: float
    frame: Optional[int]
    player_state: PlayerState
    party_status: PartyStatus
    map_data: MapData
    entities: List[Entity]
    items: List[Item]


class RAMDecoder:
    """Decodes Pokemon MD RAM data."""
    
    SPECIES_NAMES = {
        # Common Pokemon (placeholder - would need full list)
        1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur",
        4: "Charmander", 5: "Charmeleon", 6: "Charizard",
        7: "Squirtle", 8: "Wartortle", 9: "Blastoise",
        10: "Caterpie", 11: "Metapod", 12: "Butterfree",
        # Add more as needed
    }
    
    ITEM_NAMES = {
        # Common items (placeholder)
        1: "Stick", 2: "Iron Thorn", 3: "Silver Spike",
        4: "Apple", 5: "Great Apple", 6: "Max Apple",
        7: "Berry", 8: "Gummi", 9: "Seed",
        # Add more as needed
    }
    
    def __init__(self, controller: MGBAController, addresses_config: Path):
        """Initialize RAM decoder.
        
        Args:
            controller: mgba controller
            addresses_config: Path to addresses config JSON
        """
        self.controller = controller
        self.addresses_config = Path(addresses_config)
        self._addresses = self._load_addresses()
        
        logger.info("RAMDecoder initialized with config: %s", addresses_config)
    
    def _load_addresses(self) -> Dict[str, Any]:
        """Load addresses configuration.
        
        Returns:
            Addresses configuration dictionary
        """
        try:
            with open(self.addresses_config, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load addresses config: %s", e)
            raise
    
    def _read_memory(self, domain: str, address: int, size: int) -> bytes:
        """Read memory from domain.
        
        Args:
            domain: Memory domain name
            address: Memory address
            size: Number of bytes to read
            
        Returns:
            Raw byte data
        """
        return self.controller.memory_domain_read_range(domain, address, size) or b'\x00' * size
    
    def _read_u8(self, domain: str, address: int) -> int:
        """Read unsigned 8-bit value.
        
        Args:
            domain: Memory domain
            address: Memory address
            
        Returns:
            Unsigned 8-bit value
        """
        data = self._read_memory(domain, address, 1)
        return data[0] if data else 0
    
    def _read_u16(self, domain: str, address: int) -> int:
        """Read unsigned 16-bit value.
        
        Args:
            domain: Memory domain
            address: Memory address
            
        Returns:
            Unsigned 16-bit value
        """
        data = self._read_memory(domain, address, 2)
        return struct.unpack('<H', data)[0] if data and len(data) == 2 else 0
    
    def _read_u32(self, domain: str, address: int) -> int:
        """Read unsigned 32-bit value.
        
        Args:
            domain: Memory domain
            address: Memory address
            
        Returns:
            Unsigned 32-bit value
        """
        data = self._read_memory(domain, address, 4)
        return struct.unpack('<I', data)[0] if data and len(data) == 4 else 0
    
    def _read_i32(self, domain: str, address: int) -> int:
        """Read signed 32-bit value.
        
        Args:
            domain: Memory domain
            address: Memory address
            
        Returns:
            Signed 32-bit value
        """
        data = self._read_memory(domain, address, 4)
        return struct.unpack('<i', data)[0] if data and len(data) == 4 else 0
    
    def _read_ptr(self, domain: str, address: int) -> int:
        """Read pointer value.
        
        Args:
            domain: Memory domain
            address: Memory address
            
        Returns:
            Pointer value (memory address)
        """
        return self._read_u32(domain, address)
    
    def _decode_status(self, status_byte: int) -> StatusCondition:
        """Decode status condition byte.
        
        Args:
            status_byte: Status byte value
            
        Returns:
            StatusCondition flags
        """
        return StatusCondition(status_byte)
    
    def _decode_affiliation(self, affiliation_byte: int) -> Affiliation:
        """Decode affiliation byte.
        
        Args:
            affiliation_byte: Affiliation byte value
            
        Returns:
            Affiliation enum
        """
        return Affiliation(affiliation_byte)
    
    def get_player_state(self) -> Optional[PlayerState]:
        """Get player state information.
        
        Returns:
            PlayerState or None if failed
        """
        try:
            addr = self._addresses["addresses"]["player_state"]
            
            return PlayerState(
                floor_number=self._read_u8("WRAM", addr["floor_number"]["address"]),
                dungeon_id=self._read_u16("WRAM", addr["dungeon_id"]["address"]),
                turn_counter=self._read_u16("WRAM", addr["turn_counter"]["address"]),
                player_tile_x=self._read_u8("WRAM", addr["player_tile_x"]["address"]),
                player_tile_y=self._read_u8("WRAM", addr["player_tile_y"]["address"]),
                partner_tile_x=self._read_u8("WRAM", addr["partner_tile_x"]["address"]),
                partner_tile_y=self._read_u8("WRAM", addr["partner_tile_y"]["address"]),
                room_flag=bool(self._read_u8("WRAM", addr["room_flag"]["address"])),
            )
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode player state: %s", e)
            return None
    
    def get_party_status(self) -> Optional[PartyStatus]:
        """Get party status information.
        
        Returns:
            PartyStatus or None if failed
        """
        try:
            addr = self._addresses["addresses"]["party_status"]
            
            return PartyStatus(
                leader_hp=self._read_u16("WRAM", addr["leader_hp"]["address"]),
                leader_hp_max=self._read_u16("WRAM", addr["leader_hp_max"]["address"]),
                leader_belly=self._read_u16("WRAM", addr["leader_belly"]["address"]),
                leader_status=self._decode_status(
                    self._read_u32("WRAM", addr["leader_status"]["address"])
                ),
                partner_hp=self._read_u16("WRAM", addr["partner_hp"]["address"]),
                partner_hp_max=self._read_u16("WRAM", addr["partner_hp_max"]["address"]),
                partner_belly=self._read_u16("WRAM", addr["partner_belly"]["address"]),
                partner_status=self._decode_status(
                    self._read_u32("WRAM", addr["partner_status"]["address"])
                ),
            )
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode party status: %s", e)
            return None
    
    def get_map_data(self) -> Optional[MapData]:
        """Get map data information.
        
        Returns:
            MapData or None if failed
        """
        try:
            addr = self._addresses["addresses"]["map_data"]
            
            return MapData(
                stairs_x=self._read_u8("WRAM", addr["stairs_x"]["address"]),
                stairs_y=self._read_u8("WRAM", addr["stairs_y"]["address"]),
                camera_origin_x=self._read_u8("WRAM", addr["camera_origin_x"]["address"]),
                camera_origin_y=self._read_u8("WRAM", addr["camera_origin_y"]["address"]),
                weather_state=self._read_u8("WRAM", addr["weather_state"]["address"]),
                turn_phase=self._read_u8("WRAM", addr["turn_phase"]["address"]),
            )
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode map data: %s", e)
            return None
    
    def get_entities(self) -> List[Entity]:
        """Get list of entities (Pokemon/NPCs).
        
        Returns:
            List of Entity objects
        """
        try:
            addr = self._addresses["addresses"]["entities"]
            
            # Get list pointer and count
            list_ptr = self._read_ptr("WRAM", addr["monster_list_ptr"]["address"])
            monster_count = self._read_u8("WRAM", addr["monster_count"]["address"])
            
            entities = []
            
            if list_ptr and monster_count > 0:
                struct_size = addr["monster_struct_size"]["value"]
                fields = addr["monster_fields"]
                
                for i in range(monster_count):
                    entity_addr = list_ptr + (i * struct_size)
                    
                    # Read entity data
                    entity = Entity(
                        species_id=self._read_u16("WRAM", entity_addr + fields["species_id"]["offset"]),
                        level=self._read_u8("WRAM", entity_addr + fields["level"]["offset"]),
                        hp_current=self._read_u16("WRAM", entity_addr + fields["hp_current"]["offset"]),
                        hp_max=self._read_u16("WRAM", entity_addr + fields["hp_max"]["offset"]),
                        status=self._decode_status(
                            self._read_u8("WRAM", entity_addr + fields["status"]["offset"])
                        ),
                        affiliation=self._decode_affiliation(
                            self._read_u8("WRAM", entity_addr + fields["affiliation"]["offset"])
                        ),
                        tile_x=self._read_u8("WRAM", entity_addr + fields["tile_x"]["offset"]),
                        tile_y=self._read_u8("WRAM", entity_addr + fields["tile_y"]["offset"]),
                        direction=self._read_u8("WRAM", entity_addr + fields["direction"]["offset"]),
                        visible=bool(self._read_u8("WRAM", entity_addr + fields["visible"]["offset"])),
                    )
                    
                    entities.append(entity)
            
            return entities
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode entities: %s", e)
            return []
    
    def get_items(self) -> List[Item]:
        """Get list of items.
        
        Returns:
            List of Item objects
        """
        try:
            addr = self._addresses["addresses"]["items"]
            
            # Get list pointer and count
            list_ptr = self._read_ptr("WRAM", addr["item_list_ptr"]["address"])
            item_count = self._read_u8("WRAM", addr["item_count"]["address"])
            
            items = []
            
            if list_ptr and item_count > 0:
                struct_size = addr["item_struct_size"]["value"]
                fields = addr["item_fields"]
                
                for i in range(item_count):
                    item_addr = list_ptr + (i * struct_size)
                    
                    # Read item data
                    item = Item(
                        item_id=self._read_u16("WRAM", item_addr + fields["item_id"]["offset"]),
                        tile_x=self._read_u8("WRAM", item_addr + fields["tile_x"]["offset"]),
                        tile_y=self._read_u8("WRAM", item_addr + fields["tile_y"]["offset"]),
                        quantity=self._read_u16("WRAM", item_addr + fields["quantity"]["offset"]),
                    )
                    
                    items.append(item)
            
            return items
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode items: %s", e)
            return []
    
    def get_full_snapshot(self) -> Optional[RAMSnapshot]:
        """Get complete RAM snapshot.
        
        Returns:
            RAMSnapshot or None if failed
        """
        try:
            player_state = self.get_player_state()
            party_status = self.get_party_status()
            map_data = self.get_map_data()
            entities = self.get_entities()
            items = self.get_items()
            
            # Verify we got essential data
            if player_state and party_status and map_data:
                return RAMSnapshot(
                    timestamp=time.time(),
                    frame=self.controller.current_frame(),
                    player_state=player_state,
                    party_status=party_status,
                    map_data=map_data,
                    entities=entities,
                    items=items,
                )
            
            return None
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to decode full snapshot: %s", e)
            return None
    
    def get_species_name(self, species_id: int) -> str:
        """Get species name by ID.
        
        Args:
            species_id: Species ID
            
        Returns:
            Species name or "Unknown"
        """
        return self.SPECIES_NAMES.get(species_id, f"Species_{species_id}")
    
    def get_item_name(self, item_id: int) -> str:
        """Get item name by ID.

        Args:
            item_id: Item ID

        Returns:
            Item name or "Unknown"
        """
        return self.ITEM_NAMES.get(item_id, f"Item_{item_id}")

    def set_text_speed(self, speed: int) -> bool:
        """Set text speed via RAM write (when allow_memory_write enabled).

        Args:
            speed: Text speed (0=fast, 1=normal, 2=slow)

        Returns:
            True if write succeeded
        """
        try:
            addr = self._addresses["addresses"]["town_hubs"]["text_speed"]
            domain = addr["domain"]
            address = addr["address"]

            # Guard RAM write behind feature flag (would be checked at caller level)
            logger.info("Setting text speed to %d via RAM poke", speed)
            return self.controller.memory_domain_write8(domain, address, speed)

        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to set text speed: %s", e)
            return False
    
    def save_snapshot(self, snapshot: RAMSnapshot, output_dir: Path) -> Path:
        """Save RAM snapshot to file.
        
        Args:
            snapshot: RAM snapshot
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = int(snapshot.timestamp)
        filename = f"ram_snapshot_{timestamp}.json"
        output_path = output_dir / filename
        
        # Convert to dict for JSON serialization
        data = {
            "timestamp": snapshot.timestamp,
            "frame": snapshot.frame,
            "player_state": {
                "floor_number": snapshot.player_state.floor_number,
                "dungeon_id": snapshot.player_state.dungeon_id,
                "turn_counter": snapshot.player_state.turn_counter,
                "player_tile_x": snapshot.player_state.player_tile_x,
                "player_tile_y": snapshot.player_state.player_tile_y,
                "partner_tile_x": snapshot.player_state.partner_tile_x,
                "partner_tile_y": snapshot.player_state.partner_tile_y,
                "room_flag": snapshot.player_state.room_flag,
            },
            "party_status": {
                "leader_hp": snapshot.party_status.leader_hp,
                "leader_hp_max": snapshot.party_status.leader_hp_max,
                "leader_belly": snapshot.party_status.leader_belly,
                "leader_status": int(snapshot.party_status.leader_status),
                "partner_hp": snapshot.party_status.partner_hp,
                "partner_hp_max": snapshot.party_status.partner_hp_max,
                "partner_belly": snapshot.party_status.partner_belly,
                "partner_status": int(snapshot.party_status.partner_status),
            },
            "map_data": {
                "stairs_x": snapshot.map_data.stairs_x,
                "stairs_y": snapshot.map_data.stairs_y,
                "camera_origin_x": snapshot.map_data.camera_origin_x,
                "camera_origin_y": snapshot.map_data.camera_origin_y,
                "weather_state": snapshot.map_data.weather_state,
                "turn_phase": snapshot.map_data.turn_phase,
            },
            "entities": [
                {
                    "species_id": e.species_id,
                    "species_name": self.get_species_name(e.species_id),
                    "level": e.level,
                    "hp_current": e.hp_current,
                    "hp_max": e.hp_max,
                    "status": int(e.status),
                    "affiliation": int(e.affiliation),
                    "tile_x": e.tile_x,
                    "tile_y": e.tile_y,
                    "direction": e.direction,
                    "visible": e.visible,
                }
                for e in snapshot.entities
            ],
            "items": [
                {
                    "item_id": i.item_id,
                    "item_name": self.get_item_name(i.item_id),
                    "tile_x": i.tile_x,
                    "tile_y": i.tile_y,
                    "quantity": i.quantity,
                }
                for i in snapshot.items
            ],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Saved RAM snapshot to %s", output_path)
        return output_path


# Add time import
import time
