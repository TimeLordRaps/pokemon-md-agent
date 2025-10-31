"""Pure decoders for PMD Red Rescue Team RAM structures.

All decoders are ROM SHA-1 gated to ensure compatibility.
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .rom_gating import validate_rom_sha1


@dataclass
class Entity:
    """Game entity (monster/player)."""
    species_id: int
    level: int
    hp_current: int
    hp_max: int
    status: int
    affiliation: int  # 0=ally, 1=enemy, 2=neutral
    tile_x: int
    tile_y: int
    direction: int
    visible: bool


@dataclass
class Item:
    """Ground item."""
    item_id: int
    tile_x: int
    tile_y: int
    quantity: int


@dataclass
class MapData:
    """Map and camera data."""
    camera_origin_x: int
    camera_origin_y: int
    weather_state: int
    turn_phase: int
    stairs_x: int
    stairs_y: int


@dataclass
class PlayerState:
    """Player state data."""
    player_tile_x: int
    player_tile_y: int
    partner_tile_x: int
    partner_tile_y: int
    floor_number: int
    dungeon_id: int
    turn_counter: int


@dataclass
class PartyStatus:
    """Party status data."""
    leader_hp: int
    leader_hp_max: int
    leader_belly: int
    partner_hp: int
    partner_hp_max: int
    partner_belly: int


@dataclass
class RAMSnapshot:
    """Complete RAM snapshot."""
    entities: List[Entity]
    items: List[Item]
    map_data: MapData
    player_state: PlayerState
    party_status: PartyStatus
    timestamp: float


@dataclass(frozen=True)
class StructFieldSpec:
    """Description of a packed struct field."""
    name: str
    offset: int
    size: int
    field_type: str


class PMDRedDecoder:
    """Decoder for PMD Red Rescue Team RAM data."""

    ROM_SHA1 = "9f4cfc5b5f4859d17169a485462e977c7aac2b89"

    def __init__(self, addresses_config: Dict[str, Any]):
        """Initialize decoder with address configuration."""
        validate_rom_sha1(self.ROM_SHA1)
        self.addresses = addresses_config
        entities_cfg = self.addresses["entities"]
        self._monster_struct_size: int = entities_cfg["monster_struct_size"]["value"]
        self._monster_field_specs: Dict[str, StructFieldSpec] = {
            name: StructFieldSpec(
                name=name,
                offset=field_cfg["offset"],
                size=field_cfg["size"],
                field_type=field_cfg["type"],
            )
            for name, field_cfg in entities_cfg["monster_fields"].items()
        }

    def decode_uint8(self, data: bytes, offset: int) -> int:
        """Decode uint8 from data at offset. Returns 0 if buffer too short."""
        if offset + 1 > len(data):
            return 0
        return struct.unpack('<B', data[offset:offset+1])[0]

    def decode_uint16(self, data: bytes, offset: int) -> int:
        """Decode uint16 from data at offset (little-endian). Returns 0 if buffer too short."""
        if offset + 2 > len(data):
            return 0
        return struct.unpack('<H', data[offset:offset+2])[0]

    def decode_uint32(self, data: bytes, offset: int) -> int:
        """Decode uint32 from data at offset (little-endian). Returns 0 if buffer too short."""
        if offset + 4 > len(data):
            return 0
        return struct.unpack('<I', data[offset:offset+4])[0]

    def decode_bool(self, data: bytes, offset: int) -> bool:
        """Decode boolean from data at offset."""
        return self.decode_uint8(data, offset) != 0

    def decode_bitfield(self, data: bytes, offset: int, size: int) -> int:
        """Decode bitfield from data at offset."""
        if size == 1:
            return self.decode_uint8(data, offset)
        elif size == 2:
            return self.decode_uint16(data, offset)
        elif size == 4:
            return self.decode_uint32(data, offset)
        else:
            raise ValueError(f"Unsupported bitfield size: {size}")

    def decode_player_state(self, data: bytes) -> Dict[str, Any]:
        """Decode player state from RAM data."""
        base = self.addresses["player_state"]

        return {
            "floor_number": self.decode_uint8(data, base["floor_number"]["address"]),
            "dungeon_id": self.decode_uint16(data, base["dungeon_id"]["address"]),
            "turn_counter": self.decode_uint16(data, base["turn_counter"]["address"]),
            "player_tile_x": self.decode_uint8(data, base["player_tile_x"]["address"]),
            "player_tile_y": self.decode_uint8(data, base["player_tile_y"]["address"]),
            "partner_tile_x": self.decode_uint8(data, base["partner_tile_x"]["address"]),
            "partner_tile_y": self.decode_uint8(data, base["partner_tile_y"]["address"]),
            "room_flag": self.decode_bool(data, base["room_flag"]["address"])
        }

    def decode_party_status(self, data: bytes) -> Dict[str, Any]:
        """Decode party status from RAM data."""
        base = self.addresses["party_status"]

        return {
            "leader": {
                "hp": self.decode_uint16(data, base["leader_hp"]["address"]),
                "hp_max": self.decode_uint16(data, base["leader_hp_max"]["address"]),
                "belly": self.decode_uint16(data, base["leader_belly"]["address"]),
                "status": self.decode_bitfield(data, base["leader_status"]["address"],
                                             base["leader_status"]["size"])
            },
            "partner": {
                "hp": self.decode_uint16(data, base["partner_hp"]["address"]),
                "hp_max": self.decode_uint16(data, base["partner_hp_max"]["address"]),
                "belly": self.decode_uint16(data, base["partner_belly"]["address"]),
                "status": self.decode_bitfield(data, base["partner_status"]["address"],
                                              base["partner_status"]["size"])
            }
        }

    def decode_map_data(self, data: bytes) -> Dict[str, Any]:
        """Decode map data from RAM data."""
        base = self.addresses["map_data"]

        return {
            "camera_origin_x": self.decode_uint8(data, base["camera_origin_x"]["address"]),
            "camera_origin_y": self.decode_uint8(data, base["camera_origin_y"]["address"]),
            "weather_state": self.decode_uint8(data, base["weather_state"]["address"]),
            "turn_phase": self.decode_uint8(data, base["turn_phase"]["address"]),
            "stairs_x": self.decode_uint8(data, base["stairs_x"]["address"]),
            "stairs_y": self.decode_uint8(data, base["stairs_y"]["address"])
        }

    def _unpack_struct_field(self, buffer: memoryview, spec: StructFieldSpec) -> Any:
        """Decode a field from a contiguous struct buffer."""
        end_offset = spec.offset + spec.size
        if end_offset > len(buffer):
            raise ValueError(f"Field {spec.name} exceeds struct bounds")

        field_type = spec.field_type
        if field_type in {"uint8", "bitfield"}:
            value = struct.unpack_from('<B', buffer, spec.offset)[0]
        elif field_type == "bool":
            value = struct.unpack_from('<B', buffer, spec.offset)[0]
            return value != 0
        elif field_type == "uint16":
            value = struct.unpack_from('<H', buffer, spec.offset)[0]
        elif field_type == "uint32":
            value = struct.unpack_from('<I', buffer, spec.offset)[0]
        elif field_type == "int8":
            value = struct.unpack_from('<b', buffer, spec.offset)[0]
        elif field_type == "int16":
            value = struct.unpack_from('<h', buffer, spec.offset)[0]
        elif field_type == "int32":
            value = struct.unpack_from('<i', buffer, spec.offset)[0]
        else:
            # Return raw bytes for unsupported types
            return bytes(buffer[spec.offset:end_offset])

        return value

    def _decode_monster_struct(self, struct_bytes: bytes) -> Dict[str, Any]:
        """Decode a contiguous monster struct into a dictionary."""
        buffer = memoryview(struct_bytes)
        decoded: Dict[str, Any] = {}

        for name, spec in self._monster_field_specs.items():
            try:
                decoded[name] = self._unpack_struct_field(buffer, spec)
            except (ValueError, struct.error):
                decoded[name] = None

        return decoded

    def decode_monsters(self, data: bytes) -> List[Dict[str, Any]]:
        """Decode monster list from RAM data."""
        entities = self.addresses["entities"]
        count = self.decode_uint8(data, entities["monster_count"]["address"])
        ptr = self.decode_uint32(data, entities["monster_list_ptr"]["address"])

        monsters = []
        if ptr <= 0 or count == 0:
            return monsters

        max_monsters = min(count, entities["monster_count"]["max"])
        struct_size = self._monster_struct_size
        data_len = len(data)

        for i in range(max_monsters):
            struct_offset = ptr + (i * struct_size)
            struct_end = struct_offset + struct_size

            if struct_offset < 0 or struct_end > data_len:
                # Refuse partial buffers or invalid pointers
                break

            struct_bytes = data[struct_offset:struct_end]
            if len(struct_bytes) != struct_size:
                break

            monster = self._decode_monster_struct(struct_bytes)
            monsters.append(monster)

        return monsters

    def decode_items(self, data: bytes) -> List[Dict[str, Any]]:
        """Decode item list from RAM data."""
        items_config = self.addresses["items"]
        item_struct_size = items_config["item_struct_size"]["value"]

        count = self.decode_uint8(data, items_config["item_count"]["address"])
        ptr = self.decode_uint32(data, items_config["item_list_ptr"]["address"])

        items = []
        for i in range(min(count, items_config["item_count"]["max"])):
            offset = ptr + (i * item_struct_size)
            fields = items_config["item_fields"]

            item = {
                "item_id": self.decode_uint16(data, offset + fields["item_id"]["offset"]),
                "tile_x": self.decode_uint8(data, offset + fields["tile_x"]["offset"]),
                "tile_y": self.decode_uint8(data, offset + fields["tile_y"]["offset"]),
                "quantity": self.decode_uint16(data, offset + fields["quantity"]["offset"])
            }
            items.append(item)

        return items

    def decode_all(self, data: bytes) -> Dict[str, Any]:
        """Decode all game state from RAM data."""
        return {
            "player_state": self.decode_player_state(data),
            "party_status": self.decode_party_status(data),
            "map_data": self.decode_map_data(data),
            "monsters": self.decode_monsters(data),
            "items": self.decode_items(data)
        }


def load_addresses_config() -> Dict[str, Any]:
    """Load addresses configuration for PMD Red."""
    config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)["addresses"]


def create_decoder() -> PMDRedDecoder:
    """Create a PMD Red decoder instance."""
    config = load_addresses_config()
    return PMDRedDecoder(config)
