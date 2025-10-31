"""
WRAM Decoder Prototype v2 - Contiguous Reads with Struct Parsing

This module provides safe, prototype WRAM decoding functionality for Pokemon Mystery Dungeon.
It uses contiguous memory reads and struct parsing aligned to the monster entity fields
defined in the address configuration.

Key features:
- Contiguous reads for efficiency and reliability
- Struct parsing with proper endianness handling
- Safety guards and validation
- Feature flag controlled (MD_DECODER_V2=1)
"""

import os
import struct
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature flag - must be enabled to use this decoder
MD_DECODER_V2 = os.getenv("MD_DECODER_V2", "0").lower() in ("1", "true", "yes")

# Monster entity structure size (from config)
MONSTER_STRUCT_SIZE = 48

# Field definitions from config/addresses/pmd_red_us_v1.json
MONSTER_FIELDS = {
    "species_id": {"offset": 0, "size": 2, "type": "uint16", "description": "Pokemon species ID"},
    "level": {"offset": 2, "size": 1, "type": "uint8", "description": "Pokemon level"},
    "hp_current": {"offset": 4, "size": 2, "type": "uint16", "description": "Current HP"},
    "hp_max": {"offset": 6, "size": 2, "type": "uint16", "description": "Maximum HP"},
    "status": {"offset": 8, "size": 1, "type": "uint8", "description": "Status conditions"},
    "affiliation": {"offset": 9, "size": 1, "type": "uint8", "description": "0=ally, 1=enemy, 2=neutral"},
    "tile_x": {"offset": 16, "size": 1, "type": "uint8", "description": "X position"},
    "tile_y": {"offset": 17, "size": 1, "type": "uint8", "description": "Y position"},
    "direction": {"offset": 18, "size": 1, "type": "uint8", "description": "Facing direction"},
    "visible": {"offset": 19, "size": 1, "type": "uint8", "description": "Is entity visible"},
}


class WRAMDecoderV2:
    """WRAM decoder using contiguous reads and struct parsing."""

    def __init__(self, controller):
        """Initialize decoder with MGBA controller.

        Args:
            controller: MGBAController instance for memory access
        """
        self.controller = controller
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Cache address manager for efficiency
        self.address_manager = controller.address_manager

        # Validate feature flag
        if not MD_DECODER_V2:
            raise RuntimeError("WRAMDecoderV2 requires MD_DECODER_V2=1 environment variable")

    def _read_contiguous(self, address: int, size: int) -> Optional[bytes]:
        """Read contiguous bytes from memory address.

        Args:
            address: Absolute memory address
            size: Number of bytes to read

        Returns:
            Raw bytes data or None if failed
        """
        try:
            data = self.controller.peek(address, size)
            if data is None or len(data) != size:
                self.logger.warning(f"Failed to read {size} bytes from 0x{address:08X}")
                return None
            return data
        except Exception as e:
            self.logger.error(f"Error reading {size} bytes from 0x{address:08X}: {e}")
            return None

    def _parse_field(self, data: bytes, field_def: Dict[str, Any]) -> Any:
        """Parse a single field from raw bytes using struct format.

        Args:
            data: Raw bytes containing the field
            field_def: Field definition from MONSTER_FIELDS

        Returns:
            Parsed field value
        """
        offset = field_def["offset"]
        size = field_def["size"]
        field_type = field_def["type"]

        # Extract field bytes
        if offset + size > len(data):
            raise ValueError(f"Field {field_def.get('description', 'unknown')} extends beyond data")

        field_bytes = data[offset:offset + size]

        # Parse based on type (little-endian for GBA)
        if field_type == "uint8":
            return struct.unpack("<B", field_bytes)[0]
        elif field_type == "uint16":
            return struct.unpack("<H", field_bytes)[0]
        elif field_type == "uint32":
            return struct.unpack("<I", field_bytes)[0]
        elif field_type == "int8":
            return struct.unpack("<b", field_bytes)[0]
        elif field_type == "int16":
            return struct.unpack("<h", field_bytes)[0]
        elif field_type == "int32":
            return struct.unpack("<i", field_bytes)[0]
        else:
            # For unknown types, return raw bytes
            self.logger.warning(f"Unknown field type '{field_type}', returning raw bytes")
            return field_bytes

    def get_monster_list_info(self) -> Optional[Tuple[int, int]]:
        """Get monster list pointer and count.

        Returns:
            Tuple of (list_ptr, count) or None if failed
        """
        try:
            # Read monster list pointer (4 bytes)
            list_ptr_addr = self.address_manager.get_address("entities", "monster_list_ptr")
            list_ptr_data = self._read_contiguous(list_ptr_addr, 4)
            if list_ptr_data is None:
                return None
            list_ptr = struct.unpack("<I", list_ptr_data)[0]

            # Read monster count (1 byte)
            count_addr = self.address_manager.get_address("entities", "monster_count")
            count_data = self._read_contiguous(count_addr, 1)
            if count_data is None:
                return None
            count = struct.unpack("<B", count_data)[0]

            return list_ptr, count

        except Exception as e:
            self.logger.error(f"Failed to get monster list info: {e}")
            return None

    def decode_first_mon(self) -> Optional[Dict[str, Any]]:
        """Decode the first monster entity from WRAM.

        Returns:
            Dict with decoded monster data or None if failed
        """
        try:
            # Get monster list info
            list_info = self.get_monster_list_info()
            if list_info is None:
                return None

            list_ptr, count = list_info

            if count == 0:
                self.logger.info("No monsters in list")
                return None

            # Read first monster struct (48 bytes)
            monster_addr = list_ptr
            monster_data = self._read_contiguous(monster_addr, MONSTER_STRUCT_SIZE)
            if monster_data is None:
                return None

            # Parse required fields
            result = {}
            for field_name, field_def in MONSTER_FIELDS.items():
                try:
                    value = self._parse_field(monster_data, field_def)
                    result[field_name] = value
                except Exception as e:
                    self.logger.warning(f"Failed to parse field '{field_name}': {e}")
                    result[field_name] = None

            # Add metadata
            result["_metadata"] = {
                "address": monster_addr,
                "struct_size": MONSTER_STRUCT_SIZE,
                "raw_bytes": monster_data.hex(),
                "decoder_version": "v2",
            }

            self.logger.debug(f"Decoded first monster: species_id={result.get('species_id')}, level={result.get('level')}, hp={result.get('hp_current')}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to decode first monster: {e}")
            return None

    def decode_all_monsters(self) -> Optional[list[Dict[str, Any]]]:
        """Decode all monster entities from WRAM.

        Returns:
            List of decoded monster dicts or None if failed
        """
        try:
            # Get monster list info
            list_info = self.get_monster_list_info()
            if list_info is None:
                return None

            list_ptr, count = list_info

            if count == 0:
                return []

            monsters = []
            for i in range(count):
                # Calculate address for this monster
                monster_addr = list_ptr + (i * MONSTER_STRUCT_SIZE)

                # Read monster struct
                monster_data = self._read_contiguous(monster_addr, MONSTER_STRUCT_SIZE)
                if monster_data is None:
                    self.logger.warning(f"Failed to read monster {i} at 0x{monster_addr:08X}")
                    continue

                # Parse fields
                monster = {}
                for field_name, field_def in MONSTER_FIELDS.items():
                    try:
                        value = self._parse_field(monster_data, field_def)
                        monster[field_name] = value
                    except Exception as e:
                        self.logger.warning(f"Failed to parse field '{field_name}' for monster {i}: {e}")
                        monster[field_name] = None

                # Add metadata
                monster["_metadata"] = {
                    "index": i,
                    "address": monster_addr,
                    "struct_size": MONSTER_STRUCT_SIZE,
                    "raw_bytes": monster_data.hex(),
                    "decoder_version": "v2",
                }

                monsters.append(monster)

            self.logger.debug(f"Decoded {len(monsters)} monsters")
            return monsters

        except Exception as e:
            self.logger.error(f"Failed to decode all monsters: {e}")
            return None


def decode_first_mon(controller) -> Optional[Dict[str, Any]]:
    """Convenience function to decode first monster.

    Args:
        controller: MGBAController instance

    Returns:
        Decoded monster data or None
    """
    if not MD_DECODER_V2:
        logger.warning("decode_first_mon requires MD_DECODER_V2=1")
        return None

    try:
        decoder = WRAMDecoderV2(controller)
        return decoder.decode_first_mon()
    except Exception as e:
        logger.error(f"Error in decode_first_mon: {e}")
        return None


# Export for external use
__all__ = ["WRAMDecoderV2", "decode_first_mon", "MONSTER_FIELDS", "MONSTER_STRUCT_SIZE"]
