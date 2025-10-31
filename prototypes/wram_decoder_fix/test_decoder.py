"""Lightweight test harness for WRAMDecoderV2."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from unittest.mock import Mock

# Set feature flag before importing decoder_v2
os.environ["MD_DECODER_V2"] = "1"

from decoder_v2 import WRAMDecoderV2, MONSTER_STRUCT_SIZE


def build_synthetic_dump() -> bytes:
    """Create a synthetic dump to validate structural assumptions."""
    buffer = bytearray(512)
    base_offset = 0x0120
    struct_size = 32

    def write_entity(slot: int, species: int, x: int, y: int, hp: int, hp_max: int) -> None:
        offset = base_offset + slot * struct_size
        struct.pack_into("<H", buffer, offset, species)
        struct.pack_into("<H", buffer, offset + 2, x)
        struct.pack_into("<H", buffer, offset + 4, y)
        struct.pack_into("<H", buffer, offset + 6, hp)
        struct.pack_into("<H", buffer, offset + 8, hp_max)

    write_entity(0, 1, 5, 6, 35, 40)
    write_entity(1, 25, 10, 14, 20, 25)
    write_entity(2, 150, 20, 18, 88, 88)
    # Leave remaining slots empty (species=0 by default).

    return bytes(buffer)


def create_mock_controller():
    """Create a mock MGBAController for testing."""
    controller = Mock()

    # Mock address manager
    address_manager = Mock()
    address_manager.get_address.side_effect = lambda category, field: {
        ("entities", "monster_list_ptr"): 0x02004139,  # Example WRAM address
        ("entities", "monster_count"): 0x0200413D,     # Count address
    }.get((category, field), 0)

    controller.address_manager = address_manager
    return controller


def run_synthetic_test() -> None:
    """Run synthetic tests with mocked controller."""
    print("=== Synthetic Test (Functional) ===")
    
    try:

        # Test 1: Basic decoding success
        print("Test 1: Basic decoding success")
        controller = create_mock_controller()
        decoder = WRAMDecoderV2(controller)
        
        # Set up mock data
        monster_struct_addr = 0x02005000
        # Build exactly 48 bytes for monster struct using struct.pack
        monster_data = struct.pack(
            "<HBH14xBHB",
            25,  # species_id = 25 (Pikachu)
            5,   # level = 5
            50,  # hp_current = 50
            5,   # tile_x = 5
            8,   # tile_y = 8
            1    # visible = 1
        )
        # Ensure we have exactly 48 bytes (this should already be correct)
        assert len(monster_data) == 48, f"Expected 48 bytes, got {len(monster_data)}"
        
        print(f"DEBUG: monster_data size = {len(monster_data)} bytes")

        # Use side_effect function with state tracking
        call_log = []
        
        def peek_side_effect(address, size):
            result = None
            if address == 0x02004139 and size == 4:
                result = monster_struct_addr.to_bytes(4, 'little')  # list_ptr
            elif address == 0x0200413D and size == 1:
                result = b'\x02'  # count
            elif address == monster_struct_addr and size == 48:
                result = monster_data
            
            call_log.append((address, size, result))
            print(f"DEBUG: peek(0x{address:08X}, {size}) -> {type(result).__name__} (len={len(result) if result is not None else 'None'})")
            return result

        controller.peek.side_effect = peek_side_effect
        
        print(f"DEBUG: About to call decode_first_mon()")
        result = decoder.decode_first_mon()
        print(f"DEBUG: After call, result = {result}")
        print(f"DEBUG: Call log: {call_log}")
        
        assert result is not None
        assert result["species_id"] == 25
        assert result["level"] == 5
        assert result["hp_current"] == 50
        assert result["hp_max"] == 100
        assert result["tile_x"] == 5
        assert result["tile_y"] == 8
        assert result["visible"] == 1
        print("✓ Basic decoding test passed")

        # Test 2: Empty monster list
        print("Test 2: Empty monster list")
        controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x00',              # count = 0
        ]

        result = decoder.decode_first_mon()
        assert result is None
        print("✓ Empty list test passed")

        # Test 3: Read failure
        print("Test 3: Read failure handling")
        controller.peek.return_value = None

        result = decoder.decode_first_mon()
        assert result is None
        print("✓ Read failure test passed")

        print("PASS (functional tests)\n")

    except Exception as e:
        print(f"FAIL: Synthetic test failed with error: {e}")
        raise


def run_real_dumps() -> None:
    print("No real dump test for WRAMDecoderV2: requires controller and emulator integration.")
    print("PASS (placeholder)\n")


def main() -> None:
    run_synthetic_test()
    run_real_dumps()


if __name__ == "__main__":
    main()
