"""
Bug #0002: RAM Address Mismatch Between Controller and Config

Symptoms: RAM reads return incorrect values because addresses in
mgba_controller.py don't match addresses in config/addresses/pmd_red_us_v1.json

Affected Files:
- src/environment/mgba_controller.py (lines 245-280)
- config/addresses/pmd_red_us_v1.json

Root Cause: The controller hardcodes RAM addresses as absolute (0x02xxxxxx),
then converts to WRAM offsets in peek(). However, the offsets don't match
the authoritative addresses in the config file.

Example Mismatches:
1. Floor Number:
   - Controller: 0x02004139 → WRAM offset 0x4139 (16697 decimal)
   - Config: WRAM address 33544 (0x82F8 hex)
   - Difference: 16,847 bytes off!

2. Turn Counter:
   - Controller: 0x02004156 → WRAM offset 0x4156 (16726 decimal)
   - Config: WRAM address 33548 (0x82FC hex)
   - Difference: 16,822 bytes off!

3. Player Position:
   - Controller: 0x020041F8/0x020041FC → WRAM offset 0x41F8/0x41FC
   - Config: WRAM address 33550/33551 (0x82FE/0x82FF hex)
   - Difference: Significant offset mismatch!

Impact: CRITICAL - All RAM reads will return garbage data, breaking:
- Floor detection
- Player position tracking
- HP/belly monitoring
- Dungeon transition detection
- All agent decision-making based on game state

Priority: P0 - Must fix immediately. Without correct RAM addresses,
the agent cannot perceive game state.

Fix Strategy:
1. Use config file as single source of truth
2. Load addresses from config/addresses/pmd_red_us_v1.json at runtime
3. Remove hardcoded RAM_ADDRESSES dict from mgba_controller.py
4. Create AddressManager class to handle config loading and lookups
"""

import pytest
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.mgba_controller import MGBAController


class TestBug0002RamAddressMismatch:
    """Test suite to verify RAM addresses match config file."""

    def test_controller_addresses_match_config(self):
        """Test that controller RAM addresses match config file exactly."""
        # Load config file
        config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Get WRAM base address from config
        wram_base = config["memory_domains"]["WRAM"]["base_address"]
        assert wram_base == 0, "WRAM base should be 0"

        # Create controller and get its hardcoded addresses
        controller = MGBAController()

        # Test floor number
        config_floor_offset = config["addresses"]["player_state"]["floor_number"]["address"]
        controller_floor_absolute = controller.RAM_ADDRESSES["floor"]
        controller_floor_offset = controller_floor_absolute - 0x02000000  # Convert to WRAM offset

        assert controller_floor_offset == config_floor_offset, (
            f"Floor address mismatch: "
            f"controller offset={controller_floor_offset} (0x{controller_floor_offset:X}), "
            f"config offset={config_floor_offset} (0x{config_floor_offset:X})"
        )

    def test_turn_counter_matches_config(self):
        """Test turn counter address matches config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        controller = MGBAController()

        config_turn_offset = config["addresses"]["player_state"]["turn_counter"]["address"]
        controller_turn_absolute = controller.RAM_ADDRESSES["turn_counter"]
        controller_turn_offset = controller_turn_absolute - 0x02000000

        assert controller_turn_offset == config_turn_offset, (
            f"Turn counter address mismatch: "
            f"controller offset={controller_turn_offset} (0x{controller_turn_offset:X}), "
            f"config offset={config_turn_offset} (0x{config_turn_offset:X})"
        )

    def test_player_position_matches_config(self):
        """Test player position addresses match config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        controller = MGBAController()

        # X position
        config_x_offset = config["addresses"]["player_state"]["player_tile_x"]["address"]
        controller_x_absolute = controller.RAM_ADDRESSES["player_x"]
        controller_x_offset = controller_x_absolute - 0x02000000

        assert controller_x_offset == config_x_offset, (
            f"Player X address mismatch: "
            f"controller offset={controller_x_offset} (0x{controller_x_offset:X}), "
            f"config offset={config_x_offset} (0x{config_x_offset:X})"
        )

        # Y position
        config_y_offset = config["addresses"]["player_state"]["player_tile_y"]["address"]
        controller_y_absolute = controller.RAM_ADDRESSES["player_y"]
        controller_y_offset = controller_y_absolute - 0x02000000

        assert controller_y_offset == config_y_offset, (
            f"Player Y address mismatch: "
            f"controller offset={controller_y_offset} (0x{controller_y_offset:X}), "
            f"config offset={config_y_offset} (0x{config_y_offset:X})"
        )

    def test_hp_addresses_match_config(self):
        """Test HP addresses match config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        controller = MGBAController()

        # Current HP
        config_hp_offset = config["addresses"]["party_status"]["leader_hp"]["address"]
        controller_hp_absolute = controller.RAM_ADDRESSES["hp"]
        controller_hp_offset = controller_hp_absolute - 0x02000000

        assert controller_hp_offset == config_hp_offset, (
            f"HP address mismatch: "
            f"controller offset={controller_hp_offset} (0x{controller_hp_offset:X}), "
            f"config offset={config_hp_offset} (0x{config_hp_offset:X})"
        )

        # Max HP
        config_max_hp_offset = config["addresses"]["party_status"]["leader_hp_max"]["address"]
        controller_max_hp_absolute = controller.RAM_ADDRESSES["max_hp"]
        controller_max_hp_offset = controller_max_hp_absolute - 0x02000000

        assert controller_max_hp_offset == config_max_hp_offset, (
            f"Max HP address mismatch: "
            f"controller offset={controller_max_hp_offset} (0x{controller_max_hp_offset:X}), "
            f"config offset={config_max_hp_offset} (0x{config_max_hp_offset:X})"
        )

    def test_belly_address_matches_config(self):
        """Test belly address matches config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "addresses" / "pmd_red_us_v1.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        controller = MGBAController()

        config_belly_offset = config["addresses"]["party_status"]["leader_belly"]["address"]
        controller_belly_absolute = controller.RAM_ADDRESSES["belly"]
        controller_belly_offset = controller_belly_absolute - 0x02000000

        assert controller_belly_offset == config_belly_offset, (
            f"Belly address mismatch: "
            f"controller offset={controller_belly_offset} (0x{controller_belly_offset:X}), "
            f"config offset={config_belly_offset} (0x{config_belly_offset:X})"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
