"""Tests for RAM decoders."""

import json
import struct
from pathlib import Path

import pytest

from src.environment.ram_decoders import PMDRedDecoder, create_decoder, load_addresses_config


class TestPMDRedDecoder:
    """Test PMD Red decoder functionality."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_addresses_config()

    @pytest.fixture
    def decoder(self, config):
        """Create decoder instance."""
        return PMDRedDecoder(config)

    @pytest.fixture
    def sample_ram_data(self):
        """Generate sample RAM data for testing."""
        # Create 64KB of zero data, then set some test values
        data = bytearray(65536)

        # Set player state values (using actual addresses from config)
        # Floor number = 5
        data[33544] = 5
        # Dungeon ID = 10
        struct.pack_into('<H', data, 33546, 10)
        # Turn counter = 150
        struct.pack_into('<H', data, 33548, 150)
        # Player tile X = 10, Y = 8
        data[33550] = 10
        data[33551] = 8
        # Partner tile X = 12, Y = 10
        data[33552] = 12
        data[33553] = 10
        # Room flag = 1 (room)
        data[33554] = 1

        # Party status
        # Leader HP = 200/250
        struct.pack_into('<H', data, 33572, 200)
        struct.pack_into('<H', data, 33574, 250)
        # Leader belly = 75
        struct.pack_into('<H', data, 33576, 75)
        # Partner HP = 180/220
        struct.pack_into('<H', data, 33582, 180)
        struct.pack_into('<H', data, 33584, 220)
        # Partner belly = 80
        struct.pack_into('<H', data, 33586, 80)

        # Monsters (1 monster)
        # Monster count = 1
        data[33566] = 1
        # Monster pointer = 40000
        struct.pack_into('<I', data, 33562, 40000)
        # Monster data at offset 40000
        monster_offset = 40000
        # Species ID = 25 (Pikachu)
        struct.pack_into('<H', data, monster_offset, 25)
        # Level = 15
        data[monster_offset + 2] = 15
        # HP = 50/50
        struct.pack_into('<H', data, monster_offset + 4, 50)
        struct.pack_into('<H', data, monster_offset + 6, 50)
        # Tile X=15, Y=12, Direction=2 (down)
        data[monster_offset + 16] = 15
        data[monster_offset + 17] = 12
        data[monster_offset + 18] = 2

        # Items (1 item)
        # Item count = 1
        data[33571] = 1
        # Item pointer = 41000
        struct.pack_into('<I', data, 33567, 41000)
        # Item data at offset 41000
        item_offset = 41000
        # Item ID = 1, Quantity = 3
        struct.pack_into('<H', data, item_offset, 1)
        struct.pack_into('<H', data, item_offset + 6, 3)
        # Tile X=20, Y=15
        data[item_offset + 4] = 20
        data[item_offset + 5] = 15

        return bytes(data)

    def test_decode_player_state(self, decoder, sample_ram_data):
        """Test player state decoding."""
        state = decoder.decode_player_state(sample_ram_data)

        assert state["floor_number"] == 5
        assert state["dungeon_id"] == 10
        assert state["turn_counter"] == 150
        assert state["player_tile_x"] == 10
        assert state["player_tile_y"] == 8
        assert state["partner_tile_x"] == 12
        assert state["partner_tile_y"] == 10
        assert state["room_flag"] is True

    def test_decode_party_status(self, decoder, sample_ram_data):
        """Test party status decoding."""
        status = decoder.decode_party_status(sample_ram_data)

        assert status["leader"]["hp"] == 200
        assert status["leader"]["hp_max"] == 250
        assert status["leader"]["belly"] == 75
        assert status["partner"]["hp"] == 180
        assert status["partner"]["hp_max"] == 220
        assert status["partner"]["belly"] == 80

    def test_decode_monsters(self, decoder, sample_ram_data):
        """Test monster list decoding."""
        monsters = decoder.decode_monsters(sample_ram_data)

        assert len(monsters) == 1
        monster = monsters[0]
        assert monster["species_id"] == 25
        assert monster["level"] == 15
        assert monster["hp_current"] == 50
        assert monster["hp_max"] == 50
        assert monster["tile_x"] == 15
        assert monster["tile_y"] == 12
        assert monster["direction"] == 2

    def test_decode_items(self, decoder, sample_ram_data):
        """Test item list decoding."""
        items = decoder.decode_items(sample_ram_data)

        assert len(items) == 1
        item = items[0]
        assert item["item_id"] == 1
        assert item["quantity"] == 3
        assert item["tile_x"] == 20
        assert item["tile_y"] == 15

    def test_decode_all(self, decoder, sample_ram_data):
        """Test full state decoding."""
        state = decoder.decode_all(sample_ram_data)

        assert "player_state" in state
        assert "party_status" in state
        assert "map_data" in state
        assert "monsters" in state
        assert "items" in state

        assert len(state["monsters"]) == 1
        assert len(state["items"]) == 1

    def test_create_decoder(self):
        """Test decoder creation."""
        decoder = create_decoder()
        assert isinstance(decoder, PMDRedDecoder)
        assert decoder.ROM_SHA1 == "9f4cfc5b5f4859d17169a485462e977c7aac2b89"