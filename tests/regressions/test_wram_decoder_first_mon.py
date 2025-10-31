"""
Regression Tests for WRAM Decoder First Monster - test_wram_decoder_first_mon.py

Tests the WRAMDecoderV2.decode_first_mon() functionality with mocked mGBA HTTP API.
Ensures the decoder correctly parses monster entity data from contiguous memory reads.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Set feature flag for tests
os.environ["MD_DECODER_V2"] = "1"

from prototypes.wram_decoder_fix.decoder_v2 import WRAMDecoderV2, decode_first_mon, MONSTER_STRUCT_SIZE


class TestWRAMDecoderFirstMon:
    """Test suite for WRAMDecoderV2 first monster decoding."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock MGBAController with address manager."""
        controller = Mock()

        # Mock address manager
        address_manager = Mock()
        address_manager.get_address.side_effect = lambda category, field: {
            ("entities", "monster_list_ptr"): 0x02004139,  # Example WRAM address
            ("entities", "monster_count"): 0x0200413D,     # Count address
        }.get((category, field), 0)

        controller.address_manager = address_manager
        return controller

    @pytest.fixture
    def decoder(self, mock_controller):
        """Create decoder instance with mock controller."""
        return WRAMDecoderV2(mock_controller)

    def test_decode_first_mon_success(self, decoder, mock_controller):
        """Test successful decoding of first monster."""
        # Mock monster list pointer and count
        monster_data = (
            b'\x01\x00'  # species_id = 1
            b'\x0A'      # level = 10
            b'\x64\x00'  # hp_current = 100
            b'\x64\x00'  # hp_max = 100
            b'\x00'      # status = 0
            b'\x00'      # affiliation = 0
            b'\x00\x00\x00\x00\x00\x00'  # padding to offset 16
            b'\x05'      # tile_x = 5
            b'\x08'      # tile_y = 8
            b'\x00'      # direction = 0
            b'\x01'      # visible = 1
            b'\x00' * 28  # rest of struct
        )

        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr = 0x02004139 (little-endian)
            b'\x02',              # count = 2
            monster_data,         # First monster struct (48 bytes)
        ]

        result = decoder.decode_first_mon()

        assert result is not None
        assert result["species_id"] == 1
        assert result["level"] == 10
        assert result["hp_current"] == 100
        assert result["hp_max"] == 100
        assert result["tile_x"] == 5
        assert result["tile_y"] == 8
        assert result["visible"] == 1

        # Check metadata
        assert "_metadata" in result
        assert result["_metadata"]["decoder_version"] == "v2"
        assert result["_metadata"]["struct_size"] == MONSTER_STRUCT_SIZE

    def test_decode_first_mon_no_monsters(self, decoder, mock_controller):
        """Test decoding when monster count is 0."""
        # Mock empty monster list
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x00',              # count = 0
        ]

        result = decoder.decode_first_mon()

        assert result is None

    def test_decode_first_mon_read_failure(self, decoder, mock_controller):
        """Test handling of read failures."""
        # Mock failed reads
        mock_controller.peek.return_value = None

        result = decoder.decode_first_mon()

        assert result is None

    def test_decode_first_mon_partial_read_failure(self, decoder, mock_controller):
        """Test handling of partial read failures."""
        # Mock successful list info but failed monster read
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x01',              # count = 1
            None,                 # Failed monster read
        ]

        result = decoder.decode_first_mon()

        assert result is None

    def test_decode_first_mon_malformed_data(self, decoder, mock_controller):
        """Test handling of malformed monster data."""
        # Mock monster list with malformed data
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x01',              # count = 1
            b'\xFF\xFF\xFF',      # Too short data
        ]

        result = decoder.decode_first_mon()

        # Should still return result but with None values for failed fields
        assert result is not None
        assert result["species_id"] is None  # Failed to parse

    def test_get_monster_list_info_success(self, decoder, mock_controller):
        """Test successful retrieval of monster list info."""
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr = 0x02004139
            b'\x05',              # count = 5
        ]

        result = decoder.get_monster_list_info()

        assert result == (0x02004139, 5)

    def test_get_monster_list_info_read_failure(self, decoder, mock_controller):
        """Test handling of read failures in list info."""
        mock_controller.peek.return_value = None

        result = decoder.get_monster_list_info()

        assert result is None

    def test_convenience_function_success(self, mock_controller):
        """Test the convenience decode_first_mon function."""
        # Mock successful decoding
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x01',              # count = 1
            b'\x19\x00'  # species_id = 25 (Pikachu)
            b'\x05'      # level = 5
            b'\x32\x00'  # hp_current = 50
            + b'\x00' * 44  # rest of struct
        ]

        result = decode_first_mon(mock_controller)

        assert result is not None
        assert result["species_id"] == 25
        assert result["level"] == 5
        assert result["hp_current"] == 50

    def test_convenience_function_feature_flag_disabled(self, mock_controller):
        """Test convenience function when feature flag is disabled."""
        # Temporarily disable feature flag
        os.environ["MD_DECODER_V2"] = "0"

        try:
            result = decode_first_mon(mock_controller)
            assert result is None
        finally:
            # Restore feature flag
            os.environ["MD_DECODER_V2"] = "1"

    def test_convenience_function_decoder_error(self, mock_controller):
        """Test convenience function handling of decoder errors."""
        # Mock controller that will cause decoder to fail
        mock_controller.peek.return_value = None

        result = decode_first_mon(mock_controller)

        assert result is None

    @pytest.mark.parametrize("field_name,expected_value", [
        ("species_id", 42),
        ("level", 15),
        ("hp_current", 200),
        ("hp_max", 250),
        ("status", 1),
        ("affiliation", 0),
        ("tile_x", 10),
        ("tile_y", 12),
        ("direction", 2),
        ("visible", 1),
    ])
    def test_field_parsing(self, decoder, mock_controller, field_name, expected_value):
        """Test parsing of individual fields."""
        # Create mock monster data with specific field value
        monster_data = bytearray(MONSTER_STRUCT_SIZE)

        field_def = decoder._parse_field.__globals__["MONSTER_FIELDS"][field_name]
        offset = field_def["offset"]
        size = field_def["size"]
        field_type = field_def["type"]

        # Set the field value in monster data
        if field_type == "uint16":
            monster_data[offset:offset+size] = expected_value.to_bytes(2, 'little')
        elif field_type == "uint8":
            monster_data[offset:offset+size] = expected_value.to_bytes(1, 'little')

        # Mock the reads
        mock_controller.peek.side_effect = [
            b'\x39\x41\x00\x02',  # list_ptr
            b'\x01',              # count = 1
            bytes(monster_data),  # monster struct
        ]

        result = decoder.decode_first_mon()

        assert result is not None
        assert result[field_name] == expected_value


class TestWRAMDecoderV2Integration:
    """Integration tests for WRAMDecoderV2."""

    def test_decoder_initialization_requires_feature_flag(self, mock_controller):
        """Test that decoder requires MD_DECODER_V2=1."""
        # Temporarily disable feature flag
        os.environ["MD_DECODER_V2"] = "0"

        try:
            with pytest.raises(RuntimeError, match="WRAMDecoderV2 requires MD_DECODER_V2=1"):
                WRAMDecoderV2(mock_controller)
        finally:
            # Restore feature flag
            os.environ["MD_DECODER_V2"] = "1"

    def test_decoder_initialization_success(self, mock_controller):
        """Test successful decoder initialization."""
        decoder = WRAMDecoderV2(mock_controller)
        assert decoder.controller == mock_controller
        assert decoder.address_manager == mock_controller.address_manager

    def test_contiguous_read_method(self, decoder, mock_controller):
        """Test the _read_contiguous helper method."""
        mock_controller.peek.return_value = b'\x01\x02\x03\x04'

        result = decoder._read_contiguous(0x02000000, 4)

        assert result == b'\x01\x02\x03\x04'
        mock_controller.peek.assert_called_once_with(0x02000000, 4)

    def test_contiguous_read_failure(self, decoder, mock_controller):
        """Test _read_contiguous with read failure."""
        mock_controller.peek.return_value = None

        result = decoder._read_contiguous(0x02000000, 4)

        assert result is None

    def test_contiguous_read_wrong_size(self, decoder, mock_controller):
        """Test _read_contiguous with wrong returned size."""
        mock_controller.peek.return_value = b'\x01\x02'  # Only 2 bytes instead of 4

        result = decoder._read_contiguous(0x02000000, 4)

        assert result is None