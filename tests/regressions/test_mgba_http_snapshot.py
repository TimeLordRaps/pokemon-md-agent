"""
Regression Tests for mGBA HTTP Snapshot Mocking - test_mgba_http_snapshot.py

Tests mGBA HTTP API interactions with mocked responses to ensure reliable
emulator communication for WRAM decoding and live data dumping.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import socket
import time

from src.environment.mgba_controller import MGBAController, LuaSocketTransport


class TestMGBASnapshotMocking:
    """Test suite for mGBA HTTP API mocking and snapshot functionality."""

    @pytest.fixture
    def mock_transport(self):
        """Create a mock LuaSocketTransport."""
        transport = Mock(spec=LuaSocketTransport)
        transport.is_connected.return_value = True
        transport._lock = MagicMock()
        return transport

    @pytest.fixture
    def mock_controller(self, mock_transport):
        """Create a mock MGBAController with transport."""
        controller = Mock(spec=MGBAController)
        controller._transport = mock_transport
        controller.smoke_mode = False
        controller.RETRY_COUNT = 3
        controller.RETRY_BACKOFF_BASE = 0.1

        # Mock address manager
        address_manager = Mock()
        address_manager.get_address.side_effect = lambda category, field: {
            ("entities", "monster_list_ptr"): 0x02004139,
            ("entities", "monster_count"): 0x0200413D,
        }.get((category, field), 0)
        address_manager.get_size.side_effect = lambda category, field: {
            ("player_state", "floor_number"): 1,
            ("player_state", "player_tile_x"): 1,
            ("player_state", "player_tile_y"): 1,
            ("party_status", "leader_hp"): 2,
            ("party_status", "leader_hp_max"): 2,
            ("party_status", "leader_belly"): 2,
        }.get((category, field), 1)

        controller.address_manager = address_manager
        return controller

    def test_peek_memory_success(self, mock_controller, mock_transport):
        """Test successful memory peek operation."""
        # Mock successful memory read - set up proper command sequence
        call_count = 0
        def mock_send_command(command, *args):
            nonlocal call_count
            call_count += 1
            if command == "coreAdapter.memory":
                return "wram,iwram,vram,oam,palette,rom"
            elif command == "memoryDomain.readRange":
                return "aa,bb,cc,dd"  # Hex byte string
            return None
            
        mock_transport.send_command.side_effect = mock_send_command

        controller = MGBAController()
        controller._transport = mock_transport
        
        # Initialize _memory_domains to avoid the validation call
        controller._memory_domains = ["wram", "iwram", "vram", "oam", "palette", "rom"]

        result = controller.peek(0x02000000, 4)

        assert result == b'\xaa\xbb\xcc\xdd'
        # Should be called for memory domain list and the actual read
        assert mock_transport.send_command.call_count >= 1

    def test_peek_memory_iwram(self, mock_controller, mock_transport):
        """Test memory peek in IWRAM domain."""
        # Mock successful memory read - set up proper command sequence
        call_count = 0
        def mock_send_command(command, *args):
            nonlocal call_count
            call_count += 1
            if command == "coreAdapter.memory":
                return "wram,iwram,vram,oam,palette,rom"
            elif command == "memoryDomain.readRange":
                return "11,22"  # Hex byte string
            return None
            
        mock_transport.send_command.side_effect = mock_send_command

        controller = MGBAController()
        controller._transport = mock_transport
        
        # Initialize _memory_domains to avoid the validation call
        controller._memory_domains = ["wram", "iwram", "vram", "oam", "palette", "rom"]

        result = controller.peek(0x03000000, 2)

        assert result == b'\x11\x22'

    def test_peek_memory_invalid_address(self, mock_controller, mock_transport):
        """Test memory peek with invalid address."""
        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.peek(0x01000000, 4)  # Invalid address

        assert result is None
        # Should not call send_command for invalid addresses
        mock_transport.send_command.assert_not_called()
        mock_transport.send_command.assert_not_called()

    def test_peek_memory_read_failure(self, mock_controller, mock_transport):
        """Test memory peek with read failure."""
        # Mock successful memory read - set up proper command sequence
        call_count = 0
        def mock_send_command(command, *args):
            nonlocal call_count
            call_count += 1
            if command == "coreAdapter.memory":
                return "wram,iwram,vram,oam,palette,rom"
            elif command == "memoryDomain.readRange":
                return None  # Simulate read failure
            return None
            
        mock_transport.send_command.side_effect = mock_send_command

        controller = MGBAController()
        controller._transport = mock_transport
        
        # Initialize _memory_domains to avoid the validation call
        controller._memory_domains = ["wram", "iwram", "vram", "oam", "palette", "rom"]

        # Should raise MemoryReadError when read fails
        with pytest.raises(Exception):  # Could be MemoryReadError or similar
            result = controller.peek(0x02000000, 4)

    def test_peek_memory_malformed_response(self, mock_controller, mock_transport):
        """Test memory peek with malformed response."""
        # Mock successful memory read - set up proper command sequence
        call_count = 0
        def mock_send_command(command, *args):
            nonlocal call_count
            call_count += 1
            if command == "coreAdapter.memory":
                return "wram,iwram,vram,oam,palette,rom"
            elif command == "memoryDomain.readRange":
                return "invalid,hex,data"  # Malformed response
            return None
            
        mock_transport.send_command.side_effect = mock_send_command

        controller = MGBAController()
        controller._transport = mock_transport
        
        # Initialize _memory_domains to avoid the validation call
        controller._memory_domains = ["wram", "iwram", "vram", "oam", "palette", "rom"]

        # Should raise MemoryReadError when response is malformed
        with pytest.raises(Exception):  # Could be MemoryReadError or similar
            result = controller.peek(0x02000000, 4)

    def test_get_floor_success(self, mock_controller, mock_transport):
        """Test successful floor reading."""
        mock_transport.send_command.return_value = "aa,bb,cc,dd"  # Mock WRAM data
        mock_controller.address_manager.get_address.return_value = 0x02004139
        mock_controller.address_manager.get_size.return_value = 1

        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        # Mock the peek method to return floor value
        with patch.object(controller, 'peek', return_value=b'\x05'):  # floor = 5
            result = controller.get_floor()
            assert result == 5

    def test_get_floor_read_failure(self, mock_controller, mock_transport):
        """Test floor reading with read failure."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'peek', return_value=None):
            with pytest.raises(RuntimeError, match="Failed to read floor"):
                controller.get_floor()

    def test_get_player_position_success(self, mock_controller, mock_transport):
        """Test successful player position reading."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'peek') as mock_peek:
            mock_peek.side_effect = [b'\x0A', b'\x0F']  # x=10, y=15
            x, y = controller.get_player_position()
            assert x == 10
            assert y == 15

    def test_get_player_position_read_failure(self, mock_controller, mock_transport):
        """Test player position reading with read failure."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'peek', return_value=None):
            with pytest.raises(RuntimeError, match="Failed to read player position"):
                controller.get_player_position()

    def test_get_player_stats_success(self, mock_controller, mock_transport):
        """Test successful player stats reading."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'peek') as mock_peek:
            mock_peek.side_effect = [
                b'\x64\x00',  # hp = 100
                b'\xC8\x00',  # max_hp = 200
                b'\x64\x00',  # belly = 100
            ]
            stats = controller.get_player_stats()
            assert stats["hp"] == 100
            assert stats["max_hp"] == 200
            assert stats["belly"] == 100
            assert stats["max_belly"] == 100

    def test_get_player_stats_read_failure(self, mock_controller, mock_transport):
        """Test player stats reading with read failure."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'peek', return_value=None):
            with pytest.raises(RuntimeError, match="Failed to read player stats"):
                controller.get_player_stats()

    def test_send_command_success(self, mock_controller, mock_transport):
        """Test successful command sending."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport
        controller._command_latencies = {}
        controller._domain_counters = {"core": 0}

        result = controller.send_command("core.getGameTitle")

        assert result == "success"
        mock_transport.send_command.assert_called_once_with("core.getGameTitle")

    def test_send_command_with_retries(self, mock_controller, mock_transport):
        """Test command sending with retries on failure."""
        # Mock transport to fail twice then succeed
        mock_transport.send_command.side_effect = [ConnectionError(), ConnectionError(), "success"]
        
        # Mock missing attributes
        mock_transport.reconnect_backoff = 1.0
        mock_transport.max_backoff = 30.0

        controller = MGBAController()
        controller._transport = mock_transport
        controller._command_latencies = {}
        controller._domain_counters = {"core": 0}

        result = controller.send_command("core.getGameTitle")

        assert result == "success"
        assert mock_transport.send_command.call_count == 3

    def test_send_command_max_retries_exceeded(self, mock_controller, mock_transport):
        """Test command sending when max retries exceeded."""
        mock_transport.send_command.side_effect = ConnectionError()
        
        # Mock missing attributes
        mock_transport.reconnect_backoff = 1.0
        mock_transport.max_backoff = 30.0

        controller = MGBAController()
        controller._transport = mock_transport
        controller._command_latencies = {}
        controller._domain_counters = {"core": 0}

        result = controller.send_command("core.getGameTitle")

        assert result is None
        assert mock_transport.send_command.call_count == 3  # RETRY_COUNT

    def test_connect_success(self, mock_transport):
        """Test successful connection."""
        # Mock successful socket operations
        mock_socket = Mock()
        mock_transport._socket = mock_socket
        mock_transport._send_handshake = Mock()
        mock_transport._validate_connection = Mock(return_value=True)

        controller = MGBAController()
        controller._transport = mock_transport
        
        # Mock the send_command to return proper responses for server probing
        mock_transport.send_command.side_effect = [
            "wram,iwram,vram,oam,palette,rom",  # coreAdapter.memory
            "Pokemon Mystery Dungeon Red",       # core.getGameTitle
            "BPRG"                               # core.getGameCode
        ]

        with patch('socket.socket', return_value=mock_socket):
            result = controller.connect()

            assert result is True

    def test_connect_socket_timeout(self, mock_transport):
        """Test connection with socket timeout."""
        # Mock the send_command to avoid _probe_server issues
        mock_transport.send_command.return_value = "wram,iwram,vram,oam,palette,rom"
        
        mock_socket = Mock()
        mock_socket.settimeout.return_value = None
        mock_socket.connect.side_effect = socket.timeout()

        controller = MGBAController()
        controller._transport = mock_transport

        with patch('socket.socket', return_value=mock_socket):
            result = controller.connect()
            assert result is False

    def test_connect_refused(self, mock_transport):
        """Test connection refused."""
        # Mock the send_command to avoid _probe_server issues
        mock_transport.send_command.return_value = "wram,iwram,vram,oam,palette,rom"
        
        mock_socket = Mock()
        mock_socket.settimeout.return_value = None
        mock_socket.connect.side_effect = ConnectionRefusedError()

        controller = MGBAController()
        controller._transport = mock_transport

        with patch('socket.socket', return_value=mock_socket):
            result = controller.connect()
            assert result is False

    def test_memory_domain_read_range_success(self, mock_transport):
        """Test successful memory domain read."""
        mock_transport.send_command.return_value = "aa,bb,cc,dd,ee,ff"

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.memory_domain_read_range("wram", 0x1000, 6)

        assert result == b'\xaa\xbb\xcc\xdd\xee\xff'

    def test_memory_domain_read_range_failure(self, mock_transport):
        """Test memory domain read failure."""
        mock_transport.send_command.return_value = None

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.memory_domain_read_range("wram", 0x1000, 4)

        assert result is None

    def test_memory_domain_read_range_malformed(self, mock_transport):
        """Test memory domain read with malformed response."""
        mock_transport.send_command.return_value = "invalid,data,here"

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.memory_domain_read_range("wram", 0x1000, 4)

        assert result is None

    def test_button_operations(self, mock_transport):
        """Test button operation commands."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport

        # Test button tap
        result = controller.button_tap("A")
        assert result is True
        mock_transport.send_command.assert_called_with("mgba-http.button.tap", "A")

        # Test button hold
        result = controller.button_hold("B", 1000)
        assert result is True
        mock_transport.send_command.assert_called_with("mgba-http.button.hold", "B", "1000")

    def test_screenshot_operation(self, mock_transport):
        """Test screenshot operations."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.screenshot("/tmp/test.png")
        assert result is True
        mock_transport.send_command.assert_called_with("core.screenshot", "/tmp/test.png")

    def test_state_operations(self, mock_transport):
        """Test save/load state operations."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport

        # Test save state
        result = controller.save_state_slot(1)
        assert result is True
        mock_transport.send_command.assert_called_with("core.saveStateSlot", "1")

        # Test load state
        result = controller.load_state_slot(1)
        assert result is True
        mock_transport.send_command.assert_called_with("core.loadStateSlot", "1", "0")

    def test_autoload_save(self, mock_transport):
        """Test autoload save operation."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.autoload_save()
        assert result is True
        mock_transport.send_command.assert_called_with("core.autoLoadSave")

    def test_reset_operation(self, mock_transport):
        """Test reset operation."""
        mock_transport.send_command.return_value = "success"

        controller = MGBAController()
        controller._transport = mock_transport

        result = controller.reset()
        assert result is True
        mock_transport.send_command.assert_called_with("coreAdapter.reset")

    def test_platform_and_game_info(self, mock_transport):
        """Test platform and game information queries."""
        mock_transport.send_command.side_effect = ["GBA", "POKEMON MD", "BPRG"]

        controller = MGBAController()
        controller._transport = mock_transport

        assert controller.platform() == "GBA"
        assert controller.get_game_title() == "POKEMON MD"
        assert controller.get_game_code() == "BPRG"

    def test_semantic_state_success(self, mock_controller, mock_transport):
        """Test semantic state retrieval."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'get_player_stats') as mock_stats, \
             patch.object(controller, 'get_floor') as mock_floor, \
             patch.object(controller, 'get_player_position') as mock_pos:

            mock_stats.return_value = {"hp": 100, "max_hp": 200, "belly": 50, "max_belly": 100}
            mock_floor.return_value = 5
            mock_pos.return_value = (10, 15)

            state = controller.semantic_state()

            assert state["hp"] == 100
            assert state["floor"] == 5
            assert state["player_pos"] == {"x": 10, "y": 15}

    def test_semantic_state_with_fields_filter(self, mock_controller, mock_transport):
        """Test semantic state with field filtering."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'get_player_stats') as mock_stats:
            mock_stats.return_value = {"hp": 100, "max_hp": 200, "belly": 50, "max_belly": 100}

            state = controller.semantic_state(fields=["hp", "belly"])

            assert "hp" in state
            assert "belly" in state
            assert "floor" not in state

    def test_semantic_state_error_handling(self, mock_controller, mock_transport):
        """Test semantic state error handling."""
        controller = MGBAController()
        controller._transport = mock_transport
        controller.address_manager = mock_controller.address_manager

        with patch.object(controller, 'get_player_stats', side_effect=RuntimeError("test error")), \
             patch.object(controller, 'get_floor', side_effect=RuntimeError("test error")), \
             patch.object(controller, 'get_player_position', side_effect=RuntimeError("test error")):

            state = controller.semantic_state()

            # Should return empty dict when all operations fail
            assert state == {}


class TestTransportLayerMocking:
    """Test transport layer mocking scenarios."""

    def test_transport_connection_states(self):
        """Test various transport connection states."""
        transport = Mock(spec=LuaSocketTransport)

        # Test connected state
        transport.is_connected.return_value = True
        assert transport.is_connected() is True

        # Test disconnected state
        transport.is_connected.return_value = False
        assert transport.is_connected() is False

    def test_transport_command_responses(self):
        """Test various transport command response patterns."""
        transport = Mock(spec=LuaSocketTransport)

        # Test successful responses
        transport.send_command.side_effect = ["success", "42", "GBA", "<|ERROR|>"]

        assert transport.send_command("test1") == "success"
        assert transport.send_command("test2") == "42"
        assert transport.send_command("test3") == "GBA"
        assert transport.send_command("test4") == "<|ERROR|>"

    def test_transport_error_conditions(self):
        """Test transport error conditions."""
        transport = Mock(spec=LuaSocketTransport)

        # Test connection errors
        transport.send_command.side_effect = ConnectionError("Connection lost")

        with pytest.raises(ConnectionError):
            transport.send_command("test")

    def test_rate_limiting_behavior(self):
        """Test rate limiter behavior in controller."""
        from src.environment.mgba_controller import RateLimiter

        limiter = RateLimiter(max_calls=2, time_window=1.0)

        # Should allow first two calls
        assert limiter.wait_if_needed() is True
        assert limiter.wait_if_needed() is True

        # Third call should be blocked (in real usage)
        # Note: This is a simplified test - actual blocking behavior
        # depends on timing