"""Test mgba Lua Socket connection and basic functionality."""

import pytest
import socket
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mgba_controller import MGBAController, ScreenshotData


def test_mgba_controller_initialization():
    """Test mgba controller initialization."""
    controller = MGBAController(host="localhost", port=8888, timeout=10.0)
    assert controller.host == "localhost"
    assert controller.port == 8888
    assert controller.timeout == 10.0
    assert controller.TERMINATION_MARKER == "<|END|>"


@patch('socket.socket')
@patch('threading.Lock')
def test_mgba_controller_connect_success(mock_lock, mock_socket_class):
    """Test successful connection to mgba Lua socket server."""
    # Setup mocks
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    
    # Configure socket mock to simulate successful connection
    # Note: _probe_server calls getGameTitle, getGameCode, then memory
    # But in actual implementation, memory is called first, then title, then code
    mock_socket_instance.recv.side_effect = [
        b"WRAM,VRAM,OAM,PALETTE,ROM<|END|>",  # memory domains response
        b"POKEMON MYSTERY DUNGEON - RED RESCUE TEAM<|END|>",  # getGameTitle response
        b"IREX<|END|>"  # getGameCode response
    ]
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    
    # Test connection
    result = controller.connect()
    
    assert result is True
    assert controller.is_connected() is True
    assert controller._game_title == "POKEMON MYSTERY DUNGEON - RED RESCUE TEAM"
    assert controller._game_code == "IREX"
    assert "WRAM" in controller._memory_domains


@patch('socket.socket')
@patch('threading.Lock')
def test_mgba_controller_connect_failure(mock_lock, mock_socket_class):
    """Test connection failure."""
    mock_lock.return_value = MagicMock()
    mock_socket_class.side_effect = ConnectionRefusedError("Connection refused")
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    result = controller.connect()
    
    assert result is False
    assert controller.is_connected() is False


@patch('socket.socket')
@patch('threading.Lock')
def test_send_command(mock_lock, mock_socket_class):
    """Test sending commands via Lua Socket API."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    
    # Mock recv to return a response
    mock_socket_instance.recv.return_value = b"success<|END|>"
    mock_socket_instance.sendall = MagicMock()
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance  # Simulate connected state
    
    # Test command send
    response = controller.send_command("core.getGameTitle")
    
    assert response == "success"
    
    # Verify the message was sent with correct format
    mock_socket_instance.sendall.assert_called_once()
    sent_data = mock_socket_instance.sendall.call_args[0][0].decode('utf-8')
    assert "core.getGameTitle" in sent_data
    assert controller.TERMINATION_MARKER in sent_data


@patch('socket.socket')
@patch('threading.Lock')
def test_button_tap(mock_lock, mock_socket_class):
    """Test button tap functionality."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"OK<|END|>"
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    # Test button tap
    result = controller.button_tap("A")
    
    assert result is True
    
    # Verify correct command format
    mock_socket_instance.sendall.assert_called_once()
    sent_data = mock_socket_instance.sendall.call_args[0][0].decode('utf-8')
    assert "mgba-http.button.tap" in sent_data
    assert "A" in sent_data


@patch('socket.socket')
@patch('threading.Lock')
def test_button_hold(mock_lock, mock_socket_class):
    """Test button hold functionality."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"OK<|END|>"
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    # Test button hold
    result = controller.button_hold("B", 500)
    
    assert result is True
    
    # Verify correct command format
    sent_data = mock_socket_instance.sendall.call_args[0][0].decode('utf-8')
    assert "mgba-http.button.hold" in sent_data
    assert "500" in sent_data


@patch('socket.socket')
@patch('threading.Lock')
def test_memory_read_operations(mock_lock, mock_socket_class):
    """Test memory reading operations."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    
    # Test read8
    mock_socket_instance.recv.return_value = b"42<|END|>"
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    value = controller.memory_domain_read8("WRAM", 0x2000000)
    assert value == 42
    
    # Test read16
    mock_socket_instance.recv.return_value = b"1000<|END|>"
    value = controller.memory_domain_read16("WRAM", 0x2000000)
    assert value == 1000
    
    # Test read32
    mock_socket_instance.recv.return_value = b"123456<|END|>"
    value = controller.memory_domain_read32("WRAM", 0x2000000)
    assert value == 123456


@patch('socket.socket')
@patch('threading.Lock')
def test_memory_read_range(mock_lock, mock_socket_class):
    """Test memory range read operation."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"[aa,bb,cc,dd,ee]<|END|>"
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    data = controller.memory_domain_read_range("WRAM", 0x2000000, 5)
    
    assert data == b"\xaa\xbb\xcc\xdd\xee"


@patch('socket.socket')
@patch('threading.Lock')
def test_screenshot(mock_lock, mock_socket_class):
    """Test screenshot functionality."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"OK<|END|>"

    # Test default scale=2 (480×320)
    controller = MGBAController(host="localhost", port=8888, timeout=1.0, capture_scale=2)
    controller._transport.is_connected = lambda: True  # Mock transport as connected
    controller._socket = mock_socket_instance

    result = controller.screenshot("/tmp/test.png")

    assert result is True

    # Verify screenshot command with scale parameters
    sent_data = mock_socket_instance.sendall.call_args[0][0].decode('utf-8')
    assert "core.screenshot" in sent_data
    assert "/tmp/test.png" in sent_data
    assert "480" in sent_data  # width for 2x scale (240*2=480)
    assert "320" in sent_data  # height for 2x scale (160*2=320)

    # Reset mock for scale=1 test
    mock_socket_instance.sendall.reset_mock()

    # Test scale=1 (240×160)
    controller = MGBAController(host="localhost", port=8888, timeout=1.0, capture_scale=1)
    controller._transport.is_connected = lambda: True  # Mock transport as connected
    controller._socket = mock_socket_instance

    result = controller.screenshot("/tmp/test.png")

    assert result is True

    # Verify screenshot command without additional parameters
    sent_data = mock_socket_instance.sendall.call_args[0][0].decode('utf-8')
    assert "core.screenshot" in sent_data
    assert "/tmp/test.png" in sent_data
    # Should not contain scale parameters for scale=1
    assert "320" not in sent_data
    assert "480" not in sent_data


@patch('socket.socket')
@patch('threading.Lock')
def test_save_load_state_operations(mock_lock, mock_socket_class):
    """Test save and load state operations."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"OK<|END|>"
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    # Test save state slot
    result = controller.save_state_slot(1)
    assert result is True
    
    # Test load state slot
    result = controller.load_state_slot(1)
    assert result is True


@patch('socket.socket')
@patch('threading.Lock')
def test_context_manager(mock_lock, mock_socket_class):
    """Test context manager usage."""
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.side_effect = [
        b"POKEMON<|END|>",
        b"IREX<|END|>",
        b"WRAM,VRAM<|END|>"
    ]
    
    with MGBAController(host="localhost", port=8888, timeout=1.0) as controller:
        assert controller.is_connected() is True
    
    # Socket should be closed after context exit
    assert not controller.is_connected()


@patch('socket.socket')
@patch('threading.Lock')
def test_rate_limiter(mock_lock, mock_socket_class):
    """Test rate limiting functionality."""
    import time
    
    mock_lock.return_value = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_class.return_value = mock_socket_instance
    mock_socket_instance.recv.return_value = b"OK<|END|>"
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    controller._socket = mock_socket_instance
    
    # Test screenshot rate limiting
    start = time.time()
    for _ in range(35):  # Exceed 30/s limit
        controller.screenshot(f"/tmp/test_{_}.png")
    elapsed = time.time() - start
    
    # Should take at least 1 second due to rate limiting
    assert elapsed >= 1.0


if __name__ == "__main__":
    # Run basic test without pytest
    print("Testing mgba Lua Socket controller...")
    
    controller = MGBAController(host="localhost", port=8888, timeout=1.0)
    print(f"Controller initialized: {controller.host}:{controller.port}")
    print(f"Termination marker: {controller.TERMINATION_MARKER}")
    
    # Test connection check (will fail if mgba not running)
    try:
        is_connected = controller.connect()
        print(f"Connection status: {is_connected}")
        
        if is_connected:
            print(f"Game title: {controller._game_title}")
            print(f"Game code: {controller._game_code}")
            print(f"Memory domains: {controller._memory_domains}")
    except Exception as e:
        print(f"Connection failed (expected if mgba not running): {e}")
        print("Note: Start mgba with: mgba --http-server --port 8888 path/to/rom.gba")
        print("Make sure mGBASocketServer.lua is loaded")
    
    print("\nBasic tests completed!")
