"""mgba Lua Socket API controller for Pokemon MD emulator integration."""

from typing import Optional, Dict, List
from dataclasses import dataclass
from dataclasses import field
import logging
import socket
import time
import threading
from pathlib import Path
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotData:
    """Screenshot data from mgba."""
    image_data: bytes
    width: int
    height: int
    timestamp: float


@dataclass
class RateLimiter:
    """Simple rate limiter for command execution."""
    max_calls: int
    time_window: float  # seconds
    
    _calls: deque = field(default_factory=deque)
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old calls outside time window
        while self._calls and now - self._calls[0] > self.time_window:
            self._calls.popleft()
        
        # If at limit, wait
        if len(self._calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self._calls[0]) + 0.01
            if sleep_time > 0:
                logger.debug("Rate limit reached, sleeping %.2fs", sleep_time)
                time.sleep(sleep_time)
                return self.wait_if_needed()
        
        # Record this call
        self._calls.append(now)
class LuaSocketTransport:
    """Lua socket transport with <|END|> framing and line-safe buffering."""

    TERMINATION_MARKER = "<|END|>"

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._buffer = ""
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to the Lua socket server."""
        with self._lock:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))
                self._buffer = ""

                # Optional handshake
                self._send_handshake()

                logger.info("LuaSocketTransport connected to %s:%d", self.host, self.port)
                return True

            except socket.timeout:
                logger.error("Connection timeout to %s:%d", self.host, self.port)
                return False
            except ConnectionRefusedError:
                logger.error("Connection refused to %s:%d", self.host, self.port)
                return False
            except OSError as e:
                logger.error("Connection failed: %s", e)
                return False

    def _send_handshake(self) -> None:
        """Send optional handshake to confirm readiness."""
        try:
            self._send_raw("<|ACK|>" + self.TERMINATION_MARKER)
            logger.debug("Handshake sent")
        except (OSError, ConnectionError) as e:
            logger.warning("Handshake failed: %s", e)

    def send(self, command: str, *args: str | bytes) -> str:
        """Send command and return response with error on failure.

        Args:
            command: Command type
            args: Command arguments

        Returns:
            Response string

        Raises:
            ConnectionError: If command fails
        """
        response = self.send_command(command, *args)
        if response is None:
            raise ConnectionError(f"Command {command} failed")
        return response

    def send_command(self, command: str, *args: str | bytes) -> Optional[str]:
        """Send command and get response.

        Args:
            command: Command type
            args: Command arguments

        Returns:
            Response string or None if failed
        """
        # Serialize message as "type,arg1,arg2,...<|END|>"
        message_parts = [command]
        for arg in args:
            if isinstance(arg, bytes):
                hex_bytes = ",".join(f"{b:02x}" for b in arg)
                message_parts.append(f"[{hex_bytes}]")
            else:
                message_parts.append(str(arg))

        message = ",".join(message_parts) + self.TERMINATION_MARKER
        return self._send_raw(message)

    def _send_raw(self, message: str) -> Optional[str]:
        """Send raw message with partial-read loop and buffering."""
        with self._lock:
            if not self._socket:
                logger.error("Not connected")
                return None

            try:
                start_time = time.time()

                # Send message
                logger.debug("Sending: %s", message[:100])
                self._socket.sendall(message.encode('utf-8'))

                # Partial-read loop with line-safe buffering
                while self.TERMINATION_MARKER not in self._buffer:
                    try:
                        chunk = self._socket.recv(4096)
                        if not chunk:
                            logger.error("Connection closed by server")
                            self.disconnect()
                            return None
                        self._buffer += chunk.decode('utf-8', errors='ignore')
                    except socket.timeout:
                        logger.warning("Timeout during partial read")
                        continue

                # Extract response
                marker_pos = self._buffer.find(self.TERMINATION_MARKER)
                response = self._buffer[:marker_pos]
                self._buffer = self._buffer[marker_pos + len(self.TERMINATION_MARKER):]

                latency = time.time() - start_time
                logger.debug("Response latency: %.3fs", latency)
                logger.debug("Response: %s", response[:100])

                return response

            except (OSError, ConnectionError) as e:
                logger.error("Send failed: %s", e)
                self.disconnect()
                return None

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._socket is not None

    def disconnect(self) -> None:
        """Disconnect from server."""
        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None
                logger.info("LuaSocketTransport disconnected")



class MGBAController:
    """Controller for mgba emulator via Lua Socket API (mGBASocketServer 0.8.0)."""

    DEFAULT_TIMEOUT = 10.0
    RETRY_COUNT = 3
    RETRY_BACKOFF = 1.5

    # Rate limiters
    SCREENSHOT_LIMIT = RateLimiter(max_calls=30, time_window=1.0)  # 30/s max
    MEMORY_LIMIT = RateLimiter(max_calls=10, time_window=1.0)  # 10/s max
    COMMAND_LIMIT = RateLimiter(max_calls=60, time_window=1.0)  # 60/s max
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        timeout: float = 10.0,
        cache_dir: Optional[Path] = None,
        capture_scale: int = 2,
    ):
        """Initialize mgba controller.

        Args:
            host: mgba Lua socket server host
            port: mgba Lua socket server port (will auto-bump if busy)
            timeout: Socket timeout in seconds
            cache_dir: Directory for caching server info
            capture_scale: Screen capture scale factor (1=240×160, 2=320×480)
        """
        self.host = host
        self.port = self._find_available_port(port)
        self.timeout = timeout
        self.cache_dir = cache_dir or Path.home() / ".cache" / "pmd-red"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.capture_scale = capture_scale

        self._transport = LuaSocketTransport(self.host, self.port, self.timeout)
        self._server_version = "0.8.0"  # Fixed server version
        self._game_title = None
        self._game_code = None
        self._memory_domains: Optional[List[str]] = None

        # Metrics
        self._command_latencies: Dict[str, List[float]] = {}
        self._domain_counters: Dict[str, int] = {"memory": 0, "button": 0, "core": 0, "screenshot": 0}

        logger.info("Initialized MGBAController at %s:%d (scale=%dx)", self.host, self.port, self.capture_scale)

    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.bind((self.host, port))
                test_socket.close()
                return port
            except OSError:
                port += 1
        raise RuntimeError(f"No available ports found starting from {start_port}")

    def connect(self) -> bool:
        """Connect to mgba Lua socket server.

        Returns:
            True if connection succeeded
        """
        try:
            if self._transport.connect():
                # Probe server capabilities
                self._probe_server()

                # Cache connection info
                self._save_connection_cache()

                return True
        except (OSError, ConnectionError) as e:
            logger.error("Connection failed: %s", e)
            return False

        return False
    
    def _probe_server(self) -> None:
        """Probe server for capabilities and version info."""
        try:
            # Get memory domains
            response = self.send_command("coreAdapter.memory")
            if response:
                self._memory_domains = [d.strip() for d in response.split(",") if d.strip()]
                logger.info("Available memory domains: %s", self._memory_domains)
            
            # Get game title
            self._game_title = self.send_command("core.getGameTitle")
            logger.info("Game title: %s", self._game_title)
            
            # Get game code
            self._game_code = self.send_command("core.getGameCode")
            logger.info("Game code: %s", self._game_code)
            
            # Server version is known from Lua script (0.8.0)
            self._server_version = "0.8.0"
            
        except (OSError, ConnectionError) as e:
            logger.warning("Server probe failed: %s", e)
    
    def _save_connection_cache(self) -> None:
        """Save connection info to cache."""
        cache_file = self.cache_dir / "mgba-http.json"
        cache_data = {
            "host": self.host,
            "port": self.port,
            "server_version": self._server_version,
            "game_title": self._game_title,
            "game_code": self._game_code,
            "memory_domains": self._memory_domains,
            "last_connected": time.time(),
            "command_counts": self._domain_counters,
            "average_latencies": {k: sum(v)/len(v) if v else 0 for k, v in self._command_latencies.items()},
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except (OSError, IOError) as e:
            logger.debug("Failed to save connection cache: %s", e)
    
    def send_command(self, command: str, *args: str | bytes) -> Optional[str]:
        """Send command to mgba and get response with resilience.

        Args:
            command: Command type (e.g., "core.getGameTitle")
            args: Command arguments

        Returns:
            Response string or None if failed
        """
        # Track metrics
        domain = command.split('.')[0] if '.' in command else 'unknown'
        self._domain_counters[domain] += 1

        # Use transport with retries and backoff
        for attempt in range(self.RETRY_COUNT):
            try:
                start_time = time.time()
                response = self._transport.send_command(command, *args)
                latency = time.time() - start_time

                # Log structured metrics
                logger.info("Command %s latency: %.3fs", command, latency)
                if command not in self._command_latencies:
                    self._command_latencies[command] = []
                self._command_latencies[command].append(latency)

                return response

            except ConnectionError as e:
                logger.warning("Connection error on attempt %d/%d: %s", attempt + 1, self.RETRY_COUNT, e)
                if attempt == self.RETRY_COUNT - 1:
                    logger.error("Command failed after retries: %s", command)
                    return None
                # Exponential backoff
                backoff = self.RETRY_BACKOFF ** attempt * 0.1
                time.sleep(backoff)

        return None

    def send(self, command: str, *args: str | bytes) -> str:
        """Send command to mgba and get response with error on failure.

        Args:
            command: Command type (e.g., "core.getGameTitle")
            args: Command arguments

        Returns:
            Response string

        Raises:
            ConnectionError: If command fails
        """
        response = self.send_command(command, *args)
        if response is None:
            raise ConnectionError(f"Command {command} failed")
        return response
    
    
    def is_connected(self) -> bool:
        """Check if connected to mgba server.

        Returns:
            True if socket is active
        """
        return self._transport.is_connected()

    def disconnect(self) -> None:
        """Disconnect from mgba server."""
        self._transport.disconnect()
        logger.info("Disconnected from mgba server")
    
    # Core API methods
    
    def get_game_title(self) -> Optional[str]:
        """Get game title.
        
        Returns:
            Game title string
        """
        return self.send_command("core.getGameTitle")
    
    def get_game_code(self) -> Optional[str]:
        """Get game code.
        
        Returns:
            Game code string
        """
        return self.send_command("core.getGameCode")
    
    def screenshot(self, path: str) -> bool:
        """Save screenshot to file.

        Args:
            path: File path to save screenshot

        Returns:
            True if screenshot saved
        """
        self.SCREENSHOT_LIMIT.wait_if_needed()
        self._domain_counters['screenshot'] += 1

        # Scale screenshot if capture_scale > 1
        if self.capture_scale > 1:
            # mGBA scales by integer factors; 2x means 320×480 instead of 240×160
            scaled_width = 240 * self.capture_scale  # 240 * 2 = 480
            scaled_height = 160 * self.capture_scale  # 160 * 2 = 320
            response = self.send_command("core.screenshot", path, str(scaled_width), str(scaled_height))
        else:
            response = self.send_command("core.screenshot", path)

        return bool(response and response != "<|ERROR|>")
    
    def current_frame(self) -> Optional[int]:
        """Get current frame number.
        
        Returns:
            Frame number or None
        """
        response = self.send_command("core.currentFrame")
        try:
            return int(response) if response else None
        except (ValueError, TypeError):
            return None
    
    def autoload_save(self) -> bool:
        """Autoload save file.
        
        Returns:
            True if successful
        """
        response = self.send_command("core.autoloadSave")
        return bool(response and response != "<|ERROR|>")
    
    def save_state_file(self, path: str, slot: int) -> bool:
        """Save state to file.
        
        Args:
            path: File path
            slot: Save slot
            
        Returns:
            True if successful
        """
        response = self.send_command("core.saveStateFile", path, str(slot))
        return bool(response and response != "<|ERROR|>")
    
    def load_state_file(self, path: str, slot: int) -> bool:
        """Load state from file.
        
        Args:
            path: File path
            slot: Save slot
            
        Returns:
            True if successful
        """
        response = self.send_command("core.loadStateFile", path, str(slot))
        return bool(response and response != "<|ERROR|>")
    
    def save_state_slot(self, slot: int) -> bool:
        """Save state to slot.
        
        Args:
            slot: Save slot number
            
        Returns:
            True if successful
        """
        response = self.send_command("core.saveStateSlot", str(slot))
        return bool(response and response != "<|ERROR|>")
    
    def load_state_slot(self, slot: int, flags: int = 0) -> bool:
        """Load state from slot.
        
        Args:
            slot: Save slot number
            flags: Load flags
            
        Returns:
            True if successful
        """
        response = self.send_command("core.loadStateSlot", str(slot), str(flags))
        return bool(response and response != "<|ERROR|>")
    
    def reset(self) -> bool:
        """Reset the game.
        
        Returns:
            True if successful
        """
        response = self.send_command("coreAdapter.reset")
        return bool(response and response != "<|ERROR|>")
    
    def platform(self) -> Optional[str]:
        """Get platform.
        
        Returns:
            Platform string
        """
        return self.send_command("core.platform")
    
    # Button API methods
    
    def button_tap(self, button: str) -> bool:
        """Tap a button.
        
        Args:
            button: Button name (A, B, Start, Select, Up, Down, Left, Right, L, R)
            
        Returns:
            True if successful
        """
        response = self.send_command("mgba-http.button.tap", button)
        return bool(response and response != "<|ERROR|>")
    
    def button_hold(self, button: str, duration_ms: int) -> bool:
        """Hold a button for duration.
        
        Args:
            button: Button name
            duration_ms: Duration in milliseconds
            
        Returns:
            True if successful
        """
        self.COMMAND_LIMIT.wait_if_needed()
        response = self.send_command("mgba-http.button.hold", button, str(duration_ms))
        return bool(response) and response != "<|ERROR|>"
    
    def button_clear_many(self, buttons: List[str]) -> bool:
        """Clear multiple buttons.
        
        Args:
            buttons: List of button names
            
        Returns:
            True if successful
        """
        buttons_str = ";".join(buttons)
        response = self.send_command("mgba-http.button.clearMany", buttons_str)
        return bool(response and response != "<|ERROR|>")
    
    def button_get_all(self) -> Optional[str]:
        """Get all currently pressed buttons.
        
        Returns:
            Comma-separated button names or None
        """
        return self.send_command("mgba-http.button.getAll")
    
    # Memory API methods
    
    def get_memory_domains(self) -> Optional[List[str]]:
        """Get list of memory domains.
        
        Returns:
            List of memory domain names
        """
        if self._memory_domains is None:
            response = self.send_command("coreAdapter.memory")
            if response:
                self._memory_domains = [d.strip() for d in response.split(",") if d.strip()]
        return self._memory_domains
    
    def memory_domain_read8(self, domain: str, address: int) -> Optional[int]:
        """Read 8-bit value from memory domain.

        Args:
            domain: Memory domain name
            address: Memory address

        Returns:
            Value or None
        """
        self.MEMORY_LIMIT.wait_if_needed()
        self._domain_counters['memory'] += 1
        response = self.send_command("memoryDomain.read8", domain, str(address))
        try:
            return int(response) if response else None
        except (ValueError, TypeError):
            return None
    
    def memory_domain_read16(self, domain: str, address: int) -> Optional[int]:
        """Read 16-bit value from memory domain.

        Args:
            domain: Memory domain name
            address: Memory address

        Returns:
            Value or None
        """
        self.MEMORY_LIMIT.wait_if_needed()
        self._domain_counters['memory'] += 1
        response = self.send_command("memoryDomain.read16", domain, str(address))
        try:
            return int(response) if response else None
        except (ValueError, TypeError):
            return None
    
    def memory_domain_read32(self, domain: str, address: int) -> Optional[int]:
        """Read 32-bit value from memory domain.

        Args:
            domain: Memory domain name
            address: Memory address

        Returns:
            Value or None
        """
        self.MEMORY_LIMIT.wait_if_needed()
        self._domain_counters['memory'] += 1
        response = self.send_command("memoryDomain.read32", domain, str(address))
        try:
            return int(response) if response else None
        except (ValueError, TypeError):
            return None
    
    def memory_domain_read_range(self, domain: str, address: int, length: int) -> Optional[bytes]:
        """Read byte range from memory domain.

        Args:
            domain: Memory domain name
            address: Start address
            length: Number of bytes to read

        Returns:
            Byte data or None
        """
        self.MEMORY_LIMIT.wait_if_needed()
        self._domain_counters['memory'] += 1
        response = self.send_command("memoryDomain.readRange", domain, str(address), str(length))

        if not response or response == "<|ERROR|>":
            return None

        try:
            # Parse hex byte string "[aa,bb,cc,...]"
            hex_part = response.strip("[]")
            bytes_list = [int(h.strip(), 16) for h in hex_part.split(",") if h.strip()]
            return bytes(bytes_list)
        except (ValueError, IndexError, TypeError) as e:
            logger.error("Failed to parse memory read response: %s", e)
            return None
    
    def memory_domain_write8(self, domain: str, address: int, value: int, _safe: bool = True) -> bool:
        """Write 8-bit value to memory domain.

        Args:
            domain: Memory domain name
            address: Memory address
            value: Value to write
            _safe: Safety flag (currently unused but for future safety checks)

        Returns:
            True if successful
        """
        response = self.send_command("memoryDomain.write8", domain, str(address), str(value))
        return bool(response) and response != "<|ERROR|>"

    def memory_domain_write16(self, domain: str, address: int, value: int, _safe: bool = True) -> bool:
        """Write 16-bit value to memory domain.

        Args:
            domain: Memory domain name
            address: Memory address
            value: Value to write
            _safe: Safety flag (currently unused but for future safety checks)

        Returns:
            True if successful
        """
        response = self.send_command("memoryDomain.write16", domain, str(address), str(value))
        return bool(response) and response != "<|ERROR|>"

    def memory_domain_write32(self, domain: str, address: int, value: int, _safe: bool = True) -> bool:
        """Write 32-bit value to memory domain.

        Args:
            domain: Memory domain name
            address: Memory address
            value: Value to write
            _safe: Safety flag (currently unused but for future safety checks)

        Returns:
            True if successful
        """
        response = self.send_command("memoryDomain.write32", domain, str(address), str(value))
        return bool(response) and response != "<|ERROR|>"
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Compatibility aliases
Screenshot = ScreenshotData
