"""mgba Lua Socket API controller for Pokemon MD emulator integration."""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from dataclasses import field
import logging
import socket
import time
import threading
from pathlib import Path
from collections import deque
import json
import argparse
import sys
import random
from PIL import Image
import numpy as np

from .config import VideoConfig
from .fps_adjuster import FPSAdjuster

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
        self._lock = threading.RLock()  # Reentrant lock to prevent deadlocks
        self.reconnect_backoff = 1.0  # Start with 1 second backoff
        self.max_backoff = 30.0  # Max 30 seconds
        self.last_reconnect_attempt = 0.0

    def connect(self) -> bool:
        """Connect to the Lua socket server."""
        with self._lock:
            # If already connected, disconnect first
            if self.is_connected():
                self.disconnect()

            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))
                self._buffer = ""

                # Optional handshake
                self._send_handshake()

                # Validation commented out
                # if not self._validate_connection():
                #     logger.error("Connection validation failed")
                #     self.disconnect()
                #     return False

                # logger.info("LuaSocketTransport connected to %s:%d", self.host, self.port)
                return True

            except socket.timeout:
                logger.error("Connection timeout to %s:%d", self.host, self.port)
                self.disconnect()
                return False
            except ConnectionRefusedError:
                logger.error("Connection refused to %s:%d", self.host, self.port)
                self.disconnect()
                return False
            except OSError as e:
                logger.error("Connection failed: %s", e)
                self.disconnect()
                return False

    def _send_handshake(self) -> None:
        """Send optional handshake to confirm readiness."""
        # Skip handshake for now
        pass

    def _validate_connection(self) -> bool:
        """Validate that the connection is healthy by sending a simple command."""
        try:
            response = self.send_command("core.platform")
            return response is not None and response != "<|ERROR|>"
        except (OSError, ConnectionError):
            return False

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
                        break

                # Extract response
                marker_pos = self._buffer.find(self.TERMINATION_MARKER)
                if marker_pos == -1:
                    logger.error("Response incomplete - no termination marker found")
                    return None

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


class AddressManager:
    """Manages RAM address mappings from config file.

    Loads address definitions from JSON config and converts WRAM offsets
    to absolute GBA addresses for use with mGBA memory operations.
    """

    # GBA memory domain base addresses
    WRAM_BASE = 0x02000000  # Working RAM base address
    VRAM_BASE = 0x06000000  # Video RAM base address
    OAM_BASE = 0x07000000   # Object Attribute Memory base address
    PALETTE_BASE = 0x05000000  # Palette RAM base address
    ROM_BASE = 0x08000000   # ROM base address

    def __init__(self, config_path: str):
        """Load addresses from config file.

        Args:
            config_path: Path to the address configuration JSON file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.addresses = self.config.get("addresses", {})
        self.memory_domains = self.config.get("memory_domains", {})
        logger.info(f"Loaded {len(self.addresses)} address categories from {config_path}")

    def get_address(self, category: str, field: str) -> int:
        """Get absolute GBA address for a field.

        Args:
            category: Address category (e.g., "player_state", "party_status")
            field: Field name within category (e.g., "floor_number", "leader_hp")

        Returns:
            Absolute GBA memory address

        Raises:
            ValueError: If category or field not found
        """
        if category not in self.addresses:
            raise ValueError(f"Unknown address category: {category}")
        if field not in self.addresses[category]:
            raise ValueError(f"Unknown field '{field}' in category '{category}'")

        addr_info = self.addresses[category][field]
        offset = addr_info["address"]
        domain = addr_info.get("domain", "WRAM")

        # Convert WRAM offset to absolute GBA address
        if domain == "WRAM":
            return self.WRAM_BASE + offset
        elif domain == "VRAM":
            return self.VRAM_BASE + offset
        elif domain == "OAM":
            return self.OAM_BASE + offset
        elif domain == "PALETTE":
            return self.PALETTE_BASE + offset
        elif domain == "ROM":
            return self.ROM_BASE + offset
        else:
            raise ValueError(f"Unknown memory domain: {domain}")

    def get_size(self, category: str, field: str) -> int:
        """Get size in bytes for a field.

        Args:
            category: Address category
            field: Field name within category

        Returns:
            Size in bytes
        """
        if category not in self.addresses:
            raise ValueError(f"Unknown address category: {category}")
        if field not in self.addresses[category]:
            raise ValueError(f"Unknown field '{field}' in category '{category}'")
        return self.addresses[category][field]["size"]

    def get_type(self, category: str, field: str) -> str:
        """Get data type for a field.

        Args:
            category: Address category
            field: Field name within category

        Returns:
            Data type string (e.g., "uint8", "uint16", "int32")
        """
        if category not in self.addresses:
            raise ValueError(f"Unknown address category: {category}")
        if field not in self.addresses[category]:
            raise ValueError(f"Unknown field '{field}' in category '{category}'")
        return self.addresses[category][field]["type"]


class MGBAController:
    """Controller for mgba emulator via Lua Socket API (mGBASocketServer 0.8.0)."""

    DEFAULT_TIMEOUT = 3.0
    RETRY_COUNT = 3
    RETRY_BACKOFF_BASE = 0.1  # 100ms base delay
    RETRY_BACKOFF_FACTOR = 10  # Exponential factor

    # Rate limiters
    SCREENSHOT_LIMIT = RateLimiter(max_calls=30, time_window=1.0)  # 30/s max
    MEMORY_LIMIT = RateLimiter(max_calls=10, time_window=1.0)  # 10/s max
    COMMAND_LIMIT = RateLimiter(max_calls=60, time_window=1.0)  # 60/s max

    # Expose transport constants for compatibility
    TERMINATION_MARKER = LuaSocketTransport.TERMINATION_MARKER

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        timeout: float = 10.0,
        cache_dir: Optional[Path] = None,
        video_config: Optional[VideoConfig] = None,
        smoke_mode: bool = False,
        auto_reconnect: bool = True,
        config_path: Optional[str] = None,
    ):
        """Initialize mgba controller.

        Args:
            host: mgba Lua socket server host
            port: mgba Lua socket server port (will auto-bump if busy)
            timeout: Socket timeout in seconds
            cache_dir: Directory for caching server info
            video_config: Video configuration for capture resolution and scaling
            smoke_mode: Enable smoke test mode (fast timeouts, no retries)
            auto_reconnect: Enable automatic reconnection on failures
            config_path: Path to address config JSON. Defaults to config/addresses/pmd_red_us_v1.json
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.cache_dir = cache_dir or Path.home() / ".cache" / "pmd-red"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.video_config = video_config or VideoConfig()
        self.smoke_mode = smoke_mode

        # Initialize address manager with config file
        if config_path is None:
            # Default to project's config directory
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "config" / "addresses" / "pmd_red_us_v1.json")
        self.address_manager = AddressManager(config_path)

        # Initialize FPS adjuster
        self.fps_adjuster = FPSAdjuster(base_fps=30, allowed_fps=[30, 10, 5, 3, 1])

        # Build RAM_ADDRESSES from config for backward compatibility
        # Maps old hardcoded keys to config-based addresses
        self._build_ram_addresses()
        self.auto_reconnect = auto_reconnect

        # Heartbeat for connection health monitoring
        self.heartbeat_interval = 5.0  # seconds
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        self.last_heartbeat = 0.0
        self.connection_healthy = False

        # Adjust timeouts and retries for smoke mode
        if self.smoke_mode:
            self.timeout = 1.0  # Fast timeout for smoke tests
            self.RETRY_COUNT = 0  # No retries in smoke mode
            self.auto_reconnect = False  # No auto-reconnect in smoke mode

        self._transport = LuaSocketTransport(self.host, self.port, self.timeout)
        self._server_version = "0.8.0"  # Fixed server version
        self._game_title = None
        self._game_code = None
        self._memory_domains: Optional[List[str]] = None

        # Metrics
        self._command_latencies: Dict[str, List[float]] = {}
        self._domain_counters: Dict[str, int] = {"memory": 0, "button": 0, "core": 0, "screenshot": 0}

        # Frame tracking
        self.current_frame_data: Optional[np.ndarray] = None
        self._current_frame: Optional[int] = None
        self._frame_counter: int = 0

        logger.info("Initialized MGBAController at %s:%d (scale=%dx, smoke=%s)", self.host, self.port, self.video_config.scale, self.smoke_mode)

    def _start_heartbeat(self) -> None:
        """Start the heartbeat thread for connection monitoring."""
        if self.smoke_mode or self.heartbeat_thread is not None:
            return

        self.heartbeat_stop_event.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            name="mgba-heartbeat",
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.debug("Started heartbeat thread")

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        if self.heartbeat_thread is None:
            return

        self.heartbeat_stop_event.set()
        self.heartbeat_thread.join(timeout=1.0)
        self.heartbeat_thread = None
        logger.debug("Stopped heartbeat thread")

    def _heartbeat_worker(self) -> None:
        """Background worker for connection health monitoring."""
        while not self.heartbeat_stop_event.is_set():
            try:
                # Send a lightweight heartbeat command
                response = self._transport.send_command("core.platform")
                self.connection_healthy = response is not None and response != "<|ERROR|>"
                self.last_heartbeat = time.time()

                if not self.connection_healthy:
                    logger.warning("Heartbeat failed - connection may be unhealthy")
                else:
                    logger.debug("Heartbeat successful")

            except Exception as e:
                self.connection_healthy = False
                logger.warning("Heartbeat exception: %s", e)

            # Wait for next heartbeat interval
            self.heartbeat_stop_event.wait(self.heartbeat_interval)

    def is_connection_healthy(self) -> bool:
        """Check if connection is healthy based on recent heartbeat."""
        if not self.is_connected():
            return False

        # If no heartbeat configured, assume healthy
        if self.smoke_mode:
            return True

        # Check if heartbeat is recent (within 2x interval)
        time_since_heartbeat = time.time() - self.last_heartbeat
        return time_since_heartbeat < (self.heartbeat_interval * 2)

    def _find_available_port(self, start_port: int) -> int:
        """Return the specified port (for testing purposes)."""
        return start_port

    def _build_ram_addresses(self) -> None:
        """Build RAM_ADDRESSES dict from config file for backward compatibility.

        Maps old hardcoded keys to new config-based addresses loaded from JSON.
        This ensures existing code using self.RAM_ADDRESSES["key"] continues to work.
        """
        # Mapping from old keys to (category, field) tuples in config
        address_mapping = {
            # Dungeon state
            "floor": ("player_state", "floor_number"),
            "turn_counter": ("player_state", "turn_counter"),

            # Player position
            "player_x": ("player_state", "player_tile_x"),
            "player_y": ("player_state", "player_tile_y"),

            # Party stats
            "hp": ("party_status", "leader_hp"),
            "max_hp": ("party_status", "leader_hp_max"),
            "belly": ("party_status", "leader_belly"),

            # Partner stats
            "partner_hp": ("party_status", "partner_hp"),
            "partner_max_hp": ("party_status", "partner_hp_max"),
            "partner_belly": ("party_status", "partner_belly"),
        }

        # Build RAM_ADDRESSES dict from config
        self.RAM_ADDRESSES = {}
        for old_key, (category, field) in address_mapping.items():
            try:
                address = self.address_manager.get_address(category, field)
                self.RAM_ADDRESSES[old_key] = address
                # Use safe formatting for address since tests may patch AddressManager
                try:
                    addr_repr = f"0x{int(address):08X}"
                except Exception:
                    addr_repr = str(address)
                logger.debug("Mapped '%s' -> %s.%s @ %s", old_key, category, field, addr_repr)
            except ValueError as e:
                logger.warning(f"Could not map '{old_key}': {e}")

        logger.info(f"Built RAM_ADDRESSES with {len(self.RAM_ADDRESSES)} entries from config")

    def connect(self) -> bool:
        """Connect to mgba Lua socket server with strict timeout.

        Returns:
            True if connection succeeded
        """
        # If already connected, disconnect first
        if self.is_connected():
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Set strict connect timeout for smoke mode or fast failure
        connect_timeout = 1.0 if self.smoke_mode else min(self.timeout, 5.0)

        try:
            with self._transport._lock:
                self._transport._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._transport._socket.settimeout(connect_timeout)
                self._transport._socket.connect((self.host, self.port))
                self._transport._buffer = ""

            # Optional handshake
            self._transport._send_handshake()

            # Validation commented out
            # if not self._validate_connection():
            #     logger.error("Connection validation failed")
            #     self.disconnect()
            #     return False

            logger.info("Connected to mGBA at %s:%d", self.host, self.port)

            # Probe server capabilities
            self._probe_server()

            # Cache connection info
            self._save_connection_cache()

            # Start heartbeat monitoring
            self._start_heartbeat()

            return True

        except socket.timeout:
            logger.warning("Connection timeout to %s:%d after %.1fs", self.host, self.port, connect_timeout)
            self.disconnect()
            return False
        except ConnectionRefusedError:
            logger.warning("Connection refused to %s:%d", self.host, self.port)
            self.disconnect()
            return False
        except OSError as e:
            logger.error("Connection failed: %s", e)
            self.disconnect()
            return False

        return False

    def connect_with_retry(self, max_retries: int = 3, backoff_factor: float = 10) -> bool:
        """Connect with exponential backoff retry logic.

        Args:
            max_retries: Maximum number of connection attempts
            backoff_factor: Exponential backoff multiplier

        Returns:
            True if connection succeeded
        """
        if self.smoke_mode:
            # Skip retries in smoke mode
            return self.connect()

        base_delay = self.RETRY_BACKOFF_BASE
        for attempt in range(max_retries + 1):
            logger.info("Connection attempt %d/%d", attempt + 1, max_retries + 1)

            if self.connect():
                return True

            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                logger.info("Retrying connection in %.1f seconds...", delay)
                time.sleep(delay)

        logger.error("Failed to connect after %d attempts", max_retries + 1)
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
        """Send command to mgba and get response with resilience and error recovery.

        Args:
            command: Command type (e.g., "core.getGameTitle")
            args: Command arguments

        Returns:
            Response string or None if failed
        """
        # Track metrics
        domain = command.split('.')[0] if '.' in command else 'unknown'
        if domain not in self._domain_counters:
            self._domain_counters[domain] = 0
        self._domain_counters[domain] += 1

        # Skip retries in smoke mode
        if self.smoke_mode:
            try:
                start_time = time.time()
                response = self._transport.send_command(command, *args)
                latency = time.time() - start_time

                logger.info("Smoke mode command %s latency: %.3fs", command, latency)
                if command not in self._command_latencies:
                    self._command_latencies[command] = []
                self._command_latencies[command].append(latency)

                return response
            except ConnectionError as e:
                logger.error("Command failed in smoke mode: %s", e)
                return None
            except Exception as e:
                logger.error("Unexpected error in smoke mode: %s", e)
                return None

        # Use transport with retries and backoff for normal mode
        connection_lost = False
        for attempt in range(self.RETRY_COUNT):
            try:
                start_time = time.time()
                response = self._transport.send_command(command, *args)
                latency = time.time() - start_time

                # Log structured metrics
                logger.debug("Command %s latency: %.3fs", command, latency)
                if command not in self._command_latencies:
                    self._command_latencies[command] = []
                self._command_latencies[command].append(latency)

                # Reset backoff on success
                self._transport.reconnect_backoff = 1.0
                if connection_lost:
                    logger.info("Connection recovered after %d attempts", attempt + 1)

                return response

            except ConnectionError as e:
                connection_lost = True
                logger.warning("Connection error on attempt %d/%d: %s", attempt + 1, self.RETRY_COUNT, e)
                if attempt == self.RETRY_COUNT - 1:
                    logger.error("Command failed after retries: %s", command)
                    return None

                # Auto-reconnect with jittered exponential backoff (only if enabled)
                if self.auto_reconnect:
                    # Add jitter to prevent thundering herd: ±25% of base backoff
                    base_backoff = min(self._transport.reconnect_backoff, self._transport.max_backoff)
                    jitter = base_backoff * 0.25 * (random.random() * 2 - 1)  # ±25%
                    backoff_time = max(0.1, base_backoff + jitter)  # Minimum 100ms

                    logger.info("Auto-reconnecting in %.2fs (base: %.1fs, jitter: %+.2fs)",
                              backoff_time, base_backoff, jitter)
                    time.sleep(backoff_time)

                    # Try to reconnect
                    try:
                        if self._transport.connect():
                            logger.info("Auto-reconnected successfully")
                            # Reset backoff on success
                            self._transport.reconnect_backoff = 1.0
                            # Validate connection with a simple command
                            if self._transport._validate_connection():
                                logger.debug("Connection validated after auto-reconnect")
                            else:
                                logger.warning("Connection established but validation failed after auto-reconnect")
                        else:
                            # Increase backoff for next attempt (exponential with jitter)
                            self._transport.reconnect_backoff = min(
                                self._transport.reconnect_backoff * 1.5,
                                self._transport.max_backoff
                            )
                            logger.warning("Auto-reconnect failed, backoff now %.1fs", self._transport.reconnect_backoff)
                    except Exception as reconnect_error:
                        logger.error("Auto-reconnect failed with error: %s", reconnect_error)
                        # Increase backoff even on exception
                        self._transport.reconnect_backoff = min(
                            self._transport.reconnect_backoff * 1.5,
                            self._transport.max_backoff
                        )
                else:
                    logger.debug("Auto-reconnect disabled, using standard backoff")

                # Exponential backoff
                backoff = self.RETRY_BACKOFF_BASE * (self.RETRY_BACKOFF_FACTOR ** attempt)
                time.sleep(backoff)

            except Exception as e:
                logger.error("Unexpected error on attempt %d/%d: %s", attempt + 1, self.RETRY_COUNT, e)
                if attempt == self.RETRY_COUNT - 1:
                    logger.error("Command failed after retries due to unexpected error: %s", command)
                    return None

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
        # Stop heartbeat first
        self._stop_heartbeat()
        self.connection_healthy = False

        self._transport.disconnect()
        logger.info("Disconnected from mgba server")

    def __enter__(self) -> 'MGBAController':
        """Context manager entry - connect to server."""
        if not self.connect_with_retry():
            raise ConnectionError("Failed to connect to mGBA server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disconnect from server."""
        self.disconnect()

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
        """Take screenshot to file.

        Args:
            path: File path for screenshot

        Returns:
            True if successful
        """
        self.SCREENSHOT_LIMIT.wait_if_needed()
        self._domain_counters['screenshot'] += 1
        response = self.send_command("core.screenshot", path)

        return bool(response and response != "<|ERROR|>")

    def autoload_save(self) -> bool:
        """Autoload save file.

        Returns:
            True if successful
        """
        response = self.send_command("core.autoLoadSave")
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
        domain = domain.lower()
        self.MEMORY_LIMIT.wait_if_needed()
        self._domain_counters['memory'] += 1
        response = self.send_command("memoryDomain.readRange", domain, str(address), str(length))

        if not response or response == "<|ERROR|>":
            return None

        try:
            # Parse hex byte string "aa,bb,cc,..."
            bytes_list = [int(h.strip(), 16) for h in response.split(",") if h.strip()]
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

    def grab_frame(self, output_path: Optional[Path] = None, timeout: float = 5.0) -> Optional[Image.Image]:
        """Grab current frame as PIL Image with tolerant resolution detection.

        Supports multiple resolution profiles (480×320, 960×640) and automatically
        detects the resolution returned by mGBA, logging warnings for unsupported sizes.

        Args:
            output_path: Optional path to save frame with deterministic name
            timeout: Maximum time to wait for frame capture

        Returns:
            PIL Image or None if failed
        """
        start_time = time.time()

        # Get current state for deterministic naming
        try:
            floor = self.get_floor()
            x, y = self.get_player_position()
            timestamp = int(time.time() * 1000)  # milliseconds
        except RuntimeError as e:
            logger.warning("Failed to read game state for naming: %s", e)
            floor, x, y, timestamp = 0, 0, 0, int(time.time() * 1000)

        try:
            # Save screenshot to temp file
            temp_path = self.cache_dir / f"temp_frame_{timestamp}.png"

            # Save screenshot with timeout check
            if time.time() - start_time > timeout:
                logger.error("Frame capture timed out after %.1fs", timeout)
                return None

            if not self.screenshot(str(temp_path)):
                return None

            # Load as PIL Image with timeout check and retry for file locking
            if time.time() - start_time > timeout:
                logger.error("Frame load timed out after %.1fs", timeout)
                temp_path.unlink(missing_ok=True)
                return None

            # Retry opening the image file in case mGBA is still writing to it
            image = None
            for attempt in range(5):  # Try up to 5 times
                try:
                    # Use context manager to ensure file handle is closed on Windows
                    with Image.open(temp_path) as img:
                        image = img.convert('RGB').copy()
                    break  # Success, exit retry loop
                except OSError as e:
                    if attempt < 4:  # Don't sleep on last attempt
                        time.sleep(0.1)  # Wait 100ms before retry
                        if time.time() - start_time > timeout:
                            logger.error("Frame load timed out after %.1fs", timeout)
                            temp_path.unlink(missing_ok=True)
                            return None
                    else:
                        # Last attempt failed
                        logger.error("Failed to open screenshot after 5 attempts: %s", e)
                        temp_path.unlink(missing_ok=True)
                        return None

            # At this point image is guaranteed to be not None
            assert image is not None

            # Check if image size matches a supported resolution profile
            supported_sizes = self.video_config.get_supported_sizes()
            inferred_profile = self.video_config.infer_profile_from_size(image.size)

            if inferred_profile is not None:
                logger.debug("Captured frame at supported resolution: %s (%s)",
                           image.size, inferred_profile.name)
            else:
                # Find nearest supported profile and log warning
                nearest_profile = self.video_config.find_nearest_profile(image.size)
                logger.warning(
                    "Captured frame at unsupported resolution %s, nearest supported is %s (%s). "
                    "Consider updating mGBA scaling configuration.",
                    image.size, nearest_profile.size, nearest_profile.name
                )

            # Save with deterministic name if requested
            if output_path:
                deterministic_name = f"{timestamp}_{floor}_{x}_{y}.png"
                final_path = output_path / deterministic_name
                image.save(final_path)
                logger.info("Saved frame to %s (%dx%d)", final_path, image.width, image.height)

            # Cleanup temp file
            temp_path.unlink(missing_ok=True)

            # Store frame data for agent access
            self.current_frame_data = np.array(image)
            self._current_frame = self.current_frame()
            self._frame_counter += 1

            return image

        except (OSError, ValueError) as e:
            logger.error("Failed to process screenshot: %s", e)
            return None

    def capture_screenshot(self, path: str, max_retries: int = 5) -> np.ndarray:
        """
        Capture screenshot with retry logic for Windows file locking.
        
        Args:
            path: Output path for PNG screenshot
            max_retries: Max retry attempts (default 5)
        
        Returns:
            Screenshot as numpy array (H, W, 3) RGB
        
        Raises:
            RuntimeError: If screenshot fails after all retries
        """
        # Send screenshot command to mGBA
        self.send_command(f"core.screenshot,{path}")
        
        # Wait for file to exist
        file_path = Path(path)
        for attempt in range(max_retries):
            if file_path.exists():
                break
            time.sleep(0.1 * (2 ** attempt))  # 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
        else:
            raise RuntimeError(f"Screenshot file not created: {path}")
        
        # Retry opening file (handles Windows file locking)
        last_error = None
        for attempt in range(max_retries):
            try:
                with Image.open(path) as img:
                    # Load pixels immediately while file is open
                    arr = np.array(img.convert('RGB'))
                
                # Delete temp file after successful read
                try:
                    file_path.unlink()
                except PermissionError:
                    pass  # File still locked, but we got the data
                
                return arr
                
            except (PermissionError, OSError) as e:
                last_error = e
                wait_time = 0.1 * (2 ** attempt)
                time.sleep(wait_time)
        
        # All retries exhausted
        raise RuntimeError(
            f"Failed to read screenshot after {max_retries} attempts: {last_error}"
        )

    def capture_with_metadata(self, output_path: Optional[Path] = None, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Capture screenshot with metadata including timing and video config.

        Args:
            output_path: Optional path to save frame with deterministic name
            timeout: Maximum time to wait for frame capture

        Returns:
            Dict with 'image', 'metadata', and 'path' keys, or None if failed
        """
        start_time = time.time()

        # Capture the image
        image = self.grab_frame(output_path, timeout)
        if image is None:
            return None

        capture_time_ms = (time.time() - start_time) * 1000

        # Infer resolution profile from captured image
        inferred_profile = self.video_config.infer_profile_from_size(image.size)
        profile_name = inferred_profile.name if inferred_profile else "unknown"

        # Build metadata
        metadata = {
            "width": image.width,
            "height": image.height,
            "scale": inferred_profile.scale if inferred_profile else self.video_config.scale,
            "profile": profile_name,
            "capture_time_ms": capture_time_ms,
            "timestamp": time.time(),
            "frame_number": self.current_frame(),
        }

        # Get current state if available
        try:
            floor = self.get_floor()
            x, y = self.get_player_position()
            metadata.update({
                "floor": floor,
                "player_x": x,
                "player_y": y,
            })
        except RuntimeError:
            pass

        result = {
            "image": image,
            "metadata": metadata,
            "path": str(output_path) if output_path else None,
        }

        logger.info("Captured frame with metadata: %dx%d @ %s profile in %.1fms",
                   image.width, image.height, profile_name, capture_time_ms)

        return result

    def press(self, keys: List[str]) -> bool:
        """Press multiple keys simultaneously.

        Args:
            keys: List of key names (A, B, Start, Select, Up, Down, Left, Right, L, R)

        Returns:
            True if successful
        """
        if not keys:
            return True

        # Convert to button format
        key_str = ";".join(keys)
        response = self.send_command("mgba-http.button.tapMany", key_str)
        return bool(response and response != "<|ERROR|>")

    def peek(self, addr: int, n: int) -> Optional[bytes]:
        """Read n bytes from memory address.

        Args:
            addr: Memory address (absolute, e.g., 0x02004139)
            n: Number of bytes to read

        Returns:
            Bytes data or None if failed
        """
        # Determine domain and offset from absolute address
        # EWRAM: 0x02000000-0x0203FFFF (256KB)
        if 0x02000000 <= addr < 0x02040000:
            domain = "wram"  # EWRAM
            offset = addr - 0x02000000
        elif 0x03000000 <= addr < 0x03008000:
            domain = "iwram"  # IWRAM
            offset = addr - 0x03000000
        else:
            logger.error(f"Unsupported memory address: 0x{addr:08X}")
            return None

        return self.memory_domain_read_range(domain, offset, n)

    def get_floor(self) -> int:
        """Get current floor number."""
        size = self.address_manager.get_size("player_state", "floor_number")
        data = self.peek(self.RAM_ADDRESSES["floor"], size)
        if data is None:
            raise RuntimeError("Failed to read floor from memory")
        return int.from_bytes(data, byteorder='little')

    def get_player_position(self) -> tuple[int, int]:
        """Get player (x, y) tile position."""
        x_size = self.address_manager.get_size("player_state", "player_tile_x")
        y_size = self.address_manager.get_size("player_state", "player_tile_y")
        x_data = self.peek(self.RAM_ADDRESSES["player_x"], x_size)
        y_data = self.peek(self.RAM_ADDRESSES["player_y"], y_size)
        if x_data is None or y_data is None:
            raise RuntimeError("Failed to read player position from memory")
        x = int.from_bytes(x_data, byteorder='little')
        y = int.from_bytes(y_data, byteorder='little')
        return x, y

    def get_player_stats(self) -> dict[str, int]:
        """Get player stats (HP, belly)."""
        hp_size = self.address_manager.get_size("party_status", "leader_hp")
        max_hp_size = self.address_manager.get_size("party_status", "leader_hp_max")
        belly_size = self.address_manager.get_size("party_status", "leader_belly")

        hp_data = self.peek(self.RAM_ADDRESSES["hp"], hp_size)
        max_hp_data = self.peek(self.RAM_ADDRESSES["max_hp"], max_hp_size)
        belly_data = self.peek(self.RAM_ADDRESSES["belly"], belly_size)

        if any(data is None for data in [hp_data, max_hp_data, belly_data]):
            raise RuntimeError("Failed to read player stats from memory")

        # Type assertions after None check
        assert hp_data is not None
        assert max_hp_data is not None
        assert belly_data is not None

        hp = int.from_bytes(hp_data, byteorder='little')
        max_hp = int.from_bytes(max_hp_data, byteorder='little')
        belly = int.from_bytes(belly_data, byteorder='little')

        # Max belly is always 100 in PMD (not stored in RAM)
        max_belly = 100

        return {
            "hp": hp,
            "max_hp": max_hp,
            "belly": belly,
            "max_belly": max_belly,
        }

    def semantic_state(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return a lightweight semantic snapshot used by the skill runtime."""
        state: Dict[str, Any] = {}

        try:
            stats = self.get_player_stats()
            state.update(
                {
                    "hp": stats.get("hp"),
                    "max_hp": stats.get("max_hp"),
                    "belly": stats.get("belly"),
                    "max_belly": stats.get("max_belly"),
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to fetch player stats: %s", exc)

        try:
            state["floor"] = self.get_floor()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to fetch floor: %s", exc)

        try:
            px, py = self.get_player_position()
            state["player_pos"] = {"x": px, "y": py}
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to fetch position: %s", exc)

        if fields is None:
            return state
        return {field: state.get(field) for field in fields}

    def await_frames(self, n: int) -> bool:
        """Wait for n frames to pass.

        Args:
            n: Number of frames to wait

        Returns:
            True if successful
        """
        start_frame = self.current_frame()
        if start_frame is None:
            logger.error("Could not get current frame")
            return False

        target_frame = start_frame + n

        # Poll until target frame reached
        max_attempts = 100  # Avoid infinite loop
        for _ in range(max_attempts):
            current = self.current_frame()
            if current is not None and current >= target_frame:
                return True
            time.sleep(0.016)  # ~60 FPS

        logger.warning("Timeout waiting for %d frames", n)
        return False

    def wait_frames_or_ram_flag(self, frames: int, ram_addr: int, expected_value: int, timeout_frames: int = 300) -> bool:
        """Wait for either N frames to pass OR RAM address to reach expected value.

        Args:
            frames: Minimum frames to wait
            ram_addr: RAM address to monitor
            expected_value: Expected value at RAM address
            timeout_frames: Maximum frames to wait before timeout

        Returns:
            True if condition met, False if timeout
        """
        start_frame = self.current_frame()
        if start_frame is None:
            logger.error("Could not get current frame")
            return False

        target_frame = start_frame + frames
        timeout_frame = start_frame + timeout_frames

        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        start_time = time.time()

        while iteration_count < max_iterations:
            iteration_count += 1

            # Check for overall timeout
            if time.time() - start_time > 10.0:
                logger.warning("Timeout in wait_frames_or_ram_flag after 10s")
                return False

            current_frame = self.current_frame()
            if current_frame is None:
                continue

            # Check timeout
            if current_frame >= timeout_frame:
                logger.warning("Timeout waiting for sync fence (frames=%d, ram_addr=0x%x, expected=%d)",
                             frames, ram_addr, expected_value)
                return False

            # Check RAM condition
            ram_data = self.peek(ram_addr, 4)
            if ram_data is not None:
                current_value = int.from_bytes(ram_data, byteorder='little')
                if current_value == expected_value:
                    logger.debug("RAM sync fence met at frame %d (value=%d)", current_frame, current_value)
                    return True

            # Check frame condition
            if current_frame >= target_frame:
                logger.debug("Frame sync fence met at frame %d", current_frame)
                return True

            time.sleep(0.008)  # ~120 FPS polling

        logger.warning("Maximum iterations reached waiting for sync fence (frames=%d, ram_addr=0x%x, expected=%d)",
                     frames, ram_addr, expected_value)
        return False

    def sync_after_input(self, input_keys: List[str], sync_frames: int = 5) -> bool:
        """Press input and wait for sync fence (frames or RAM change).

        Args:
            input_keys: Keys to press
            sync_frames: Minimum frames to wait after input

        Returns:
            True if sync successful
        """
        # Press the input
        if not self.press(input_keys):
            logger.error("Failed to press keys: %s", input_keys)
            return False

        # Wait for sync fence - either frames pass or player position changes
        # This ensures input has been processed
        initial_x, initial_y = self.get_player_position()

        return self.wait_frames_or_ram_flag(
            frames=sync_frames,
            ram_addr=self.RAM_ADDRESSES["player_x"],  # Monitor X position change
            expected_value=initial_x,  # Wait for it to change from initial
            timeout_frames=60  # 1 second timeout
        )


    def set_fps(self, fps: int) -> bool:
        """Set emulator FPS.

        Args:
            fps: Target FPS (1-60)

        Returns:
            True if successful
        """
        if not (1 <= fps <= 60):
            logger.warning("FPS %d out of range (1-60)", fps)
            return False

        response = self.send_command("core.setFrameRate", str(fps))
        return bool(response and response != "<|ERROR|>")

    def get_fps(self) -> Optional[int]:
        """Get current emulator FPS.

        Returns:
            Current FPS or None if failed
        """
        response = self.send_command("core.getFrameRate")
        if response and response != "<|ERROR|>":
            try:
                return int(response)
            except ValueError:
                return None
        return None

    def set_frame_multiplier(self, multiplier: int) -> bool:
        """Set frame multiplier (speed control).

        Args:
            multiplier: Frame multiplier (1, 2, 4, 8, 16, 32, 64)

        Returns:
            True if successful
        """
        if multiplier not in [1, 2, 4, 8, 16, 32, 64]:
            logger.warning("Invalid frame multiplier %d", multiplier)
            return False

        response = self.send_command("core.setFrameMultiplier", str(multiplier))
        return bool(response and response != "<|ERROR|>")

    def adjust_fps(self, target_fps: int) -> bool:
        """Adjust FPS using the FPS adjuster and send command to mGBA.

        Args:
            target_fps: Target FPS level

        Returns:
            True if adjustment succeeded
        """
        if not self.fps_adjuster.set_fps(target_fps):
            return False

        # Send command to mGBA
        return self.set_fps(target_fps)

    def adjust_frame_multiplier(self, multiplier: int) -> bool:
        """Adjust frame multiplier using the FPS adjuster and send command to mGBA.

        Args:
            multiplier: Frame multiplier

        Returns:
            True if adjustment succeeded
        """
        if not self.fps_adjuster.set_multiplier(multiplier):
            return False

        # Send command to mGBA
        return self.set_frame_multiplier(multiplier)

    def get_current_effective_fps(self) -> int:
        """Get current effective FPS from the adjuster.

        Returns:
            Current effective FPS
        """
        return self.fps_adjuster.get_current_fps()

    def current_frame(self) -> Optional[int]:
        """Get current frame number from emulator.

        Returns:
            Current frame number or None if failed
        """
        if self._current_frame is not None:
            return self._current_frame

        try:
            response = self.send_command("core.currentFrame")
            if response and response != "<|ERROR|>":
                self._current_frame = int(response)
                return self._current_frame
        except (ValueError, TypeError):
            logger.debug("Failed to parse frame number from response")
        return None

    def zoom_out_temporally(self) -> bool:
        """Zoom out temporally (lower FPS for longer time spans).

        Returns:
            True if adjustment succeeded
        """
        target_fps = self.fps_adjuster.get_current_fps()
        lower_levels = [fps for fps in self.fps_adjuster.allowed_fps if fps < target_fps]

        if lower_levels:
            target_fps = max(lower_levels)
        else:
            # Try increasing multiplier
            current_multiplier = self.fps_adjuster.frame_multiplier
            if current_multiplier < 64:
                target_multiplier = current_multiplier * 2
                return self.adjust_frame_multiplier(target_multiplier)

        return self.adjust_fps(target_fps)

    def zoom_in_temporally(self) -> bool:
        """Zoom in temporally (higher FPS for more detail).

        Returns:
            True if adjustment succeeded
        """
        target_fps = self.fps_adjuster.get_current_fps()
        higher_levels = [fps for fps in self.fps_adjuster.allowed_fps if fps > target_fps]

        if higher_levels:
            target_fps = min(higher_levels)
        else:
            # Try decreasing multiplier
            current_multiplier = self.fps_adjuster.frame_multiplier
            if current_multiplier > 1:
                target_multiplier = current_multiplier // 2
                return self.adjust_frame_multiplier(target_multiplier)

        return self.adjust_fps(target_fps)


# Compatibility aliases
Screenshot = ScreenshotData


def main():
    """CLI entry point for mgba controller."""
    parser = argparse.ArgumentParser(description="mGBA Controller for Pokemon MD")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Capture one 480×320 frame and exit"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="mGBA Lua socket server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="mGBA Lua socket server port"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Socket timeout in seconds"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Video capture width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Video capture height"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Video capture scale factor"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    controller = MGBAController(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        video_config=VideoConfig(width=args.width, height=args.height, scale=args.scale),
        smoke_mode=args.smoke,
        auto_reconnect=False  # CLI mode doesn't need auto-reconnect
    )

    try:
        if not controller.connect_with_retry():
            logger.error("Failed to connect to mGBA after retries")
            sys.exit(1)

        if args.smoke:
            logger.info("Running smoke test - capturing frame at configured resolution...")

            # Create temp directory for smoke test output
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="pmd_smoke_"))
            logger.info("Smoke test output directory: %s", temp_dir)

            # Capture frame with metadata
            result = controller.capture_with_metadata(output_path=temp_dir, timeout=2.0)
            if result is None:
                logger.error("Failed to capture frame with metadata")
                sys.exit(1)

            # Save metadata to JSON file
            metadata_path = temp_dir / "capture_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(result["metadata"], f, indent=2)

            # Verify frame dimensions
            image = result["image"]
            supported_sizes = controller.video_config.get_supported_sizes()
            if image.size not in supported_sizes:
                logger.error("Frame dimensions incorrect: got %s, expected one of %s", image.size, supported_sizes)
                sys.exit(1)

            logger.info("Smoke test completed successfully - %dx%d frame saved to %s",
                       controller.video_config.scaled_width, controller.video_config.scaled_height, temp_dir)
            logger.info("Metadata saved to %s", metadata_path)
            print(f"SMOKE_SUCCESS:{temp_dir}")  # For automated testing
            sys.exit(0)

        # Interactive mode (placeholder)
        logger.info("Connected to mGBA. Use --smoke for testing.")

    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
