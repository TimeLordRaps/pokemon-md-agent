"""Pytest configuration and shared fixtures for integration testing.

This module provides shared fixtures for testing the Pokemon Mystery Dungeon agent system,
including mGBA controller, RAM decoders, and model router fixtures.

IMPORTANT: Integration tests assume:
- mGBA emulator is running with Lua socket server on port 8888
- ROM is loaded (Pokemon Mystery Dungeon: Red Rescue Team US v1.0)
- Save state is loaded with player in a dungeon
- Lua script server is active ("mGBA script server 0.8.0 ready")
"""

import pytest
import tempfile
from pathlib import Path
from typing import Generator
import os
import socket
import time
import faulthandler
from unittest.mock import Mock

# Global variable to track session start time
_session_start_time = None

# Track test durations for slow test reporting
_test_durations = []

# Import project modules
from src.environment.mgba_controller import MGBAController, AddressManager
from src.environment.config import VideoConfig


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: Integration tests requiring live mGBA server (slow)"
    )
    config.addinivalue_line(
        "markers",
        "live_emulator: Tests that require live emulator connection"
    )
    config.addinivalue_line(
        "markers",
        "model_test: Tests that load ML models (very slow, requires GPU)"
    )
    config.addinivalue_line(
        "markers",
        "ram_test: Tests that read/write RAM from emulator"
    )


def pytest_sessionstart(session):
    """Set up session-level deadlock detection."""
    global _session_start_time
    _session_start_time = time.time()
    
    # Enable faulthandler traceback dumping on timeout
    faulthandler.dump_traceback_later(int(os.getenv("PYTEST_FDUMP_S", "60")), repeat=True)


# ============================================================================
# Test Timeout Configuration
# ============================================================================

# Default timeout for integration tests (seconds)
INTEGRATION_TEST_TIMEOUT = 30

# Default timeout for model tests (seconds)
MODEL_TEST_TIMEOUT = 120

# Default timeout for RAM tests (seconds)
RAM_TEST_TIMEOUT = 10


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    # Go up from tests/ to project root
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Get config directory."""
    return project_root / "config"


@pytest.fixture(scope="session")
def address_config_path(config_dir: Path) -> str:
    """Get path to RAM address config file."""
    return str(config_dir / "addresses" / "pmd_red_us_v1.json")


@pytest.fixture(scope="session")
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create temporary cache directory for tests."""
    with tempfile.TemporaryDirectory(prefix="pmd_test_cache_") as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# mGBA Controller Fixtures
# ============================================================================

@pytest.fixture
def mgba_controller(address_config_path: str, temp_cache_dir: Path) -> Generator[MGBAController, None, None]:
    """Create MGBAController instance for integration testing.

    This fixture creates a controller connected to the live mGBA server.
    It automatically handles connection and cleanup.

    Yields:
        MGBAController: Connected controller instance

    Notes:
        - Assumes mGBA server is running on localhost:8888
        - Timeout is set to 3 seconds to prevent hanging
        - Auto-reconnect is disabled for predictable test behavior
    """
    controller = MGBAController(
        host="localhost",
        port=8888,
        timeout=3.0,  # Prevent hanging per integration test requirements
        cache_dir=temp_cache_dir,
        smoke_mode=False,
        auto_reconnect=False,  # Disable for predictable tests
        config_path=address_config_path,
    )

    # Don't auto-connect here - let tests control connection
    yield controller

    # Cleanup: ensure disconnection
    try:
        if controller.is_connected():
            controller.disconnect()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def connected_mgba_controller(mgba_controller: MGBAController) -> Generator[MGBAController, None, None]:
    """Create and connect MGBAController instance.

    This is a convenience fixture that automatically connects to the server.
    Use this for tests that need an active connection from the start.

    Yields:
        MGBAController: Connected controller instance

    Raises:
        RuntimeError: If connection to mGBA server fails
    """
    if not is_mgba_server_available(timeout=0.5):
        pytest.skip("mGBA server not available - ensure emulator is running")

    max_attempts = 2  # Initial try + single retry
    for attempt in range(1, max_attempts + 1):
        if mgba_controller.connect():
            break
        if attempt < max_attempts:
            time.sleep(0.5)
    else:
        pytest.skip("mGBA server not available after retry (3s timeout) - ensure emulator is running")

    yield mgba_controller

    # Cleanup is handled by mgba_controller fixture


@pytest.fixture
def smoke_mgba_controller(address_config_path: str, temp_cache_dir: Path) -> Generator[MGBAController, None, None]:
    """Create MGBAController in smoke test mode.

    Smoke mode has:
    - Fast timeouts (1 second)
    - No retries
    - No auto-reconnect

    Use this for tests that need fast failure behavior.

    Yields:
        MGBAController: Controller in smoke mode
    """
    controller = MGBAController(
        host="localhost",
        port=8888,
        timeout=1.0,
        cache_dir=temp_cache_dir,
        smoke_mode=True,
        auto_reconnect=False,
        config_path=address_config_path,
    )

    yield controller

    # Cleanup
    try:
        if controller.is_connected():
            controller.disconnect()
    except Exception:
        pass


# ============================================================================
# Address Manager Fixtures
# ============================================================================

@pytest.fixture
def mock_controller(address_manager: AddressManager) -> Mock:
    """Create a mock MGBAController for unit testing.

    This fixture provides a mock controller with the necessary attributes
    for testing decoder functionality without requiring a live emulator.

    Returns:
        Mock: Mocked MGBAController instance
    """
    controller = Mock()
    controller.address_manager = address_manager
    controller.peek = Mock()
    return controller


# ============================================================================
# Video Config Fixtures
# ============================================================================

@pytest.fixture
def video_config_1x() -> VideoConfig:
    """Create VideoConfig with 1x scaling (240x160)."""
    return VideoConfig(scale=1)


@pytest.fixture
def video_config_2x() -> VideoConfig:
    """Create VideoConfig with 2x scaling (480x320)."""
    return VideoConfig(scale=2)


@pytest.fixture
def video_config_4x() -> VideoConfig:
    """Create VideoConfig with 4x scaling (960x640)."""
    return VideoConfig(scale=4)


# ============================================================================
# Helper Functions
# ============================================================================

def is_mgba_server_available(host: str = "localhost", port: int = 8888, timeout: float = 1.0) -> bool:
    """Check if mGBA server is available.

    Args:
        host: Server host
        port: Server port
        timeout: Connection timeout

    Returns:
        True if server is reachable
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_runtest_setup(item):
    """Pre-test setup hook.

    Adds timeout markers to integration tests to prevent hanging.
    """
    # Add timeout to integration tests
    if item.get_closest_marker("integration"):
        if not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(INTEGRATION_TEST_TIMEOUT))

    # Add timeout to model tests
    if item.get_closest_marker("model_test"):
        if not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(MODEL_TEST_TIMEOUT))

    # Add timeout to RAM tests
    if item.get_closest_marker("ram_test"):
        if not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(RAM_TEST_TIMEOUT))


def pytest_runtest_call(item):
    """Track test execution time."""
    import time
    start_time = time.time()
    
    def track_duration():
        duration = time.time() - start_time
        _test_durations.append((item.nodeid, duration))
    
    item.addfinalizer(track_duration)


# ============================================================================
# Environment Validation
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def validate_test_environment():
    """Validate test environment on session start.

    This fixture runs once at the start of the test session and logs
    important environment information.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check if mGBA server is available
    server_available = is_mgba_server_available()

    if server_available:
        logger.info("✅ mGBA server is available on localhost:8888")
    else:
        logger.warning("⚠️  mGBA server NOT available - integration tests will be skipped")

    # Log environment info
    logger.info(f"Python: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Test session starting...")

    yield

    # Session teardown
    logger.info("Test session complete")


def pytest_sessionfinish(session, exitstatus):
    """Clean up session-level resources and report timing."""
    import logging
    logger = logging.getLogger(__name__)

    # Cancel faulthandler alarm
    faulthandler.cancel_dump_traceback_later()

    # Report elapsed time
    if _session_start_time is not None:
        duration = time.time() - _session_start_time
        logger.info(f"Test session elapsed time: {duration:.1f}s")
    else:
        logger.warning("Session start time not recorded")

    # Print top slow tests
    if _test_durations:
        sorted_durations = sorted(_test_durations, key=lambda x: x[1], reverse=True)
        logger.info("Top slow tests:")
        for i, (nodeid, duration) in enumerate(sorted_durations[:5]):
            logger.info(f"  {i+1}. {nodeid}: {duration:.2f}s")

    logger.info("Session cleanup complete")
