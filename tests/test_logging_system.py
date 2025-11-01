"""Test script to exercise the logging system and verify telemetry capture.

This script tests:
- JSON telemetry logging (.jsonl files)
- Human-readable logging (.log files)
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Module-specific logging
- Log rotation functionality
- Structured metadata in logs
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.utils.logging_setup import LoggerSetup, setup_logging, get_logger


class TestLoggingSystem:
    """Test the complete logging system functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger_setup(self, temp_log_dir):
        """Create a logger setup instance for testing."""
        return LoggerSetup(
            log_dir=str(temp_log_dir),
            log_level="DEBUG",
            enable_json=True,
            enable_console=False,  # Disable console for clean test output
            app_name="test-agent",
            max_bytes=1024,  # Small size to test rotation
            backup_count=2
        )

    def test_basic_logging_setup(self, logger_setup, temp_log_dir):
        """Test basic logger setup and file creation."""
        # Get a logger
        logger = logger_setup.get_logger("test.module")

        # Log a message
        logger.info("Test message")

        # Check files exist
        json_file = temp_log_dir / "test-agent.jsonl"
        human_file = temp_log_dir / "test-agent.log"

        assert json_file.exists(), "JSON log file should be created"
        assert human_file.exists(), "Human-readable log file should be created"

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_log_levels(self, logger_setup, temp_log_dir):
        """Test different logging levels are captured."""
        logger = logger_setup.get_logger("test.levels")

        # Log at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Check JSON log content
        json_file = temp_log_dir / "test-agent.jsonl"
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        assert len(lines) == 4, "Should have 4 log entries"

        # Parse and verify levels
        entries = [json.loads(line.strip()) for line in lines]
        levels = [entry['level'] for entry in entries]

        assert "DEBUG" in levels
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_module_specific_logging(self, logger_setup, temp_log_dir):
        """Test logging from different modules/namespaces."""
        logger1 = logger_setup.get_logger("module1.submodule")
        logger2 = logger_setup.get_logger("module2.submodule")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Check JSON logs
        json_file = temp_log_dir / "test-agent.jsonl"
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        entries = [json.loads(line.strip()) for line in lines]

        loggers = [entry['logger'] for entry in entries]
        assert "module1.submodule" in loggers
        assert "module2.submodule" in loggers

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_json_telemetry_metadata(self, logger_setup, temp_log_dir):
        """Test that JSON logs contain proper metadata."""
        logger = logger_setup.get_logger("test.metadata")

        # Log with extra fields (simulating telemetry data)
        extra_data = {
            "step": 42,
            "latency_ms": 150.5,
            "model": "qwen-vl",
            "action": "explore"
        }

        # Create a log record with extra fields
        logger.info("Telemetry event", extra=extra_data)

        # Check JSON structure
        json_file = temp_log_dir / "test-agent.jsonl"
        with open(json_file, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            entry = json.loads(line)

        # Verify standard fields
        assert "timestamp" in entry
        assert "level" in entry
        assert "logger" in entry
        assert "message" in entry

        # Verify extra fields are included
        assert entry["step"] == 42
        assert entry["latency_ms"] == 150.5
        assert entry["model"] == "qwen-vl"
        assert entry["action"] == "explore"

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_human_readable_format(self, logger_setup, temp_log_dir):
        """Test human-readable log format."""
        logger = logger_setup.get_logger("test.human")

        logger.info("Human readable message")

        # Check human-readable log
        human_file = temp_log_dir / "test-agent.log"
        with open(human_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should contain timestamp, level, logger, message
        assert "INFO" in content
        assert "test.human" in content
        assert "Human readable message" in content

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_log_rotation(self, logger_setup, temp_log_dir):
        """Test log rotation when file size limit is reached."""
        logger = logger_setup.get_logger("test.rotation")

        # Generate enough log messages to trigger rotation
        # Each message is ~100 chars, max_bytes=1024, so ~10 messages should rotate
        for i in range(15):
            logger.info(f"Rotation test message {i:02d}: " + "x" * 50)

        # Check for rotated files
        json_files = list(temp_log_dir.glob("test-agent.jsonl*"))
        human_files = list(temp_log_dir.glob("test-agent.log*"))

        # Should have original + at least one backup
        assert len(json_files) >= 2, f"Expected rotation files, found: {json_files}"
        assert len(human_files) >= 2, f"Expected rotation files, found: {human_files}"

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_exception_logging(self, logger_setup, temp_log_dir):
        """Test logging of exceptions."""
        logger = logger_setup.get_logger("test.exception")

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred")

        # Check JSON log contains exception info
        json_file = temp_log_dir / "test-agent.jsonl"
        with open(json_file, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            entry = json.loads(line)

        assert "exception" in entry
        assert "Test exception" in entry["exception"]

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_global_setup_function(self, temp_log_dir):
        """Test the global setup_logging function."""
        setup = setup_logging(
            log_dir=str(temp_log_dir),
            log_level="INFO",
            enable_json=True,
            enable_console=False,
            app_name="global-test"
        )

        logger = get_logger("global.test")
        logger.info("Global setup test")

        # Verify files created
        json_file = temp_log_dir / "global-test.jsonl"
        human_file = temp_log_dir / "global-test.log"

        assert json_file.exists()
        assert human_file.exists()

        # Shutdown the global logger setup
        from src.utils.logging_setup import shutdown_logging
        shutdown_logging()

    def test_logger_reuse(self, logger_setup):
        """Test that getting the same logger multiple times returns the same instance."""
        logger1 = logger_setup.get_logger("reuse.test")
        logger2 = logger_setup.get_logger("reuse.test")

        assert logger1 is logger2, "Same logger name should return same instance"

        # Properly shutdown to close file handles
        logger_setup.shutdown()

    def test_multiple_loggers_isolation(self, logger_setup, temp_log_dir):
        """Test that multiple loggers don't interfere with each other."""
        logger1 = logger_setup.get_logger("isolation.one")
        logger2 = logger_setup.get_logger("isolation.two")

        logger1.setLevel(logging.ERROR)  # Only log errors
        logger2.setLevel(logging.DEBUG)  # Log everything

        logger1.info("This should not appear")  # Below ERROR level
        logger1.error("This should appear")      # At ERROR level

        logger2.debug("This should appear")      # At DEBUG level
        logger2.info("This should appear")       # At INFO level

        # Count entries in log file
        json_file = temp_log_dir / "test-agent.jsonl"
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        entries = [json.loads(line.strip()) for line in lines]

        # Should have 3 entries: one ERROR from logger1, two from logger2
        assert len(entries) == 3

        loggers = [entry['logger'] for entry in entries]
        messages = [entry['message'] for entry in entries]

        assert "isolation.one" in loggers  # Only the ERROR message
        assert loggers.count("isolation.two") == 2  # Both DEBUG and INFO
        assert "This should appear" in messages

        # Properly shutdown to close file handles
        logger_setup.shutdown()


if __name__ == "__main__":
    # Run basic functionality test when executed directly
    print("Running logging system test...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_log_dir = Path(tmpdir)

        # Setup logging
        setup = LoggerSetup(
            log_dir=str(temp_log_dir),
            log_level="DEBUG",
            enable_json=True,
            enable_console=True,
            app_name="manual-test"
        )

        # Test different loggers and levels
        agent_logger = setup.get_logger("agent.core")
        vision_logger = setup.get_logger("vision.grid_parser")
        telemetry_logger = setup.get_logger("telemetry.events")

        # Generate various log messages
        agent_logger.debug("Agent initialization started")
        agent_logger.info("Agent configured successfully")

        vision_logger.warning("Grid parsing encountered minor issue")
        vision_logger.error("Failed to parse grid data")

        # Telemetry-style logging with extra data
        telemetry_logger.info("Model inference completed", extra={
            "step": 123,
            "latency_ms": 250.0,
            "model": "qwen-vl-7b",
            "tokens": 150,
            "action": "decision_making"
        })

        # Test exception logging
        try:
            raise RuntimeError("Simulated runtime error")
        except RuntimeError:
            agent_logger.exception("Runtime error in agent loop")

        # Wait a bit to ensure file writes complete
        time.sleep(0.1)

        # Verify output files
        json_file = temp_log_dir / "manual-test.jsonl"
        human_file = temp_log_dir / "manual-test.log"

        print(f"JSON telemetry log: {json_file}")
        print(f"Human-readable log: {human_file}")

        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                json_lines = f.readlines()
            print(f"JSON log entries: {len(json_lines)}")

            # Show first JSON entry as example
            if json_lines:
                try:
                    entry = json.loads(json_lines[0].strip())
                    print("Sample JSON entry:", json.dumps(entry, indent=2))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")

        if human_file.exists():
            with open(human_file, 'r', encoding='utf-8') as f:
                human_content = f.read()
            print(f"Human log length: {len(human_content)} chars")
            print("First few lines of human log:")
            print('\n'.join(human_content.split('\n')[:3]))

        print("Manual test completed successfully!")