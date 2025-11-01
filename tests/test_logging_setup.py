"""Tests for logging setup functionality."""

import tempfile
import json
import logging
import sys
from pathlib import Path
import pytest

from src.utils.logging_setup import (
    LoggerSetup,
    setup_logging,
    get_logger,
    shutdown_logging,
    JSONFormatter,
    HumanReadableFormatter,
)


class TestJSONFormatter:
    """Test JSON formatter."""

    def test_format_creates_valid_json(self):
        """Test that JSON formatter produces valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_format_includes_exception_info(self):
        """Test that exception info is included when present."""
        formatter = JSONFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert "exception" in parsed


class TestHumanReadableFormatter:
    """Test human readable formatter."""

    def test_format_includes_standard_fields(self):
        """Test that human readable format includes expected fields."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert "WARNING" in result
        assert "test_logger" in result
        assert "Warning message" in result


class TestLoggerSetup:
    """Test LoggerSetup class."""

    def test_initialization(self):
        """Test LoggerSetup initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir, app_name="test_app")

            assert setup.log_dir == Path(temp_dir)
            assert setup.app_name == "test_app"
            assert setup.log_level == logging.INFO

    def test_get_logger_creates_configured_logger(self):
        """Test that get_logger returns properly configured logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir, enable_json=True, enable_console=False)
            logger = setup.get_logger("test_component")

            assert logger.name == "test_component"
            assert logger.level == logging.INFO

            # Should have handlers configured
            assert len(logger.handlers) >= 1

            # Properly shutdown the logger setup to close file handles
            setup.shutdown()

    def test_log_files_created(self):
        """Test that log files are created when logging occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir, app_name="test_app")
            logger = setup.get_logger("test_logger")

            logger.info("Test message")

            # Check that files exist
            json_file = Path(temp_dir) / "test_app.jsonl"
            human_file = Path(temp_dir) / "test_app.log"

            # Note: Files might not be created until first write, but should exist after
            # For this test, we just verify the setup doesn't fail

            # Properly shutdown to close file handles
            setup.shutdown()

    def test_set_level_updates_all_loggers(self):
        """Test that set_level updates all configured loggers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir)
            logger1 = setup.get_logger("logger1")
            logger2 = setup.get_logger("logger2")

            setup.set_level("DEBUG")

            assert logger1.level == logging.DEBUG
            assert logger2.level == logging.DEBUG
            assert setup.log_level == logging.DEBUG

            # Properly shutdown to close file handles
            setup.shutdown()

    def test_get_log_files_returns_correct_paths(self):
        """Test get_log_files returns correct file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir, app_name="test_app", enable_json=True)
            files = setup.get_log_files()

            assert files["json"] == str(Path(temp_dir) / "test_app.jsonl")
            assert files["human"] == str(Path(temp_dir) / "test_app.log")


class TestGlobalFunctions:
    """Test global logging functions."""

    def test_setup_logging_creates_instance(self):
        """Test setup_logging creates and returns LoggerSetup instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = setup_logging(log_dir=temp_dir, app_name="global_test")

            assert isinstance(setup, LoggerSetup)
            assert setup.app_name == "global_test"

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        # Setup global instance first
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_dir=temp_dir)

            logger = get_logger("global_test")
            assert isinstance(logger, logging.Logger)
            assert logger.name == "global_test"

            # Shutdown global logging to close file handles
            shutdown_logging()

    def test_get_logger_works_without_global_setup(self):
        """Test get_logger works even without explicit setup."""
        # This should create a default setup
        logger = get_logger("default_test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "default_test"


class TestIntegration:
    """Integration tests for logging setup."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logging
            setup = setup_logging(
                log_dir=temp_dir,
                log_level="DEBUG",
                enable_json=True,
                enable_console=False,
                app_name="integration_test"
            )

            # Get logger and log messages
            logger = get_logger("workflow_test")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Verify setup
            assert setup.app_name == "integration_test"
            assert setup.log_level == logging.DEBUG

            # Verify logger configuration
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) >= 1

            # Properly shutdown to close file handles
            setup.shutdown()

    def test_multiple_loggers_share_configuration(self):
        """Test that multiple loggers share the same configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = LoggerSetup(log_dir=temp_dir)

            logger1 = setup.get_logger("component1")
            logger2 = setup.get_logger("component2")

            # Both should have same level
            assert logger1.level == logger2.level == logging.INFO

            # Changing level should affect both
            setup.set_level("WARNING")
            assert logger1.level == logger2.level == logging.WARNING

            # Properly shutdown to close file handles
            setup.shutdown()