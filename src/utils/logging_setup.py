"""Standard logging setup for PMD-Red agent with JSON/telemetry and human-readable debug output."""

from typing import Dict, Any, Optional
import logging
import logging.handlers
import json
import sys
from pathlib import Path
import os


class JSONFormatter(logging.Formatter):
    """JSON formatter for telemetry logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add custom fields from record (extra parameters passed to logging calls)
        # Python logging puts extra fields directly as attributes on the LogRecord
        standard_fields = {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info', 
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process', 'message', 
                          'asctime', 'getMessage'}
        
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for debug/console output."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class LoggerSetup:
    """Centralized logging configuration with multiple output formats."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        enable_json: bool = True,
        enable_console: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB per file
        backup_count: int = 5,
        app_name: str = "pmd-red-agent",
    ):
        """Initialize logger setup.

        Args:
            log_dir: Directory for log files (defaults to project logs/)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_json: Enable JSON telemetry logging
            enable_console: Enable console human-readable logging
            max_bytes: Max bytes per log file before rotation
            backup_count: Number of backup files to keep
            app_name: Application name for log file naming
        """
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_json = enable_json
        self.enable_console = enable_console
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.app_name = app_name

        # Track configured loggers to avoid duplicate setup
        self._configured_loggers: set[str] = set()

        # Create formatters
        self.json_formatter = JSONFormatter()
        self.human_formatter = HumanReadableFormatter()

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a configured logger.

        Args:
            name: Logger name (usually __name__)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Avoid reconfiguring the same logger
        if name in self._configured_loggers:
            return logger

        logger.setLevel(self.log_level)

        # Remove any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add JSON telemetry handler if enabled
        if self.enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.app_name}.jsonl",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            json_handler.setFormatter(self.json_formatter)
            json_handler.setLevel(self.log_level)
            logger.addHandler(json_handler)

        # Add human-readable file handler
        human_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name}.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        human_handler.setFormatter(self.human_formatter)
        human_handler.setLevel(self.log_level)
        logger.addHandler(human_handler)

        # Add console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.human_formatter)
            console_handler.setLevel(self.log_level)
            logger.addHandler(console_handler)

        self._configured_loggers.add(name)
        return logger

    def configure_root_logger(self) -> None:
        """Configure the root logger with basic setup."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.human_formatter)
            console_handler.setLevel(self.log_level)
            root_logger.addHandler(console_handler)

    def set_level(self, level: str) -> None:
        """Set logging level for all configured loggers.

        Args:
            level: New logging level
        """
        new_level = getattr(logging, level.upper(), logging.INFO)
        self.log_level = new_level

        # Update all configured loggers
        for logger_name in self._configured_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setLevel(new_level)

    def shutdown(self) -> None:
        """Shutdown all loggers and close file handles."""
        # Close all handlers for configured loggers
        for logger_name in self._configured_loggers:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                # Flush any buffered data before closing
                if hasattr(handler, 'flush'):
                    handler.flush()
                if hasattr(handler, 'close'):
                    handler.close()
                logger.removeHandler(handler)
        
        # Clear the configured loggers set
        self._configured_loggers.clear()

    def get_log_files(self) -> Dict[str, str]:
        """Get paths to current log files.

        Returns:
            Dict mapping log type to file path
        """
        files = {}
        if self.enable_json:
            files["json"] = str(self.log_dir / f"{self.app_name}.jsonl")
        files["human"] = str(self.log_dir / f"{self.app_name}.log")
        return files


# Global logger setup instance
_logger_setup: Optional[LoggerSetup] = None


def get_logger_setup() -> LoggerSetup:
    """Get the global logger setup instance, creating it if needed."""
    global _logger_setup
    if _logger_setup is None:
        _logger_setup = LoggerSetup()
    return _logger_setup


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_console: bool = True,
    app_name: str = "pmd-red-agent",
) -> LoggerSetup:
    """Convenience function to setup logging globally.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        enable_json: Enable JSON telemetry logging
        enable_console: Enable console output
        app_name: Application name

    Returns:
        LoggerSetup instance
    """
    global _logger_setup
    _logger_setup = LoggerSetup(
        log_dir=log_dir,
        log_level=log_level,
        enable_json=enable_json,
        enable_console=enable_console,
        app_name=app_name,
    )
    _logger_setup.configure_root_logger()
    return _logger_setup


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return get_logger_setup().get_logger(name)


def shutdown_logging() -> None:
    """Shutdown the global logger setup and close all file handles."""
    global _logger_setup
    if _logger_setup is not None:
        _logger_setup.shutdown()
        _logger_setup = None