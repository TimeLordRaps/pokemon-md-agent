"""Screenshot guard with debounce and single-flight pattern.

Prevents thundering herd of screenshot requests:
- Debounces rapid calls (collapses within debounce window)
- Implements single-flight pattern (concurrent requests wait for same result)
- Non-intrusive: wraps the controller without modification
"""

import threading
import time
import logging
from typing import Optional, Callable, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ScreenshotGuard:
    """Debounces rapid screenshot calls and implements single-flight pattern.

    Concurrent requests within debounce window are collapsed to a single call.
    """

    def __init__(self, debounce_ms: int = 100):
        """Initialize screenshot guard.

        Args:
            debounce_ms: Milliseconds to wait before executing screenshot after last call
        """
        self.debounce_s = debounce_ms / 1000.0
        self._pending_calls = defaultdict(dict)  # key -> {last_call_time, timer, event, result}
        self._lock = threading.RLock()

    def take_screenshot(
        self,
        screenshot_func: Callable[[str], bool],
        path: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Take a screenshot with debounce and single-flight.

        If multiple calls arrive with the same path within debounce window,
        only one actual screenshot is taken and result is shared.

        Args:
            screenshot_func: Callable that takes path and returns bool
            path: File path for screenshot
            timeout: Seconds to wait for result (None = wait forever)

        Returns:
            Result from screenshot_func or False if rejected/timed out
        """
        key = path

        with self._lock:
            if key not in self._pending_calls:
                # First call for this path
                self._pending_calls[key] = {
                    "result": None,
                    "last_call_time": time.monotonic(),
                    "timer": None,
                    "event": threading.Event(),
                    "executing": False,
                }

            call_info = self._pending_calls[key]

            # Cancel previous timer if it exists
            if call_info["timer"]:
                call_info["timer"].cancel()

            # Update last call time
            call_info["last_call_time"] = time.monotonic()

            # If already executing, wait for result
            if call_info["executing"]:
                # Another thread is executing, wait for result
                event = call_info["event"]

            else:
                # Schedule execution after debounce delay
                def _execute():
                    with self._lock:
                        if call_info["executing"]:
                            return  # Already executing
                        call_info["executing"] = True

                    try:
                        result = screenshot_func(path)
                        with self._lock:
                            call_info["result"] = result
                            call_info["event"].set()
                        logger.debug(f"Screenshot executed for {path}: {result}")
                    except Exception as e:
                        logger.exception(f"Screenshot failed for {path}: {e}")
                        with self._lock:
                            call_info["result"] = False
                            call_info["event"].set()
                    finally:
                        with self._lock:
                            call_info["executing"] = False

                call_info["timer"] = threading.Timer(self.debounce_s, _execute)
                call_info["timer"].daemon = True
                call_info["timer"].start()

                event = call_info["event"]

        # Wait for result (outside lock to prevent deadlock)
        if event.wait(timeout=timeout):
            result = call_info["result"]
            # Clean up after success
            with self._lock:
                if key in self._pending_calls:
                    del self._pending_calls[key]
            return result or False

        logger.warning(f"Screenshot request timed out for {path}")
        return False

    def cancel_pending(self, path: str) -> None:
        """Cancel pending screenshot for given path.

        Args:
            path: File path to cancel
        """
        with self._lock:
            if path in self._pending_calls:
                call_info = self._pending_calls[path]
                if call_info["timer"]:
                    call_info["timer"].cancel()
                del self._pending_calls[path]
                logger.debug(f"Cancelled pending screenshot for {path}")

    def cancel_all_pending(self) -> None:
        """Cancel all pending screenshots."""
        with self._lock:
            for call_info in self._pending_calls.values():
                if call_info["timer"]:
                    call_info["timer"].cancel()
            self._pending_calls.clear()
            logger.info("Cancelled all pending screenshots")

    def get_pending_count(self) -> int:
        """Get number of pending screenshot requests."""
        with self._lock:
            return len(self._pending_calls)
