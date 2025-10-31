"""Tests for screenshot debounce and single-flight guard."""

import pytest
import time
import threading
from unittest.mock import Mock
from src.environment.netio.screenshot_guard import ScreenshotGuard


class TestScreenshotGuard:
    """Test screenshot debounce and single-flight pattern."""

    def test_screenshot_guard_single_call(self):
        """Test single screenshot call executes normally."""
        guard = ScreenshotGuard(debounce_ms=50)

        screenshot_func = Mock(return_value=True)

        result = guard.take_screenshot(screenshot_func, "/tmp/screen.png", timeout=1.0)

        assert result is True
        assert screenshot_func.call_count == 1

    def test_screenshot_guard_debounce_collapses_calls(self):
        """Test rapid calls within debounce window are collapsed to one."""
        guard = ScreenshotGuard(debounce_ms=100)

        screenshot_func = Mock(return_value=True)

        # Rapid fire 5 calls to same path
        threads = []
        results = []

        def _call():
            result = guard.take_screenshot(
                screenshot_func, "/tmp/screen.png", timeout=1.0
            )
            results.append(result)

        for _ in range(5):
            t = threading.Thread(target=_call)
            t.start()
            threads.append(t)
            time.sleep(0.01)  # Stagger slightly but within debounce

        for t in threads:
            t.join()

        # All should succeed
        assert all(results)

        # But only one actual execution
        assert screenshot_func.call_count == 1

    def test_screenshot_guard_different_paths(self):
        """Test different paths execute independently."""
        guard = ScreenshotGuard(debounce_ms=50)

        screenshot_func = Mock(return_value=True)

        result1 = guard.take_screenshot(screenshot_func, "/tmp/screen1.png", timeout=1.0)
        result2 = guard.take_screenshot(screenshot_func, "/tmp/screen2.png", timeout=1.0)

        assert result1 is True
        assert result2 is True

        # Each path should have one call
        assert screenshot_func.call_count == 2

    def test_screenshot_guard_timeout(self):
        """Test timeout waiting for debounced result."""
        guard = ScreenshotGuard(debounce_ms=500)

        # Function that never completes
        screenshot_func = Mock()

        def _slow_screenshot(path):
            time.sleep(10)  # Simulate slow execution
            return True

        # Call with short timeout
        result = guard.take_screenshot(
            _slow_screenshot, "/tmp/slow.png", timeout=0.1
        )

        # Should timeout and return False
        assert result is False

    def test_screenshot_guard_function_failure(self):
        """Test handling of function exceptions."""
        guard = ScreenshotGuard(debounce_ms=50)

        screenshot_func = Mock(side_effect=Exception("screenshot failed"))

        result = guard.take_screenshot(screenshot_func, "/tmp/fail.png", timeout=1.0)

        assert result is False

    def test_screenshot_guard_pending_count(self):
        """Test tracking of pending requests."""
        guard = ScreenshotGuard(debounce_ms=200)

        screenshot_func = Mock(side_effect=lambda p: time.sleep(0.1) or True)

        # Start a debounced call (will be pending)
        threading.Thread(
            target=guard.take_screenshot,
            args=(screenshot_func, "/tmp/pending.png"),
            kwargs={"timeout": 2.0},
        ).start()

        # Immediately check pending count
        time.sleep(0.01)
        pending = guard.get_pending_count()
        assert pending >= 1

        # Wait for completion
        time.sleep(0.5)
        pending = guard.get_pending_count()
        assert pending == 0

    def test_screenshot_guard_cancel_pending(self):
        """Test cancelling a pending screenshot."""
        guard = ScreenshotGuard(debounce_ms=200)

        screenshot_func = Mock(return_value=True)

        # Start debounced call
        t = threading.Thread(
            target=guard.take_screenshot,
            args=(screenshot_func, "/tmp/cancel.png"),
            kwargs={"timeout": 2.0},
        )
        t.start()

        # Cancel it
        time.sleep(0.05)
        guard.cancel_pending("/tmp/cancel.png")

        # Wait for thread
        t.join(timeout=1.0)

        # Function should not have been called (cancelled before debounce elapsed)
        assert screenshot_func.call_count == 0

    def test_screenshot_guard_cancel_all(self):
        """Test cancelling all pending screenshots."""
        guard = ScreenshotGuard(debounce_ms=200)

        screenshot_func = Mock(return_value=True)

        # Start multiple debounced calls
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=guard.take_screenshot,
                args=(screenshot_func, f"/tmp/cancel{i}.png"),
                kwargs={"timeout": 2.0},
            )
            t.start()
            threads.append(t)

        # Cancel all
        time.sleep(0.05)
        guard.cancel_all_pending()

        # Wait for threads
        for t in threads:
            t.join(timeout=1.0)

        # No executions should have happened
        assert screenshot_func.call_count == 0

    def test_screenshot_guard_acceptance_concurrent_collapse_to_one(self):
        """Test acceptance criteria: 50 concurrent calls collapse to single execution.

        When spammed with 50 screenshot calls to same path, only 1 reaches underlying func.
        """
        guard = ScreenshotGuard(debounce_ms=100)

        screenshot_func = Mock(return_value=True)
        results = []
        errors = []

        def _spam_call(index):
            try:
                result = guard.take_screenshot(
                    screenshot_func, "/tmp/spam.png", timeout=2.0
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Launch 50 concurrent calls
        threads = []
        start = time.monotonic()

        for i in range(50):
            t = threading.Thread(target=_spam_call, args=(i,))
            t.start()
            threads.append(t)

        # Wait for all to complete
        for t in threads:
            t.join(timeout=3.0)

        elapsed = time.monotonic() - start

        # All 50 calls should return True (single result shared)
        assert len(results) == 50
        assert all(results)

        # Only ONE actual screenshot execution
        assert screenshot_func.call_count == 1
        assert len(errors) == 0

        # Should complete quickly (debounce + single call)
        assert elapsed < 1.0

    def test_screenshot_guard_multiple_sequential_calls_after_debounce(self):
        """Test that calls after debounce elapse execute independently."""
        guard = ScreenshotGuard(debounce_ms=100)

        screenshot_func = Mock(return_value=True)

        # First call
        result1 = guard.take_screenshot(screenshot_func, "/tmp/seq.png", timeout=1.0)
        assert result1
        assert screenshot_func.call_count == 1

        # Wait longer than debounce
        time.sleep(0.15)

        # Second call should execute independently
        result2 = guard.take_screenshot(screenshot_func, "/tmp/seq.png", timeout=1.0)
        assert result2
        assert screenshot_func.call_count == 2

    def test_screenshot_guard_false_return_value(self):
        """Test handling of False return from screenshot function."""
        guard = ScreenshotGuard(debounce_ms=50)

        screenshot_func = Mock(return_value=False)

        result = guard.take_screenshot(screenshot_func, "/tmp/false.png", timeout=1.0)

        assert result is False  # Should propagate False
        assert screenshot_func.call_count == 1
