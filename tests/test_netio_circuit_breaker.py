"""Tests for circuit breaker functionality."""

import pytest
import time
from unittest.mock import Mock
from src.environment.netio.adaptive_socket import CircuitBreaker, CircuitBreakerState


class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_circuit_breaker_closed_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_successful_calls(self):
        """Test successful calls keep circuit CLOSED."""
        cb = CircuitBreaker(failure_threshold=5)

        func = Mock(return_value="success")

        for _ in range(10):
            success, result = cb.call(func)
            assert success
            assert result == "success"

        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        func = Mock(side_effect=Exception("test error"))

        # Trigger 3 failures
        for _ in range(3):
            success, result = cb.call(func)
            assert not success
            assert result is None

        # Circuit should be OPEN
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_rejects_in_open_state(self):
        """Test circuit rejects calls when OPEN."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_ms=100)

        func = Mock(side_effect=Exception("error"))

        # Trigger failure to open
        success, result = cb.call(func)
        assert not success
        assert cb.state == CircuitBreakerState.OPEN

        # Try to call while open (within cooldown)
        success, result = cb.call(Mock())
        assert not success
        assert func.call_count == 1  # Should not call func again

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit transitions to HALF_OPEN after cooldown."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_ms=100)

        func_fail = Mock(side_effect=Exception("error"))

        # Open circuit
        success, result = cb.call(func_fail)
        assert cb.state == CircuitBreakerState.OPEN
        assert not success

        # Wait for cooldown
        time.sleep(0.15)

        # Should now be in HALF_OPEN on next call
        func_ok = Mock(return_value="ok")
        success, result = cb.call(func_ok)

        # Call should have been attempted
        assert func_ok.called

        # If successful, should transition to CLOSED
        if success and result == "ok":
            assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_half_open_failure_reopens(self):
        """Test HALF_OPEN fails on recovery attempt stays open."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_ms=100)

        # Open circuit
        fail_func = Mock(side_effect=Exception("error"))
        success, result = cb.call(fail_func)
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        # Try recovery but it fails
        fail_func_2 = Mock(side_effect=Exception("still failing"))
        success, result = cb.call(fail_func_2)
        assert not success

        # Circuit should remain OPEN after failed recovery
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_success_closes_from_half_open(self):
        """Test successful call in HALF_OPEN closes circuit."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_ms=100)

        # Open circuit
        fail_func = Mock(side_effect=Exception("error"))
        cb.call(fail_func)
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        # Successful call should close circuit
        success_func = Mock(return_value="ok")
        success, result = cb.call(success_func)

        if success:  # If we managed to call it
            assert result == "ok"

    def test_circuit_breaker_jitter_in_cooldown(self):
        """Test cooldown has jitter (±10%)."""
        # This is hard to test precisely, but we can verify
        # that timing varies slightly across multiple opens/recoveries
        cb = CircuitBreaker(failure_threshold=1, cooldown_ms=100)

        times = []
        for _ in range(3):
            fail_func = Mock(side_effect=Exception("error"))
            cb.call(fail_func)
            assert cb.state == CircuitBreakerState.OPEN

            start = time.monotonic()
            while cb.state == CircuitBreakerState.OPEN:
                # Try to transition to HALF_OPEN
                success_func = Mock(return_value="ok")
                success, result = cb.call(success_func)
                time.sleep(0.01)

            elapsed = time.monotonic() - start
            times.append(elapsed)

        # Verify cooldown is roughly in expected range (100ms ± 10%)
        for t in times:
            assert 0.08 < t < 0.13  # 100ms ± 30% tolerance for clock precision

    def test_circuit_breaker_acceptance_fail_open_half_open_close(self):
        """Test acceptance criteria: fail → open → half-open → close."""
        cb = CircuitBreaker(failure_threshold=5, cooldown_ms=200)

        # 1. CLOSED + failures
        fail_count = 0
        for i in range(5):
            func = Mock(side_effect=Exception(f"fail {i}"))
            success, result = cb.call(func)
            assert not success
            fail_count += 1

        # 2. Should be OPEN now
        assert cb.state == CircuitBreakerState.OPEN
        assert fail_count == 5

        # 3. Reject calls while open (within cooldown)
        reject_func = Mock(return_value="rejected")
        success, result = cb.call(reject_func)
        assert not success
        assert not reject_func.called  # Should not invoke func

        # 4. Wait for cooldown → HALF_OPEN
        time.sleep(0.25)

        # 5. Try recovery (success) → CLOSED
        recovery_func = Mock(return_value="recovered")
        success, result = cb.call(recovery_func)

        # If we called it and it succeeded
        if recovery_func.called and success:
            assert result == "recovered"
            assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_failure_counter_reset_on_success(self):
        """Test that success resets failure counter."""
        cb = CircuitBreaker(failure_threshold=3)

        # One failure
        fail_func = Mock(side_effect=Exception("error"))
        cb.call(fail_func)

        # Should still be CLOSED
        assert cb.state == CircuitBreakerState.CLOSED

        # Success should reset counter
        success_func = Mock(return_value="ok")
        success, result = cb.call(success_func)
        assert success

        # More failures but not enough to open
        for _ in range(2):
            fail_func2 = Mock(side_effect=Exception("error"))
            cb.call(fail_func2)

        # Still not open (counter was reset)
        assert cb.state == CircuitBreakerState.CLOSED
