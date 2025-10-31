"""Adaptive socket wrapper with rate limiting and circuit breaker for mGBA I/O.

Provides:
- Token-bucket rate limiter for screenshot & memory reads
- Circuit breaker with half-open retry and jitter
- Context-manager for lifecycle (connect, close, idempotent cleanup)
- No modification to original mgba_controller.py
"""

import time
import threading
import random
import logging
from typing import Optional, Callable, Any
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """States for circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimiter:
    """Token-bucket rate limiter for screenshot and memory read operations.

    Allows bursts up to `max_tokens` but enforces average rate of `max_rps` (requests per second).
    """

    def __init__(self, max_rps: float = 15.0, max_burst: Optional[int] = None):
        """Initialize token bucket.

        Args:
            max_rps: Maximum requests per second (rate limit)
            max_burst: Maximum burst tokens. If None, defaults to max_rps.
        """
        self.max_rps = max_rps
        self.max_burst = max_burst or int(max_rps * 2)  # 2-second burst capacity
        self._tokens = float(self.max_burst)
        self._last_refill = time.monotonic()
        self._lock = threading.RLock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_burst,
            self._tokens + (elapsed * self.max_rps)
        )
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket (non-blocking by default).

        Args:
            tokens: Number of tokens to acquire (default: 1.0)
            timeout: Max time to wait (not implemented for non-blocking version)

        Returns:
            True if tokens acquired, False if insufficient
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_if_needed(self, tokens: float = 1.0) -> None:
        """Block until tokens are available.

        Args:
            tokens: Number of tokens to acquire
        """
        while not self.acquire(tokens):
            # Sleep a small amount to avoid busy-waiting
            time.sleep(0.001)


class CircuitBreaker:
    """Circuit breaker with half-open retry logic and jitter.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests
    - HALF_OPEN: Testing if service recovered, allow retry
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_ms: int = 1200,
        max_half_open_requests: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            cooldown_ms: Milliseconds to wait before half-open (with jitter)
            max_half_open_requests: Concurrent requests allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_ms / 1000.0
        self.max_half_open_requests = max_half_open_requests

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_open_time = 0.0
        self._half_open_requests = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state."""
        return self._state

    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                # Close after successful half-open test
                if self._success_count >= 1:
                    self._state = CircuitBreakerState.CLOSED
                    self._success_count = 0
                    logger.info("Circuit breaker CLOSED after successful half-open test")

    def record_failure(self) -> None:
        """Record failed request."""
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._last_open_time = time.monotonic()
                self._failure_count = 0
                logger.warning(
                    f"Circuit breaker OPEN after {self.failure_threshold} failures"
                )

    def call(self, func: Callable[[], Any], *args, **kwargs) -> tuple[bool, Optional[Any]]:
        """Execute function with circuit breaker protection.

        Args:
            func: Callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            (success, result) tuple where success=False if rejected by circuit breaker
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                pass  # Allow request
            elif self._state == CircuitBreakerState.OPEN:
                # Check if cooldown has elapsed (with jitter)
                elapsed = time.monotonic() - self._last_open_time
                jitter = random.uniform(0, 0.1) * self.cooldown_s  # ±10% jitter
                adjusted_cooldown = self.cooldown_s + jitter

                if elapsed >= adjusted_cooldown:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info(
                        f"Circuit breaker HALF_OPEN after {adjusted_cooldown:.2f}s cooldown"
                    )
                else:
                    # Still in cooldown, reject request
                    return (False, None)

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Rate-limit half-open retries
                if self._half_open_requests >= self.max_half_open_requests:
                    return (False, None)
                self._half_open_requests += 1

        # Execute request outside lock
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return (True, result)
        except Exception as e:
            self.record_failure()
            logger.exception(f"Circuit breaker request failed: {e}")
            return (False, None)


class AdaptiveSocket:
    """Wraps a socket-like object with rate limiting and circuit breaker.

    Non-intrusive: wraps the transport without modifying its interface.
    """

    def __init__(
        self,
        transport: Any,
        max_rps: float = 15.0,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_ms: int = 1200,
    ):
        """Initialize adaptive socket wrapper.

        Args:
            transport: Underlying transport object (e.g., LuaSocketTransport)
            max_rps: Rate limit in requests per second
            circuit_failure_threshold: Failures before opening circuit
            circuit_cooldown_ms: Cooldown in milliseconds
        """
        self._transport = transport
        self._rate_limiter = RateLimiter(max_rps=max_rps)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cooldown_ms=circuit_cooldown_ms,
        )
        self._lock = threading.RLock()
        self._is_closed = False

    def connect(self) -> bool:
        """Connect the transport."""
        return self._transport.connect()

    def disconnect(self) -> None:
        """Disconnect the transport."""
        if hasattr(self._transport, "disconnect"):
            self._transport.disconnect()

    def is_connected(self) -> bool:
        """Check if transport is connected."""
        if hasattr(self._transport, "is_connected"):
            return self._transport.is_connected()
        return False

    def send_command(self, command: str, *args: str) -> Optional[str]:
        """Send command with rate limiting and circuit breaker protection.

        Args:
            command: Command to send
            *args: Command arguments

        Returns:
            Response string or None if rejected/failed
        """
        # Check if already closed
        if self._is_closed:
            logger.warning("Cannot send command: adapter is closed")
            return None

        # Rate limit the request
        self._rate_limiter.wait_if_needed()

        # Try to execute through circuit breaker
        def _send():
            return self._transport.send_command(command, *args)

        success, result = self._circuit_breaker.call(_send)
        if not success:
            logger.warning(f"Command rejected by circuit breaker: {command}")
            return None
        return result

    def close(self) -> None:
        """Close the adapter and underlying transport (idempotent)."""
        with self._lock:
            if not self._is_closed:
                self.disconnect()
                self._is_closed = True
                logger.info("AdaptiveSocket closed")

    @contextmanager
    def managed(self):
        """Context manager for safe lifecycle.

        Usage:
            with adapter.managed():
                adapter.send_command(...)
        """
        try:
            yield self
        finally:
            self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (idempotent cleanup)."""
        self.close()
        return False
