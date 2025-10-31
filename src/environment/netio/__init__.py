"""Non-intrusive I/O hardening module for mGBA controller.

Provides:
- adaptive_socket: Rate-limited socket wrapper with circuit breaker
- screenshot_guard: Debounced, single-flight screenshot requests
- Config integration for opt-in rate-limiting and resilience
"""

from .adaptive_socket import AdaptiveSocket, RateLimiter, CircuitBreaker
from .screenshot_guard import ScreenshotGuard

__all__ = [
    "AdaptiveSocket",
    "RateLimiter",
    "CircuitBreaker",
    "ScreenshotGuard",
]
