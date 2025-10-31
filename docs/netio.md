# mGBA I/O Hardening Module (netio)

## Overview

The `netio` module provides non-intrusive I/O hardening for the mGBA controller, eliminating intermittent socket/screenshot faults without modifying the core `mgba_controller.py`.

### Key Features

- **Rate Limiting**: Token-bucket algorithm for screenshot and memory read operations
- **Circuit Breaker**: Automatic failure detection with graceful half-open recovery
- **Screenshot Guard**: Debounce and single-flight pattern for concurrent requests
- **Opt-in Design**: Drop-in adapter with configuration-driven activation
- **Lifecycle Safety**: Context managers and idempotent cleanup

## Architecture

### Components

#### AdaptiveSocket

Wraps a socket-like transport with rate limiting and circuit breaker protection.

**Key Methods:**
- `send_command(command, *args)`: Send command with rate limiting and circuit breaker
- `connect()`: Establish connection
- `disconnect()`: Close connection
- `close()`: Idempotent cleanup
- `__enter__()` / `__exit__()`: Context manager support

**Configuration:**
- `max_rps`: Max requests per second (default: 15.0)
- `circuit_failure_threshold`: Failures before opening (default: 5)
- `circuit_cooldown_ms`: Cooldown before half-open retry (default: 1200ms)

#### RateLimiter

Token-bucket rate limiter for controlling request throughput.

**Key Methods:**
- `acquire(tokens=1.0)`: Non-blocking token acquisition (returns bool)
- `wait_if_needed(tokens=1.0)`: Blocking wait until tokens available

**Behavior:**
- Burst capacity: 2× `max_rps` tokens by default
- Refills at `max_rps` tokens per second
- Thread-safe with RLock

#### CircuitBreaker

Circuit breaker with three states: CLOSED → OPEN → HALF_OPEN → CLOSED

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Too many failures, requests rejected, retry after cooldown
- **HALF_OPEN**: Testing recovery, allow limited probe requests

**Key Methods:**
- `call(func, *args, **kwargs)`: Execute function with protection
- `record_success()`: Mark successful request
- `record_failure()`: Mark failed request
- `state`: Current state property

**Features:**
- Configurable failure threshold and cooldown
- Jitter on cooldown (±10%) to prevent thundering herd
- Thread-safe state transitions

#### ScreenshotGuard

Debounces rapid screenshot calls and implements single-flight pattern.

**Key Methods:**
- `take_screenshot(func, path, timeout=None)`: Take screenshot with protection
- `cancel_pending(path)`: Cancel pending screenshot
- `cancel_all_pending()`: Cancel all pending
- `get_pending_count()`: Query pending request count

**Behavior:**
- Collapses concurrent calls to same path into single execution
- Debounce window (default: 100ms) prevents rapid repeated calls
- Concurrent callers wait for shared result
- Thread-safe with condition variables

## Usage Guide

### Basic Usage: Opt-in with AdaptiveSocket

```python
from src.environment.mgba_controller import MGBAController
from src.environment.netio import AdaptiveSocket

# Create original controller
controller = MGBAController(host="localhost", port=8888)

# Wrap with adaptive socket for rate limiting + circuit breaker
adaptive = AdaptiveSocket(
    transport=controller._transport,
    max_rps=15.0,
    circuit_failure_threshold=5,
    circuit_cooldown_ms=1200,
)

# Use context manager for safe lifecycle
with adaptive:
    # Send commands through adaptive socket
    response = adaptive.send_command("core.screenshot", "/path/to/screenshot.png")
```

### Configuration-Driven Setup

```python
from src.environment.netio import AdaptiveSocket

# Read from config
config = {
    'IO_MAX_RPS': 15.0,
    'IO_CIRCUIT_FAILS': 5,
    'IO_CIRCUIT_COOLDOWN_MS': 1200,
}

# Wrap transport
adaptive = AdaptiveSocket(
    transport=controller._transport,
    max_rps=config['IO_MAX_RPS'],
    circuit_failure_threshold=config['IO_CIRCUIT_FAILS'],
    circuit_cooldown_ms=config['IO_CIRCUIT_COOLDOWN_MS'],
)
```

### Screenshot Protection

```python
from src.environment.netio import ScreenshotGuard

guard = ScreenshotGuard(debounce_ms=100)

# Collapse rapid calls to same path
result = guard.take_screenshot(
    screenshot_func=controller.screenshot,
    path="/tmp/screen.png",
    timeout=2.0,
)

# Clean up pending requests on exit
guard.cancel_all_pending()
```

### Combined Protection Pattern

```python
from src.environment.netio import AdaptiveSocket, ScreenshotGuard

# Create wrapped transport
adaptive = AdaptiveSocket(
    transport=controller._transport,
    max_rps=15.0,
    circuit_failure_threshold=5,
    circuit_cooldown_ms=1200,
)

# Create screenshot guard
guard = ScreenshotGuard(debounce_ms=100)

# Use both together
with adaptive:
    result = guard.take_screenshot(
        screenshot_func=lambda p: adaptive.send_command("core.screenshot", p),
        path="/tmp/hardened_screenshot.png",
        timeout=2.0,
    )
    guard.cancel_all_pending()
```

## Configuration

### config/mgba_config.ini

```ini
[io_hardening]
enable_adaptive_socket = false
IO_MAX_RPS = 15.0
IO_CIRCUIT_FAILS = 5
IO_CIRCUIT_COOLDOWN_MS = 1200

[screenshot_guard]
enable_screenshot_guard = false
SCREENSHOT_DEBOUNCE_MS = 100
```

## Acceptance Criteria

### Rate Limiting

**Criteria:** When spammed with 50 screenshot calls in 2 seconds, only ≤30 reach mGBA.

**Configuration:**
- `IO_MAX_RPS = 15.0`
- Default burst = 30 tokens (2× rate)
- Result: ~30 requests succeed, 20 are rate-limited

### Circuit Breaker

**Criteria:** Fail → Open → Half-Open → Close recovery cycle.

**Behavior:**
1. Failures accumulate until threshold (5) → OPEN
2. Requests rejected with jittered cooldown (1200ms ± 10%)
3. Cooldown expires → HALF_OPEN
4. Test request succeeds → CLOSED
5. Back to normal operation

### Screenshot Guard

**Criteria:** Concurrent calls collapse to single execution.

**Behavior:**
- 50 concurrent calls to same path within debounce window
- Only 1 actual screenshot execution
- All 50 callers receive shared result
- No lingering resources after completion

## Testing

Run all netio tests:

```bash
pytest tests/test_netio_*.py -v
```

Individual test modules:
- `test_netio_rate_limits.py`: Rate limiting and burst behavior
- `test_netio_circuit_breaker.py`: State transitions and recovery
- `test_netio_screenshot_guard.py`: Debounce and single-flight patterns

## Thread Safety

All components are thread-safe:
- RateLimiter: RLock on token bucket
- CircuitBreaker: RLock on state transitions
- ScreenshotGuard: RLock on pending tracking, threading.Event for coordination
- AdaptiveSocket: Thread-safe command dispatch

## Performance Impact

- **Minimal overhead** for rate limiting (sub-microsecond per check)
- **Jitter-based cooldown** prevents thundering herd on recovery
- **Debounce/single-flight** reduces actual network calls significantly
- **No blocking** in normal (CLOSED, not rate-limited) case

## Limitations & Caveats

1. **Circuit breaker jitter** (±10%) is to prevent synchronized recovery storms
2. **Screenshot debounce** means requests are delayed by up to 100ms (configurable)
3. **Half-open probes** are serialized (1 at a time) to safely test recovery
4. **No cross-process coordination**: Each process has independent circuit state

## Integration with MGBAController

The netio module is intentionally **non-intrusive**:

- No modifications to `mgba_controller.py`
- No monkey-patching or global state
- Pure composition-based design
- Opt-in via simple wrapper instantiation
- Compatible with all MGBAController versions

## Future Enhancements

- Metrics export (Prometheus-compatible)
- Adaptive rate limiting based on response times
- Circuit breaker state persistence across restarts
- Per-command rate limiting (screenshot vs memory read)
