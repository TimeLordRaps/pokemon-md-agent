"""Time budgets and rate limiting configuration for agent operations.

Environment variables are read once at import time to establish operational limits
for routing, queuing, socket operations, and resource management.
"""

import os
from typing import Final


def _get_model_aware_batch_size() -> int:
    """Get batch size based on model parameter count.

    Returns:
        Batch size: 8 for 2B models, 4 for 4B models, 2 for 8B models
    """
    model_size_b = int(os.environ.get('MODEL_SIZE_B', 4))
    if model_size_b == 2:
        return 8
    elif model_size_b == 8:
        return 2
    else:  # Default to 4B behavior
        return 4


# Router timing budgets
ROUTER_MAX_WALL_S: Final[int] = int(os.environ.get('ROUTER_MAX_WALL_S', 20))
ROUTER_FLUSH_TICK_MS: Final[int] = int(os.environ.get('ROUTER_FLUSH_TICK_MS', 50))

# Model batching configuration (model-aware defaults)
BATCH_MAX_SIZE: Final[int] = _get_model_aware_batch_size()

# Socket operation timeouts
SOCKET_OP_TIMEOUT_S: Final[int] = int(os.environ.get('SOCKET_OP_TIMEOUT_S', 5))

# Rate limiting for vision operations
SCREENSHOT_RATE_LIMIT_HZ: Final[int] = int(os.environ.get('SCREENSHOT_RATE_LIMIT_HZ', 20))

# Caching limits
PROMPT_CACHE_SIZE: Final[int] = int(os.environ.get('PROMPT_CACHE_SIZE', 5))

# Per-stage budgets for deadline-aware scheduling (seconds)
TOKENIZE_BUDGET_S: Final[float] = float(os.environ.get('TOKENIZE_BUDGET_S', 0.5))
FORWARD_BUDGET_S: Final[float] = float(os.environ.get('FORWARD_BUDGET_S', 2.0))
DECODE_BUDGET_S: Final[float] = float(os.environ.get('DECODE_BUDGET_S', 1.0))