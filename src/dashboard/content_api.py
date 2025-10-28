"""Content API wrapper for You.com search and web content fetching.

Handles multi-URL batch fetching, budget management, and rate limiting for content retrieval.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """A fetched web page."""
    url: str
    title: str
    content: str
    format: str  # 'markdown' or 'html'
    fetched_at: float = field(default_factory=time.time)
    status_code: int = 200
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.status_code == 200 and self.error is None


@dataclass
class BudgetTracker:
    """Tracks API usage budget."""
    monthly_limit: int = 1000
    cache_file: Path = field(default_factory=lambda: Path.home() / '.cache' / 'pmd-red' / 'youcom_budget.json')

    used_this_month: int = field(init=False, default=0)
    month_start: float = field(init=False, default_factory=time.time)

    def __post_init__(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Load budget state from cache file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.used_this_month = data.get('used_this_month', 0)
                    self.month_start = data.get('month_start', time.time())
        except Exception as e:
            logger.warning(f"Failed to load budget cache: {e}")

    def _save_state(self):
        """Save budget state to cache file."""
        try:
            data = {
                'used_this_month': self.used_this_month,
                'month_start': self.month_start
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save budget cache: {e}")

    def _reset_if_new_month(self):
        """Reset counter if it's a new month."""
        current_month = time.gmtime(time.time()).tm_mon
        start_month = time.gmtime(self.month_start).tm_mon
        if current_month != start_month:
            logger.info("New month detected, resetting budget counter")
            self.used_this_month = 0
            self.month_start = time.time()
            self._save_state()

    def can_consume(self, amount: int = 1) -> bool:
        """Check if we can consume the given amount."""
        self._reset_if_new_month()
        return self.used_this_month + amount <= self.monthly_limit

    def consume(self, amount: int = 1) -> bool:
        """Consume budget. Returns True if successful."""
        if not self.can_consume(amount):
            return False
        self.used_this_month += amount
        self._save_state()
        return True

    def remaining(self) -> int:
        """Get remaining budget for this month."""
        self._reset_if_new_month()
        return max(0, self.monthly_limit - self.used_this_month)


class ContentAPI:
    """You.com Contents API wrapper with budget and rate limiting."""

    def __init__(self, api_key: Optional[str] = None, budget_tracker: Optional[BudgetTracker] = None):
        self.api_key = api_key or os.getenv('YOUCOM_API_KEY')
        self.base_url = "https://api.you.com"

        # Budget tracking
        self.budget = budget_tracker or BudgetTracker()

        # HTTP session with retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Per-gate cool-down tracking
        self.gate_tokens: Dict[str, List[float]] = {}  # gate_token -> list of timestamps
        self.max_calls_per_gate = 2

        # Stats
        self.stats = {
            'calls_made': 0,
            'pages_fetched': 0,
            'errors': 0,
            'budget_exceeded': 0
        }

    async def fetch(self, urls: List[str], format: str = "markdown") -> List[Page]:
        """Fetch multiple URLs in a batch. Counts as 1 budget call."""
        if not urls:
            return []

        if not self.budget.can_consume(1):
            logger.warning("Monthly budget exceeded")
            self.stats['budget_exceeded'] += 1
            return [Page(url, "", "", format, error="Budget exceeded") for url in urls]

        # Consume budget
        if not self.budget.consume(1):
            return [Page(url, "", "", format, error="Budget exceeded") for url in urls]

        try:
            # Batch fetch
            results = await self._batch_fetch(urls, format)
            self.stats['calls_made'] += 1
            self.stats['pages_fetched'] += len([r for r in results if r.is_success()])
            return results

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            self.stats['errors'] += 1
            return [Page(url, "", "", format, error=str(e)) for url in urls]

    async def _batch_fetch(self, urls: List[str], format: str) -> List[Page]:
        """Perform the actual batch fetch."""
        # You.com Contents API expects individual calls, but we batch them
        # In practice, we'd make parallel requests but count as one budget call
        tasks = []
        for url in urls:
            tasks.append(self._fetch_single(url, format))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pages.append(Page(urls[i], "", "", format, error=str(result)))
            else:
                pages.append(result)

        return pages

    async def _fetch_single(self, url: str, format: str) -> Page:
        """Fetch a single URL."""
        # Simulate API call - in real implementation, this would call You.com API
        # For now, return mock data
        await asyncio.sleep(0.1)  # Simulate network delay

        return Page(
            url=url,
            title=f"Mock Title for {url}",
            content=f"# Mock Content\n\nThis is mock content for {url}",
            format=format
        )

    def check_gate_token(self, gate_token: str) -> bool:
        """Check if gate token allows another call (max 2 per gate)."""
        now = time.time()
        # Count calls for this specific gate token in the last hour
        recent_calls = [t for t in self.gate_tokens.get(gate_token, []) if now - t < 3600]

        if len(recent_calls) >= self.max_calls_per_gate:
            return False

        return True

    def consume_gate_token(self, gate_token: str) -> bool:
        """Consume a gate token. Returns True if allowed."""
        if not self.check_gate_token(gate_token):
            return False

        if gate_token not in self.gate_tokens:
            self.gate_tokens[gate_token] = []
        self.gate_tokens[gate_token].append(time.time())
        return True

    async def fetch_guide(self) -> List[Page]:
        """Fetch bulk default pages for guide content."""
        guide_urls = [
            "https://example.com/bulk-defaults",
            "https://example.com/current/index.json",
            "https://example.com/faq/index.html",
            "https://example.com/indexes/manifest.json",
            "https://example.com/trajectories/latest"
        ]
        return await self.fetch(guide_urls, format="markdown")

    async def search_old_memories(self, query: str) -> List[Page]:
        """Search for old memories using site-side indexes first."""
        # First try site-side search
        site_results = await self._search_site_indexes(query)
        if site_results:
            return site_results

        # Fallback to targeted pages
        memory_urls = [
            f"https://example.com/indexes/search?q={query}",
            "https://example.com/memories/recent"
        ]
        return await self.fetch(memory_urls, format="markdown")

    async def _search_site_indexes(self, query: str) -> List[Page]:
        """Search local site indexes (would be client-side in real implementation)."""
        # Mock implementation - in reality this would search FAISS indexes
        await asyncio.sleep(0.05)
        return []  # Return empty to trigger fallback

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            'monthly_limit': self.budget.monthly_limit,
            'used_this_month': self.budget.used_this_month,
            'remaining': self.budget.remaining(),
            'reset_days': self._days_until_reset()
        }

    def _days_until_reset(self) -> int:
        """Days until monthly budget resets."""
        import calendar
        now = time.gmtime(time.time())
        _, last_day = calendar.monthrange(now.tm_year, now.tm_mon)
        days_left = last_day - now.tm_mday
        return max(0, days_left)

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return dict(self.stats)

    def reset_gate_tokens(self):
        """Reset all gate tokens (for testing)."""
        self.gate_tokens.clear()