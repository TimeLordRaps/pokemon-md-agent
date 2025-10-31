"""Content API wrapper for You.com search and web content fetching.

Handles multi-URL batch fetching, budget management, and rate limiting for content retrieval.
"""

import asyncio
import json
import logging
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib
import os
from collections import deque

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


@dataclass
class LocalCache:
    """Local cache for fetched content."""
    cache_dir: Path = field(default_factory=lambda: Path.home() / '.cache' / 'pmd-red' / 'content')
    max_age_days: int = 7
    max_entries: int = 1000

    def __init__(self, cache_dir: Optional[Path] = None, max_age_days: int = 7, max_entries: int = 1000):
        self.cache_dir = cache_dir or (Path.home() / '.cache' / 'pmd-red' / 'content')
        self.cache_dir = Path(self.cache_dir)
        self.max_age_days = max_age_days
        self.max_entries = max_entries
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[Page]:
        """Get cached page if available and not expired."""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check if expired (per-entry expiry takes precedence over global max_age_days)
            expiry_time = data.get('expiry_time')
            if expiry_time and time.time() > expiry_time:
                cache_file.unlink()
                return None

            # Check global expiry if no per-entry expiry
            if not expiry_time:
                age_days = (time.time() - data['fetched_at']) / (24 * 3600)
                if age_days > self.max_age_days:
                    cache_file.unlink()
                    return None

            return Page(**{k: v for k, v in data.items() if k != 'expiry_time'})

        except Exception as e:
            logger.warning(f"Failed to read cache for {url}: {e}")
            return None

    def put(self, page_or_url, page=None, max_age_seconds=None):
        """Cache a page. Supports both put(page) and put(url, page, max_age_seconds=...) signatures."""
        if page is None:
            # Called as put(page)
            page = page_or_url
            expiry_time = None
        else:
            # Called as put(url, page, max_age_seconds=...)
            url = page_or_url
            # Create a Page object from url and page content
            if isinstance(page, str):
                # Assume page is content string
                page = Page(url=url, title="", content=page, format="markdown")
            expiry_time = time.time() + max_age_seconds if max_age_seconds else None

        # Now cache the page
        cache_key = self._get_cache_key(page.url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            data = {
                'url': page.url,
                'title': page.title,
                'content': page.content,
                'format': page.format,
                'fetched_at': page.fetched_at,
                'status_code': page.status_code,
                'error': page.error,
                'expiry_time': expiry_time
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            # Clean up old entries if we have too many
            self._cleanup_cache()

        except Exception as e:
            logger.warning(f"Failed to write cache for {page.url}: {e}")

    def _cleanup_cache(self):
        """Remove oldest entries if cache is too large."""
        cache_files = list(self.cache_dir.glob("*.json"))
        if len(cache_files) <= self.max_entries:
            return

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest files
        to_remove = len(cache_files) - self.max_entries
        for i in range(to_remove):
            try:
                cache_files[i].unlink()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            'entries': len(cache_files),
            'max_entries': self.max_entries,
            'cache_dir': str(self.cache_dir)
        }


@dataclass
class FetchQueue:
    """Queue for managing fetch requests."""
    max_concurrent: int = 5
    max_queue_size: int = 100

    queue: deque = field(default_factory=deque)
    active: set = field(default_factory=set)
    semaphore: asyncio.Semaphore = field(init=False)
    _acquired_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def enqueue(self, urls_or_item, priority: int = 0) -> bool:
        """Enqueue URLs or single item for fetching. Returns True if queued."""
        if isinstance(urls_or_item, str):
            # Single item
            items = [urls_or_item]
        else:
            # List of URLs
            items = urls_or_item

        if len(self.queue) + len(items) > self.max_queue_size:
            return False

        # Add with priority (higher priority = processed first)
        for item in items:
            if item not in self.active:
                self.queue.appendleft((item, priority))

        return True

    def dequeue(self) -> Optional[str]:
        """Dequeue next item to process."""
        if not self.queue:
            return None

        # Get highest priority item
        self.queue = deque(sorted(self.queue, key=lambda x: x[1], reverse=True))
        item, _ = self.queue.popleft()
        self.active.add(item)
        return item

    def complete(self, item: str):
        """Mark item as completed."""
        self.active.discard(item)

    def acquire(self) -> bool:
        """Try to acquire semaphore. Returns True if acquired."""
        if self._acquired_count < self.max_concurrent:
            self._acquired_count += 1
            return True
        return False

    def release(self):
        """Release semaphore."""
        if self._acquired_count > 0:
            self._acquired_count -= 1

    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)


class ContentAPI:
    """You.com Contents API wrapper with budget, cache, and queued fetching."""

    def __init__(self, api_key: Optional[str] = None, budget_tracker: Optional[BudgetTracker] = None, cache_dir: Optional[Path] = None, max_concurrent_fetches: int = 5):
        self.api_key = api_key or os.getenv('YOU_API_KEY') or os.getenv('YOUCOM_API_KEY')
        self.base_url = "https://api.you.com"

        # Budget tracking
        self.budget = budget_tracker or BudgetTracker()

        # Local cache
        self.cache = LocalCache(cache_dir=cache_dir) if cache_dir else LocalCache()

        # Fetch queue
        self.fetch_queue = FetchQueue(max_concurrent=max_concurrent_fetches)

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
            'cache_hits': 0,
            'errors': 0,
            'budget_exceeded': 0,
            'queued_requests': 0
        }

    async def fetch(self, urls: List[str], format: str = "markdown", priority: int = 0) -> List[Page]:
        """Fetch multiple URLs with caching and bulk fetching support."""
        if not urls:
            return []

        # Check cache first
        results = []
        uncached_urls = []

        for url in urls:
            cached = self.cache.get(url)
            if cached and cached.is_success():
                results.append(cached)
                self.stats['cache_hits'] += 1
            else:
                uncached_urls.append(url)

        if not uncached_urls:
            return results

        # Use bulk fetch for uncached URLs
        bulk_results = await self.fetch_bulk(uncached_urls, format)
        results.extend(bulk_results)

        return results

    async def _process_queue(self, format: str) -> List[Page]:
        """Process queued fetch requests."""
        results = []
        processed_urls = []

        while True:
            url = self.fetch_queue.dequeue()
            if not url:
                break

            try:
                async with self.fetch_queue.semaphore:
                    page = await self._fetch_single(url, format)
                    results.append(page)

                    # Cache successful results
                    if page.is_success():
                        self.cache.put(page)

            except Exception as e:
                logger.error(f"Failed to fetch queued URL {url}: {e}")
                results.append(Page(url, "", "", format, error=str(e)))

            finally:
                self.fetch_queue.complete(url)
                processed_urls.append(url)

        self.stats['queued_requests'] += len(processed_urls)
        return results

    async def fetch_bulk(self, urls: List[str], format: str = "markdown") -> List[Page]:
        """Fetch multiple URLs in bulk if provider allows. Counts as 1 budget call."""
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
            # Bulk fetch - provider allows up to 10 URLs per call
            batch_size = 10
            all_results = []

            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_results = await self._batch_fetch(batch_urls, format)
                all_results.extend(batch_results)

            self.stats['calls_made'] += 1
            self.stats['pages_fetched'] += len([r for r in all_results if r.is_success()])
            return all_results

        except Exception as e:
            logger.error(f"Bulk fetch failed: {e}")
            self.stats['errors'] += 1
            # Fallback to individual fetches
            logger.info("Falling back to individual fetches")
            individual_results = []
            for url in urls:
                try:
                    page = await self._fetch_single(url, format)
                    individual_results.append(page)
                    if page.is_success():
                        self.stats['pages_fetched'] += 1
                except Exception as fetch_error:
                    logger.error(f"Individual fetch failed for {url}: {fetch_error}")
                    individual_results.append(Page(url, "", "", format, error=str(fetch_error)))
            self.stats['calls_made'] += len(urls)  # Count each individual fetch as a call
            return individual_results

    async def _batch_fetch(self, urls: List[str], format: str) -> List[Page]:
        """Perform the actual batch fetch with concurrency limiting."""
        # You.com Contents API expects individual calls, but we batch them
        # In practice, we'd make parallel requests but count as one budget call
        # Respect concurrency limits using semaphore

        async def fetch_with_semaphore(url):
            async with self.fetch_queue.semaphore:
                return await self._fetch_single(url, format)

        tasks = [fetch_with_semaphore(url) for url in urls]
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

    async def fetch_guide(self, shallow_hits: int = 0) -> List[Page]:
        """Fetch bulk default pages for guide content.

        Args:
            shallow_hits: Number of shallow hits that triggered this fetch (>=3 required)
        """
        if shallow_hits < 3:
            logger.warning("Insufficient shallow hits for fetch_guide: %d < 3", shallow_hits)
            return []

        guide_urls = [
            "https://example.com/bulk-defaults",
            "https://example.com/current/index.json",
            "https://example.com/faq/index.html",
            "https://example.com/indexes/manifest.json",
            "https://example.com/trajectories/latest"
        ]
        return await self.fetch_bulk(guide_urls, format="markdown")

    async def search_old_memories(self, query: str, shallow_hits: int = 0) -> List[Page]:
        """Search for old memories using site-side indexes first.

        Args:
            query: The search query
            shallow_hits: Number of shallow hits that triggered this search (>=3 required)
        """
        if shallow_hits < 3:
            logger.warning("Insufficient shallow hits for search_old_memories: %d < 3", shallow_hits)
            return []

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