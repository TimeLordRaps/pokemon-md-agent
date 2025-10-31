"""Test content API cooldown arithmetic and bulk/queue fallbacks."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import time

from src.dashboard.content_api import ContentAPI, BudgetTracker, LocalCache, FetchQueue, Page


class TestCooldownArithmetic:
    """Test cooldown gate token arithmetic."""

    @pytest.fixture
    def api(self):
        """Create test API instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / 'budget.json'
            budget = BudgetTracker(monthly_limit=1000, cache_file=cache_file)
            api = ContentAPI(budget_tracker=budget)
            yield api

    def test_gate_token_initial_state(self, api):
        """Test gate token initial state allows two calls."""
        gate_token = "test_gate"

        # Initially should allow
        assert api.check_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is True

        # Should allow one more
        assert api.check_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is True

        # Should deny third call
        assert api.check_gate_token(gate_token) is False
        assert api.consume_gate_token(gate_token) is False

    def test_gate_token_reset(self, api):
        """Test gate token reset restores allowance."""
        gate_token = "test_gate"

        # Exhaust tokens
        api.consume_gate_token(gate_token)
        api.consume_gate_token(gate_token)
        assert api.check_gate_token(gate_token) is False

        # Reset should allow two calls again
        api.reset_gate_tokens()
        assert api.check_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is True
        assert api.check_gate_token(gate_token) is False

    def test_multiple_gate_tokens_independent(self, api):
        """Test multiple gate tokens are independent."""
        gate1 = "gate1"
        gate2 = "gate2"

        # Exhaust gate1
        api.consume_gate_token(gate1)
        api.consume_gate_token(gate1)
        assert api.check_gate_token(gate1) is False

        # gate2 should still allow calls
        assert api.check_gate_token(gate2) is True
        assert api.consume_gate_token(gate2) is True
        assert api.consume_gate_token(gate2) is True
        assert api.check_gate_token(gate2) is False

    def test_gate_token_cooldown_timing(self, api):
        """Test gate token cooldown timing logic."""
        gate_token = "test_gate"

        # Exhaust tokens
        api.consume_gate_token(gate_token)
        api.consume_gate_token(gate_token)

        # Should be denied immediately
        assert api.check_gate_token(gate_token) is False

        # After reset, should allow again
        api.reset_gate_tokens()
        assert api.check_gate_token(gate_token) is True


class TestBulkQueueFallbacks:
    """Test bulk fetch and queue fallback mechanisms."""

    @pytest.fixture
    def api(self):
        """Create test API with cache and queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'cache'
            cache_dir.mkdir()

            budget_file = Path(tmpdir) / 'budget.json'
            budget = BudgetTracker(monthly_limit=1000, cache_file=budget_file)

            api = ContentAPI(
                budget_tracker=budget,
                cache_dir=cache_dir,
                max_concurrent_fetches=3
            )
            yield api

    @pytest.mark.asyncio
    async def test_bulk_fetch_success(self, api):
        """Test successful bulk fetch of multiple URLs."""
        urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]

        # Mock the internal fetch method
        with patch.object(api, '_batch_fetch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                Page(urls[0], "Title 1", "Content 1", "markdown"),
                Page(urls[1], "Title 2", "Content 2", "markdown"),
                Page(urls[2], "Title 3", "Content 3", "markdown")
            ]

            results = await api.fetch(urls)

            # Should call batch fetch once
            mock_batch.assert_called_once_with(urls, "markdown")

            # Should consume 1 budget call
            assert api.get_budget_status()['used_this_month'] == 1

            # Should return all results
            assert len(results) == 3
            assert all(r.is_success() for r in results)

    @pytest.mark.asyncio
    async def test_fetch_guide_insufficient_shallow_hits(self, api):
        """Test fetch_guide rejects calls with insufficient shallow_hits."""
        # Should return empty list for shallow_hits < 3
        results = await api.fetch_guide(shallow_hits=2)
        assert results == []

        results = await api.fetch_guide(shallow_hits=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_guide_sufficient_shallow_hits(self, api):
        """Test fetch_guide accepts calls with sufficient shallow_hits."""
        # Mock the fetch_bulk method
        with patch.object(api, 'fetch_bulk', new_callable=AsyncMock) as mock_fetch_bulk:
            mock_fetch_bulk.return_value = [
                Page("https://example.com/guide1", "Guide 1", "Content 1", "markdown")
            ]

            results = await api.fetch_guide(shallow_hits=3)

            # Should call fetch_bulk
            mock_fetch_bulk.assert_called_once()
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_old_memories_insufficient_shallow_hits(self, api):
        """Test search_old_memories rejects calls with insufficient shallow_hits."""
        # Should return empty list for shallow_hits < 3
        results = await api.search_old_memories("test query", shallow_hits=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_old_memories_sufficient_shallow_hits(self, api):
        """Test search_old_memories accepts calls with sufficient shallow_hits."""
        # Mock site-side search to return results
        with patch.object(api, '_search_site_indexes', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                Page("https://example.com/memory1", "Memory 1", "Content 1", "markdown")
            ]

            results = await api.search_old_memories("test query", shallow_hits=5)

            # Should call site search and return results
            mock_search.assert_called_once_with("test query")
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_bulk_fetch_partial_failure(self, api):
        """Test bulk fetch with some failures."""
        urls = ["http://example.com/page1", "http://example.com/page2"]

        # Mock batch fetch with one failure
        with patch.object(api, '_batch_fetch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                Page(urls[0], "Title 1", "Content 1", "markdown"),
                Page(urls[1], "", "", "markdown", error="Network timeout")
            ]

            results = await api.fetch(urls)

            # Should return mixed results
            assert len(results) == 2
            assert results[0].is_success()
            assert not results[1].is_success()
            assert "timeout" in results[1].error.lower()

    @pytest.mark.asyncio
    async def test_queue_fallback_on_bulk_failure(self, api):
        """Test fallback to queued individual fetches when bulk fails."""
        urls = ["http://example.com/page1", "http://example.com/page2"]

        # Mock batch fetch to fail
        with patch.object(api, '_batch_fetch', new_callable=AsyncMock) as mock_batch:
            mock_batch.side_effect = Exception("Bulk fetch failed")

            # Mock individual fetch to succeed
            with patch.object(api, '_fetch_single', new_callable=AsyncMock) as mock_single:
                mock_single.side_effect = [
                    Page(urls[0], "Title 1", "Content 1", "markdown"),
                    Page(urls[1], "Title 2", "Content 2", "markdown")
                ]

                results = await api.fetch(urls)

                # Should fallback to individual fetches
                assert mock_single.call_count == 2
                assert len(results) == 2
                assert all(r.is_success() for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforcement(self, api):
        """Test concurrent fetch limit enforcement."""
        urls = ["http://example.com/page1", "http://example.com/page2",
                "http://example.com/page3", "http://example.com/page4"]

        # Mock individual fetches with delays to test concurrency
        async def delayed_fetch(url, format):
            await asyncio.sleep(0.1)  # Small delay
            return Page(url, f"Title for {url}", f"Content for {url}", format)

        with patch.object(api, '_fetch_single', side_effect=delayed_fetch):
            start_time = time.time()
            results = await api.fetch(urls)
            elapsed = time.time() - start_time

            # Should respect concurrent limit (max_concurrent_fetches=3)
            # With 4 URLs and concurrency limit of 3, should take longer than 0.1s but less than 0.3s
            assert 0.1 < elapsed < 0.3
            assert len(results) == 4
            assert all(r.is_success() for r in results)

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_fetch(self, api):
        """Test cache hits avoid network fetches."""
        url = "http://example.com/cached_page"
        cached_page = Page(url, "Cached Title", "Cached Content", "markdown")

        # Pre-populate cache
        api.cache.put(url, cached_page)

        results = await api.fetch([url])

        # Should return cached result without network call
        assert len(results) == 1
        assert results[0].is_success()
        assert results[0].title == "Cached Title"
        assert results[0].content == "Cached Content"

        # Should not consume budget for cache hits
        assert api.get_budget_status()['used_this_month'] == 0


class TestLocalCache:
    """Test local cache functionality."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create test cache."""
        return LocalCache(cache_dir=tmp_path / 'cache', max_entries=10)

    def test_cache_put_get(self, cache):
        """Test basic cache put and get operations."""
        url = "http://example.com/test"
        page = Page(url, "Test Title", "Test Content", "markdown")

        # Put in cache
        cache.put(url, page)

        # Get from cache
        cached = cache.get(url)
        assert cached is not None
        assert cached.title == "Test Title"
        assert cached.content == "Test Content"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("http://example.com/missing")
        assert result is None

    def test_cache_expiry(self, cache):
        """Test cache expiry removes old entries."""
        url = "http://example.com/expire"
        page = Page(url, "Expire Title", "Expire Content", "markdown")

        # Put with very short expiry
        cache.put(url, page, max_age_seconds=0.1)

        # Should be available immediately
        assert cache.get(url) is not None

        # Wait for expiry
        time.sleep(0.2)

        # Should be expired
        assert cache.get(url) is None

    def test_cache_size_limit(self, cache):
        """Test cache respects size limits."""
        # Fill cache beyond limit
        for i in range(15):  # More than max_entries=10
            url = f"http://example.com/page{i}"
            page = Page(url, f"Title {i}", f"Content {i}", "markdown")
            cache.put(url, page)

        # Should have cleaned up old entries
        stats = cache.get_stats()
        assert stats['entries'] <= 10


class TestFetchQueue:
    """Test fetch queue functionality."""

    @pytest.fixture
    def queue(self):
        """Create test queue."""
        return FetchQueue(max_concurrent=3)

    def test_queue_priority_ordering(self, queue):
        """Test queue maintains priority ordering."""
        # Add items with different priorities
        queue.enqueue("low_priority", priority=1)
        queue.enqueue("high_priority", priority=3)
        queue.enqueue("medium_priority", priority=2)

        # Should dequeue in priority order (highest first)
        assert queue.dequeue() == "high_priority"
        assert queue.dequeue() == "medium_priority"
        assert queue.dequeue() == "low_priority"

    def test_queue_semaphore_limits(self, queue):
        """Test semaphore limits concurrent operations."""
        # Initially should allow up to max_concurrent
        assert queue.acquire() is True
        assert queue.acquire() is True
        assert queue.acquire() is True

        # Should deny additional concurrent operations
        assert queue.acquire() is False

        # Release one, should allow another
        queue.release()
        assert queue.acquire() is True

        # Still at limit
        assert queue.acquire() is False

    def test_queue_empty_dequeue(self, queue):
        """Test dequeue on empty queue returns None."""
        assert queue.dequeue() is None

    @pytest.mark.asyncio
    async def test_queue_async_operations(self, queue):
        """Test async queue operations."""
        # Test concurrent enqueue/dequeue
        tasks = []

        async def producer():
            for i in range(5):
                queue.enqueue(f"item_{i}", priority=i)
                await asyncio.sleep(0.01)

        async def consumer():
            items = []
            while len(items) < 5:
                item = queue.dequeue()
                if item:
                    items.append(item)
                await asyncio.sleep(0.005)
            return items

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await producer_task
        items = await consumer_task

        # Should have consumed all items (order may vary due to concurrency)
        assert len(items) == 5
        assert set(items) == {"item_0", "item_1", "item_2", "item_3", "item_4"}