"""Tests for content API batching and budget management."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import tempfile

from src.dashboard.content_api import ContentAPI, BudgetTracker, Page


class TestBudgetTracker:
    """Test budget tracker functionality."""

    def test_initial_budget(self):
        """Test initial budget state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / 'budget.json'
            tracker = BudgetTracker(monthly_limit=1000, cache_file=cache_file)
            assert tracker.monthly_limit == 1000
            assert tracker.used_this_month == 0
            assert tracker.can_consume(1) is True
            assert tracker.remaining() == 1000

    def test_consume_budget(self):
        """Test budget consumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / 'budget.json'
            tracker = BudgetTracker(monthly_limit=1000, cache_file=cache_file)

            # Can consume initially
            assert tracker.consume(100) is True
            assert tracker.used_this_month == 100
            assert tracker.remaining() == 900

            # Can consume more
            assert tracker.consume(200) is True
            assert tracker.used_this_month == 300

            # Cannot consume beyond limit
            assert tracker.consume(800) is False
            assert tracker.used_this_month == 300  # Unchanged

    def test_budget_persistence(self):
        """Test budget persistence across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / 'budget.json'

            # First tracker
            tracker1 = BudgetTracker(monthly_limit=1000, cache_file=cache_file)
            tracker1.consume(50)

            # Second tracker should load the same state
            tracker2 = BudgetTracker(monthly_limit=1000, cache_file=cache_file)
            assert tracker2.used_this_month == 50


class TestContentAPI:
    """Test ContentAPI functionality."""

    @pytest.fixture
    def api(self):
        """Create test API instance."""
        # Use a custom budget tracker for testing
        import tempfile
        tmpdir = tempfile.mkdtemp()
        cache_file = Path(tmpdir) / 'budget.json'
        budget = BudgetTracker(monthly_limit=1000, cache_file=cache_file)
        api = ContentAPI(budget_tracker=budget)
        yield api
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_state(self, api):
        """Test initial API state."""
        stats = api.get_stats()
        assert stats['calls_made'] == 0
        assert stats['pages_fetched'] == 0
        assert stats['errors'] == 0

        budget = api.get_budget_status()
        assert budget['monthly_limit'] == 1000
        assert budget['used_this_month'] == 0

    @pytest.mark.asyncio
    async def test_batch_fetch(self, api):
        """Test multi-URL batch fetch."""
        urls = ["http://example.com/page1", "http://example.com/page2"]

        # Mock the internal fetch method
        with patch.object(api, '_batch_fetch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                Page(urls[0], "Title 1", "Content 1", "markdown"),
                Page(urls[1], "Title 2", "Content 2", "markdown")
            ]

            results = await api.fetch(urls)

            # Should call batch fetch once
            mock_batch.assert_called_once_with(urls, "markdown")

            # Should consume 1 budget call
            assert api.get_budget_status()['used_this_month'] == 1

            # Should return results
            assert len(results) == 2
            assert all(r.is_success() for r in results)

    @pytest.mark.asyncio
    async def test_budget_exceeded(self, api):
        """Test behavior when budget is exceeded."""
        # Exhaust budget
        api.budget.used_this_month = 1000

        urls = ["http://example.com/page1"]
        results = await api.fetch(urls)

        # Should return error pages
        assert len(results) == 1
        assert not results[0].is_success()
        assert "Budget exceeded" in results[0].error

        # Stats should reflect budget exceeded
        assert api.get_stats()['budget_exceeded'] == 1

    def test_gate_token_logic(self, api):
        """Test gate token cool-down logic."""
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
        """Test gate token reset for testing."""
        api.reset_gate_tokens()
        gate_token = "test_gate"

        # Should allow two calls after reset
        assert api.consume_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is True
        assert api.consume_gate_token(gate_token) is False

    @pytest.mark.asyncio
    async def test_fetch_guide(self, api):
        """Test fetch_guide method."""
        with patch.object(api, 'fetch', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [Page("http://example.com/guide", "Guide", "Content", "markdown")]

            results = await api.fetch_guide()

            mock_fetch.assert_called_once()
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_old_memories(self, api):
        """Test search_old_memories method."""
        query = "test query"

        with patch.object(api, '_search_site_indexes', new_callable=AsyncMock) as mock_search:
            with patch.object(api, 'fetch', new_callable=AsyncMock) as mock_fetch:
                # Site search returns empty (fallback)
                mock_search.return_value = []
                mock_fetch.return_value = [Page("http://example.com/search", "Results", "Content", "markdown")]

                results = await api.search_old_memories(query)

                mock_search.assert_called_once_with(query)
                mock_fetch.assert_called_once()
                assert len(results) == 1