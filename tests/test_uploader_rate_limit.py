"""Tests for dashboard uploader rate limiting and batching."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from src.dashboard.uploader import (
    DashboardUploader, DashboardConfig, UploadMode,
    RateLimiter, FileBatch
)


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_initial_state(self):
        """Test rate limiter initial state."""
        limiter = RateLimiter(capacity=10, refill_rate=1.0)

        assert limiter.capacity == 10
        assert limiter.refill_rate == 1.0
        assert limiter.tokens == 10  # Starts full

    def test_consume_tokens(self):
        """Test token consumption."""
        limiter = RateLimiter(capacity=10, refill_rate=1.0)

        # Can consume available tokens
        assert limiter.consume(5) is True
        assert limiter.tokens == 5

        # Can consume remaining
        assert limiter.consume(5) is True
        assert limiter.tokens == 0

        # Cannot consume more than available
        assert limiter.consume(1) is False
        assert limiter.tokens == 0

    def test_refill_over_time(self):
        """Test token refill over time."""
        limiter = RateLimiter(capacity=10, refill_rate=2.0)  # 2 tokens per second

        # Consume all tokens
        limiter.consume(10)
        assert limiter.tokens == 0

        # Simulate time passing
        limiter.last_refill -= 2.5  # 2.5 seconds ago

        # Should have refilled 5 tokens (2.5 * 2)
        assert limiter.consume(5) is True
        assert limiter.tokens == 0  # Exactly 5 available, consumed 5

    def test_time_until_tokens(self):
        """Test time calculation for token availability."""
        limiter = RateLimiter(capacity=10, refill_rate=1.0)

        limiter.consume(10)  # Empty
        wait_time = limiter.time_until_tokens(5)

        # Should take 5 seconds to refill 5 tokens
        assert abs(wait_time - 5.0) < 0.1


class TestFileBatch:
    """Test file batch functionality."""

    def test_initial_state(self):
        """Test batch initial state."""
        batch = FileBatch()
        assert batch.is_empty()
        assert batch.total_bytes == 0
        assert batch.age_seconds() < 1.0

    def test_add_file(self):
        """Test adding files to batch."""
        batch = FileBatch()

        # Add small file
        success = batch.add_file("test1.txt", b"hello")
        assert success is True
        assert batch.total_bytes == 5
        assert len(batch.files) == 1

        # Add another file
        success = batch.add_file("test2.txt", b"world!")
        assert success is True
        assert batch.total_bytes == 11
        assert len(batch.files) == 2

    def test_batch_size_limit(self):
        """Test batch size limits."""
        batch = FileBatch()

        # Try to add file larger than 8MB limit
        large_content = b"x" * (8 * 1024 * 1024 + 1)
        success = batch.add_file("large.txt", large_content)
        assert success is False
        assert batch.is_empty()

    def test_age_calculation(self):
        """Test batch age calculation."""
        batch = FileBatch()
        initial_age = batch.age_seconds()

        # Simulate time passing
        batch.created_at -= 10
        age = batch.age_seconds()

        assert abs(age - initial_age - 10) < 1.0


class TestDashboardUploader:
    """Test dashboard uploader functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DashboardConfig(
            enabled=True,
            branch="pages",
            site_root="docs",
            flush_seconds=30.0,
            max_batch_bytes=8 * 1024 * 1024,
            max_files_per_minute=30
        )

    @pytest.fixture
    def uploader(self, config):
        """Create test uploader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            uploader = DashboardUploader(config, cache_dir)
            # Force no-op mode for testing
            uploader.upload_mode = UploadMode.NO_OP
            yield uploader

    def test_initial_state(self, uploader):
        """Test uploader initial state."""
        assert uploader.upload_mode == UploadMode.NO_OP
        assert uploader.current_batch.is_empty()
        assert len(uploader.stats) > 0

    @pytest.mark.asyncio
    async def test_queue_file(self, uploader):
        """Test file queuing."""
        test_content = b"test content"
        success = await uploader.queue_file("test.txt", test_content)

        assert success is True
        assert not uploader.current_batch.is_empty()
        assert uploader.current_batch.total_bytes == len(test_content)

    @pytest.mark.asyncio
    async def test_batch_flush_on_size(self, uploader):
        """Test automatic batch flush when size limit reached."""
        # Set very small batch limit
        uploader.config.max_batch_bytes = 10

        # Add file that exceeds limit
        large_content = b"x" * 15
        success = await uploader.queue_file("large.txt", large_content)

        assert success is True
        # Should have flushed and started new batch
        assert uploader.current_batch.is_empty()

    @pytest.mark.asyncio
    async def test_batch_flush_on_time(self, uploader):
        """Test automatic batch flush when time limit reached."""
        # Set very short time limit
        uploader.config.flush_seconds = 0.1

        # Add file
        await uploader.queue_file("test.txt", b"content")

        # Simulate batch being old
        uploader.current_batch.created_at -= 1.0

        # Next queue should trigger flush
        await uploader.queue_file("test2.txt", b"content2")
        assert uploader.current_batch.is_empty()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, uploader):
        """Test file upload rate limiting."""
        # Exhaust file rate limiter
        uploader.file_limiter.tokens = 0

        # Try to queue file
        success = await uploader.queue_file("test.txt", b"content")

        assert success is True  # Still succeeds but should wait
        # In real scenario, this would wait, but in test we can't easily verify

    @pytest.mark.asyncio
    async def test_build_budget_limiting(self, uploader):
        """Test build budget rate limiting."""
        # Exhaust build limiter
        uploader.build_limiter.tokens = 0

        # Mock flush to test build limiting
        with patch.object(uploader, '_flush_batch', new_callable=AsyncMock) as mock_flush:
            await uploader.flush()
            # Should still call flush but with delay
            mock_flush.assert_called_once()

    def test_stats_tracking(self, uploader):
        """Test statistics tracking."""
        initial_stats = uploader.get_stats().copy()

        # Simulate some activity
        uploader.stats['files_uploaded'] = 5
        uploader.stats['bytes_uploaded'] = 1000

        stats = uploader.get_stats()
        assert stats['files_uploaded'] == 5
        assert stats['bytes_uploaded'] == 1000

    @pytest.mark.asyncio
    async def test_flush_cleanup(self, uploader):
        """Test flush and cleanup."""
        # Add some files
        await uploader.queue_file("test1.txt", b"content1")
        await uploader.queue_file("test2.txt", b"content2")

        assert not uploader.current_batch.is_empty()

        # Force flush
        await uploader.flush()

        assert uploader.current_batch.is_empty()
        assert uploader.stats['batches_flushed'] == 1

    def test_upload_mode_detection(self, config):
        """Test upload mode detection logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Test disabled
            config.enabled = False
            uploader = DashboardUploader(config, cache_dir)
            assert uploader.upload_mode == UploadMode.NO_OP

            # Test git repo detection (mock)
            config.enabled = True
            with patch('src.dashboard.uploader.DashboardUploader._is_git_repo', return_value=True):
                uploader = DashboardUploader(config, cache_dir)
                assert uploader.upload_mode == UploadMode.GIT_PUSH

            # Test API mode
            config.github_token = "token"
            config.github_repo = "user/repo"
            with patch('src.dashboard.uploader.DashboardUploader._is_git_repo', return_value=False):
                uploader = DashboardUploader(config, cache_dir)
                assert uploader.upload_mode == UploadMode.GITHUB_API