"""Tests for token-bucket rate limiter."""

import pytest
import time
from src.environment.netio.adaptive_socket import RateLimiter


class TestRateLimiter:
    """Test token-bucket rate limiting."""

    def test_rate_limiter_acquisition(self):
        """Test acquiring tokens up to max burst."""
        limiter = RateLimiter(max_rps=10.0, max_burst=5)

        # Should acquire up to max_burst tokens
        assert limiter.acquire(1.0)
        assert limiter.acquire(1.0)
        assert limiter.acquire(1.0)
        assert limiter.acquire(1.0)
        assert limiter.acquire(1.0)

        # 6th token should fail (exceeded burst)
        assert not limiter.acquire(1.0)

    def test_rate_limiter_refill(self):
        """Test token refill over time."""
        limiter = RateLimiter(max_rps=10.0, max_burst=5)

        # Drain all tokens
        for _ in range(5):
            assert limiter.acquire(1.0)

        # Should be empty
        assert not limiter.acquire(1.0)

        # Wait for some tokens to refill
        time.sleep(0.15)  # At 10 RPS, should get ~1.5 tokens

        # Should be able to get 1 token
        assert limiter.acquire(1.0)

    def test_rate_limiter_wait_if_needed(self):
        """Test blocking wait for tokens."""
        limiter = RateLimiter(max_rps=10.0, max_burst=2)

        # Drain tokens
        assert limiter.acquire(1.0)
        assert limiter.acquire(1.0)
        assert not limiter.acquire(1.0)

        # This should block and then succeed
        start = time.monotonic()
        limiter.wait_if_needed(1.0)
        elapsed = time.monotonic() - start

        # Should have waited ~0.1 seconds for 1 token at 10 RPS
        assert elapsed >= 0.08  # Some tolerance for timing variance

    def test_rate_limiter_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(max_rps=10.0, max_burst=10)

        # Acquire 5 tokens
        assert limiter.acquire(5.0)

        # Should have 5 left
        assert limiter.acquire(5.0)

        # Should be empty now
        assert not limiter.acquire(1.0)

    def test_rate_limiter_burst_handling(self):
        """Test burst behavior when spammed with requests."""
        limiter = RateLimiter(max_rps=5.0, max_burst=10)

        # Rapid requests should succeed up to burst
        success_count = 0
        for _ in range(20):
            if limiter.acquire(1.0):
                success_count += 1
            else:
                break

        # Should have gotten ~10 (burst capacity)
        assert success_count >= 9  # Some tolerance
        assert success_count <= 11

    def test_rate_limiter_high_rps(self):
        """Test high RPS configuration."""
        limiter = RateLimiter(max_rps=100.0, max_burst=50)

        # Should be able to get burst quickly
        for i in range(50):
            assert limiter.acquire(1.0), f"Failed at token {i}"

        # Next should fail
        assert not limiter.acquire(1.0)

    @pytest.mark.timeout(5)
    def test_rate_limiter_acceptance_criteria_burst_under_30(self):
        """Test acceptance criteria: spammed with 50 screenshot calls.

        Only ≤30 should succeed immediately (burst capacity) when IO_MAX_RPS=15.
        """
        limiter = RateLimiter(max_rps=15.0)  # 15 RPS = ~30 burst tokens

        # Simulate 50 rapid requests (no waiting, just immediate acquire)
        success_count = 0
        failed_count = 0

        for i in range(50):
            if limiter.acquire(1.0):
                success_count += 1
            else:
                failed_count += 1

        # With 15 RPS and burst capacity of 30 tokens,
        # rapid fire should get ~30, rest rejected
        assert success_count <= 31  # Allow tiny margin for timing
        assert failed_count >= 19

    @pytest.mark.timeout(5)
    def test_rate_limiter_sustained_rate_15_rps(self):
        """Test sustained rate over time at 15 RPS."""
        limiter = RateLimiter(max_rps=15.0)
        
        # Skip the burst by draining tokens first
        for _ in range(30):
            limiter.acquire(1.0)
        
        # Now wait for refill and measure rate
        start = time.monotonic()
        requests_made = 0
        duration = 2.0  # Measure for 2 seconds
        
        while time.monotonic() - start < duration:
            limiter.wait_if_needed(1.0)
            requests_made += 1
        
        elapsed = time.monotonic() - start
        rate = requests_made / elapsed
        
        # Should be close to 15 RPS
        assert 14.0 <= rate <= 16.0, f"Rate was {rate:.2f} RPS, expected ~15 RPS"
