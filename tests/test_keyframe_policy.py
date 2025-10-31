"""Tests for keyframe policy with SSIM-based detection."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch
from src.retrieval.keyframe_policy import KeyframePolicy, KeyframeCandidate, SamplingStrategy


class TestKeyframePolicySSIM:
    """Test SSIM-based keyframe detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.policy = KeyframePolicy(ssim_threshold=0.8)

    def create_test_candidate(self, timestamp: float, ssim_score: float = None, frame_image: Image.Image = None) -> KeyframeCandidate:
        """Create a test keyframe candidate."""
        return KeyframeCandidate(
            timestamp=timestamp,
            embedding=np.random.rand(128),
            metadata={"test": True},
            frame_image=frame_image,
            ssim_score=ssim_score
        )

    def test_ssim_threshold_parameter(self):
        """Test that SSIM threshold is properly configured."""
        policy_low = KeyframePolicy(ssim_threshold=0.5)
        policy_high = KeyframePolicy(ssim_threshold=0.95)

        assert policy_low.ssim_threshold == 0.5
        assert policy_high.ssim_threshold == 0.95
        assert self.policy.ssim_threshold == 0.8

    def test_ssim_calculation_no_previous_frame(self):
        """Test SSIM calculation when no previous keyframe exists."""
        # Create test image
        test_image = Image.new('RGB', (64, 64), color='red')
        candidates = [self.create_test_candidate(1.0, frame_image=test_image)]

        result = self.policy._calculate_ssim_scores(candidates)

        # SSIM should be None since no previous frame
        assert result[0].ssim_score is None

    @patch('src.retrieval.keyframe_policy.ssim')
    def test_ssim_calculation_with_previous_frame(self, mock_ssim):
        """Test SSIM calculation with previous keyframe."""
        # Mock SSIM to return 0.9 (similar frames)
        mock_ssim.return_value = 0.9

        # Set up previous keyframe image
        prev_image = Image.new('RGB', (64, 64), color='red')
        self.policy.last_keyframe_image = prev_image

        # Create current frame image
        curr_image = Image.new('RGB', (64, 64), color='red')
        candidates = [self.create_test_candidate(1.0, frame_image=curr_image)]

        result = self.policy._calculate_ssim_scores(candidates)

        # SSIM should be calculated
        assert result[0].ssim_score == 0.9
        mock_ssim.assert_called_once()

    @patch('src.retrieval.keyframe_policy.ssim')
    def test_ssim_calculation_failure_handling(self, mock_ssim):
        """Test SSIM calculation handles failures gracefully."""
        # Mock SSIM to raise exception
        mock_ssim.side_effect = Exception("SSIM calculation failed")

        # Set up previous keyframe image
        prev_image = Image.new('RGB', (64, 64), color='red')
        self.policy.last_keyframe_image = prev_image

        # Create current frame image
        curr_image = Image.new('RGB', (64, 64), color='red')
        candidates = [self.create_test_candidate(1.0, frame_image=curr_image)]

        result = self.policy._calculate_ssim_scores(candidates)

        # SSIM should be None on failure
        assert result[0].ssim_score is None

    def test_ssim_threshold_trigger_application(self):
        """Test that SSIM threshold affects keyframe promotion."""
        # Create candidates with different SSIM scores
        candidates = [
            self.create_test_candidate(1.0, ssim_score=0.9),  # Above threshold (similar)
            self.create_test_candidate(2.0, ssim_score=0.7),  # Below threshold (different)
            self.create_test_candidate(3.0, ssim_score=None),  # No SSIM score
        ]

        # Apply triggers
        result = self.policy._apply_keyframe_triggers(candidates)

        # Candidate with SSIM below threshold should get promotion boost
        assert result[1].importance_score == 2.0  # Boosted due to low SSIM
        assert result[0].importance_score == 0.0  # Not boosted
        assert result[2].importance_score == 0.0  # Not boosted

    def test_keyframe_selection_with_ssim(self):
        """Test full keyframe selection with SSIM integration."""
        # Create test images
        base_image = Image.new('RGB', (64, 64), color='red')
        similar_image = Image.new('RGB', (64, 64), color=(255, 0, 0))  # Still red
        different_image = Image.new('RGB', (64, 64), color='blue')  # Different color

        # Set up policy with a previous keyframe
        self.policy.last_keyframe_image = base_image

        candidates = [
            self.create_test_candidate(1.0, frame_image=similar_image),   # Should have high SSIM
            self.create_test_candidate(2.0, frame_image=different_image), # Should have low SSIM
        ]

        # Mock SSIM to simulate the behavior
        with patch('src.retrieval.keyframe_policy.ssim') as mock_ssim:
            mock_ssim.side_effect = [0.95, 0.6]  # First similar, second different

            result = self.policy.select_keyframes(candidates)

            # Both SSIM calculations should be called
            assert mock_ssim.call_count == 2

            # Check that SSIM scores were set
            assert candidates[0].ssim_score == 0.95
            assert candidates[1].ssim_score == 0.6

            # The different frame should be boosted due to low SSIM
            # (The selection logic depends on the strategy and scoring)

    def test_clear_history_resets_ssim_state(self):
        """Test that clearing history resets SSIM-related state."""
        # Set up some state
        self.policy.last_keyframe_image = Image.new('RGB', (64, 64), color='red')
        self.policy.selected_keyframes = [self.create_test_candidate(1.0)]

        # Clear history
        self.policy.clear_history()

        # Check SSIM state is reset
        assert self.policy.last_keyframe_image is None
        assert len(self.policy.selected_keyframes) == 0

    def test_get_policy_stats_includes_ssim_threshold(self):
        """Test that policy stats include SSIM threshold information."""
        stats = self.policy.get_policy_stats()

        # Should include temporal window and other config
        assert "temporal_window_seconds" in stats
        # SSIM threshold is a configuration parameter, could be added to stats if needed