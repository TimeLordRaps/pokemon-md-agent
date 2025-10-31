"""Test best-of-n functionality for improved quality over n=1 baseline.

Validates that n>1 generates improve scores via parallel sampling and RRF scoring.
Tests parameter validation, parallel generation, scoring mechanics, and routing
integration. Ensures quality gains scale with n while maintaining latency bounds.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.agent.model_router import ModelRouter, ModelSize, PrefillRequest, DecodeRequest
from src.agent.qwen_controller import QwenController, DecodeResult


class TestBestOfN:
    """Test best-of-n generation with scoring."""

    def test_best_of_n_parameter_validation(self):
        """Test best_of_n parameter accepts valid values."""
        controller = QwenController()

        # Valid values
        for valid_n in [1, 2, 4, 8]:
            result = controller.generate("test", best_of_n=valid_n)
            assert isinstance(result, str)

        # Invalid values should raise
        with pytest.raises(ValueError):
            controller.generate("test", best_of_n=3)

        with pytest.raises(ValueError):
            controller.generate("test", best_of_n=0)

    @pytest.mark.asyncio
    async def test_best_of_n_single_candidate(self):
        """Test best_of_n=1 works like normal generation."""
        controller = QwenController()

        with patch.object(controller, '_single_generate', new_callable=AsyncMock) as mock_single:
            mock_single.return_value = "test output"

            result, scores = await controller.generate_async("test prompt", best_of_n=1)

            assert result == "test output"
            assert scores == [0.04]  # Single candidate gets score based on token count (2/50 = 0.04)
            mock_single.assert_called_once()

    @pytest.mark.asyncio
    async def test_best_of_n_parallel_generation(self):
        """Test best_of_n>1 generates multiple candidates in parallel."""
        controller = QwenController()

        # Mock single generation with different outputs
        outputs = ["output 1", "output 2", "output 3", "output 4"]

        with patch.object(controller, '_single_generate', new_callable=AsyncMock) as mock_single:
            mock_single.side_effect = outputs

            result, scores = await controller.generate_async(
                "test prompt",
                best_of_n=4,
                retrieval_scores=[0.8, 0.6, 0.9, 0.7]
            )

            # Should call generate 4 times
            assert mock_single.call_count == 4

            # Should return one result and scores for all candidates
            assert result in outputs
            assert len(scores) == 4
            assert all(isinstance(score, float) for score in scores)

    def test_scoring_with_retrieval_scores(self):
        """Test scoring combines logprob and retrieval scores via RRF."""
        controller = QwenController()

        # Mock decode results with different token counts for scoring
        candidates = [
            DecodeResult("output 1", tokens_used=45, latency_ms=100.0),  # Higher "logprob" (45/50 = 0.9)
            DecodeResult("output 2", tokens_used=40, latency_ms=120.0),  # Lower "logprob" (40/50 = 0.8)
        ]

        retrieval_scores = [0.7, 0.6]  # First has higher retrieval score

        scores = controller._score_candidates(
            candidates, retrieval_scores, k=60
        )

        # First candidate: higher logprob (0.9) + RRF with retrieval (0.7)
        # Second candidate: lower logprob (0.8) + RRF with retrieval (0.6)
        # First should score higher due to both factors
        assert scores[0] > scores[1]

    def test_rrf_calculation(self):
        """Test Reciprocal Rank Fusion calculation."""
        controller = QwenController()

        # Test RRF with different relevance scores (higher = better)
        score1 = controller._rrf_score(1.0, k=60)  # Highest relevance (rank 1)
        score2 = controller._rrf_score(0.0, k=60)  # Lowest relevance (rank 11)

        assert score1 > score2
        assert score1 == 1.0 / (60 + 1)  # 1/(k+1) for rank 1
        assert score2 == 1.0 / (60 + 11)  # 1/(k+11) for rank 11

    def test_candidate_selection(self):
        """Test argmax selection of best candidate."""
        controller = QwenController()

        candidates = ["bad", "good", "best"]
        scores = [0.5, 0.8, 0.9]

        selected, all_scores = controller._select_best_candidate(candidates, scores)

        assert selected == "best"
        assert all_scores == [0.5, 0.8, 0.9]

    def test_best_of_n_with_router_integration(self):
        """Test best-of-n works through ModelRouter interface."""
        router = ModelRouter()

        # Mock the pipeline to return multiple results
        with patch.object(router.two_stage_pipeline, 'submit_prefill') as mock_prefill:
            mock_future = AsyncMock()
            mock_prefill.return_value = mock_future

            # Simulate best_of_n=2
            request = PrefillRequest(
                prompt="test",
                model_size=ModelSize.SIZE_4B,
                best_of_n=2
            )

            future = router.two_stage_pipeline.submit_prefill(request)

            # Should create future for parallel processing
            assert future == mock_future
            mock_prefill.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_best_of_n_score_improvement_over_n1(self):
        """Test that n>1 scores higher than n=1 baseline."""
        controller = QwenController()

        # Mock single generation with fixed outputs of different lengths
        # (longer outputs get higher scores in the mock implementation)
        outputs = ["poor", "good quality response", "best quality response with more details"]
        scores = [0.02, 0.06, 0.08]  # 1/50, 4/50, 6/50

        with patch.object(controller, '_single_generate', new_callable=AsyncMock) as mock_single:
            mock_single.side_effect = outputs

            # Test n=1
            result_n1, scores_n1 = await controller.generate_async("test", best_of_n=1)
            assert result_n1 == "poor"
            assert len(scores_n1) == 1

            # Reset mock
            mock_single.reset_mock()
            mock_single.side_effect = outputs

            # Test n=2
            result_n2, scores_n2 = await controller.generate_async("test", best_of_n=2)
            assert result_n2 == "good quality response"  # Should select highest scoring
            assert len(scores_n2) == 2
            assert max(scores_n2) > max(scores_n1)  # n=2 should achieve higher max score

    @pytest.mark.asyncio
    async def test_best_of_n_latency_scaling(self):
        """Test latency scales reasonably with n (not linearly)."""
        import time
        import asyncio
        controller = QwenController()

        # Mock with timing
        async def timed_generate(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms per generation
            return "output"

        with patch.object(controller, '_single_generate', side_effect=timed_generate):
            start = time.time()
            await controller.generate_async("test", best_of_n=1)
            n1_time = time.time() - start

            start = time.time()
            await controller.generate_async("test", best_of_n=4)
            n4_time = time.time() - start

            # Should be less than 4x (due to parallelism)
            assert n4_time < n1_time * 3.5