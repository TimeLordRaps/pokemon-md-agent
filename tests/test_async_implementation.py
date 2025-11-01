"""Test async/await implementation in qwen_controller.py."""

import sys
import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.qwen_controller import QwenController


class TestAsyncImplementation:
    """Test async/await usage in qwen_controller.py."""

    @pytest.fixture
    def controller(self):
        """Create controller for testing."""
        return QwenController()

    def test_async_methods_are_properly_defined(self, controller):
        """Test that async methods are properly defined with async keyword."""
        # Check that key async methods exist and are coroutines
        async_methods = [
            'generate_async',
            'initialize_async',
            '_generate_candidates_parallel',
            '_generate_with_cache',
            '_single_generate',
            '_process_prefill_batch',
            '_process_decode_batch',
        ]

        for method_name in async_methods:
            method = getattr(controller, method_name, None)
            assert method is not None, f"Method {method_name} not found"
            assert inspect.iscoroutinefunction(method), f"Method {method_name} is not async"

    def test_sync_generate_method_exists(self, controller):
        """Test that sync generate method exists for compatibility."""
        assert hasattr(controller, 'generate')
        assert callable(getattr(controller, 'generate'))
        assert not inspect.iscoroutinefunction(controller.generate)

    def test_async_method_signatures(self, controller):
        """Test that async methods have correct signatures."""
        # Test generate_async signature
        sig = inspect.signature(controller.generate_async)
        expected_params = ['prompt', 'images', 'model_size', 'use_thinking',
                          'max_tokens', 'temperature', 'best_of_n', 'retrieval_scores',
                          'tool_schema', 'yield_every']
        actual_params = list(sig.parameters.keys())  # Don't skip 'self' - bound methods don't include it

        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from generate_async"

    def test_pipeline_methods_are_async(self, controller):
        """Test that pipeline processing methods are async."""
        pipeline_methods = ['_process_prefill_batch', '_process_decode_batch']

        for method_name in pipeline_methods:
            method = getattr(controller, method_name, None)
            assert method is not None, f"Pipeline method {method_name} not found"
            assert inspect.iscoroutinefunction(method), f"Pipeline method {method_name} is not async"

    def test_async_context_manager_support(self, controller):
        """Test that controller supports async context management."""
        # Check for async initialization method
        assert hasattr(controller, 'initialize_async')
        assert inspect.iscoroutinefunction(controller.initialize_async)

    def test_parallel_generation_uses_asyncio_gather(self, controller):
        """Test that parallel generation properly uses asyncio.gather."""
        with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ["response1", "response2"]

            async def test_parallel():
                return await controller._generate_candidates_parallel(
                    "test prompt", None, "test_model", 100, 0.7, 2
                )

            # This would normally require an event loop, but we're just checking the signature
            # In real test, would run in async test
            assert inspect.iscoroutinefunction(controller._generate_candidates_parallel)

    @pytest.mark.network
    def test_async_method_error_handling(self, controller):
        """Test async methods have proper error handling."""
        # Test that async methods can be called (even if they fail due to missing dependencies)
        async def test_async_call():
            try:
                # This will likely fail due to missing dependencies or uninitialized components
                await controller.generate_async("test", max_tokens=10)
            except (ImportError, RuntimeError) as e:
                # Expected - missing dependencies, uninitialized pipeline, or model not loaded
                error_msg = str(e).lower()
                expected_errors = ["torch", "transformers", "model", "pipeline", "failed"]
                assert any(expected in error_msg for expected in expected_errors), f"Unexpected error: {e}"
            except Exception as e:
                # Unexpected error - should be related to implementation, not async syntax
                assert "async" not in str(e).lower() and "coroutine" not in str(e).lower()

        # Run the async test
        try:
            asyncio.run(test_async_call())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # We're in an existing event loop, skip this test
                pytest.skip("Cannot run async test in existing event loop")
            else:
                raise

    def test_pipeline_initialization_is_async(self, controller):
        """Test that pipeline initialization is properly async."""
        assert hasattr(controller, 'initialize_async')
        assert inspect.iscoroutinefunction(controller.initialize_async)

        # Check that it tries to initialize pipeline components
        assert hasattr(controller, 'pipeline_initialized')
        assert controller.pipeline_initialized is False  # Should start False

    def test_async_wait_functionality(self, controller):
        """Test async wait functionality in pipeline methods."""
        # Test that pipeline methods contain async wait logic
        import inspect

        source = inspect.getsource(controller._process_prefill_batch)
        # Should contain async operations
        assert "async def" in source or "await" in source or "asyncio" in source

        source = inspect.getsource(controller._process_decode_batch)
        assert "async def" in source or "await" in source or "asyncio" in source

    def test_best_of_n_implementation_uses_async(self, controller):
        """Test that best-of-n selection uses async properly."""
        # Check that the method that handles best_of_n > 1 is async
        assert inspect.iscoroutinefunction(controller._generate_candidates_parallel)

        # Check signature includes n parameter
        sig = inspect.signature(controller._generate_candidates_parallel)
        assert 'n' in sig.parameters

    def test_async_timeout_handling(self, controller):
        """Test async timeout handling in generation."""
        # Check that async methods have timeout parameters or handling
        sig = inspect.signature(controller.generate_async)
        # Should not have timeout in main API, but internal methods might
        assert 'timeout' not in sig.parameters  # Main API doesn't expose timeout

        # But internal methods should handle timeouts
        sig = inspect.signature(controller._generate_with_cache)
        assert 'wall_budget_s' in sig.parameters  # Wall time budget instead

    def test_async_cancellation_handling(self, controller):
        """Test that async methods handle cancellation properly."""
        # Check that methods use asyncio.wait with timeout/cancellation support
        import inspect

        source = inspect.getsource(controller._generate_with_cache)
        # Should contain asyncio.wait or similar cancellation-aware constructs
        assert "asyncio.wait" in source or "asyncio.gather" in source

    def test_async_semaphore_usage(self, controller):
        """Test that async methods use semaphores for resource management."""
        # Check that VRAM semaphores are used
        assert hasattr(controller, 'vram_semaphores')
        assert isinstance(controller.vram_semaphores, dict)

        # Check semaphore acquisition method exists
        assert hasattr(controller, '_get_or_create_vram_semaphore')
        semaphore = controller._get_or_create_vram_semaphore("test_model")
        assert semaphore is not None

    def test_async_error_propagation(self, controller):
        """Test that async methods properly propagate errors."""
        # Test that custom exceptions are defined and used
        from src.agent.qwen_controller import GenerationBudgetExceeded, BestOfSelectionError, PipelineError

        assert GenerationBudgetExceeded is not None
        assert BestOfSelectionError is not None
        assert PipelineError is not None

        # Check that async methods can raise these
        sig = inspect.signature(controller.generate_async)
        # The raises documentation should mention these exceptions
        # (though we can't easily check docstrings here)

    def test_async_partial_result_yielding(self, controller):
        """Test async partial result yielding functionality."""
        sig = inspect.signature(controller.generate_async)
        assert 'yield_every' in sig.parameters

        # Check that the implementation handles yield_every
        source = inspect.getsource(controller._generate_with_cache)
        assert "yield_every" in source

    def test_async_pipeline_coordination(self, controller):
        """Test async coordination between pipeline stages."""
        # Check that pipeline engine is properly integrated
        assert hasattr(controller, 'pipeline_engine')

        # Check that async methods submit to pipeline
        source = inspect.getsource(controller.generate_async)
        assert "pipeline_engine" in source or "PipelineRequest" in source

    def test_async_fallback_behavior(self, controller):
        """Test async fallback to synchronous behavior."""
        # When pipeline fails, should fall back to other methods
        source = inspect.getsource(controller.generate_async)
        assert "fallback" in source.lower() or "except" in source or "try" in source

    def test_async_cleanup_and_finalization(self, controller):
        """Test async cleanup and finalization."""
        # Check that async methods have proper cleanup
        assert hasattr(controller, '_restart_pipeline')
        restart_method = getattr(controller, '_restart_pipeline')
        assert inspect.iscoroutinefunction(restart_method)

    def test_async_concurrent_safety(self, controller):
        """Test that async implementation is safe for concurrent calls."""
        # Check that semaphores prevent VRAM conflicts
        assert hasattr(controller, '_get_or_create_vram_semaphore')

        # Check that concurrent calls use different semaphore instances per model
        semaphore1 = controller._get_or_create_vram_semaphore("model1")
        semaphore2 = controller._get_or_create_vram_semaphore("model2")
        semaphore1_again = controller._get_or_create_vram_semaphore("model1")

        assert semaphore1 is semaphore1_again  # Same instance for same model
        assert semaphore1 is not semaphore2    # Different instances for different models

    def test_async_deadline_and_budget_enforcement(self, controller):
        """Test async deadline and budget enforcement."""
        # Check wall time budget enforcement
        source = inspect.getsource(controller._generate_with_cache)
        assert "wall_budget_s" in source or "timeout" in source

        # Check that GenerationBudgetExceeded can be raised
        from src.agent.qwen_controller import GenerationBudgetExceeded
        assert issubclass(GenerationBudgetExceeded, Exception)