"""Benchmark Unsloth backend against Transformers baseline for tokens/sec measurement."""

import os
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest
import torch

try:
    import unsloth  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    unsloth = None  # type: ignore

from src.agent.qwen_controller import QwenController


@pytest.fixture
def sample_prompts():
    """Sample prompts for benchmarking."""
    return [
        "Hello, how are you today?",
        "Describe the weather in a tropical paradise.",
        "What is artificial intelligence and how does it work?",
        "Tell me about the game Pokemon Mystery Dungeon.",
        "How do I navigate a dungeon in Pokemon Mystery Dungeon?",
        "What are the best items to bring on a dungeon adventure?",
        "Explain the difference between thinking and reasoning.",
        "How can I solve problems more effectively?",
        "What are some tips for playing strategy games?",
        "Describe a typical day in an AI research lab."
    ]


@pytest.fixture
def dummy_image():
    """Create a dummy image tensor for multimodal requests."""
    return torch.randint(0, 256, (3, 480, 320), dtype=torch.uint8)


@pytest.fixture
def benchmark_config():
    """Benchmark configuration."""
    return {
        "model_size": "2B",
        "use_thinking": False,
        "max_tokens": 100,
        "temperature": 0.7,
        "iterations": 5,
        "warmup_iterations": 2
    }


def measure_tokens_per_second(prompt: str, controller: QwenController, config: dict, image: torch.Tensor) -> float:
    """Measure tokens per second for a single prompt."""
    start_time = time.time()

    # Force sync call (no async in test context)
    result = controller.generate(
        prompt=prompt,
        images=[image.clone()],
        model_size=config["model_size"],
        use_thinking=config["use_thinking"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
    )

    elapsed_time = time.time() - start_time
    # Rough token count estimation (words * 1.3 for token approximation)
    token_count = len(result.split()) * 1.3
    return token_count / elapsed_time if elapsed_time > 0 else 0


@pytest.mark.parametrize("backend", ["hf", "unsloth"])
def test_backend_benchmark(backend, sample_prompts, benchmark_config, dummy_image):
    """Benchmark tokens/sec for different backends."""
    # Set environment
    original_backend = os.environ.get("MODEL_BACKEND")
    os.environ["MODEL_BACKEND"] = backend
    os.environ["REAL_MODELS_DRYRUN"] = "0"  # Allow actual loads

    try:
        controller = QwenController(use_pipeline=False)  # Disable pipeline for benchmark test

        # Warmup
        for _ in range(benchmark_config["warmup_iterations"]):
            controller.generate(
                prompt="Warmup prompt",
                images=[dummy_image.clone()],
                model_size=benchmark_config["model_size"],
                max_tokens=10
            )

        # Benchmark
        total_tps = 0
        for prompt in sample_prompts[:benchmark_config["iterations"]]:
            tps = measure_tokens_per_second(prompt, controller, benchmark_config, dummy_image)
            total_tps += tps
            print(f"{backend}: {tps:.2f} tokens/sec for prompt")

        avg_tps = total_tps / benchmark_config["iterations"]
        print(f"{backend} average: {avg_tps:.2f} tokens/sec")

        # Store result for comparison
        setattr(test_backend_benchmark, f"{backend}_tps", avg_tps)

        # Basic assertion - should be positive
        assert avg_tps > 0, f"Tokens per second should be positive, got {avg_tps}"

    finally:
        # Restore environment
        if original_backend:
            os.environ["MODEL_BACKEND"] = original_backend
        else:
            os.environ.pop("MODEL_BACKEND", None)


def test_unsloth_vs_transformers_performance(sample_prompts, benchmark_config, dummy_image):
    """Compare Unsloth vs Transformers performance."""
    # Run HF benchmark
    os.environ["MODEL_BACKEND"] = "hf"
    os.environ["REAL_MODELS_DRYRUN"] = "0"
    controller_hf = QwenController(use_pipeline=False)  # Disable pipeline

    hf_tps_list = []
    for prompt in sample_prompts[:benchmark_config["iterations"]]:
        tps = measure_tokens_per_second(prompt, controller_hf, benchmark_config, dummy_image)
        hf_tps_list.append(tps)

    hf_avg_tps = sum(hf_tps_list) / len(hf_tps_list)
    print(f"Transformers baseline: {hf_avg_tps:.2f} tokens/sec")

    # Run Unsloth benchmark
    os.environ["MODEL_BACKEND"] = "unsloth"
    os.environ["REAL_MODELS_DRYRUN"] = "0"
    controller_unsloth = QwenController(use_pipeline=False)  # Disable pipeline

    unsloth_tps_list = []
    for prompt in sample_prompts[:benchmark_config["iterations"]]:
        tps = measure_tokens_per_second(prompt, controller_unsloth, benchmark_config, dummy_image)
        unsloth_tps_list.append(tps)

    unsloth_avg_tps = sum(unsloth_tps_list) / len(unsloth_tps_list)
    print(f"Unsloth: {unsloth_avg_tps:.2f} tokens/sec")

    # Calculate performance change
    if hf_avg_tps > 0:
        percent_change = ((unsloth_avg_tps - hf_avg_tps) / hf_avg_tps) * 100
        print(f"Performance change: {percent_change:.1f}%")

        # Assertions based on requirements
        # Should not be more than 20% slower
        assert percent_change >= -20.0, f"Unsloth is {abs(percent_change):.1f}% slower than Transformers (threshold: 20%)"

        # Should achieve at least 95% of baseline performance
        assert unsloth_avg_tps >= hf_avg_tps * 0.95, f"Unsloth achieves only {unsloth_avg_tps/hf_avg_tps*100:.1f}% of Transformers performance"

    print("All performance requirements met!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
