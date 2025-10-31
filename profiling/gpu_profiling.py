#!/usr/bin/env python3
"""
GPU Profiling Baseline Script for PMD-Red Agent

Profiles CUDA kernel times and memory transfers for Qwen3-VL model inference.
Uses QwenController for real model loading and inference profiling.
Requires torch.profiler and CUDA-enabled PyTorch.

Run with: HF_HOME=E:\transformer_models python profiling/gpu_profiling.py
"""

import datetime
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set HF_HOME environment variable for model downloads
hf_home = os.environ.get('HF_HOME', 'E:\\transformer_models')
os.environ['HF_HOME'] = hf_home

# Set HF token if available
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HUGGINGFACE_TOKEN'] = hf_token
    print(f"Using HuggingFace token for authentication")
else:
    print("No HF_TOKEN found - downloads may fail if authentication required")

try:
    import torch
    import torch.profiler as profiler
    HAS_CUDA = torch.cuda.is_available()
    from src.agent.qwen_controller import QwenController
    from src.agent.model_router import ModelSize
except ImportError as e:
    HAS_CUDA = False
    print(f"WARNING: Import failed: {e}. GPU profiling disabled.")

def run_gpu_profiling() -> Dict[str, Any]:
    """
    Profile GPU usage during real Qwen3-VL model inference using QwenController.

    Returns:
        Dict containing profiling results and metrics
    """
    if not HAS_CUDA:
        return {"error": "CUDA not available"}

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cuda_available": True,
        "device": torch.cuda.get_device_name(),
        "models_profiled": [],
        "memory_usage": {},
        "kernel_times": {}
    }

    # Initialize QwenController
    controller = QwenController()

    # Profile each Qwen3-VL model variant using controller's registry
    registry = controller.get_armada_registry()

    # Map registry keys to model configurations - using EXACTLY the specified model names
    model_configs = [
        ("Qwen/Qwen3-VL-2B-Thinking-FP8", "Qwen/Qwen3-VL-2B-Thinking-FP8"),
        ("unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit", "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit", "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit", "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit", "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    ]

    for model_id, model_id in model_configs:
        try:
            model_result = profile_single_model(controller, model_id, model_id)
            results["models_profiled"].append(model_result)
            print(f"✓ Profiled {model_id}")
        except Exception as e:
            print(f"✗ Failed to profile {model_id}: {e}")
            results["models_profiled"].append({
                "model": model_id,
                "error": str(e)
            })

    # Record overall GPU memory usage
    if torch.cuda.is_available():
        results["memory_usage"] = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024
        }

    return results

def profile_single_model(controller: QwenController, model_key: str, model_id: str) -> Dict[str, Any]:
    """
    Profile a single Qwen3-VL model using QwenController with automatic download and VRAM tracking.

    Args:
        controller: QwenController instance
        model_key: Registry key for the model
        model_id: HuggingFace model ID

    Returns:
        Profiling results for this model
    """
    result = {
        "model": model_key,
        "model_id": model_id,
        "inference_time_ms": None,
        "memory_peak_mb": None,
        "cuda_events": [],
        "error": None
    }

    try:
        # Clear any existing CUDA cache and reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Record initial memory state
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        # Load model using controller - this will trigger automatic download if needed
        print(f"Loading model {model_key} ({model_id})...")
        load_success = controller.load_model(model_id)
        if not load_success:
            raise RuntimeError(f"Failed to load model {model_id}")

        # Record memory after loading
        post_load_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        result["load_memory_mb"] = post_load_memory - initial_memory

        # Create test image for vision inference (use PMD-appropriate size)
        test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Reset peak memory stats before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Profile actual vision generation with 1000-step simulation
        start_time = time.time()
        total_inference_time = 0
        inference_count = 1000
        memory_peaks = []

        print(f"Running {inference_count} inference steps for {model_id}...")

        for step in range(inference_count):
            if step % 100 == 0:
                print(f"Step {step}/{inference_count}")

            # Reset peak memory stats before each inference
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            step_start = time.time()

            # Run actual inference with consistent input
            vision_result = controller.generate_vision(
                images=[test_image],
                text="Analyze this Pokemon Mystery Dungeon screenshot and describe what you see.",
                mode="instruct" if "instruct" in model_id else "thinking"
            )

            step_end = time.time()
            step_time_ms = (step_end - step_start) * 1000
            total_inference_time += step_time_ms

            # Record peak memory for this step
            if torch.cuda.is_available():
                memory_peaks.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_inference_time_ms = total_inference_time / inference_count

        # Calculate memory statistics
        if memory_peaks:
            result["memory_peak_mb"] = max(memory_peaks)
            result["memory_avg_mb"] = sum(memory_peaks) / len(memory_peaks)
        else:
            result["memory_peak_mb"] = 0
            result["memory_avg_mb"] = 0

        # Calculate additional metrics
        result["total_time_ms"] = total_time_ms
        result["avg_inference_time_ms"] = avg_inference_time_ms
        result["inference_steps"] = inference_count
        result["vision_result"] = {
            "response_length": len(vision_result.response),
            "input_tokens": vision_result.input_tokens,
            "output_tokens": vision_result.output_tokens,
            "generation_time_ms": vision_result.generation_time
        }

        print(f"✓ {model_id}: {avg_inference_time_ms:.1f}ms avg, {result['memory_peak_mb']:.1f}MB peak VRAM")

    except Exception as e:
        result["error"] = str(e)
        print(f"✗ {model_key}: {e}")

    return result

def generate_baseline_report(results: Dict[str, Any]) -> str:
    """
    Generate a structured model comparison report.

    Args:
        results: Profiling results dictionary

    Returns:
        Formatted comparison report
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("GPU PROFILING MODEL COMPARISON REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"CUDA Device: {results.get('device', 'Unknown')}")
    report_lines.append("")

    # Extract model results
    models_profiled = results.get("models_profiled", [])
    successful_models = [m for m in models_profiled if "error" not in m]

    if not successful_models:
        report_lines.append("NO SUCCESSFUL MODEL PROFILING RESULTS")
        return "\n".join(report_lines)

    # Table header
    report_lines.append("MODEL BASELINE TABLE")
    report_lines.append("-" * 100)
    report_lines.append("<25")
    report_lines.append("-" * 100)

    # Model rows
    for model in successful_models:
        name = model["model"]
        total_time = model.get("total_time_ms", 0)
        avg_time = model.get("avg_inference_time_ms", 0)
        memory_peak = model.get("memory_peak_mb", 0)
        memory_avg = model.get("memory_avg_mb", 0)
        steps = model.get("inference_steps", 0)

        report_lines.append("<25")

    report_lines.append("-" * 80)
    report_lines.append("")

    # Summary statistics
    if successful_models:
        times = [m.get("inference_time_ms", 0) for m in successful_models if m.get("inference_time_ms")]
        memories = [m.get("memory_peak_mb", 0) for m in successful_models if m.get("memory_peak_mb")]

        if times:
            total_times = [m.get("total_time_ms", 0) for m in successful_models if m.get("total_time_ms")]
            avg_times = [m.get("avg_inference_time_ms", 0) for m in successful_models if m.get("avg_inference_time_ms")]

            report_lines.append("SUMMARY STATISTICS:")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append("")

        if memories:
            peak_memories = [m.get("memory_peak_mb", 0) for m in successful_models if m.get("memory_peak_mb")]
            avg_memories = [m.get("memory_avg_mb", 0) for m in successful_models if m.get("memory_avg_mb")]

            report_lines.append("MEMORY USAGE:")
            report_lines.append(".1f")
            report_lines.append(".1f")
            report_lines.append(".1f")
            report_lines.append(".1f")

    # Overall GPU memory usage
    memory_usage = results.get("memory_usage", {})
    if memory_usage:
        report_lines.append("")
        report_lines.append("OVERALL GPU MEMORY USAGE:")
        report_lines.append(".1f")
        report_lines.append(".1f")
        report_lines.append(".1f")
        report_lines.append(".1f")

    # Failed models
    failed_models = [m for m in models_profiled if "error" in m]
    if failed_models:
        report_lines.append("")
        report_lines.append("FAILED MODELS:")
        for model in failed_models:
            report_lines.append(f"  - {model['model']}: {model['error']}")

    report_lines.append("")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)

if __name__ == "__main__":
    print("Starting GPU profiling...")

    results = run_gpu_profiling()

    # Save JSON results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    json_file = f"profiling/gpu_baseline_{timestamp}.json"

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"GPU profiling complete. JSON results saved to {json_file}")

    # Generate and save baseline report in specified format
    baseline_report = generate_baseline_report(results)
    txt_file = f"profiling/gpu_baseline_{timestamp}.txt"

    with open(txt_file, 'w') as f:
        f.write(baseline_report)

    print(f"Baseline report saved to {txt_file}")

    # Print summary
    if "error" not in results:
        print(f"CUDA Device: {results.get('device', 'Unknown')}")
        print(f"Models profiled: {len(results.get('models_profiled', []))}")
        memory = results.get('memory_usage', {})
        if memory:
            print(".1f")
            print(".1f")
        print(f"Results saved as: {json_file} and {txt_file}")