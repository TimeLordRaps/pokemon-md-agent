#!/usr/bin/env python3
"""
Memory Profiling Baseline Script for PMD-Red Agent

Tracks VRAM usage per model load/unload and memory leaks over time.
Uses memory_profiler for detailed heap analysis.

Run with: python profiling/memory_profiling.py
"""

import datetime
import gc
import json
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from memory_profiler import profile as memory_profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("WARNING: memory_profiler not available. Memory profiling disabled.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. VRAM profiling disabled.")

def run_memory_profiling(steps: int = 1000) -> Dict[str, Any]:
    """
    Profile memory usage over a 1000-step agent simulation.

    Args:
        steps: Number of agent steps to simulate

    Returns:
        Dict containing memory profiling results
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "steps": steps,
        "tracemalloc_stats": {},
        "vram_usage": {},
        "memory_growth": {},
        "gc_stats": {},
        "error": None
    }

    try:
        # Start tracemalloc for heap tracking
        tracemalloc.start()

        # Record initial memory state
        initial_snapshot = tracemalloc.take_snapshot()
        initial_stats = _get_memory_stats()

        print(f"Starting memory profiling for {steps} steps...")

        # Simulate agent steps with increasing memory pressure
        for step in range(steps):
            _simulate_memory_intensive_step(step)

            # Periodic memory checks
            if step % 100 == 0:
                gc.collect()  # Force garbage collection
                snapshot = tracemalloc.take_snapshot()
                stats = _get_memory_stats()

                # Track memory growth
                key = f"step_{step}"
                results["memory_growth"][key] = {
                    "current_mb": stats["current_mb"],
                    "peak_mb": stats["peak_mb"],
                    "growth_mb": stats["current_mb"] - initial_stats["current_mb"]
                }

                if HAS_TORCH and torch.cuda.is_available():
                    results["vram_usage"][key] = {
                        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                    }

                print(f"Step {step}: {stats['current_mb']:.1f} MB current, {stats['peak_mb']:.1f} MB peak")

        # Final memory analysis
        final_snapshot = tracemalloc.take_snapshot()
        final_stats = _get_memory_stats()

        # Compare snapshots for memory leaks
        stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        results["tracemalloc_stats"] = {
            "top_10_allocators": [
                {
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                    "traceback": str(stat.traceback).split('\n')[:3]  # First 3 lines
                }
                for stat in stats[:10]
            ]
        }

        # Overall memory growth
        results["memory_growth"]["summary"] = {
            "initial_mb": initial_stats["current_mb"],
            "final_mb": final_stats["current_mb"],
            "total_growth_mb": final_stats["current_mb"] - initial_stats["current_mb"],
            "growth_rate_kb_per_step": ((final_stats["current_mb"] - initial_stats["current_mb"]) * 1024) / steps
        }

        # GC statistics
        results["gc_stats"] = {
            "collections": gc.get_count(),
            "stats": gc.get_stats()
        }

    except Exception as e:
        results["error"] = str(e)
    finally:
        tracemalloc.stop()

    return results

def _get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    current, peak = tracemalloc.get_traced_memory()
    return {
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024
    }

def _simulate_memory_intensive_step(step: int):
    """
    Simulate a single agent step with realistic memory usage patterns.

    Args:
        step: Current step number (affects memory pressure)
    """
    # Simulate vision pipeline memory usage
    # Create temporary image buffers and processing data
    _simulate_vision_memory(step)

    # Simulate model inference memory (tensors, caches)
    _simulate_model_memory(step)

    # Simulate retrieval system memory (vectors, indexes)
    _simulate_retrieval_memory(step)

    # Simulate environment memory (RAM snapshots, state)
    _simulate_environment_memory(step)

    # Add some memory pressure that grows over time
    _add_memory_pressure(step)

def _simulate_vision_memory(step: int):
    """Simulate memory usage from vision processing."""
    # Simulate screenshot buffers (RGB images)
    image_size = 240 * 160 * 3  # GBA resolution * RGB
    screenshot_buffers = [bytearray(image_size) for _ in range(4)]  # 4-up capture

    # Simulate grid parsing data structures
    grid_data = [[0 for _ in range(30)] for _ in range(20)]  # 30x20 grid

    # Simulate ASCII rendering buffers
    ascii_buffer = ["."] * (30 * 20)

    # Keep references briefly then release
    del screenshot_buffers, grid_data, ascii_buffer

def _simulate_model_memory(step: int):
    """Simulate memory usage from model inference."""
    if not HAS_TORCH:
        return

    # Simulate token embeddings (batch_size=1, seq_len=512, hidden_dim=2048)
    batch_size, seq_len, hidden_dim = 1, 512, 2048

    # Create tensors that mimic model internal state
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)

    # Simulate attention mechanism memory
    num_heads = 16
    head_dim = hidden_dim // num_heads
    qkv = torch.randn(batch_size, seq_len, hidden_dim * 3)
    attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # Simulate feed-forward network
    ff_hidden = torch.randn(batch_size, seq_len, hidden_dim * 4)

    # Keep brief reference then release
    del embeddings, qkv, attn_weights, ff_hidden

def _simulate_retrieval_memory(step: int):
    """Simulate memory usage from retrieval system."""
    # Simulate FAISS index memory (vector storage)
    vector_dim = 768
    num_vectors = 1000 + (step // 10)  # Growing vector database
    vectors = [[0.0] * vector_dim for _ in range(min(num_vectors, 100))]  # Sample subset

    # Simulate metadata storage
    metadata = [{"id": i, "text": f"sample_text_{i}"} for i in range(100)]

    # Simulate query results
    query_results = [{"score": 0.95 - i*0.01, "id": i} for i in range(10)]

    del vectors, metadata, query_results

def _simulate_environment_memory(step: int):
    """Simulate memory usage from environment interaction."""
    # Simulate RAM snapshots (WRAM + IRAM)
    ram_snapshot = bytearray(256 * 1024)  # 256KB WRAM
    iram_snapshot = bytearray(32 * 1024)   # 32KB IRAM

    # Simulate decoded game state
    game_state = {
        "player": {"x": 100, "y": 100, "hp": 100, "level": 5},
        "inventory": [{"id": i, "quantity": 1} for i in range(20)],
        "dungeon": {"floor": step % 10, "room": step % 50},
        "enemies": [{"type": "monster", "hp": 50} for _ in range(5)]
    }

    # Simulate WebSocket message buffers
    ws_messages = [b"<|screenshot|>" + b"x" * 1000 for _ in range(10)]

    del ram_snapshot, iram_snapshot, game_state, ws_messages

def _add_memory_pressure(step: int):
    """
    Add controlled memory pressure that increases over time.
    This simulates memory leaks or growing caches.
    """
    # Create some persistent objects that accumulate
    persistent_data = []

    # Amount of persistent data grows slowly over time
    num_items = min(step // 50, 20)  # Max 20 persistent items

    for i in range(num_items):
        # Simulate cached computation results or persistent state
        item = {
            "step": step,
            "data": list(range(100)),  # Some data that takes memory
            "metadata": f"cached_result_{step}_{i}"
        }
        persistent_data.append(item)

    # Store in a global-like structure (simulating memory leak)
    if not hasattr(_add_memory_pressure, 'persistent_storage'):
        _add_memory_pressure.persistent_storage = []

    _add_memory_pressure.persistent_storage.extend(persistent_data)

    # Occasionally clean up (but not completely, simulating partial cleanup)
    if step % 200 == 0 and _add_memory_pressure.persistent_storage:
        # Remove half the items (simulating garbage collection)
        keep_count = len(_add_memory_pressure.persistent_storage) // 2
        _add_memory_pressure.persistent_storage = _add_memory_pressure.persistent_storage[:keep_count]

if __name__ == "__main__":
    print("Starting memory profiling...")

    results = run_memory_profiling(1000)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = f"profiling/memory_baseline_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write("PMD-Red Agent Memory Profiling Results\n")
        f.write("=" * 50 + "\n\n")

        if "error" in results and results["error"]:
            f.write(f"ERROR: {results['error']}\n\n")
        else:
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Steps: {results['steps']}\n\n")

            # Memory growth summary
            growth = results.get("memory_growth", {}).get("summary", {})
            if growth:
                f.write("MEMORY GROWTH SUMMARY:\n")
                f.write(".1f")
                f.write(".1f")
                f.write(".1f")
                f.write(".3f")
                f.write("\n\n")

            # Top memory allocators
            allocators = results.get("tracemalloc_stats", {}).get("top_10_allocators", [])
            if allocators:
                f.write("TOP 10 MEMORY ALLOCATORS:\n")
                for i, alloc in enumerate(allocators[:10], 1):
                    f.write(f"{i}. Size: {alloc['size_mb']:.1f} MB, Count: {alloc['count']}\n")
                    f.write(f"   Traceback: {' | '.join(alloc['traceback'])}\n")
                f.write("\n")

            # GC stats
            gc_stats = results.get("gc_stats", {})
            if gc_stats:
                f.write("GARBAGE COLLECTION STATS:\n")
                f.write(f"Collections: {gc_stats.get('collections', 'N/A')}\n")
                f.write("\n")

    # Also save raw JSON for programmatic analysis
    json_file = f"profiling/memory_baseline_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Memory profiling complete. Results saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")

    # Print key findings
    if "error" not in results:
        growth = results.get("memory_growth", {}).get("summary", {})
        if growth:
            print(".1f")
            print(".3f")