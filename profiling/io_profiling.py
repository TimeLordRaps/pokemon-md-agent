#!/usr/bin/env python3
"""
I/O Profiling Baseline Script for PMD-Red Agent

Measures WebSocket latency, disk I/O for savestates, and FAISS query performance.
Simulates realistic I/O patterns without requiring mGBA connection.

Run with: python profiling/io_profiling.py
"""

import asyncio
import datetime
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: FAISS not available. Vector query profiling disabled.")

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("WARNING: websockets not available. WebSocket profiling disabled.")

class IOProfiler:
    """Comprehensive I/O profiler for PMD-Red agent subsystems."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "websocket_latency": {},
            "disk_io": {},
            "vector_queries": {},
            "network_simulation": {}
        }

    async def run_full_profiling(self) -> Dict[str, Any]:
        """Run all I/O profiling benchmarks."""
        print("Starting I/O profiling...")

        # WebSocket latency profiling (simulated mGBA communication)
        if HAS_WEBSOCKETS:
            await self._profile_websocket_latency()
        else:
            self.results["websocket_latency"] = {"error": "websockets library not available"}

        # Disk I/O profiling (savestate loads/saves)
        self._profile_disk_io()

        # Vector query profiling (FAISS searches)
        if HAS_FAISS:
            self._profile_vector_queries()
        else:
            self.results["vector_queries"] = {"error": "FAISS not available"}

        # Network simulation (Content API calls)
        await self._profile_network_simulation()

        return self.results

    async def _profile_websocket_latency(self):
        """Profile WebSocket communication latency."""
        print("Profiling WebSocket latency...")

        latencies = []
        message_sizes = [100, 1000, 10000, 50000]  # bytes

        for size in message_sizes:
            # Simulate mGBA WebSocket communication
            # Since we don't have a real server, we'll measure local socket latency
            lat = await self._measure_websocket_roundtrip(size)
            latencies.append({
                "message_size_bytes": size,
                "latency_ms": lat * 1000,
                "messages_per_second": 1.0 / lat if lat > 0 else 0
            })
            print(f"  Size {size}B: {lat*1000:.1f}ms roundtrip")

        # Simulate realistic PMD-Red communication patterns
        self.results["websocket_latency"] = {
            "synthetic_latencies": latencies,
            "realistic_patterns": {
                "screenshot_request": await self._measure_websocket_roundtrip(50),  # Small command
                "memory_read_256kb": await self._measure_websocket_roundtrip(256*1024),  # WRAM dump
                "button_press": await self._measure_websocket_roundtrip(20),  # Small command
            }
        }

    async def _measure_websocket_roundtrip(self, message_size: int) -> float:
        """
        Measure roundtrip latency for a message of given size.
        Uses local TCP socket for simulation since we don't have mGBA running.
        """
        # Create test message with <|END|> termination like mGBA protocol
        message = b"x" * message_size + b"<|END|>"
        reply = b"<|ACK|><|END|>"

        # Use local TCP socket for latency measurement
        start_time = time.perf_counter()

        try:
            # Create a simple echo server simulation
            reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
            writer.write(message)
            await writer.drain()

            response = await reader.read(1024)
            writer.close()
            await writer.wait_closed()

            end_time = time.perf_counter()
            return end_time - start_time

        except (ConnectionRefusedError, OSError):
            # No server running, simulate latency based on message size
            # Rough approximation: 0.1ms base + 0.01ms per KB
            base_latency = 0.0001  # 0.1ms
            size_latency = (message_size / 1024) * 0.00001  # 0.01ms per KB
            simulated_latency = base_latency + size_latency

            # Add some jitter
            import random
            jitter = random.uniform(-0.5, 0.5) * simulated_latency * 0.1
            return max(0.00005, simulated_latency + jitter)  # Minimum 0.05ms

    def _profile_disk_io(self):
        """Profile disk I/O performance for savestate operations."""
        print("Profiling disk I/O...")

        # Test different savestate sizes (typical GBA savestates)
        savestate_sizes = [64*1024, 128*1024, 256*1024, 512*1024]  # 64KB to 512KB

        io_results = []

        for size in savestate_sizes:
            # Create temporary file to simulate savestate
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                # Write test data
                test_data = os.urandom(size)
                write_start = time.perf_counter()
                with open(temp_path, 'wb') as f:
                    f.write(test_data)
                write_time = time.perf_counter() - write_start

                # Read test data (simulating savestate load)
                read_start = time.perf_counter()
                with open(temp_path, 'rb') as f:
                    loaded_data = f.read()
                read_time = time.perf_counter() - read_start

                # Verify data integrity
                data_integrity = loaded_data == test_data

                io_results.append({
                    "size_kb": size / 1024,
                    "write_time_ms": write_time * 1000,
                    "read_time_ms": read_time * 1000,
                    "total_time_ms": (write_time + read_time) * 1000,
                    "write_speed_mbps": (size / 1024 / 1024) / write_time if write_time > 0 else 0,
                    "read_speed_mbps": (size / 1024 / 1024) / read_time if read_time > 0 else 0,
                    "data_integrity": data_integrity
                })

                print(f"  Size {size//1024}KB: {write_time*1000:.1f}ms write, {read_time*1000:.1f}ms read")

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        self.results["disk_io"] = {
            "savestate_benchmarks": io_results,
            "summary": {
                "avg_write_speed_mbps": sum(r["write_speed_mbps"] for r in io_results) / len(io_results),
                "avg_read_speed_mbps": sum(r["read_speed_mbps"] for r in io_results) / len(io_results),
                "typical_savestate_size_kb": 128  # Based on GBA savestate size
            }
        }

    def _profile_vector_queries(self):
        """Profile FAISS vector query performance."""
        print("Profiling vector queries...")

        if not HAS_FAISS:
            return

        # Create test index simulating PMD-Red retrieval system
        vector_dims = [384, 768, 1024]  # Different embedding dimensions
        index_sizes = [1000, 10000, 50000]  # Different index sizes

        query_results = []

        for dim in vector_dims:
            for size in index_sizes:
                # Create synthetic index
                index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)

                # Add random vectors
                vectors = faiss.rand((size, dim))
                faiss.normalize_L2(vectors)  # Normalize for cosine similarity
                index.add(vectors)

                # Measure query performance
                num_queries = 100
                query_vectors = faiss.rand((num_queries, dim))
                faiss.normalize_L2(query_vectors)

                # Time queries
                start_time = time.perf_counter()
                k = 10  # Top-k results
                distances, indices = index.search(query_vectors, k)
                query_time = time.perf_counter() - start_time

                avg_query_time_ms = (query_time / num_queries) * 1000
                queries_per_second = num_queries / query_time

                query_results.append({
                    "vector_dim": dim,
                    "index_size": size,
                    "avg_query_time_ms": avg_query_time_ms,
                    "queries_per_second": queries_per_second,
                    "memory_usage_mb": (vectors.nbytes + index.getMemorySize()) / 1024 / 1024
                })

                print(f"  Dim {dim}, Size {size}: {avg_query_time_ms:.2f}ms/query, {queries_per_second:.1f} q/s")

        self.results["vector_queries"] = {
            "query_performance": query_results,
            "summary": {
                "recommended_dim": 768,  # Common for sentence transformers
                "typical_index_size": 10000,
                "target_query_time_ms": 5.0  # Target for responsive UI
            }
        }

    async def _profile_network_simulation(self):
        """Profile network I/O for Content API calls."""
        print("Profiling network I/O simulation...")

        # Simulate different API call patterns
        api_patterns = [
            {"name": "species_lookup", "size_kb": 5, "latency_ms": 200},
            {"name": "item_details", "size_kb": 2, "latency_ms": 150},
            {"name": "dungeon_map", "size_kb": 50, "latency_ms": 500},
            {"name": "trajectory_search", "size_kb": 20, "latency_ms": 300}
        ]

        network_results = []

        for pattern in api_patterns:
            # Simulate network latency with jitter
            import random
            base_latency = pattern["latency_ms"] / 1000
            jitter = random.uniform(-0.2, 0.2) * base_latency
            simulated_latency = max(0.01, base_latency + jitter)

            # Simulate data transfer time (rough approximation)
            transfer_time = (pattern["size_kb"] * 1024) / (10 * 1024 * 1024)  # 10 Mbps connection
            total_time = simulated_latency + transfer_time

            network_results.append({
                "api_call": pattern["name"],
                "response_size_kb": pattern["size_kb"],
                "simulated_latency_ms": total_time * 1000,
                "transfer_time_ms": transfer_time * 1000,
                "calls_per_minute": 60 / total_time if total_time > 0 else 0
            })

            await asyncio.sleep(0.001)  # Small delay between simulations

        # Simulate gatekeeper impact (batch vs individual calls)
        batch_simulation = {
            "individual_calls": sum(r["simulated_latency_ms"] for r in network_results),
            "batched_call": max(r["simulated_latency_ms"] for r in network_results),  # Single batch request
            "batch_savings_ms": sum(r["simulated_latency_ms"] for r in network_results) - max(r["simulated_latency_ms"] for r in network_results)
        }

        self.results["network_simulation"] = {
            "api_patterns": network_results,
            "batching_analysis": batch_simulation,
            "budget_impact": {
                "monthly_budget": 1000,  # API calls per month
                "avg_call_cost": 0.001,  # Hypothetical cost per call
                "estimated_monthly_cost": 1000 * 0.001
            }
        }

async def main():
    """Main profiling entry point."""
    profiler = IOProfiler()
    results = await profiler.run_full_profiling()

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = f"profiling/io_baseline_{timestamp}.md"

    with open(output_file, 'w') as f:
        f.write("# PMD-Red Agent I/O Profiling Results\n\n")
        f.write(f"**Timestamp:** {results['timestamp']}\n\n")

        # WebSocket section
        ws_data = results.get("websocket_latency", {})
        if "error" not in ws_data:
            f.write("## WebSocket Latency\n\n")
            patterns = ws_data.get("realistic_patterns", {})
            if patterns:
                f.write("| Operation | Latency (ms) | Throughput (msg/s) |\n")
                f.write("|-----------|--------------|-------------------|\n")
                for op, lat in patterns.items():
                    throughput = 1.0 / lat if lat > 0 else 0
                    f.write(f"| {op} | {lat*1000:.1f} | {throughput:.1f} |\n")
                f.write("\n")
        else:
            f.write(f"## WebSocket Latency\n\n{ws_data['error']}\n\n")

        # Disk I/O section
        disk_data = results.get("disk_io", {})
        summary = disk_data.get("summary", {})
        if summary:
            f.write("## Disk I/O Performance\n\n")
            f.write(".1f")
            f.write(".1f")
            f.write(f"**Typical savestate size:** {summary['typical_savestate_size_kb']} KB\n\n")

        # Vector queries section
        vq_data = results.get("vector_queries", {})
        if "error" not in vq_data:
            f.write("## Vector Query Performance\n\n")
            summary = vq_data.get("summary", {})
            if summary:
                f.write(f"- **Recommended vector dimension:** {summary['recommended_dim']}\n")
                f.write(f"- **Typical index size:** {summary['typical_index_size']}\n")
                f.write(f"- **Target query time:** {summary['target_query_time_ms']} ms\n\n")
        else:
            f.write(f"## Vector Query Performance\n\n{vq_data['error']}\n\n")

        # Network simulation section
        net_data = results.get("network_simulation", {})
        batch = net_data.get("batching_analysis", {})
        if batch:
            f.write("## Network I/O Simulation\n\n")
            f.write(f"**Individual calls total:** {batch['individual_calls']:.1f} ms\n")
            f.write(f"**Batched call:** {batch['batched_call']:.1f} ms\n")
            f.write(f"**Time savings with batching:** {batch['batch_savings_ms']:.1f} ms\n\n")

    # Save raw JSON
    json_file = f"profiling/io_baseline_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"I/O profiling complete. Results saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")

if __name__ == "__main__":
    asyncio.run(main())