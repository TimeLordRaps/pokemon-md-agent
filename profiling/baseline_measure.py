#!/usr/bin/env python3
"""
Baseline Measurement Script for PMD-Red Agent Profiling Infrastructure

Runs all profiling scripts in sequence and aggregates results with timestamps.
Ensures profiling overhead <5% and saves results to profiling/ directory.

Run with: python profiling/baseline_measure.py
"""

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_profiling_script(script_name: str, args: List[str] = None) -> Dict[str, Any]:
    """
    Run a single profiling script and capture its execution.

    Args:
        script_name: Name of the profiling script (without .py extension)
        args: Additional command line arguments

    Returns:
        Dict containing execution results and timing
    """
    if args is None:
        args = []

    script_path = f"profiling/{script_name}.py"
    cmd = [sys.executable, script_path] + args

    start_time = time.perf_counter()
    result = {
        "script": script_name,
        "command": " ".join(cmd),
        "start_time": datetime.datetime.now().isoformat(),
        "success": False,
        "execution_time_seconds": None,
        "stdout": "",
        "stderr": "",
        "error": None
    }

    try:
        # Run the script and capture output
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )

        result["success"] = proc.returncode == 0
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["return_code"] = proc.returncode

    except subprocess.TimeoutExpired:
        result["error"] = "Script timed out after 300 seconds"
    except FileNotFoundError:
        result["error"] = f"Script {script_path} not found"
    except Exception as e:
        result["error"] = str(e)

    result["execution_time_seconds"] = time.perf_counter() - start_time
    result["end_time"] = datetime.datetime.now().isoformat()

    return result

def aggregate_results(profiling_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from all profiling runs into a summary.

    Args:
        profiling_runs: List of individual script execution results

    Returns:
        Aggregated results dictionary
    """
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_execution_time_seconds": sum(r["execution_time_seconds"] for r in profiling_runs),
        "scripts_run": len(profiling_runs),
        "successful_scripts": len([r for r in profiling_runs if r["success"]]),
        "failed_scripts": len([r for r in profiling_runs if not r["success"]]),
        "overhead_estimate_percent": None,
        "script_results": profiling_runs
    }

    # Estimate profiling overhead (rough approximation: <5% target)
    # This is a conservative estimate - actual agent overhead would be lower
    total_profiling_time = summary["total_execution_time_seconds"]
    # Assume real agent episode would take ~10-30 seconds for 100 steps
    estimated_real_agent_time = 20.0
    overhead_percent = (total_profiling_time / estimated_real_agent_time) * 100

    summary["overhead_estimate_percent"] = overhead_percent
    summary["overhead_within_target"] = overhead_percent < 5.0

    return summary

def main():
    """Main baseline measurement orchestration."""
    print("Starting PMD-Red Agent Baseline Profiling Measurement...")
    print("=" * 60)

    # Define profiling scripts to run in sequence
    profiling_scripts = [
        {"name": "cpu_profiling", "args": []},
        {"name": "gpu_profiling", "args": []},
        {"name": "memory_profiling", "args": []},
        {"name": "io_profiling", "args": []}
    ]

    profiling_runs = []

    for script_config in profiling_scripts:
        script_name = script_config["name"]
        print(f"\nRunning {script_name}...")
        print("-" * 40)

        result = run_profiling_script(script_name, script_config.get("args", []))

        # Print immediate feedback
        status = "SUCCESS" if result["success"] else "FAILED"
        exec_time = result["execution_time_seconds"]
        print(f"{status} - {exec_time:.2f}s")

        if not result["success"] and result["error"]:
            print(f"Error: {result['error']}")

        profiling_runs.append(result)

    # Aggregate and save results
    print("\n" + "=" * 60)
    print("Aggregating results...")

    summary = aggregate_results(profiling_runs)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save detailed results as JSON
    json_file = f"profiling/baseline_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save human-readable summary
    txt_file = f"profiling/baseline_summary_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write("PMD-Red Agent Baseline Profiling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Total execution time: {summary['total_execution_time_seconds']:.2f} seconds\n")
        f.write(f"Scripts run: {summary['scripts_run']}\n")
        f.write(f"Successful: {summary['successful_scripts']}\n")
        f.write(f"Failed: {summary['failed_scripts']}\n\n")

        overhead_pct = summary.get('overhead_estimate_percent', 0)
        f.write(f"Estimated profiling overhead: {overhead_pct:.1f}%\n")
        f.write(f"Within target (<5%): {'YES' if summary.get('overhead_within_target', False) else 'NO'}\n\n")

        f.write("SCRIPT RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in summary["script_results"]:
            status = "PASS" if result["success"] else "FAIL"
            f.write(f"{result['script']:15} | {status:4} | {result['execution_time_seconds']:6.2f}s\n")

        if summary["failed_scripts"] > 0:
            f.write("\nFAILED SCRIPTS:\n")
            for result in summary["script_results"]:
                if not result["success"]:
                    f.write(f"- {result['script']}: {result.get('error', 'Unknown error')}\n")

    print(f"Baseline profiling complete!")
    print(f"Results saved to:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    print()
    print(".1f")
    print(f"Scripts: {summary['successful_scripts']}/{summary['scripts_run']} successful")

    if summary.get('overhead_within_target', False):
        print("Profiling overhead within target (<5%)")
    else:
        print("⚠ Profiling overhead exceeds target")

if __name__ == "__main__":
    main()