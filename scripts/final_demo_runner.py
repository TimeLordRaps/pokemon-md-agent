"""
Final demo coordinator - runs 50-step sequence with live monitoring
"""
import subprocess
import time
from pathlib import Path

def run_final_demo():
    print("=== FINAL 50-STEP DEMO (T-minus 90 min) ===")

    # Start mGBA if not running
    # (assume user already started it per instructions)

    # Run demo with timeout
    proc = subprocess.Popen(
        ["python", "demo_agent.py", "--max-steps", "50"],
        cwd="pokemon-md-agent",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Monitor for errors
    start = time.time()
    while proc.poll() is None:
        if time.time() - start > 600:  # 10 min timeout
            proc.kill()
            print("✗ Demo timed out after 10 minutes")
            return False
        time.sleep(5)

    # Validate outputs
    runs_dir = Path("runs")
    latest_run = max(runs_dir.glob("demo_*"), key=lambda p: p.stat().st_mtime)
    traj_file = next(latest_run.glob("trajectory_*.jsonl"))

    with open(traj_file) as f:
        lines = f.readlines()
        if len(lines) < 50:
            print(f"✗ Only {len(lines)} steps completed (expected 50)")
            return False

    print(f"✓ Demo completed: {len(lines)} steps logged")
    print(f"✓ Outputs in: {latest_run}")
    return True

if __name__ == "__main__":
    success = run_final_demo()
    exit(0 if success else 1)