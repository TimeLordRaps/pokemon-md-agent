"""
Final demo coordinator - runs agent, validates output, generates video montage
"""
import subprocess
import sys
import time
from pathlib import Path


def run_agent_demo(max_steps: int = 50) -> bool:
    """Run agent demo script."""
    print("=" * 60)
    print("PHASE 1: AGENT AUTONOMOUS DEMO")
    print("=" * 60)
    print(f"Starting agent demo ({max_steps} steps)...")

    # Run demo
    cmd = [sys.executable, "demo_agent.py"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).parent.parent)  # Pokemon-md-agent root
    )

    # Monitor for errors
    start = time.time()
    while proc.poll() is None:
        elapsed = time.time() - start
        if elapsed > 600:  # 10 min timeout
            proc.kill()
            print("✗ Demo timed out after 10 minutes")
            return False
        time.sleep(2)

    stdout, stderr = proc.communicate()
    elapsed = time.time() - start

    print(f"\nAgent completed in {elapsed:.1f}s")
    if proc.returncode != 0:
        print(f"✗ Agent failed with return code {proc.returncode}")
        if stderr:
            print("STDERR:", stderr[:500])
        return False

    print("✓ Agent demo completed successfully")
    return True


def validate_outputs() -> Path:
    """Validate agent outputs and return latest run directory."""
    print("\n" + "=" * 60)
    print("PHASE 2: VALIDATION")
    print("=" * 60)

    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("✗ 'runs' directory not found")
        return None

    # Find latest run
    all_runs = list(runs_dir.glob("demo_*")) + list(runs_dir.glob("run_*"))
    if not all_runs:
        print("✗ No run directories found in 'runs'")
        return None

    latest_run = max(all_runs, key=lambda p: p.stat().st_mtime)
    print(f"✓ Found latest run: {latest_run}")

    # Check for trajectory file
    traj_files = list(latest_run.glob("trajectory_*.jsonl"))
    if not traj_files:
        print("✗ No trajectory file found")
        return None

    traj_file = traj_files[0]
    with open(traj_file) as f:
        frame_count = sum(1 for _ in f)

    print(f"✓ Trajectory: {frame_count} frames logged")
    return latest_run


def generate_video(run_dir: Path) -> bool:
    """Generate montage video from agent run."""
    print("\n" + "=" * 60)
    print("PHASE 3: VIDEO GENERATION")
    print("=" * 60)

    script_path = Path(__file__).parent / "generate_montage_video.py"
    output_path = Path("docs/assets/agent_demo.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--run-dir", str(run_dir),
        "--output", str(output_path),
        "--fps", "30",
        "--duration", "150",  # Target 120-180s
        "--voiceover",
        "--voiceover-voice", "af_heart",
        "--voiceover-lang", "a",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).parent.parent)  # Pokemon-md-agent root
    )

    stdout, stderr = proc.communicate()
    print(stdout)

    if proc.returncode != 0:
        print(f"✗ Video generation failed")
        if stderr:
            print("STDERR:", stderr[:500])
        return False

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Video generated: {output_path} ({size_mb:.1f} MB)")

        # Validate video
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(output_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                duration = float(data.get("format", {}).get("duration", 0))
                print(f"  Duration: {duration:.1f}s (target: 120-180s)")
                if duration < 120:
                    print("  ⚠️  Duration shorter than 120s - may need more frames")
                elif duration > 180:
                    print("  ⚠️  Duration longer than 180s - consider increasing sampling")
            else:
                print("  ⚠️  Could not validate video with ffprobe")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            print("  ⚠️  ffprobe not available for validation")

        return True

    return False


def run_final_demo():
    """Orchestrate full demo pipeline."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Pokemon MD Agent — Final 3-Minute Demo  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Phase 1: Run agent
    if not run_agent_demo(max_steps=50):
        print("\n✗ DEMO FAILED at agent execution")
        return False

    # Phase 2: Validate outputs
    run_dir = validate_outputs()
    if not run_dir:
        print("\n✗ DEMO FAILED at validation")
        return False

    # Phase 3: Generate video
    if not generate_video(run_dir):
        print("\n✗ DEMO FAILED at video generation")
        return False

    # Success!
    print("\n" + "=" * 60)
    print("✓ DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  • Run directory: {run_dir}")
    print(f"  • Video montage: agent_demo.mp4")
    print(f"\nReady for presentation/recording!")

    return True


if __name__ == "__main__":
    success = run_final_demo()
    sys.exit(0 if success else 1)
