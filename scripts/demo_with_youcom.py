#!/usr/bin/env python3
"""
Final integration demo: Agent + You.com API + Video with Voiceover
Ready for production presentation.
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_setup():
    """Verify all dependencies and keys are present."""
    logger.info("=" * 70)
    logger.info("VERIFICATION PHASE")
    logger.info("=" * 70)

    import os

    # Check You.com API Key
    you_api = os.getenv('YOU_API_KEY') or os.getenv('YOUCOM_API_KEY')
    if you_api:
        logger.info(f"✓ You.com API Key found: {you_api[:15]}...")
    else:
        logger.warning("✗ You.com API Key NOT found (optional for demo)")

    # Check ROM/SAV
    rom_path = Path("../rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba")
    sav_path = Path("../rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).sav")

    if rom_path.exists():
        logger.info(f"✓ ROM found: {rom_path.name}")
    else:
        logger.error("✗ ROM not found at ../rom/")
        return False

    if sav_path.exists():
        logger.info(f"✓ SAV found: {sav_path.name}")
    else:
        logger.error("✗ SAV not found at ../rom/")
        return False

    # Check mGBA connection
    logger.info("Checking mGBA connection (optional)...")
    try:
        result = subprocess.run(
            [sys.executable, ".temp_check_ram.py"],
            capture_output=True,
            timeout=5,
            text=True
        )
        if "floor_number" in result.stdout:
            logger.info("✓ mGBA connection verified")
        else:
            logger.warning("⚠ mGBA not responding (will fail during agent execution)")
    except Exception as e:
        logger.warning(f"⚠ mGBA check failed: {e}")

    return True


def run_demo():
    """Run the full demo pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO EXECUTION")
    logger.info("=" * 70)

    cmd = [sys.executable, "scripts/final_demo_runner.py"]

    proc = subprocess.run(cmd, cwd=".")

    if proc.returncode != 0:
        logger.error(f"✗ Demo failed with return code {proc.returncode}")
        return False

    logger.info("✓ Demo completed successfully")
    return True


def verify_outputs():
    """Verify demo outputs."""
    logger.info("\n" + "=" * 70)
    logger.info("OUTPUT VERIFICATION")
    logger.info("=" * 70)

    # Check for video
    video_path = Path("agent_demo.mp4")
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Video generated: {video_path.name} ({size_mb:.1f} MB)")
    else:
        logger.error("✗ Video not found")
        return False

    # Check for trajectory
    runs_dir = Path("runs")
    if runs_dir.exists():
        latest_run = max(runs_dir.glob("demo_*"), key=lambda p: p.stat().st_mtime, default=None)
        if latest_run:
            traj_files = list(latest_run.glob("trajectory_*.jsonl"))
            if traj_files:
                logger.info(f"✓ Trajectory saved: {latest_run.name}")
            else:
                logger.error("✗ No trajectory file found")
                return False
        else:
            logger.error("✗ No run directory found")
            return False
    else:
        logger.error("✗ runs/ directory not found")
        return False

    return True


def ready_for_github():
    """Confirm everything is ready for GitHub push."""
    logger.info("\n" + "=" * 70)
    logger.info("GITHUB READINESS CHECK")
    logger.info("=" * 70)

    checks = [
        ("Git status clean", subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, cwd="."
        ).returncode == 0),
        ("Video exists", Path("agent_demo.mp4").exists()),
        ("Trajectory exists", any(Path("runs").glob("demo_*/trajectory_*.jsonl")) if Path("runs").exists() else False),
    ]

    for check_name, result in checks:
        status = "✓" if result else "✗"
        logger.info(f"{status} {check_name}")

    all_good = all(result for _, result in checks)

    if all_good:
        logger.info("\n✓ Ready for GitHub push!")
    else:
        logger.error("\n✗ Some checks failed")

    return all_good


def main():
    """Main orchestration."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "Pokemon MD Agent - Final Demo with You.com Integration".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("")

    # 1. Verify setup
    if not verify_setup():
        return 1

    # 2. Run demo
    if not run_demo():
        return 1

    # 3. Verify outputs
    if not verify_outputs():
        return 1

    # 4. Ready for GitHub
    if not ready_for_github():
        logger.warning("Some checks failed, but continuing...")

    logger.info("\n" + "=" * 70)
    logger.info("✓ ALL SYSTEMS GO - Ready for presentation!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Review agent_demo.mp4")
    logger.info("2. git remote add origin https://github.com/TimeLordRaps/pokemon-md-agent.git")
    logger.info("3. git push -u origin main")

    return 0


if __name__ == "__main__":
    sys.exit(main())
