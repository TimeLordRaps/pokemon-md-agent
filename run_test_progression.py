#!/usr/bin/env python3
"""Test run progression script for skill learning and bootstrap validation.

Executes iterative test runs with increasing durations:
- 15 minutes (test bootstrap infrastructure)
- 30 minutes (observe skill discovery)
- 1 hour (measure learning accumulation)
- 5 hours (full convergence analysis)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s: %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_runs.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


async def run_test_progression(max_duration_seconds: int, run_name: str) -> dict:
    """Execute a single test run with duration limit.

    Args:
        max_duration_seconds: Maximum runtime in seconds
        run_name: Name for this test run

    Returns:
        Dictionary with run results
    """
    from src.runners.live_armada import ArmadaConfig, LiveArmadaRunner
    import os

    logger.info(f"Starting test run: {run_name} ({max_duration_seconds}s max)")

    # Configuration
    rom_path = Path(os.getenv("PMD_ROM", "config/pokemon_md_red.gba"))
    save_path = Path(os.getenv("PMD_SAVE", "config/save_files/game_start.sav"))
    lua_path = Path(os.getenv("MGBALUA", "scripts/mGBASocketServer.lua"))
    mgba_exe = Path(os.getenv("MGBAX", "C:/Program Files/mGBA/mGBA.exe"))

    dashboard_dir = Path(f"docs/test_runs/{run_name}")

    config = ArmadaConfig(
        rom=rom_path,
        save=save_path,
        lua=lua_path,
        mgba_exe=mgba_exe,
        host="localhost",
        port=8888,
        capture_fps=6.0,
        dashboard_dir=dashboard_dir,
        trace_jsonl=dashboard_dir / "traces" / "latest.jsonl",
        dry_run=False,  # Real gameplay
    )

    if not config.validate():
        logger.error(f"Configuration validation failed for {run_name}")
        return {"success": False, "error": "Config validation failed"}

    runner = LiveArmadaRunner(config=config)
    logger.info(f"Run ID: {runner.run_id}")

    # Calculate max steps based on FPS and duration
    max_steps = int(max_duration_seconds * config.capture_fps)
    logger.info(f"Target: {max_steps} steps ({max_duration_seconds}s @ {config.capture_fps} FPS)")

    # Run with time limit
    start_time = time.time()
    result_code = await runner.run(max_steps=max_steps)
    elapsed_time = time.time() - start_time

    # Collect results
    results = {
        "success": result_code == 0,
        "run_id": runner.run_id,
        "elapsed_seconds": elapsed_time,
        "target_steps": max_steps,
        "actual_steps": runner.step_count,
        "discovered_skills": len(runner.discovered_skills),
        "total_skills": len(runner.skill_manager.list_skills()),
        "dashboard_dir": str(dashboard_dir),
        "run_name": run_name
    }

    logger.info(f"Test run {run_name} completed:")
    logger.info(f"  Duration: {elapsed_time:.1f}s / {max_duration_seconds}s")
    logger.info(f"  Steps: {runner.step_count} / {max_steps}")
    logger.info(f"  Skills discovered this run: {len(runner.discovered_skills)}")
    logger.info(f"  Total skills available: {len(runner.skill_manager.list_skills())}")
    logger.info(f"  Dashboard: {dashboard_dir}")

    return results


async def run_progression() -> None:
    """Execute the full test progression."""
    logger.info("="*60)
    logger.info("POKEMON MD AGENT - TEST PROGRESSION")
    logger.info("="*60)

    # Test runs: (duration_seconds, name)
    test_runs = [
        (15 * 60, "15min_bootstrap_test"),    # 15 minutes
        (30 * 60, "30min_skill_discovery"),   # 30 minutes
        (60 * 60, "1hr_learning_accumulation"),  # 1 hour
        (5 * 60 * 60, "5hr_convergence"),    # 5 hours
    ]

    all_results = []

    for duration_secs, run_name in test_runs:
        try:
            logger.info("\n" + "="*60)
            logger.info(f"Starting: {run_name}")
            logger.info("="*60)

            result = await run_test_progression(duration_secs, run_name)
            all_results.append(result)

            logger.info(f"✓ {run_name} completed successfully")

            # Brief pause between runs
            if run_name != "5hr_convergence":
                logger.info("Pausing before next run...")
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"✗ {run_name} failed: {e}", exc_info=True)
            result = {
                "success": False,
                "run_name": run_name,
                "error": str(e)
            }
            all_results.append(result)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST PROGRESSION SUMMARY")
    logger.info("="*60)

    for result in all_results:
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        logger.info(f"{status} | {result.get('run_name')}")
        if result.get("success"):
            logger.info(f"        Steps: {result.get('actual_steps')} | Skills: {result.get('discovered_skills')} discovered, {result.get('total_skills')} total")
        else:
            logger.info(f"        Error: {result.get('error')}")

    logger.info("="*60)
    logger.info("ANALYSIS RESULTS SAVED TO:")
    for result in all_results:
        if result.get("success"):
            logger.info(f"  - {result.get('dashboard_dir')}")
    logger.info("="*60)

    return all_results


async def main() -> int:
    """Main entry point."""
    try:
        results = await run_progression()

        # Check if all runs succeeded
        all_success = all(r.get("success", False) for r in results)
        return 0 if all_success else 1

    except KeyboardInterrupt:
        logger.info("Test progression interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
