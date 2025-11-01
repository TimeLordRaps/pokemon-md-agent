#!/usr/bin/env python3
"""Dry-run version of test progression for infrastructure validation.

Runs the same test progression but without requiring mGBA connection,
useful for validating skill discovery, bootstrap, and dashboard output
without real emulator access.
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
        logging.FileHandler('test_runs_dryrun.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


async def run_test_progression_dryrun(max_steps: int, run_name: str) -> dict:
    """Execute a dry-run test without mGBA.

    Args:
        max_steps: Maximum steps to simulate
        run_name: Name for this test run

    Returns:
        Dictionary with run results
    """
    from src.runners.live_armada import ArmadaConfig, LiveArmadaRunner

    logger.info(f"Starting dry-run test: {run_name} ({max_steps} steps)")

    # Configuration with dry_run=True
    dashboard_dir = Path(f"docs/test_runs/{run_name}")

    config = ArmadaConfig(
        rom=Path("/fake/rom.gba"),
        save=Path("/fake/save.sav"),
        lua=Path("/fake/lua.lua"),
        mgba_exe=Path("/fake/mgba.exe"),
        host="localhost",
        port=8888,
        capture_fps=6.0,
        dashboard_dir=dashboard_dir,
        trace_jsonl=dashboard_dir / "traces" / "latest.jsonl",
        dry_run=True,  # DRY RUN MODE
    )

    runner = LiveArmadaRunner(config=config)
    logger.info(f"Run ID: {runner.run_id}")
    logger.info(f"Target: {max_steps} steps")

    # Run
    start_time = time.time()
    result_code = await runner.run(max_steps=max_steps)
    elapsed_time = time.time() - start_time

    # Collect results
    results = {
        "success": result_code == 0,
        "run_id": runner.run_id,
        "elapsed_seconds": elapsed_time,
        "actual_steps": runner.step_count,
        "discovered_skills": len(runner.discovered_skills),
        "total_skills": len(runner.skill_manager.list_skills()),
        "dashboard_dir": str(dashboard_dir),
        "run_name": run_name
    }

    logger.info(f"Dry-run test {run_name} completed:")
    logger.info(f"  Duration: {elapsed_time:.1f}s")
    logger.info(f"  Steps: {runner.step_count} / {max_steps}")
    logger.info(f"  Skills discovered this run: {len(runner.discovered_skills)}")
    logger.info(f"  Total skills available: {len(runner.skill_manager.list_skills())}")
    logger.info(f"  Dashboard: {dashboard_dir}")

    # Verify directories were created
    assert dashboard_dir.exists(), f"Dashboard dir not created: {dashboard_dir}"
    assert (dashboard_dir / "keyframes").exists(), "Keyframes dir not created"
    assert (dashboard_dir / "traces").exists(), "Traces dir not created"
    logger.info("✓ All output directories created successfully")

    return results


async def run_progression_dryrun() -> None:
    """Execute the dry-run test progression."""
    logger.info("="*60)
    logger.info("POKEMON MD AGENT - DRY-RUN TEST PROGRESSION")
    logger.info("="*60)

    # Dry-run tests with fewer steps
    test_runs = [
        (10, "10step_bootstrap_test"),
        (20, "20step_skill_discovery"),
        (30, "30step_learning_accumulation"),
        (50, "50step_convergence"),
    ]

    all_results = []

    for max_steps, run_name in test_runs:
        try:
            logger.info("\n" + "="*60)
            logger.info(f"Starting: {run_name}")
            logger.info("="*60)

            result = await run_test_progression_dryrun(max_steps, run_name)
            all_results.append(result)

            logger.info(f"✓ {run_name} completed successfully")

            # Brief pause between runs
            if run_name != "50step_convergence":
                logger.info("Pausing before next run...")
                await asyncio.sleep(1)

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
    logger.info("DRY-RUN TEST PROGRESSION SUMMARY")
    logger.info("="*60)

    for result in all_results:
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        logger.info(f"{status} | {result.get('run_name')}")
        if result.get("success"):
            logger.info(f"        Steps: {result.get('actual_steps')} | Skills: {result.get('discovered_skills')} discovered, {result.get('total_skills')} total")
        else:
            logger.info(f"        Error: {result.get('error')}")

    logger.info("="*60)

    if all(r.get("success", False) for r in all_results):
        logger.info("✓ ALL DRY-RUN TESTS PASSED")
        logger.info("Infrastructure is ready for real test runs with mGBA")
    else:
        logger.error("✗ SOME DRY-RUN TESTS FAILED")

    logger.info("="*60)

    return all_results


async def main() -> int:
    """Main entry point."""
    try:
        results = await run_progression_dryrun()

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
