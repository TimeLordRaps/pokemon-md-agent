#!/usr/bin/env python3
"""Test progression with GPU support via agent-hackathon env."""

import asyncio
import logging
import sys
import time
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s: %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_runs_gpu.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

async def run_single_test(max_steps: int, run_name: str) -> dict:
    """Execute a single test run."""
    from src.runners.live_armada import ArmadaConfig, LiveArmadaRunner

    logger.info(f"Starting test: {run_name} ({max_steps} steps)")

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
        dry_run=False,
    )

    if not config.validate():
        logger.error(f"Configuration validation failed")
        return {"success": False, "error": "Config validation failed"}

    runner = LiveArmadaRunner(config=config)
    logger.info(f"Run ID: {runner.run_id}")

    start_time = time.time()
    result_code = await runner.run(max_steps=max_steps)
    elapsed_time = time.time() - start_time

    return {
        "success": result_code == 0,
        "run_id": runner.run_id,
        "elapsed_seconds": elapsed_time,
        "actual_steps": runner.step_count,
        "discovered_skills": len(runner.discovered_skills),
        "total_skills": len(runner.skill_manager.list_skills()),
        "run_name": run_name,
    }

async def main() -> int:
    """Main entry point."""
    try:
        logger.info("="*70)
        logger.info("POKEMON MD AGENT - GPU TEST PROGRESSION (agent-hackathon env)")
        logger.info("="*70)

        # Single test: 30 steps to validate game progression
        logger.info("\nRunning validation test: 30 steps")
        result = await run_single_test(30, "gpu_validation_test_30steps")

        if result.get("success"):
            logger.info(f"✓ Test passed: {result['actual_steps']} steps executed")
            logger.info(f"  Output: {result}")
            return 0
        else:
            logger.error(f"✗ Test failed: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
