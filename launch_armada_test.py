#!/usr/bin/env python3
"""Launch armada test progression with idempotent startup verification."""

import os
import asyncio
import sys
import subprocess
from pathlib import Path

# Set environment variables for mGBA integration
os.environ['PMD_ROM'] = r'C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba'
os.environ['PMD_SAVE'] = r'C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).sav'
os.environ['MGBALUA'] = r'C:\Homework\agent_hackathon\pokemon-md-agent\src\mgba-harness\mgba-http\mGBASocketServer.lua'
os.environ['MGBAX'] = r'C:\Program Files\mGBA\mGBA.exe'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', r'E:\transformer_models')

sys.path.insert(0, '.')

def run_startup_verification():
    """Run idempotent setup verification before test starts."""
    print("\n" + "=" * 70)
    print("STARTUP VERIFICATION")
    print("=" * 70)

    verify_script = Path(__file__).parent / "verify_setup.py"

    if not verify_script.exists():
        print("ERROR: verify_setup.py not found!")
        return False

    # Run verification script
    result = subprocess.run(
        [sys.executable, str(verify_script)],
        cwd=Path(__file__).parent,
        capture_output=False
    )

    print("=" * 70 + "\n")

    return result.returncode == 0

from run_test_progression import main

if __name__ == '__main__':
    # Run verification first
    if not run_startup_verification():
        print("FATAL: Setup verification failed. Aborting test.")
        sys.exit(1)

    # Proceed with test
    print("Starting 6-model armada test progression...\n")
    sys.exit(asyncio.run(main()))
