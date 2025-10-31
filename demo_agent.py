#!/usr/bin/env python3
"""Demo script for Pokemon MD Agent Core.

Runs the agent for a short demo period to show autonomous gameplay.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.agent_core import AgentCore

async def main():
    """Run agent demo."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Pokemon MD Agent Demo")
    print("=" * 40)

    try:
        # Create agent (will attempt mGBA connection)
        print("Initializing agent...")
        agent = AgentCore(
            objective="Navigate to stairs and progress through dungeon",
            test_mode=False,  # Real mGBA connection
            enable_retrieval=False  # Skip retrieval for demo
        )

        print("Agent initialized successfully!")
        print("Starting autonomous gameplay demo...")

        # Run for 30 seconds (about 30-60 steps depending on timing)
        await agent.run(max_steps=50)

        print("\nDemo completed successfully!")
        print("Agent ran for 50 steps autonomously.")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("Make sure mGBA is running with Pokemon Mystery Dungeon loaded.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))