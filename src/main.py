"""Main entry point for Pokemon MD autonomous agent."""

import asyncio
import logging
import sys
from pathlib import Path

from .agent.agent_core import PokemonMDAgent, AgentConfig

logger = logging.getLogger(__name__)


async def run_agent():
    """Run the Pokemon MD agent."""
    # Configuration
    rom_path = Path("../../rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba")
    save_dir = Path("../../saves")
    
    config = AgentConfig(
        screenshot_interval=1.0,
        memory_poll_interval=0.1,
        decision_interval=0.5,
        max_runtime_hours=1.0,  # Short test run
        enable_4up_capture=True,
        enable_trajectory_logging=True,
        enable_stuck_detection=False,  # Disable for now
    )
    
    # Create and run agent
    agent = PokemonMDAgent(rom_path, save_dir, config)
    
    try:
        print("Starting Pokemon MD Agent...")
        await agent.run()
        print("Agent completed successfully")
    except KeyboardInterrupt:
        print("Agent interrupted by user")
        agent.stop()
    except Exception as e:
        print(f"Agent failed: {e}")
        agent.stop()
        raise


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from .agent.agent_core import PokemonMDAgent
        print("✓ PokemonMDAgent imported successfully")
        
        from .agent.model_router import ModelRouter
        print("✓ ModelRouter imported successfully")
        
        from .agent.memory_manager import MemoryManager
        print("✓ MemoryManager imported successfully")
        
        from ..environment.mgba_controller import MGBAController
        print("✓ MGBAController imported successfully")
        
        from ..environment.ram_decoders import RAMDecoder
        print("✓ RAMDecoder imported successfully")
        
        from ..retrieval.trajectory_logger import TrajectoryLogger
        print("✓ TrajectoryLogger imported successfully")
        
        from ..models.world_model import WorldModel
        print("✓ WorldModel imported successfully")
        
        from ..vision.quad_capture import QuadCapture
        print("✓ QuadCapture imported successfully")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    print("Pokemon MD Autonomous Agent - Main Entry Point")
    print("=" * 50)
    
    if not test_imports():
        print("\nImport test failed")
        return 1
    
    # Check if ROM exists
    rom_path = Path("../../rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba")
    if not rom_path.exists():
        print(f"ROM not found at {rom_path}")
        print("Please ensure the ROM file is in the correct location")
        return 1
    
    # Run agent
    try:
        asyncio.run(run_agent())
        return 0
    except KeyboardInterrupt:
        print("Agent execution interrupted by user")
        return 0
    except Exception as e:
        print(f"Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
