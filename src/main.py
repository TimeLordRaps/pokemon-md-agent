"""Main entry point for Pokemon MD autonomous agent."""

import asyncio
import logging
import sys
import time
from pathlib import Path

from .agent.agent_core import PokemonMDAgent, AgentConfig

logger = logging.getLogger(__name__)


async def run_demo_agent():
    """Run the Pokemon MD agent demo with 3 turns."""
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
        enable_stuck_detection=True,  # Enable for demo
    )

    # Create agent
    agent = PokemonMDAgent(rom_path, save_dir, config)

    try:
        print("Starting Pokemon MD Agent Demo...")
        print("Connecting to mGBA on port 8888...")

        # Initialize
        await agent._initialize()
        print("✓ Connected to mGBA (port 8888)")
        print("✓ Loaded save state")

        # Demo loop - 3 turns
        decisions = [
            {"action": "move", "direction": "north", "description": "Move North"},
            {"action": "use_item", "item": "apple", "description": "Use Apple"},
            {"action": "interact", "description": "Take Stairs"}
        ]

        for turn_num, decision in enumerate(decisions, 1):
            print(f"\n--- Turn {turn_num} ---")

            # Gather current context
            context = await agent._gather_decision_context()

            # Render ASCII grid
            if hasattr(agent, 'ram_decoder') and agent.ram_decoder:
                player_state = agent.ram_decoder.get_player_state()
                entities = agent.ram_decoder.get_entities()
                items = agent.ram_decoder.get_items()
                map_data = agent.ram_decoder.get_map_data()

                if player_state and entities is not None and items is not None:
                    # Create mock grid frame for ASCII rendering
                    from ..vision.grid_parser import GridFrame, GridCell, TileType
                    tiles = [[GridCell(tile_type=TileType.FLOOR, visible=True) for _ in range(32)] for _ in range(32)]
                    grid = GridFrame(width=32, height=32, tiles=tiles, tile_size_px=8, camera_tile_origin=(0, 0), view_rect_tiles=(0, 0, 32, 32), timestamp=time.time())

                    # Create mock RAM snapshot
                    from ..environment.ram_decoders import RAMSnapshot, PartyStatus, MapData, PlayerState
                    snapshot = RAMSnapshot(
                        player_state=PlayerState(**player_state) if player_state else PlayerState(0, 0, 0, 0, 0, 0, 0),
                        entities=entities or [],
                        items=items or [],
                        map_data=MapData(0, 0, 0, 0, -1, -1),
                        party_status=PartyStatus(
                            leader_hp=100,
                            leader_hp_max=100,
                            leader_belly=50,
                            partner_hp=100,
                            partner_hp_max=100,
                            partner_belly=50
                        ),
                        timestamp=time.time()
                    )

                    # Render ASCII grid
                    from ..vision.ascii_renderer import ASCIIRenderer
                    renderer = ASCIIRenderer()
                    ascii_grid = renderer.render_environment_with_entities(grid, snapshot)
                    print(ascii_grid)

            # Execute decision
            print(f"Decision: {decision['description']}")
            await agent._execute_decision(decision)

            # Check stuckness
            if agent.config.enable_stuck_detection and hasattr(agent, 'stuck_detector'):
                # Create mock embedding for stuckness check
                import numpy as np
                mock_embedding = np.random.normal(0, 0.1, 1024)
                analysis = agent.stuck_detector.analyze(
                    current_embedding=mock_embedding,
                    current_position=(player_state.player_tile_x, player_state.player_tile_y) if player_state else None,
                    current_action=decision['action'],
                    current_time=time.time()
                )
                print(f"Stuckness: {analysis.status.value.replace('_', ' ').title()} ({len(agent.stuck_detector.snapshots)} unique states)")

            # Wait between turns
            await asyncio.sleep(1.0)

        print("\nDemo completed successfully!")
        print("✓ Made 3 decisions in sequence")
        print("✓ Rendered ASCII grids correctly")
        print("✓ Logged stuckness detector state")

    except KeyboardInterrupt:
        print("Demo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent._cleanup()


async def run_agent():
    """Run the full Pokemon MD agent."""
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


def run_demo():
    """Run the demo version."""
    logging.basicConfig(level=logging.INFO)

    print("Pokemon MD Autonomous Agent - Demo Mode")
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

    # Run demo
    try:
        asyncio.run(run_demo_agent())
        return 0
    except KeyboardInterrupt:
        print("Demo execution interrupted by user")
        return 0
    except Exception as e:
        print(f"Demo execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
    # Check for demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        exit_code = run_demo()
    else:
        exit_code = main()
    sys.exit(exit_code)
