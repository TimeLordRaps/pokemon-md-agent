"""Quick start example for Pokemon MD autonomous agent."""

import logging
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import agent modules
from src.agent.model_router import ModelRouter, ModelSize
from src.agent.memory_manager import MemoryManager, MemoryAllocation
from src.agent.qwen_controller import QwenController, AgentState, InferenceResult
from src.environment.fps_adjuster import FPSAdjuster
from src.embeddings.temporal_silo import TemporalSiloManager
from src.embeddings.vector_store import VectorStore
from src.orchestrator.runtime import build_router_runtime


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_model_routing():
    """Demonstrate model routing functionality."""
    print("\n=== Model Routing Demo ===")
    
    router = ModelRouter(
        confidence_2b_threshold=0.8,
        confidence_4b_threshold=0.6,
        stuck_escalation_threshold=5
    )
    
    # Test different scenarios
    scenarios = [
        {"confidence": 0.9, "stuck_counter": 0},
        {"confidence": 0.7, "stuck_counter": 0},
        {"confidence": 0.5, "stuck_counter": 0},
        {"confidence": 0.9, "stuck_counter": 5},
    ]
    
    for scenario in scenarios:
        decision = router.select_model(**scenario)
        print(
            f"Confidence: {scenario['confidence']}, "
            f"Stuck: {scenario['stuck_counter']} → "
            f"Model: {decision.selected_model.value} "
            f"({decision.reasoning})"
        )


def demonstrate_memory_management():
    """Demonstrate memory management functionality."""
    print("\n=== Memory Management Demo ===")
    
    # Create memory manager with custom allocation
    allocation = MemoryAllocation(
        last_5_minutes=0.7,
        last_30_minutes=0.2,
        active_missions=0.1
    )
    
    memory_mgr = MemoryManager(total_context_budget=256_000, allocation=allocation)
    
    # Show allocation
    budgets = memory_mgr.allocate()
    print(f"Memory allocation: {budgets}")
    
    # Use scratchpad
    memory_mgr.scratchpad.write("Floor 5: Found rare berry", priority=1)
    memory_mgr.scratchpad.write("Enemy Pokemon ahead", priority=0)
    
    entries = memory_mgr.scratchpad.read()
    print(f"Scratchpad entries: {entries}")


def demonstrate_fps_adjustment():
    """Demonstrate FPS adjustment functionality."""
    print("\n=== FPS Adjustment Demo ===")
    
    fps_adjuster = FPSAdjuster(base_fps=30, initial_multiplier=4)
    
    print(f"Initial effective FPS: {fps_adjuster.get_current_fps()}")
    print(f"Temporal span info: {fps_adjuster.get_temporal_span_info()}")
    
    # Zoom out temporally
    fps_adjuster.set_fps(5)
    print(f"After zoom out: {fps_adjuster.get_current_fps()} FPS")
    
    # Adjust frame multiplier
    fps_adjuster.set_multiplier(8)
    print(f"After 2x multiplier: {fps_adjuster.get_effective_fps()} FPS")
    
    # Get adjustment summary
    summary = fps_adjuster.get_adjustment_summary()
    print(f"Adjustment summary: {summary}")


def demonstrate_temporal_silos():
    """Demonstrate temporal silo functionality."""
    print("\n=== Temporal Silos Demo ===")
    
    silo_manager = TemporalSiloManager(base_fps=30)
    
    # Show silo configurations
    stats = silo_manager.get_silo_stats()
    print("Silo configurations:")
    for silo_id, silo_stats in stats.items():
        print(f"  {silo_id}: {silo_stats['total_entries']}/{silo_stats['max_capacity']} entries")
    
    # Simulate storing embeddings
    import numpy as np
    
    current_time = time.time()
    dummy_embedding = np.random.normal(0, 0.1, 1024)
    
    # Store in multiple silos
    for silo_id in ["temporal_1frame", "temporal_4frame", "temporal_16frame"]:
        silo_manager.store(
            embedding=dummy_embedding,
            trajectory_id="test_trajectory_001",
            silo_id=silo_id,
            current_time=current_time,
            metadata={"action": "move_right"}
        )
    
    # Show updated stats
    updated_stats = silo_manager.get_silo_stats()
    print("\nAfter storing embeddings:")
    for silo_id, silo_stats in updated_stats.items():
        if silo_stats['total_entries'] > 0:
            print(f"  {silo_id}: {silo_stats['total_entries']} entries")
    
    # Cross-silo search
    query_embedding = np.random.normal(0, 0.1, 1024)
    results = silo_manager.cross_silo_search(query_embedding, top_k=2)
    print(f"\nCross-silo search found matches in {len(results)} silos")


def demonstrate_vector_store():
    """Demonstrate vector store functionality."""
    print("\n=== Vector Store Demo ===")
    
    # Test memory backend
    store = VectorStore(backend="memory", embedding_dimension=1024)
    
    # Add some entries
    import numpy as np
    
    for i in range(3):
        embedding = np.random.normal(0, 0.1, 1024)
        store.add_entry(
            entry_id=f"entry_{i}",
            embedding=embedding,
            metadata={"trajectory_id": f"traj_{i}", "action": "move"},
            silo_id="temporal_4frame"
        )
    
    # Search
    query = np.random.normal(0, 0.1, 1024)
    results = store.search(query, top_k=2)
    
    print(f"Search returned {len(results)} results:")
    for entry_id, similarity, metadata in results:
        print(f"  {entry_id}: {similarity:.3f} similarity")
    
    # Get stats
    stats = store.get_stats()
    print(f"Vector store stats: {stats}")


def demonstrate_router_runtime():
    """Demonstrate RouterGlue runtime wiring with maintenance daemon."""
    print("\n=== Router Runtime Demo ===")

    silo_manager = TemporalSiloManager(base_fps=30, silos=[1, 2])

    router, maintenance = build_router_runtime(
        silo_manager=silo_manager,
        cadence_seconds=0,
        cadence_steps=1,
    )

    import numpy as np

    base_time = time.time()
    for idx in range(4):
        embedding = np.random.normal(0, 0.1, 1024)
        silo_manager.store(
            embedding=embedding,
            trajectory_id=f"router_demo_{idx}",
            metadata={"action": "move", "floor": 1},
            current_time=base_time + idx,
        )

    metrics = maintenance.run(force=True)
    print(f"Maintenance per-silo counts: {metrics.per_silo_counts}")
    print(f"Removed (compact, expire): {metrics.total_removed_compaction}, {metrics.total_removed_retention}")
    assert router.maintenance_daemon is maintenance
    print("RouterGlue shares the maintenance daemon instance.")


def demonstrate_agent_controller():
    """Demonstrate agent controller functionality."""
    print("\n=== Agent Controller Demo ===")
    
    # Create components
    router = ModelRouter()
    memory_mgr = MemoryManager()
    
    # Create agent controller
    agent = QwenController(model_router=router, memory_manager=memory_mgr)
    
    # Simulate perception
    perception = agent.perceive(
        screenshot=None,  # Would be actual screenshot
        sprite_detections=[
            {"type": "player", "x": 100, "y": 200, "confidence": 0.9},
            {"type": "enemy", "x": 150, "y": 180, "confidence": 0.8},
        ]
    )
    
    print(f"Perception: {perception}")
    
    # Make decision
    result = agent.think_and_decide(perception)
    print(f"Decision: {result.action} (confidence: {result.confidence:.2f})")
    print(f"Reasoning: {result.reasoning}")
    
    # Update stuck counter
    agent.update_stuck_counter(is_stuck=True)
    agent.update_stuck_counter(is_stuck=True)
    agent.update_stuck_counter(is_stuck=True)
    agent.update_stuck_counter(is_stuck=True)
    agent.update_stuck_counter(is_stuck=True)
    
    # Try decision again (should escalate to 8B)
    result2 = agent.think_and_decide(perception)
    print(f"After stuck: {result2.model_used} model selected")


def main():
    """Run the quickstart demonstration."""
    print("Pokemon MD Autonomous Agent - Quick Start Demo")
    print("=" * 50)

    setup_logging()

    try:
        demonstrate_model_routing()
        demonstrate_memory_management()
        demonstrate_fps_adjustment()
        demonstrate_temporal_silos()
        demonstrate_vector_store()
        demonstrate_router_runtime()
        demonstrate_agent_controller()

        print("\n=== Demo Complete ===")
        print("All components working correctly!")
        print("\nNext steps:")
        print("1. Install mgba and start mgba-http server")
        print("2. Load Pokemon Mystery Dungeon Red ROM")
        print("3. Run: python -m src --demo")
        print("4. Implement actual Qwen3-VL model loading")
        print("5. Connect vision processing pipeline")
        print("6. Run full agent loop")
        print("7. Integrate RouterGlue runtime into production orchestrator")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
