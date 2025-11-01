"""Pokemon MD Agent Core.

Main agent loop that coordinates all components for autonomous gameplay.
"""

import asyncio
import logging
import random
import time
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.environment.mgba_controller import MGBAController
from src.environment.rom_gating import find_rom_files, ROMValidationError
from src.vision.grid_parser import GridParser
from src.vision.ascii_renderer import ASCIIRenderer
from src.agent.qwen_controller import QwenController
from src.agent.gatekeeper import Gatekeeper
from src.skills.runtime import SkillRuntime
from src.retrieval.stuckness_detector import StucknessDetector, StucknessAnalysis, StucknessStatus
from src.retrieval.auto_retrieve import AutoRetriever, RetrievalQuery, RetrievedTrajectory
from src.environment.ram_decoders import RAMSnapshot, PlayerState, MapData, PartyStatus
from src.environment.ram_watch import RAMWatcher, create_ram_watcher
from src.retrieval.ann_search import VectorSearch
from .agent_config import AgentConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AgentCore",
    "PokemonMDAgent",
    "RAMWatcher",
    "create_ram_watcher",
]


class AgentConfig:
    """Configuration for PokemonMDAgent behavior."""
 
    def __init__(
        self,
        screenshot_interval: float = 1.0,
        memory_poll_interval: float = 0.1,
        decision_interval: float = 0.5,
        max_runtime_hours: float = 1.0,
        enable_4up_capture: bool = True,
        enable_trajectory_logging: bool = True,
        enable_stuck_detection: bool = True,
        enable_skill_triggers: bool = False,
        skill_belly_threshold: float = 0.3,
        skill_hp_threshold: float = 0.25,
        skill_backoff_seconds: float = 5.0
    ):
        """Initialize agent configuration.
 
        Args:
            screenshot_interval: Seconds between screenshots
            memory_poll_interval: Seconds between memory polls
            decision_interval: Seconds between decisions
            max_runtime_hours: Maximum runtime in hours
            enable_4up_capture: Enable 4-up screenshot capture
            enable_trajectory_logging: Enable trajectory logging
            enable_stuck_detection: Enable stuckness detection
            enable_skill_triggers: Enable automatic skill triggers
            skill_belly_threshold: Belly threshold for triggers (0-1)
            skill_hp_threshold: HP threshold for triggers (0-1)
            skill_backoff_seconds: Seconds to wait after skill execution
        """
        self.screenshot_interval = screenshot_interval
        self.memory_poll_interval = memory_poll_interval
        self.decision_interval = decision_interval
        self.max_runtime_hours = max_runtime_hours
        self.enable_4up_capture = enable_4up_capture
        self.enable_trajectory_logging = enable_trajectory_logging
        self.enable_stuck_detection = enable_stuck_detection
        self.enable_skill_triggers = enable_skill_triggers
        self.skill_belly_threshold = skill_belly_threshold
        self.skill_hp_threshold = skill_hp_threshold
        self.skill_backoff_seconds = skill_backoff_seconds


class AgentCore:
    """Main agent loop: perceive → reason → act."""

    def __init__(self, objective: str = "Navigate to stairs", test_mode: bool = False, enable_retrieval: bool = False):
        # Initialize all components
        self.test_mode = test_mode
        self.mgba = MGBAController()
        self.grid_parser = GridParser()
        self.ascii_renderer = ASCIIRenderer()
        self.qwen = QwenController(use_pipeline=not test_mode)
        self.skills = SkillRuntime(self.mgba)
        self.stuckness = StucknessDetector()
        self.retriever = None

        # Initialize gatekeeper with ANN search if retrieval enabled
        if enable_retrieval:
            # Initialize ANN search from retrieval components
            try:
                from src.retrieval.ann_search import VectorSearch
                # Assume ANN index path is configured - would be passed as parameter in production
                ann_search = VectorSearch(index_path="data/ann_index.faiss")
                logger.info("ANN search initialized for gatekeeper")
            except Exception as e:
                logger.warning(f"Could not initialize ANN search for gatekeeper: {e}")
                ann_search = None
        else:
            ann_search = None

        self.gatekeeper = Gatekeeper(
            ann_search=ann_search,
            min_hits=3
        )

        # Set objective
        self.objective = objective

        # State tracking for stuckness detection
        self.step_count = 0
        self.last_state = None
        self.embedding_history = []

        # Logging setup
        self.log_file = Path("agent_log.txt")
        self._setup_logging()

        # Optional retrieval system
        if enable_retrieval and not test_mode:
            # Would initialize AutoRetriever with proper dependencies
            # self.retriever = AutoRetriever(silo_manager, vector_store, deduplicator)
            logger.info("Retrieval system enabled (placeholder)")

        # Connect to mGBA with retry (skip in test mode)
        if not test_mode:
            # Validate ROM files before connecting to mGBA
            try:
                rom_files = find_rom_files()
                if not rom_files:
                    raise ROMValidationError(
                        "No ROM files found. Please ensure Pokemon Mystery Dungeon - Red Rescue Team "
                        "is present in the rom/ directory."
                    )
                logger.info(f"Found {len(rom_files)} ROM file(s): {[f.name for f in rom_files]}")
            except ROMValidationError as e:
                logger.error(f"ROM validation failed: {e}")
                raise RuntimeError(f"Agent initialization failed: {e}") from e

            # Don't fail init if connection fails - try during first perceive
            connected = self.mgba.connect_with_retry(max_retries=1)
            if connected:
                logger.info("Successfully connected to mGBA during initialization")
            else:
                logger.warning("Failed to connect to mGBA during initialization - will retry during first perceive()")

        logger.info(f"Agent initialized with objective: {objective} (test_mode: {test_mode})")

    def _setup_logging(self) -> None:
        """Setup file logging."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info("File logging initialized")

    def perceive(self) -> dict:
        """Get current game state via full perception pipeline."""
        if self.test_mode:
            # Return mock data for testing
            return {
                "screenshot": None,
                "grid": {"width": 32, "height": 32, "cells": []},
                "ascii": "Test ASCII view\nPlayer at (10,10)\nStairs at (15,15)",
                "ram": {"player_x": 10, "player_y": 10, "floor_number": 1}
            }

        try:
            # Capture screenshot
            screenshot = self.mgba.grab_frame()
            logger.debug("Captured screenshot")

            # Parse RAM snapshot for comprehensive state
            ram_snapshot = self._read_ram_snapshot()
            logger.debug(f"Read RAM snapshot: floor={ram_snapshot.player_state.floor_number}, pos=({ram_snapshot.player_state.player_tile_x},{ram_snapshot.player_state.player_tile_y})")

            # Parse grid from RAM data
            grid_frame = self.grid_parser.parse_ram_snapshot(ram_snapshot)
            logger.debug(f"Parsed grid: {grid_frame.width}x{grid_frame.height}")

            # Render ASCII representation
            ascii_view = self.ascii_renderer.render_environment_with_entities(grid_frame, ram_snapshot)
            logger.debug("Rendered ASCII view")

            # Create embedding for stuckness detection
            embedding = self._create_state_embedding(grid_frame, ram_snapshot)

            return {
                "screenshot": screenshot,
                "grid": grid_frame,
                "ascii": ascii_view,
                "ram": {
                    "player_x": ram_snapshot.player_state.player_tile_x,
                    "player_y": ram_snapshot.player_state.player_tile_y,
                    "floor_number": ram_snapshot.player_state.floor_number,
                    "snapshot": ram_snapshot
                },
                "embedding": embedding,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Perception failed: {e}")
            # Fallback to basic RAM reading
            ram_state = self._read_ram_state()
            return {
                "screenshot": None,
                "grid": {"width": 32, "height": 32, "cells": []},
                "ascii": f"Fallback view\nPlayer at ({ram_state['player_x']},{ram_state['player_y']})\nFloor: {ram_state['floor_number']}",
                "ram": ram_state,
                "embedding": np.zeros(128),  # Placeholder embedding
                "timestamp": time.time()
            }

    def _read_ram_snapshot(self) -> RAMSnapshot:
        """Read comprehensive RAM snapshot."""
        from src.environment.ram_decoders import create_decoder, PlayerState, MapData, PartyStatus

        # Check connection and reconnect if needed
        if not self.mgba.is_connected():
            logger.warning("mGBA not connected, attempting reconnection...")
            if not self.mgba.connect_with_retry(max_retries=2):
                logger.error("Failed to reconnect to mGBA, returning default snapshot")
                # Return minimal snapshot with defaults
                return RAMSnapshot(
                    player_state=PlayerState(
                        player_tile_x=0, player_tile_y=0, floor_number=1,
                        partner_tile_x=0, partner_tile_y=0, dungeon_id=1, turn_counter=0
                    ),
                    map_data=MapData(
                        stairs_x=-1, stairs_y=-1, camera_origin_x=0, camera_origin_y=0,
                        weather_state=0, turn_phase=0
                    ),
                    party_status=PartyStatus(
                        leader_hp=50, leader_hp_max=50, leader_belly=50,
                        partner_hp=50, partner_hp_max=50, partner_belly=50
                    ),
                    entities=[],
                    items=[],
                    timestamp=time.time()
                )

        decoder = create_decoder()
        raw_data = self.mgba.memory_domain_read_range("WRAM", 0x02000000, 2048)
        if not raw_data:
            logger.warning("Failed to read RAM data from WRAM, returning default snapshot")
            # Return minimal snapshot with defaults
            return RAMSnapshot(
                player_state=PlayerState(
                    player_tile_x=0, player_tile_y=0, floor_number=1,
                    partner_tile_x=0, partner_tile_y=0, dungeon_id=1, turn_counter=0
                ),
                map_data=MapData(
                    stairs_x=-1, stairs_y=-1, camera_origin_x=0, camera_origin_y=0,
                    weather_state=0, turn_phase=0
                ),
                party_status=PartyStatus(
                    leader_hp=50, leader_hp_max=50, leader_belly=50,
                    partner_hp=50, partner_hp_max=50, partner_belly=50
                ),
                entities=[],
                items=[],
                timestamp=time.time()
            )

        # Use decoder to get snapshot (assuming decode_player_state etc.)
        player_state_dict = decoder.decode_player_state(raw_data)
        player_state = PlayerState(
            player_tile_x=player_state_dict.get("player_tile_x", 0),
            player_tile_y=player_state_dict.get("player_tile_y", 0),
            floor_number=player_state_dict.get("floor_number", 1),
            dungeon_id=player_state_dict.get("dungeon_id", 1),
            turn_counter=player_state_dict.get("turn_counter", 0),
            partner_tile_x=player_state_dict.get("partner_tile_x", 0),
            partner_tile_y=player_state_dict.get("partner_tile_y", 0)
        )
        
        map_data_dict = decoder.decode_map_data(raw_data)
        map_data = MapData(
            camera_origin_x=map_data_dict.get("camera_origin_x", 0),
            camera_origin_y=map_data_dict.get("camera_origin_y", 0),
            weather_state=map_data_dict.get("weather_state", 0),
            turn_phase=map_data_dict.get("turn_phase", 0),
            stairs_x=map_data_dict.get("stairs_x", -1),
            stairs_y=map_data_dict.get("stairs_y", -1)
        )
        
        party_status_dict = decoder.decode_party_status(raw_data)
        party_status = PartyStatus(
            leader_hp=party_status_dict.get("leader", {}).get("hp", 50),
            leader_hp_max=party_status_dict.get("leader", {}).get("hp_max", 50),
            leader_belly=party_status_dict.get("leader", {}).get("belly", 50),
            partner_hp=party_status_dict.get("partner", {}).get("hp", 50),
            partner_hp_max=party_status_dict.get("partner", {}).get("hp_max", 50),
            partner_belly=party_status_dict.get("partner", {}).get("belly", 50)
        )
        
        return RAMSnapshot(
            player_state=player_state,
            map_data=map_data,
            party_status=party_status,
            entities=[],  # Would need entity decoder
            items=[],    # Would need item decoder
            timestamp=time.time()
        )

    def _read_ram_state(self) -> dict:
        """Read key RAM addresses (legacy fallback)."""
        snapshot = self._read_ram_snapshot()
        return {
            "player_x": snapshot.player_state.player_tile_x,
            "player_y": snapshot.player_state.player_tile_y,
            "floor_number": snapshot.player_state.floor_number
        }

    def _create_state_embedding(self, grid_frame, ram_snapshot: RAMSnapshot) -> np.ndarray:
        """Create state embedding for stuckness detection."""
        # Simple embedding: position + floor + entity count
        embedding = np.array([
            ram_snapshot.player_state.player_tile_x / 32.0,
            ram_snapshot.player_state.player_tile_y / 32.0,
            ram_snapshot.player_state.floor_number / 10.0,
            len(ram_snapshot.entities) / 10.0,
            len(ram_snapshot.items) / 5.0
        ])
        # Pad to 128 dimensions
        embedding = np.pad(embedding, (0, 123), mode='constant')
        return embedding

    async def reason(self, state: dict) -> dict:
        """Decide what to do with full reasoning pipeline."""
        try:
            # Build prompt from state
            prompt = self._build_prompt(state)

            # Check stuckness detection (every N steps)
            stuck_analysis = None
            if self.step_count % 5 == 0 and 'embedding' in state:  # Check every 5 steps
                stuck_analysis = self.stuckness.analyze(
                    current_embedding=state['embedding'],
                    current_position=(state['ram']['player_x'], state['ram']['player_y']),
                    current_action=None,  # Would track from previous action
                    current_time=state.get('timestamp', time.time())
                )
                logger.debug(f"Stuckness analysis: {stuck_analysis.status.value} (conf: {stuck_analysis.confidence:.2f})")

            # If stuck, try retrieval or fallback
            if stuck_analysis and stuck_analysis.status in [StucknessStatus.STUCK, StucknessStatus.VERY_STUCK]:
                logger.warning(f"Agent stuck: {stuck_analysis.reasons[0]}")

                # Try retrieval if available
                if self.retriever:
                    query = RetrievalQuery(
                        current_embedding=state['embedding'],
                        current_position=(state['ram']['player_x'], state['ram']['player_y']),
                        current_floor=state['ram']['floor_number'],
                        time_window_seconds=120.0
                    )
                    trajectories = self.retriever.retrieve(query)
                    if trajectories:
                        logger.info(f"Retrieved {len(trajectories)} relevant trajectories")
                        # Would integrate trajectories into prompt

                # Fallback to random escape action
                return {"action": "random", "rationale": f"Stuck detected: {stuck_analysis.reasons[0]}"}

            # Query Qwen with async generation
            decision_text, scores = await self.qwen.generate_async(prompt)
            logger.debug(f"Qwen response: {decision_text[:100]}...")

            # Parse decision - get potential actions
            decision = self._parse_decision(decision_text)

            # Filter actions through gatekeeper
            valid_actions = self._extract_valid_actions(decision_text)
            filtered_actions = await self.gatekeeper.filter(valid_actions, state)

            if not filtered_actions:
                logger.warning("All actions filtered out by gatekeeper")
                return {"action": "random", "rationale": "Gatekeeper filtered all actions"}

            # Use first filtered action
            decision['action'] = filtered_actions[0]
            logger.debug(f"Gatekeeper filtered actions: {valid_actions} -> {filtered_actions}")

            # Try skill execution if action maps to skill
            if decision['action'] in ['TAKE_STAIRS', 'USE_ITEM', 'ATTACK']:
                skill_success = await self._try_skill_execution(decision['action'], state)
                if skill_success:
                    decision['rationale'] += " (executed via skill)"

            return decision

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            # Fallback to random action
            return {"action": "random", "rationale": f"Fallback due to error: {e}"}

    def _is_stuck_simple(self, state: dict) -> bool:
        """Simple stuck detection - check if position hasn't changed."""
        # This is a placeholder - real implementation would track history
        return False

    def _build_prompt(self, state: dict) -> str:
        """Construct prompt for Qwen."""
        # Include objective, current state, ASCII view
        prompt = f"""Objective: {self.objective}

Current State:
{state['ascii']}

Player Position: ({state['ram']['player_x']}, {state['ram']['player_y']})
Floor: {state['ram']['floor_number']}

What action should the agent take? Choose from:
- MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
- USE_ITEM, ATTACK, TAKE_STAIRS
- WAIT

Respond with: ACTION | RATIONALE
Example: MOVE_UP | Stairs likely to the north
"""
        return prompt

    def _parse_decision(self, text: str) -> dict:
        """Extract action and rationale from Qwen output."""
        # Simple parsing: split on |
        parts = text.split("|")
        if len(parts) >= 2:
            action = parts[0].strip()
            rationale = parts[1].strip()
        else:
            action = "WAIT"
            rationale = "Could not parse decision"

        return {"action": action, "rationale": rationale}

    def _extract_valid_actions(self, text: str) -> List[str]:
        """Extract potential valid actions from Qwen response."""
        # Parse actions from prompt format
        # Example: "What action should the agent take? Choose from:
        # - MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
        # - USE_ITEM, ATTACK, TAKE_STAIRS
        # - WAIT"

        actions = []
        if "MOVE_UP" in text or "move up" in text.lower():
            actions.extend(["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"])
        if "USE_ITEM" in text or "use item" in text.lower():
            actions.append("USE_ITEM")
        if "ATTACK" in text or "attack" in text.lower():
            actions.append("ATTACK")
        if "TAKE_STAIRS" in text or "take stairs" in text.lower():
            actions.append("TAKE_STAIRS")
        if "WAIT" in text or "wait" in text.lower():
            actions.append("WAIT")

        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)

        return unique_actions if unique_actions else ["WAIT"]

    def act(self, decision: dict):
        """Execute action."""
        if self.test_mode:
            # Just log in test mode
            logger.info(f"[TEST MODE] Action: {decision['action']} | Rationale: {decision['rationale']}")
            return

        action = decision["action"]

        # Map action to button sequence
        button_map = {
            "MOVE_UP": ["UP"],
            "MOVE_DOWN": ["DOWN"],
            "MOVE_LEFT": ["LEFT"],
            "MOVE_RIGHT": ["RIGHT"],
            "USE_ITEM": ["A"],
            "ATTACK": ["A"],
            "TAKE_STAIRS": ["A"],
            "WAIT": [],
            "random": self._random_action()
        }

        buttons = button_map.get(action, [])

        # Press buttons via mGBA
        for button in buttons:
            self.mgba.button_tap(button)
            self.mgba.await_frames(1)  # Wait 1 frame between inputs

        logger.info(f"Action: {action} | Rationale: {decision['rationale']}")

    async def _try_skill_execution(self, action: str, state: dict) -> bool:
        """Try to execute action via skill runtime."""
        try:
            # Create simple skill for basic actions
            from src.skills.dsl import Skill, Tap
            from src.skills.dsl import Button

            button_map = {
                "TAKE_STAIRS": Button.A,
                "USE_ITEM": Button.A,
                "ATTACK": Button.A
            }

            if action in button_map:
                skill = Skill(name=f"simple_{action.lower()}", actions=[Tap(button=button_map[action])])
                success = await self.skills.execute_skill(skill)
                logger.info(f"Skill execution for {action}: {'success' if success else 'failed'}")
                return success

        except Exception as e:
            logger.warning(f"Skill execution failed: {e}")

        return False

    def _random_action(self) -> list:
        """Random action for escaping stuck states."""
        directions = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"]]
        return random.choice(directions)

    async def run(self, max_steps: int = 100):
        """Main agent loop with full autonomous operation."""
        logger.info(f"Starting agent loop (max {max_steps} steps)")

        # Initialize async components
        if not self.test_mode:
            await self.qwen.initialize_async()
            logger.info("Async components initialized")

        for step in range(max_steps):
            try:
                self.step_count = step + 1

                # Perceive
                state = self.perceive()

                # Update stuckness history
                if 'embedding' in state:
                    from src.retrieval.stuckness_detector import TemporalSnapshot
                    snapshot = TemporalSnapshot(
                        timestamp=state['timestamp'],
                        embedding=state['embedding'],
                        position=(state['ram']['player_x'], state['ram']['player_y']),
                        floor=state['ram']['floor_number']
                    )
                    self.stuckness.add_snapshot(snapshot)

                # Reason
                decision = await self.reason(state)

                # Act
                self.act(decision)

                # Brief pause between steps for game processing
                if not self.test_mode:
                    await asyncio.sleep(0.1)

                # Log progress
                logger.info(f"Step {step+1}/{max_steps} complete - Action: {decision['action']}")

            except KeyboardInterrupt:
                logger.info("Agent interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error at step {step+1}: {e}")
                # Try reconnection on connection errors
                if "connection" in str(e).lower() and not self.test_mode:
                    logger.info("Attempting reconnection...")
                    if self.mgba.connect_with_retry(max_retries=2):
                        logger.info("Reconnected successfully")
                        continue
                # Continue on other errors (don't crash the demo)

        logger.info("Agent loop complete")


class PokemonMDAgent(AgentCore):
    """Pokemon MD Agent - orchestrates all components for autonomous gameplay."""
 
    def __init__(
        self,
        rom_path: Path,
        save_dir: Path,
        config: AgentConfig,
        test_mode: bool = False
    ):
        """Initialize PokemonMDAgent.
 
        Args:
            rom_path: Path to the ROM file
            save_dir: Directory for save states
            config: Agent configuration
            test_mode: Enable test mode with mock components
        """
        # Initialize parent with default objective
        super().__init__(
            objective="Autonomous Pokemon MD gameplay",
            test_mode=test_mode,
            enable_retrieval=config.enable_trajectory_logging
        )
 
        self.rom_path = rom_path
        self.save_dir = save_dir
        self.config = config
        self.running = False
        self._stop_event = None
 
        # Set up save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
 
        logger.info(
            f"PokemonMDAgent initialized: ROM={rom_path.name}, "
            f"runtime={config.max_runtime_hours}h, "
            f"stuck_detection={config.enable_stuck_detection}"
        )
 
    async def _initialize(self) -> None:
        """Initialize agent components (async version)."""
        logger.info("Initializing PokemonMDAgent...")
 
        # Connect to mGBA
        if not self.mgba.connect_with_retry(max_retries=3):
            raise RuntimeError("Failed to connect to mGBA")
 
        logger.info("Connected to mGBA")
 
        # Load save state
        self.mgba.autoload_save()
        logger.info("Loaded save state")
 
        # Initialize async components
        await self.qwen.initialize_async()
        logger.info("Async components initialized")
 
    async def _gather_decision_context(self) -> dict:
        """Gather context for decision making."""
        return self.perceive()
 
    async def _execute_decision(self, decision: dict) -> None:
        """Execute a decision.
 
        Args:
            decision: Decision dict with action and parameters
        """
        action = decision.get("action")
 
        if action == "move":
            direction = decision.get("direction")
            if direction:
                self.mgba.button_tap(direction.upper())
 
        elif action == "use_item":
            # Simplified item usage
            self.mgba.button_tap("A")  # Open menu
            await asyncio.sleep(0.5)
            self.mgba.button_tap("B")  # Cancel for now
 
        elif action == "interact":
            self.mgba.button_tap("A")  # Interact
 
        logger.info(f"Executed decision: {decision}")
 
    async def _cleanup(self) -> None:
        """Clean up agent resources."""
        logger.info("Cleaning up agent...")
 
        # Stop any running processes
        if self._stop_event:
             self._stop_event.set()
 
        # Disconnect from mGBA
        self.mgba.disconnect()
 
        # Clean up async components
        if hasattr(self.qwen, 'cleanup'):
            await self.qwen.cleanup()
 
        logger.info("Agent cleanup complete")
 
    def stop(self) -> None:
        """Stop the agent."""
        self.running = False
        if self._stop_event:
            self._stop_event.set()
 
    async def run(self) -> None:
        """Main agent run loop."""
        self.running = True
        self._stop_event = asyncio.Event()
 
        try:
            await self._initialize()
 
            logger.info("Starting agent run loop...")
 
            while self.running and not self._stop_event.is_set():
                try:
                    # Gather context
                    context = await self._gather_decision_context()
 
                    # Make decision
                    decision = await self.reason(context)
 
                    # Execute decision
                    await self._execute_decision(decision)
 
                    # Check for skill triggers
                    if self.config.enable_skill_triggers and 'ram' in context:
                        await self._check_and_execute_skills(context['ram'])
 
                    # Brief pause
                    await asyncio.sleep(self.config.decision_interval)
 
                except KeyboardInterrupt:
                    logger.info("Agent interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}")
                    # Try to continue
                    await asyncio.sleep(1.0)
 
        finally:
            await self._cleanup()
 
    async def _check_and_execute_skills(self, ram_state: dict) -> None:
        """Check for skill trigger conditions and execute if needed.
 
        Args:
            ram_state: RAM state from perception
        """
        # Simplified trigger check - in real implementation would decode actual party status
        try:
            # Mock party status for testing
            party_status = ram_state.get('party_status', {})
            if party_status:
                self._check_skill_triggers(party_status)
        except Exception as e:
            logger.warning(f"Skill trigger check failed: {e}")
 
    def _check_skill_triggers(self, party_status: dict) -> bool:
        """Check if skill triggers should activate.
 
        Args:
            party_status: Party status data from RAM
 
        Returns:
            True if trigger should activate
        """
        # Mock implementation for testing
        leader = party_status.get('leader', {})
        hp = leader.get('hp', 100)
        hp_max = leader.get('hp_max', 100)
        belly = leader.get('belly', 100)
 
        # Calculate percentages
        hp_percent = hp / hp_max if hp_max > 0 else 1.0
        belly_percent = belly / 200.0 if belly > 0 else 1.0  # Assume max belly is 200
 
        # Check thresholds
        if hp_percent < self.config.skill_hp_threshold:
            logger.info(f"HP trigger: {hp_percent:.1%} < {self.config.skill_hp_threshold:.1%}")
            return True
 
        if belly_percent < self.config.skill_belly_threshold:
            logger.info(f"Belly trigger: {belly_percent:.1%} < {self.config.skill_belly_threshold:.1%}")
            return True
 
        return False

