"""Pokemon MD Agent Core.

Main agent loop that coordinates all components for autonomous gameplay.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..environment.mgba_controller import MGBAController
from ..environment.save_manager import SaveManager
from ..environment.ram_decoders import RAMDecoder
from ..environment.ram_watch import RAMWatcher
from ..retrieval.trajectory_logger import TrajectoryLogger, CombatEvent, MovementTrajectory, DecisionLog
from ..models.world_model import WorldModel, FloorModel, Entity, Position, EntityType
from ..vision.quad_capture import QuadCapture
from .model_router import ModelRouter
from .memory_manager import MemoryManager
from ..retrieval.stuckness_detector import StucknessDetector

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    screenshot_interval: float = 1.0  # seconds
    memory_poll_interval: float = 0.1  # seconds
    decision_interval: float = 0.5  # seconds
    max_runtime_hours: float = 24.0
    enable_4up_capture: bool = True
    enable_trajectory_logging: bool = True
    enable_stuck_detection: bool = True
    auto_save_interval: int = 300  # frames
    capture_scale: int = 2  # Capture scale factor (1=240×160, 2=320×480)

    # Dashboard configuration
    dashboard_enabled: bool = True
    dashboard_branch: str = "pages"
    dashboard_site_root: str = "docs"
    dashboard_flush_seconds: float = 30.0
    dashboard_max_batch_bytes: int = 8 * 1024 * 1024  # 8 MB
    dashboard_max_files_per_minute: int = 30


class PokemonMDAgent:
    """Main Pokemon Mystery Dungeon autonomous agent."""
    
    def __init__(self, rom_path: Path, save_dir: Path, config: Optional[AgentConfig] = None):
        """Initialize the agent.
        
        Args:
            rom_path: Path to the Pokemon MD ROM
            save_dir: Directory for saves and logs
            config: Agent configuration
        """
        self.rom_path = Path(rom_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AgentConfig()
        
        # Core components
        self.controller = MGBAController()
        self.save_manager = SaveManager(self.controller, self.save_dir / "saves")
        addresses_config = Path("config/addresses/pmd_red_us_v1.json")
        self.ram_decoder = RAMDecoder(self.controller, addresses_config)
        self.ram_watcher = RAMWatcher(self.controller, self.ram_decoder, 
                                    self.save_dir / "ram_logs")
        
        # Agent components
        self.model_router = ModelRouter()
        self.memory_manager = MemoryManager(total_context_budget=256000)
        self.stuck_detector = StucknessDetector()
        
        # World modeling
        self.world_model = WorldModel(self.save_dir / "world_model")
        
        # Logging and capture
        self.trajectory_logger = TrajectoryLogger(self.save_dir / "trajectories")
        self.quad_capture = QuadCapture(self.controller, self.save_dir / "captures")
        
        # State tracking
        self.running = False
        self.start_time = 0.0
        self.frame_count = 0
        self.last_screenshot = 0.0
        self.last_memory_poll = 0.0
        self.last_decision = 0.0
        self.last_auto_save = 0.0
        
        logger.info("Pokemon MD Agent initialized")
    
    async def run(self) -> None:
        """Run the main agent loop."""
        try:
            # Initialize
            await self._initialize()
            
            # Main loop
            while self.running and self._should_continue():
                current_time = time.time()
                
                # Update frame count
                self.frame_count += 1
                
                # Periodic tasks
                await self._handle_screenshots(current_time)
                await self._handle_memory_polling(current_time)
                await self._handle_decisions(current_time)
                await self._handle_auto_save(current_time)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
            
            logger.info("Agent loop completed")
            
        except Exception as e:
            logger.error("Agent loop failed: %s", e)
            raise
        finally:
            await self._cleanup()
    
    async def _initialize(self) -> None:
        """Initialize the agent."""
        logger.info("Initializing agent...")
        
        # Connect to emulator
        if not self.controller.connect():
            raise RuntimeError("Failed to connect to mGBA emulator")
        
        # Load ROM - mGBA should already have it loaded, just ensure connection
        
        # Ensure clean start state
        self.save_manager.ensure_startable_state()
        
        # Start RAM watching
        await self.ram_watcher.start_watching()
        
        # Initialize components (no async init methods)
        
        self.start_time = time.time()
        self.running = True
        
        logger.info("Agent initialization complete")
    
    async def _handle_screenshots(self, current_time: float) -> None:
        """Handle periodic screenshots and captures."""
        if current_time - self.last_screenshot >= self.config.screenshot_interval:
            try:
                # Basic screenshot
                screenshot_path = self.save_dir / "screenshots" / "04d"
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                self.controller.screenshot(str(screenshot_path))
                
                # 4-up capture if enabled
                if self.config.enable_4up_capture:
                    # Get current state for metadata
                    player_state = self.ram_decoder.get_player_state()
                    entities = self.ram_decoder.get_entities()
                    items = self.ram_decoder.get_items()
                    
                    floor = player_state.floor_number if player_state else 0
                    dungeon_id = player_state.dungeon_id if player_state else 0
                    
                    self.quad_capture.capture_quad_view(
                        frame=self.frame_count,
                        floor=floor,
                        dungeon_id=dungeon_id,
                        room_kind="unknown",  # TODO: detect room type
                        player_pos=(player_state.player_tile_x, player_state.player_tile_y) if player_state else (0, 0),
                        entities_count=len(entities) if entities else 0,
                        items_count=len(items) if items else 0
                    )
                
                self.last_screenshot = current_time
                
            except Exception as e:
                logger.warning("Screenshot capture failed: %s", e)
    
    async def _handle_memory_polling(self, current_time: float) -> None:
        """Handle periodic memory polling and world model updates."""
        if current_time - self.last_memory_poll >= self.config.memory_poll_interval:
            try:
                # Get current state
                player_state = self.ram_decoder.get_player_state()
                entities = self.ram_decoder.get_entities()
                items = self.ram_decoder.get_items()
                map_data = self.ram_decoder.get_map_data()
                
                if player_state and entities is not None and items is not None:
                    # Update world model
                    await self._update_world_model(player_state, entities, items, map_data)
                    
                    # Basic stuckness check (simplified)
                    if self.config.enable_stuck_detection:
                        # Simple check: if player hasn't moved in last few polls
                        current_pos = (player_state.player_tile_x, player_state.player_tile_y)
                        # TODO: implement proper stuckness detection
                        pass
                
                self.last_memory_poll = current_time
                
            except Exception as e:
                logger.warning("Memory polling failed: %s", e)
    
    async def _handle_decisions(self, current_time: float) -> None:
        """Handle periodic decision making."""
        if current_time - self.last_decision >= self.config.decision_interval:
            try:
                # Get current context
                context = await self._gather_decision_context()
                
                # For now, just log context - TODO: implement actual decision making
                logger.debug("Decision context: floor=%s, entities=%d", 
                           context.get('floor', 'unknown'), 
                           context.get('entities_count', 0))
                
                # TODO: implement model routing and decision execution
                
                self.last_decision = current_time
                
            except Exception as e:
                logger.warning("Decision making failed: %s", e)
    
    async def _handle_auto_save(self, current_time: float) -> None:
        """Handle periodic auto-saving."""
        if self.frame_count - self.last_auto_save >= self.config.auto_save_interval:
            try:
                self.save_manager.save_slot(self.frame_count % 10)  # Rotate through slots 0-9
                self.world_model.save_state()
                self.last_auto_save = self.frame_count
                
                logger.debug("Auto-saved at frame %d", self.frame_count)
                
            except Exception as e:
                logger.warning("Auto-save failed: %s", e)
    
    async def _update_world_model(self, player_state: Dict[str, Any], 
                                entities: List[Dict[str, Any]], 
                                items: List[Dict[str, Any]],
                                map_data: Optional[Dict[str, Any]]) -> None:
        """Update the world model with current state."""
        try:
            floor = player_state.get('floor', 0)
            dungeon_id = player_state.get('dungeon_id', 0)
            
            # Create/update floor model
            floor_model = FloorModel(
                floor_number=floor,
                dungeon_id=dungeon_id,
                width=map_data.get('width', 32) if map_data else 32,
                height=map_data.get('height', 32) if map_data else 32,
                last_updated=time.time()
            )
            
            # Add entities
            for entity_data in entities:
                entity = Entity(
                    id=entity_data.get('id', 0),
                    type=EntityType(entity_data.get('type', 0)),
                    position=Position(
                        x=entity_data.get('x', 0),
                        y=entity_data.get('y', 0),
                        floor=floor
                    ),
                    species_id=entity_data.get('species_id'),
                    level=entity_data.get('level'),
                    hp=entity_data.get('hp'),
                    max_hp=entity_data.get('max_hp'),
                    is_hostile=entity_data.get('is_hostile', False)
                )
                floor_model.entities[entity.id] = entity
                self.world_model.update_entity(entity)
            
            # Add items
            for item_data in items:
                item_entity = Entity(
                    id=item_data.get('id', 0),
                    type=EntityType.ITEM,
                    position=Position(
                        x=item_data.get('x', 0),
                        y=item_data.get('y', 0),
                        floor=floor
                    ),
                    item_id=item_data.get('item_id')
                )
                floor_model.entities[item_entity.id] = item_entity
                self.world_model.update_entity(item_entity)
            
            self.world_model.update_floor(floor_model)
            
        except Exception as e:
            logger.warning("World model update failed: %s", e)
    
    async def _gather_decision_context(self) -> Dict[str, Any]:
        """Gather context for decision making."""
        context = {
            'frame': self.frame_count,
            'timestamp': time.time(),
        }
        
        try:
            # Get current state
            player_state = self.ram_decoder.get_player_state()
            if player_state:
                context['floor'] = player_state.floor_number
                context['dungeon_id'] = player_state.dungeon_id
                context['player_x'] = player_state.player_tile_x
                context['player_y'] = player_state.player_tile_y
            
            # Get recent screenshots/captures
            recent_captures = self.quad_capture.get_recent_captures(3)
            if recent_captures:
                context['recent_captures'] = recent_captures
            
            # Get world model state
            world_stats = self.world_model.get_stats()
            context['world_model'] = world_stats
            
        except Exception as e:
            logger.warning("Failed to gather decision context: %s", e)
        
        return context
    
    async def _execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute a decision."""
        action = decision.get('action', 'wait')
        
        try:
            if action == 'move':
                direction = decision.get('direction', 'up')
                self.controller.button_tap(direction)
                
            elif action == 'interact':
                self.controller.button_tap('a')
                
            elif action == 'run_away':
                self.controller.button_hold('b', 500)  # Hold B for 500ms
                
            elif action == 'wait':
                pass  # Do nothing
                
            else:
                logger.warning("Unknown action: %s", action)
                
        except Exception as e:
            logger.error("Failed to execute decision: %s", e)
    
    async def _handle_stuck_situation(self) -> None:
        """Handle when the agent is detected as stuck."""
        try:
            # Try a few random actions to unstuck
            for _ in range(3):
                direction = ['up', 'down', 'left', 'right'][self.frame_count % 4]
                self.controller.button_tap(direction)
                await asyncio.sleep(0.1)
            
            logger.info("Attempted to unstuck agent")
            
        except Exception as e:
            logger.error("Failed to handle stuck situation: %s", e)
    
    def _should_continue(self) -> bool:
        """Check if the agent should continue running."""
        if not self.running:
            return False
        
        # Check time limit
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.config.max_runtime_hours:
            logger.info("Reached maximum runtime of %.1f hours", self.config.max_runtime_hours)
            return False
        
        return True
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up agent resources...")
        
        try:
            # Stop watching
            await self.ram_watcher.stop_watching()
            
            # Save final state
            self.save_manager.save_slot(99)  # Use slot 99 for final save
            self.world_model.save_state()
            
            # Disconnect
            self.controller.disconnect()
            
        except Exception as e:
            logger.error("Cleanup failed: %s", e)
    
    def stop(self) -> None:
        """Stop the agent."""
        logger.info("Stopping agent...")
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'running': self.running,
            'frame_count': self.frame_count,
            'runtime_seconds': time.time() - self.start_time if self.start_time else 0,
            'trajectory_stats': self.trajectory_logger.get_stats(),
            'world_stats': self.world_model.get_stats(),
        }