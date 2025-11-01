"""Game state reset and checkpoint recovery system.

Handles resetting to title screen, loading from ROM, and recovering
from saved checkpoints for fresh starts or mid-run recovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GameResetMode(Enum):
    """Enum for different game reset strategies."""

    FRESH_ROM = "fresh_rom"  # Start new game from ROM
    CHECKPOINT_LOAD = "checkpoint_load"  # Load from .ss0 checkpoint
    SOFT_RESET = "soft_reset"  # Reset via in-game menu to title
    TITLE_SCREEN = "title_screen"  # Just reach title, don't start game


@dataclass
class GameResetConfig:
    """Configuration for game reset behavior."""

    reset_mode: GameResetMode = GameResetMode.TITLE_SCREEN
    checkpoint_file: Optional[Path] = None  # For checkpoint_load mode
    rom_path: Optional[Path] = None  # For fresh_rom mode
    auto_start_game: bool = False  # Whether to auto-start new game
    soft_reset_button_sequence: list = None  # Custom button sequence for soft reset

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.soft_reset_button_sequence is None:
            # Default: Menu → Reset
            self.soft_reset_button_sequence = ["Start", "A"]


@dataclass
class GameState:
    """Current game state information."""

    is_at_title_screen: bool = False
    is_in_game: bool = False
    current_hp: Optional[int] = None
    dungeon_level: Optional[int] = None
    position: Optional[tuple] = None
    frame_count: int = 0
    last_action: Optional[str] = None
    reset_detected: bool = False  # Whether a reset was detected


class GameResetHandler:
    """Handle game state resets and checkpoint recovery."""

    def __init__(
        self,
        controller: Optional[object] = None,
        config: Optional[GameResetConfig] = None
    ):
        """Initialize game reset handler.

        Args:
            controller: MGBAController instance for emulator control
            config: GameResetConfig for reset behavior
        """
        self.controller = controller
        self.config = config or GameResetConfig()
        self.current_state = GameState()
        self.reset_history: list = []
        self._reset_in_progress = False

    def update_game_state(
        self,
        is_at_title: bool = False,
        is_in_game: bool = False,
        hp: Optional[int] = None,
        dungeon_level: Optional[int] = None,
        position: Optional[tuple] = None,
        frame_count: int = 0,
        action: Optional[str] = None
    ) -> None:
        """Update current game state tracking.

        Args:
            is_at_title: Whether at title screen
            is_in_game: Whether actively in a dungeon
            hp: Current HP value
            dungeon_level: Current dungeon level
            position: Current position tuple
            frame_count: Current frame number
            action: Last action taken
        """
        self.current_state.is_at_title_screen = is_at_title
        self.current_state.is_in_game = is_in_game
        self.current_state.current_hp = hp
        self.current_state.dungeon_level = dungeon_level
        self.current_state.position = position
        self.current_state.frame_count = frame_count
        self.current_state.last_action = action

    def detect_reset(self, frame_delta: int, state_hash: Optional[str] = None) -> bool:
        """Detect if a game reset occurred.

        Args:
            frame_delta: Change in frame count since last check
            state_hash: Optional hash of game state for comparison

        Returns:
            True if reset was detected
        """
        # Reset detected if frame count went backwards significantly
        if frame_delta < -1000:
            self.current_state.reset_detected = True
            logger.info("Reset detected: frame count decreased significantly")
            return True

        # Reset detected if returned to title screen unexpectedly
        if self.current_state.is_at_title_screen and self.current_state.last_action:
            # Was in game, now at title (likely crashed or reset)
            logger.info("Reset detected: unexpectedly returned to title screen")
            self.current_state.reset_detected = True
            return True

        return False

    async def wait_for_title_screen(
        self,
        timeout: float = 30.0,
        check_interval: float = 0.5
    ) -> bool:
        """Wait for game to reach title screen.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check state in seconds

        Returns:
            True if reached title screen, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.current_state.is_at_title_screen:
                logger.info("Title screen reached")
                return True

            await asyncio.sleep(check_interval)

        logger.warning(f"Timeout waiting for title screen ({timeout}s)")
        return False

    async def reset_to_title_screen(self) -> bool:
        """Reset game to title screen.

        Attempts to use soft reset (menu-based) if available,
        otherwise relies on game shutdown/restart.

        Returns:
            True if reset successful
        """
        if self._reset_in_progress:
            logger.warning("Reset already in progress")
            return False

        self._reset_in_progress = True

        try:
            if not self.controller:
                logger.error("No controller available for reset")
                return False

            logger.info("Resetting to title screen...")

            # Execute soft reset button sequence
            for button in self.config.soft_reset_button_sequence:
                try:
                    self.controller.button_tap(button)
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"Failed to press {button}: {e}")

            # Wait for title screen
            if await self.wait_for_title_screen():
                self.current_state.is_in_game = False
                self.reset_history.append({
                    "timestamp": time.time(),
                    "mode": "soft_reset",
                    "success": True
                })
                return True
            else:
                logger.error("Failed to reach title screen after reset")
                return False

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            return False

        finally:
            self._reset_in_progress = False

    async def load_checkpoint(
        self,
        checkpoint_path: Path,
        timeout: float = 10.0
    ) -> bool:
        """Load game from a checkpoint file.

        Args:
            checkpoint_path: Path to .ss0 or save file
            timeout: Maximum time to wait for load

        Returns:
            True if load successful
        """
        if not self.controller:
            logger.error("No controller available for checkpoint load")
            return False

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            # Request load via controller
            # Note: Actual implementation depends on mGBA Lua API
            # This is a placeholder for the conceptual interface
            if hasattr(self.controller, 'load_save'):
                self.controller.load_save(str(checkpoint_path))
            else:
                logger.warning("Controller doesn't support load_save, skipping")

            # Wait for load to complete
            await asyncio.sleep(timeout)

            self.current_state.is_in_game = True
            self.reset_history.append({
                "timestamp": time.time(),
                "mode": "checkpoint_load",
                "checkpoint": str(checkpoint_path),
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    async def start_new_game(self, timeout: float = 30.0) -> bool:
        """Start a new game from title screen.

        Assumes game is at title screen and navigates
        through new game menus.

        Args:
            timeout: Maximum time for new game startup

        Returns:
            True if game started successfully
        """
        if not self.controller:
            logger.error("No controller available for game start")
            return False

        if not self.current_state.is_at_title_screen:
            logger.warning("Not at title screen, cannot start new game")
            return False

        try:
            logger.info("Starting new game...")

            # Standard menu navigation: A to start, A again to confirm
            self.controller.button_tap("A")
            await asyncio.sleep(1.0)
            self.controller.button_tap("A")
            await asyncio.sleep(2.0)

            # Wait for game to be playable
            await asyncio.sleep(timeout)

            self.current_state.is_in_game = True
            return True

        except Exception as e:
            logger.error(f"Failed to start new game: {e}")
            return False

    async def perform_reset(
        self,
        mode: Optional[GameResetMode] = None
    ) -> bool:
        """Perform a game reset according to configuration.

        Args:
            mode: Override reset mode for this operation

        Returns:
            True if reset successful
        """
        reset_mode = mode or self.config.reset_mode

        logger.info(f"Performing reset: {reset_mode.value}")

        try:
            if reset_mode == GameResetMode.TITLE_SCREEN:
                success = await self.reset_to_title_screen()

            elif reset_mode == GameResetMode.SOFT_RESET:
                success = await self.reset_to_title_screen()

            elif reset_mode == GameResetMode.CHECKPOINT_LOAD:
                if not self.config.checkpoint_file:
                    logger.error("Checkpoint load requested but no checkpoint specified")
                    return False
                success = await self.load_checkpoint(self.config.checkpoint_file)

            elif reset_mode == GameResetMode.FRESH_ROM:
                # For fresh ROM, we'd need emulator restart
                # This is typically handled by the harness, not the agent
                logger.info("Fresh ROM mode: emulator restart needed externally")
                success = True

            else:
                logger.error(f"Unknown reset mode: {reset_mode}")
                return False

            if success and self.config.auto_start_game:
                success = await self.start_new_game()

            return success

        except Exception as e:
            logger.error(f"Error during reset operation: {e}")
            return False

    def get_reset_status(self) -> dict:
        """Get current reset status and history.

        Returns:
            Dictionary with status information
        """
        return {
            "current_state": {
                "at_title_screen": self.current_state.is_at_title_screen,
                "in_game": self.current_state.is_in_game,
                "hp": self.current_state.current_hp,
                "dungeon_level": self.current_state.dungeon_level,
                "frame_count": self.current_state.frame_count,
                "reset_detected": self.current_state.reset_detected
            },
            "reset_history": self.reset_history[-5:],  # Last 5 resets
            "total_resets": len(self.reset_history),
            "reset_mode": self.config.reset_mode.value
        }

    async def recovery_flow(
        self,
        target_mode: GameResetMode = GameResetMode.CHECKPOINT_LOAD
    ) -> bool:
        """Execute a full recovery flow for mid-run recovery.

        This handles resetting and getting back to a playable state,
        typically after detecting a crash or hung state.

        Args:
            target_mode: What mode to end up in

        Returns:
            True if recovery successful
        """
        logger.info("Starting recovery flow...")

        try:
            # First, soft reset to title
            if not await self.reset_to_title_screen():
                logger.error("Failed to reset to title during recovery")
                return False

            # Then load from checkpoint if specified
            if target_mode == GameResetMode.CHECKPOINT_LOAD:
                if not self.config.checkpoint_file:
                    logger.warning("No checkpoint specified, skipping load")
                else:
                    if not await self.load_checkpoint(self.config.checkpoint_file):
                        logger.error("Failed to load checkpoint during recovery")
                        return False

            # Finally, start new game if needed
            if self.config.auto_start_game and target_mode == GameResetMode.FRESH_ROM:
                if not await self.start_new_game():
                    logger.error("Failed to start new game during recovery")
                    return False

            logger.info("Recovery flow completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during recovery flow: {e}")
            return False
