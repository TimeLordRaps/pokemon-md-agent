"""Action executor for sending button presses to mgba emulator."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time

from .mgba_controller import MGBAController

logger = logging.getLogger(__name__)


class Button(Enum):
    """Available controller buttons."""
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Action:
    """Represents a single action (button press)."""
    button: Button
    duration_ms: int = 100
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ActionSequence:
    """Sequence of actions to execute."""
    name: str
    actions: List[Action]
    delay_after_ms: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ActionExecutor:
    """Executes actions on the mgba emulator."""
    
    def __init__(
        self,
        mgba_controller: MGBAController,
        default_button_duration: int = 100,
        default_delay_ms: int = 50,
    ):
        """Initialize action executor.
        
        Args:
            mgba_controller: mgba controller instance
            default_button_duration: Default duration for button presses in ms
            default_delay_ms: Default delay between actions in ms
        """
        self.mgba = mgba_controller
        self.default_button_duration = default_button_duration
        self.default_delay_ms = default_delay_ms
        
        # Track execution statistics
        self.actions_executed = 0
        self.sequences_executed = 0
        self.last_action_time = 0.0
        
        logger.info(
            "Initialized ActionExecutor: duration=%dms, delay=%dms",
            default_button_duration,
            default_delay_ms
        )
    
    def press_button(
        self,
        button: Button,
        duration_ms: Optional[int] = None,
        delay_ms: int = 0,
    ) -> bool:
        """Press a single button.
        
        Args:
            button: Button to press
            duration_ms: How long to press (uses default if None)
            delay_ms: Delay before pressing
            
        Returns:
            True if press succeeded
        """
        duration = duration_ms or self.default_button_duration
        
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        
        success = self.mgba.button_tap(button.value)
        
        if success:
            self.actions_executed += 1
            self.last_action_time = time.time()
            
            logger.debug("Pressed %s for %dms", button.value, duration)
        else:
            logger.warning("Failed to press %s", button.value)
        
        return success
    
    def release_button(self, button: Button, delay_ms: int = 0) -> bool:
        """Release a button.
        
        Args:
            button: Button to release
            delay_ms: Delay before releasing
            
        Returns:
            True if release succeeded
        """
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        
        success = self.mgba.release_button(button.value)
        
        if success:
            logger.debug("Released %s", button.value)
        else:
            logger.warning("Failed to release %s", button.value)
        
        return success
    
    def execute_action(self, action: Action) -> bool:
        """Execute a single action.
        
        Args:
            action: Action to execute
            
        Returns:
            True if action succeeded
        """
        return self.press_button(
            action.button,
            action.duration_ms,
            0  # Action already has timestamp, no additional delay
        )
    
    def execute_sequence(
        self,
        sequence: ActionSequence,
        delay_after: bool = True,
    ) -> bool:
        """Execute an action sequence.
        
        Args:
            sequence: Sequence of actions to execute
            delay_after: Whether to apply delay after sequence
            
        Returns:
            True if all actions succeeded
        """
        logger.debug("Executing action sequence: %s", sequence.name)
        
        all_succeeded = True
        
        for i, action in enumerate(sequence.actions):
            # Add small delay between actions (except for first)
            if i > 0:
                time.sleep(self.default_delay_ms / 1000.0)
            
            success = self.execute_action(action)
            
            if not success:
                all_succeeded = False
                logger.error(
                    "Action %d/%d failed in sequence %s",
                    i + 1,
                    len(sequence.actions),
                    sequence.name
                )
                break
        
        if all_succeeded:
            self.sequences_executed += 1
            
            if sequence.delay_after_ms > 0 and delay_after:
                time.sleep(sequence.delay_after_ms / 1000.0)
            
            logger.info(
                "Completed sequence %s (%d actions)",
                sequence.name,
                len(sequence.actions)
            )
        else:
            logger.error("Sequence %s failed", sequence.name)
        
        return all_succeeded
    
    def move_up(self, steps: int = 1, delay_ms: int = 100) -> bool:
        """Move up specified number of steps.
        
        Args:
            steps: Number of up movements
            delay_ms: Delay between steps
            
        Returns:
            True if all movements succeeded
        """
        return self._move_direction(Button.UP, steps, delay_ms)
    
    def move_down(self, steps: int = 1, delay_ms: int = 100) -> bool:
        """Move down specified number of steps.
        
        Args:
            steps: Number of down movements
            delay_ms: Delay between steps
            
        Returns:
            True if all movements succeeded
        """
        return self._move_direction(Button.DOWN, steps, delay_ms)
    
    def move_left(self, steps: int = 1, delay_ms: int = 100) -> bool:
        """Move left specified number of steps.
        
        Args:
            steps: Number of left movements
            delay_ms: Delay between steps
            
        Returns:
            True if all movements succeeded
        """
        return self._move_direction(Button.LEFT, steps, delay_ms)
    
    def move_right(self, steps: int = 1, delay_ms: int = 100) -> bool:
        """Move right specified number of steps.
        
        Args:
            steps: Number of right movements
            delay_ms: Delay between steps
            
        Returns:
            True if all movements succeeded
        """
        return self._move_direction(Button.RIGHT, steps, delay_ms)
    
    def _move_direction(
        self,
        direction: Button,
        steps: int,
        delay_ms: int,
    ) -> bool:
        """Move in a direction for specified steps.
        
        Args:
            direction: Direction to move
            steps: Number of steps
            delay_ms: Delay between steps
            
        Returns:
            True if all movements succeeded
        """
        if steps <= 0:
            return True
        
        all_succeeded = True
        
        for step in range(steps):
            success = self.press_button(direction, delay_ms=delay_ms)
            
            if not success:
                all_succeeded = False
                logger.error("Move %s step %d/%d failed", direction.value, step + 1, steps)
                break
            
            # Add delay between steps
            if step < steps - 1 and delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
        
        if all_succeeded:
            logger.debug("Moved %s %d steps", direction.value, steps)
        
        return all_succeeded
    
    def interact(self, delay_ms: int = 100, textbox_pacing: bool = False) -> bool:
        """Press A button to interact.

        Args:
            delay_ms: Delay before interaction
            textbox_pacing: If True, throttle for OCR capture during textboxes

        Returns:
            True if interaction succeeded
        """
        if textbox_pacing:
            # Throttle A taps during textboxes to ensure ≥1 fps OCR capture
            pacing_delay = 1000  # 1 second minimum between taps
            logger.debug("Textbox pacing enabled, using %dms delay", pacing_delay)
            return self.press_button(Button.A, delay_ms=pacing_delay)
        return self.press_button(Button.A, delay_ms=delay_ms)
    
    def cancel(self, delay_ms: int = 100) -> bool:
        """Press B button to cancel.
        
        Args:
            delay_ms: Delay before cancel
            
        Returns:
            True if cancel succeeded
        """
        return self.press_button(Button.B, delay_ms=delay_ms)
    
    def open_menu(self, delay_ms: int = 100) -> bool:
        """Press Start to open menu.
        
        Args:
            delay_ms: Delay before opening menu
            
        Returns:
            True if menu open succeeded
        """
        return self.press_button(Button.START, delay_ms=delay_ms)
    
    def wait(self, duration_ms: int) -> None:
        """Wait for specified duration.
        
        Args:
            duration_ms: Duration to wait in milliseconds
        """
        time.sleep(duration_ms / 1000.0)
        logger.debug("Waited %dms", duration_ms)
    
    def create_navigation_sequence(
        self,
        directions: List[str],
        delay_ms: int = 100,
    ) -> ActionSequence:
        """Create navigation sequence from direction list.
        
        Args:
            directions: List of directions ("up", "down", "left", "right")
            delay_ms: Delay between movements
            
        Returns:
            ActionSequence for navigation
        """
        actions = []
        
        for direction in directions:
            direction = direction.lower()
            
            if direction == "up":
                button = Button.UP
            elif direction == "down":
                button = Button.DOWN
            elif direction == "left":
                button = Button.LEFT
            elif direction == "right":
                button = Button.RIGHT
            else:
                logger.warning("Unknown direction: %s", direction)
                continue
            
            action = Action(button=button, duration_ms=delay_ms)
            actions.append(action)
        
        sequence = ActionSequence(
            name="navigation",
            actions=actions,
            delay_after_ms=0,
            metadata={"type": "navigation", "directions": directions}
        )
        
        return sequence
    
    def execute_navigation(
        self,
        directions: List[str],
        delay_ms: int = 100,
    ) -> bool:
        """Execute navigation sequence.
        
        Args:
            directions: List of directions to move
            delay_ms: Delay between movements
            
        Returns:
            True if navigation succeeded
        """
        sequence = self.create_navigation_sequence(directions, delay_ms)
        return self.execute_sequence(sequence)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "actions_executed": self.actions_executed,
            "sequences_executed": self.sequences_executed,
            "default_button_duration": self.default_button_duration,
            "default_delay_ms": self.default_delay_ms,
            "last_action_time": self.last_action_time,
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.actions_executed = 0
        self.sequences_executed = 0
        self.last_action_time = 0.0
        logger.debug("Reset execution statistics")
