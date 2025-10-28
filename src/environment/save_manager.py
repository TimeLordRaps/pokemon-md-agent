"""Save state management for Pokemon MD agent."""

from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import time

from .mgba_controller import MGBAController

logger = logging.getLogger(__name__)


@dataclass
class SaveSlotInfo:
    """Information about a save slot."""
    slot: int
    path: Path
    timestamp: float
    frame: Optional[int]
    description: Optional[str] = None


class SaveManager:
    """Manages save/load operations for Pokemon MD."""
    
    # Reserved slots
    SLOT_TITLE_SCREEN = 0  # Clean title screen for reset
    SLOT_FLOOR_READY = 1   # Floor ready for benchmark loops
    SLOT_AUTO = 2          # Last autosave
    
    def __init__(
        self,
        controller: MGBAController,
        save_dir: Path,
        auto_save_interval: int = 300,  # 5 minutes
    ):
        """Initialize save manager.
        
        Args:
            controller: mgba controller instance
            save_dir: Directory for save files
            auto_save_interval: Auto-save interval in seconds
        """
        self.controller = controller
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self._last_auto_save = 0.0
        self._slot_registry: Dict[int, SaveSlotInfo] = {}
        
        logger.info("SaveManager initialized with dir: %s", self.save_dir)
    
    def ensure_startable_state(self) -> bool:
        """Ensure the game is in a startable state.

        This will:
        1. Autoload save if available
        2. Load state slot 1 if available (for agent loops)
        3. Else noop

        Returns:
            True if game is startable
        """
        logger.info("Ensuring startable state...")

        # Try autoload save
        if self.controller.autoload_save():
            logger.info("Autoload save successful")
            time.sleep(0.5)  # Wait for autoload to complete

            # Load state slot 1 for agent loops
            if self.controller.load_state_slot(1, 0):
                logger.info("Loaded state slot 1")
                return True
            else:
                logger.info("Failed to load state slot 1")
        else:
            logger.info("No autoload save available")

        logger.warning("Could not ensure startable state")
        return False
    
    def save_slot(self, slot: int, description: Optional[str] = None) -> bool:
        """Save current state to slot.
        
        Args:
            slot: Save slot number (0-99)
            description: Optional description
            
        Returns:
            True if save successful
        """
        if not self.controller.is_connected():
            logger.error("Not connected to mgba")
            return False
        
        # Build save path
        slot_path = self.save_dir / f"slot_{slot:02d}.state"
        
        # Save state
        if self.controller.save_state_file(str(slot_path), slot):
            frame = self.controller.current_frame()
            
            # Update registry
            self._slot_registry[slot] = SaveSlotInfo(
                slot=slot,
                path=slot_path,
                timestamp=time.time(),
                frame=frame,
                description=description,
            )
            
            # Save registry
            self._save_slot_registry()
            
            logger.info("Saved to slot %d: %s", slot, description or "no description")
            return True
        
        logger.error("Failed to save slot %d", slot)
        return False
    
    def load_slot(self, slot: int) -> bool:
        """Load state from slot.
        
        Args:
            slot: Save slot number
            
        Returns:
            True if load successful
        """
        if not self.controller.is_connected():
            logger.error("Not connected to mgba")
            return False
        
        # Build save path
        slot_path = self.save_dir / f"slot_{slot:02d}.state"
        
        if not slot_path.exists():
            logger.warning("Save slot %d does not exist", slot)
            return False
        
        # Load state
        if self.controller.load_state_file(str(slot_path), slot):
            # Update registry
            if slot in self._slot_registry:
                self._slot_registry[slot].timestamp = time.time()
                self._slot_registry[slot].frame = self.controller.current_frame()
            
            logger.info("Loaded slot %d", slot)
            return True
        
        logger.error("Failed to load slot %d", slot)
        return False
    
    def auto_save_if_needed(self) -> bool:
        """Auto-save if interval has passed.
        
        Returns:
            True if auto-save performed
        """
        now = time.time()
        
        if now - self._last_auto_save >= self.auto_save_interval:
            logger.info("Performing auto-save")
            success = self.save_slot(
                self.SLOT_AUTO,
                description=f"Auto-save at frame {self.controller.current_frame()}"
            )
            
            if success:
                self._last_auto_save = now
                return True
        
        return False
    
    def list_slots(self) -> List[SaveSlotInfo]:
        """List all save slots.
        
        Returns:
            List of save slot info
        """
        # Refresh registry from disk
        self._load_slot_registry()
        
        # Filter to existing files
        existing_slots = []
        for slot_info in self._slot_registry.values():
            if slot_info.path.exists():
                existing_slots.append(slot_info)
        
        return sorted(existing_slots, key=lambda s: s.slot)
    
    def get_slot_info(self, slot: int) -> Optional[SaveSlotInfo]:
        """Get info for a specific slot.
        
        Args:
            slot: Save slot number
            
        Returns:
            SaveSlotInfo or None if not found
        """
        self._load_slot_registry()
        return self._slot_registry.get(slot)
    
    def delete_slot(self, slot: int) -> bool:
        """Delete a save slot.
        
        Args:
            slot: Save slot number
            
        Returns:
            True if deletion successful
        """
        slot_path = self.save_dir / f"slot_{slot:02d}.state"
        
        if slot_path.exists():
            try:
                slot_path.unlink()
                logger.info("Deleted slot %d", slot)
                
                # Remove from registry
                if slot in self._slot_registry:
                    del self._slot_registry[slot]
                    self._save_slot_registry()
                
                return True
            except OSError as e:
                logger.error("Failed to delete slot %d: %s", slot, e)
                return False
        else:
            logger.warning("Slot %d does not exist", slot)
            return False
    
    def backup_slot(self, slot: int, backup_dir: Optional[Path] = None) -> Optional[Path]:
        """Create a backup of a save slot.
        
        Args:
            slot: Save slot number
            backup_dir: Backup directory (defaults to save_dir/backup)
            
        Returns:
            Path to backup file or None if failed
        """
        if backup_dir is None:
            backup_dir = self.save_dir / "backup"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        slot_path = self.save_dir / f"slot_{slot:02d}.state"
        
        if not slot_path.exists():
            logger.warning("Slot %d does not exist", slot)
            return None
        
        # Create backup with timestamp
        timestamp = int(time.time())
        backup_path = backup_dir / f"slot_{slot:02d}_{timestamp}.state"
        
        try:
            # Copy file
            import shutil
            shutil.copy2(slot_path, backup_path)
            
            logger.info("Backed up slot %d to %s", slot, backup_path)
            return backup_path
            
        except OSError as e:
            logger.error("Failed to backup slot %d: %s", slot, e)
            return None
    
    def create_title_screen_slot(self) -> bool:
        """Create a save slot at the title screen.
        
        This should be called manually when the game is at title screen.
        The agent will use this slot for resetting.
        
        Returns:
            True if successful
        """
        logger.info("Creating title screen slot...")
        
        # Wait for user to navigate to title screen
        input("Press Enter when at title screen, then press Start to enter file select...")
        
        # Save to slot 0
        return self.save_slot(
            self.SLOT_TITLE_SCREEN,
            description="Title screen (for reset)"
        )
    
    def create_floor_ready_slot(self) -> bool:
        """Create a save slot ready for floor exploration.
        
        This should be called when the agent is positioned at the start of a dungeon floor.
        The agent will use this slot for benchmark loops.
        
        Returns:
            True if successful
        """
        logger.info("Creating floor-ready slot...")
        
        # Wait for user to navigate to floor start
        input("Press Enter when positioned at floor start...")
        
        # Save to slot 1
        return self.save_slot(
            self.SLOT_FLOOR_READY,
            description="Floor ready (for benchmark loops)"
        )
    
    def _load_slot_registry(self) -> None:
        """Load slot registry from disk."""
        registry_path = self.save_dir / "slot_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._slot_registry = {}
                for slot, info in data.items():
                    self._slot_registry[int(slot)] = SaveSlotInfo(**info)
                    
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.debug("Failed to load slot registry: %s", e)
                self._slot_registry = {}
    
    def _save_slot_registry(self) -> None:
        """Save slot registry to disk."""
        registry_path = self.save_dir / "slot_registry.json"
        
        try:
            data = {}
            for slot, info in self._slot_registry.items():
                data[str(slot)] = {
                    "slot": info.slot,
                    "path": str(info.path),
                    "timestamp": info.timestamp,
                    "frame": info.frame,
                    "description": info.description,
                }
            
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except OSError as e:
            logger.debug("Failed to save slot registry: %s", e)
    
    def reset_to_title_screen(self) -> bool:
        """Reset to title screen slot.
        
        Returns:
            True if successful
        """
        if not self.controller.is_connected():
            logger.error("Not connected to mgba")
            return False
        
        # Clear any stuck buttons first
        self.controller.button_clear_many([
            "A", "B", "Start", "Select", "Up", "Down", "Left", "Right", "L", "R"
        ])
        
        # Try to reset and load title screen
        if self.controller.reset():
            time.sleep(1.0)  # Wait for reset
            
            return self.load_slot(self.SLOT_TITLE_SCREEN)
        
        # If reset fails, just load title screen slot
        return self.load_slot(self.SLOT_TITLE_SCREEN)
