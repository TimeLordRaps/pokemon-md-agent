"""RAM watcher for streaming real-time game state updates."""

import asyncio
import time
from typing import AsyncGenerator, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from .ram_decoders import RAMSnapshot
from .mgba_controller import MGBAController

logger = logging.getLogger(__name__)


@dataclass
class RAMChange:
    """Represents a change in RAM state."""
    timestamp: float
    field_path: str  # e.g., "player_state.player_tile_x"
    old_value: Any
    new_value: Any
    frame: Optional[int] = None


@dataclass
class WatcherConfig:
    """Configuration for RAM watcher."""
    poll_interval: float = 0.1  # 100ms polling
    watch_fields: set = field(default_factory=lambda: {
        "player_state.player_tile_x",
        "player_state.player_tile_y", 
        "player_state.floor_number",
        "player_state.turn_counter",
        "party_status.leader_hp",
        "party_status.partner_hp",
        "map_data.stairs_x",
        "map_data.stairs_y",
    })
    callback: Optional[Callable[[RAMChange], None]] = None
    output_dir: Optional[Path] = None


class RAMWatcher:
    """Watches RAM changes and streams updates."""
    
    def __init__(
        self,
        controller: MGBAController,
        decoder,
        config: Optional[WatcherConfig] = None
    ):
        """Initialize RAM watcher.
        
        Args:
            controller: mgba controller
            decoder: RAM decoder instance
            config: Watcher configuration
        """
        self.controller = controller
        self.decoder = decoder
        self.config = config or WatcherConfig()
        
        self._running = False
        self._previous_snapshot: Optional[RAMSnapshot] = None
        self._subscribers: list[Callable[[RAMChange], None]] = []
        
        # Output directory for change logs
        self.output_dir = self.config.output_dir or Path("data/ram_changes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAMWatcher initialized with poll interval: %.2fs", 
                   self.config.poll_interval)
    
    def subscribe(self, callback: Callable[[RAMChange], None]) -> None:
        """Subscribe to RAM changes.
        
        Args:
            callback: Function to call on changes
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[RAMChange], None]) -> None:
        """Unsubscribe from RAM changes.
        
        Args:
            callback: Function to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    async def watch_changes(self) -> AsyncGenerator[RAMChange, None]:
        """Watch for RAM changes.
        
        Yields:
            RAMChange objects when state changes
        """
        self._running = True
        
        logger.info("Starting RAM change watching...")
        
        while self._running:
            try:
                # Get current snapshot
                snapshot = self.decoder.get_full_snapshot()
                
                if snapshot and self._previous_snapshot:
                    # Compare with previous snapshot
                    changes = self._compare_snapshots(
                        self._previous_snapshot, 
                        snapshot
                    )
                    
                    for change in changes:
                        yield change
                        
                        # Notify subscribers
                        for callback in self._subscribers:
                            try:
                                callback(change)
                            except Exception as e:
                                logger.error("Subscriber callback failed: %s", e)
                
                self._previous_snapshot = snapshot
                
            except Exception as e:
                logger.error("Error during RAM watching: %s", e)
            
            # Poll interval
            await asyncio.sleep(self.config.poll_interval)
    
    def _compare_snapshots(
        self, 
        old: RAMSnapshot, 
        new: RAMSnapshot
    ) -> list[RAMChange]:
        """Compare two snapshots and return changes.
        
        Args:
            old: Previous snapshot
            new: Current snapshot
            
        Returns:
            List of RAMChange objects
        """
        changes = []
        
        # Player state changes
        if old.player_state.floor_number != new.player_state.floor_number:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="player_state.floor_number",
                old_value=old.player_state.floor_number,
                new_value=new.player_state.floor_number,
                frame=new.frame
            ))
        
        if old.player_state.player_tile_x != new.player_state.player_tile_x:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="player_state.player_tile_x",
                old_value=old.player_state.player_tile_x,
                new_value=new.player_state.player_tile_x,
                frame=new.frame
            ))
        
        if old.player_state.player_tile_y != new.player_state.player_tile_y:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="player_state.player_tile_y",
                old_value=old.player_state.player_tile_y,
                new_value=new.player_state.player_tile_y,
                frame=new.frame
            ))
        
        if old.player_state.turn_counter != new.player_state.turn_counter:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="player_state.turn_counter",
                old_value=old.player_state.turn_counter,
                new_value=new.player_state.turn_counter,
                frame=new.frame
            ))
        
        # Party status changes
        if old.party_status.leader_hp != new.party_status.leader_hp:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="party_status.leader_hp",
                old_value=old.party_status.leader_hp,
                new_value=new.party_status.leader_hp,
                frame=new.frame
            ))
        
        if old.party_status.partner_hp != new.party_status.partner_hp:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="party_status.partner_hp",
                old_value=old.party_status.partner_hp,
                new_value=new.party_status.partner_hp,
                frame=new.frame
            ))
        
        if old.party_status.leader_belly != new.party_status.leader_belly:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="party_status.leader_belly",
                old_value=old.party_status.leader_belly,
                new_value=new.party_status.leader_belly,
                frame=new.frame
            ))
        
        if old.party_status.partner_belly != new.party_status.partner_belly:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="party_status.partner_belly",
                old_value=old.party_status.partner_belly,
                new_value=new.party_status.partner_belly,
                frame=new.frame
            ))
        
        # Map data changes
        if old.map_data.stairs_x != new.map_data.stairs_x:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="map_data.stairs_x",
                old_value=old.map_data.stairs_x,
                new_value=new.map_data.stairs_x,
                frame=new.frame
            ))
        
        if old.map_data.stairs_y != new.map_data.stairs_y:
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="map_data.stairs_y",
                old_value=old.map_data.stairs_y,
                new_value=new.map_data.stairs_y,
                frame=new.frame
            ))
        
        # Entity changes (count and positions)
        if len(old.entities) != len(new.entities):
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="entities.count",
                old_value=len(old.entities),
                new_value=len(new.entities),
                frame=new.frame
            ))
        
        # Check for entity position changes
        for i, (old_e, new_e) in enumerate(zip(old.entities, new.entities)):
            if old_e.tile_x != new_e.tile_x or old_e.tile_y != new_e.tile_y:
                changes.append(RAMChange(
                    timestamp=new.timestamp,
                    field_path=f"entities[{i}].position",
                    old_value=(old_e.tile_x, old_e.tile_y),
                    new_value=(new_e.tile_x, new_e.tile_y),
                    frame=new.frame
                ))
        
        # Item changes
        if len(old.items) != len(new.items):
            changes.append(RAMChange(
                timestamp=new.timestamp,
                field_path="items.count",
                old_value=len(old.items),
                new_value=len(new.items),
                frame=new.frame
            ))
        
        return changes
    
    def log_changes(self, changes: list[RAMChange]) -> Path:
        """Log changes to file.
        
        Args:
            changes: List of changes to log
            
        Returns:
            Path to log file
        """
        timestamp = int(time.time())
        log_file = self.output_dir / f"changes_{timestamp}.json"
        
        data = {
            "timestamp": time.time(),
            "changes": [
                {
                    "timestamp": c.timestamp,
                    "field_path": c.field_path,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "frame": c.frame,
                }
                for c in changes
            ]
        }
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug("Logged %d changes to %s", len(changes), log_file)
        return log_file
    
    def start_watching(
        self,
        callback: Optional[Callable[[RAMChange], None]] = None
    ) -> None:
        """Start watching changes in a background task.
        
        Args:
            callback: Optional callback for changes
        """
        if callback:
            self.subscribe(callback)
        
        if not self._running:
            self._running = True
            logger.info("Started RAM watcher")
    
    def stop_watching(self) -> None:
        """Stop watching changes."""
        self._running = False
        logger.info("Stopped RAM watcher")
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current state as dictionary.
        
        Returns:
            Current state dictionary or None
        """
        snapshot = self.decoder.get_full_snapshot()
        
        if not snapshot:
            return None
        
        return {
            "timestamp": snapshot.timestamp,
            "frame": snapshot.frame,
            "player": {
                "tile_x": snapshot.player_state.player_tile_x,
                "tile_y": snapshot.player_state.player_tile_y,
                "floor": snapshot.player_state.floor_number,
                "turn": snapshot.player_state.turn_counter,
            },
            "party": {
                "leader_hp": snapshot.party_status.leader_hp,
                "leader_hp_max": snapshot.party_status.leader_hp_max,
                "leader_belly": snapshot.party_status.leader_belly,
                "partner_hp": snapshot.party_status.partner_hp,
                "partner_hp_max": snapshot.party_status.partner_hp_max,
                "partner_belly": snapshot.party_status.partner_belly,
            },
            "map": {
                "stairs_x": snapshot.map_data.stairs_x,
                "stairs_y": snapshot.map_data.stairs_y,
            },
            "entities": [
                {
                    "species": e.species_id,
                    "tile_x": e.tile_x,
                    "tile_y": e.tile_y,
                    "hp": e.hp_current,
                    "visible": e.visible,
                }
                for e in snapshot.entities
            ],
            "items": [
                {
                    "item_id": i.item_id,
                    "tile_x": i.tile_x,
                    "tile_y": i.tile_y,
                    "quantity": i.quantity,
                }
                for i in snapshot.items
            ]
        }
    
    async def run_forever(self) -> None:
        """Run watcher forever until stopped."""
        logger.info("Starting RAM watcher forever loop...")
        
        async for change in self.watch_changes():
            # Log significant changes
            if change.field_path in {"player_state.player_tile_x", 
                                    "player_state.player_tile_y",
                                    "party_status.leader_hp"}:
                logger.info("RAM Change: %s = %s (was %s)", 
                           change.field_path, change.new_value, change.old_value)
