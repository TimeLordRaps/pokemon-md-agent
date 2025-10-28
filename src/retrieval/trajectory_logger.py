"""Trajectory logging for Pokemon MD agent.

Logs combat events, movement trajectories, and decision outcomes
for retrieval and analysis.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from enum import IntEnum

logger = logging.getLogger(__name__)


class RoomKind(IntEnum):
    """Types of rooms in dungeons."""
    ROOM = 0
    CORRIDOR = 1
    DEAD_END = 2
    SPECIAL = 3


class DistanceBucket(IntEnum):
    """Distance buckets for combat analysis."""
    CLOSE = 1      # 1 tile
    NEAR = 2        # 2 tiles
    MEDIUM = 3      # 3-5 tiles
    FAR = 4         # 6-10 tiles
    VERY_FAR = 5    # >10 tiles


@dataclass
class CombatEvent:
    """A single combat event."""
    timestamp: float
    frame: int
    floor: int
    dungeon_id: int
    room_kind: RoomKind
    species_id: int
    level: int
    distance_bucket: DistanceBucket
    move_used: str
    damage_dealt: int
    damage_taken: int
    status_proc: Optional[str] = None
    success: bool = False
    hp_before: int = 0
    hp_after: int = 0
    enemy_hp_before: int = 0
    enemy_hp_after: int = 0


@dataclass
class MovementTrajectory:
    """A movement trajectory segment."""
    timestamp: float
    frame_start: int
    frame_end: int
    floor: int
    dungeon_id: int
    path: List[Dict[str, int]]  # [{"x": x, "y": y}, ...]
    duration_seconds: float
    distance_tiles: int
    objective: Optional[str] = None  # "explore", "combat", "stairs", etc.


@dataclass
class DecisionLog:
    """A decision and its outcome."""
    timestamp: float
    frame: int
    floor: int
    dungeon_id: int
    model_used: str  # "2B", "4B", "8B"
    confidence: float
    action: str
    reasoning: str
    outcome_success: Optional[bool] = None
    stuck_counter: int = 0
    shallow_hits_used: int = 0
    web_fetches_used: int = 0


class TrajectoryLogger:
    """Logs trajectories and combat events for analysis."""
    
    def __init__(self, log_dir: Path, max_file_size_mb: int = 10):
        """Initialize trajectory logger.
        
        Args:
            log_dir: Directory to store log files
            max_file_size_mb: Maximum size per log file before rotation
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        
        # Current log files
        self.combat_log = self.log_dir / "combat_events.jsonl"
        self.trajectory_log = self.log_dir / "trajectories.jsonl"
        self.decision_log = self.log_dir / "decisions.jsonl"
        
        # Rolling success rates per combat key
        self.success_rates: Dict[str, List[bool]] = {}
        self.max_history_per_key = 20
        
        logger.info(f"TrajectoryLogger initialized with log dir: {log_dir}")
    
    def log_combat_event(self, event: CombatEvent) -> None:
        """Log a combat event.
        
        Args:
            event: Combat event to log
        """
        # Convert to dict and add key for retrieval
        event_dict = asdict(event)
        event_dict["combat_key"] = self._make_combat_key(event)
        
        # Append to log
        self._append_json_line(self.combat_log, event_dict)
        
        # Update success rates
        key = event_dict["combat_key"]
        if key not in self.success_rates:
            self.success_rates[key] = []
        
        self.success_rates[key].append(event.success)
        
        # Keep only recent history
        if len(self.success_rates[key]) > self.max_history_per_key:
            self.success_rates[key] = self.success_rates[key][-self.max_history_per_key:]
        
        logger.debug(f"Logged combat event: {event.species_id} lvl{event.level} with {event.move_used}")
    
    def log_trajectory(self, trajectory: MovementTrajectory) -> None:
        """Log a movement trajectory.
        
        Args:
            trajectory: Movement trajectory to log
        """
        trajectory_dict = asdict(trajectory)
        self._append_json_line(self.trajectory_log, trajectory_dict)
        
        logger.debug(f"Logged trajectory: {len(trajectory.path)} steps, {trajectory.distance_tiles} tiles")
    
    def log_decision(self, decision: DecisionLog) -> None:
        """Log a decision.
        
        Args:
            decision: Decision to log
        """
        decision_dict = asdict(decision)
        self._append_json_line(self.decision_log, decision_dict)
        
        logger.debug(f"Logged decision: {decision.model_used} -> {decision.action}")
    
    def get_success_rate(self, species_id: int, floor_range: tuple[int, int], 
                        room_kind: RoomKind, distance_bucket: DistanceBucket, 
                        move_used: str) -> Optional[float]:
        """Get success rate for a combat scenario.
        
        Args:
            species_id: Pokemon species ID
            floor_range: (min_floor, max_floor)
            room_kind: Type of room
            distance_bucket: Distance bucket
            move_used: Move used
            
        Returns:
            Success rate (0.0-1.0) or None if no data
        """
        key = self._make_combat_key_from_params(
            species_id, floor_range, room_kind, distance_bucket, move_used
        )
        
        if key not in self.success_rates or not self.success_rates[key]:
            return None
        
        successes = sum(self.success_rates[key])
        total = len(self.success_rates[key])
        return successes / total
    
    def get_recent_combat_events(self, species_id: int, limit: int = 5) -> List[CombatEvent]:
        """Get recent combat events for a species.
        
        Args:
            species_id: Pokemon species ID
            limit: Maximum number of events to return
            
        Returns:
            List of recent combat events
        """
        events = []
        
        try:
            with open(self.combat_log, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    event_dict = json.loads(line)
                    if event_dict.get("species_id") == species_id:
                        # Convert back to CombatEvent
                        event = CombatEvent(**{k: v for k, v in event_dict.items() 
                                              if k in CombatEvent.__dataclass_fields__})
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
        except FileNotFoundError:
            pass
        
        return events
    
    def _make_combat_key(self, event: CombatEvent) -> str:
        """Make a combat key from event."""
        floor_min = max(1, event.floor - 2)
        floor_max = event.floor + 2
        return self._make_combat_key_from_params(
            event.species_id, (floor_min, floor_max), 
            event.room_kind, event.distance_bucket, event.move_used
        )
    
    def _make_combat_key_from_params(self, species_id: int, floor_range: tuple[int, int],
                                   room_kind: RoomKind, distance_bucket: DistanceBucket,
                                   move_used: str) -> str:
        """Make a combat key from parameters."""
        return f"{species_id}_{floor_range[0]}-{floor_range[1]}_{room_kind.value}_{distance_bucket.value}_{move_used}"
    
    def _append_json_line(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Append a JSON line to a file.
        
        Args:
            file_path: File to append to
            data: Data to serialize
        """
        # Check if we need to rotate the file
        if file_path.exists() and file_path.stat().st_size > self.max_file_size:
            self._rotate_file(file_path)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, default=str)
            f.write('\n')
    
    def _rotate_file(self, file_path: Path) -> None:
        """Rotate a log file by renaming with timestamp.
        
        Args:
            file_path: File to rotate
        """
        timestamp = int(time.time())
        rotated_path = file_path.with_suffix(f".{timestamp}.jsonl")
        
        try:
            file_path.rename(rotated_path)
            logger.info(f"Rotated log file: {file_path} -> {rotated_path}")
        except Exception as e:
            logger.warning(f"Failed to rotate log file {file_path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            "combat_events_logged": 0,
            "trajectories_logged": 0,
            "decisions_logged": 0,
            "success_rates_tracked": len(self.success_rates),
        }
        
        # Count lines in each file
        for file_path, key in [
            (self.combat_log, "combat_events_logged"),
            (self.trajectory_log, "trajectories_logged"),
            (self.decision_log, "decisions_logged"),
        ]:
            try:
                with open(file_path, 'r') as f:
                    stats[key] = sum(1 for line in f if line.strip())
            except FileNotFoundError:
                stats[key] = 0
        
        return stats