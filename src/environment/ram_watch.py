"""Async RAM watcher for PMD Red Rescue Team.

Streams decoded game state updates with field deltas and triggers snapshots.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TYPE_CHECKING

from .ram_decoders import PMDRedDecoder, create_decoder

if TYPE_CHECKING:
    from ..agent.agent_config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class FieldDelta:
    """Change in a specific field."""
    field_path: str
    old_value: Any
    new_value: Any


class RAMWatcher:
    """Async RAM watcher that streams state updates and handles snapshots."""

    def __init__(self, decoder: PMDRedDecoder, snapshot_interval: int = 100, config: Optional["AgentConfig"] = None):
        """Initialize RAM watcher.

        Args:
            decoder: PMD Red decoder instance
            snapshot_interval: Turns between snapshots (0 = disable)
            config: Agent configuration with thresholds and backoff settings
        """
        self.decoder = decoder
        self.snapshot_interval = snapshot_interval
        self.config = config
        self.last_snapshot_turn = 0
        self.last_state: Optional[Dict[str, Any]] = None
        self.snapshots_dir = Path("snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

        # Backoff tracking for skill triggers
        self.last_belly_trigger_time = 0.0
        self.last_hp_trigger_time = 0.0

    @property
    def belly_pct(self) -> float:
        """Get current belly percentage (0.0-1.0)."""
        if self.last_state is None:
            return 1.0

        party_status = self.last_state.get("party_status", {})
        leader = party_status.get("leader", {})
        belly = leader.get("belly", 100)
        return min(belly / 200.0, 1.0)  # Assume max belly is 200

    @property
    def hp_pct(self) -> float:
        """Get current HP percentage (0.0-1.0)."""
        if self.last_state is None:
            return 1.0

        party_status = self.last_state.get("party_status", {})
        leader = party_status.get("leader", {})
        hp = leader.get("hp", 100)
        hp_max = leader.get("hp_max", 100)
        return hp / hp_max if hp_max > 0 else 1.0

    def should_trigger_belly(self, threshold_pct: float) -> bool:
        """Check if belly trigger should activate with backoff logic."""
        if not self.config or not self.config.enable_skill_triggers:
            return False

        if self.belly_pct <= threshold_pct:
            current_time = time.time()
            if current_time - self.last_belly_trigger_time >= self.config.skill_backoff_seconds:
                self.last_belly_trigger_time = current_time
                return True

        return False

    def should_trigger_hp(self, threshold_pct: float) -> bool:
        """Check if HP trigger should activate with backoff logic."""
        if not self.config or not self.config.enable_skill_triggers:
            return False

        if self.hp_pct <= threshold_pct:
            current_time = time.time()
            if current_time - self.last_hp_trigger_time >= self.config.skill_backoff_seconds:
                self.last_hp_trigger_time = current_time
                return True

        return False

    def _compute_deltas(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> List[FieldDelta]:
        """Compute field deltas between two states."""
        deltas = []

        def recurse(path: str, old: Any, new: Any):
            if isinstance(old, dict) and isinstance(new, dict):
                for key in set(old.keys()) | set(new.keys()):
                    recurse(f"{path}.{key}", old.get(key), new.get(key))
            elif isinstance(old, list) and isinstance(new, list):
                # Simple list comparison - could be enhanced for entity changes
                if old != new:
                    deltas.append(FieldDelta(path, old, new))
            else:
                if old != new:
                    deltas.append(FieldDelta(path, old, new))

        recurse("root", old_state, new_state)
        return deltas

    def _should_snapshot(self, state: Dict[str, Any]) -> bool:
        """Check if snapshot should be taken."""
        if self.snapshot_interval == 0:
            return False

        current_turn = state["player_state"]["turn_counter"]
        current_floor = state["player_state"]["floor_number"]

        # Snapshot on floor change or turn interval
        floor_changed = (self.last_state is not None and
                        self.last_state["player_state"]["floor_number"] != current_floor)
        turn_interval = (current_turn - self.last_snapshot_turn) >= self.snapshot_interval

        return floor_changed or turn_interval

    def _save_snapshot(self, state: Dict[str, Any], raw_bytes: bytes) -> None:
        """Save snapshot to disk."""
        player_state = state["player_state"]
        turn = player_state["turn_counter"]
        floor = player_state["floor_number"]

        # Save decoded JSON
        json_path = self.snapshots_dir / f"dungeon_{floor}_turn_{turn}.ram.json"
        with open(json_path, 'w') as f:
            json.dump(state, f, indent=2)

        # Save raw bytes
        bin_path = self.snapshots_dir / f"dungeon_{floor}_turn_{turn}.bin"
        with open(bin_path, 'wb') as f:
            f.write(raw_bytes)

        logger.info(f"Snapshot saved: floor {floor}, turn {turn}")

    async def watch_ram(self, ram_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[Tuple[Dict[str, Any], List[FieldDelta]], None]:
        """Watch RAM stream and yield state updates with deltas.

        Args:
            ram_stream: Async generator yielding raw RAM bytes

        Yields:
            Tuple of (current_state, deltas_since_last)
        """
        async for raw_bytes in ram_stream:
            try:
                current_state = self.decoder.decode_all(raw_bytes)

                # Compute deltas
                deltas = []
                if self.last_state is not None:
                    deltas = self._compute_deltas(self.last_state, current_state)
                else:
                    # First state - consider all fields as new
                    deltas = [FieldDelta("root", None, current_state)]

                # Check for snapshot
                if self._should_snapshot(current_state):
                    self._save_snapshot(current_state, raw_bytes)
                    self.last_snapshot_turn = current_state["player_state"]["turn_counter"]

                self.last_state = current_state
                yield current_state, deltas

            except Exception as e:
                logger.error(f"Error decoding RAM: {e}")
                continue


async def create_ram_watcher(snapshot_interval: int = 100, config: Optional["AgentConfig"] = None) -> RAMWatcher:
    """Create a RAM watcher with default decoder."""
    decoder = create_decoder()
    return RAMWatcher(decoder, snapshot_interval, config)