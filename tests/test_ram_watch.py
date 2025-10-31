"""Tests for RAM watcher."""

import asyncio
import json
import struct
import tempfile
from pathlib import Path

import pytest

from src.environment.ram_decoders import create_decoder
from src.environment.ram_watch import RAMWatcher, create_ram_watcher, FieldDelta


class TestRAMWatcher:
    """Test RAM watcher functionality."""

    @pytest.fixture
    def decoder(self):
        """Create decoder instance."""
        return create_decoder()

    @pytest.fixture
    def watcher(self, decoder):
        """Create RAM watcher instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = RAMWatcher(decoder, snapshot_interval=10)
            watcher.snapshots_dir = Path(tmpdir) / "snapshots"
            watcher.snapshots_dir.mkdir()
            yield watcher

    @pytest.fixture
    def sample_ram_sequence(self):
        """Generate sequence of RAM data with changes."""
        # Initial state
        data1 = bytearray(65536)
        data1[33544] = 1  # Floor 1
        struct.pack_into('<H', data1, 33548, 0)  # Turn 0

        # State 2: floor change
        data2 = bytearray(data1)
        data2[33544] = 2  # Floor 2

        # State 3: turn change
        data3 = bytearray(data2)
        struct.pack_into('<H', data3, 33548, 15)  # Turn 15

        # State 4: another floor change
        data4 = bytearray(data3)
        data4[33544] = 3  # Floor 3

        return [bytes(data1), bytes(data2), bytes(data3), bytes(data4)]

    def test_field_delta_creation(self):
        """Test field delta creation."""
        delta = FieldDelta("player_state.floor_number", 1, 2)
        assert delta.field_path == "player_state.floor_number"
        assert delta.old_value == 1
        assert delta.new_value == 2

    def test_compute_deltas(self, watcher, sample_ram_sequence):
        """Test delta computation."""
        state1 = watcher.decoder.decode_all(sample_ram_sequence[0])
        state2 = watcher.decoder.decode_all(sample_ram_sequence[1])

        deltas = watcher._compute_deltas(state1, state2)

        # Should have floor change delta
        floor_deltas = [d for d in deltas if "floor_number" in d.field_path]
        assert len(floor_deltas) == 1
        assert floor_deltas[0].old_value == 1
        assert floor_deltas[0].new_value == 2

    def test_should_snapshot_floor_change(self, watcher, sample_ram_sequence):
        """Test snapshot triggering on floor change."""
        # Set initial state
        watcher.last_state = watcher.decoder.decode_all(sample_ram_sequence[0])

        # Floor change should trigger snapshot
        new_state = watcher.decoder.decode_all(sample_ram_sequence[1])
        assert watcher._should_snapshot(new_state)

    def test_should_snapshot_turn_interval(self, watcher, sample_ram_sequence):
        """Test snapshot triggering on turn interval."""
        # Set initial state with turn 0
        watcher.last_state = watcher.decoder.decode_all(sample_ram_sequence[0])
        watcher.last_snapshot_turn = 0

        # Turn 15 should trigger snapshot (interval=10)
        new_state = watcher.decoder.decode_all(sample_ram_sequence[2])
        assert watcher._should_snapshot(new_state)

    def test_should_snapshot_no_trigger(self, watcher, sample_ram_sequence):
        """Test snapshot not triggering when conditions not met."""
        # Set initial state to floor 2
        watcher.last_state = watcher.decoder.decode_all(sample_ram_sequence[1])
        watcher.last_snapshot_turn = 10

        # Turn 15 with last snapshot at 10 should not trigger (interval=10, floor same)
        new_state = watcher.decoder.decode_all(sample_ram_sequence[2])
        assert not watcher._should_snapshot(new_state)

    @pytest.mark.asyncio
    async def test_watch_ram_stream(self, watcher, sample_ram_sequence):
        """Test watching RAM stream."""
        async def ram_stream():
            for data in sample_ram_sequence:
                yield data
                await asyncio.sleep(0.01)  # Small delay

        states_and_deltas = []
        async for state, deltas in watcher.watch_ram(ram_stream()):
            states_and_deltas.append((state, deltas))

        assert len(states_and_deltas) == 4

        # First state should have deltas (initial)
        assert len(states_and_deltas[0][1]) > 0

        # Subsequent states should have floor/turn changes
        assert any("floor_number" in d.field_path for d in states_and_deltas[1][1])
        assert any("turn_counter" in d.field_path for d in states_and_deltas[2][1])

    @pytest.mark.asyncio
    async def test_create_ram_watcher(self):
        """Test RAM watcher creation."""
        watcher = await create_ram_watcher(snapshot_interval=50)
        assert isinstance(watcher, RAMWatcher)
        assert watcher.snapshot_interval == 50

    def test_snapshot_saving(self, watcher, sample_ram_sequence):
        """Test snapshot file saving."""
        state = watcher.decoder.decode_all(sample_ram_sequence[1])

        watcher._save_snapshot(state, sample_ram_sequence[1])

        # Check files were created
        json_files = list(watcher.snapshots_dir.glob("*.ram.json"))
        bin_files = list(watcher.snapshots_dir.glob("*.bin"))

        assert len(json_files) == 1
        assert len(bin_files) == 1

        # Verify JSON content
        with open(json_files[0], 'r') as f:
            saved_state = json.load(f)
        assert saved_state["player_state"]["floor_number"] == 2