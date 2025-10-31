"""Tests for CircularBuffer class."""

import time
import json
import tempfile
import os
import numpy as np
import pytest
from unittest.mock import patch

from src.retrieval.circular_buffer import CircularBuffer, BufferEntry


class TestCircularBuffer:
    """Test cases for CircularBuffer."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        buffer = CircularBuffer()
        assert buffer.window_seconds == 3600.0  # 60 minutes
        assert buffer.max_entries == 108000  # 30 * 60 * 60
        assert len(buffer.buffer) == 0

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        buffer = CircularBuffer(window_seconds=1800.0, max_entries=50000)
        assert buffer.window_seconds == 1800.0
        assert buffer.max_entries == 50000

    def test_add_frame_basic(self):
        """Test basic frame addition."""
        buffer = CircularBuffer(max_entries=10)
        frame_data = np.array([[1, 2], [3, 4]])

        success = buffer.add_frame(frame_data)
        assert success is True
        assert len(buffer.buffer) == 1

        entry = buffer.buffer[0]
        assert entry.data.shape == (2, 2)
        assert entry.timestamp is not None
        assert entry.metadata == {}

    def test_add_frame_with_metadata(self):
        """Test frame addition with custom metadata."""
        buffer = CircularBuffer(max_entries=10)
        frame_data = np.array([1, 2, 3])
        metadata = {"fps": 30, "resolution": "640x480"}

        success = buffer.add_frame(frame_data, metadata=metadata)
        assert success is True
        assert len(buffer.buffer) == 1

        entry = buffer.buffer[0]
        assert entry.metadata == metadata

    def test_add_frame_with_timestamp(self):
        """Test frame addition with custom timestamp."""
        buffer = CircularBuffer(max_entries=10)
        frame_data = np.array([1, 2, 3])
        custom_timestamp = 1234567890.0

        success = buffer.add_frame(frame_data, timestamp=custom_timestamp)
        assert success is True
        assert len(buffer.buffer) == 1

        entry = buffer.buffer[0]
        assert entry.timestamp == custom_timestamp

    def test_rolling_window_eviction(self):
        """Test that old frames are evicted based on time window."""
        buffer = CircularBuffer(window_seconds=2.0, max_entries=10)

        # Add first frame
        buffer.add_frame(np.array([1]), timestamp=100.0)
        assert len(buffer.buffer) == 1

        # Add second frame within window (no eviction check yet since time hasn't advanced)
        with patch('time.time', return_value=100.5):  # Time hasn't advanced enough for eviction
            buffer.add_frame(np.array([2]), timestamp=101.0)
            assert len(buffer.buffer) == 2

        # Add third frame that causes first to be evicted (current time = 103.0, window = 2.0)
        # So frames older than 101.0 should be evicted
        with patch('time.time', return_value=103.0):
            buffer.add_frame(np.array([3]), timestamp=102.0)
            assert len(buffer.buffer) == 2  # Should have evicted the 100.0 timestamp frame

            # Check remaining timestamps
            timestamps = [entry.timestamp for entry in buffer.buffer]
            assert 100.0 not in timestamps  # Should be evicted
            assert 101.0 in timestamps
            assert 102.0 in timestamps

    def test_max_entries_limit(self):
        """Test that buffer respects max_entries limit."""
        buffer = CircularBuffer(max_entries=2)

        # Add frames with timestamps to prevent time-based eviction
        with patch('time.time', return_value=100.5):
            buffer.add_frame(np.array([1]), timestamp=100.0)
        with patch('time.time', return_value=101.5):
            buffer.add_frame(np.array([2]), timestamp=101.0)
        assert len(buffer.buffer) == 2

        # Try to add another frame - should fail since buffer is full
        with patch('time.time', return_value=102.5):
            success = buffer.add_frame(np.array([3]), timestamp=102.0)
        assert success is False
        assert len(buffer.buffer) == 2

    def test_get_entries_time_window_filtering(self):
        """Test get_entries with time window filtering."""
        buffer = CircularBuffer(max_entries=10)

        # Add frames with different timestamps
        with patch('time.time', return_value=100.5):
            buffer.add_frame(np.array([1]), timestamp=100.0)
        with patch('time.time', return_value=101.5):
            buffer.add_frame(np.array([2]), timestamp=101.0)
        with patch('time.time', return_value=102.5):
            buffer.add_frame(np.array([3]), timestamp=102.0)

        # Get entries from last 2 seconds (current time = 103.0)
        with patch('time.time', return_value=103.0):
            entries = buffer.get_entries(time_window=2.0)
            assert len(entries) == 2  # Should get frames at 101.0 and 102.0

            timestamps = [entry.timestamp for entry in entries]
            assert 100.0 not in timestamps  # Too old
            assert 101.0 in timestamps
            assert 102.0 in timestamps

    def test_get_buffer_stats(self):
        """Test buffer statistics reporting."""
        buffer = CircularBuffer(max_entries=10)

        # Empty buffer stats
        stats = buffer.get_buffer_stats()
        assert stats['current_entries'] == 0
        assert stats['max_entries'] == 10
        assert stats['window_seconds'] == 3600.0
        assert stats['oldest_timestamp'] is None
        assert stats['newest_timestamp'] is None

        # Add some frames
        with patch('time.time', return_value=100.5):
            buffer.add_frame(np.array([1]), timestamp=100.0)
        with patch('time.time', return_value=101.5):
            buffer.add_frame(np.array([2]), timestamp=101.0)

        stats = buffer.get_buffer_stats()
        assert stats['current_entries'] == 2
        assert stats['oldest_timestamp'] == 100.0
        assert stats['newest_timestamp'] == 101.0
        assert stats['total_added'] == 2
        assert stats['total_evicted'] == 0

    def test_thread_safety(self):
        """Test that buffer operations are thread-safe."""
        import threading

        buffer = CircularBuffer(max_entries=100)
        results = []
        errors = []

        def add_frames(thread_id: int):
            try:
                for i in range(10):
                    success = buffer.add_frame(np.array([thread_id, i]))
                    results.append((thread_id, i, success))
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_frames, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 frames each

        # All operations should have succeeded
        assert all(success for _, _, success in results)

        # Buffer should contain exactly the number added (since we added less than max)
        assert len(buffer.buffer) == 50

    def test_clear_buffer(self):
        """Test buffer clearing."""
        buffer = CircularBuffer(max_entries=10)

        # Add some frames
        with patch('time.time', return_value=100.5):
            buffer.add_frame(np.array([1]), timestamp=100.0)
        with patch('time.time', return_value=101.5):
            buffer.add_frame(np.array([2]), timestamp=101.0)
        assert len(buffer.buffer) == 2

        # Clear buffer
        buffer.clear()
        assert len(buffer.buffer) == 0

        # Stats should be reset
        stats = buffer.get_buffer_stats()
        assert stats['total_added'] == 0
        assert stats['total_evicted'] == 0
        assert stats['keyframes_added'] == 0

    def test_keyframe_hooks(self):
        """Test keyframe detection hooks."""
        buffer = CircularBuffer(max_entries=10)

        # Test floor keyframe detection
        assert buffer.check_floor_keyframe(1) is True  # First floor change
        assert buffer.check_floor_keyframe(1) is False  # Same floor
        assert buffer.check_floor_keyframe(2) is True  # Floor change

        # Test combat keyframe detection
        assert buffer.check_combat_keyframe(True) is True  # Enter combat
        assert buffer.check_combat_keyframe(True) is False  # Still in combat
        assert buffer.check_combat_keyframe(False) is True  # Exit combat

        # Test inventory keyframe detection
        inv1 = {"item1": 5, "item2": 3}
        inv2 = {"item1": 5, "item2": 3}
        inv3 = {"item1": 4, "item2": 3}

        assert buffer.check_inventory_keyframe(inv1) is True  # First inventory
        assert buffer.check_inventory_keyframe(inv2) is False  # Same inventory
        assert buffer.check_inventory_keyframe(inv3) is True  # Changed inventory

    def test_keyframe_addition(self):
        """Test adding keyframes to buffer."""
        buffer = CircularBuffer(max_entries=10)

        # Add regular frame
        buffer.add_frame(np.array([1, 2]), is_keyframe=False)
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0].priority == 1.0
        assert buffer.buffer[0].is_keyframe is False

        # Add keyframe
        buffer.add_frame(np.array([3, 4]), is_keyframe=True)
        assert len(buffer.buffer) == 2
        assert buffer.buffer[1].priority == 2.0
        assert buffer.buffer[1].is_keyframe is True

        # Check stats
        stats = buffer.get_buffer_stats()
        assert stats['keyframes_added'] == 1

    def test_keyframe_eviction_priority(self):
        """Test that keyframes are preserved longer during eviction."""
        buffer = CircularBuffer(window_seconds=2.0, keyframe_window_multiplier=3.0, max_entries=10)

        # Add a keyframe at t=0
        buffer.add_frame(np.array([1]), timestamp=0.0, is_keyframe=True)

        # Simulate time passing to t=3: keyframe should still be within extended window (3*2=6)
        with patch('time.time', return_value=3.0):
            buffer.add_frame(np.array([2]), timestamp=3.0, is_keyframe=False)  # Force eviction check

            # Keyframe should still be there (age=3.0 < 6.0)
            remaining_timestamps = [entry.timestamp for entry in buffer.buffer]
            assert 0.0 in remaining_timestamps

        # Simulate time passing to t=7: keyframe should now be evicted (age=7.0 > 6.0)
        with patch('time.time', return_value=7.0):
            buffer.add_frame(np.array([3]), timestamp=7.0, is_keyframe=False)  # Force eviction check

            # Keyframe should now be evicted
            remaining_timestamps = [entry.timestamp for entry in buffer.buffer]
            assert 0.0 not in remaining_timestamps

    def test_save_to_json_basic(self):
        """Test basic save to JSON functionality."""
        buffer = CircularBuffer(max_entries=10)

        # Add some frames with fixed current time to prevent eviction
        with patch('time.time', return_value=200.0):  # Future time relative to entries
            buffer.add_frame(np.array([1, 2, 3]), timestamp=100.0, metadata={"test": "data"})
            buffer.add_frame(np.array([[4, 5], [6, 7]]), timestamp=101.0, is_keyframe=True)

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            # Save buffer
            buffer.save_to_json(file_path)

            # Verify file was created and contains expected data
            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                data = json.load(f)

            assert data['window_seconds'] == 3600.0
            assert data['max_entries'] == 10
            assert len(data['entries']) == 2

            # Check first entry
            entry1 = data['entries'][0]
            assert entry1['id'] == 'frame_100.0'
            assert entry1['data'] == [1, 2, 3]
            assert entry1['timestamp'] == 100.0
            assert entry1['metadata'] == {"test": "data"}
            assert entry1['is_keyframe'] is False

            # Check second entry (keyframe)
            entry2 = data['entries'][1]
            assert entry2['id'] == 'frame_101.0'
            assert entry2['data'] == [[4, 5], [6, 7]]
            assert entry2['timestamp'] == 101.0
            assert entry2['is_keyframe'] is True

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_from_json_basic(self):
        """Test basic load from JSON functionality."""
        # Create a buffer and save it
        original_buffer = CircularBuffer(window_seconds=1800.0, max_entries=5)
        original_buffer.add_frame(np.array([1, 2]), timestamp=100.0)
        original_buffer.add_frame(np.array([3, 4]), timestamp=101.0, is_keyframe=True)

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            # Save and then load
            original_buffer.save_to_json(file_path)
            loaded_buffer = CircularBuffer.load_from_json(file_path)

            # Verify the loaded buffer matches the original
            assert loaded_buffer.window_seconds == original_buffer.window_seconds
            assert loaded_buffer.max_entries == original_buffer.max_entries
            assert loaded_buffer.keyframe_window_multiplier == original_buffer.keyframe_window_multiplier
            assert len(loaded_buffer.buffer) == len(original_buffer.buffer)

            # Check entries
            for orig_entry, loaded_entry in zip(original_buffer.buffer, loaded_buffer.buffer):
                assert orig_entry.id == loaded_entry.id
                np.testing.assert_array_equal(orig_entry.data, loaded_entry.data)
                assert orig_entry.timestamp == loaded_entry.timestamp
                assert orig_entry.metadata == loaded_entry.metadata
                assert orig_entry.priority == loaded_entry.priority
                assert orig_entry.is_keyframe == loaded_entry.is_keyframe

            # Check stats
            orig_stats = original_buffer.get_buffer_stats()
            loaded_stats = loaded_buffer.get_buffer_stats()
            assert loaded_stats['total_added'] == orig_stats['total_added']
            assert loaded_stats['keyframes_added'] == orig_stats['keyframes_added']

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_save_load_roundtrip_with_state(self):
        """Test save/load roundtrip preserves all internal state."""
        buffer = CircularBuffer(window_seconds=1200.0, max_entries=10, keyframe_window_multiplier=2.0)

        # Add frames and trigger some state changes
        buffer.add_frame(np.array([1]), timestamp=100.0)
        buffer.add_frame(np.array([2]), timestamp=101.0, is_keyframe=True)

        # Trigger keyframe checks to set internal state
        buffer.check_floor_keyframe(1)
        buffer.check_combat_keyframe(True)
        buffer.check_inventory_keyframe({"item1": 5})

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            # Save and reload
            buffer.save_to_json(file_path)
            loaded = CircularBuffer.load_from_json(file_path)

            # Verify all state is preserved
            assert loaded.window_seconds == buffer.window_seconds
            assert loaded.max_entries == buffer.max_entries
            assert loaded.keyframe_window_multiplier == buffer.keyframe_window_multiplier
            assert loaded._last_floor == buffer._last_floor
            assert loaded._last_combat_state == buffer._last_combat_state
            assert loaded._last_inventory == buffer._last_inventory

            # Verify entries
            assert len(loaded.buffer) == len(buffer.buffer)
            for orig, loaded_entry in zip(buffer.buffer, loaded.buffer):
                assert orig.id == loaded_entry.id
                np.testing.assert_array_equal(orig.data, loaded_entry.data)
                assert orig.timestamp == loaded_entry.timestamp
                assert orig.is_keyframe == loaded_entry.is_keyframe

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_save_to_json_invalid_path(self):
        """Test save_to_json handles invalid file paths."""
        buffer = CircularBuffer(max_entries=5)
        buffer.add_frame(np.array([1, 2, 3]))

        # Try to save to invalid path - on Windows this might create dirs, so test with non-existent drive
        with pytest.raises((IOError, OSError)):
            buffer.save_to_json('Z:\\invalid\\drive\\buffer.json')

    def test_load_from_json_missing_file(self):
        """Test load_from_json handles missing files."""
        with pytest.raises(IOError):
            CircularBuffer.load_from_json('/nonexistent/file.json')

    def test_load_from_json_invalid_json(self):
        """Test load_from_json handles invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            file_path = f.name

        try:
            with pytest.raises(ValueError):
                CircularBuffer.load_from_json(file_path)
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_from_json_missing_required_fields(self):
        """Test load_from_json fails with missing required fields."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            # Write JSON missing required fields
            json.dump({
                'window_seconds': 3600.0,
                'max_entries': 100,
                'enable_async': True,
                # Missing 'entries' and 'keyframe_window_multiplier'
            }, f)
            file_path = f.name

        try:
            with pytest.raises((ValueError, IOError)):
                CircularBuffer.load_from_json(file_path)
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)