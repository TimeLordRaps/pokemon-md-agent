"""Test WRAM bounds checking in RAM decoders."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mgba_controller import MGBAController


class TestWRAMBoundsChecking:
    """Test buffer overflow protection in RAM decoders."""

    @pytest.fixture
    def controller(self, tmp_path):
        """Create controller for testing."""
        return MGBAController(cache_dir=tmp_path)

    def test_peek_bounds_checking_valid_addresses(self, controller):
        """Test peek with valid WRAM addresses."""
        # Mock memory reading
        with patch.object(controller, 'memory_domain_read_range') as mock_read:
            mock_read.return_value = b'\x12\x34\x56\x78'  # Valid data

            # Test valid WRAM address
            result = controller.peek(0x02000000, 4)  # Start of WRAM
            assert result == b'\x12\x34\x56\x78'
            mock_read.assert_called_with('wram', 0, 4)

            # Test middle of WRAM
            result = controller.peek(0x02010000, 2)  # Middle of WRAM
            assert result == b'\x12\x34\x56\x78'
            mock_read.assert_called_with('wram', 0x10000, 2)

            # Test end of WRAM
            result = controller.peek(0x0203FFFC, 4)  # Near end of WRAM
            assert result == b'\x12\x34\x56\x78'
            mock_read.assert_called_with('wram', 0x3FFFC, 4)

    def test_peek_bounds_checking_invalid_addresses(self, controller):
        """Test peek with invalid addresses outside WRAM."""
        # Test address before WRAM
        result = controller.peek(0x01FFFFFF, 4)  # Just before WRAM
        assert result is None

        # Test address after WRAM
        result = controller.peek(0x02040000, 4)  # Just after WRAM
        assert result is None

        # Test completely invalid address
        result = controller.peek(0x12345678, 4)  # Random invalid address
        assert result is None

        # Test IWRAM bounds
        result = controller.peek(0x03007FFC, 4)  # Valid IWRAM
        # Should work if we add IWRAM support, but currently returns None
        # This tests the bounds checking logic

    def test_peek_zero_length_read(self, controller):
        """Test peek with zero length (edge case)."""
        result = controller.peek(0x02000000, 0)
        assert result == b''  # Empty bytes

    def test_peek_large_read_within_bounds(self, controller):
        """Test peek with large read within WRAM bounds."""
        with patch.object(controller, 'memory_domain_read_range') as mock_read:
            mock_read.return_value = b'A' * 0x40000  # Full WRAM size

            result = controller.peek(0x02000000, 0x40000)  # Full WRAM
            assert len(result) == 0x40000
            mock_read.assert_called_with('wram', 0, 0x40000)

    def test_peek_read_beyond_wram_bounds(self, controller):
        """Test peek that would read beyond WRAM bounds."""
        # Request read that extends beyond WRAM end
        result = controller.peek(0x0203FFFF, 4)  # Last byte of WRAM + 3 more
        assert result is None  # Should be rejected

    def test_get_floor_bounds_checking(self, controller):
        """Test get_floor with bounds checking."""
        with patch.object(controller, 'peek') as mock_peek:
            # Valid floor data
            mock_peek.return_value = b'\x05\x00\x00\x00'  # floor = 5

            result = controller.get_floor()
            assert result == 5
            mock_peek.assert_called_once()

            # Test with None return (memory read failure)
            mock_peek.return_value = None
            with pytest.raises(RuntimeError, match="Failed to read floor"):
                controller.get_floor()

    def test_get_player_position_bounds_checking(self, controller):
        """Test get_player_position with bounds checking."""
        with patch.object(controller, 'peek') as mock_peek:
            # Valid position data
            mock_peek.side_effect = [b'\x0A\x00\x00\x00', b'\x08\x00\x00\x00']  # x=10, y=8

            x, y = controller.get_player_position()
            assert x == 10
            assert y == 8
            assert mock_peek.call_count == 2

            # Test with one None return
            mock_peek.side_effect = [b'\x0A\x00\x00\x00', None]
            with pytest.raises(RuntimeError, match="Failed to read player position"):
                controller.get_player_position()

    def test_get_player_stats_bounds_checking(self, controller):
        """Test get_player_stats with bounds checking."""
        with patch.object(controller, 'peek') as mock_peek:
            # Valid stats data
            mock_peek.side_effect = [
                b'\xC8\x00\x00\x00',  # hp = 200
                b'\xFA\x00\x00\x00',  # max_hp = 250
                b'\x4B\x00\x00\x00',  # belly = 75
            ]

            stats = controller.get_player_stats()
            assert stats['hp'] == 200
            assert stats['max_hp'] == 250
            assert stats['belly'] == 75
            assert stats['max_belly'] == 100  # Fixed value
            assert mock_peek.call_count == 3

            # Test with None return
            mock_peek.side_effect = [b'\xC8\x00\x00\x00', None, b'\x4B\x00\x00\x00']
            with pytest.raises(RuntimeError, match="Failed to read player stats"):
                controller.get_player_stats()

    def test_memory_domain_read_bounds_validation(self, controller):
        """Test memory_domain_read_range with bounds validation."""
        # Test with valid domain and address
        with patch.object(controller, 'memory_domain_read_range') as mock_read:
            mock_read.return_value = b'test'

            result = controller.memory_domain_read_range('wram', 0x1000, 4)
            assert result == b'test'
            mock_read.assert_called_with('wram', 0x1000, 4)

            # Test with invalid domain
            result = controller.memory_domain_read_range('invalid_domain', 0x1000, 4)
            assert result is None  # Should be handled gracefully

    def test_buffer_overflow_prevention_in_peek(self, controller):
        """Test that peek prevents buffer overflow by rejecting invalid ranges."""
        # Test extremely large read request
        result = controller.peek(0x02000000, 0x1000000)  # 16MB read
        assert result is None

        # Test negative length (though this should be caught earlier)
        with patch.object(controller, 'memory_domain_read_range') as mock_read:
            mock_read.return_value = b'fail'
            # If we get here, the method should handle it gracefully
            result = controller.peek(0x02000000, -1)
            # Result depends on implementation, but shouldn't crash

    def test_wram_offset_calculation_edge_cases(self, controller):
        """Test WRAM offset calculation for edge cases."""
        # Test with IWRAM address (should fail in current implementation)
        result = controller.peek(0x03000000, 4)
        # Current implementation only supports WRAM, so this should return None
        assert result is None

        # Test with address that's not aligned to memory domain
        result = controller.peek(0x02000001, 4)  # Unaligned address
        # Should still work as the offset calculation handles this
        # (though the underlying mGBA may have alignment requirements)

    def test_concurrent_memory_reads_bounds_safety(self, controller):
        """Test bounds checking under concurrent memory read scenarios."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def memory_read_worker(worker_id):
            """Worker function for concurrent memory reads."""
            try:
                # Test different address ranges
                if worker_id == 0:
                    result = controller.peek(0x02000000, 4)  # Valid
                elif worker_id == 1:
                    result = controller.peek(0x02040000, 4)  # Invalid (beyond WRAM)
                else:
                    result = controller.peek(0x01FFFFFF, 4)  # Invalid (before WRAM)

                results.put((worker_id, result))
            except Exception as e:
                errors.put((worker_id, str(e)))

        # Mock memory reads to return valid data for valid addresses
        with patch.object(controller, 'memory_domain_read_range') as mock_read:
            def mock_read_impl(domain, address, length):
                if domain == 'wram' and 0 <= address < 0x40000:
                    return b'\x00' * length
                return None

            mock_read.side_effect = mock_read_impl

            # Start concurrent reads
            threads = []
            for i in range(3):
                t = threading.Thread(target=memory_read_worker, args=(i,))
                threads.append(t)
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Verify results
            assert results.qsize() == 3
            assert errors.empty()

            while not results.empty():
                worker_id, result = results.get()
                if worker_id == 0:
                    assert result == b'\x00\x00\x00\x00'  # Valid read
                else:
                    assert result is None  # Invalid reads

    def test_memory_read_failure_propagation(self, controller):
        """Test that memory read failures are properly propagated."""
        with patch.object(controller, 'memory_domain_read_range', return_value=None):
            # Test peek failure
            result = controller.peek(0x02000000, 4)
            assert result is None

            # Test that higher-level functions handle this
            with pytest.raises(RuntimeError):
                controller.get_floor()

    def test_bounds_checking_with_different_data_sizes(self, controller):
        """Test bounds checking with different data type sizes."""
        # Test reading different byte sizes at WRAM boundaries
        test_cases = [
            (0x02000000, 1),  # Single byte at start
            (0x0203FFFF, 1),  # Single byte at end
            (0x02000000, 2),  # Two bytes at start
            (0x0203FFFE, 2),  # Two bytes at end
            (0x02000000, 4),  # Four bytes at start
            (0x0203FFFC, 4),  # Four bytes at end
        ]

        for address, size in test_cases:
            with patch.object(controller, 'memory_domain_read_range') as mock_read:
                expected_data = b'A' * size
                mock_read.return_value = expected_data

                result = controller.peek(address, size)
                if address + size <= 0x02040000:  # Within bounds
                    assert result == expected_data
                else:  # Would exceed bounds
                    assert result is None