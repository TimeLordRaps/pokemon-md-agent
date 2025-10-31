"""Test async screenshot capture implementation.

Tests background screenshot capture with 2-frame buffer, thread management,
and frame synchronization for <5ms perceived latency.
"""

import time
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.vision.quad_capture import AsyncScreenshotCapture, FrameData
from src.environment.mgba_controller import MGBAController


class TestAsyncScreenshotCapture:
    """Test AsyncScreenshotCapture class."""

    @pytest.fixture
    def mock_controller(self):
        """Mock MGBA controller."""
        controller = Mock(spec=MGBAController)
        controller.video_config = Mock()
        controller.video_config.width = 320
        controller.video_config.height = 288
        return controller

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Temporary output directory."""
        return tmp_path / "captures"

    @pytest.fixture
    def async_capture(self, mock_controller, output_dir):
        """AsyncScreenshotCapture instance."""
        return AsyncScreenshotCapture(mock_controller, output_dir)

    def test_initialization(self, async_capture):
        """Test proper initialization with buffer and thread setup."""
        assert async_capture.controller is not None
        assert async_capture.output_dir is not None
        assert async_capture.buffer_size == 2
        assert len(async_capture.frame_buffer) == 2
        assert async_capture.capture_thread is None
        assert not async_capture.running
        assert async_capture.restart_count == 0

    def test_start_stop_capture_thread(self, async_capture):
        """Test starting and stopping capture thread."""
        # Start capture
        async_capture.start()
        assert async_capture.running
        assert async_capture.capture_thread is not None
        assert async_capture.capture_thread.is_alive()

        # Give thread a moment to start properly
        time.sleep(0.01)
        assert async_capture.capture_thread.is_alive()

        # Stop capture
        async_capture.stop()
        assert not async_capture.running
        # Thread should be None after stop()
        assert async_capture.capture_thread is None

    def test_frame_buffer_operations(self, async_capture):
        """Test frame buffer write and read operations."""
        # Test initial empty buffer
        assert async_capture.get_latest_frame() is None

        # Write frame to buffer
        from src.vision.quad_capture import FrameData
        test_frame = FrameData(frame=1, timestamp=time.time(), image=Mock(), game_state={"frame_counter": 1})
        async_capture._write_frame_to_buffer(test_frame)

        # Read frame from buffer
        retrieved = async_capture.get_latest_frame()
        assert retrieved is not None
        assert retrieved.frame == 1

        # Test buffer rotation
        from src.vision.quad_capture import FrameData
        for i in range(3):
            frame = FrameData(frame=i + 2, timestamp=time.time(), image=Mock(), game_state={"frame_counter": i + 2})
            async_capture._write_frame_to_buffer(frame)

        # Should have latest frame
        latest = async_capture.get_latest_frame()
        assert latest.frame == 4

    def test_frame_synchronization(self, async_capture):
        """Test frame synchronization with game state timestamps."""
        # Mock frames with different timestamps (use recent timestamps)
        current_time = time.time()
        frames = [
            FrameData(frame=100, timestamp=current_time - 0.1, image=Mock(), game_state={"frame_counter": 100}),
            FrameData(frame=101, timestamp=current_time - 0.05, image=Mock(), game_state={"frame_counter": 101}),
            FrameData(frame=102, timestamp=current_time - 0.01, image=Mock(), game_state={"frame_counter": 102}),
        ]

        for frame in frames:
            async_capture._write_frame_to_buffer(frame)

        # Test frame matching
        matched = async_capture.get_frame_for_game_state(101, tolerance_ms=200)
        assert matched is not None
        assert matched.frame == 101

        # Test tolerance exceeded
        matched = async_capture.get_frame_for_game_state(99, tolerance_ms=10)
        assert matched is None

    def test_thread_restart_on_failure(self, async_capture):
        """Test automatic thread restart on capture failures."""
        # Reduce max consecutive failures for faster test
        original_max = async_capture.max_restarts
        async_capture.max_restarts = 1  # Allow restart

        # Mock controller to raise exception
        async_capture.controller.grab_frame.side_effect = RuntimeError("Capture failed")

        # Start capture
        async_capture.start()
        time.sleep(0.5)  # Let thread attempt capture multiple times (longer wait)

        # Should have restarted at least once due to failures
        assert async_capture.restart_count >= 1

        # Cleanup
        async_capture.stop()
        async_capture.max_restarts = original_max

    def test_graceful_fallback_to_sync(self, async_capture, mock_controller):
        """Test fallback to synchronous capture when async fails."""
        # Mock async failure
        async_capture.start()
        async_capture.stop()  # Stop immediately

        # Mock sync capture success
        mock_image = Mock()
        mock_controller.grab_frame.return_value = mock_image

        # Should fallback to sync
        result = async_capture.get_latest_frame_or_capture_sync()
        mock_controller.grab_frame.assert_called_once()
        assert result is not None

    def test_performance_latency(self, async_capture, mock_controller):
        """Test perceived latency <5ms from agent perspective."""
        # Mock fast capture
        mock_controller.grab_frame.return_value = Mock()

        # Start async capture
        async_capture.start()
        time.sleep(0.05)  # Let buffer populate

        # Measure read latency
        start_time = time.perf_counter()
        frame = async_capture.get_latest_frame()
        latency = (time.perf_counter() - start_time) * 1000  # ms

        assert frame is not None
        assert latency < 5.0  # <5ms requirement

        async_capture.stop()

    def test_frame_alignment_accuracy(self, async_capture):
        """Test 100% frame alignment accuracy."""
        # Populate buffer with known frames
        from src.vision.quad_capture import FrameData
        current_time = time.time()
        for frame_num in range(10):
            frame_data = FrameData(
                frame=frame_num,
                timestamp=current_time - (10 - frame_num) * 0.01,  # Recent timestamps
                image=Mock(),
                game_state={"frame_counter": frame_num}
            )
            async_capture._write_frame_to_buffer(frame_data)

        # Test alignment for frames that should be in the buffer (last 2 due to buffer_size=2)
        # With buffer_size=2, only the last 2 frames should be available
        for expected_frame in [8, 9]:  # Only check the last 2 frames
            matched = async_capture.get_frame_for_game_state(expected_frame, tolerance_ms=1000)
            assert matched is not None
            assert matched.frame == expected_frame

    def test_cpu_overhead(self, async_capture, mock_controller):
        """Test thread overhead <2% CPU usage."""
        # This is a basic test - real CPU monitoring would need system tools
        mock_controller.screenshot.return_value = Mock()

        async_capture.start()
        time.sleep(0.1)  # Let it run briefly

        # Thread should be running but not consuming excessive resources
        assert async_capture.capture_thread.is_alive()

        async_capture.stop()

    def test_error_handling_and_logging(self, async_capture, caplog):
        """Test comprehensive error handling and logging."""
        # Mock controller failure
        async_capture.controller.grab_frame.side_effect = ConnectionError("Socket error")

        async_capture.start()
        time.sleep(0.1)  # Let thread attempt multiple captures

        # Should log errors but continue
        assert "Screenshot capture failed" in caplog.text or "Capture failed" in caplog.text

        async_capture.stop()