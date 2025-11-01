"""Unit tests for MGBAController frame capture functionality."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mgba_controller import MGBAController, VideoConfig


class TestMGBAControllerFrameCapture:
    """Test MGBAController frame capture and current_frame_data property."""

    @pytest.fixture
    def mock_controller(self, tmp_path):
        """Create a mock MGBAController for testing."""
        with patch('src.environment.mgba_controller.AddressManager'):
            with patch('src.environment.mgba_controller.FPSAdjuster'):
                controller = MGBAController(
                    host="localhost",
                    port=8888,
                    timeout=1.0,
                    cache_dir=tmp_path,
                    smoke_mode=True,
                    auto_reconnect=False
                )
                # Mock the transport connection
                controller._transport = Mock()
                controller._transport.is_connected = Mock(return_value=True)
        return controller

    def test_current_frame_data_initialized_as_none(self, mock_controller):
        """Test that current_frame_data is initialized as None."""
        assert mock_controller.current_frame_data is None

    def test_current_frame_data_set_after_frame_capture(self, mock_controller, tmp_path):
        """Test that current_frame_data is set when a frame is captured."""
        # Create a dummy frame
        test_image = Image.new('RGB', (480, 320), color='red')
        frame_file = tmp_path / "test_frame.png"
        test_image.save(frame_file)

        # Mock the screenshot method
        mock_controller.screenshot = Mock(return_value=True)
        mock_controller.get_floor = Mock(return_value=1)
        mock_controller.get_player_position = Mock(return_value=(10, 15))

        # Mock the Image.open call
        with patch('src.environment.mgba_controller.Image.open', return_value=test_image):
            with patch('src.environment.mgba_controller.Path.unlink'):
                result = mock_controller.grab_frame()

        # Verify frame was captured and stored
        assert result is not None
        assert mock_controller.current_frame_data is not None
        assert isinstance(mock_controller.current_frame_data, np.ndarray)
        assert mock_controller.current_frame_data.shape == (320, 480, 3)

    def test_current_frame_method_returns_frame_number(self, mock_controller):
        """Test that current_frame() method returns frame number from emulator."""
        # Mock the send_command to return a frame number
        mock_controller.send_command = Mock(return_value="12345")

        frame_num = mock_controller.current_frame()

        assert frame_num == 12345
        mock_controller.send_command.assert_called_once_with("core.currentFrame")

    def test_current_frame_method_handles_error_response(self, mock_controller):
        """Test that current_frame() handles error responses gracefully."""
        mock_controller.send_command = Mock(return_value="<|ERROR|>")

        frame_num = mock_controller.current_frame()

        assert frame_num is None

    def test_current_frame_method_handles_invalid_response(self, mock_controller):
        """Test that current_frame() handles non-numeric responses."""
        mock_controller.send_command = Mock(return_value="invalid")

        frame_num = mock_controller.current_frame()

        assert frame_num is None

    def test_current_frame_method_handles_none_response(self, mock_controller):
        """Test that current_frame() handles None responses."""
        mock_controller.send_command = Mock(return_value=None)

        frame_num = mock_controller.current_frame()

        assert frame_num is None


@pytest.mark.integration
@pytest.mark.live_emulator
class TestMGBAControllerFrameCaptureIntegration:
    """Integration tests for frame capture (requires running emulator)."""

    @pytest.mark.timeout(10)
    def test_grab_frame_stores_current_frame_data(self, connected_mgba_controller):
        """Test that grab_frame() stores frame data in current_frame_data."""
        controller = connected_mgba_controller

        # Ensure current_frame_data is initially None
        assert controller.current_frame_data is None

        # Grab a frame
        image = controller.grab_frame()

        # Skip test if grab_frame fails (emulator not ready, ROM not loaded, etc.)
        if image is None:
            pytest.skip("grab_frame() returned None - emulator may not be ready or ROM not loaded")

        # Verify frame was captured and stored
        assert controller.current_frame_data is not None
        assert isinstance(controller.current_frame_data, np.ndarray)

        # Verify dimensions match
        assert controller.current_frame_data.shape[0] > 0  # height
        assert controller.current_frame_data.shape[1] > 0  # width
        assert controller.current_frame_data.shape[2] == 3  # RGB channels

    @pytest.mark.timeout(5)
    @pytest.mark.network
    def test_current_frame_returns_valid_number(self, connected_mgba_controller):
        """Test that current_frame() returns a valid frame number from emulator."""
        controller = connected_mgba_controller

        frame1 = controller.current_frame()
        assert frame1 is not None
        assert isinstance(frame1, int)
        assert frame1 > 0
