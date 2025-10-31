"""Test current_frame property in MGBAController."""

import pytest
from unittest.mock import Mock, patch
from src.environment.mgba_controller import MGBAController


class TestCurrentFrame:
    """Test cases for current_frame property implementation."""

    def test_current_frame_initialized_none(self):
        """Test that current_frame is initialized to None."""
        controller = MGBAController(smoke_mode=True)
        assert controller._current_frame is None
        assert controller._frame_counter == 0

    def test_current_frame_returns_cached_value(self):
        """Test that current_frame returns cached value without new calls."""
        controller = MGBAController(smoke_mode=True)
        controller._current_frame = 42

        # Should return cached value without calling send_command
        with patch.object(controller, 'send_command') as mock_send:
            result = controller.current_frame()
            assert result == 42
            mock_send.assert_not_called()

    def test_current_frame_fetches_from_emulator(self):
        """Test that current_frame fetches from emulator when not cached."""
        controller = MGBAController(smoke_mode=True)

        with patch.object(controller, 'send_command', return_value="100") as mock_send:
            result = controller.current_frame()
            assert result == 100
            mock_send.assert_called_once_with("core.currentFrame")
            # Should cache the result
            assert controller._current_frame == 100

    def test_current_frame_handles_send_command_failure(self):
        """Test that current_frame handles send_command failure gracefully."""
        controller = MGBAController(smoke_mode=True)

        with patch.object(controller, 'send_command', return_value=None) as mock_send:
            result = controller.current_frame()
            assert result is None
            mock_send.assert_called_once_with("core.currentFrame")
            # Should remain None
            assert controller._current_frame is None

    def test_current_frame_handles_parse_error(self):
        """Test that current_frame handles invalid response parsing."""
        controller = MGBAController(smoke_mode=True)

        with patch.object(controller, 'send_command', return_value="invalid") as mock_send:
            result = controller.current_frame()
            assert result is None
            mock_send.assert_called_once_with("core.currentFrame")
            # Should remain None
            assert controller._current_frame is None

    def test_grab_frame_sets_current_frame(self):
        """Test that grab_frame sets current_frame and increments counter."""
        controller = MGBAController(smoke_mode=True)

        # Mock the screenshot and image processing
        with patch.object(controller, 'screenshot', return_value=True), \
             patch('PIL.Image.open') as mock_open, \
             patch('numpy.array') as mock_array, \
             patch.object(controller, 'current_frame', return_value=50):

            mock_image = Mock()
            mock_image.size = (480, 320)
            mock_open.return_value.__enter__.return_value = mock_image
            mock_array.return_value = Mock()  # Mock numpy array

            # Mock the video config
            controller.video_config.get_supported_sizes.return_value = [(480, 320)]
            controller.video_config.infer_profile_from_size.return_value = None

            # Mock Path and cache_dir
            with patch('pathlib.Path') as mock_path_class:
                mock_path_instance = Mock()
                mock_path_instance.unlink = Mock()
                mock_path_class.return_value = mock_path_instance
                controller.cache_dir = mock_path_instance

                result = controller.grab_frame()

                # Should have called current_frame to get the current frame number
                assert controller._current_frame == 50
                assert controller._frame_counter == 1
                assert result is not None