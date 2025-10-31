"""Test screenshot file locking issues on Windows."""

import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mgba_controller import MGBAController


class TestScreenshotLocking:
    """Test screenshot capture with Windows file locking issues."""

    @pytest.fixture
    def controller(self, tmp_path):
        """Create controller for testing."""
        return MGBAController(cache_dir=tmp_path)

    def test_screenshot_file_locking_during_high_frequency_capture(self, controller, tmp_path):
        """Test that reproduces WinError 32 (sharing violation) during rapid screenshot capture.

        This test simulates the scenario where mGBA is still writing to the screenshot file
        while the controller tries to read it, causing file locking issues on Windows.
        """
        screenshot_path = tmp_path / "test_screenshot.png"

        # Mock the send_command to simulate successful screenshot command
        with patch.object(controller, 'send_command', return_value="OK"):
            # Mock PIL Image.open to raise PermissionError on first attempt (file locked)
            # then succeed on retry
            original_open = None
            call_count = 0

            def mock_image_open(path):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call - file still locked by mGBA
                    raise PermissionError("The process cannot access the file because it is being used by another process")
                else:
                    # Subsequent calls - file available
                    # Create a dummy image file for the test
                    from PIL import Image
                    dummy_img = Image.new('RGB', (160, 240), color='red')
                    dummy_img.save(path)
                    return dummy_img

            with patch('PIL.Image.open', side_effect=mock_image_open):
                # This should succeed despite the initial file locking error
                img = controller.capture_screenshot(str(screenshot_path))

                # Verify the screenshot was captured successfully
                assert img is not None
                assert isinstance(img, np.ndarray)
                assert img.shape == (160, 240, 3)  # GBA resolution

                # Verify retry mechanism was triggered
                assert call_count > 1

    def test_screenshot_file_locking_persistent_failure(self, controller, tmp_path):
        """Test that persistent file locking leads to proper error handling.

        Simulates a scenario where the file remains locked for the entire retry period.
        """
        screenshot_path = tmp_path / "persistent_lock.png"

        with patch.object(controller, 'send_command', return_value="OK"):
            # Mock PIL Image.open to always raise PermissionError
            with patch('PIL.Image.open', side_effect=PermissionError("File locked")):
                # This should raise RuntimeError after exhausting retries
                with pytest.raises(RuntimeError, match="Failed to read screenshot after"):
                    controller.capture_screenshot(str(screenshot_path), max_retries=3)

    def test_screenshot_file_deleted_during_retry(self, controller, tmp_path):
        """Test behavior when screenshot file is deleted during retry attempts."""
        screenshot_path = tmp_path / "deleted_during_retry.png"

        with patch.object(controller, 'send_command', return_value="OK"):
            call_count = 0

            def mock_image_open(path):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # File exists but locked
                    raise PermissionError("File locked")
                elif call_count == 2:
                    # File deleted during retry
                    raise FileNotFoundError("No such file or directory")
                else:
                    # Should not reach here
                    raise RuntimeError("Unexpected call")

            with patch('PIL.Image.open', side_effect=mock_image_open):
                with pytest.raises(RuntimeError, match="Failed to read screenshot"):
                    controller.capture_screenshot(str(screenshot_path), max_retries=2)

    def test_screenshot_corrupted_during_write(self, controller, tmp_path):
        """Test handling of corrupted image files written by mGBA."""
        screenshot_path = tmp_path / "corrupted.png"

        with patch.object(controller, 'send_command', return_value="OK"):
            # Mock PIL Image.open to raise OSError for corrupted file
            with patch('PIL.Image.open', side_effect=OSError("Truncated PNG file")):
                with pytest.raises(RuntimeError, match="Failed to read screenshot"):
                    controller.capture_screenshot(str(screenshot_path))

    def test_concurrent_screenshot_requests(self, controller, tmp_path):
        """Test multiple concurrent screenshot capture requests.

        This reproduces the scenario where multiple threads try to capture screenshots
        simultaneously, potentially causing file conflicts.
        """
        results = []
        errors = []

        def capture_screenshot_thread(thread_id):
            """Worker function for concurrent screenshot capture."""
            try:
                time.sleep(thread_id * 0.01)  # Slight stagger
                img = controller.capture_screenshot(str(tmp_path / f"screenshot_{thread_id}.png"))
                results.append((thread_id, img))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Mock successful commands and delayed file access
        with patch.object(controller, 'send_command', return_value="OK"):
            call_count = 0

            def mock_image_open(path):
                nonlocal call_count
                call_count += 1
                # Simulate variable delay in file availability
                time.sleep(0.05)
                from PIL import Image
                dummy_img = Image.new('RGB', (160, 240), color='blue')
                dummy_img.save(path)
                return dummy_img

            with patch('PIL.Image.open', side_effect=mock_image_open):
                # Start multiple threads
                threads = []
                for i in range(5):
                    t = threading.Thread(target=capture_screenshot_thread, args=(i,))
                    threads.append(t)
                    t.start()

                # Wait for all threads
                for t in threads:
                    t.join()

                # Verify results - should all succeed or fail gracefully
                assert len(results) + len(errors) == 5

                # At least some should succeed
                assert len(results) > 0

                # Verify each successful result
                for thread_id, img in results:
                    assert img is not None
                    assert isinstance(img, np.ndarray)
                    assert img.shape == (160, 240, 3)