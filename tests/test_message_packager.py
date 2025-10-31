"""
Unit tests for message_packager.py with Copilot input support and multi-image packs.

Tests three-message protocol: MSG[-2] episodic_map, MSG[-1] retrieval, MSG[0] now.
Tests Copilot {png,meta.json} parsing and multi-image pack support.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch
from PIL import Image
from src.orchestrator.message_packager import (
    Message,
    CopilotInput,
    pack,
    pack_from_copilot,
    parse_copilot_input,
    MODEL_PRESETS,
    _validate_image_dimensions
)


class TestMessagePackager:
    """Test message packager functionality."""

    def test_copilot_input_dataclass(self):
        """Test CopilotInput dataclass structure."""
        copilot_input = CopilotInput(
            png_path="/path/to/env.png",
            meta_json_path="/path/to/meta.json",
            retrieved_thumbnails=["thumb1.png", "thumb2.png"]
        )

        assert copilot_input.png_path == "/path/to/env.png"
        assert copilot_input.meta_json_path == "/path/to/meta.json"
        assert copilot_input.retrieved_thumbnails == ["thumb1.png", "thumb2.png"]

    def test_parse_copilot_input_basic(self):
        """Test parsing basic Copilot input files."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            # Create dummy PNG file
            with open(png_path, 'wb') as f:
                f.write(b'dummy png content')

            # Create meta.json
            meta_data = {
                "dynamic_map": "/path/to/map.png",
                "event_log": ["Event 1", "Event 2"],
                "grid_overlay": "/path/to/grid.png",
                "retrieved_trajectories": [
                    {"summary": "Trajectory 1", "frames": ["frame1.png", "frame2.png"]}
                ]
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=["thumb1.png"]
            )

            step_state = parse_copilot_input(copilot_input)

            assert step_state['dynamic_map'] == "/path/to/map.png"
            assert step_state['event_log'] == ["Event 1", "Event 2"]
            assert step_state['now']['env'] == png_path
            assert step_state['now']['grid'] == "/path/to/grid.png"
            assert step_state['retrieved_thumbnails'] == ["thumb1.png"]
            assert len(step_state['retrieved_trajs']) == 1

    def test_parse_copilot_input_missing_files(self):
        """Test error handling for missing input files."""
        copilot_input = CopilotInput(
            png_path="/nonexistent/env.png",
            meta_json_path="/nonexistent/meta.json",
            retrieved_thumbnails=[]
        )

        with pytest.raises(FileNotFoundError, match="Copilot PNG not found"):
            parse_copilot_input(copilot_input)

        # Test missing meta.json
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            with open(png_path, 'wb') as f:
                f.write(b'dummy')

            copilot_input.png_path = png_path
            with pytest.raises(FileNotFoundError, match="Copilot meta.json not found"):
                parse_copilot_input(copilot_input)

    def test_parse_copilot_input_malformed_json(self):
        """Test error handling for malformed meta.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')

            # Write invalid JSON
            with open(meta_path, 'w') as f:
                f.write("invalid json content")

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=[]
            )

            with pytest.raises(ValueError):
                parse_copilot_input(copilot_input)

    def test_pack_from_copilot_basic(self):
        """Test packing messages from Copilot input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')

            meta_data = {
                "dynamic_map": "/path/to/map.png",
                "event_log": ["Event 1"],
                "grid_overlay": "/path/to/grid.png",
                "retrieved_trajectories": []
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=["thumb1.png"]
            )

            messages = pack_from_copilot(copilot_input, "explore", "4B")

            assert len(messages) == 3
            assert messages[0].role == "user"  # episodic_map
            assert messages[1].role == "assistant"  # retrieval
            assert messages[2].role == "user"  # now

            # Check now message includes thumbnails
            now_msg = messages[2]
            assert "thumb1.png" in now_msg.images
            assert png_path in now_msg.images  # env image
            assert "/path/to/grid.png" in now_msg.images  # grid overlay

    def test_pack_from_copilot_with_retrieved_trajectories(self):
        """Test packing with retrieved trajectories in meta.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            # Create a valid 480x320 dummy PNG file
            from PIL import Image
            img = Image.new('RGB', (480, 320), color='red')
            img.save(png_path)

            meta_data = {
                "retrieved_trajectories": [
                    {
                        "summary": "Fight sequence",
                        "frames": ["fight1.png", "fight2.png", "fight3.png"]
                    }
                ]
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=[]
            )

            messages = pack_from_copilot(copilot_input, "fight", "4B")

            # Check retrieval message
            retrieval_msg = messages[1]
            assert "Fight sequence" in retrieval_msg.text
            assert "fight1.png" in retrieval_msg.images

    def test_multi_image_pack_support(self):
        """Test multi-image packs with env+grid+thumbnails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')

            meta_data = {
                "grid_overlay": "/path/to/grid.png"
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=["thumb1.png", "thumb2.png", "thumb3.png"]
            )

            messages = pack_from_copilot(copilot_input, "explore", "4B")

            now_msg = messages[2]  # MSG[0]
            expected_images = [png_path, "/path/to/grid.png", "thumb1.png", "thumb2.png", "thumb3.png"]
            assert set(now_msg.images) == set(expected_images)
            assert "Retrieved thumbnails (3)" in now_msg.text

    def test_thumbnail_limit(self):
        """Test that thumbnails are limited to prevent budget overflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')

            with open(meta_path, 'w') as f:
                json.dump({}, f)

            # Create many thumbnails
            thumbnails = [f"thumb{i}.png" for i in range(10)]
            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=thumbnails
            )

            messages = pack_from_copilot(copilot_input, "explore", "4B")

            now_msg = messages[2]
            # Should limit to 5 thumbnails max
            thumbnail_images = [img for img in now_msg.images if img and img.startswith("thumb")]
            assert len(thumbnail_images) <= 5

    def test_budget_constraints_with_thumbnails(self):
        """Test budget constraints apply correctly with multi-image packs."""
        step_state = {
            'dynamic_map': 'map.png',
            'event_log': ['event1', 'event2'],
            'retrieved_trajs': [{'summary': 'traj1', 'frames': ['frame1.png']}],
            'now': {'env': 'env.png', 'grid': 'grid.png'},
            'retrieved_thumbnails': ['thumb1.png', 'thumb2.png', 'thumb3.png', 'thumb4.png', 'thumb5.png']
        }

        # Test 2B model with strict budget
        messages = pack(step_state, "explore", "2B")

        total_images = sum(len(msg.images) for msg in messages)
        assert total_images <= MODEL_PRESETS['2B']['max_images']

    def test_model_size_validation(self):
        """Test model size validation in pack_from_copilot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')
            with open(meta_path, 'w') as f:
                json.dump({}, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=[]
            )

            # Valid model sizes
            for model_size in ['2B', '4B', '8B']:
                messages = pack_from_copilot(copilot_input, "explore", model_size)
                assert len(messages) == 3

            # Invalid model size
            with pytest.raises(ValueError, match="Invalid model_size"):
                pack_from_copilot(copilot_input, "explore", "invalid")

    def test_policy_hint_integration(self):
        """Test that policy hints are correctly integrated into messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')
            with open(meta_path, 'w') as f:
                json.dump({}, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=[]
            )

            messages = pack_from_copilot(copilot_input, "fight", "4B")

            now_msg = messages[2]
            assert "Policy hint: fight" in now_msg.text

    @patch('src.orchestrator.message_packager.logger')
    def test_logging_integration(self, mock_logger):
        """Test that appropriate logging occurs during packing."""
        step_state = {
            'now': {'env': 'env.png'},
            'retrieved_thumbnails': ['thumb1.png']
        }

        pack(step_state, "explore", "4B")

        # Check that info logging occurred
        mock_logger.info.assert_called()

    def test_backward_compatibility(self):
        """Test that existing pack() function still works."""
        step_state = {
            'dynamic_map': 'map.png',
            'event_log': ['event1'],
            'retrieved_trajs': [],
            'now': {'env': 'env.png'}
        }

        messages = pack(step_state, "explore", "4B")

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"

    def test_empty_retrieved_thumbnails(self):
        """Test handling of empty retrieved thumbnails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")
            meta_path = os.path.join(temp_dir, "meta.json")

            with open(png_path, 'wb') as f:
                f.write(b'dummy png')
            with open(meta_path, 'w') as f:
                json.dump({"grid_overlay": "grid.png"}, f)

            copilot_input = CopilotInput(
                png_path=png_path,
                meta_json_path=meta_path,
                retrieved_thumbnails=[]
            )

            messages = pack_from_copilot(copilot_input, "explore", "4B")

            now_msg = messages[2]
            # Should still have env and grid images
            assert png_path in now_msg.images
            assert "grid.png" in now_msg.images
            # Should not mention thumbnails
            assert "Retrieved thumbnails" not in now_msg.text

    def test_image_validation_480x320_only(self):
        """Test that only 480×320 images are accepted, no rescaling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "env.png")

            # Create a 240×160 image
            base_img = Image.new('RGB', (240, 160), color='red')
            base_img.save(png_path)

            # Call validation which should reject non-480×320 images
            with pytest.raises(ValueError, match="required exactly \\(480, 320\\)"):
                _validate_image_dimensions([png_path])

            # Test with correct size
            correct_img = Image.new('RGB', (480, 320), color='blue')
            correct_img.save(png_path)

            # Should not raise
            _validate_image_dimensions([png_path])