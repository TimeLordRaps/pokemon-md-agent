"""Integration tests for message_packager with vision prompts (Phase 2)."""

import pytest
import tempfile
import json
from pathlib import Path
from src.orchestrator.message_packager import (
    CopilotInput,
    Message,
    pack,
    pack_from_copilot,
    get_vision_system_prompt_for_model,
    pack_with_vision_prompts,
    pack_from_copilot_with_vision,
)


class TestVisionPromptIntegration:
    """Test integration of vision prompts with message packaging."""

    def test_get_vision_system_prompt_for_2b(self):
        """Get vision prompt optimized for 2B model."""
        prompt = get_vision_system_prompt_for_model("2B")
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "JSON" in prompt
        assert "player_pos" in prompt

    def test_get_vision_system_prompt_for_4b(self):
        """Get vision prompt optimized for 4B model."""
        prompt = get_vision_system_prompt_for_model("4B")
        assert isinstance(prompt, str)
        # 4B should use instruct variant
        assert "JSON" in prompt

    def test_get_vision_system_prompt_for_8b(self):
        """Get vision prompt for 8B model."""
        prompt = get_vision_system_prompt_for_model("8B")
        assert isinstance(prompt, str)
        # 8B might use thinking variant
        assert "JSON" in prompt or "STEP" in prompt

    def test_invalid_model_size_defaults_to_thinking(self):
        """Invalid model size defaults to thinking variant."""
        prompt = get_vision_system_prompt_for_model("invalid")
        # Should return thinking variant for unknown sizes
        assert isinstance(prompt, str)
        assert "JSON" in prompt


class TestPackWithVisionPrompts:
    """Test pack_with_vision_prompts function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.step_state = {
            'dynamic_map': None,
            'event_log': ['Started exploration'],
            'retrieved_trajs': [],
            'now': {
                'env': None,
                'grid': None,
            },
            'retrieved_thumbnails': [],
        }

    def test_pack_with_vision_instruct(self):
        """Pack messages with vision prompt (instruct variant)."""
        system_prompt, messages = pack_with_vision_prompts(
            self.step_state,
            policy_hint="explore",
            model_size="4B"
        )

        # Verify system prompt
        assert isinstance(system_prompt, str)
        assert "JSON" in system_prompt
        assert len(system_prompt) > 100

        # Verify messages still work
        assert isinstance(messages, list)
        assert len(messages) == 3  # Three-message protocol

        # Verify each message
        for msg in messages:
            assert isinstance(msg, Message)
            assert msg.role in ["user", "assistant"]
            assert isinstance(msg.text, str)
            assert isinstance(msg.images, list)

    def test_pack_with_vision_2b(self):
        """Pack with 2B model size."""
        system_prompt, messages = pack_with_vision_prompts(
            self.step_state,
            policy_hint="fight",
            model_size="2B"
        )

        assert system_prompt is not None
        assert messages is not None
        assert len(messages) == 3

    def test_pack_with_vision_8b(self):
        """Pack with 8B model size."""
        system_prompt, messages = pack_with_vision_prompts(
            self.step_state,
            policy_hint="boss_battle",
            model_size="8B"
        )

        assert system_prompt is not None
        assert messages is not None

    def test_pack_returns_tuple(self):
        """Function returns tuple of (system_prompt, messages)."""
        result = pack_with_vision_prompts(self.step_state, "explore")

        assert isinstance(result, tuple)
        assert len(result) == 2
        system_prompt, messages = result
        assert isinstance(system_prompt, str)
        assert isinstance(messages, list)

    def test_system_prompt_covers_game_state_fields(self):
        """System prompt documents GameState fields."""
        system_prompt, _ = pack_with_vision_prompts(
            self.step_state,
            "explore",
            model_size="4B"
        )

        # Key fields should be documented
        required_fields = ["player_pos", "floor", "state", "enemies", "items", "confidence"]
        for field in required_fields:
            assert field in system_prompt, f"Missing field in system prompt: {field}"


class TestPackFromCopilotWithVision:
    """Test integration with Copilot input format."""

    def create_temp_copilot_input(self):
        """Create temporary Copilot input files."""
        tmpdir = tempfile.mkdtemp()

        # Create dummy PNG (just a text file for testing)
        png_path = Path(tmpdir) / "test.png"
        png_path.write_text("dummy png")

        # Create meta.json
        meta_path = Path(tmpdir) / "meta.json"
        meta_data = {
            "dynamic_map": None,
            "event_log": ["Exploring"],
            "grid_overlay": None,
            "retrieved_trajectories": [],
        }
        meta_path.write_text(json.dumps(meta_data))

        return CopilotInput(
            png_path=str(png_path),
            meta_json_path=str(meta_path),
            retrieved_thumbnails=[]
        )

    def test_pack_from_copilot_with_vision(self):
        """Pack from Copilot input with vision prompts."""
        copilot_input = self.create_temp_copilot_input()

        system_prompt, messages = pack_from_copilot_with_vision(
            copilot_input,
            policy_hint="explore",
            model_size="4B"
        )

        # Verify system prompt
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 100

        # Verify messages
        assert isinstance(messages, list)
        assert len(messages) == 3

    def test_pack_from_copilot_with_vision_returns_tuple(self):
        """Function returns (system_prompt, messages) tuple."""
        copilot_input = self.create_temp_copilot_input()

        result = pack_from_copilot_with_vision(
            copilot_input,
            "explore"
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nonexistent_copilot_files_raise(self):
        """Raise FileNotFoundError for nonexistent files."""
        bad_input = CopilotInput(
            png_path="/nonexistent/path.png",
            meta_json_path="/nonexistent/meta.json",
            retrieved_thumbnails=[]
        )

        with pytest.raises(FileNotFoundError):
            pack_from_copilot_with_vision(bad_input, "explore")


class TestVisionPromptConsistency:
    """Test consistency between vision prompts and pack functions."""

    def test_all_model_sizes_supported(self):
        """All standard model sizes return prompts."""
        for model_size in ['2B', '4B', '8B']:
            prompt = get_vision_system_prompt_for_model(model_size)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_different_model_sizes_produce_consistent_messages(self):
        """Different model sizes produce same message structure."""
        step_state = {
            'dynamic_map': None,
            'event_log': [],
            'retrieved_trajs': [],
            'now': {'env': None, 'grid': None},
            'retrieved_thumbnails': [],
        }

        results = {}
        for model_size in ['2B', '4B', '8B']:
            _, messages = pack_with_vision_prompts(
                step_state,
                "explore",
                model_size=model_size
            )
            results[model_size] = messages

        # All should have 3 messages
        for model_size, messages in results.items():
            assert len(messages) == 3, f"Model {model_size} didn't return 3 messages"

        # Message roles should be consistent
        for model_size, messages in results.items():
            roles = [msg.role for msg in messages]
            assert roles == ["user", "assistant", "user"], f"Model {model_size} has inconsistent roles"


class TestVisionPromptBackwardCompatibility:
    """Test backward compatibility with existing pack() function."""

    def test_pack_still_works(self):
        """Original pack() function still works."""
        step_state = {
            'dynamic_map': None,
            'event_log': [],
            'retrieved_trajs': [],
            'now': {'env': None, 'grid': None},
            'retrieved_thumbnails': [],
        }

        messages = pack(step_state, "explore", "4B")

        assert isinstance(messages, list)
        assert len(messages) == 3
        for msg in messages:
            assert isinstance(msg, Message)

    def test_pack_from_copilot_still_works(self):
        """Original pack_from_copilot() function still works."""
        tmpdir = tempfile.mkdtemp()
        png_path = Path(tmpdir) / "test.png"
        png_path.write_text("dummy")
        meta_path = Path(tmpdir) / "meta.json"
        meta_path.write_text(json.dumps({
            "event_log": [],
            "retrieved_trajectories": []
        }))

        copilot_input = CopilotInput(
            png_path=str(png_path),
            meta_json_path=str(meta_path),
            retrieved_thumbnails=[]
        )

        messages = pack_from_copilot(copilot_input, "explore", "4B")

        assert isinstance(messages, list)
        assert len(messages) == 3


class TestVisionPromptBenchmark:
    """Performance benchmarks for vision prompt integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.step_state = {
            'dynamic_map': None,
            'event_log': ['test'],
            'retrieved_trajs': [],
            'now': {'env': None, 'grid': None},
            'retrieved_thumbnails': [],
        }

    def test_pack_with_vision_prompts_fast(self):
        """pack_with_vision_prompts completes quickly."""
        import time

        start = time.time()
        for _ in range(10):
            pack_with_vision_prompts(self.step_state, "explore", "4B")
        elapsed = time.time() - start

        # 10 packs should be fast
        assert elapsed < 1.0

    def test_get_vision_system_prompt_for_model_fast(self):
        """get_vision_system_prompt_for_model is fast."""
        import time

        start = time.time()
        for _ in range(100):
            get_vision_system_prompt_for_model("4B")
        elapsed = time.time() - start

        assert elapsed < 0.1  # 100 calls < 100ms


class TestVisionPromptEdgeCases:
    """Test edge cases with vision prompts."""

    def test_pack_with_empty_step_state(self):
        """Pack handles minimal step_state."""
        minimal_state = {
            'now': {'env': None, 'grid': None},
        }

        system_prompt, messages = pack_with_vision_prompts(
            minimal_state,
            "explore"
        )

        assert system_prompt is not None
        assert messages is not None

    def test_pack_with_various_policy_hints(self):
        """Pack handles different policy hints."""
        step_state = {
            'now': {'env': None, 'grid': None},
        }

        hints = ["explore", "fight", "retreat", "boss_battle", "shop"]
        for hint in hints:
            system_prompt, messages = pack_with_vision_prompts(
                step_state,
                policy_hint=hint
            )
            assert system_prompt is not None
            assert messages is not None
