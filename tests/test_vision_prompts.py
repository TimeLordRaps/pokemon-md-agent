"""Tests for vision prompt module (Phase 2)."""

import pytest
import json
from src.models.vision_prompts import (
    VISION_SYSTEM_PROMPT_INSTRUCT,
    VISION_SYSTEM_PROMPT_THINKING,
    PromptBuilder,
    get_vision_system_prompt,
    format_vision_prompt_with_examples,
    get_schema_guidance,
)
from src.models.game_state_schema import GameState, GameStateEnum


class TestVisionSystemPrompts:
    """Test vision system prompt definitions."""

    def test_instruct_prompt_exists(self):
        """Instruct variant is non-empty."""
        assert VISION_SYSTEM_PROMPT_INSTRUCT
        assert isinstance(VISION_SYSTEM_PROMPT_INSTRUCT, str)
        assert len(VISION_SYSTEM_PROMPT_INSTRUCT) > 100

    def test_thinking_prompt_exists(self):
        """Thinking variant is non-empty."""
        assert VISION_SYSTEM_PROMPT_THINKING
        assert isinstance(VISION_SYSTEM_PROMPT_THINKING, str)
        assert len(VISION_SYSTEM_PROMPT_THINKING) > 100

    def test_instruct_prompt_contains_requirements(self):
        """Instruct prompt includes critical requirements."""
        required_keywords = [
            "GameState",
            "player_pos",
            "confidence",
            "JSON",
            "enemies",
            "items",
            "state",
        ]
        prompt_lower = VISION_SYSTEM_PROMPT_INSTRUCT.lower()
        for keyword in required_keywords:
            assert keyword.lower() in prompt_lower, f"Missing keyword: {keyword}"

    def test_thinking_prompt_contains_steps(self):
        """Thinking prompt includes step-by-step reasoning."""
        required_steps = [
            "STEP 1",
            "STEP 2",
            "STEP 3",
            "STEP 4",
            "STEP 5",
            "STEP 6",
        ]
        prompt = VISION_SYSTEM_PROMPT_THINKING
        for step in required_steps:
            assert step in prompt, f"Missing reasoning step: {step}"

    def test_both_prompts_require_json(self):
        """Both prompts emphasize JSON output."""
        assert "JSON" in VISION_SYSTEM_PROMPT_INSTRUCT
        assert "JSON" in VISION_SYSTEM_PROMPT_THINKING

    def test_both_prompts_cover_game_state_fields(self):
        """Both prompts document key GameState fields."""
        fields = ["player_pos", "floor", "state", "enemies", "items", "confidence"]
        for prompt in [VISION_SYSTEM_PROMPT_INSTRUCT, VISION_SYSTEM_PROMPT_THINKING]:
            for field in fields:
                assert field in prompt, f"Missing field documentation: {field}"


class TestGetVisionSystemPrompt:
    """Test prompt variant selector."""

    def test_get_instruct_variant(self):
        """Get instruct variant by name."""
        prompt = get_vision_system_prompt("instruct")
        assert prompt == VISION_SYSTEM_PROMPT_INSTRUCT

    def test_get_thinking_variant(self):
        """Get thinking variant by name."""
        prompt = get_vision_system_prompt("thinking")
        assert prompt == VISION_SYSTEM_PROMPT_THINKING

    def test_default_is_instruct(self):
        """Default variant is instruct."""
        prompt = get_vision_system_prompt()
        assert prompt == VISION_SYSTEM_PROMPT_INSTRUCT

    def test_invalid_variant_raises(self):
        """Raise ValueError for invalid variant."""
        with pytest.raises(ValueError):
            get_vision_system_prompt("invalid_variant")


class TestPromptBuilder:
    """Test PromptBuilder class for composing complete prompts."""

    def test_instantiate_instruct(self):
        """Create builder with instruct variant."""
        builder = PromptBuilder("instruct")
        assert builder.model_variant == "instruct"
        assert builder.system_prompt == VISION_SYSTEM_PROMPT_INSTRUCT

    def test_instantiate_thinking(self):
        """Create builder with thinking variant."""
        builder = PromptBuilder("thinking")
        assert builder.model_variant == "thinking"
        assert builder.system_prompt == VISION_SYSTEM_PROMPT_THINKING

    def test_default_variant_is_instruct(self):
        """Default variant is instruct."""
        builder = PromptBuilder()
        assert builder.model_variant == "instruct"

    def test_add_few_shot_examples(self):
        """Add few-shot examples from Phase 1."""
        builder = PromptBuilder()
        result = builder.add_few_shot_examples(num_examples=3)

        # Check method chaining
        assert result is builder
        assert len(builder.few_shot_examples) == 3

        # Check examples are valid
        for example in builder.few_shot_examples:
            assert "description" in example
            assert "state" in example
            assert isinstance(example["state"], GameState)

    def test_add_context(self):
        """Add execution context."""
        builder = PromptBuilder()
        result = builder.add_context(policy_hint="explore", model_size="4B")

        # Check method chaining
        assert result is builder
        assert builder.context["policy_hint"] == "explore"
        assert builder.context["model_size"] == "4B"

    def test_method_chaining(self):
        """Builder supports method chaining."""
        builder = (PromptBuilder("instruct")
                   .add_few_shot_examples(2)
                   .add_context(policy_hint="fight", model_size="2B"))

        assert len(builder.few_shot_examples) == 2
        assert builder.context["policy_hint"] == "fight"

    def test_get_system_prompt(self):
        """Retrieve system prompt."""
        builder = PromptBuilder("thinking")
        prompt = builder.get_system_prompt()
        assert prompt == VISION_SYSTEM_PROMPT_THINKING

    def test_build_user_prompt_without_examples(self):
        """Build user prompt without examples."""
        builder = PromptBuilder()
        user_prompt = builder.build_user_prompt()

        assert isinstance(user_prompt, str)
        assert "EXAMPLE OUTPUTS" not in user_prompt  # No examples added
        assert "CURRENT SCREENSHOT ANALYSIS" in user_prompt

    def test_build_user_prompt_with_examples(self):
        """Build user prompt with examples."""
        builder = PromptBuilder().add_few_shot_examples(2)
        user_prompt = builder.build_user_prompt()

        assert "EXAMPLE OUTPUTS" in user_prompt
        assert "Example 1:" in user_prompt
        assert "Example 2:" in user_prompt
        # Examples should contain JSON
        assert "{" in user_prompt and "}" in user_prompt

    def test_build_user_prompt_with_context(self):
        """Build user prompt with policy context."""
        builder = PromptBuilder().add_context(policy_hint="explore")
        user_prompt = builder.build_user_prompt()

        assert "Policy Hint: explore" in user_prompt

    def test_build_complete_prompt(self):
        """Build complete prompt dict with system and user."""
        builder = (PromptBuilder("thinking")
                   .add_few_shot_examples(3)
                   .add_context(policy_hint="boss_battle", model_size="4B"))

        complete = builder.build_complete_prompt()

        assert isinstance(complete, dict)
        assert "system" in complete
        assert "user" in complete
        assert complete["system"] == VISION_SYSTEM_PROMPT_THINKING
        assert "EXAMPLE OUTPUTS" in complete["user"]
        assert "boss_battle" in complete["user"]

    def test_prompt_includes_few_shot_json(self):
        """Few-shot examples in prompt are valid JSON."""
        builder = PromptBuilder().add_few_shot_examples(1)
        user_prompt = builder.build_user_prompt()

        # Extract JSON from prompt (between first { and last })
        start = user_prompt.find('{')
        end = user_prompt.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = user_prompt[start:end]
            # Should be parseable as JSON
            parsed = json.loads(json_str)
            assert "player_pos" in parsed
            assert "state" in parsed


class TestFormatVisionPromptWithExamples:
    """Test high-level prompt formatting function."""

    def test_basic_format(self):
        """Format prompt with defaults."""
        result = format_vision_prompt_with_examples()

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result
        assert result["system"] == VISION_SYSTEM_PROMPT_INSTRUCT

    def test_with_thinking_variant(self):
        """Format with thinking variant."""
        result = format_vision_prompt_with_examples(model_variant="thinking")
        assert result["system"] == VISION_SYSTEM_PROMPT_THINKING

    def test_with_policy_hint(self):
        """Include policy hint in prompt."""
        result = format_vision_prompt_with_examples(policy_hint="retreat")
        assert "retreat" in result["user"]

    def test_with_num_examples(self):
        """Control number of examples."""
        result = format_vision_prompt_with_examples(num_examples=5)
        # Should have 5 examples
        assert result["user"].count("Example") == 5

    def test_with_model_size(self):
        """Include model size context."""
        result = format_vision_prompt_with_examples(model_size="2B")
        # Model size affects variant selection (4B/smaller = instruct)
        assert "system" in result

    def test_all_parameters(self):
        """Format with all parameters."""
        result = format_vision_prompt_with_examples(
            policy_hint="explore",
            model_variant="thinking",
            num_examples=3,
            model_size="4B"
        )

        assert result["system"] == VISION_SYSTEM_PROMPT_THINKING
        assert "explore" in result["user"]
        assert result["user"].count("Example") == 3


class TestGetSchemaGuidance:
    """Test schema guidance export."""

    def test_get_schema_guidance(self):
        """Get compact schema for LM guidance."""
        guidance = get_schema_guidance()

        assert isinstance(guidance, str)
        assert len(guidance) > 50
        # Should contain key field names
        assert "player_pos" in guidance
        assert "state" in guidance

    def test_schema_guidance_is_valid_json(self):
        """Schema guidance parses as valid JSON."""
        guidance = get_schema_guidance()
        parsed = json.loads(guidance)

        assert isinstance(parsed, dict)
        assert "player_pos" in parsed
        assert "state" in parsed


class TestPromptBenchmark:
    """Performance benchmarks for prompt operations."""

    def test_prompt_builder_instantiation_fast(self):
        """Builder instantiation is fast."""
        import time

        start = time.time()
        for _ in range(100):
            PromptBuilder()
        elapsed = time.time() - start

        assert elapsed < 0.1  # 100 instantiations < 100ms

    def test_get_vision_system_prompt_fast(self):
        """Prompt selector is fast."""
        import time

        start = time.time()
        for _ in range(1000):
            get_vision_system_prompt("instruct")
            get_vision_system_prompt("thinking")
        elapsed = time.time() - start

        assert elapsed < 0.05  # 2000 calls < 50ms

    def test_build_complete_prompt_fast(self):
        """Building complete prompt is fast."""
        import time

        builder = (PromptBuilder()
                   .add_few_shot_examples(3)
                   .add_context(policy_hint="explore"))

        start = time.time()
        for _ in range(50):
            builder.build_complete_prompt()
        elapsed = time.time() - start

        assert elapsed < 0.5  # 50 builds < 500ms


class TestPromptEdgeCases:
    """Test edge cases and error conditions."""

    def test_builder_with_zero_examples(self):
        """Builder handles zero examples gracefully."""
        builder = PromptBuilder().add_few_shot_examples(0)
        assert len(builder.few_shot_examples) == 0
        user_prompt = builder.build_user_prompt()
        assert "EXAMPLE OUTPUTS" not in user_prompt

    def test_builder_with_max_examples(self):
        """Builder handles maximum examples (5)."""
        builder = PromptBuilder().add_few_shot_examples(5)
        assert len(builder.few_shot_examples) == 5

    def test_builder_with_empty_policy_hint(self):
        """Builder handles empty policy hint."""
        builder = PromptBuilder().add_context(policy_hint="")
        user_prompt = builder.build_user_prompt()
        # Should not have "Policy Hint: " line if empty
        assert "Policy Hint:" not in user_prompt

    def test_prompt_variant_names_case_sensitive(self):
        """Prompt variant names are case-sensitive."""
        with pytest.raises(ValueError):
            get_vision_system_prompt("INSTRUCT")  # Wrong case

    def test_format_with_invalid_variant(self):
        """format_vision_prompt_with_examples validates variant."""
        with pytest.raises(ValueError):
            format_vision_prompt_with_examples(model_variant="invalid")
