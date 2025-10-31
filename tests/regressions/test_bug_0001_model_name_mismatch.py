"""
Bug #0001: Model name inconsistency - "Reasoning" vs "Thinking"

Symptoms: Model loading will fail because the code references models with
"Reasoning" in the name, but the actual Qwen3-VL models use "Thinking".

Affected Files:
- src/agent/model_router.py (lines 31-44)
- src/agent/qwen_controller.py (lines 298-320)

Root Cause: Inconsistent naming convention across codebase. MODEL_NAMES dict
uses "Reasoning" suffix, but actual HuggingFace model IDs use "Thinking".
The ArmadaRegistry in qwen_controller.py has correct names (lines 69-126),
but load_models() method uses incorrect names.

Expected Model Names (per mission spec):
1. Qwen/Qwen3-VL-2B-Thinking-FP8
2. unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
3. unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit
4. unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit
5. unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit
6. unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit

Actual (Incorrect) Names in Code:
- model_router.py uses "Reasoning" for all thinking variants
- qwen_controller.py uses "Qwen/Qwen3-VL-2B-Reasoning-FP8" (should be Thinking)
- qwen_controller.py uses "Qwen3-VL-4B-Reasoning" and "Qwen3-VL-8B-Reasoning"

Fix: Replace all "Reasoning" with "Thinking" in model names.

Impact: HIGH - Model loading will fail, blocking all agent functionality.
Priority: P0 - Must fix before any model can be loaded.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.model_router import MODEL_NAMES, ModelSize
from src.agent.qwen_controller import QwenController


class TestBug0001ModelNameMismatch:
    """Test suite to verify correct model naming convention."""

    # Expected model names per mission spec
    EXPECTED_MODELS = {
        "2B": {
            "instruct": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
            "thinking": "Qwen/Qwen3-VL-2B-Thinking-FP8",  # FP8 only, no bnb-4bit
        },
        "4B": {
            "instruct": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
            "thinking": "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
        },
        "8B": {
            "instruct": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
            "thinking": "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
        },
    }

    def test_model_router_names_correct(self):
        """Test that MODEL_NAMES dict contains correct model IDs."""
        # Check 2B Thinking (special FP8 case)
        assert MODEL_NAMES[ModelSize.SIZE_2B]["thinking"] == self.EXPECTED_MODELS["2B"]["thinking"], (
            f"2B Thinking model incorrect: expected {self.EXPECTED_MODELS['2B']['thinking']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_2B]['thinking']}"
        )

        # Check 2B Instruct
        assert MODEL_NAMES[ModelSize.SIZE_2B]["instruct"] == self.EXPECTED_MODELS["2B"]["instruct"], (
            f"2B Instruct model incorrect: expected {self.EXPECTED_MODELS['2B']['instruct']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_2B]['instruct']}"
        )

        # Check 4B models
        assert MODEL_NAMES[ModelSize.SIZE_4B]["thinking"] == self.EXPECTED_MODELS["4B"]["thinking"], (
            f"4B Thinking model incorrect: expected {self.EXPECTED_MODELS['4B']['thinking']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_4B]['thinking']}"
        )
        assert MODEL_NAMES[ModelSize.SIZE_4B]["instruct"] == self.EXPECTED_MODELS["4B"]["instruct"], (
            f"4B Instruct model incorrect: expected {self.EXPECTED_MODELS['4B']['instruct']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_4B]['instruct']}"
        )

        # Check 8B models
        assert MODEL_NAMES[ModelSize.SIZE_8B]["thinking"] == self.EXPECTED_MODELS["8B"]["thinking"], (
            f"8B Thinking model incorrect: expected {self.EXPECTED_MODELS['8B']['thinking']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_8B]['thinking']}"
        )
        assert MODEL_NAMES[ModelSize.SIZE_8B]["instruct"] == self.EXPECTED_MODELS["8B"]["instruct"], (
            f"8B Instruct model incorrect: expected {self.EXPECTED_MODELS['8B']['instruct']}, "
            f"got {MODEL_NAMES[ModelSize.SIZE_8B]['instruct']}"
        )

    def test_armada_registry_names_correct(self):
        """Test that ArmadaRegistry contains correct model IDs."""
        controller = QwenController()
        registry = controller.get_armada_registry()

        # Check that we have exactly 6 models
        assert len(registry) == 6, f"Expected 6 models in registry, got {len(registry)}"

        # Check 2B Thinking (FP8)
        assert "qwen3-vl-2b-thinking" in registry
        assert registry["qwen3-vl-2b-thinking"]["model_name"] == self.EXPECTED_MODELS["2B"]["thinking"], (
            f"Registry 2B Thinking incorrect: expected {self.EXPECTED_MODELS['2B']['thinking']}, "
            f"got {registry['qwen3-vl-2b-thinking']['model_name']}"
        )

        # Check 2B Instruct
        assert "qwen3-vl-2b-instruct" in registry
        assert registry["qwen3-vl-2b-instruct"]["model_name"] == self.EXPECTED_MODELS["2B"]["instruct"]

        # Check 4B models
        assert "qwen3-vl-4b-thinking" in registry
        assert registry["qwen3-vl-4b-thinking"]["model_name"] == self.EXPECTED_MODELS["4B"]["thinking"]
        assert "qwen3-vl-4b-instruct" in registry
        assert registry["qwen3-vl-4b-instruct"]["model_name"] == self.EXPECTED_MODELS["4B"]["instruct"]

        # Check 8B models
        assert "qwen3-vl-8b-thinking" in registry
        assert registry["qwen3-vl-8b-thinking"]["model_name"] == self.EXPECTED_MODELS["8B"]["thinking"]
        assert "qwen3-vl-8b-instruct" in registry
        assert registry["qwen3-vl-8b-instruct"]["model_name"] == self.EXPECTED_MODELS["8B"]["instruct"]

    def test_no_reasoning_in_model_names(self):
        """Test that no model names contain 'Reasoning' (should be 'Thinking')."""
        # Check MODEL_NAMES dict
        for size, variants in MODEL_NAMES.items():
            for variant, model_name in variants.items():
                assert "Reasoning" not in model_name, (
                    f"Found 'Reasoning' in {size.value} {variant} model: {model_name}. "
                    f"Should use 'Thinking' instead."
                )

        # Check ArmadaRegistry
        controller = QwenController()
        registry = controller.get_armada_registry()
        for model_key, entry in registry.items():
            model_name = entry["model_name"]
            assert "Reasoning" not in model_name, (
                f"Found 'Reasoning' in registry key '{model_key}': {model_name}. "
                f"Should use 'Thinking' instead."
            )

    def test_2b_thinking_is_fp8_not_bnb4bit(self):
        """Test that 2B Thinking model uses FP8 quantization, not bnb4bit."""
        # Check that 2B Thinking model name explicitly mentions FP8
        thinking_model = MODEL_NAMES[ModelSize.SIZE_2B]["thinking"]
        assert "FP8" in thinking_model, (
            f"2B Thinking model should be FP8 quantized: {thinking_model}"
        )
        assert "bnb-4bit" not in thinking_model, (
            f"2B Thinking model should NOT be bnb-4bit (only FP8 available): {thinking_model}"
        )

        # Verify it's from Qwen org, not unsloth
        assert thinking_model.startswith("Qwen/"), (
            f"2B Thinking FP8 model should be from Qwen org: {thinking_model}"
        )

        # Check registry entry
        controller = QwenController()
        registry = controller.get_armada_registry()
        thinking_entry = registry["qwen3-vl-2b-thinking"]
        assert thinking_entry["quantization"] == "fp8", (
            f"2B Thinking registry entry should have quantization='fp8', "
            f"got '{thinking_entry['quantization']}'"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
