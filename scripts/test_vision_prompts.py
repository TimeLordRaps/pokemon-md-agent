#!/usr/bin/env python
"""Quick test script for Phase 2 vision prompts validation."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run quick validation of vision prompts."""
    try:
        from src.models.vision_prompts import (
            VISION_SYSTEM_PROMPT_INSTRUCT,
            VISION_SYSTEM_PROMPT_THINKING,
            PromptBuilder,
            get_vision_system_prompt,
            format_vision_prompt_with_examples,
        )
        from src.orchestrator.message_packager import (
            get_vision_system_prompt_for_model,
            pack_with_vision_prompts,
        )

        print("[INFO] Vision prompts imports successful")

        # Test 1: Instruct prompt exists
        assert VISION_SYSTEM_PROMPT_INSTRUCT
        print("[OK] Instruct system prompt: PASS")
        print(f"  - Length: {len(VISION_SYSTEM_PROMPT_INSTRUCT)} characters")
        print(f"  - Contains 'JSON': {'JSON' in VISION_SYSTEM_PROMPT_INSTRUCT}")

        # Test 2: Thinking prompt exists
        assert VISION_SYSTEM_PROMPT_THINKING
        print("[OK] Thinking system prompt: PASS")
        print(f"  - Length: {len(VISION_SYSTEM_PROMPT_THINKING)} characters")
        print(f"  - Contains 'STEP': {'STEP' in VISION_SYSTEM_PROMPT_THINKING}")

        # Test 3: Prompt variant selector
        instruct = get_vision_system_prompt("instruct")
        thinking = get_vision_system_prompt("thinking")
        assert instruct == VISION_SYSTEM_PROMPT_INSTRUCT
        assert thinking == VISION_SYSTEM_PROMPT_THINKING
        print("[OK] Prompt variant selector: PASS")

        # Test 4: PromptBuilder
        builder = PromptBuilder("instruct")
        builder.add_few_shot_examples(3)
        builder.add_context(policy_hint="explore", model_size="4B")
        complete = builder.build_complete_prompt()
        assert "system" in complete
        assert "user" in complete
        assert complete["system"] == VISION_SYSTEM_PROMPT_INSTRUCT
        print("[OK] PromptBuilder: PASS")
        print(f"  - System prompt length: {len(complete['system'])} chars")
        print(f"  - User prompt length: {len(complete['user'])} chars")
        print(f"  - Few-shot examples: 3")

        # Test 5: Model-specific prompts
        for model_size in ['2B', '4B', '8B']:
            prompt = get_vision_system_prompt_for_model(model_size)
            assert prompt
            print(f"[OK] Vision prompt for {model_size}: PASS")

        # Test 6: Integration with message packager
        step_state = {
            'dynamic_map': None,
            'event_log': [],
            'retrieved_trajs': [],
            'now': {'env': None, 'grid': None},
            'retrieved_thumbnails': [],
        }

        system_prompt, messages = pack_with_vision_prompts(
            step_state,
            policy_hint="explore",
            model_size="4B"
        )

        assert system_prompt
        assert messages
        assert len(messages) == 3
        print("[OK] Message packager integration: PASS")
        print(f"  - System prompt length: {len(system_prompt)} chars")
        print(f"  - Messages: {len(messages)} (3-message protocol)")

        # Test 7: format_vision_prompt_with_examples
        result = format_vision_prompt_with_examples(
            policy_hint="fight",
            model_variant="thinking",
            num_examples=3,
            model_size="4B"
        )
        assert "system" in result
        assert "user" in result
        assert "fight" in result["user"]
        print("[OK] format_vision_prompt_with_examples: PASS")

        print("\n[OK] All validation checks passed!")
        return 0

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
