"""
Vision model system prompts for Pokemon MD-Red agent.

Provides structured prompts for Qwen3-VL to output GameState-compliant JSON.
Supports both instruct and thinking model variants.

Key features:
- Instruct variant: Direct JSON output for 2B/4B models
- Thinking variant: Chain-of-thought reasoning for reasoning-enabled models
- Few-shot examples: 3-5 in-context examples for learning
- Schema guidance: Full GameState schema structure
- Model-specific optimization: Tailored for Qwen3-VL
"""

from typing import List, Optional, Dict, Any
from src.models.game_state_schema import GameState, GameStateEnum, RoomType
from src.models.game_state_utils import (
    generate_few_shot_examples,
    schema_to_prompt_json,
)

# ============================================================================
# INSTRUCT VARIANT - For 2B/4B models (Direct JSON output)
# ============================================================================

VISION_SYSTEM_PROMPT_INSTRUCT = """You are a Pokemon Mystery Dungeon game state analyzer.
Your task is to analyze game screenshots and extract structured game state information in JSON format.

CRITICAL REQUIREMENTS:
1. Always output valid JSON matching the GameState schema (provided below)
2. Player coordinate must be [x, y] with 0-indexed values (never negative)
3. For each entity (enemy/item), provide exact coordinates and type
4. Classify game state as one of: exploring, battle, menu, stairs, boss, unknown
5. Provide confidence score 0-1 indicating how certain you are about the observation
6. Keep threats and opportunities to maximum 3 items each
7. Return JSON only - no explanations or markdown

GAME STATE SCHEMA:
{
  "player_pos": "[x, y] - integer coordinates, 0-indexed",
  "player_hp": "integer or null (optional)",
  "floor": "integer 1-50",
  "dungeon_name": "string or null (optional)",
  "state": "exploring|battle|menu|stairs|boss|unknown",
  "enemies": [
    {
      "x": "integer coordinate",
      "y": "integer coordinate",
      "type": "enemy",
      "species": "Pokemon name",
      "hp": "integer or null",
      "level": "integer or null",
      "status_effects": ["status1", "status2"]
    }
  ],
  "items": [
    {
      "x": "integer coordinate",
      "y": "integer coordinate",
      "type": "item",
      "name": "Item name"
    }
  ],
  "confidence": "float 0.0-1.0",
  "threats": ["threat1", "threat2"],
  "opportunities": ["action1", "action2"],
  "notes": "Additional observations (optional)"
}

ANALYSIS RULES:
- Coordinate System: Screen is grid-based with 0,0 at top-left
- Entity Detection: Identify all visible enemies and items on screen
- State Classification: Determine if player is exploring, in battle, at menu, found stairs, or in boss battle
- Confidence: Rate your certainty about coordinates and entity types (0.5-1.0 range realistic)
- Threats: List immediate dangers (nearby enemies, status effects, low HP)
- Opportunities: List beneficial actions (available items, clear paths, retreats)

IMPORTANT:
- Return ONLY the JSON object, nothing else
- All required fields must be present
- Use null for optional fields without data
- Keep coordinates accurate and within bounds
- Confidence must be between 0 and 1"""

# ============================================================================
# THINKING VARIANT - For thinking models (Chain-of-thought reasoning)
# ============================================================================

VISION_SYSTEM_PROMPT_THINKING = """You are a Pokemon Mystery Dungeon game state analyzer with reasoning capability.
Your task is to analyze game screenshots, reason about the state, and extract structured game state in JSON.

STEP 1 - OBSERVATION:
First, describe what you see on screen:
- Player position and appearance
- Any enemies/NPCs present with their positions
- Items on the ground
- Environmental features (walls, doors, stairs)
- Status indicators (HP bar, floor number, dungeon name)

STEP 2 - COORDINATE EXTRACTION:
Map visual positions to grid coordinates:
- Identify the grid dimensions
- Player position: [x, y] with 0,0 at top-left
- For each entity, determine exact grid position
- Ensure all coordinates are non-negative integers

STEP 3 - ENTITY CLASSIFICATION:
For each visible entity:
- Is it an enemy? What species/type?
- Is it an item? What kind?
- What status effects/conditions are visible?
- Can you estimate HP or level from appearance?

STEP 4 - STATE DETERMINATION:
Determine game state:
- Is player in combat (enemy adjacent)?
- Is player exploring (moving freely)?
- Is player at menu/shop (UI visible)?
- Are stairs visible?
- Is this a boss room?

STEP 5 - THREAT & OPPORTUNITY ANALYSIS:
Threats: What immediate dangers exist?
- Nearby enemies and distances
- Player status effects
- Low HP condition
- Environmental hazards

Opportunities: What beneficial actions available?
- Movement paths
- Item locations and types
- Retreat options
- Healing opportunities

STEP 6 - CONFIDENCE ASSESSMENT:
Rate your confidence (0.0-1.0):
- High confidence (0.85+): Clear, unambiguous observations
- Medium confidence (0.65-0.84): Some uncertainty in details
- Low confidence (<0.65): Unclear/ambiguous scene

FINAL OUTPUT - RETURN ONLY JSON:
After reasoning through the above, output the GameState JSON:

{
  "player_pos": "[x, y]",
  "player_hp": "int or null",
  "floor": "int 1-50",
  "dungeon_name": "string or null",
  "state": "exploring|battle|menu|stairs|boss|unknown",
  "enemies": [...],
  "items": [...],
  "confidence": "float 0.0-1.0",
  "threats": ["threat1", "threat2"],
  "opportunities": ["action1", "action2"],
  "notes": "Additional observations"
}

CRITICAL REQUIREMENTS:
1. Player position must be valid [x, y] coordinates (non-negative integers)
2. All coordinates must be 0-indexed from top-left
3. Confidence must be between 0.0 and 1.0
4. Threats and opportunities limited to 3 items each
5. State must be one of: exploring, battle, menu, stairs, boss, unknown
6. Return ONLY valid JSON - no additional text"""


# ============================================================================
# PROMPT BUILDER CLASS
# ============================================================================

class PromptBuilder:
    """
    Type-safe builder for vision model prompts.

    Combines system prompt, schema context, few-shot examples, and user query.
    """

    def __init__(self, model_variant: str = "instruct"):
        """
        Initialize prompt builder.

        Args:
            model_variant: "instruct" or "thinking"
        """
        self.model_variant = model_variant
        self.system_prompt = get_vision_system_prompt(model_variant)
        self.few_shot_examples: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def add_few_shot_examples(self, num_examples: int = 3) -> "PromptBuilder":
        """
        Add few-shot examples from Phase 1 utilities.

        Args:
            num_examples: Number of examples to include (1-5)

        Returns:
            Self for method chaining
        """
        self.few_shot_examples = generate_few_shot_examples(num_examples)
        return self

    def add_context(self, policy_hint: str = "", model_size: str = "4B") -> "PromptBuilder":
        """
        Add execution context.

        Args:
            policy_hint: Current action policy (e.g., "explore", "fight", "retreat")
            model_size: Model size ("2B", "4B", "8B") for optimization hints

        Returns:
            Self for method chaining
        """
        self.context = {
            "policy_hint": policy_hint,
            "model_size": model_size,
        }
        return self

    def build_user_prompt(self) -> str:
        """
        Build complete user prompt with context and examples.

        Returns:
            Complete user prompt text
        """
        lines = []

        # Context
        if self.context.get("policy_hint"):
            lines.append(f"Policy Hint: {self.context['policy_hint']}")
            lines.append("")

        # Few-shot examples
        if self.few_shot_examples:
            lines.append("EXAMPLE OUTPUTS:")
            lines.append("-" * 60)
            for i, example in enumerate(self.few_shot_examples, 1):
                lines.append(f"Example {i}: {example['description']}")
                lines.append("Output:")
                import json
                state_json = example["state"].model_dump_json()
                lines.append(state_json)
                lines.append("")

        # Query
        lines.append("CURRENT SCREENSHOT ANALYSIS:")
        lines.append("Please analyze the current screenshot and output the game state as JSON.")

        return "\n".join(lines)

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt

    def build_complete_prompt(self) -> Dict[str, str]:
        """
        Build complete prompt structure.

        Returns:
            Dict with 'system' and 'user' keys
        """
        return {
            "system": self.get_system_prompt(),
            "user": self.build_user_prompt(),
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_vision_system_prompt(model_variant: str = "instruct") -> str:
    """
    Get vision system prompt for specified model variant.

    Args:
        model_variant: "instruct" or "thinking"

    Returns:
        System prompt string

    Raises:
        ValueError: If model_variant invalid
    """
    if model_variant == "instruct":
        return VISION_SYSTEM_PROMPT_INSTRUCT
    elif model_variant == "thinking":
        return VISION_SYSTEM_PROMPT_THINKING
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}. Use 'instruct' or 'thinking'.")


def format_vision_prompt_with_examples(
    policy_hint: str = "",
    model_variant: str = "instruct",
    num_examples: int = 3,
    model_size: str = "4B"
) -> Dict[str, str]:
    """
    Build complete vision prompt with examples and context.

    Args:
        policy_hint: Current action policy hint
        model_variant: "instruct" or "thinking"
        num_examples: Number of few-shot examples (1-5)
        model_size: Model size for context ("2B", "4B", "8B")

    Returns:
        Dict with 'system' and 'user' keys for model input
    """
    builder = PromptBuilder(model_variant)
    builder.add_few_shot_examples(num_examples)
    builder.add_context(policy_hint=policy_hint, model_size=model_size)
    return builder.build_complete_prompt()


def get_schema_guidance() -> str:
    """
    Get GameState schema in prompt-friendly format.

    Returns:
        Compact JSON schema for LM guidance
    """
    return schema_to_prompt_json()


__all__ = [
    "VISION_SYSTEM_PROMPT_INSTRUCT",
    "VISION_SYSTEM_PROMPT_THINKING",
    "PromptBuilder",
    "get_vision_system_prompt",
    "format_vision_prompt_with_examples",
    "get_schema_guidance",
]
