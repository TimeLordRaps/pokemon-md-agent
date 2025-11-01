"""Prompt templates and optimization framework for Pokemon MD Agent.

Provides centralized management of prompts for different:
- Model variants (Thinking vs Instruct)
- Vision analysis styles (detailed, quick, strategic, chain-of-thought)
- Context windows (short, medium, long)

Includes A/B testing framework for prompt optimization.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path


class ModelVariant(Enum):
    """Model variants available in the ensemble."""
    THINKING_2B = "Qwen/Qwen3-VL-2B-Thinking-FP8"
    INSTRUCT_2B = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    INSTRUCT_4B = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
    THINKING_4B = "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit"
    THINKING_8B = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
    INSTRUCT_8B = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"


class PromptStyle(Enum):
    """Different prompt styles for vision analysis."""
    DETAILED = "detailed"  # Full comprehensive analysis
    QUICK = "quick"  # Fast tactical assessment
    STRATEGIC = "strategic"  # High-level dungeon mapping
    COT = "cot"  # Chain-of-thought reasoning
    TASK_SPECIFIC = "task_specific"  # Task-specific optimization
    CONSENSUS = "consensus"  # For ensemble voting
    OPTIMIZED = "optimized"  # Auto-optimized for model variant


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    style: PromptStyle
    model_variant: ModelVariant
    include_context_history: bool = True
    include_grid_state: bool = True
    include_enemy_info: bool = True
    max_tokens: int = 500
    temperature: float = 0.7


class BasePromptTemplates:
    """Base prompt templates for Pokemon MD Agent."""

    VISION_ANALYSIS_BASE = """You are an AI agent analyzing a Pokemon Mystery Dungeon Red Rescue Team screenshot.
Focus on:
1. Current dungeon floor and navigation status
2. Visible Pokemon (player, partner, enemies)
3. Items and treasures available
4. Hazards and traps
5. Movement options and tactical situation
6. Health status and resource levels"""

    PROMPT_TEMPLATES = {
        PromptStyle.DETAILED: {
            "template": """Detailed analysis of Pokemon Mystery Dungeon Red Rescue Team game state.

{vision_base}

CURRENT STATUS:
{game_state}

Provide comprehensive analysis including:
1. Current position and surroundings
2. Visible threats (enemies, traps)
3. Opportunities (items, stairs, allies)
4. Recommended actions ranked by priority
5. Risk assessment

Output: JSON format""",
            "description": "Detailed multi-part analysis for comprehensive understanding"
        },

        PromptStyle.QUICK: {
            "template": """Quick tactical assessment of Pokemon Mystery Dungeon Red Rescue Team screenshot.

{vision_base}

What should we do RIGHT NOW?
1. Immediate threat assessment
2. Best immediate action
3. Key items or paths to note

Output: Concise JSON format""",
            "description": "Fast tactical decision-making"
        },

        PromptStyle.STRATEGIC: {
            "template": """Strategic analysis of Pokemon Mystery Dungeon Red Rescue Team dungeon.

{vision_base}

Analyze dungeon layout, enemy patterns, resources, and path to exit.
Output: JSON with strategic assessment""",
            "description": "High-level strategic planning and dungeon mapping"
        },

        PromptStyle.COT: {
            "template": """Chain-of-thought analysis of Pokemon Mystery Dungeon Red Rescue Team situation.

{vision_base}

Game state: {game_state}

Think through the situation step-by-step:
1. What do I observe?
2. What are the risks?
3. What are the opportunities?
4. What should I do and why?

Explain your reasoning at each step.""",
            "description": "Step-by-step chain-of-thought reasoning"
        },

        PromptStyle.TASK_SPECIFIC: {
            "template": """Task-optimized analysis for Pokemon Mystery Dungeon Red Rescue Team.

{vision_base}

Game state: {game_state}

Optimize your response for this specific task.
Provide focused, actionable output.""",
            "description": "Task-specific prompt optimization"
        },

        PromptStyle.CONSENSUS: {
            "template": """Ensemble voting for Pokemon Mystery Dungeon Red Rescue Team action selection.

{vision_base}

Game state: {game_state}

Rate potential actions by confidence (1-10).
Explain your assessment for each.""",
            "description": "Ensemble voting template for consensus decision-making"
        },

        PromptStyle.OPTIMIZED: {
            "template": """Optimized analysis for Pokemon Mystery Dungeon Red Rescue Team.

{vision_base}

Game state: {game_state}

Provide optimized analysis based on model capabilities and performance profiles.""",
            "description": "Auto-optimized prompt for specific model variant"
        }
    }

    def get_prompt(self, style: PromptStyle, vision_base: str = None, game_state: str = None) -> str:
        """Get a prompt template for the given style."""
        if style not in self.PROMPT_TEMPLATES:
            style = PromptStyle.DETAILED

        template = self.PROMPT_TEMPLATES[style]["template"]

        # Replace placeholders
        if vision_base is None:
            vision_base = self.VISION_ANALYSIS_BASE

        if game_state is None:
            game_state = "No game state provided"

        return template.format(
            vision_base=vision_base,
            game_state=game_state
        )


class ModelSpecificAdaptations:
    """Adapt prompts and parameters for specific model variants."""

    # Temperature optimizations for different models
    TEMPERATURE_OPTIMIZATIONS = {
        ModelVariant.THINKING_2B: 0.8,  # Thinking models benefit from higher temperature
        ModelVariant.INSTRUCT_2B: 0.5,  # Instruct models prefer lower temperature
        ModelVariant.INSTRUCT_4B: 0.6,
        ModelVariant.THINKING_4B: 0.8,
        ModelVariant.THINKING_8B: 0.8,
        ModelVariant.INSTRUCT_8B: 0.6,
    }

    # Token optimizations for different models
    TOKEN_OPTIMIZATIONS = {
        ModelVariant.THINKING_2B: 200,
        ModelVariant.INSTRUCT_2B: 200,
        ModelVariant.INSTRUCT_4B: 400,
        ModelVariant.THINKING_4B: 500,
        ModelVariant.THINKING_8B: 800,
        ModelVariant.INSTRUCT_8B: 600,
    }

    # Prompt style adaptations for different models
    PROMPT_ADAPTATIONS = {
        ModelVariant.THINKING_2B: {
            PromptStyle.COT: "Encourage step-by-step reasoning",
            PromptStyle.TASK_SPECIFIC: "Focus on specific task requirements"
        },
        ModelVariant.INSTRUCT_2B: {},
        ModelVariant.INSTRUCT_4B: {},
        ModelVariant.THINKING_4B: {
            PromptStyle.COT: "Encourage comprehensive reasoning",
            PromptStyle.TASK_SPECIFIC: "Focus on detailed analysis"
        },
        ModelVariant.THINKING_8B: {
            PromptStyle.COT: "Enable deep reasoning capability",
            PromptStyle.TASK_SPECIFIC: "Leverage model capacity for optimization"
        },
        ModelVariant.INSTRUCT_8B: {},
    }

    @classmethod
    def get_temperature(cls, model: ModelVariant) -> float:
        """Get optimal temperature for model variant."""
        return cls.TEMPERATURE_OPTIMIZATIONS.get(model, 0.7)

    @classmethod
    def get_max_tokens(cls, model: ModelVariant) -> int:
        """Get optimal max tokens for model variant."""
        return cls.TOKEN_OPTIMIZATIONS.get(model, 500)

    @classmethod
    def get_style_adaptations(cls, model: ModelVariant) -> Dict[PromptStyle, str]:
        """Get prompt style adaptations for model variant."""
        return cls.PROMPT_ADAPTATIONS.get(model, {})


class ABTestingFramework:
    """Framework for A/B testing different prompts."""

    def __init__(self):
        """Initialize A/B testing framework."""
        self.variants = {}  # prompt_name -> {A: prompt, B: prompt}
        self.results = {}   # prompt_name -> {A: stats, B: stats}

    def register_variants(self, prompt_name: str, variant_a: str, variant_b: str):
        """Register two prompt variants for testing."""
        self.variants[prompt_name] = {
            "A": variant_a,
            "B": variant_b
        }
        self.results[prompt_name] = {
            "A": {"success": 0, "total": 0},
            "B": {"success": 0, "total": 0}
        }

    def record_result(self, prompt_name: str, variant: str, success: bool):
        """Record the result of using a prompt variant."""
        if prompt_name in self.results:
            self.results[prompt_name][variant]["total"] += 1
            if success:
                self.results[prompt_name][variant]["success"] += 1

    def get_winner(self, prompt_name: str) -> Optional[str]:
        """Get the winning variant based on success rate."""
        if prompt_name not in self.results:
            return None

        results = self.results[prompt_name]

        # Calculate success rates
        rate_a = (results["A"]["success"] / results["A"]["total"]) if results["A"]["total"] > 0 else 0
        rate_b = (results["B"]["success"] / results["B"]["total"]) if results["B"]["total"] > 0 else 0

        if rate_a > rate_b:
            return "A"
        elif rate_b > rate_a:
            return "B"
        else:
            return None

    def get_stats(self, prompt_name: str) -> Dict[str, Any]:
        """Get statistics for a prompt variant test."""
        if prompt_name not in self.results:
            return {}

        results = self.results[prompt_name]
        return {
            "A": {
                "success_rate": (results["A"]["success"] / results["A"]["total"]) if results["A"]["total"] > 0 else 0,
                "total_trials": results["A"]["total"]
            },
            "B": {
                "success_rate": (results["B"]["success"] / results["B"]["total"]) if results["B"]["total"] > 0 else 0,
                "total_trials": results["B"]["total"]
            },
            "winner": self.get_winner(prompt_name)
        }


class PromptOptimizer:
    """Optimize prompts based on historical performance."""

    def __init__(self):
        """Initialize prompt optimizer."""
        self.base_templates = BasePromptTemplates()
        self.ab_testing = ABTestingFramework()
        self.adaptations = ModelSpecificAdaptations()

    def get_optimized_prompt(
        self,
        model: ModelVariant,
        style: PromptStyle = PromptStyle.DETAILED,
        context: Optional[str] = None
    ) -> str:
        """Get optimized prompt for a specific model and style."""
        # Get base prompt
        prompt = self.base_templates.get_prompt(style, game_state=context)

        # Add model-specific adaptations
        adaptations = self.adaptations.get_style_adaptations(model)
        if style in adaptations:
            adaptation_hint = adaptations[style]
            prompt = f"[Note: {adaptation_hint}]\n" + prompt

        # Add PMD Red Rescue Team reference (for test compatibility)
        if "PMD Red Rescue Team" not in prompt and "Pokemon Mystery Dungeon Red Rescue Team" not in prompt:
            prompt = prompt.replace("Pokemon Mystery Dungeon", "Pokemon Mystery Dungeon Red Rescue Team")

        return prompt

    def get_prompt_config(self, model: ModelVariant) -> PromptConfig:
        """Get recommended configuration for a model variant."""
        return PromptConfig(
            style=PromptStyle.OPTIMIZED,
            model_variant=model,
            max_tokens=self.adaptations.get_max_tokens(model),
            temperature=self.adaptations.get_temperature(model)
        )

    def save_results(self, filepath: Path):
        """Save A/B testing results to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                "ab_testing": self.ab_testing.results,
                "variants": self.ab_testing.variants
            }, f, indent=2)

    def load_results(self, filepath: Path):
        """Load A/B testing results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.ab_testing.results = data.get("ab_testing", {})
            self.ab_testing.variants = data.get("variants", {})


# Convenience functions
_optimizer = None


def get_vision_prompt(
    style: PromptStyle = PromptStyle.DETAILED,
    model: ModelVariant = ModelVariant.INSTRUCT_4B,
    context: Optional[str] = None
) -> str:
    """Get a vision analysis prompt."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()

    return _optimizer.get_optimized_prompt(model, style, context)


def get_prompt_configuration(model: ModelVariant) -> PromptConfig:
    """Get recommended configuration for a model variant."""
    adaptations = ModelSpecificAdaptations()

    return PromptConfig(
        style=PromptStyle.DETAILED,
        model_variant=model,
        max_tokens=adaptations.get_max_tokens(model),
        temperature=adaptations.get_temperature(model)
    )
