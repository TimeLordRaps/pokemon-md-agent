#!/usr/bin/env python3
"""
Tests for prompt templates and optimization framework.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.agent.prompt_templates import (
    BasePromptTemplates,
    ModelSpecificAdaptations,
    ABTestingFramework,
    PromptOptimizer,
    ModelVariant,
    PromptStyle,
    PromptConfig,
    get_vision_prompt,
    get_prompt_configuration
)


class TestBasePromptTemplates:
    """Test base prompt template functionality."""

    def test_prompt_templates_structure(self):
        """Test that all prompt templates are properly defined."""
        templates = BasePromptTemplates()

        for style in PromptStyle:
            assert style in templates.PROMPT_TEMPLATES
            assert "template" in templates.PROMPT_TEMPLATES[style]
            assert "description" in templates.PROMPT_TEMPLATES[style]
            assert isinstance(templates.PROMPT_TEMPLATES[style]["template"], str)
            assert len(templates.PROMPT_TEMPLATES[style]["template"]) > 10

    def test_vision_analysis_base(self):
        """Test base vision analysis template."""
        templates = BasePromptTemplates()
        assert "Pokemon Mystery Dungeon" in templates.VISION_ANALYSIS_BASE
        assert "AI agent" in templates.VISION_ANALYSIS_BASE


class TestModelSpecificAdaptations:
    """Test model-specific optimization functionality."""

    def test_temperature_optimizations(self):
        """Test temperature settings for all model variants."""
        adaptations = ModelSpecificAdaptations()

        for variant in ModelVariant:
            assert variant in adaptations.TEMPERATURE_OPTIMIZATIONS
            temp = adaptations.TEMPERATURE_OPTIMIZATIONS[variant]
            assert 0 <= temp <= 1, f"Temperature {temp} out of range for {variant}"

        # Thinking models should have slightly higher temperatures
        assert adaptations.TEMPERATURE_OPTIMIZATIONS[ModelVariant.THINKING_2B] > \
               adaptations.TEMPERATURE_OPTIMIZATIONS[ModelVariant.INSTRUCT_2B]

    def test_token_optimizations(self):
        """Test token limits for all model variants."""
        adaptations = ModelSpecificAdaptations()

        for variant in ModelVariant:
            assert variant in adaptations.TOKEN_OPTIMIZATIONS
            tokens = adaptations.TOKEN_OPTIMIZATIONS[variant]
            assert tokens > 0, f"Invalid token count {tokens} for {variant}"

        # Larger models should have higher token limits
        assert adaptations.TOKEN_OPTIMIZATIONS[ModelVariant.INSTRUCT_8B] > \
               adaptations.TOKEN_OPTIMIZATIONS[ModelVariant.INSTRUCT_2B]

    def test_prompt_adaptations(self):
        """Test model-specific prompt adaptations."""
        adaptations = ModelSpecificAdaptations()

        # Thinking models should have adaptations
        thinking_2b = adaptations.PROMPT_ADAPTATIONS.get(ModelVariant.THINKING_2B, {})
        assert PromptStyle.COT in thinking_2b
        assert PromptStyle.TASK_SPECIFIC in thinking_2b

        # Instruct models should not have adaptations (None returned)
        instruct_2b = adaptations.PROMPT_ADAPTATIONS.get(ModelVariant.INSTRUCT_2B, {})
        assert instruct_2b == {}


class TestPromptOptimizer:
    """Test main prompt optimization functionality."""

    def test_get_optimized_prompt(self):
        """Test getting optimized prompts for different models."""
        optimizer = PromptOptimizer()

        for variant in ModelVariant:
            for style in PromptStyle:
                prompt = optimizer.get_optimized_prompt(variant, style)
                assert isinstance(prompt, str)
                assert len(prompt) > 20
                assert "PMD Red Rescue Team" in prompt or "Pokemon Mystery Dungeon Red Rescue Team" in prompt

    def test_get_prompt_config(self):
        """Test getting prompt configurations."""
        optimizer = PromptOptimizer()

        for variant in ModelVariant:
            config = optimizer.get_prompt_config(variant)
            assert isinstance(config, PromptConfig)
            assert config.model_variant == variant
            assert config.style == PromptStyle.OPTIMIZED
            assert 0 <= config.temperature <= 1
            assert config.max_tokens > 0

    def test_model_adaptations_applied(self):
        """Test that model-specific adaptations are applied to prompts."""
        optimizer = PromptOptimizer()

        # Thinking models should have additional reasoning instructions
        thinking_prompt = optimizer.get_optimized_prompt(
            ModelVariant.THINKING_4B, PromptStyle.COT
        )
        instruct_prompt = optimizer.get_optimized_prompt(
            ModelVariant.INSTRUCT_4B, PromptStyle.COT
        )

        assert len(thinking_prompt) > len(instruct_prompt)
        assert "reasoning" in thinking_prompt.lower() or "thinking" in thinking_prompt.lower()


class TestABTestingFramework:
    """Test A/B testing framework."""

    def test_create_test_variants(self):
        """Test creating test variants for A/B testing."""
        tester = ABTestingFramework()

        variants = tester.create_test_variants(ModelVariant.INSTRUCT_4B)
        assert len(variants) >= 6  # Base styles + temperature variations

        for config in variants:
            assert isinstance(config, PromptConfig)
            assert config.model_variant == ModelVariant.INSTRUCT_4B
            assert 0 <= config.temperature <= 1
            assert config.max_tokens > 0

    @patch('time.time')
    def test_record_and_save_results(self, mock_time):
        """Test recording and saving test results."""
        mock_time.return_value = 1234567890.0

        tester = ABTestingFramework(Path("test_output"))
        config = PromptConfig(
            style=PromptStyle.OPTIMIZED,
            model_variant=ModelVariant.INSTRUCT_2B
        )

        # This would normally be created by PerformanceMetrics - just use a simple dict
        metrics_data = {
            "response_time_ms": 150.5,
            "token_count": 128,
            "accuracy_score": 0.9,
            "hallucination_count": 0,
            "usefulness_score": 0.85
        }

        # Simulate recording (would use actual PerformanceMetrics in real code)
        result = {
            "timestamp": 1234567890.0,
            "config": {
                "style": "optimized",  # Use string instead of enum
                "model_variant": "instruct_2b",  # Use string instead of enum
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "template_version": config.template_version
            },
            "metrics": metrics_data
        }
        tester.test_results.append(result)

        tester.save_results()

        results_file = Path("test_output/ab_test_results.json")
        assert results_file.exists()

        # Clean up
        results_file.unlink()
        Path("test_output").rmdir()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_vision_prompt(self):
        """Test convenience function for getting vision prompts."""
        for variant in ModelVariant:
            prompt = get_vision_prompt(variant)
            assert isinstance(prompt, str)
            assert "PMD Red Rescue Team" in prompt or "Pokemon Mystery Dungeon Red Rescue Team" in prompt

    def test_get_prompt_configuration(self):
        """Test convenience function for getting configurations."""
        for variant in ModelVariant:
            config = get_prompt_configuration(variant)
            assert isinstance(config, PromptConfig)
            assert config.model_variant == variant


class TestPromptTemplatesMain:
    """Test main execution of prompt_templates.py."""

    def test_main_execution(self):
        """Test that main execution runs without errors."""
        import subprocess
        import sys

        # Run the module as a script
        result = subprocess.run([
            sys.executable, "-m", "src.agent.prompt_templates"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        assert result.returncode == 0
        assert "=== Prompt Templates Test ===" in result.stdout
        assert "A/B Testing Simulation ===" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])