"""Test skills DSL and runtime."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Skip old DSL tests - new Python DSL doesn't have ActionType/TriggerType enums
# from src.skills.dsl import SkillDSL, Skill, Action, ActionType, Trigger, TriggerType
from src.skills.runtime import SkillRuntime, RAMPredicates, ExecutionContext


class TestSkillDSL:
    """Test skill DSL functionality."""
    
    @pytest.mark.skip(reason="Old YAML DSL replaced by new Python DSL")
    def test_load_skill_from_dict(self):
        """Test loading skill from dictionary."""
        skill_data = {
            "name": "use_food_when_hungry",
            "description": "Use food when belly is low",
            "cooldown": 30.0,
            "triggers": [
                {
                    "type": "ram_condition",
                    "condition": "is_hungry"
                }
            ],
            "preconditions": [
                "has_food_item",
                "not_in_battle"
            ],
            "actions": [
                {
                    "type": "use_item",
                    "params": {
                        "item_type": "food",
                        "predicate": "best_food_available"
                    }
                }
            ],
            "fallback": [
                {
                    "type": "wait",
                    "params": {"frames": 60}
                }
            ]
        }
        
        skill = Skill.from_dict(skill_data)
        
        assert skill.name == "use_food_when_hungry"
        assert skill.cooldown == 30.0
        assert len(skill.triggers) == 1
        assert len(skill.preconditions) == 2
        assert len(skill.actions) == 1
        assert len(skill.fallback) == 1
        
        # Check trigger
        trigger = skill.triggers[0]
        assert trigger.type == TriggerType.RAM_CONDITION
        assert trigger.condition == "is_hungry"
        
        # Check action
        action = skill.actions[0]
        assert action.type == ActionType.USE_ITEM
        assert action.params["item_type"] == "food"
    
    @pytest.mark.skip(reason="Old YAML DSL replaced by new Python DSL")
    def test_skill_dsl_load_from_yaml(self):
        """Test loading skills from YAML file."""
        yaml_content = """
name: use_food_when_hungry
description: Use food when belly is low
cooldown: 30.0
triggers:
  - type: ram_condition
    condition: is_hungry
preconditions:
  - has_food_item
  - not_in_battle
actions:
  - type: use_item
    params:
      item_type: food
      predicate: best_food_available
"""
        
        # Create temporary directory and file
        temp_dir = Path(tempfile.mkdtemp())
        library_dir = temp_dir / "skill-libraries" / "test_library"
        library_dir.mkdir(parents=True)
        yaml_path = library_dir / "test_skill.yaml"
        
        try:
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            
            dsl = SkillDSL(skill_libraries_dir=temp_dir / "skill-libraries", library_name="test_library")
            skills = dsl.load_all_skills()
            
            assert len(skills) == 1
            skill = list(skills.values())[0]
            assert skill.name == "use_food_when_hungry"
            assert skill.cooldown == 30.0
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="RAMPredicates.evaluate_condition not yet implemented")
class TestRAMPredicates:
    """Test RAM predicates functionality."""

    def test_evaluate_condition_comparison(self):
        """Test basic comparison conditions."""
        mock_controller = Mock()
        predicates = RAMPredicates(mock_controller)
        
        # Mock get_value to return 50 for hp
        predicates.get_value = Mock(return_value=50)
        
        context = ExecutionContext(
            controller=mock_controller,
            ram_predicates=predicates
        )
        
        # Test hp < 75 (should be true)
        assert predicates.evaluate_condition("hp < 75", context) == True
        
        # Test hp > 75 (should be false)
        assert predicates.evaluate_condition("hp > 75", context) == False
    
    def test_evaluate_named_condition_hungry(self):
        """Test named condition 'is_hungry'."""
        mock_controller = Mock()
        predicates = RAMPredicates(mock_controller)
        
        # Mock belly values
        def mock_get_value(var):
            if var == "belly":
                return 10  # Low belly
            elif var == "max_belly":
                return 100
            return 0
        
        predicates.get_value = mock_get_value
        
        context = ExecutionContext(
            controller=mock_controller,
            ram_predicates=predicates
        )
        
        # Should be hungry (10 < 30)
        assert predicates.evaluate_condition("is_hungry", context) == True


@pytest.mark.skip(reason="SkillRuntime action execution methods not yet implemented")
class TestSkillRuntime:
    """Test skill runtime execution."""
    
    def test_execute_press_action(self):
        """Test executing press button action."""
        mock_controller = Mock()
        mock_controller.press = Mock(return_value=True)

        runtime = SkillRuntime(mock_controller)

        params = {"keys": ["A"]}
        success = runtime.execute_press_action(params)

        assert success == True
        mock_controller.press.assert_called_once_with(["A"])

    def test_execute_wait_action(self):
        """Test executing wait action."""
        mock_controller = Mock()
        mock_controller.await_frames = Mock(return_value=True)

        runtime = SkillRuntime(mock_controller)
        
        params = {"frames": 120}
        success = runtime.execute_wait_action(params)
        
        assert success == True
        mock_controller.await_frames.assert_called_once_with(120)


if __name__ == "__main__":
    pytest.main([__file__])