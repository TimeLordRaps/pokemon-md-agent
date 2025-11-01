"""Skill management system for autonomous agent learning and reuse.

Skills are composable action sequences that the agent can save, load, and invoke.
They represent learned behaviors that successfully accomplish specific tasks.
"""

import json
import logging
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillStep:
    """Single step in a skill execution."""
    action: str  # e.g., "move_down", "confirm"
    confidence: float  # 0.0-1.0, from model output
    observation: Optional[Dict[str, Any]] = None  # Game state at this step
    reasoning: Optional[str] = None  # Why this action was chosen


@dataclass
class Skill:
    """Learned behavior/skill that can be reused."""
    name: str
    description: str
    steps: List[SkillStep] = field(default_factory=list)
    precondition: Optional[str] = None  # e.g., "at_entrance_hall"
    postcondition: Optional[str] = None  # e.g., "reached_stairs"
    success_rate: float = 0.0  # 0.0-1.0
    times_used: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [asdict(step) for step in self.steps],
            "precondition": self.precondition,
            "postcondition": self.postcondition,
            "success_rate": self.success_rate,
            "times_used": self.times_used,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create skill from dictionary."""
        data_copy = data.copy()
        if "steps" in data_copy:
            data_copy["steps"] = [
                SkillStep(**step) if isinstance(step, dict) else step
                for step in data_copy["steps"]
            ]
        return cls(**data_copy)


class SkillManager:
    """Manage skill persistence and retrieval."""

    def __init__(self, skills_dir: Path = Path("config/skills")):
        """Initialize skill manager.

        Args:
            skills_dir: Directory to store skills
        """
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.skills: Dict[str, Skill] = {}
        self._load_all_skills()

    def _load_all_skills(self) -> None:
        """Load all skills from disk."""
        if not self.skills_dir.exists():
            return

        for skill_file in self.skills_dir.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    data = json.load(f)
                    skill = Skill.from_dict(data)
                    self.skills[skill.name] = skill
                    logger.debug(f"Loaded skill: {skill.name}")
            except Exception as e:
                logger.error(f"Failed to load skill {skill_file}: {e}")

    def _get_skill_path(self, name: str) -> Path:
        """Get file path for skill."""
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        return self.skills_dir / f"{safe_name}.json"

    def save_skill(self, skill: Skill) -> bool:
        """Save skill to disk.

        Args:
            skill: Skill to save

        Returns:
            True if successful
        """
        try:
            skill_path = self._get_skill_path(skill.name)
            with open(skill_path, "w") as f:
                json.dump(skill.to_dict(), f, indent=2)
            self.skills[skill.name] = skill
            logger.info(f"Saved skill: {skill.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save skill {skill.name}: {e}")
            return False

    def load_skill(self, name: str) -> Optional[Skill]:
        """Load skill by name.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise
        """
        if name in self.skills:
            return self.skills[name]

        skill_path = self._get_skill_path(name)
        if not skill_path.exists():
            logger.warning(f"Skill not found: {name}")
            return None

        try:
            with open(skill_path, "r") as f:
                data = json.load(f)
                skill = Skill.from_dict(data)
                self.skills[name] = skill
                return skill
        except Exception as e:
            logger.error(f"Failed to load skill {name}: {e}")
            return None

    def delete_skill(self, name: str) -> bool:
        """Delete skill.

        Args:
            name: Skill name

        Returns:
            True if successful
        """
        try:
            skill_path = self._get_skill_path(name)
            if skill_path.exists():
                skill_path.unlink()
            if name in self.skills:
                del self.skills[name]
            logger.info(f"Deleted skill: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete skill {name}: {e}")
            return False

    def list_skills(self, tags: Optional[List[str]] = None) -> List[Skill]:
        """List available skills.

        Args:
            tags: Optional filter by tags

        Returns:
            List of skills matching criteria
        """
        skills = list(self.skills.values())

        if tags:
            skills = [
                s for s in skills
                if any(tag in s.tags for tag in tags)
            ]

        return sorted(skills, key=lambda s: s.name)

    def create_skill_from_trajectory(
        self,
        name: str,
        actions: List[str],
        confidences: List[float],
        description: str = "",
        precondition: Optional[str] = None,
        postcondition: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Skill]:
        """Create a skill from a successful trajectory.

        Args:
            name: Skill name
            actions: List of actions taken
            confidences: Confidence for each action
            description: Skill description
            precondition: What must be true before skill
            postcondition: What will be true after skill
            tags: Skill tags for filtering

        Returns:
            Created skill or None if failed
        """
        if len(actions) != len(confidences):
            logger.error("Actions and confidences length mismatch")
            return None

        steps = [
            SkillStep(
                action=action,
                confidence=conf,
                reasoning=f"Step {i+1}/{len(actions)}"
            )
            for i, (action, conf) in enumerate(zip(actions, confidences))
        ]

        skill = Skill(
            name=name,
            description=description or f"Skill: {name}",
            steps=steps,
            precondition=precondition,
            postcondition=postcondition,
            tags=tags or [],
        )

        return skill if self.save_skill(skill) else None

    def update_skill_success(self, name: str, succeeded: bool) -> bool:
        """Update skill success statistics.

        Args:
            name: Skill name
            succeeded: Whether the skill execution succeeded

        Returns:
            True if updated
        """
        skill = self.load_skill(name)
        if not skill:
            return False

        skill.times_used += 1
        skill.last_used_at = datetime.now().isoformat()

        if succeeded:
            # Update success rate with exponential moving average
            skill.success_rate = 0.9 * skill.success_rate + 0.1 * 1.0
        else:
            skill.success_rate = 0.9 * skill.success_rate + 0.1 * 0.0

        return self.save_skill(skill)

    def find_similar_skills(
        self,
        precondition: str,
        postcondition: Optional[str] = None,
        max_results: int = 5
    ) -> List[Skill]:
        """Find skills matching precondition/postcondition.

        Args:
            precondition: Precondition to match
            postcondition: Optional postcondition to match
            max_results: Max skills to return

        Returns:
            Matching skills sorted by success rate
        """
        candidates = [
            s for s in self.skills.values()
            if s.precondition == precondition
        ]

        if postcondition:
            candidates = [
                s for s in candidates
                if s.postcondition == postcondition
            ]

        # Sort by success rate (highest first), then by times used
        candidates.sort(
            key=lambda s: (s.success_rate, s.times_used),
            reverse=True
        )

        return candidates[:max_results]
