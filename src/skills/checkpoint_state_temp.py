"""State management for skill execution checkpoints.

This module provides infrastructure for creating, validating, and serializing
checkpoints during skill execution. Checkpoints capture the execution state
at specific points in a skill, allowing recovery and resumption from those points.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Captures the execution state at a checkpoint.

    A checkpoint stores:
    - The execution context (variables, state snapshots)
    - Metadata about when it was created and what skill created it
    - Notes about what was accomplished before the checkpoint
    - Frame captures for debugging and visualization

    Checkpoints can be serialized to JSON and restored later to resume
    execution from the checkpoint point.
    """

    checkpoint_id: str
    """Unique identifier for this checkpoint (e.g., 'explore_ready', 'before_boss')."""

    timestamp: float
    """Unix timestamp when checkpoint was created."""

    skill_name: str
    """Name of the skill that created this checkpoint."""

    execution_context: Dict[str, Any] = field(default_factory=dict)
    """Serialized execution context at checkpoint time.

    This includes:
    - state: Dict of semantic game state (HP, items, dungeon floor, etc.)
    - snapshots: List of state snapshots captured during execution
    - params: Original parameters passed to the skill
    """

    parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters passed to the skill that created this checkpoint."""

    notes: List[str] = field(default_factory=list)
    """Execution notes and annotations up to this checkpoint."""

    frames_captured: int = 0
    """Number of frames (screenshots) captured before this checkpoint."""

    description: Optional[str] = None
    """Optional human-readable description of what this checkpoint represents."""

    visual_metadata: Dict[str, Any] = field(default_factory=dict)
    """Visual metadata extracted from screenshots.

    This includes:
    - screenshot_path: Path to the checkpoint screenshot
    - floor_number: Extracted dungeon floor number
    - player_position: Extracted player (x, y) coordinates
    - visible_enemies: List of visible enemy names and positions
    - visible_items: List of visible item names and positions
    - player_hp: Extracted player HP value
    - player_status: Extracted status effects
    - menu_state: Current menu/UI state
    """

    screenshot_analysis: Dict[str, Any] = field(default_factory=dict)
    """Detailed screenshot analysis results.

    This includes:
    - ocr_text: Raw OCR text extracted from screenshot
    - detected_regions: Bounding boxes of detected UI elements
    - confidence_scores: Confidence scores for extracted data
    - analysis_timestamp: When the analysis was performed
    - analyzer_version: Version of the screenshot analyzer used
    """

    save_slot: Optional[int] = None
    """SaveManager slot number if this checkpoint has a saved game state."""

    parent_checkpoint_id: Optional[str] = None
    """ID of the parent checkpoint this was derived from (for trajectory tracking)."""

    trajectory_metadata: Dict[str, Any] = field(default_factory=dict)
    """Trajectory-specific metadata for organizing checkpoints.

    This includes:
    - floor: Dungeon floor number for this checkpoint
    - branch_id: Branch identifier for alternative execution paths
    - depth: Depth in the checkpoint tree
    - outcome: Result of executing from this checkpoint (success/failure/pending)
    - children: List of child checkpoint IDs
    """

    """Optional human-readable description of what this checkpoint represents."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint state to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.

        Raises:
            ValueError: If any field contains non-serializable data.
        """
        try:
            data = {
                "checkpoint_id": self.checkpoint_id,
                "timestamp": self.timestamp,
                "skill_name": self.skill_name,
                "execution_context": self.execution_context,
                "parameters": self.parameters,
                "notes": self.notes,
                "frames_captured": self.frames_captured,
                "description": self.description,
            }

            # Validate JSON serializability
            json.dumps(data)
            return data
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to serialize checkpoint {self.checkpoint_id}: {e}"
            ) from e

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointState:
        """Deserialize checkpoint state from a dictionary.

        Args:
            data: Dictionary containing checkpoint data (typically from JSON).

        Returns:
            CheckpointState instance.

        Raises:
            ValueError: If required fields are missing or have wrong types.
            KeyError: If critical fields are absent.
        """
        try:
            # Validate required fields exist
            required_fields = {"checkpoint_id", "timestamp", "skill_name"}
            missing = required_fields - set(data.keys())
            if missing:
                raise KeyError(f"Missing required fields: {missing}")

            # Validate and extract required fields with type checking
            checkpoint_id = data["checkpoint_id"]
            if not isinstance(checkpoint_id, str):
                raise ValueError(
                    f"checkpoint_id must be string, got {type(checkpoint_id).__name__}"
                )

            timestamp = data["timestamp"]
            if not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"timestamp must be numeric, got {type(timestamp).__name__}"
                )

            skill_name = data["skill_name"]
            if not isinstance(skill_name, str):
                raise ValueError(
                    f"skill_name must be string, got {type(skill_name).__name__}"
                )

            # Validate and extract optional fields with type checking
            execution_context = data.get("execution_context", {})
            if not isinstance(execution_context, dict):
                raise ValueError(
                    f"execution_context must be dict, got {type(execution_context).__name__}"
                )

            parameters = data.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"parameters must be dict, got {type(parameters).__name__}"
                )

            notes = data.get("notes", [])
            if not isinstance(notes, list):
                raise ValueError(
                    f"notes must be list, got {type(notes).__name__}"
                )

            frames_captured = data.get("frames_captured", 0)
            if not isinstance(frames_captured, int):
                raise ValueError(
                    f"frames_captured must be int, got {type(frames_captured).__name__}"
                )

            description = data.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f"description must be string or None, got {type(description).__name__}"
                )

            return cls(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                skill_name=skill_name,
                execution_context=execution_context,
                parameters=parameters,
                notes=notes,
                frames_captured=frames_captured,
                description=description,
            )

        except KeyError as e:
            raise ValueError(
                f"Missing required field in checkpoint data: {e}"
            ) from e
        except ValueError:
            raise

    def validate(self) -> List[str]:
        """Validate checkpoint state integrity.

        Returns:
            List of validation error messages. Empty list means checkpoint is valid.
        """
        errors: List[str] = []

        # Validate checkpoint_id
        if not self.checkpoint_id or not isinstance(self.checkpoint_id, str):
            errors.append("checkpoint_id must be a non-empty string")
        elif len(self.checkpoint_id) > 64:
            errors.append("checkpoint_id must not exceed 64 characters")
        elif not self.checkpoint_id.replace("_", "").replace("-", "").isalnum():
            errors.append(
                "checkpoint_id must contain only alphanumeric, underscore, or hyphen characters"
            )

        # Validate timestamp
        if not isinstance(self.timestamp, (int, float)):
            errors.append("timestamp must be a numeric Unix timestamp")
        elif self.timestamp < 0:
            errors.append("timestamp must be non-negative")

        # Validate skill_name
        if not self.skill_name or not isinstance(self.skill_name, str):
            errors.append("skill_name must be a non-empty string")
        elif len(self.skill_name) > 128:
            errors.append("skill_name must not exceed 128 characters")

        # Validate execution_context
        if not isinstance(self.execution_context, dict):
            errors.append("execution_context must be a dictionary")

        # Validate parameters
        if not isinstance(self.parameters, dict):
            errors.append("parameters must be a dictionary")

        # Validate notes
        if not isinstance(self.notes, list):
            errors.append("notes must be a list")
        else:
            for i, note in enumerate(self.notes):
                if not isinstance(note, str):
                    errors.append(f"notes[{i}] must be a string")

        # Validate frames_captured
        if not isinstance(self.frames_captured, int):
            errors.append("frames_captured must be an integer")
        elif self.frames_captured < 0:
            errors.append("frames_captured must be non-negative")

        # Validate description if present
        if self.description is not None:
            if not isinstance(self.description, str):
                errors.append("description must be a string or None")
            elif len(self.description) > 500:
                errors.append("description must not exceed 500 characters")

        return errors

    def is_valid(self) -> bool:
        """Check if checkpoint state is valid.

        Returns:
            True if checkpoint passes all validation checks, False otherwise.
        """
        return len(self.validate()) == 0

    def __repr__(self) -> str:
        """Return a detailed string representation of the checkpoint."""
        timestamp_str = datetime.fromtimestamp(self.timestamp).isoformat()
        return (
            f"CheckpointState("
            f"id={self.checkpoint_id!r}, "
            f"skill={self.skill_name!r}, "
            f"time={timestamp_str}, "
            f"frames={self.frames_captured})"
        )
