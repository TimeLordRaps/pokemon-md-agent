"""Cross-run state bootstrap system for memory and skill continuity.

Enables the agent to save and load learned behaviors and memory
across multiple game runs, with support for fresh ROM restarts
and checkpoint-based recovery.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BootstrapCheckpoint:
    """Checkpoint for saving agent state between runs."""

    run_id: str  # Unique run identifier
    timestamp: str  # ISO timestamp when checkpoint was created

    # State snapshots
    learned_skills: List[Dict[str, Any]] = field(default_factory=list)
    memory_buffer: Dict[str, Any] = field(default_factory=dict)
    trajectory_embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # Game state metadata
    last_known_position: Optional[Dict[str, int]] = None
    last_known_hp: Optional[int] = None
    dungeon_level: Optional[int] = None

    # Bootstrap flags
    is_bootstrap: bool = False  # Whether this was loaded from a checkpoint
    parent_run_id: Optional[str] = None  # Which run this bootstrapped from

    # Statistics
    total_steps: int = 0
    success_rate: float = 0.0
    skills_discovered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(**data)


class BootstrapStateManager:
    """Manage bootstrap state persistence and retrieval."""

    def __init__(
        self,
        bootstrap_dir: Path = Path("config/bootstrap"),
        enable_bootstrap: bool = True
    ):
        """Initialize bootstrap state manager.

        Args:
            bootstrap_dir: Directory to store bootstrap checkpoints
            enable_bootstrap: Whether to enable state bootstrapping
        """
        self.bootstrap_dir = Path(bootstrap_dir)
        self.bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self.enable_bootstrap = enable_bootstrap

        self.current_checkpoint: Optional[BootstrapCheckpoint] = None
        self.checkpoint_history: List[BootstrapCheckpoint] = []

        self._load_checkpoint_history()

    def _load_checkpoint_history(self) -> None:
        """Load list of all previous checkpoints."""
        if not self.bootstrap_dir.exists():
            return

        for checkpoint_file in sorted(self.bootstrap_dir.glob("*.json")):
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)
                    checkpoint = BootstrapCheckpoint.from_dict(data)
                    self.checkpoint_history.append(checkpoint)
                    logger.debug(f"Loaded checkpoint history: {checkpoint.run_id}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")

    def _get_checkpoint_path(self, run_id: str) -> Path:
        """Get file path for checkpoint."""
        # Sanitize run_id for filename
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_id)
        return self.bootstrap_dir / f"{safe_id}.json"

    def create_checkpoint(self, run_id: str) -> BootstrapCheckpoint:
        """Create a new checkpoint for a run.

        Args:
            run_id: Unique identifier for this run

        Returns:
            New BootstrapCheckpoint instance
        """
        checkpoint = BootstrapCheckpoint(
            run_id=run_id,
            timestamp=datetime.now().isoformat()
        )
        self.current_checkpoint = checkpoint
        return checkpoint

    def save_checkpoint(
        self,
        run_id: str,
        learned_skills: List[Dict[str, Any]],
        memory_buffer: Dict[str, Any],
        trajectory_embeddings: Optional[Dict[str, List[float]]] = None,
        game_state: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save checkpoint with agent state.

        Args:
            run_id: Run identifier
            learned_skills: List of discovered skills
            memory_buffer: Memory buffer state
            trajectory_embeddings: Learned trajectory embeddings
            game_state: Current game state metadata
            stats: Run statistics

        Returns:
            True if successful
        """
        if not self.enable_bootstrap:
            return False

        try:
            checkpoint = BootstrapCheckpoint(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                learned_skills=learned_skills,
                memory_buffer=memory_buffer,
                trajectory_embeddings=trajectory_embeddings or {},
            )

            # Add game state if provided
            if game_state:
                checkpoint.last_known_position = game_state.get("position")
                checkpoint.last_known_hp = game_state.get("hp")
                checkpoint.dungeon_level = game_state.get("dungeon_level")

            # Add statistics if provided
            if stats:
                checkpoint.total_steps = stats.get("total_steps", 0)
                checkpoint.success_rate = stats.get("success_rate", 0.0)
                checkpoint.skills_discovered = stats.get("skills_discovered", 0)

            # Save to disk
            checkpoint_path = self._get_checkpoint_path(run_id)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            self.current_checkpoint = checkpoint
            self.checkpoint_history.append(checkpoint)
            logger.info(f"Saved checkpoint: {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {run_id}: {e}")
            return False

    def load_latest_checkpoint(self) -> Optional[BootstrapCheckpoint]:
        """Load the most recent checkpoint.

        Returns:
            Latest checkpoint or None if none exist
        """
        if not self.checkpoint_history:
            return None

        checkpoint = self.checkpoint_history[-1]
        self.current_checkpoint = checkpoint
        return checkpoint

    def load_checkpoint(self, run_id: str) -> Optional[BootstrapCheckpoint]:
        """Load a specific checkpoint.

        Args:
            run_id: Run identifier

        Returns:
            Checkpoint if found, None otherwise
        """
        checkpoint_path = self._get_checkpoint_path(run_id)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {run_id}")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                checkpoint = BootstrapCheckpoint.from_dict(data)
                self.current_checkpoint = checkpoint
                return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {run_id}: {e}")
            return None

    def list_checkpoints(
        self,
        limit: Optional[int] = None
    ) -> List[BootstrapCheckpoint]:
        """List available checkpoints.

        Args:
            limit: Maximum number to return (most recent first)

        Returns:
            List of checkpoints sorted by recency
        """
        checkpoints = sorted(
            self.checkpoint_history,
            key=lambda c: c.timestamp,
            reverse=True
        )

        if limit:
            return checkpoints[:limit]
        return checkpoints

    def get_bootstrap_status(self) -> Dict[str, Any]:
        """Get current bootstrap status.

        Returns:
            Status information
        """
        return {
            "enabled": self.enable_bootstrap,
            "current_checkpoint": self.current_checkpoint.run_id if self.current_checkpoint else None,
            "total_checkpoints": len(self.checkpoint_history),
            "recent_checkpoints": [
                {
                    "run_id": c.run_id,
                    "timestamp": c.timestamp,
                    "skills": len(c.learned_skills),
                    "steps": c.total_steps
                }
                for c in self.list_checkpoints(limit=5)
            ]
        }

    def create_bootstrap_run(
        self,
        new_run_id: str,
        parent_run_id: str
    ) -> BootstrapCheckpoint:
        """Create a new run that bootstraps from a previous one.

        Args:
            new_run_id: ID for the new run
            parent_run_id: ID of run to bootstrap from

        Returns:
            New checkpoint with parent data
        """
        parent = self.load_checkpoint(parent_run_id)
        if not parent:
            logger.warning(f"Parent checkpoint not found: {parent_run_id}")
            return self.create_checkpoint(new_run_id)

        # Create new checkpoint with parent's learned data
        checkpoint = BootstrapCheckpoint(
            run_id=new_run_id,
            timestamp=datetime.now().isoformat(),
            learned_skills=parent.learned_skills.copy(),
            memory_buffer=parent.memory_buffer.copy(),
            trajectory_embeddings=parent.trajectory_embeddings.copy(),
            last_known_position=parent.last_known_position,
            last_known_hp=parent.last_known_hp,
            dungeon_level=parent.dungeon_level,
            is_bootstrap=True,
            parent_run_id=parent_run_id,
        )

        self.current_checkpoint = checkpoint
        logger.info(f"Created bootstrap run {new_run_id} from {parent_run_id}")
        return checkpoint

    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """Remove old checkpoints, keeping only the most recent.

        Args:
            keep_count: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        if not self.enable_bootstrap:
            return 0

        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda c: c.timestamp,
            reverse=True
        )

        to_delete = sorted_checkpoints[keep_count:]
        deleted_count = 0

        for checkpoint in to_delete:
            try:
                checkpoint_path = self._get_checkpoint_path(checkpoint.run_id)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old checkpoint: {checkpoint.run_id}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint.run_id}: {e}")

        # Update history
        self.checkpoint_history = sorted_checkpoints[:keep_count]

        return deleted_count
