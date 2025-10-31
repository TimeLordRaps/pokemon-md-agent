"""RAG schema definitions for Pokemon MD agent memory and retrieval."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEntry:
    """A single trajectory entry in the RAG system."""
    id: str
    timestamp: float
    floor: int
    silo: str  # Episode/silo identifier
    emb_vector: List[float]  # Embedding vector for ANN search
    screenshot_path: Optional[str] = None
    sprite_map: Dict[str, Any] = field(default_factory=dict)  # Sprite detection results
    notes: str = ""  # Human-readable description

    # Additional metadata
    action_taken: Optional[str] = None
    confidence: Optional[float] = None
    outcome: Optional[str] = None
    reward: Optional[float] = None

    @property
    def composite_index(self) -> Tuple[int, str, float]:
        """Composite index (floor, silo, ts) for efficient retrieval."""
        return (self.floor, self.silo, self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "floor": self.floor,
            "silo": self.silo,
            "emb_vector": self.emb_vector,
            "screenshot_path": self.screenshot_path,
            "sprite_map": self.sprite_map,
            "notes": self.notes,
            "action_taken": self.action_taken,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryEntry':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            floor=data["floor"],
            silo=data["silo"],
            emb_vector=data["emb_vector"],
            screenshot_path=data.get("screenshot_path"),
            sprite_map=data.get("sprite_map", {}),
            notes=data.get("notes", ""),
            action_taken=data.get("action_taken"),
            confidence=data.get("confidence"),
            outcome=data.get("outcome"),
            reward=data.get("reward"),
        )


@dataclass
class EmbeddingEntry:
    """An embedding entry for different types of content."""
    id: str
    content_type: str  # "input", "think_step", "instruct_response", etc.
    content: str
    emb_vector: List[float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content_type": self.content_type,
            "content": self.content,
            "emb_vector": self.emb_vector,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingEntry':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content_type=data["content_type"],
            content=data["content"],
            emb_vector=data["emb_vector"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    entry: TrajectoryEntry
    score: float
    rank: int
    source: str  # "ann", "rrf", "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry": self.entry.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
        }


@dataclass
class QueryContext:
    """Context for a retrieval query."""
    query_text: str
    query_embedding: List[float]
    current_floor: Optional[int] = None
    current_silo: Optional[str] = None
    max_results: int = 10
    recency_bias: float = 0.1  # How much to weight recent entries
    dedup_by_episode: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_text": self.query_text,
            "query_embedding": self.query_embedding,
            "current_floor": self.current_floor,
            "current_silo": self.current_silo,
            "max_results": self.max_results,
            "recency_bias": self.recency_bias,
            "dedup_by_episode": self.dedup_by_episode,
        }