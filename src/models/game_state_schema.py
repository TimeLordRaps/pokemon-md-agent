"""Game state schema for vision model outputs.

Provides Pydantic models for structured extraction of Pokemon Mystery Dungeon
game state from screenshots. Used for prompt guidance and output validation.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GameStateEnum(str, Enum):
    """Possible game states."""
    EXPLORING = "exploring"
    BATTLE = "battle"
    MENU = "menu"
    STAIRS = "stairs_found"
    BOSS = "boss_battle"
    UNKNOWN = "unknown"


class RoomType(str, Enum):
    """Dungeon room types."""
    CORRIDOR = "corridor"
    CHAMBER = "chamber"
    BOSS = "boss_room"
    SHOP = "shop"
    TREASURE = "treasure_room"
    UNKNOWN = "unknown"


class Entity(BaseModel):
    """Represents a visible game entity (player, enemy, item, object)."""

    x: int = Field(..., ge=0, description="X coordinate (0-indexed)")
    y: int = Field(..., ge=0, description="Y coordinate (0-indexed)")
    type: str = Field(..., description="Entity type: player|enemy|item|door|stairs|trap")
    species: Optional[str] = Field(None, description="Pokemon species (for enemies/allies)")
    name: Optional[str] = Field(None, description="Item name (for items)")
    status_effects: List[str] = Field(
        default_factory=list,
        description="Status effects: poison|burn|sleep|paralysis|confuse"
    )
    hp: Optional[int] = Field(None, ge=0, description="HP if visible")
    level: Optional[int] = Field(None, ge=1, description="Level if visible")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        """Validate entity type is recognized."""
        valid_types = {"player", "enemy", "ally", "item", "door", "stairs", "trap"}
        if v.lower() not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {v}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "x": 10,
                "y": 8,
                "type": "enemy",
                "species": "Geodude",
                "name": None,
                "status_effects": [],
                "hp": 15,
                "level": 5
            }
        }


class GameState(BaseModel):
    """Complete game state observation from a single screenshot."""

    # === CORE POSITIONING ===
    player_pos: tuple[int, int] = Field(..., description="Player position [x, y]")
    player_hp: Optional[int] = Field(None, ge=0, description="Player HP if visible")
    player_max_hp: Optional[int] = Field(None, ge=1, description="Player max HP if visible")
    player_status: List[str] = Field(
        default_factory=list,
        description="Player status effects: poison|burn|sleep|paralysis"
    )

    # === ENVIRONMENT ===
    floor: int = Field(..., ge=1, description="Current dungeon floor (1-indexed)")
    dungeon_name: Optional[str] = Field(None, description="Dungeon name if visible")
    room_type: RoomType = Field(
        default=RoomType.CORRIDOR,
        description="Room type classification"
    )

    # === ENTITIES ===
    enemies: List[Entity] = Field(
        default_factory=list,
        description="Visible hostile enemies"
    )
    allies: List[Entity] = Field(
        default_factory=list,
        description="Visible allied Pokemon"
    )
    items: List[Entity] = Field(
        default_factory=list,
        description="Visible pickupable items"
    )
    special_objects: List[Entity] = Field(
        default_factory=list,
        description="Stairs, doors, traps, etc"
    )

    # === GAME STATE ===
    state: GameStateEnum = Field(..., description="Overall game state")
    is_day: bool = Field(default=True, description="Day vs night mode")
    weather: Optional[str] = Field(None, description="Weather effect if any")

    # === CONTEXT & CHANGES ===
    significant_change: str = Field(
        default="",
        description="What changed from previous state (5-50 words)"
    )
    threats: List[str] = Field(
        default_factory=list,
        description="Immediate threats/dangers (up to 3 items)"
    )
    opportunities: List[str] = Field(
        default_factory=list,
        description="Available actions/opportunities (up to 3 items)"
    )

    # === CONFIDENCE & METADATA ===
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Model confidence in this extraction (0-1)"
    )
    notes: str = Field(
        default="",
        description="Additional observations or caveats"
    )

    @field_validator("floor")
    @classmethod
    def validate_floor(cls, v):
        """Validate floor is reasonable."""
        if v < 1 or v > 50:  # PMD floors typically 1-50
            raise ValueError(f"floor must be 1-50, got {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Confidence must be valid probability."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {v}")
        return v

    @field_validator("threats", "opportunities")
    @classmethod
    def validate_threat_limit(cls, v):
        """Limit threats and opportunities to 3 items each."""
        if len(v) > 3:
            return v[:3]
        return v

    def to_prompt_json(self) -> str:
        """Export schema as JSON for LM prompt guidance."""
        return self.model_dump_json(
            include={"state", "player_pos", "floor", "enemies", "items"},
            indent=2
        )

    class Config:
        json_schema_extra = {
            "example": {
                "player_pos": [12, 8],
                "player_hp": 45,
                "player_max_hp": 60,
                "player_status": ["poison"],
                "floor": 3,
                "dungeon_name": "Mt. Horn",
                "room_type": "corridor",
                "enemies": [
                    {
                        "x": 14,
                        "y": 8,
                        "type": "enemy",
                        "species": "Geodude",
                        "name": None,
                        "status_effects": [],
                        "hp": 20,
                        "level": 5
                    }
                ],
                "allies": [],
                "items": [
                    {
                        "x": 10,
                        "y": 6,
                        "type": "item",
                        "species": None,
                        "name": "Apple",
                        "status_effects": [],
                        "hp": None,
                        "level": None
                    }
                ],
                "special_objects": [],
                "state": "exploring",
                "is_day": True,
                "weather": None,
                "significant_change": "Geodude moved 2 tiles closer (was at 16,8)",
                "threats": ["Geodude 2 tiles away, closing in"],
                "opportunities": ["Move up to dodge", "Use ranged attack"],
                "confidence": 0.92,
                "notes": "Enemy movement pattern suggests patrolling behavior"
            }
        }
