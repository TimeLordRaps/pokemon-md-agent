"""Sprite detection using Qwen3-VL for Pokemon MD autonomous gameplay."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SpriteType(Enum):
    """Types of sprites that can be detected."""
    PLAYER = "player"
    ENEMY_POKEMON = "enemy_pokemon"
    ALLY_POKEMON = "ally_pokemon"
    ITEM = "item"
    WALL = "wall"
    DOOR = "door"
    STAIRS = "stairs"
    TRAP = "trap"
    BERRIES = "berries"
    RECOVERY = "recovery"
    ORB = "orb"
    FOOD = "food"
    TREASURE = "treasure"


@dataclass
class SpriteDetection:
    """Individual sprite detection result."""
    sprite_type: SpriteType
    position: Tuple[int, int]  # x, y coordinates
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    species: Optional[str] = None  # Pokemon species if applicable
    metadata: Optional[Dict[str, Any]] = None


class SpriteDetector:
    """Qwen3-VL based sprite detection for Pokemon MD game state."""
    
    def __init__(self, model_name: str = "Qwen3-VL-4B-Instruct"):
        """Initialize sprite detector.
        
        Args:
            model_name: Qwen3-VL model to use for detection
        """
        self.model_name = model_name
        self.model = None  # Will be initialized when model is loaded
        
        # Pokemon species mapping for recognition
        self.pokemon_species = {
            # Starter Pokemon and common ones (placeholder)
            "charmander": SpriteType.ALLY_POKEMON,
            "squirtle": SpriteType.ALLY_POKEMON,
            "bulbasaur": SpriteType.ALLY_POKEMON,
            "pikachu": SpriteType.ALLY_POKEMON,
            "meowth": SpriteType.ENEMY_POKEMON,
            "rattata": SpriteType.ENEMY_POKEMON,
            "spearow": SpriteType.ENEMY_POKEMON,
        }
        
        # Item type mappings
        self.item_types = {
            "berry": SpriteType.ITEM,
            "food": SpriteType.FOOD,
            "treasure": SpriteType.TREASURE,
            "orb": SpriteType.ORB,
        }
        
        logger.info("Initialized SpriteDetector with model: %s", model_name)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load Qwen3-VL model for sprite detection.
        
        Args:
            model_path: Path to model (auto-download if None)
        """
        logger.info("Loading sprite detection model: %s", self.model_name)
        
        # TODO: Implement actual model loading
        # This will involve:
        # 1. Loading Qwen3-VL model
        # 2. Setting up for vision inference
        # 3. Configuring for sprite detection tasks
        
        logger.info("Sprite detection model loading complete (placeholder implementation)")
    
    def detect_sprites(
        self,
        screenshot: Any,
        detection_threshold: float = 0.7,
    ) -> List[SpriteDetection]:
        """Detect sprites in screenshot.
        
        Args:
            screenshot: Screenshot image data
            detection_threshold: Minimum confidence threshold for detections
            
        Returns:
            List of SpriteDetection objects
        """
        logger.debug("Detecting sprites with threshold: %.2f", detection_threshold)
        
        # TODO: Implement actual sprite detection
        # This will involve:
        # 1. Preprocessing screenshot
        # 2. Running Qwen3-VL inference with sprite detection prompt
        # 3. Parsing response to extract sprite positions and types
        # 4. Returning structured detection results
        
        # Placeholder implementation
        return self._generate_dummy_detections(screenshot, detection_threshold)
    
    def detect_player(self, detections: List[SpriteDetection]) -> Optional[SpriteDetection]:
        """Find player character in sprite detections.
        
        Args:
            detections: List of sprite detections
            
        Returns:
            Player detection or None if not found
        """
        for detection in detections:
            if detection.sprite_type == SpriteType.PLAYER:
                return detection
        
        logger.warning("Player not found in detections")
        return None
    
    def detect_enemies(self, detections: List[SpriteDetection]) -> List[SpriteDetection]:
        """Find enemy Pokemon in sprite detections.
        
        Args:
            detections: List of sprite detections
            
        Returns:
            List of enemy Pokemon detections
        """
        enemies = []
        
        for detection in detections:
            if detection.sprite_type == SpriteType.ENEMY_POKEMON:
                enemies.append(detection)
        
        return enemies
    
    def detect_items(self, detections: List[SpriteDetection]) -> List[SpriteDetection]:
        """Find items in sprite detections.
        
        Args:
            detections: List of sprite detections
            
        Returns:
            List of item detections
        """
        items = []
        
        item_types = {
            SpriteType.ITEM,
            SpriteType.BERRIES,
            SpriteType.RECOVERY,
            SpriteType.ORB,
            SpriteType.FOOD,
            SpriteType.TREASURE,
        }
        
        for detection in detections:
            if detection.sprite_type in item_types:
                items.append(detection)
        
        return items
    
    def detect_exits(self, detections: List[SpriteDetection]) -> List[SpriteDetection]:
        """Find exits (stairs, doors) in sprite detections.
        
        Args:
            detections: List of sprite detections
            
        Returns:
            List of exit detections
        """
        exits = []
        
        exit_types = {SpriteType.STAIRS, SpriteType.DOOR}
        
        for detection in detections:
            if detection.sprite_type in exit_types:
                exits.append(detection)
        
        return exits
    
    def get_spatial_layout(self, detections: List[SpriteDetection]) -> Dict[str, Any]:
        """Get spatial layout of detected sprites.
        
        Args:
            detections: List of sprite detections
            
        Returns:
            Dictionary with spatial layout information
        """
        if not detections:
            return {"player_position": None, "enemies": [], "items": [], "exits": []}
        
        player_pos = None
        enemies = []
        items = []
        exits = []
        walls = []
        
        for detection in detections:
            pos = detection.position
            
            if detection.sprite_type == SpriteType.PLAYER:
                player_pos = pos
            
            elif detection.sprite_type == SpriteType.ENEMY_POKEMON:
                enemies.append({
                    "position": pos,
                    "confidence": detection.confidence,
                    "species": detection.species,
                })
            
            elif detection.sprite_type in {
                SpriteType.ITEM, SpriteType.BERRIES, SpriteType.RECOVERY,
                SpriteType.ORB, SpriteType.FOOD, SpriteType.TREASURE
            }:
                items.append({
                    "position": pos,
                    "confidence": detection.confidence,
                    "type": detection.sprite_type.value,
                })
            
            elif detection.sprite_type in {SpriteType.STAIRS, SpriteType.DOOR}:
                exits.append({
                    "position": pos,
                    "confidence": detection.confidence,
                    "type": detection.sprite_type.value,
                })
            
            elif detection.sprite_type == SpriteType.WALL:
                walls.append({
                    "position": pos,
                    "confidence": detection.confidence,
                })
        
        return {
            "player_position": player_pos,
            "enemies": enemies,
            "items": items,
            "exits": exits,
            "walls": walls,
            "total_detections": len(detections),
        }
    
    def calculate_distances(
        self,
        from_pos: Tuple[int, int],
        to_positions: List[Tuple[int, int]],
    ) -> List[float]:
        """Calculate distances from one position to multiple positions.
        
        Args:
            from_pos: Starting position (x, y)
            to_positions: List of target positions
            
        Returns:
            List of distances
        """
        distances = []
        
        for to_pos in to_positions:
            distance = np.sqrt(
                (from_pos[0] - to_pos[0]) ** 2 +
                (from_pos[1] - to_pos[1]) ** 2
            )
            distances.append(distance)
        
        return distances
    
    def find_nearest_enemy(
        self,
        player_pos: Tuple[int, int],
        enemy_detections: List[SpriteDetection],
    ) -> Optional[SpriteDetection]:
        """Find nearest enemy to player position.
        
        Args:
            player_pos: Player position
            enemy_detections: List of enemy detections
            
        Returns:
            Nearest enemy detection or None
        """
        if not enemy_detections:
            return None
        
        enemy_positions = [enemy.position for enemy in enemy_detections]
        distances = self.calculate_distances(player_pos, enemy_positions)
        
        min_distance_idx = np.argmin(distances)
        return enemy_detections[min_distance_idx]
    
    def find_nearest_item(
        self,
        player_pos: Tuple[int, int],
        item_detections: List[SpriteDetection],
    ) -> Optional[SpriteDetection]:
        """Find nearest item to player position.
        
        Args:
            player_pos: Player position
            item_detections: List of item detections
            
        Returns:
            Nearest item detection or None
        """
        if not item_detections:
            return None
        
        item_positions = [item.position for item in item_detections]
        distances = self.calculate_distances(player_pos, item_positions)
        
        min_distance_idx = np.argmin(distances)
        return item_detections[min_distance_idx]
    
    def _generate_dummy_detections(
        self,
        screenshot: Any,
        threshold: float,
    ) -> List[SpriteDetection]:
        """Generate dummy sprite detections for testing.
        
        Args:
            screenshot: Screenshot data
            threshold: Detection threshold
            
        Returns:
            List of dummy detections
        """
        # Generate some deterministic "random" positions
        np.random.seed(42)  # For consistent dummy data
        
        detections = []
        
        # Player
        detections.append(SpriteDetection(
            sprite_type=SpriteType.PLAYER,
            position=(100, 200),
            confidence=0.95,
            species="Charmander",
        ))
        
        # Some enemies
        detections.append(SpriteDetection(
            sprite_type=SpriteType.ENEMY_POKEMON,
            position=(150, 180),
            confidence=0.88,
            species="Meowth",
        ))
        
        detections.append(SpriteDetection(
            sprite_type=SpriteType.ENEMY_POKEMON,
            position=(200, 250),
            confidence=0.82,
            species="Rattata",
        ))
        
        # Some items
        detections.append(SpriteDetection(
            sprite_type=SpriteType.BERRIES,
            position=(80, 180),
            confidence=0.90,
        ))
        
        detections.append(SpriteDetection(
            sprite_type=SpriteType.TREASURE,
            position=(300, 150),
            confidence=0.75,
        ))
        
        # Exit
        detections.append(SpriteDetection(
            sprite_type=SpriteType.STAIRS,
            position=(400, 200),
            confidence=0.92,
        ))
        
        # Filter by threshold
        filtered_detections = [
            det for det in detections
            if det.confidence >= threshold
        ]
        
        logger.debug("Generated %d dummy detections", len(filtered_detections))
        return filtered_detections
    
    def get_detection_stats(self, detections: List[SpriteDetection]) -> Dict[str, int]:
        """Get statistics about sprite detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection counts by type
        """
        stats = {}
        
        for detection in detections:
            sprite_type = detection.sprite_type.value
            stats[sprite_type] = stats.get(sprite_type, 0) + 1
        
        stats["total"] = len(detections)
        return stats
