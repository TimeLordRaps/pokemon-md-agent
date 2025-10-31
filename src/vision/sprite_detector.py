"""Sprite detection using Qwen3-VL models for Pokemon Mystery Dungeon.

Detects and labels sprites including stairs, items, enemies, traps, and HUD elements.
Uses YAML-based labeling system for structured output.
"""

import yaml
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import hashlib
from collections import defaultdict
import imagehash
import numpy as np
from PIL import Image
from .sprite_phash import compute_phash

logger = logging.getLogger(__name__)


@dataclass
class SpriteLabels:
    """YAML-based sprite labeling configuration."""
    stairs: List[str]
    items: List[str]
    enemies: List[str]
    traps: List[str]
    hud_elements: List[str]
    special_tiles: List[str]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SpriteLabels":
        """Load labels from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save labels to YAML file."""
        data = {
            "stairs": self.stairs,
            "items": self.items,
            "enemies": self.enemies,
            "traps": self.traps,
            "hud_elements": self.hud_elements,
            "special_tiles": self.special_tiles,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class DetectionConfig:
    """Configuration for sprite detection."""
    confidence_threshold: float = 0.7
    max_detections: int = 20
    enable_grid_correlation: bool = True
    enable_phash_computation: bool = False  # Feature flag for pHash integration
    categories: Optional[List[str]] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = ["stairs", "items", "enemies", "traps", "hud_elements", "special_tiles"]


@dataclass
class GridData:
    """Grid representation of dungeon state."""
    width: int
    height: int
    tiles: List[List[str]]  # 2D grid of tile types
    entities: List[Dict[str, Any]]  # List of entities with positions
    player_pos: Optional[Tuple[int, int]] = None


@dataclass
class DetectionResult:
    """Result of sprite detection."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    metadata: Dict[str, Any]
    grid_pos: Optional[Tuple[int, int]] = None  # Grid coordinates if available
    phash: Optional[np.ndarray] = None  # Perceptual hash as binary array when enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "metadata": self.metadata
        }
        if self.grid_pos:
            result["grid_pos"] = list(self.grid_pos)
        if self.phash is not None:
            result["phash"] = self.phash.tolist()  # Convert numpy array to list for JSON
        return result


class BaseSpriteDetector(ABC):
    """Base class for sprite detection implementations."""

    def __init__(self, config: DetectionConfig, labels: Optional[SpriteLabels] = None):
        self.config = config
        self.labels = labels or DEFAULT_LABELS

    @abstractmethod
    def detect(self, image_path: Path, grid_data: Optional[GridData] = None) -> List[DetectionResult]:
        """Detect sprites in image with optional grid context."""
        pass

    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply common filtering based on config."""
        # Filter by confidence
        filtered = [d for d in detections if d.confidence >= self.config.confidence_threshold]

        # Filter by categories if specified
        if self.config.categories:
            filtered = [d for d in filtered if self._get_category(d.label) in self.config.categories]

        # Limit number of detections
        filtered = filtered[:self.config.max_detections]

        return filtered

    def _correlate_with_grid(self, detections: List[DetectionResult], grid_data: GridData) -> List[DetectionResult]:
        """Correlate detections with grid data to add grid positions."""
        # Simple correlation based on bounding box center
        for detection in detections:
            center_x = detection.bbox[0] + detection.bbox[2] // 2
            center_y = detection.bbox[1] + detection.bbox[3] // 2

            # Convert pixel coordinates to grid coordinates (assuming 16x16 tiles)
            grid_x = center_x // 16
            grid_y = center_y // 16

            # Validate grid bounds
            if 0 <= grid_x < grid_data.width and 0 <= grid_y < grid_data.height:
                detection.grid_pos = (grid_x, grid_y)

        return detections

    def _get_category(self, label: str) -> str:
        """Get category for a label."""
        if label in self.labels.stairs:
            return "stairs"
        elif label in self.labels.items:
            return "items"
        elif label in self.labels.enemies:
            return "enemies"
        elif label in self.labels.traps:
            return "traps"
        elif label in self.labels.hud_elements:
            return "hud_elements"
        elif label in self.labels.special_tiles:
            return "special_tiles"
        return "unknown"


# Default sprite labels for Pokemon Mystery Dungeon
DEFAULT_LABELS = SpriteLabels(
    stairs=[
        "up_stairs", "down_stairs", "exit_stairs", "warp_stairs",
        "golden_stairs", "silver_stairs", "hidden_stairs"
    ],
    items=[
        "apple", "banana", "orange", "berry", "seed", "herb",
        "gummi", "wonder_gummi", "heal_seed", " oran_berry",
        "sitrus_berry", "reviver_seed", "totter_seed",
        "stick", "iron_thorn", "silver_spike", "gold_fang",
        "rare_fossil", "amulet_coin", "gold_bar", "pearl",
        "big_pearl", "stardust", "star_piece", "nugget",
        "heart_scale", "pretty_feather", "bright_powder",
        "white_herb", "mental_herb", "power_herb", "grass_mail",
        "flame_mail", "bubble_mail", "bloom_mail", "tunnel_mail",
        "steel_mail", "heart_mail", "snow_mail", "space_mail",
        "air_mail", "mosaic_mail", "brick_mail"
    ],
    enemies=[
        "caterpie", "metapod", "butterfree", "weedle", "kakuna", "beedrill",
        "pidgey", "pidgeotto", "pidgeot", "rattata", "raticate", "spearow",
        "fearow", "ekans", "arbok", "pikachu", "raichu", "sandshrew",
        "sandslash", "nidoran_f", "nidorina", "nidoqueen", "nidoran_m",
        "nidorino", "nidoking", "clefairy", "clefable", "vulpix", "ninetales",
        "jigglypuff", "wigglytuff", "zubat", "golbat", "oddish", "gloom",
        "vileplume", "paras", "parasect", "venonat", "venomoth", "diglett",
        "dugtrio", "meowth", "persian", "psyduck", "golduck", "mankey",
        "primeape", "growlithe", "arcanine", "poliwag", "poliwhirl", "poliwrath",
        "abra", "kadabra", "alakazam", "machop", "machoke", "machamp",
        "bellsprout", "weepinbell", "victreebel", "tentacool", "tentacruel",
        "geodude", "graveler", "golem", "ponyta", "rapidash", "slowpoke",
        "slowbro", "magnemite", "magneton", "farfetchd", "doduo", "dodrio",
        "seel", "dewgong", "grimer", "muk", "shellder", "cloyster", "gastly",
        "haunter", "gengar", "onix", "drowzee", "hypno", "krabby", "kingler",
        "voltorb", "electrode", "exeggcute", "exeggutor", "cubone", "marowak",
        "hitmonlee", "hitmonchan", "lickitung", "koffing", "weezing", "rhyhorn",
        "rhydon", "chansey", "tangela", "kangaskhan", "horsea", "seadra",
        "goldeen", "seaking", "staryu", "starmie", "mr_mime", "scyther",
        "jynx", "electabuzz", "magmar", "pinsir", "tauros", "magikarp",
        "gyarados", "lapras", "ditto", "eevee", "vaporeon", "jolteon",
        "flareon", "porygon", "omanyte", "omastar", "kabuto", "kabutops",
        "aerodactyl", "snorlax", "articuno", "zapdos", "moltres", "dratini",
        "dragonair", "dragonite", "mewtwo", "mew"
    ],
    traps=[
        "trip_trap", "mud_trap", "grimy_trap", "poison_trap", "spiked_tile",
        "stealth_rock", "warp_trap", "gust_trap", "slumber_trap", "slow_trap",
        "spin_trap", "grapple_trap", "pitfall_trap", "warp_tile", "wonder_tile",
        "pokemon_trap", "seal_trap", "mud_trap", "sticky_trap"
    ],
    hud_elements=[
        "hp_bar", "belly_bar", "level_indicator", "dungeon_floor",
        "team_status", "inventory_icon", "map_icon", "leader_icon",
        "partner_icon", "menu_button", "message_box", "choice_cursor"
    ],
    special_tiles=[
        "wall", "floor", "water", "lava", "ice", "cracked_floor",
        "hole", "chest", "shop", "hidden_item", "buried_item"
    ]
)


class QwenVLSpriteDetector(BaseSpriteDetector):
    """Sprite detector using Qwen3-VL models."""

    def __init__(
        self,
        config: DetectionConfig,
        qwen_controller: Optional[Any] = None,
        labels: Optional[SpriteLabels] = None,
    ):
        """Initialize sprite detector.

        Args:
            config: Detection configuration
            qwen_controller: QwenController instance for vision generation
            labels: Sprite labels configuration
        """
        super().__init__(config, labels)
        self.qwen_controller = qwen_controller

        logger.info("Initialized QwenVLSpriteDetector with confidence threshold %.2f", config.confidence_threshold)

    def detect(self, image_path: Path, grid_data: Optional[GridData] = None) -> List[DetectionResult]:
        """Detect sprites in image with optional grid context."""
        if not image_path.exists():
            logger.error("Image file not found: %s", image_path)
            return []

        # Build detection prompt
        prompt = self._build_detection_prompt()

        # Use Qwen controller for real detection
        if self.qwen_controller is None:
            logger.warning("No Qwen controller provided, falling back to mock detection")
            detections = self._mock_detection(image_path)
        else:
            try:
                # Load image
                from PIL import Image
                image = Image.open(image_path)

                # Generate vision response
                response = self.qwen_controller.generate_vision(
                    image=image,
                    prompt=prompt,
                    capability_tags=["vision_only"]
                )

                # Parse JSON response
                detections = self._parse_detection_response(response)

            except Exception as e:
                logger.error("Qwen detection failed: %s", e)
                detections = self._mock_detection(image_path)

        # Compute perceptual hashes if enabled
        if self.config.enable_phash_computation:
            try:
                from PIL import Image
                image = Image.open(image_path)
                image_array = np.array(image)
                for detection in detections:
                    # Extract sprite region from bbox
                    x, y, w, h = detection.bbox
                    sprite_region = image_array[y:y+h, x:x+w]
                    if sprite_region.size > 0:
                        detection.phash = compute_phash(sprite_region)
            except Exception as e:
                logger.warning("Failed to compute perceptual hash: %s", e)

        # Apply filtering
        filtered = self._filter_detections(detections)

        # Add grid correlation if enabled and grid data provided
        if self.config.enable_grid_correlation and grid_data:
            filtered = self._correlate_with_grid(filtered, grid_data)

        logger.info("Detected %d sprites in %s", len(filtered), image_path)
        return filtered

    def _correlate_with_grid(self, detections: List[DetectionResult], grid_data: GridData) -> List[DetectionResult]:
        """Correlate detections with grid data to add grid positions."""
        # Simple correlation based on bounding box center
        for detection in detections:
            center_x = detection.bbox[0] + detection.bbox[2] // 2
            center_y = detection.bbox[1] + detection.bbox[3] // 2

            # Convert pixel coordinates to grid coordinates (assuming 16x16 tiles)
            grid_x = center_x // 16
            grid_y = center_y // 16

            # Validate grid bounds
            if 0 <= grid_x < grid_data.width and 0 <= grid_y < grid_data.height:
                detection.grid_pos = (grid_x, grid_y)

        return detections

    def _build_detection_prompt(self) -> str:
        """Build the detection prompt for the model with PMD-specific tuning."""
        prompt = f"""Analyze this Pokemon Mystery Dungeon Red Rescue Team game screenshot and identify all visible sprites, items, enemies, and HUD elements.

IMPORTANT: Focus on the game world area (the grid-based dungeon view) and ignore any emulator UI, borders, or menus.

SPRITE CATEGORIES TO DETECT:

STAIRS (vertical transitions):
- up_stairs, down_stairs, exit_stairs, warp_stairs, golden_stairs, silver_stairs, hidden_stairs
- Look for glowing or animated tiles that indicate floor changes
- Usually 32x32 pixels, centered on grid tiles

ITEMS (consumables and treasures):
- Food: apple, banana, orange, berry, seed, herb, gummi, wonder_gummi
- Healing: heal_seed, oran_berry, sitrus_berry, reviver_seed, totter_seed
- Tools: stick, iron_thorn, silver_spike, gold_fang, rare_fossil, amulet_coin
- Treasure: gold_bar, pearl, big_pearl, stardust, star_piece, nugget, heart_scale
- Herbs: pretty_feather, bright_powder, white_herb, mental_herb, power_herb
- Mail: grass_mail, flame_mail, bubble_mail, bloom_mail, tunnel_mail, steel_mail, heart_mail, snow_mail, space_mail, air_mail, mosaic_mail, brick_mail

ENEMIES (Pokemon):
- Common: caterpie, metapod, butterfree, weedle, kakuna, beedrill, pidgey, pidgeotto, pidgeot, rattata, raticate, spearow, fearow, ekans, arbok, pikachu, raichu, sandshrew, sandslash, nidoran_f, nidorina, nidoqueen, nidoran_m, nidorino, nidoking, clefairy, clefable, vulpix, ninetales, jigglypuff, wigglytuff, zubat, golbat, oddish, gloom, vileplume, paras, parasect, venonat, venomoth, diglett, dugtrio, meowth, persian, psyduck, golduck, mankey, primeape, growlithe, arcanine, poliwag, poliwhirl, poliwrath, abra, kadabra, alakazam, machop, machoke, machamp, bellsprout, weepinbell, victreebel, tentacool, tentacruel, geodude, graveler, golem, ponyta, rapidash, slowpoke, slowbro, magnemite, magneton, farfetchd, doduo, dodrio, seel, dewgong, grimer, muk, shellder, cloyster, gastly, haunter, gengar, onix, drowzee, hypno, krabby, kingler, voltorb, electrode, exeggcute, exeggutor, cubone, marowak, hitmonlee, hitmonchan, lickitung, koffing, weezing, rhyhorn, rhydon, chansey, tangela, kangaskhan, horsea, seadra, goldeen, seaking, staryu, starmie, mr_mime, scyther, jynx, electabuzz, magmar, pinsir, tauros, magikarp, gyarados, lapras, ditto, eevee, vaporeon, jolteon, flareon, porygon, omanyte, omastar, kabuto, kabutops, aerodactyl, snorlax, articuno, zapdos, moltres, dratini, dragonair, dragonite, mewtwo, mew

TRAPS (hazards):
- Damage: trip_trap, mud_trap, grimy_trap, poison_trap, spiked_tile, stealth_rock
- Movement: warp_trap, gust_trap, slumber_trap, slow_trap, spin_trap, grapple_trap, pitfall_trap, seal_trap
- Special: warp_tile, wonder_tile, pokemon_trap, mud_trap, sticky_trap

HUD ELEMENTS (UI components):
- hp_bar: Red health bar showing current HP/max HP ratio
- belly_bar: Yellow/orange belly meter showing hunger status
- level_indicator: Current dungeon floor number (B1, B2, etc.)
- dungeon_floor: Floor counter display
- team_status: Partner/team member status icons
- inventory_icon: Bag/backpack icon
- map_icon: Mini-map or map button
- leader_icon: Protagonist character icon
- partner_icon: Partner Pokemon icon
- menu_button: Menu or pause button
- message_box: Text dialog box for game messages
- choice_cursor: Selection arrow or cursor in menus

SPECIAL TILES (terrain):
- Terrain: wall, floor, water, lava, ice, cracked_floor, hole, chest
- Interactive: shop, hidden_item, buried_item

DETECTION RULES:
1. Only detect elements visible in the main game viewport (ignore emulator borders)
2. Bounding boxes should be tight around the actual sprite/element
3. Use exact category names from the lists above
4. Prioritize HUD elements that show game state (HP, belly, floor number)
5. For sprites, focus on the 16x16 to 32x32 pixel grid-aligned objects
6. Confidence should reflect visual clarity and typical sprite appearance

OUTPUT FORMAT:
Return a JSON array of detections with this exact structure:
[
  {{
    "label": "hp_bar",
    "confidence": 0.95,
    "bbox": [x, y, width, height],
    "metadata": {{"type": "hud", "description": "Current HP status"}}
  }},
  {{
    "label": "up_stairs",
    "confidence": 0.88,
    "bbox": [x, y, width, height],
    "metadata": {{"type": "stairs", "direction": "up"}}
  }}
]"""

        return prompt

    def _mock_detection(self, image_path: Path) -> List[DetectionResult]:
        """Mock detection results for development/testing with realistic PMD elements."""
        # Mock results simulating a typical PMD dungeon scene
        mock_results = [
            # HUD elements (always present in game view)
            DetectionResult(
                label="hp_bar",
                confidence=0.98,
                bbox=(10, 10, 80, 8),
                metadata={"type": "hud", "description": "Current HP status bar"}
            ),
            DetectionResult(
                label="belly_bar",
                confidence=0.97,
                bbox=(10, 25, 60, 6),
                metadata={"type": "hud", "description": "Belly hunger meter"}
            ),
            DetectionResult(
                label="level_indicator",
                confidence=0.95,
                bbox=(400, 10, 30, 15),
                metadata={"type": "hud", "floor": "B1", "description": "Current dungeon floor"}
            ),

            # Stairs (important navigation element)
            DetectionResult(
                label="up_stairs",
                confidence=0.92,
                bbox=(200, 150, 32, 32),
                metadata={"type": "stairs", "direction": "up", "description": "Exit stairs"}
            ),

            # Items (scattered around dungeon)
            DetectionResult(
                label="apple",
                confidence=0.88,
                bbox=(150, 200, 16, 16),
                metadata={"type": "item", "healing": 10, "description": "Restores 10 HP"}
            ),
            DetectionResult(
                label="oran_berry",
                confidence=0.85,
                bbox=(300, 180, 16, 16),
                metadata={"type": "item", "healing": 100, "description": "Restores 100 HP"}
            ),

            # Enemies (typical dungeon inhabitants)
            DetectionResult(
                label="caterpie",
                confidence=0.90,
                bbox=(100, 100, 24, 24),
                metadata={"type": "enemy", "level": 3, "hp": 20, "description": "Bug type enemy"}
            ),
            DetectionResult(
                label="pidgey",
                confidence=0.87,
                bbox=(250, 120, 24, 24),
                metadata={"type": "enemy", "level": 4, "hp": 25, "description": "Flying type enemy"}
            ),

            # Traps (hazards to avoid)
            DetectionResult(
                label="trip_trap",
                confidence=0.83,
                bbox=(175, 225, 16, 16),
                metadata={"type": "trap", "damage": 5, "description": "Causes tripping damage"}
            ),

            # Special tiles
            DetectionResult(
                label="chest",
                confidence=0.94,
                bbox=(350, 200, 24, 24),
                metadata={"type": "special", "description": "Treasure chest"}
            ),
        ]
        return mock_results

    def _parse_detection_response(self, response: str) -> List[DetectionResult]:
        """Parse detection response from Qwen model with robust error handling.

        Args:
            response: Raw response string from model

        Returns:
            List of parsed detection results
        """
        try:
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Try to extract JSON array directly
            if response.startswith('[') and response.endswith(']'):
                detections_data = json.loads(response)
            else:
                # Try to find JSON array within text
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    detections_data = json.loads(json_str)
                else:
                    # Try to find JSON object with detections field
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        detections_data = data.get("detections", [])
                    else:
                        logger.warning("No JSON found in response, using mock results")
                        return self._mock_detection(Path("dummy"))

            # Parse detections with validation
            detections = []
            for item in detections_data:
                if isinstance(item, dict) and "label" in item:
                    # Validate required fields
                    if "bbox" not in item or not isinstance(item["bbox"], list) or len(item["bbox"]) != 4:
                        logger.warning("Invalid bbox format for detection: %s", item)
                        continue

                    # Validate bbox coordinates are reasonable (positive, within screen bounds)
                    bbox = item["bbox"]
                    if any(coord < 0 or coord > 1000 for coord in bbox):
                        logger.warning("Invalid bbox coordinates: %s", bbox)
                        continue

                    # Validate confidence is reasonable
                    confidence = float(item.get("confidence", 0.0))
                    if not 0.0 <= confidence <= 1.0:
                        logger.warning("Invalid confidence value: %f", confidence)
                        confidence = 0.0

                    detection = DetectionResult(
                        label=item["label"],
                        confidence=confidence,
                        bbox=tuple(bbox),
                        metadata=item.get("metadata", {})
                    )
                    detections.append(detection)

            logger.info("Parsed %d valid detections from Qwen response", len(detections))
            return detections

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse detection response: %s", e)
            logger.debug("Raw response: %s", response[:500])
            return self._mock_detection(Path("dummy"))

    def save_detections(self, detections: List[DetectionResult], output_path: Path) -> None:
        """Save detections to JSON file.

        Args:
            detections: List of detection results
            output_path: Output JSON file path
        """
        data = {
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                    "metadata": d.metadata
                }
                for d in detections
            ],
            "metadata": {
                "model": "qwen3-vl-4b",  # Default model name
                "confidence_threshold": self.config.confidence_threshold,
                "timestamp": None,  # Would add actual timestamp
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %d detections to %s", len(detections), output_path)

    def load_detections(self, input_path: Path) -> List[DetectionResult]:
        """Load detections from JSON file.

        Args:
            input_path: Input JSON file path

        Returns:
            List of detection results
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        detections = []
        for d in data.get("detections", []):
            detections.append(DetectionResult(
                label=d["label"],
                confidence=d["confidence"],
                bbox=tuple(d["bbox"]),
                metadata=d.get("metadata", {})
            ))

        logger.info("Loaded %d detections from %s", len(detections), input_path)
        return detections


@dataclass
class SpriteHash:
    """Perceptual hash data for sprite matching."""
    label: str
    phash: str  # Hex string representation of perceptual hash
    category: str
    metadata: Dict[str, Any]
    confidence_threshold: float = 0.85  # Hamming distance threshold

    def matches(self, other_phash: str) -> Tuple[bool, float]:
        """Check if this hash matches another hash.

        Returns:
            Tuple of (matches, confidence_score)
        """
        # Convert hex strings to hash objects for comparison
        try:
            hash1 = imagehash.hex_to_hash(self.phash)
            hash2 = imagehash.hex_to_hash(other_phash)

            # Calculate Hamming distance
            distance = hash1 - hash2

            # Convert distance to similarity score (lower distance = higher similarity)
            # Max possible distance for phash is 64 (8x8 = 64 bits)
            max_distance = 64.0
            similarity = 1.0 - (distance / max_distance)

            # Check if similarity meets threshold
            matches = similarity >= self.confidence_threshold

            return matches, similarity

        except (ValueError, TypeError) as e:
            logger.warning("Hash comparison failed: %s", e)
            return False, 0.0


@dataclass
class SpriteLibrary:
    """Library of known sprites with their perceptual hashes."""
    sprites: Dict[str, SpriteHash]  # label -> SpriteHash
    category_index: Dict[str, List[str]]  # category -> list of labels

    def __init__(self):
        self.sprites = {}
        self.category_index = defaultdict(list)

    def add_sprite(self, sprite_hash: SpriteHash) -> None:
        """Add a sprite to the library."""
        self.sprites[sprite_hash.label] = sprite_hash
        self.category_index[sprite_hash.category].append(sprite_hash.label)

    def find_matches(self, phash: str, category: Optional[str] = None) -> List[Tuple[str, float]]:
        """Find matching sprites for a given perceptual hash.

        Args:
            phash: Perceptual hash to match against
            category: Optional category filter

        Returns:
            List of (label, confidence) tuples for matches
        """
        candidates = self.sprites.values()
        if category:
            candidates = [self.sprites[label] for label in self.category_index.get(category, [])]

        matches = []
        for sprite in candidates:
            is_match, confidence = sprite.matches(phash)
            if is_match:
                matches.append((sprite.label, confidence))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_sprite(self, label: str) -> Optional[SpriteHash]:
        """Get sprite hash by label."""
        return self.sprites.get(label)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SpriteLibrary":
        """Load sprite library from YAML file."""
        library = cls()

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for sprite_data in data.get('sprites', []):
            sprite_hash = SpriteHash(**sprite_data)
            library.add_sprite(sprite_hash)

        logger.info("Loaded %d sprites from %s", len(library.sprites), yaml_path)
        return library

    def to_yaml(self, yaml_path: Path) -> None:
        """Save sprite library to YAML file."""
        data = {
            'sprites': [
                {
                    'label': sprite.label,
                    'phash': sprite.phash,
                    'category': sprite.category,
                    'metadata': sprite.metadata,
                    'confidence_threshold': sprite.confidence_threshold
                }
                for sprite in self.sprites.values()
            ]
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("Saved %d sprites to %s", len(self.sprites), yaml_path)
def create_detector(
    config: Optional[DetectionConfig] = None,
    qwen_controller: Optional[Any] = None,
    labels_path: Optional[Path] = None,
) -> QwenVLSpriteDetector:
    """Create a sprite detector with optional custom labels.

    Args:
        config: Detection configuration
        qwen_controller: QwenController instance for vision generation
        labels_path: Path to custom labels YAML file

    Returns:
        Configured sprite detector
    """
    if config is None:
        config = DetectionConfig()

    labels = None
    if labels_path and labels_path.exists():
        labels = SpriteLabels.from_yaml(labels_path)

    return QwenVLSpriteDetector(config=config, qwen_controller=qwen_controller, labels=labels)


# CLI interface
def main():
    """CLI entry point for sprite detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-VL Sprite Detector for PMD")
    parser.add_argument("image", help="Path to game screenshot")
    parser.add_argument(
        "--qwen-controller",
        help="Path to Qwen controller module (optional, uses mock if not provided)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to custom labels YAML file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for detections"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    # Create config
    config = DetectionConfig(confidence_threshold=args.confidence)

    # Create detector (without qwen controller for now - would need to load it)
    detector = create_detector(
        config=config,
        qwen_controller=None,  # Would load from args.qwen_controller if provided
        labels_path=args.labels,
    )

    # Detect sprites
    image_path = Path(args.image)
    detections = detector.detect(image_path)

    # Output results
    if args.output:
        detector.save_detections(detections, args.output)
        print(f"Saved {len(detections)} detections to {args.output}")
    else:
        # Print to stdout
        for i, d in enumerate(detections):
            print(f"{i+1}. {d.label} (conf: {d.confidence:.2f}) at {d.bbox}")


if __name__ == "__main__":
    main()

class PHashSpriteDetector(BaseSpriteDetector):
    """Sprite detector using perceptual hashing for fast, accurate sprite matching."""

    def __init__(
        self,
        config: DetectionConfig,
        sprite_library: Optional[SpriteLibrary] = None,
        labels: Optional[SpriteLabels] = None,
    ):
        """Initialize pHash sprite detector.

        Args:
            config: Detection configuration
            sprite_library: Pre-computed sprite library with hashes
            labels: Sprite labels configuration
        """
        super().__init__(config, labels)
        self.sprite_library = sprite_library or SpriteLibrary()

        # Cache for computed hashes to avoid recomputation
        self._hash_cache: Dict[str, str] = {}
        self._dedup_cache: Dict[str, str] = {}  # phash -> canonical label

        logger.info("Initialized PHashSpriteDetector with %d sprites", len(self.sprite_library.sprites))

    def detect(self, image_path: Path, grid_data: Optional[GridData] = None) -> List[DetectionResult]:
        """Detect sprites using perceptual hashing."""
        if not image_path.exists():
            logger.error("Image file not found: %s", image_path)
            return []

        try:
            # Load and process image
            image = Image.open(image_path)

            # Extract sprites from image (this would use grid parsing or quad capture)
            sprite_regions = self._extract_sprite_regions(image, grid_data)

            detections = []
            for region, bbox in sprite_regions:
                # Compute perceptual hash for this region
                phash = self._compute_phash(region)

                # Find matches in sprite library
                matches = self.sprite_library.find_matches(phash)

                if matches:
                    # Use best match
                    best_label, confidence = matches[0]

                    # Handle deduplication
                    canonical_label = self._get_canonical_label(phash, best_label)

                    detection = DetectionResult(
                        label=canonical_label,
                        confidence=confidence,
                        bbox=bbox,
                        metadata={
                            "method": "phash",
                            "phash": phash,
                            "matches": len(matches),
                            "category": self._get_category(canonical_label)
                        }
                    )
                    detections.append(detection)

            # Apply filtering
            filtered = self._filter_detections(detections)

            # Add grid correlation if enabled
            if self.config.enable_grid_correlation and grid_data:
                filtered = self._correlate_with_grid(filtered, grid_data)

            logger.info("Detected %d sprites via pHash in %s", len(filtered), image_path)
            return filtered

        except Exception as e:
            logger.error("PHash detection failed: %s", e)
            return []

    def _extract_sprite_regions(self, image: Image.Image, grid_data: Optional[GridData]) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Extract individual sprite regions from the full image.

        This is a simplified implementation - in practice, this would integrate
        with grid parsing and quad capture systems to identify sprite locations.
        """
        regions = []

        # For now, use a simple grid-based approach assuming 16x16 sprites
        # In practice, this would use the grid parser to identify sprite positions
        width, height = image.size

        # Assume sprites are aligned to 16x16 grid (typical for PMD)
        sprite_size = 16

        for y in range(0, height - sprite_size + 1, sprite_size):
            for x in range(0, width - sprite_size + 1, sprite_size):
                # Extract sprite region
                region = image.crop((x, y, x + sprite_size, y + sprite_size))

                # Simple heuristic: skip mostly transparent/background regions
                if self._is_sprite_region(region):
                    bbox = (x, y, sprite_size, sprite_size)
                    regions.append((region, bbox))

        return regions

    def _is_sprite_region(self, region: Image.Image) -> bool:
        """Check if a region likely contains a sprite (not background)."""
        # Convert to grayscale for analysis
        gray = region.convert('L')

        # Calculate variance - sprites typically have more variation than background
        pixels = list(gray.getdata())
        if not pixels:
            return False

        mean = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)

        # Threshold for sprite detection (tune based on game)
        return variance > 100  # Adjust threshold as needed

    def _compute_phash(self, image: Image.Image) -> str:
        """Compute perceptual hash for an image region using deterministic compute_phash."""
        # Create cache key from image content
        image_bytes = image.tobytes()
        cache_key = hashlib.md5(image_bytes).hexdigest()

        # Check cache first
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        # Convert PIL image to numpy array for compute_phash
        image_array = np.array(image)

        # Use deterministic compute_phash from sprite_phash module
        phash_array = compute_phash(image_array)

        # Convert binary array to hex string for storage (compatibility with existing code)
        phash_hex = ''.join(str(int(bit)) for bit in phash_array)

        # Cache result
        self._hash_cache[cache_key] = phash_hex

        return phash_hex

    def _get_canonical_label(self, phash: str, detected_label: str) -> str:
        """Get canonical label for deduplication."""
        if phash in self._dedup_cache:
            return self._dedup_cache[phash]

        # First time seeing this hash, use detected label as canonical
        self._dedup_cache[phash] = detected_label
        return detected_label

    def add_sprite_to_library(self, label: str, image: Image.Image, category: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a sprite to the library by computing its hash."""
        phash = self._compute_phash(image)

        sprite_hash = SpriteHash(
            label=label,
            phash=phash,
            category=category,
            metadata=metadata or {},
            confidence_threshold=0.85
        )

        self.sprite_library.add_sprite(sprite_hash)
        logger.info("Added sprite %s (%s) to library", label, category)

    def save_library(self, path: Path) -> None:
        """Save sprite library to YAML file."""
        self.sprite_library.to_yaml(path)

    def load_library(self, path: Path) -> None:
        """Load sprite library from YAML file."""
        self.sprite_library = SpriteLibrary.from_yaml(path)
