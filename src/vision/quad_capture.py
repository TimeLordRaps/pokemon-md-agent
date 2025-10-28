"""4-up capture system for Pokemon MD agent.

Captures quad-view screenshots (environment, map, grid, meta) and saves
with ASCII variants for LLM consumption.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from ..environment.mgba_controller import MGBAController

logger = logging.getLogger(__name__)


@dataclass
class CaptureMetadata:
    """Metadata for a capture."""
    timestamp: float
    frame: int
    floor: int
    dungeon_id: int
    room_kind: str
    player_pos: tuple[int, int]
    entities_count: int
    items_count: int
    ascii_available: bool = False


class QuadCapture:
    """4-up capture system."""
    
    def __init__(self, controller: MGBAController, output_dir: Path):
        """Initialize quad capture.
        
        Args:
            controller: MGBA controller instance
            output_dir: Directory to save captures
        """
        self.controller = controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture subdirectories
        self.screens_dir = self.output_dir / "screens"
        self.ascii_dir = self.output_dir / "ascii"
        self.screens_dir.mkdir(exist_ok=True)
        self.ascii_dir.mkdir(exist_ok=True)
        
        # Font for ASCII rendering (fallback to default if not found)
        try:
            self.font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        except (OSError, IOError):
            try:
                self.font = ImageFont.truetype("Courier New", 12)
            except (OSError, IOError):
                self.font = ImageFont.load_default()
        
        logger.info("QuadCapture initialized with output dir: %s", output_dir)
    
    def capture_quad_view(self, frame: int, floor: int, dungeon_id: int, 
                         room_kind: str, player_pos: tuple[int, int],
                         entities_count: int, items_count: int) -> Optional[str]:
        """Capture 4-up view and return capture ID.
        
        Args:
            frame: Current frame number
            floor: Current floor
            dungeon_id: Current dungeon ID
            room_kind: Type of room
            player_pos: Player position (x, y)
            entities_count: Number of entities visible
            items_count: Number of items visible
            
        Returns:
            Capture ID or None if failed
        """
        timestamp = time.time()
        capture_id = "04d"
        
        try:
            # Capture individual views
            env_path = self.screens_dir / f"temp_env_{frame}.png"
            env_img = self.controller.screenshot(str(env_path))
            if env_path.exists():
                env_img = Image.open(env_path)
                env_path.unlink()  # Clean up temp file
            else:
                env_img = None
            map_img = self._capture_minimap()
            grid_img = self._capture_grid_overlay()
            meta_img = self._create_meta_view(floor, dungeon_id, room_kind, 
                                             player_pos, entities_count, items_count)
            
            if not all([env_img, map_img, grid_img, meta_img]):
                logger.warning("Failed to capture all views for frame %d", frame)
                return None
            
            # Create 4-up composite
            quad_img = self._create_quad_composite(env_img, map_img, grid_img, meta_img)
            if quad_img is None:
                logger.warning("Failed to create quad composite for frame %d", frame)
                return None
            
            # Save images
            screen_path = self.screens_dir / "04d" / "04d"
            screen_path.parent.mkdir(parents=True, exist_ok=True)
            quad_img.save(screen_path)
            
            # Generate and save ASCII variants
            ascii_data = self._generate_ascii_variants(env_img, map_img, grid_img, meta_img)
            ascii_path = self.ascii_dir / "04d" / "04d"
            ascii_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ascii_path, 'w', encoding='utf-8') as f:
                json.dump(ascii_data, f, indent=2)
            
            # Save metadata
            metadata = CaptureMetadata(
                timestamp=timestamp,
                frame=frame,
                floor=floor,
                dungeon_id=dungeon_id,
                room_kind=room_kind,
                player_pos=player_pos,
                entities_count=entities_count,
                items_count=items_count,
                ascii_available=True
            )
            
            metadata_path = screen_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(vars(metadata), f, indent=2, default=str)
            
            logger.debug("Captured quad view: %s", capture_id)
            return capture_id
            
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to capture quad view: %s", e)
            return None
    
    def _capture_minimap(self) -> Optional[Image.Image]:
        """Capture minimap view.
        
        Returns:
            Minimap image or None
        """
        # For now, return a placeholder - would need minimap RAM parsing
        img = Image.new('RGB', (160, 144), color='gray')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "MINIMAP\n(TBD)", fill='white', font=self.font)
        return img
    
    def _capture_grid_overlay(self) -> Optional[Image.Image]:
        """Capture grid overlay view.
        
        Returns:
            Grid overlay image or None
        """
        # For now, return a placeholder - would need grid parsing
        img = Image.new('RGB', (160, 144), color='black')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "GRID\n(TBD)", fill='white', font=self.font)
        return img
    
    def _create_meta_view(self, floor: int, dungeon_id: int, room_kind: str,
                         player_pos: tuple[int, int], entities_count: int, 
                         items_count: int) -> Optional[Image.Image]:
        """Create metadata view.
        
        Args:
            floor: Current floor
            dungeon_id: Dungeon ID
            room_kind: Room type
            player_pos: Player position
            entities_count: Entity count
            items_count: Item count
            
        Returns:
            Meta view image
        """
        img = Image.new('RGB', (160, 144), color='navy')
        draw = ImageDraw.Draw(img)
        
        # Draw metadata
        text = ".2f"".2f"f"""META VIEW
Floor: {floor}
Dungeon: {dungeon_id}
Room: {room_kind}
Player: ({player_pos[0]}, {player_pos[1]})
Entities: {entities_count}
Items: {items_count}
Time: {time.strftime('%H:%M:%S')}"""
        
        draw.text((5, 5), text, fill='white', font=self.font)
        return img
    
    def _create_quad_composite(self, env: Optional[Image.Image], map_: Optional[Image.Image],
                               grid: Optional[Image.Image], meta: Optional[Image.Image]) -> Optional[Image.Image]:
        """Create 4-up composite image with layout: (env | dynamic-map+grid)/(env+grid | meta)

        Args:
            env: Environment screenshot
            map_: Minimap view
            grid: Grid overlay
            meta: Metadata view

        Returns:
            Composite 4-up image or None if any input is None
        """
        if not all([env, map_, grid, meta]):
            return None

        # Type assertions after null check
        assert env is not None
        assert map_ is not None
        assert grid is not None
        assert meta is not None

        # Create 2x2 grid with new layout
        width, height = env.size
        composite = Image.new('RGB', (width * 2, height * 2))

        # Layout: (env | dynamic-map+grid)/(env+grid | meta)
        # Top-left: env
        composite.paste(env, (0, 0))

        # Top-right: dynamic-map+grid (overlay map and grid)
        top_right = Image.new('RGB', (width, height))
        top_right.paste(map_, (0, 0))
        top_right.paste(grid, (0, 0), grid)  # Composite with alpha if available
        composite.paste(top_right, (width, 0))

        # Bottom-left: env+grid (overlay env and grid)
        bottom_left = Image.new('RGB', (width, height))
        bottom_left.paste(env, (0, 0))
        bottom_left.paste(grid, (0, 0), grid)  # Composite with alpha if available
        composite.paste(bottom_left, (0, height))

        # Bottom-right: meta
        composite.paste(meta, (width, height))

        # Add labels
        draw = ImageDraw.Draw(composite)
        label_font = ImageFont.load_default()

        labels = [
            ("ENV", 10, 10),
            ("MAP+GRID", width + 10, 10),
            ("ENV+GRID", 10, height + 10),
            ("META", width + 10, height + 10)
        ]

        for label, x, y in labels:
            draw.text((x, y), label, fill='yellow', font=label_font)

        return composite
    
    def _generate_ascii_variants(self, env: Optional[Image.Image], map_: Optional[Image.Image],
                               grid: Optional[Image.Image], meta: Optional[Image.Image]) -> Dict[str, str]:
        """Generate ASCII variants of all views.
        
        Args:
            env: Environment image
            map_: Minimap image
            grid: Grid overlay image
            meta: Metadata image
            
        Returns:
            Dictionary with ASCII representations
        """
        if not all([env, map_, grid, meta]):
            return {}
            
        # Type assertions after null check
        assert env is not None
        assert map_ is not None
        assert grid is not None
        assert meta is not None
            
        return {
            "environment": self._image_to_ascii(env),
            "map": self._image_to_ascii(map_),
            "grid": self._image_to_ascii(grid),
            "meta": self._image_to_ascii(meta),
            "combined": self._create_combined_ascii(env, map_, grid, meta)
        }
    
    def _image_to_ascii(self, img: Image.Image, width: int = 80, height: int = 24) -> str:
        """Convert image to ASCII art.
        
        Args:
            img: Input image
            width: ASCII width
            height: ASCII height
            
        Returns:
            ASCII art string
        """
        # Resize image
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img = img.convert('L')  # Grayscale
        
        # ASCII characters from dark to light
        chars = " .:-=+*#%@"
        
        pixels = list(img.getdata())
        ascii_str = ""
        
        for i in range(height):
            for j in range(width):
                pixel = pixels[i * width + j]
                char_index = int(pixel / 255 * (len(chars) - 1))
                ascii_str += chars[char_index]
            ascii_str += "\n"
        
        return ascii_str
    
    def _create_combined_ascii(self, env: Image.Image, map_: Image.Image,
                             grid: Image.Image, meta: Image.Image) -> str:
        """Create combined ASCII layout.
        
        Args:
            env: Environment image
            map_: Minimap image
            grid: Grid overlay image
            meta: Metadata image
            
        Returns:
            Combined ASCII string
        """
        env_ascii = self._image_to_ascii(env, 40, 12)
        map_ascii = self._image_to_ascii(map_, 40, 12)
        grid_ascii = self._image_to_ascii(grid, 40, 12)
        meta_ascii = self._image_to_ascii(meta, 40, 12)
        
        # Split into lines
        env_lines = env_ascii.strip().split('\n')
        map_lines = map_ascii.strip().split('\n')
        grid_lines = grid_ascii.strip().split('\n')
        meta_lines = meta_ascii.strip().split('\n')
        
        combined = ["ENVIRONMENT          | MAP"]
        combined.append("-" * 80)
        
        for i in range(12):
            env_line = env_lines[i] if i < len(env_lines) else ""
            map_line = map_lines[i] if i < len(map_lines) else ""
            combined.append(f"{env_line:<40} | {map_line}")
        
        combined.append("")
        combined.append("GRID                | META")
        combined.append("-" * 80)
        
        for i in range(12):
            grid_line = grid_lines[i] if i < len(grid_lines) else ""
            meta_line = meta_lines[i] if i < len(meta_lines) else ""
            combined.append(f"{grid_line:<40} | {meta_line}")
        
        return "\n".join(combined)
    
    def get_recent_captures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent captures.
        
        Args:
            limit: Maximum number of captures to return
            
        Returns:
            List of capture metadata
        """
        captures = []
        
        try:
            # Find all metadata files
            metadata_files = list(self.screens_dir.glob("**/*.json"))
            metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for metadata_file in metadata_files[:limit]:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['capture_id'] = metadata_file.stem
                        captures.append(data)
                except (OSError, ValueError):
                    continue
                    
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to get recent captures: %s", e)
        
        return captures
    
    def cleanup_old_captures(self, max_age_days: int = 7) -> int:
        """Clean up old captures.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of captures cleaned up
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned = 0
        
        try:
            # Clean screens
            for json_file in self.screens_dir.glob("**/*.json"):
                if json_file.stat().st_mtime < cutoff_time:
                    # Remove associated files
                    capture_id = json_file.stem
                    timestamp = json_file.parent.name
                    
                    json_file.unlink()
                    
                    png_file = json_file.with_suffix('.png')
                    if png_file.exists():
                        png_file.unlink()
                    
                    ascii_file = self.ascii_dir / timestamp / f"{capture_id}.json"
                    if ascii_file.exists():
                        ascii_file.unlink()
                    
                    cleaned += 1
                    
        except (OSError, ValueError) as e:
            logger.error("Failed to cleanup captures: %s", e)
        
        logger.info("Cleaned up %d old captures", cleaned)
        return cleaned