"""4-up capture system for Pokemon MD agent.

Captures quad-view screenshots (environment, map, grid, meta) and saves
with ASCII variants for LLM consumption.
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from ..environment.mgba_controller import MGBAController
from ..environment.ram_decoders import RAMSnapshot
from .grid_parser import GridParser

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


@dataclass
class FrameData:
    """Frame data with synchronization info."""
    frame: int
    timestamp: float
    image: Optional[Image.Image] = None
    game_state: Optional[Dict[str, Any]] = None


class QuadCapture:
    """4-up capture system."""

    def __init__(self, controller: MGBAController, output_dir: Path, video_config=None):
        """Initialize quad capture.

        Args:
            controller: MGBA controller instance
            output_dir: Directory to save captures
            video_config: Video configuration for dynamic resolution
        """
        self.controller = controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_config = video_config or controller.video_config

        # Initialize grid parser for overlay generation
        self.grid_parser = GridParser(video_config=self.video_config)
        
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


class AsyncScreenshotCapture:
    """Async screenshot capture with background buffering for <5ms latency.

    Maintains a 2-frame circular buffer with background capture thread.
    Agent reads instantly from buffer, never blocking on capture operations.
    """

    def __init__(self, controller: MGBAController, output_dir: Path, buffer_size: int = 2):
        """Initialize async screenshot capture.

        Args:
            controller: MGBA controller for screenshot operations
            output_dir: Directory to save captures
            buffer_size: Size of circular buffer (default: 2)
        """
        self.controller = controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_config = controller.video_config
        self.buffer_size = buffer_size
        self.frame_buffer: List[Optional[FrameData]] = [None] * buffer_size
        self.buffer_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 3
        self.last_frame_counter = 0

        # Initialize directories for quad capture functionality
        self.screens_dir = self.output_dir / "screens"
        self.ascii_dir = self.output_dir / "ascii"
        self.screens_dir.mkdir(exist_ok=True)
        self.ascii_dir.mkdir(exist_ok=True)

        # Font for ASCII rendering
        try:
            self.font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        except (OSError, IOError):
            try:
                self.font = ImageFont.truetype("Courier New", 12)
            except (OSError, IOError):
                self.font = ImageFont.load_default()

        logger.info("AsyncScreenshotCapture initialized with buffer size: %d, output dir: %s", buffer_size, output_dir)

    def start(self) -> None:
        """Start background capture thread."""
        if self.running:
            logger.warning("Capture thread already running")
            return

        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="async-screenshot-capture",
            daemon=True
        )
        self.capture_thread.start()
        logger.info("Async screenshot capture thread started")

    def stop(self) -> None:
        """Stop background capture thread gracefully."""
        if not self.running:
            return

        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")

        self.capture_thread = None
        logger.info("Async screenshot capture thread stopped")

    def _capture_loop(self) -> None:
        """Background capture loop with error handling and restart logic."""
        consecutive_failures = 0
        max_consecutive_failures = 5

        while self.running:
            try:
                # Capture screenshot
                frame_counter = self.last_frame_counter + 1
                timestamp = time.time()

                # Capture screenshot (this is the blocking operation)
                temp_path = f"temp_async_capture_{frame_counter}.png"
                image = self.controller.grab_frame()

                # Create frame data
                frame_data = FrameData(
                    frame=frame_counter,
                    timestamp=timestamp,
                    image=image,
                    game_state={"frame_counter": frame_counter}
                )

                # Write to buffer
                self._write_frame_to_buffer(frame_data)
                self.last_frame_counter = frame_counter

                # Reset failure counter on success
                consecutive_failures = 0

                # Rate limit to ~30 FPS
                time.sleep(1/30)

            except Exception as e:
                consecutive_failures += 1
                logger.error("Screenshot capture failed (attempt %d/%d): %s",
                           consecutive_failures, max_consecutive_failures, e)

                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive failures, restarting thread")
                    self._restart_thread()
                    consecutive_failures = 0

                # Brief pause before retry
                time.sleep(0.1)

    def _restart_thread(self) -> None:
        """Restart capture thread on failures."""
        if self.restart_count >= self.max_restarts:
            logger.error("Max restart attempts (%d) exceeded, stopping async capture", self.max_restarts)
            self.running = False
            return

        self.restart_count += 1
        logger.warning("Restarting capture thread (attempt %d/%d)", self.restart_count, self.max_restarts)

        # Don't try to join the current thread - just mark it as stopping
        # The thread will exit naturally when self.running becomes False
        self.running = False

        # Start new thread
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"async-screenshot-capture-restart-{self.restart_count}",
            daemon=True
        )
        self.capture_thread.start()

    def _write_frame_to_buffer(self, frame_data: FrameData) -> None:
        """Write frame data to circular buffer."""
        with self.buffer_lock:
            # Simple circular buffer: keep two most recent frames
            self.frame_buffer[0] = self.frame_buffer[1]  # Shift older frame
            self.frame_buffer[1] = frame_data  # New frame

    def get_latest_frame(self) -> Optional[FrameData]:
        """Get latest frame from buffer (non-blocking)."""
        with self.buffer_lock:
            return self.frame_buffer[1] if self.frame_buffer[1] else self.frame_buffer[0]

    def get_frame_for_game_state(self, game_frame: int, tolerance_ms: int = 10) -> Optional[FrameData]:
        """Get frame synchronized with game state.

        Args:
            game_frame: Game frame counter to match
            tolerance_ms: Maximum age tolerance in milliseconds

        Returns:
            Synchronized frame data or None if no match within tolerance
        """
        with self.buffer_lock:
            for frame_data in reversed(self.frame_buffer):
                if frame_data is None:
                    continue

                # Check frame counter match
                if frame_data.frame == game_frame:
                    age_ms = (time.time() - frame_data.timestamp) * 1000
                    if age_ms <= tolerance_ms:
                        return frame_data

        return None

    def get_latest_frame_or_capture_sync(self) -> Optional[FrameData]:
        """Get latest frame or fall back to synchronous capture."""
        frame = self.get_latest_frame()
        if frame is not None:
            return frame

        # Fallback to sync capture
        logger.warning("No buffered frames available, falling back to sync capture")
        try:
            timestamp = time.time()
            frame_counter = self.last_frame_counter + 1
            temp_path = f"temp_sync_fallback_{frame_counter}.png"
            image = self.controller.grab_frame()

            return FrameData(
                frame=frame_counter,
                timestamp=timestamp,
                image=image,
                game_state={"frame_counter": frame_counter}
            )
        except Exception as e:
            logger.error("Sync fallback capture failed: %s", e)
            return None
    
    def capture_quad_view(self, frame: int, floor: int, dungeon_id: int,
                          room_kind: str, player_pos: tuple[int, int],
                          entities_count: int, items_count: int,
                          enable_overlay: bool = True) -> Optional[str]:
        """Capture 4-up view and return capture ID.

        Args:
            frame: Current frame number
            floor: Current floor
            dungeon_id: Current dungeon ID
            room_kind: Type of room
            player_pos: Player position (x, y)
            entities_count: Number of entities visible
            items_count: Number of items visible
            enable_overlay: Enable grid overlay rendering (default: True)

        Returns:
            Capture ID or None if failed
        """
        timestamp = time.time()
        capture_id = "04d"
        
        try:
            # Capture individual views
            env_img = self.controller.grab_frame()
            map_img = self._capture_minimap()
            grid_img = self._capture_grid_overlay(enable_overlay=enable_overlay)
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
    
    def _capture_grid_overlay(self, enable_overlay: bool = True) -> Optional[Image.Image]:
        """Capture grid overlay view.

        Args:
            enable_overlay: Enable grid overlay rendering (default: True)

        Returns:
            Grid overlay image or None
        """
        if not enable_overlay:
            # Return blank image if overlay disabled
            return Image.new('RGB', (160, 144), color='black')

        # Generate grid overlay using grid parser
        try:
            # For now, create a simple overlay - full implementation would parse RAM
            # and use grid_parser to generate proper overlay
            img = Image.new('RGBA', (160, 144), (0, 0, 0, 0))  # Transparent background
            draw = ImageDraw.Draw(img)

            # Add grid lines (placeholder for actual grid parsing)
            grid_color = (255, 255, 255, 128)  # Semi-transparent white
            for x in range(0, 160, 16):  # 16px tiles
                draw.line([(x, 0), (x, 144)], fill=grid_color, width=1)
            for y in range(0, 144, 16):
                draw.line([(0, y), (160, y)], fill=grid_color, width=1)

            # Add coordinate labels (placeholder)
            label_font = ImageFont.load_default()
            for r in range(0, 9):  # 9 rows
                for c in range(0, 10):  # 10 columns
                    label_x = c * 16 + 2
                    label_y = r * 16 + 2
                    draw.text((label_x, label_y), f"({r},{c})", fill=(255, 255, 255, 200), font=label_font)

            logger.debug("Generated grid overlay image")
            return img

        except (OSError, ValueError) as e:
            logger.error("Failed to generate grid overlay: %s", e)
            # Fallback to placeholder
            img = Image.new('RGB', (160, 144), color='black')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "GRID\n(ERROR)", fill='white', font=self.font)
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
        text = f"""META VIEW
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
        half_width = self.video_config.width // 2
        half_height = self.video_config.height // 2
        composite = Image.new('RGB', (self.video_config.width, self.video_config.height))

        # Layout: (env | dynamic-map+grid)/(env+grid | meta)
        # Top-left: env (full size)
        env_resized = env.resize((half_width, half_height), Image.Resampling.LANCZOS)
        composite.paste(env_resized, (0, 0))

        # Top-right: dynamic-map+grid (overlay map and grid)
        top_right = Image.new('RGB', (half_width, half_height))
        map_resized = map_.resize((half_width, half_height), Image.Resampling.LANCZOS)
        grid_resized = grid.resize((half_width, half_height), Image.Resampling.LANCZOS)
        top_right.paste(map_resized, (0, 0))
        top_right.paste(grid_resized, (0, 0), grid_resized)  # Composite with alpha if available
        composite.paste(top_right, (half_width, 0))

        # Bottom-left: env+grid (overlay env and grid)
        bottom_left = Image.new('RGB', (half_width, half_height))
        bottom_left.paste(env_resized, (0, 0))
        bottom_left.paste(grid_resized, (0, 0), grid_resized)  # Composite with alpha if available
        composite.paste(bottom_left, (0, half_height))

        # Bottom-right: meta
        meta_resized = meta.resize((half_width, half_height), Image.Resampling.LANCZOS)
        composite.paste(meta_resized, (half_width, half_height))

        # Add labels
        draw = ImageDraw.Draw(composite)
        label_font = ImageFont.load_default()

        labels = [
            ("ENV", 10, 10),
            ("MAP+GRID", half_width + 10, 10),
            ("ENV+GRID", 10, half_height + 10),
            ("META", half_width + 10, half_height + 10)
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