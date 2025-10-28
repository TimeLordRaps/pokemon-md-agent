"""Meta view writer for 2×2 grid generation and layout."""

from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
import asyncio
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

logger = logging.getLogger(__name__)


@dataclass
class ViewTile:
    """Individual tile in the meta view grid."""
    image: Image.Image
    metadata: Dict[str, Any]
    position: Tuple[int, int]  # (row, col) in grid
    importance_score: float = 1.0


@dataclass
class MetaViewResult:
    """Result of meta view generation."""
    composite_image: Image.Image
    grid_layout: List[List[Optional[ViewTile]]]
    metadata: Dict[str, Any]
    generation_time: float


class MetaViewWriter:
    """Generates 2×2 grid meta views from temporal keyframes."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (2, 2),
        tile_size: Tuple[int, int] = (240, 160),  # 2× resolution: 480×320 total canvas
        padding: int = 4,
        background_color: Tuple[int, int, int] = (32, 32, 32),
        enable_async: bool = True,
    ):
        """Initialize meta view writer.

        Args:
            grid_size: (rows, cols) for grid layout
            tile_size: (width, height) for each tile
            padding: Padding between tiles
            background_color: Background color (R, G, B)
            enable_async: Enable async operations
        """
        self.grid_rows, self.grid_cols = grid_size
        self.tile_width, self.tile_height = tile_size
        self.padding = padding
        self.background_color = background_color
        self._enable_async = enable_async

        # Calculate total canvas size
        self.canvas_width = self.grid_cols * (self.tile_width + self.padding) + self.padding
        self.canvas_height = self.grid_rows * (self.tile_height + self.padding) + self.padding

        # Font for metadata overlay (optional)
        self.font = None
        try:
            # Try to load a default font
            self.font = ImageFont.load_default()
        except (OSError, ImportError):
            pass

        logger.info(
            "Initialized MetaViewWriter: grid=%dx%d, tile=%dx%d, canvas=%dx%d",
            self.grid_rows, self.grid_cols, self.tile_width, self.tile_height,
            self.canvas_width, self.canvas_height
        )

    def generate_meta_view(
        self,
        tiles: List[ViewTile],
        layout_strategy: str = "importance",
        title: Optional[str] = None,
    ) -> MetaViewResult:
        """Generate 2×2 meta view from tiles.

        Args:
            tiles: List of view tiles to arrange
            layout_strategy: "importance", "temporal", or "random"
            title: Optional title for the view

        Returns:
            MetaViewResult with composite image and metadata
        """
        import time
        start_time = time.time()

        try:
            # Select and arrange tiles
            arranged_tiles = self._arrange_tiles(tiles, layout_strategy)

            # Create composite image
            composite = self._create_composite_image(arranged_tiles, title)

            generation_time = time.time() - start_time

            # Create result
            result = MetaViewResult(
                composite_image=composite,
                grid_layout=arranged_tiles,
                metadata={
                    "layout_strategy": layout_strategy,
                    "total_tiles": len(tiles),
                    "selected_tiles": len([t for row in arranged_tiles for t in row if t is not None]),
                    "title": title,
                    "grid_size": (self.grid_rows, self.grid_cols),
                    "tile_size": (self.tile_width, self.tile_height),
                },
                generation_time=generation_time,
            )

            logger.debug(
                "Generated meta view: %d tiles arranged in %.3fs",
                len([t for row in arranged_tiles for t in row if t is not None]),
                generation_time
            )

            return result

        except Exception as e:
            logger.error("Failed to generate meta view: %s", e)
            # Return empty result
            empty_grid: List[List[Optional[ViewTile]]] = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
            empty_image = Image.new('RGB', (self.canvas_width, self.canvas_height), self.background_color)

            return MetaViewResult(
                composite_image=empty_image,
                grid_layout=empty_grid,
                metadata={"error": str(e)},
                generation_time=time.time() - start_time,
            )

    async def generate_meta_view_async(
        self,
        tiles: List[ViewTile],
        layout_strategy: str = "importance",
        title: Optional[str] = None,
    ) -> MetaViewResult:
        """Async version of generate_meta_view."""
        if not self._enable_async:
            return self.generate_meta_view(tiles, layout_strategy, title)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_meta_view, tiles, layout_strategy, title
        )

    def _arrange_tiles(
        self,
        tiles: List[ViewTile],
        strategy: str,
    ) -> List[List[Optional[ViewTile]]]:
        """Arrange tiles into grid layout."""
        # Sort tiles based on strategy
        if strategy == "importance":
            sorted_tiles = sorted(tiles, key=lambda t: t.importance_score, reverse=True)
        elif strategy == "temporal":
            # Assume tiles have temporal metadata
            sorted_tiles = sorted(tiles, key=lambda t: t.metadata.get('timestamp', 0), reverse=True)
        else:  # random or default
            sorted_tiles = tiles.copy()
            random.shuffle(sorted_tiles)

        # Create grid
        grid: List[List[Optional[ViewTile]]] = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        # Fill grid with available tiles
        tile_idx = 0
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if tile_idx < len(sorted_tiles):
                    tile = sorted_tiles[tile_idx]
                    tile.position = (row, col)
                    grid[row][col] = tile
                    tile_idx += 1

        return grid

    def _create_composite_image(
        self,
        grid: List[List[Optional[ViewTile]]],
        title: Optional[str],
    ) -> Image.Image:
        """Create composite image from grid layout."""
        # Create canvas
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), self.background_color)
        draw = ImageDraw.Draw(canvas)

        # Draw each tile
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                tile = grid[row][col]
                if tile is not None:
                    # Calculate tile position
                    x = col * (self.tile_width + self.padding) + self.padding
                    y = row * (self.tile_height + self.padding) + self.padding

                    # Resize tile image to fit
                    resized_tile = self._resize_image(tile.image, (self.tile_width, self.tile_height))

                    # Paste tile
                    canvas.paste(resized_tile, (x, y))

                    # Optional: draw border
                    self._draw_tile_border(draw, x, y, self.tile_width, self.tile_height)

        # Add title if provided
        if title and self.font:
            self._draw_title(draw, title)

        return canvas

    def _resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to fit tile while maintaining aspect ratio."""
        target_width, target_height = size

        # Calculate resize dimensions maintaining aspect ratio
        img_width, img_height = image.size
        ratio = min(target_width / img_width, target_height / img_height)

        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and center the resized image
        final_image = Image.new('RGB', size, self.background_color)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        final_image.paste(resized, (x_offset, y_offset))

        return final_image

    def _draw_tile_border(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw border around tile."""
        border_color = (64, 64, 64)  # Dark gray
        draw.rectangle([x, y, x + width, y + height], outline=border_color, width=1)

    def _draw_title(self, draw: ImageDraw.ImageDraw, title: str) -> None:
        """Draw title at top of canvas."""
        # Calculate text position (centered at top)
        bbox = draw.textbbox((0, 0), title, font=self.font)
        text_width = bbox[2] - bbox[0]

        x = (self.canvas_width - text_width) // 2
        y = self.padding // 2

        # Draw text with shadow for visibility
        shadow_color = (0, 0, 0)
        text_color = (255, 255, 255)

        draw.text((x + 1, y + 1), title, font=self.font, fill=shadow_color)
        draw.text((x, y), title, font=self.font, fill=text_color)

    def create_view_tiles_from_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        image_generator: Optional[Callable[[np.ndarray, Dict[str, Any]], Image.Image]] = None,
    ) -> List[ViewTile]:
        """Create view tiles from embeddings (for visualization).

        Args:
            embeddings: List of embedding vectors
            metadata_list: Optional metadata for each embedding
            image_generator: Optional function to generate images from embeddings

        Returns:
            List of ViewTile objects
        """
        tiles = []

        for i, embedding in enumerate(embeddings):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}

            # Generate or create placeholder image
            if image_generator:
                image = image_generator(embedding, metadata)
            else:
                # Create a simple visualization based on embedding
                image = self._create_embedding_visualization(embedding)

            # Calculate importance score from metadata or embedding properties
            importance_score = metadata.get('importance_score', 1.0)
            if importance_score == 1.0 and 'similarity' in metadata:
                importance_score = metadata['similarity']

            tile = ViewTile(
                image=image,
                metadata=metadata,
                position=(0, 0),  # Will be set during arrangement
                importance_score=importance_score,
            )

            tiles.append(tile)

        return tiles

    def _create_embedding_visualization(self, embedding: np.ndarray) -> Image.Image:
        """Create a simple visualization of an embedding vector."""
        # Create a small image representing the embedding
        img_size = (64, 64)
        image = Image.new('RGB', img_size, (64, 64, 64))
        draw = ImageDraw.Draw(image)

        # Use embedding values to create a pattern
        # Normalize to 0-255 range
        if len(embedding) > 0:
            normalized = ((embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding) + 1e-8) * 255).astype(int)

            # Create a grid pattern
            grid_size = min(8, int(np.sqrt(len(normalized))))
            cell_width = img_size[0] // grid_size
            cell_height = img_size[1] // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    idx = (i * grid_size + j) % len(normalized)
                    intensity = normalized[idx]
                    color = (intensity, intensity // 2, 255 - intensity)
                    x = j * cell_width
                    y = i * cell_height
                    draw.rectangle([x, y, x + cell_width, y + cell_height], fill=color)

        return image

    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics."""
        return {
            "grid_size": (self.grid_rows, self.grid_cols),
            "tile_size": (self.tile_width, self.tile_height),
            "canvas_size": (self.canvas_width, self.canvas_height),
            "padding": self.padding,
            "background_color": self.background_color,
        }