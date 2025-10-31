"""Sprite library for extracting, normalizing, and hashing sprites from GBA memory.

Extracts unique sprites from OAM/VRAM/PALETTE domains, normalizes them for consistent
representation, computes perceptual hashes and CRCs, and exports an atlas with index.json.
"""

import json
import zlib
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import logging

from ..environment.mgba_controller import MGBAController

logger = logging.getLogger(__name__)


@dataclass
class SpriteEntry:
    """Individual sprite entry with metadata and hashes."""
    sprite_id: str
    vram_offset: int
    oam_index: int
    palette_id: int
    width: int
    height: int
    perceptual_hash: str
    crc32: str
    normalized_pixels: bytes
    metadata: Dict[str, Any]


class SpriteLibrary:
    """Library for managing GBA sprite extraction and indexing."""

    def __init__(self, output_dir: Path, controller: Optional[MGBAController] = None):
        """Initialize sprite library.

        Args:
            output_dir: Directory to store sprite atlas and index
            controller: MGBA controller for memory access
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.sprites: Dict[str, SpriteEntry] = {}
        self.index_file = self.output_dir / "index.json"

        # Load existing index if available
        self._load_index()

    def extract_sprites(self) -> List[SpriteEntry]:
        """Extract unique sprites from GBA memory domains.

        Returns:
            List of extracted sprite entries
        """
        if not self.controller:
            logger.warning("No controller available for sprite extraction")
            return []

        try:
            # Read OAM for sprite attributes
            oam_data = self._read_oam()
            # Read VRAM for sprite pixel data
            vram_data = self._read_vram()
            # Read palette data
            palette_data = self._read_palette()

            sprites = []
            for oam_idx, oam_entry in enumerate(oam_data):
                sprite = self._extract_single_sprite(
                    oam_idx, oam_entry, vram_data, palette_data
                )
                if sprite:
                    sprites.append(sprite)

            # Deduplicate and normalize
            unique_sprites = self._deduplicate_sprites(sprites)
            self.sprites.update({s.sprite_id: s for s in unique_sprites})

            logger.info(f"Extracted {len(unique_sprites)} unique sprites")
            return unique_sprites

        except Exception as e:
            logger.error(f"Sprite extraction failed: {e}")
            return []

    def _read_oam(self) -> List[Dict[str, Any]]:
        """Read Object Attribute Memory for sprite attributes."""
        oam_entries = []

        # OAM is 512 bytes, 8 bytes per entry (128 entries max)
        oam_size = 512
        oam_data = self.controller.memory_domain_read_range("OAM", 0, oam_size)

        if not oam_data:
            return []

        for i in range(0, len(oam_data), 8):
            if i + 8 > len(oam_data):
                break

            # Parse OAM entry (GBA format)
            entry_bytes = oam_data[i:i+8]
            y_pos = entry_bytes[0]
            x_pos = entry_bytes[1]
            tile_idx = entry_bytes[2]
            attr0 = entry_bytes[3]
            attr1 = entry_bytes[4]
            attr2 = entry_bytes[5]

            # Extract sprite properties
            shape = (attr0 >> 6) & 0x3
            size = (attr1 >> 6) & 0x3
            palette_bank = (attr2 >> 12) & 0xF

            width, height = self._get_sprite_dimensions(shape, size)

            oam_entries.append({
                'x': x_pos,
                'y': y_pos,
                'tile_idx': tile_idx,
                'width': width,
                'height': height,
                'palette_bank': palette_bank,
                'visible': (attr0 & 0x100) == 0,  # Bit 8: Display
            })

        return oam_entries

    def _read_vram(self) -> bytes:
        """Read Video RAM for sprite tile data."""
        # VRAM is 64KB, sprites typically in upper regions
        vram_size = 16384  # 16KB for sprites
        return self.controller.memory_domain_read_range("VRAM", 0, vram_size) or b''

    def _read_palette(self) -> bytes:
        """Read palette RAM for sprite colors."""
        palette_size = 512  # 256 colors * 2 bytes each
        return self.controller.memory_domain_read_range("PALETTE", 0, palette_size) or b''

    def _get_sprite_dimensions(self, shape: int, size: int) -> Tuple[int, int]:
        """Get sprite width/height from GBA shape/size bits."""
        dimensions = [
            [(8, 8), (16, 16), (32, 32), (64, 64)],  # Square
            [(16, 8), (32, 8), (32, 16), (64, 32)],  # Horizontal
            [(8, 16), (8, 32), (16, 32), (32, 64)],  # Vertical
        ]
        return dimensions[shape][size]

    def _extract_single_sprite(
        self,
        oam_idx: int,
        oam_entry: Dict[str, Any],
        vram_data: bytes,
        palette_data: bytes
    ) -> Optional[SpriteEntry]:
        """Extract and normalize a single sprite."""
        if not oam_entry['visible']:
            return None

        try:
            # Calculate tile data offset in VRAM
            tile_idx = oam_entry['tile_idx']
            palette_bank = oam_entry['palette_bank']
            width, height = oam_entry['width'], oam_entry['height']

            # Extract tile data (4bpp tiles)
            tile_data = self._extract_tile_data(
                vram_data, tile_idx, width, height
            )

            # Apply palette
            pixels = self._apply_palette(tile_data, palette_data, palette_bank)

            # Normalize sprite
            normalized = self._normalize_sprite(pixels, width, height)

            # Compute hashes
            phash = self._compute_perceptual_hash(normalized)
            crc32 = self._compute_crc32(normalized)

            # Create unique ID
            sprite_id = f"sprite_{oam_idx:03d}_{phash[:8]}"

            return SpriteEntry(
                sprite_id=sprite_id,
                vram_offset=tile_idx * 32,  # 32 bytes per 4bpp tile
                oam_index=oam_idx,
                palette_id=palette_bank,
                width=width,
                height=height,
                perceptual_hash=phash,
                crc32=crc32,
                normalized_pixels=normalized,
                metadata={
                    'x_pos': oam_entry['x'],
                    'y_pos': oam_entry['y'],
                    'shape': oam_entry.get('shape', 0),
                    'size': oam_entry.get('size', 0),
                }
            )

        except Exception as e:
            logger.debug(f"Failed to extract sprite {oam_idx}: {e}")
            return None

    def _extract_tile_data(
        self, vram_data: bytes, tile_idx: int, width: int, height: int
    ) -> bytes:
        """Extract tile data from VRAM."""
        tiles_wide = width // 8
        tiles_high = height // 8
        tile_data = b''

        for ty in range(tiles_high):
            for tx in range(tiles_wide):
                tile_offset = tile_idx + ty * 32 + tx  # Assuming 32 tiles per row
                start = tile_offset * 32  # 32 bytes per tile
                end = start + 32
                if end <= len(vram_data):
                    tile_data += vram_data[start:end]

        return tile_data

    def _apply_palette(
        self, tile_data: bytes, palette_data: bytes, palette_bank: int
    ) -> np.ndarray:
        """Apply palette to 4bpp tile data to get RGB pixels."""
        pixels = []

        for byte_idx in range(0, len(tile_data), 1):
            byte_val = tile_data[byte_idx]

            # 4bpp: two pixels per byte
            for nibble in [(byte_val >> 4) & 0xF, byte_val & 0xF]:
                color_idx = nibble + palette_bank * 16
                if color_idx * 2 + 1 < len(palette_data):
                    color_bytes = palette_data[color_idx * 2:color_idx * 2 + 2]
                    rgb555 = int.from_bytes(color_bytes, 'little')

                    # Convert GBA RGB555 to RGB888
                    r = ((rgb555 >> 0) & 0x1F) * 8
                    g = ((rgb555 >> 5) & 0x1F) * 8
                    b = ((rgb555 >> 10) & 0x1F) * 8

                    pixels.extend([r, g, b])

        return np.array(pixels, dtype=np.uint8).reshape(-1, 3)

    def _normalize_sprite(self, pixels: np.ndarray, width: int, height: int) -> bytes:
        """Normalize sprite pixels for consistent hashing."""
        # Convert to PIL Image for processing
        img = Image.fromarray(pixels.reshape(height, width, 3), 'RGB')

        # Resize to standard size for hashing (maintain aspect ratio)
        target_size = (32, 32)
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert back to bytes
        return img.tobytes()

    def _compute_perceptual_hash(self, pixel_data: bytes) -> str:
        """Compute perceptual hash (pHash) of sprite."""
        # Simple DCT-based pHash implementation
        pixels = np.frombuffer(pixel_data, dtype=np.uint8).reshape(32, 32, 3)

        # Convert to grayscale
        gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114])

        # DCT
        dct = np.fft.fft2(gray)
        dct_shift = np.fft.fftshift(dct)

        # Keep low frequencies
        low_freq = dct_shift[:8, :8]

        # Compute median
        median = np.median(low_freq)

        # Create hash
        hash_bits = low_freq > median
        hash_int = 0
        for bit in hash_bits.flatten():
            hash_int = (hash_int << 1) | int(bit)

        return f"{hash_int:016x}"

    def _compute_crc32(self, data: bytes) -> str:
        """Compute CRC32 hash of sprite data."""
        return f"{zlib.crc32(data):08x}"

    def _deduplicate_sprites(self, sprites: List[SpriteEntry]) -> List[SpriteEntry]:
        """Remove duplicate sprites based on perceptual hash."""
        seen_hashes = set()
        unique_sprites = []

        for sprite in sprites:
            if sprite.perceptual_hash not in seen_hashes:
                seen_hashes.add(sprite.perceptual_hash)
                unique_sprites.append(sprite)

        return unique_sprites

    def export_atlas(self) -> None:
        """Export sprite atlas and index.json."""
        # Create atlas image
        atlas_width = 256
        atlas_height = ((len(self.sprites) * 32) + 255) // 256 * 32

        atlas = Image.new('RGBA', (atlas_width, atlas_height), (0, 0, 0, 0))
        sprite_positions = {}

        x, y = 0, 0
        for sprite in self.sprites.values():
            # Convert normalized pixels back to image
            img = Image.frombytes('RGB', (32, 32), sprite.normalized_pixels)
            img = img.convert('RGBA')

            # Add to atlas
            atlas.paste(img, (x, y))
            sprite_positions[sprite.sprite_id] = {'x': x, 'y': y, 'w': 32, 'h': 32}

            x += 32
            if x >= atlas_width:
                x = 0
                y += 32

        # Save atlas
        atlas_path = self.output_dir / "atlas.png"
        atlas.save(atlas_path)
        logger.info(f"Saved sprite atlas to {atlas_path}")

        # Save index
        index_data = {
            'version': '1.0',
            'total_sprites': len(self.sprites),
            'atlas_path': 'atlas.png',
            'atlas_size': {'width': atlas_width, 'height': atlas_height},
            'sprites': {}
        }

        for sprite_id, sprite in self.sprites.items():
            index_data['sprites'][sprite_id] = {
                'atlas_pos': sprite_positions[sprite_id],
                'original_size': {'width': sprite.width, 'height': sprite.height},
                'vram_offset': sprite.vram_offset,
                'oam_index': sprite.oam_index,
                'palette_id': sprite.palette_id,
                'perceptual_hash': sprite.perceptual_hash,
                'crc32': sprite.crc32,
                'metadata': sprite.metadata
            }

        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved sprite index to {self.index_file}")

    def _load_index(self) -> None:
        """Load existing sprite index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)

                for sprite_id, sprite_data in data.get('sprites', {}).items():
                    self.sprites[sprite_id] = SpriteEntry(
                        sprite_id=sprite_id,
                        vram_offset=sprite_data['vram_offset'],
                        oam_index=sprite_data['oam_index'],
                        palette_id=sprite_data['palette_id'],
                        width=sprite_data['original_size']['width'],
                        height=sprite_data['original_size']['height'],
                        perceptual_hash=sprite_data['perceptual_hash'],
                        crc32=sprite_data['crc32'],
                        normalized_pixels=b'',  # Not stored in index
                        metadata=sprite_data['metadata']
                    )

                logger.info(f"Loaded {len(self.sprites)} sprites from index")
            except Exception as e:
                logger.warning(f"Failed to load sprite index: {e}")

    def get_sprite_by_hash(self, phash: str) -> Optional[SpriteEntry]:
        """Find sprite by perceptual hash."""
        for sprite in self.sprites.values():
            if sprite.perceptual_hash == phash:
                return sprite
        return None