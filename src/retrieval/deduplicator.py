"""Deduplication utilities using pHash and sprite-hash."""

from typing import Dict, Any, Optional, Set, List, Tuple
import logging
import hashlib
import imagehash
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class Deduplicator:
    """Handles deduplication of content using perceptual hashing."""

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        """Initialize deduplicator.

        Args:
            hash_size: Size of perceptual hash
            highfreq_factor: High frequency factor for pHash
        """
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.seen_hashes: Set[str] = set()
        logger.info(f"Initialized Deduplicator with hash_size={hash_size}")

    def compute_phash(self, image: Image.Image) -> str:
        """Compute perceptual hash for image.

        Args:
            image: PIL Image

        Returns:
            Hex string representation of pHash
        """
        try:
            # Convert to grayscale for consistent hashing
            gray_image = image.convert('L')

            # Compute perceptual hash
            phash = imagehash.phash(gray_image, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor)

            return str(phash)
        except Exception as e:
            logger.error(f"Failed to compute pHash: {e}")
            return ""

    def compute_sprite_hash(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Compute specialized hash for sprites.

        Args:
            image: Sprite image
            metadata: Optional sprite metadata

        Returns:
            Combined hash string
        """
        try:
            # Get perceptual hash
            phash = self.compute_phash(image)

            # Add sprite-specific features
            sprite_features = ""

            # Color palette hash (simplified)
            if image.mode == 'P':
                palette = image.getpalette()
                if palette:
                    palette_hash = hashlib.md5(bytes(palette[:256])).hexdigest()[:8]
                    sprite_features += f"_pal{palette_hash}"

            # Size-based features
            width, height = image.size
            sprite_features += f"_sz{width}x{height}"

            # Metadata-based features
            if metadata:
                species = metadata.get('species', '')
                if species:
                    sprite_features += f"_sp{species[:4]}"

            return f"{phash}{sprite_features}"

        except Exception as e:
            logger.error(f"Failed to compute sprite hash: {e}")
            return ""

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content hash has been seen before.

        Args:
            content_hash: Hash to check

        Returns:
            True if duplicate
        """
        if content_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(content_hash)
        return False

    def compute_text_hash(self, text: str) -> str:
        """Compute hash for text content.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def deduplicate_images(self, images: List[Image.Image], threshold: float = 0.9) -> Tuple[List[Image.Image], List[str]]:
        """Deduplicate list of images using pHash similarity.

        Args:
            images: List of PIL Images
            threshold: Similarity threshold (0-1)

        Returns:
            Tuple of (deduplicated_images, hashes)
        """
        deduplicated = []
        hashes = []

        for img in images:
            phash = self.compute_phash(img)

            # Check similarity with existing images
            is_duplicate = False
            for existing_hash in hashes:
                try:
                    # Compute Hamming distance between hashes
                    distance = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing_hash)
                    similarity = 1 - (distance / (self.hash_size * self.hash_size * 4))  # Normalize

                    if similarity >= threshold:
                        is_duplicate = True
                        break
                except:
                    continue

            if not is_duplicate:
                deduplicated.append(img)
                hashes.append(phash)

        return deduplicated, hashes

    def deduplicate_sprites(
        self,
        sprite_data: List[Tuple[Image.Image, Dict[str, Any]]],
        threshold: float = 0.95
    ) -> Tuple[List[Tuple[Image.Image, Dict[str, Any]]], List[str]]:
        """Deduplicate sprites using specialized sprite hashing.

        Args:
            sprite_data: List of (image, metadata) tuples
            threshold: Similarity threshold

        Returns:
            Tuple of (deduplicated_sprites, hashes)
        """
        deduplicated = []
        hashes = []

        for img, metadata in sprite_data:
            sprite_hash = self.compute_sprite_hash(img, metadata)

            # Check exact match first
            if sprite_hash not in hashes:
                deduplicated.append((img, metadata))
                hashes.append(sprite_hash)

        return deduplicated, hashes

    def batch_deduplicate(
        self,
        items: List[Dict[str, Any]],
        content_type: str = "image",
        threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Batch deduplicate items by content type.

        Args:
            items: List of items with 'content' field
            content_type: Type of content ('image', 'sprite', 'text')
            threshold: Similarity threshold

        Returns:
            Deduplicated list
        """
        if content_type == "image":
            images = [item['content'] for item in items]
            deduplicated_images, hashes = self.deduplicate_images(images, threshold)

            result = []
            for img, h in zip(deduplicated_images, hashes):
                item = next(item for item in items if item['content'] == img)
                item_copy = item.copy()
                item_copy['dedup_hash'] = h
                result.append(item_copy)

            return result

        elif content_type == "sprite":
            sprite_data = [(item['content'], item.get('metadata', {})) for item in items]
            deduplicated_sprites, hashes = self.deduplicate_sprites(sprite_data, threshold)

            result = []
            for (img, metadata), h in zip(deduplicated_sprites, hashes):
                item = next(item for item in items if item['content'] == img)
                item_copy = item.copy()
                item_copy['dedup_hash'] = h
                result.append(item_copy)

            return result

        elif content_type == "text":
            seen_hashes = set()
            result = []

            for item in items:
                text_hash = self.compute_text_hash(item['content'])
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    item_copy = item.copy()
                    item_copy['dedup_hash'] = text_hash
                    result.append(item_copy)

            return result

        else:
            logger.warning(f"Unknown content type: {content_type}")
            return items

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "seen_hashes_count": len(self.seen_hashes),
            "hash_size": self.hash_size,
            "highfreq_factor": self.highfreq_factor,
            "supported_types": ["image", "sprite", "text"],
        }

    def clear(self) -> None:
        """Clear deduplication state."""
        self.seen_hashes.clear()
        logger.info("Cleared deduplicator state")