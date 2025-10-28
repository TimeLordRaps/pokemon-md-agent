"""Embedding generator for ASCII/grid JSON and keyframe images."""

from typing import Dict, Any, Optional, List
import logging
import hashlib
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for ASCII/grid JSON and keyframe images."""

    def __init__(self, vector_dim: int = 1024):
        """Initialize embedding generator.

        Args:
            vector_dim: Dimension of output embeddings
        """
        self.vector_dim = vector_dim
        logger.info(f"Initialized EmbeddingGenerator with vector_dim={vector_dim}")

    def generate_text_embedding(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding from text (ASCII/grid JSON).

        Args:
            text: Text content to embed
            metadata: Optional metadata for context

        Returns:
            Embedding vector
        """
        # Simple hash-based embedding for text content
        # In production, this would use a proper embedding model
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Convert to float array and normalize
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        embedding = embedding / 255.0  # Normalize to [0, 1]

        # Pad or truncate to target dimension
        if len(embedding) < self.vector_dim:
            padding = np.zeros(self.vector_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:self.vector_dim]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def generate_image_embedding(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding from keyframe image.

        Args:
            image: PIL Image to embed
            metadata: Optional metadata for context

        Returns:
            Embedding vector
        """
        # Convert to grayscale and resize for consistency
        gray_image = image.convert('L').resize((64, 64), Image.Resampling.LANCZOS)

        # Convert to numpy array and flatten
        img_array = np.array(gray_image, dtype=np.float32).flatten()

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Pad or truncate to target dimension
        if len(img_array) < self.vector_dim:
            padding = np.zeros(self.vector_dim - len(img_array), dtype=np.float32)
            embedding = np.concatenate([img_array, padding])
        else:
            embedding = img_array[:self.vector_dim]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def generate_ascii_embedding(self, ascii_text: str, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding specifically for ASCII art.

        Args:
            ascii_text: ASCII art text
            metadata: Optional metadata

        Returns:
            Embedding vector
        """
        # Add ASCII-specific prefix for better differentiation
        ascii_content = f"ASCII:{ascii_text}"
        return self.generate_text_embedding(ascii_content, metadata)

    def generate_grid_embedding(self, grid_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding for grid/maze data.

        Args:
            grid_data: Grid data structure
            metadata: Optional metadata

        Returns:
            Embedding vector
        """
        # Serialize grid data to JSON for consistent embedding
        grid_json = json.dumps(grid_data, sort_keys=True)
        grid_content = f"GRID:{grid_json}"
        return self.generate_text_embedding(grid_content, metadata)

    def generate_sprite_embedding(self, sprite_image: Image.Image, sprite_hash: str, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding for sprite with hash-based deduplication.

        Args:
            sprite_image: Sprite image
            sprite_hash: Perceptual hash for deduplication
            metadata: Optional metadata

        Returns:
            Embedding vector
        """
        # Combine image embedding with hash for uniqueness
        image_embedding = self.generate_image_embedding(sprite_image, metadata)

        # Incorporate hash into embedding
        hash_embedding = np.frombuffer(hashlib.sha256(sprite_hash.encode()).digest()[:32], dtype=np.uint8).astype(np.float32) / 255.0

        # Concatenate and normalize
        combined = np.concatenate([image_embedding, hash_embedding])
        if len(combined) > self.vector_dim:
            combined = combined[:self.vector_dim]

        combined = combined / np.linalg.norm(combined)
        return combined

    def batch_generate_embeddings(
        self,
        items: List[Dict[str, Any]],
        content_type: str = "text"
    ) -> List[np.ndarray]:
        """Generate embeddings for batch of items.

        Args:
            items: List of items with 'content' and optional 'metadata'
            content_type: Type of content ('text', 'ascii', 'grid', 'image', 'sprite')

        Returns:
            List of embeddings
        """
        embeddings = []

        for item in items:
            content = item.get('content')
            metadata = item.get('metadata', {})

            if content_type == "text":
                embedding = self.generate_text_embedding(content, metadata)
            elif content_type == "ascii":
                embedding = self.generate_ascii_embedding(content, metadata)
            elif content_type == "grid":
                embedding = self.generate_grid_embedding(content, metadata)
            elif content_type == "image":
                embedding = self.generate_image_embedding(content, metadata)
            elif content_type == "sprite":
                sprite_hash = item.get('sprite_hash', '')
                embedding = self.generate_sprite_embedding(content, sprite_hash, metadata)
            else:
                # Default to text
                embedding = self.generate_text_embedding(str(content), metadata)

            embeddings.append(embedding)

        return embeddings

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        return {
            "vector_dim": self.vector_dim,
            "supported_types": ["text", "ascii", "grid", "image", "sprite"],
            "embedding_method": "hash_based",  # In production: "transformer" or "vision_model"
        }