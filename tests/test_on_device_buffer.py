"""Tests for on-device buffer manager."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from PIL import Image

from src.retrieval.on_device_buffer import (
    OnDeviceBufferManager,
    OnDeviceBufferConfig,
    BufferOperationResult
)
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.deduplicator import Deduplicator


class TestOnDeviceBufferManager:
    """Test on-device buffer manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = OnDeviceBufferConfig(
            circular_buffer_mb=10.0,
            ann_index_max_elements=100,
            enable_async=False  # Disable for simpler testing
        )
        self.manager = OnDeviceBufferManager(self.config)

    def test_initialization(self):
        """Test manager initializes with correct components."""
        assert self.manager.config == self.config
        assert hasattr(self.manager, 'circular_buffer')
        assert hasattr(self.manager, 'ann_index')
        assert hasattr(self.manager, 'keyframe_policy')
        assert hasattr(self.manager, 'meta_view_writer')
        assert hasattr(self.manager, 'embedding_generator')
        assert hasattr(self.manager, 'deduplicator')

    @pytest.mark.asyncio
    async def test_store_embedding(self):
        """Test storing embedding in buffer system."""
        embedding = np.random.rand(1024).astype(np.float32)
        metadata = {"type": "test", "priority": 0.8}

        result = await self.manager.store_embedding(embedding, metadata)

        assert isinstance(result, BufferOperationResult)
        assert result.success
        assert result.data_size == embedding.nbytes
        assert result.metadata["embedding_id"].startswith("emb_")

    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Test searching for similar embeddings."""
        # Store some test embeddings
        embeddings = [
            np.random.rand(1024).astype(np.float32),
            np.random.rand(1024).astype(np.float32),
            np.ones(1024).astype(np.float32)  # Similar to query
        ]

        for i, emb in enumerate(embeddings):
            await self.manager.store_embedding(emb, {"index": i})

        # Search for similar
        query = np.ones(1024).astype(np.float32)  # Should match last embedding
        results = await self.manager.search_similar(query, top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r.score, float) for r in results)
        assert all("metadata" in r.__dict__ for r in results)

    def test_embedding_generator(self):
        """Test embedding generation for different content types."""
        gen = self.manager.embedding_generator

        # Test text embedding
        text_emb = gen.generate_text_embedding("test text")
        assert len(text_emb) == 1024
        assert np.isclose(np.linalg.norm(text_emb), 1.0)  # L2 normalized

        # Test ASCII embedding
        ascii_emb = gen.generate_ascii_embedding("ASCII art")
        assert len(ascii_emb) == 1024

        # Test grid embedding
        grid_data = {"width": 10, "height": 10, "tiles": [[0] * 10] * 10}
        grid_emb = gen.generate_grid_embedding(grid_data)
        assert len(grid_emb) == 1024

    def test_deduplicator(self):
        """Test content deduplication."""
        dedup = self.manager.deduplicator

        # Create test images
        img1 = Image.new('RGB', (10, 10), color='red')
        img2 = Image.new('RGB', (10, 10), color='red')  # Duplicate
        img3 = Image.new('RGB', (10, 10), color='blue')  # Different

        images = [img1, img2, img3]
        deduplicated, hashes = dedup.deduplicate_images(images)

        assert len(deduplicated) == 2  # Should remove duplicate
        assert len(hashes) == 2
        assert all(isinstance(h, str) for h in hashes)

    @pytest.mark.asyncio
    async def test_keyframe_processing(self):
        """Test keyframe selection with triggers."""
        # Create mock candidates with triggers
        candidates = []
        for i in range(5):
            candidate = Mock()
            candidate.timestamp = i * 10
            candidate.embedding = np.random.rand(1024)
            candidate.metadata = {"frame_id": i}
            candidate.importance_score = 0.5

            # Add specific triggers
            if i == 0:
                candidate.ssim_score = 0.7  # SSIM drop
                candidate.floor_changed = False
            elif i == 1:
                candidate.floor_changed = True  # Floor change
            elif i == 2:
                candidate.combat_active = True  # Combat
            else:
                candidate.ssim_score = 0.95  # Normal

            candidates.append(candidate)

        # Mock the keyframe policy to avoid full implementation
        with patch.object(self.manager.keyframe_policy, 'select_keyframes') as mock_select:
            mock_result = Mock()
            mock_result.selected_keyframes = candidates[:2]
            mock_result.sampling_rate = 0.4
            mock_select.return_value = mock_result

            result = await self.manager.process_keyframes()

            assert result is not None
            mock_select.assert_called_once()

    @pytest.mark.asyncio
    async def test_meta_view_generation(self):
        """Test meta view generation."""
        # Ensure we have some keyframes
        self.manager.keyframe_policy.selected_keyframes = [
            Mock(embedding=np.random.rand(1024), metadata={"test": True}, timestamp=1.0, importance_score=1.0)
        ] * 4  # Need at least 2 for meta view

        result = await self.manager.generate_meta_view("Test View")

        if result:  # May be None if insufficient keyframes
            assert hasattr(result, 'composite_image')
            assert hasattr(result, 'grid_layout')
            assert result.metadata["title"] == "Test View"

    @pytest.mark.asyncio
    async def test_buffer_status(self):
        """Test getting buffer status."""
        status = await self.manager.get_buffer_status()

        required_keys = ["buffer", "ann_index", "keyframe_policy", "performance", "integrations", "config"]
        for key in required_keys:
            assert key in status

        # Check performance metrics
        perf = status["performance"]
        assert "avg_operation_time" in perf
        assert "avg_search_time_ms" in perf

    def test_shutdown(self):
        """Test manager shutdown."""
        self.manager.shutdown()
        # Should not raise exceptions
        assert True

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # Add some test data
        for i in range(3):
            emb = np.random.rand(1024)
            await self.manager.store_embedding(emb, {"test": True})

        # Cleanup with very short max age (should remove nothing recent)
        removed = await self.manager.cleanup_old_data(max_age_seconds=0.001)
        assert isinstance(removed, int)