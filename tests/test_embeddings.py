"""Unit tests for embeddings module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.embeddings.extractor import QwenEmbeddingExtractor, EmbeddingMode
from src.embeddings.temporal_silo import TemporalSiloManager, SiloConfig


class TestQwenEmbeddingExtractor:
    """Test QwenEmbeddingExtractor functionality."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = QwenEmbeddingExtractor("test-model")
        assert extractor.model_name == "test-model"
        assert extractor.device == "auto"
        assert not extractor._is_loaded

    def test_valid_modes(self):
        """Test that all embedding modes are valid."""
        extractor = QwenEmbeddingExtractor("test-model")
        assert "input" in extractor.VALID_MODES
        assert "think_full" in extractor.VALID_MODES
        assert "instruct_eos" in extractor.VALID_MODES

    def test_extract_dummy_mode(self):
        """Test extraction with dummy embeddings."""
        extractor = QwenEmbeddingExtractor("test-model")

        # Test different modes produce different sized embeddings
        input_emb = extractor.extract("test", mode="input")
        think_emb = extractor.extract("test", mode="think_full")

        assert isinstance(input_emb, np.ndarray)
        assert isinstance(think_emb, np.ndarray)
        assert input_emb.shape != think_emb.shape  # Different sizes for different modes

    def test_extract_enum_mode(self):
        """Test extraction with enum mode."""
        extractor = QwenEmbeddingExtractor("test-model")

        emb = extractor.extract("test", mode=EmbeddingMode.INPUT)
        assert isinstance(emb, np.ndarray)

    def test_invalid_mode(self):
        """Test invalid mode raises ValueError."""
        extractor = QwenEmbeddingExtractor("test-model")

        with pytest.raises(ValueError, match="Invalid embedding mode"):
            extractor.extract("test", mode="invalid_mode")

    def test_batch_extract(self):
        """Test batch extraction."""
        extractor = QwenEmbeddingExtractor("test-model")

        inputs = ["test1", "test2", "test3"]
        embeddings = extractor.extract_batch(inputs, mode="input")

        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)

    def test_preprocess_input_text(self):
        """Test preprocessing text input."""
        extractor = QwenEmbeddingExtractor("test-model")

        # Mock tokenizer
        extractor.tokenizer = Mock()
        extractor.tokenizer.tokenize.return_value = ["hello", "world"]
        extractor.tokenizer.convert_tokens_to_ids.return_value = [1, 2]

        result = extractor.preprocess_input("hello world", EmbeddingMode.INPUT)

        assert result["has_text"] is True
        assert result["preprocessed"] is True

    def test_preprocess_input_image(self):
        """Test preprocessing image input."""
        extractor = QwenEmbeddingExtractor("test-model")

        # Mock processor
        extractor.processor = Mock()
        # Use numpy array instead of torch tensor for mocking
        extractor.processor.return_value = {"pixel_values": np.random.randn(1, 3, 224, 224)}

        image_data = np.random.rand(224, 224, 3)
        result = extractor.preprocess_input(image_data, EmbeddingMode.INPUT)

        assert result["has_image"] is True
        assert result["preprocessed"] is True

    def test_compare_embeddings(self):
        """Test embedding comparison."""
        extractor = QwenEmbeddingExtractor("test-model")

        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])

        similarity = extractor.compare_embeddings(emb1, emb2, method="cosine")
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_get_embedding_info(self):
        """Test getting embedding info."""
        extractor = QwenEmbeddingExtractor("test-model")

        info = extractor.get_embedding_info()
        assert "model_name" in info
        assert "supported_modes" in info
        assert "input_types" in info


class TestTemporalSiloManager:
    """Test TemporalSiloManager functionality."""

    def test_init_default_silos(self):
        """Test initialization with default 7 silos."""
        manager = TemporalSiloManager()

        assert len(manager.silos) == 7
        expected_silos = [
            "temporal_1frame", "temporal_2frame", "temporal_4frame",
            "temporal_8frame", "temporal_16frame", "temporal_32frame", "temporal_64frame"
        ]
        assert list(manager.silos.keys()) == expected_silos

    def test_store_and_retrieve(self):
        """Test basic store and retrieve."""
        manager = TemporalSiloManager(silos=[1])  # Only 1frame silo

        test_embedding = np.array([0.1, 0.2, 0.3])
        trajectory_id = "test_traj_1"

        # Store
        manager.store(test_embedding, trajectory_id, floor=5)

        # Retrieve recent
        recent = manager.get_recent_trajectories(time_window_seconds=10.0)
        assert "temporal_1frame" in recent
        assert len(recent["temporal_1frame"]) > 0

    def test_cross_silo_search(self):
        """Test cross-silo search."""
        manager = TemporalSiloManager(silos=[1, 2])

        # Store in different silos
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])

        manager.store(emb1, "traj1", floor=1)
        manager.store(emb2, "traj2", floor=2)

        # Search
        results = manager.cross_silo_search(emb1, top_k=2)
        assert len(results) > 0

    def test_composite_index(self):
        """Test composite index functionality."""
        manager = TemporalSiloManager(silos=[1])

        emb = np.array([0.5, 0.5])
        manager.store(emb, "traj1", floor=7)

        # Search by composite index
        results = manager.search_by_composite_index(floor=7)
        assert len(results) > 0

        # Check composite index structure
        entry = results[0]
        assert entry.composite_index == (7, "temporal_1frame", entry.timestamp)

    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        manager = TemporalSiloManager(silos=[1])

        stats = manager.get_memory_usage()
        assert "total_entries" in stats
        assert "total_capacity" in stats
        assert "overall_utilization" in stats

    def test_silo_capacity_limits(self):
        """Test silo capacity limits."""
        config = SiloConfig(
            silo_id="test_silo",
            sample_rate=1000,
            time_span_seconds=1.0,
            max_entries=2
        )

        from src.embeddings.temporal_silo import TemporalSilo
        silo = TemporalSilo(config)

        # Store more than capacity
        for i in range(5):
            silo.store(np.array([float(i)]), float(i), f"traj{i}")

        # Should only keep most recent
        assert len(silo.entries) <= 2


if __name__ == "__main__":
    pytest.main([__file__])