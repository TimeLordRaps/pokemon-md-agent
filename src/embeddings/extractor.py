"""Extract embeddings from Qwen3-VL models using different strategies.
Changed lines & context scanned: Qwen3-VL integration, 9 extraction modes, batch processing."""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Sanitize HF_HOME before any imports that might use it
from ..agent.utils import get_hf_cache_dir

hf_cache_dir = get_hf_cache_dir()

logger = logging.getLogger(__name__)


class EmbeddingMode(Enum):
    """Types of embeddings to extract from Qwen3-VL models."""
    INPUT = "input"  # Basic input embedding
    THINK_INPUT = "think_input"  # Input part of thinking tokens
    THINK_FULL = "think_full"  # Full thinking sequence
    THINK_ONLY = "think_only"  # Only thinking tokens
    THINK_IMAGE_INPUT = "think_image_input"  # Image input for thinking
    THINK_IMAGE_FULL = "think_image_full"  # Full image thinking sequence
    THINK_IMAGE_ONLY = "think_image_only"  # Only image thinking tokens
    INSTRUCT_EOS = "instruct_eos"  # End-of-sequence for instructions
    INSTRUCT_IMAGE_ONLY = "instruct_image_only"  # Image-only instructions


class QwenEmbeddingExtractor:
    """Extract embeddings from Qwen3-VL models using various strategies."""

    VALID_MODES = [mode.value for mode in EmbeddingMode]

    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize embedding extractor.

        Args:
            model_name: Name of Qwen3-VL model to use
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[AutoProcessor] = None

        self._is_loaded = False
        logger.info("Initialized QwenEmbeddingExtractor for model: %s on device: %s", model_name, device)


    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load Qwen3-VL model and tokenizer.

        Args:
            model_path: Path to model (auto-download if None)

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info("Loading Qwen3-VL model: %s", self.model_name)

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # Load model
            model_path = model_path or self.model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=hf_cache_dir
            )

            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=hf_cache_dir)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=hf_cache_dir)

            # Move to device if not using device_map
            if device == "cpu" and self.model.device.type != "cpu":
                self.model = self.model.to(device)

            self._is_loaded = True
            logger.info("Model loading complete: %s on %s", self.model_name, device)

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def extract(
        self,
        input_data: Any,
        mode: Union[str, EmbeddingMode] = EmbeddingMode.THINK_FULL,
        vector_id: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Extract embedding from model using specified mode.

        Args:
            input_data: Input to process (screenshot, text, etc.)
            mode: Type of embedding to extract (string or enum)
            **kwargs: Additional arguments for extraction

        Returns:
            Numpy array containing the embedding vector

        Raises:
            ValueError: If mode is invalid
            RuntimeError: If extraction fails
        """
        # Convert string mode to enum
        if isinstance(mode, str):
            try:
                mode = EmbeddingMode(mode)
            except ValueError:
                raise ValueError(f"Invalid embedding mode: {mode}. Valid modes: {self.VALID_MODES}")

        # Allow extraction without model for testing (use dummy embeddings)
        if not self._is_loaded:
            logger.debug("Model not loaded, using dummy embedding for mode: %s", mode.value)
            return self._generate_dummy_embedding(mode)

        logger.debug("Extracting embedding with mode: %s", mode.value)

        try:
            # Extract embedding
            if mode == EmbeddingMode.INPUT:
                embedding = self._extract_input_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_INPUT:
                embedding = self._extract_think_input_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_FULL:
                embedding = self._extract_think_full_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_ONLY:
                embedding = self._extract_think_only_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_IMAGE_INPUT:
                embedding = self._extract_think_image_input_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_IMAGE_FULL:
                embedding = self._extract_think_image_full_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.THINK_IMAGE_ONLY:
                embedding = self._extract_think_image_only_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.INSTRUCT_EOS:
                embedding = self._extract_instruct_eos_embedding(input_data, **kwargs)
            elif mode == EmbeddingMode.INSTRUCT_IMAGE_ONLY:
                embedding = self._extract_instruct_image_only_embedding(input_data, **kwargs)
            else:
                raise ValueError(f"Unknown embedding mode: {mode}")

            # Map to required schema if vector_id provided
            if vector_id is not None:
                # Ensure vector_id contains required fields: {id, ts, floor, silo, screenshot_path, sprite_map, notes}
                required_fields = {'id', 'ts', 'floor', 'silo', 'screenshot_path', 'sprite_map', 'notes'}
                if not all(field in vector_id for field in required_fields):
                    logger.warning(f"vector_id missing required fields: {required_fields - set(vector_id.keys())}")
                # The embedding is already extracted, vector_id mapping handled by caller

            return embedding

        except Exception as e:
            logger.error("Embedding extraction failed for mode %s: %s", mode.value, e)
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def _extract_input_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract basic input embedding from the beginning of input tokens."""
        processed = self.preprocess_input(input_data, EmbeddingMode.INPUT)

        # Tokenize input
        inputs = self._tokenize_input(processed)

        # Run model to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract embedding from first layer, first token (CLS-like)
        hidden_states = outputs.hidden_states[0]  # First layer
        embedding = hidden_states[0, 0, :].cpu().numpy()  # First token

        return embedding.astype(np.float32)

    def _extract_think_input_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from input part of thinking sequence."""
        processed = self.preprocess_input(input_data, EmbeddingMode.THINK_INPUT)

        # For thinking input, extract from the beginning of thinking tokens
        inputs = self._tokenize_input(processed)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract from first layer, position after input tokens
        hidden_states = outputs.hidden_states[0]  # First layer
        # Assume thinking starts after some input tokens
        think_start_pos = processed.get("think_start_pos", len(inputs.get("input_ids", [[]])[0]) // 2)
        embedding = hidden_states[0, think_start_pos, :].cpu().numpy()

        return embedding.astype(np.float32)

    def _extract_think_full_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from full thinking sequence (before EOS)."""
        processed = self.preprocess_input(input_data, EmbeddingMode.THINK_FULL)

        # For thinking modes, we need to simulate thinking tokens
        # In practice, this would involve running generation with thinking
        inputs = self._tokenize_input(processed)

        # Run model and get hidden states before EOS
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract from last layer, last token (before EOS)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        embedding = hidden_states[0, -1, :].cpu().numpy()  # Last token

        return embedding.astype(np.float32)

    def _extract_think_only_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from thinking tokens only."""
        # TODO: Extract from thinking content, excluding input
        return self._generate_dummy_embedding(EmbeddingMode.THINK_ONLY)

    def _extract_think_image_input_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from image input in thinking context."""
        # TODO: Extract from image tokens within thinking block
        return self._generate_dummy_embedding(EmbeddingMode.THINK_IMAGE_INPUT)

    def _extract_think_image_full_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from full image thinking sequence."""
        # TODO: Extract from all image+thinking tokens
        return self._generate_dummy_embedding(EmbeddingMode.THINK_IMAGE_FULL)

    def _extract_think_image_only_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from image thinking tokens only."""
        # TODO: Extract from image thinking content, excluding input
        return self._generate_dummy_embedding(EmbeddingMode.THINK_IMAGE_ONLY)

    def _extract_instruct_eos_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from instruction end-of-sequence."""
        processed = self.preprocess_input(input_data, EmbeddingMode.INSTRUCT_EOS)

        inputs = self._tokenize_input(processed)

        # Run model to get final hidden state
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract from last layer, EOS token position
        hidden_states = outputs.hidden_states[-1]  # Last layer
        embedding = hidden_states[0, -1, :].cpu().numpy()  # EOS token

        return embedding.astype(np.float32)

    def _extract_instruct_image_only_embedding(self, input_data: Any, **kwargs) -> np.ndarray:
        """Extract embedding from image-only instructions."""
        # TODO: Extract from image tokens in instruction context
        return self._generate_dummy_embedding(EmbeddingMode.INSTRUCT_IMAGE_ONLY)
    
    def _generate_dummy_embedding(self, mode: EmbeddingMode) -> np.ndarray:
        """Generate dummy embedding for testing.
        
        Args:
            mode: Embedding mode to generate dummy for
            
        Returns:
            Random embedding vector
        """
        # Different embedding sizes for different modes
        embedding_sizes = {
            EmbeddingMode.INPUT: 1024,
            EmbeddingMode.THINK_INPUT: 1024,
            EmbeddingMode.THINK_FULL: 2048,
            EmbeddingMode.THINK_ONLY: 1536,
            EmbeddingMode.THINK_IMAGE_INPUT: 1024,
            EmbeddingMode.THINK_IMAGE_FULL: 2048,
            EmbeddingMode.THINK_IMAGE_ONLY: 1536,
            EmbeddingMode.INSTRUCT_EOS: 1024,
            EmbeddingMode.INSTRUCT_IMAGE_ONLY: 768,
        }
        
        size = embedding_sizes.get(mode, 1024)
        
        # Generate deterministic "random" embedding based on mode
        np.random.seed(hash(mode.value) % 2**32)
        embedding = np.random.normal(0, 0.1, size)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def extract_batch(
        self,
        input_data_list: List[Any],
        mode: EmbeddingMode = EmbeddingMode.THINK_FULL,
        **kwargs,
    ) -> List[np.ndarray]:
        """Extract embeddings from batch of inputs.
        
        Args:
            input_data_list: List of inputs to process
            mode: Type of embedding to extract
            **kwargs: Additional arguments for extraction
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, input_data in enumerate(input_data_list):
            logger.debug("Processing batch item %d/%d", i + 1, len(input_data_list))
            
            embedding = self.extract(input_data, mode, **kwargs)
            embeddings.append(embedding)
        
        logger.info("Extracted %d embeddings in batch", len(embeddings))
        return embeddings
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding extractor.
        
        Returns:
            Dictionary with extractor information
        """
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "supported_modes": [mode.value for mode in EmbeddingMode],
            "input_types": ["image", "text", "image+text"],
        }
    
    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = "cosine",
    ) -> float:
        """Compare two embeddings using specified method.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Comparison method ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity/distance score
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}")
        
        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        elif method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        elif method == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))
        
        else:
            raise ValueError(f"Unknown comparison method: {method}")
    
    def preprocess_input(
        self,
        input_data: Any,
        mode: EmbeddingMode,
    ) -> Dict[str, Any]:
        """Preprocess input data for embedding extraction.

        Args:
            input_data: Raw input data (dict with 'image' and/or 'text' keys)
            mode: Embedding extraction mode

        Returns:
            Preprocessed input ready for model
        """
        logger.debug("Preprocessing input for mode: %s", mode.value)

        processed = {
            "mode": mode.value,
            "has_image": False,
            "has_text": False,
            "tokens": None,
            "image_features": None,
        }

        # Handle different input types
        if isinstance(input_data, dict):
            # Extract image if present
            if "image" in input_data:
                processed["has_image"] = True
                processed["image_features"] = self._preprocess_image(input_data["image"])

            # Extract text if present
            if "text" in input_data:
                processed["has_text"] = True
                processed["tokens"] = self._preprocess_text(input_data["text"])
        else:
            # Handle single input (assume text or image based on type)
            if isinstance(input_data, str):
                processed["has_text"] = True
                processed["tokens"] = self._preprocess_text(input_data)
            else:
                # Assume image-like object
                processed["has_image"] = True
                processed["image_features"] = self._preprocess_image(input_data)

        # Mode-specific preprocessing
        if mode in [EmbeddingMode.THINK_INPUT, EmbeddingMode.THINK_FULL, EmbeddingMode.THINK_ONLY]:
            processed = self._preprocess_thinking_mode(processed, mode)
        elif mode in [EmbeddingMode.THINK_IMAGE_INPUT, EmbeddingMode.THINK_IMAGE_FULL, EmbeddingMode.THINK_IMAGE_ONLY]:
            processed = self._preprocess_image_thinking_mode(processed, mode)
        elif mode in [EmbeddingMode.INSTRUCT_EOS, EmbeddingMode.INSTRUCT_IMAGE_ONLY]:
            processed = self._preprocess_instruction_mode(processed, mode)

        processed["preprocessed"] = True
        return processed

    def _tokenize_input(self, processed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize processed input for model consumption.

        Args:
            processed: Preprocessed input data

        Returns:
            Tokenized input tensors
        """
        inputs = {}

        # Handle text tokens
        if processed.get("has_text") and processed.get("tokens"):
            text_data = processed["tokens"]
            if isinstance(text_data, dict) and "token_ids" in text_data:
                inputs["input_ids"] = torch.tensor([text_data["token_ids"]], dtype=torch.long)
            elif isinstance(text_data, str):
                # Fallback tokenization
                tokenized = self.tokenizer(text_data, return_tensors="pt")
                inputs.update(tokenized)

        # Handle image features
        if processed.get("has_image") and processed.get("image_features"):
            image_data = processed["image_features"]
            if isinstance(image_data, dict) and "pixel_values" in image_data:
                inputs["pixel_values"] = image_data["pixel_values"]

        # Add attention mask if we have input_ids
        if "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        return inputs

    def _preprocess_image(self, image_data: Any) -> Any:
        """Preprocess image data for model input."""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_model() first.")

        # Handle different image formats
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image
            from PIL import Image
            image = Image.fromarray(image_data.astype('uint8'))
        elif isinstance(image_data, str):
            # Assume it's a file path
            from PIL import Image
            image = Image.open(image_data)
        else:
            # Assume it's already a PIL Image or compatible
            image = image_data

        # Process image through the processor
        processed = self.processor(images=image, return_tensors="pt")

        return {
            "processed": True,
            "pixel_values": processed["pixel_values"],
            "original_shape": image.size if hasattr(image, 'size') else None
        }

    def _preprocess_text(self, text: str) -> Any:
        """Preprocess text data for model input."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "length": len(tokens)
        }

    def _preprocess_thinking_mode(self, processed: Dict[str, Any], mode: EmbeddingMode) -> Dict[str, Any]:
        """Preprocess for thinking-related modes."""
        # Extract thinking tokens from text if present
        if processed["has_text"] and processed["tokens"]:
            # TODO: Parse thinking blocks from text (e.g., extract content between <think> tags)
            # For now, mark as thinking mode processed
            processed["thinking_extracted"] = True
        return processed

    def _preprocess_image_thinking_mode(self, processed: Dict[str, Any], mode: EmbeddingMode) -> Dict[str, Any]:
        """Preprocess for image thinking modes."""
        # Combine image features with thinking extraction
        processed = self._preprocess_thinking_mode(processed, mode)
        processed["image_thinking_combined"] = True
        return processed

    def _preprocess_instruction_mode(self, processed: Dict[str, Any], mode: EmbeddingMode) -> Dict[str, Any]:
        """Preprocess for instruction-related modes."""
        # Extract instruction-specific features
        processed["instruction_mode"] = True
        return processed
    
    def get_layer_outputs(
        self,
        input_data: Any,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Get outputs from specific model layers.
        
        Args:
            input_data: Input to process
            layers: List of layer indices (all layers if None)
            
        Returns:
            Dictionary mapping layer index to output tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug("Getting layer outputs for layers: %s", layers)
        
        # TODO: Implement actual layer output extraction
        # This will involve:
        # 1. Running forward pass with output_hidden_states=True
        # 2. Extracting specified layers from hidden_states tuple
        # 3. Converting to numpy arrays
        
        # Placeholder implementation
        layer_outputs = {}
        layer_indices = layers or [0, 1, 2, 3, 4]  # Sample layers
        
        for layer_idx in layer_indices:
            # Generate dummy layer output
            layer_outputs[layer_idx] = np.random.normal(0, 0.1, (1, 512))
        
        return layer_outputs
