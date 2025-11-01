# Embedding Types Documentation

This document details the various embedding extraction modes supported by the Qwen3-VL model controller.

## Overview

The system supports multiple embedding extraction strategies to capture different aspects of model processing:

## Input Embeddings

- **`input`**: Hidden states captured from the model's input processing
  - Captures the initial representation of the input text/images
  - Useful for similarity matching of raw inputs

## Thinking Model Embeddings

For reasoning-enabled models (Qwen3-VL Thinking variants):

- **`think_input`**: Hidden state at/before `</think>` + input
  - Captures the model's reasoning process combined with input
  - Best for understanding model thought process

- **`think_full`**: Hidden state before `</s>` (full input+output)
  - Complete representation including reasoning and final output
  - Most comprehensive but computationally expensive

- **`think_only`**: Embedding of only `<think>...</think>` block
  - Isolated reasoning content without input/output
  - Useful for analyzing reasoning patterns

- **`think_image_input`**: Like `think_input` but image-only input
  - Specialized for vision reasoning tasks

- **`think_image_full`**: Like `think_full` but image-only input
  - Complete vision + reasoning representation

- **`think_image_only`**: Image-only reasoning (experimental)
  - Pure visual reasoning without text context

## Instruct Model Embeddings

For fast, direct models (Qwen3-VL Instruct variants):

- **`instruct_eos`**: Hidden state at `</s>` token
  - Final output representation
  - Fastest extraction, good for simple tasks

- **`instruct_image_only`**: Image tokens only
  - Vision-focused embeddings without text

## Usage Examples

```python
from src.embeddings.extractor import QwenEmbeddingExtractor

extractor = QwenEmbeddingExtractor(model_name="Qwen3-VL-4B-Thinking")

# Extract different types of embeddings
input_emb = extractor.extract(input_data, mode="input")
reasoning_emb = extractor.extract(input_data, mode="think_only")
final_emb = extractor.extract(input_data, mode="think_full")
```

## Model-Specific Recommendations

- **2B/4B Models**: Use `instruct_eos` for speed, `think_full` for quality
- **8B+ Models**: Use `think_full` for best reasoning capture
- **Vision Tasks**: Use `think_image_*` variants for vision-specific embeddings

## Performance Considerations

- `think_full` provides highest quality but slowest extraction
- `instruct_eos` provides fastest extraction with good quality
- Image-only variants reduce computational overhead for vision tasks

## Cache Strategy

Embeddings are cached per model and input to improve performance:

- Cache key: `model_name + input_hash + mode`
- RAM cache with LRU eviction
- Optional disk spillover for large deployments