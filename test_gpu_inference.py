#!/usr/bin/env python3
"""Test GPU availability and model loading."""

import torch
import sys

print("="*70)
print("GPU & MODEL AVAILABILITY CHECK")
print("="*70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("WARNING: CUDA not available - models will run on CPU (very slow)")

print("\nTrying to load model...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "Qwen/Qwen3-VL-2B-Thinking-FP8"
    print(f"Loading: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")
    
    # Try loading tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"✓ Tokenizer loaded")
    
    # Try loading model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print(f"✓ Model loaded on {device}")
    print(f"Model type: {type(model)}")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS: GPU ready for inference")
print("="*70)
