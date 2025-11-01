#!/usr/bin/env python3
"""Test correct Qwen3-VL model loading."""

import torch
from transformers import AutoTokenizer
from PIL import Image
import io

print("Testing Qwen3-VL model loading...")
print(f"CUDA available: {torch.cuda.is_available()}")

model_id = "Qwen/Qwen3-VL-2B-Thinking-FP8"

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"✓ Tokenizer loaded")
    
    # Try loading with trust_remote_code for custom model class
    model = torch.jit.script(None)  # Won't work, but testing approach
    
except:
    pass

# Try alternative: Use transformers pipeline for vision tasks
try:
    from transformers import pipeline
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        device=0 if torch.cuda.is_available() else -1,
    )
    print(f"✓ Vision pipeline loaded")
    
    # Test with a simple image
    test_img = Image.new('RGB', (240, 160), color=(73, 109, 137))
    result = pipe(images=test_img, prompt="What do you see?", top_k=1)
    print(f"✓ Inference works: {result}")
    
except Exception as e:
    print(f"Pipeline approach failed: {e}")
    
    # Fallback: Direct model loading
    try:
        model = __import__('transformers').AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"✓ Model loaded with AutoModel.from_pretrained")
    except Exception as e2:
        print(f"✗ Model loading failed: {e2}")

