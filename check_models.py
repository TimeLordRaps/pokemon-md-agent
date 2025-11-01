#!/usr/bin/env python3
"""
Idempotent model verification and download script.
Runs at startup to ensure all 6 models are available in HF_HOME cache.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# Ensure HF_HOME is set
HF_HOME = os.environ.get("HF_HOME", "E:\\transformer_models")
os.environ["HF_HOME"] = HF_HOME

print("=" * 70)
print("POKEMON MD AGENT - MODEL CACHE VERIFICATION")
print("=" * 70)
print(f"HF_HOME: {HF_HOME}")
print()

# All 6 required models
REQUIRED_MODELS = [
    "Qwen/Qwen3-VL-2B-Thinking-FP8",
    "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
]

def get_model_cache_dir(model_id: str) -> Path:
    """Get the expected cache directory for a model."""
    # HuggingFace uses format: models--org--modelname
    safe_name = model_id.replace("/", "--")
    return Path(HF_HOME) / "hub" / f"models--{safe_name}"


def check_model_available(model_id: str) -> Tuple[bool, Path]:
    """Check if a model is available in cache."""
    cache_dir = get_model_cache_dir(model_id)
    exists = cache_dir.exists()
    return exists, cache_dir


def verify_model_integrity(cache_dir: Path) -> bool:
    """Verify model has essential files."""
    required_files = [
        "refs/main",  # or "snapshots/..."
        "blobs" or "snapshots",
    ]
    # At minimum, check snapshots dir exists
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return False

    # Check for at least one snapshot with files
    snapshot_dirs = list(snapshots.glob("*"))
    if not snapshot_dirs:
        return False

    return True


def download_model(model_id: str) -> bool:
    """Download model using huggingface_hub."""
    try:
        print(f"  Downloading {model_id}...")
        from huggingface_hub import snapshot_download

        # Download model to cache
        snapshot_download(
            model_id,
            cache_dir=Path(HF_HOME) / "hub",
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"    ✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return False


def main():
    """Verify and download models as needed."""
    print("Checking 6 required models:")
    print()

    all_available = True

    for i, model_id in enumerate(REQUIRED_MODELS, 1):
        print(f"[{i}/6] {model_id}")

        exists, cache_dir = check_model_available(model_id)

        if exists:
            # Check integrity
            if verify_model_integrity(cache_dir):
                print(f"      ✓ Found in cache: {cache_dir}")
            else:
                print(f"      ⚠ Cache incomplete, re-downloading...")
                if download_model(model_id):
                    print(f"      ✓ Downloaded successfully")
                else:
                    print(f"      ✗ Failed to download")
                    all_available = False
        else:
            print(f"      ⚠ Not in cache, downloading...")
            if download_model(model_id):
                print(f"      ✓ Downloaded successfully")
            else:
                print(f"      ✗ Failed to download")
                all_available = False

        print()

    print("=" * 70)
    if all_available:
        print("✓ ALL MODELS AVAILABLE - Ready to start test")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME MODELS MISSING - Check errors above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
