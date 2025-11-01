"""Small helper to inspect and optionally load HF models listed in configs/qwen_vl_models_hf.txt.

This file intentionally avoids heavy downloads by default. Set REAL_MODELS_DRYRUN=0 to allow actual model loads.
Uses Unsloth for supported models to enable 4-bit memory loading with dynamic precision computation.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HF_MODELS_FILE = Path(os.environ.get("HF_MODELS_FILE", ROOT / "configs" / "qwen_vl_models_hf.txt"))

try:
    from unsloth import FastLanguageModel  # type: ignore[import-untyped]
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    FastLanguageModel = None  # type: ignore[assignment,misc]


def read_model_list() -> list[str]:
    if not HF_MODELS_FILE.exists():
        print(f"Model list file not found: {HF_MODELS_FILE}")
        return []
    return [l.strip() for l in HF_MODELS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_unsloth_model(model_id: str) -> bool:
    """Check if the model is a supported Unsloth model."""
    return model_id.startswith("unsloth/Qwen3-VL-") and "unsloth-bnb-4bit" in model_id


def check_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def smoke_load(model_id: str, backend: str | None = None) -> None:
    """Attempt a minimal load of tokenizer/processor to validate credentials and that model exists.

    This will only run if REAL_MODELS_DRYRUN is not set to a truthy value (default is 1 meaning dry-run).
    Uses Unsloth for supported models to enable 4-bit loading with dynamic precision.
    """
    dry = os.environ.get("REAL_MODELS_DRYRUN", "1")
    if dry not in ("0", "false", "False"):
        print("DRY RUN: not actually loading model. Unset REAL_MODELS_DRYRUN or set to 0 to load.")
        return

    # Determine HF cache dir from environment (HF_HOME is honored on Windows/macOS/Linux)
    # Read HF cache dir and sanitize quotes (some Windows envs include surrounding quotes)
    from src.agent.utils import get_hf_cache_dir
    cache_dir = get_hf_cache_dir()

    # Determine backend
    use_unsloth = (backend == "unsloth" or (backend is None and HAS_UNSLOTH and is_unsloth_model(model_id)))

    if use_unsloth and FastLanguageModel:
        print(f"Loading Unsloth model {model_id} in 4-bit inference mode...")
        try:
            # Pass cache_dir when available so HF_HOME is used for caching and local files
            flm_kwargs = dict(
                model_name=model_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            if cache_dir:
                flm_kwargs["cache_dir"] = cache_dir

            model, tokenizer = FastLanguageModel.from_pretrained(**flm_kwargs)
            print("Unsloth model loaded successfully. Vocab size:", getattr(tokenizer, 'vocab_size', 'unknown'))
            return
        except Exception as e:
            print("Failed to load Unsloth model:", e)
            # Fall through to Transformers tokenizer-only check when Unsloth cannot load (e.g., Windows/time_limit)

    # Fallback to standard Transformers for non-Unsloth models or when Unsloth is forced off
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("transformers not installed or import failed:", e)
        return

    print(f"Attempting to load tokenizer for {model_id} (using Transformers)...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
        print("Tokenizer loaded. vocab size:", getattr(tok, 'vocab_size', 'unknown'))
    except Exception as e:
        print("Failed to load tokenizer:", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List models from configs/qwen_vl_models_hf.txt")
    parser.add_argument("--check-token", action="store_true", help="Check HF_TOKEN presence")
    parser.add_argument("--smoke", type=str, help="Run a smoke load for a single model id (only if dryrun disabled)")
    parser.add_argument("--backend", choices=["unsloth", "transformers"], help="Specify model loading backend (default: auto-detect)")
    args = parser.parse_args(argv)

    if args.list:
        models = read_model_list()
        if not models:
            print("No models found in", HF_MODELS_FILE)
            return 1
        print("Models:")
        for m in models:
            print(" -", m)
        return 0

    if args.check_token:
        ok = check_token()
        print("HF token present:" , ok)
        return 0 if ok else 2

    if args.smoke:
        smoke_load(args.smoke, args.backend)
        return 0

    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
