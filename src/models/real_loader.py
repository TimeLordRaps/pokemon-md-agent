"""Small helper to inspect and optionally load HF models listed in configs/qwen_vl_models_hf.txt.

This file intentionally avoids heavy downloads by default. Set REAL_MODELS_DRYRUN=0 to allow actual model loads.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HF_MODELS_FILE = Path(os.environ.get("HF_MODELS_FILE", ROOT / "configs" / "qwen_vl_models_hf.txt"))


def read_model_list() -> list[str]:
    if not HF_MODELS_FILE.exists():
        print(f"Model list file not found: {HF_MODELS_FILE}")
        return []
    return [l.strip() for l in HF_MODELS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def smoke_load(model_id: str) -> None:
    """Attempt a minimal load of tokenizer/processor to validate credentials and that model exists.

    This will only run if REAL_MODELS_DRYRUN is not set to a truthy value (default is 1 meaning dry-run).
    """
    dry = os.environ.get("REAL_MODELS_DRYRUN", "1")
    if dry not in ("0", "false", "False"):
        print("DRY RUN: not actually loading model. Unset REAL_MODELS_DRYRUN or set to 0 to load.")
        return

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("transformers not installed or import failed:", e)
        return

    print(f"Attempting to load tokenizer for {model_id} ...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        print("Tokenizer loaded. vocab size:", getattr(tok, 'vocab_size', 'unknown'))
    except Exception as e:
        print("Failed to load tokenizer:", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List models from configs/qwen_vl_models_hf.txt")
    parser.add_argument("--check-token", action="store_true", help="Check HF_TOKEN presence")
    parser.add_argument("--smoke", type=str, help="Run a smoke load for a single model id (only if dryrun disabled)")
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
        smoke_load(args.smoke)
        return 0

    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
