#!/usr/bin/env bash
# Enable use of Hugging Face models for a shell session (WSL/Git Bash)
set -euo pipefail

echo "Enabling real model backend (temporary for this shell)"
export MODEL_BACKEND=hf
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running heavy downloads."
fi
export REAL_MODELS_DRYRUN=1

echo "MODEL_BACKEND=$MODEL_BACKEND"
echo "REAL_MODELS_DRYRUN=${REAL_MODELS_DRYRUN}"
echo "To test credentials run: python -m src.models.real_loader --list"
