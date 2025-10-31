# Enabling real Hugging Face models for Pokemon MD Agent

This document explains how to switch the codebase from mocked model calls to real Hugging Face models. Follow these steps carefully — loading models can consume large VRAM and may require additional setup (bitsandbytes, accelerate, GPU drivers).

Prerequisites
- A Hugging Face access token set as `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment.
- Optional: a GPU with appropriate CUDA drivers. For large models, use GPU. Smaller 2B models can run (slow) on CPU if you have enough RAM.
- Optional packages: `transformers`, `accelerate`, `bitsandbytes`, `torch` (compatible with your CUDA or CPU), `safetensors`.

Quick switch (recommended workflow)

1. Confirm you have the HF model list in `configs/qwen_vl_models_hf.txt` (this repo file contains the requested model IDs).
2. Enable real models in your shell for a single run:

```bash
# bash (WSL / Git Bash)
export MODEL_BACKEND=hf
export HF_TOKEN="<your token here>"
# optional: mark dry run to avoid heavy loads during checks
export REAL_MODELS_DRYRUN=1
```

Or in PowerShell:

```powershell
$env:MODEL_BACKEND = 'hf'
$env:HF_TOKEN = '<your token here>'
$env:REAL_MODELS_DRYRUN = '1'
```

3. Run the small smoke loader provided in `src/models/real_loader.py` to validate credentials and list models (won't download unless you remove the DRYRUN). Example:

```bash
python -m src.models.real_loader --list
```

Notes and safety
- By default tests and CI should keep using mocked models. The repo uses unittest.mock extensively for fast unit tests. We recommend marking real-model tests with `@pytest.mark.real_model` (not run by default).
- The `qwen_vl_models_hf.txt` file lists the HF repo IDs. If you prefer local checkpoints, use `configs/qwen_vl_models.txt` which points to local directories on the developer machine.
- For 4-bit / bnb models: install `bitsandbytes` and use `device_map='auto'` + `load_in_4bit=True` patterns in `transformers` or use accelerate for multi-GPU setups.

Resource estimates (very approximate)
- Qwen3-VL-2B (FP8 / quantized): ~6–12 GB VRAM (quantized lower)
- Qwen3-VL-4B: ~12–22 GB VRAM (depends on quantization)
- Qwen3-VL-8B: ~24+ GB VRAM (use 4-bit/bnb quantization for consumer GPUs)

If you want help wiring a specific model loader (bitsandbytes/quantized) or adding a CI gating job that runs a single lightweight generation on a paid GPU runner, tell me which model to prioritize and I will add the loader and a smoke test.
