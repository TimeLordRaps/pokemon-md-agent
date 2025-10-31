# PyTorch CUDA Compatibility Research

## Summary
- **CUDA Version Detected**: 12.9 (Driver 576.02, RTX 4090)
- **PyTorch Compatibility**: CUDA 12.8 wheels work with CUDA 12.9 (backward compatible)
- **Unsloth Version**: Latest git version supports Qwen3-VL models
- **Installation Order**: PyTorch CUDA first, then unsloth
- **Model Selection**: Qwen3-VL-2B-Thinking for balance of speed and reasoning

## Key Findings

### CUDA Compatibility
- RTX 4090 with CUDA 12.9 detected via `nvidia-smi`
- PyTorch 2.9.0 supports CUDA 12.8 (cu128 wheels)
- CUDA 12.9 is backward compatible with CUDA 12.8
- Installation command: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

### Unsloth Support
- Latest unsloth git version supports Qwen3-VL models
- Available models: 2B/4B/8B Instruct and Thinking variants
- Ampere architecture (RTX 4090) requires `ampere` extra
- PyTorch 2.9.0 compatibility: `torch290` extra

### Installation Issues Resolved
- **Problem**: Fresh env installed CPU torch (2.9.0) instead of CUDA
- **Root Cause**: Unsloth brings torch as dependency without CUDA specification
- **Solution**: Install PyTorch CUDA manually before unsloth
- **Updated pyproject.toml**: `unsloth[cu128-ampere-torch290] @ git+https://github.com/unslothai/unsloth.git`

### Model Selection
- **Qwen3-VL-2B-Thinking**: Recommended for agent
  - Small size (2B parameters) for speed
  - Thinking capability for reasoning
  - Vision support for game screenshots
  - FP8 variant for faster inference on long contexts

## Updated Installation Instructions

```bash
# Install PyTorch CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install package
pip install -e .
```

## Installation Results

### ✅ Successful Installation
- **PyTorch**: 2.9.0+cu128 (CUDA 12.8, compatible with CUDA 12.9)
- **CUDA Available**: True
- **GPU Detected**: NVIDIA GeForce RTX 4090
- **Unsloth**: 2025.10.10 (latest version with Qwen3-VL support)
- **FastVisionModel**: Imports successfully

### Package Versions
```
torch                 2.9.0+cu128
torchao               0.14.1
torchaudio            2.9.0+cu128
torchvision           0.24.0+cu128
transformers          4.56.2
accelerate            1.11.0
unsloth               2025.10.10
unsloth_zoo           2025.10.12
```

### Key Success Factors
1. **Manual PyTorch CUDA Install**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
2. **Force Reinstall**: Used `--force-reinstall` to override cached CPU wheels
3. **Correct Unsloth Extras**: `unsloth[cu128-ampere-torch290]` for Ampere GPU architecture
4. **Installation Order**: PyTorch CUDA first, then `pip install -e .`

### Model Selection Confirmed
- **Qwen3-VL-2B-Thinking**: Recommended for agent (2B params, thinking capability, vision support)
- **Available Variants**: 2B/4B/8B Instruct and Thinking models all supported
- **Import Test**: `from unsloth import FastVisionModel` works correctly

## Next Steps
1. Test Qwen3-VL-2B-Thinking model loading and basic inference
2. Implement model router with 2B/4B/8B escalation logic
3. Test vision input processing for Pokemon game screenshots
4. Integrate with agent architecture