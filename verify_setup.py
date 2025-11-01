#!/usr/bin/env python3
"""
Indempotent setup verification script.
Runs at the start of EVERY test run to verify:
1. Emulator executable exists
2. ROM file exists
3. Save file exists
4. Lua script exists
5. All 6 models are available in HF_HOME cache

Must be called before any test execution begins.
"""

import os
import sys
import json
from pathlib import Path

# Get settings from environment
ROM_PATH = os.environ.get("PMD_ROM")
SAVE_PATH = os.environ.get("PMD_SAVE")
LUA_PATH = os.environ.get("MGBALUA")
MGBA_EXE = os.environ.get("MGBAX")
HF_HOME = os.environ.get("HF_HOME", "E:\\transformer_models")

# Set HF_HOME for model loading
os.environ["HF_HOME"] = HF_HOME

print("=" * 70)
print("POKEMON MD AGENT - ENVIRONMENT VERIFICATION")
print("=" * 70)
print()

REQUIRED_MODELS = [
    "Qwen/Qwen3-VL-2B-Thinking-FP8",
    "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
]

all_ok = True

# 1. Check Emulator
print("[1/5] Emulator executable")
if MGBA_EXE and Path(MGBA_EXE).exists():
    print(f"      ✓ Found: {MGBA_EXE}")
else:
    print(f"      ✗ NOT FOUND: {MGBA_EXE}")
    all_ok = False
print()

# 2. Check ROM
print("[2/5] ROM file")
if ROM_PATH and Path(ROM_PATH).exists():
    size_mb = Path(ROM_PATH).stat().st_size / (1024 * 1024)
    print(f"      ✓ Found: {ROM_PATH} ({size_mb:.1f} MB)")
else:
    print(f"      ✗ NOT FOUND: {ROM_PATH}")
    all_ok = False
print()

# 3. Check Save file
print("[3/5] Save file")
if SAVE_PATH and Path(SAVE_PATH).exists():
    size_kb = Path(SAVE_PATH).stat().st_size / 1024
    print(f"      ✓ Found: {SAVE_PATH} ({size_kb:.1f} KB)")
else:
    print(f"      ✗ NOT FOUND: {SAVE_PATH}")
    all_ok = False
print()

# 4. Check Lua script
print("[4/5] Lua script")
if LUA_PATH and Path(LUA_PATH).exists():
    print(f"      ✓ Found: {LUA_PATH}")
else:
    print(f"      ✗ NOT FOUND: {LUA_PATH}")
    all_ok = False
print()

# 5. Check Models in HF_HOME cache
print("[5/5] HF_HOME Model Cache")
print(f"      HF_HOME={HF_HOME}")
hub_dir = Path(HF_HOME) / "hub"
print(f"      Hub directory: {hub_dir}")
print()

models_ok = 0
for i, model_id in enumerate(REQUIRED_MODELS, 1):
    safe_name = model_id.replace("/", "--")
    model_cache_dir = hub_dir / f"models--{safe_name}"

    if model_cache_dir.exists():
        # Check for snapshots
        snapshots_dir = model_cache_dir / "snapshots"
        if snapshots_dir.exists() and list(snapshots_dir.glob("*")):
            print(f"      [{i}/6] ✓ {model_id}")
            models_ok += 1
        else:
            print(f"      [{i}/6] ⚠ {model_id} (cache incomplete)")
    else:
        print(f"      [{i}/6] ✗ {model_id} (not in cache)")

print()
if models_ok < len(REQUIRED_MODELS):
    print(f"      WARNING: Only {models_ok}/{len(REQUIRED_MODELS)} models in cache")
    print(f"      Models will be downloaded on first use (may take 10-30 minutes)")
    all_ok = False
else:
    print(f"      ✓ All {models_ok}/6 models available")
print()

print("=" * 70)
if all_ok and models_ok == len(REQUIRED_MODELS):
    print("✓ SETUP VERIFIED - Ready to start test")
    print("=" * 70)
    sys.exit(0)
elif all_ok and models_ok > 0:
    print("⚠ PARTIAL SETUP - Some models need download on first use")
    print("=" * 70)
    sys.exit(0)  # Allow to proceed, models will download
else:
    print("✗ SETUP INCOMPLETE - Check errors above")
    print("=" * 70)
    sys.exit(1)
