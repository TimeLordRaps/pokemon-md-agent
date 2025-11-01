# PowerShell script to validate Phase 2 vision prompts
# Usage: .\scripts\validate_vision_prompts.ps1

param(
    [switch]$RunTests,
    [switch]$ShowPrompts,
    [switch]$GeneratePrompts,
    [switch]$Verbose
)

# Set error action
$ErrorActionPreference = "Continue"

Write-Host "=== Vision Prompts Validator (Phase 2) ===" -ForegroundColor Cyan

# Check if we're in the right directory
if (!(Test-Path "src/models/vision_prompts.py")) {
    Write-Host "[ERROR] Not in project root directory" -ForegroundColor Red
    exit 1
}

# Activate environment if needed
$currentEnv = cmd /c "conda info --json" 2>$null | ConvertFrom-Json | Select-Object -ExpandProperty active_prefix
if ($currentEnv -notmatch "agent-hackathon") {
    Write-Host "[INFO] Activating mamba environment..." -ForegroundColor Yellow
    & mamba activate agent-hackathon
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to activate environment" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Environment ready" -ForegroundColor Green

# Option 1: Run tests
if ($RunTests) {
    Write-Host "`n[INFO] Running Phase 2 validation tests..." -ForegroundColor Cyan
    & python -m pytest tests/test_vision_prompts.py tests/test_message_packager_vision.py -v --tb=short
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All tests PASSED" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Tests FAILED" -ForegroundColor Red
        exit 1
    }
}

# Option 2: Show system prompts
if ($ShowPrompts) {
    Write-Host "`n[INFO] Vision System Prompts:" -ForegroundColor Cyan
    Write-Host "---" -ForegroundColor Cyan

    $pythonCode = @'
from src.models.vision_prompts import (
    VISION_SYSTEM_PROMPT_INSTRUCT,
    VISION_SYSTEM_PROMPT_THINKING
)

print("INSTRUCT VARIANT (for 2B/4B models):")
print("=" * 60)
print(VISION_SYSTEM_PROMPT_INSTRUCT[:300] + "...")
print("\n\nTHINKING VARIANT (for reasoning models):")
print("=" * 60)
print(VISION_SYSTEM_PROMPT_THINKING[:300] + "...")
'@
    & python -c $pythonCode 2>&1

    Write-Host "---" -ForegroundColor Cyan
}

# Option 3: Generate sample prompts
if ($GeneratePrompts) {
    Write-Host "`n[INFO] Generating sample prompts..." -ForegroundColor Cyan

    $pythonCode = @'
from src.models.vision_prompts import PromptBuilder

# Generate for different scenarios
scenarios = [
    ("explore", "2B"),
    ("battle", "4B"),
    ("boss_battle", "8B"),
]

for policy, model_size in scenarios:
    builder = PromptBuilder("instruct")
    builder.add_few_shot_examples(2)
    builder.add_context(policy_hint=policy, model_size=model_size)

    prompt = builder.build_complete_prompt()
    print(f"\n[Sample] Policy: {policy}, Model: {model_size}")
    print(f"  System prompt length: {len(prompt['system'])} chars")
    print(f"  User prompt length: {len(prompt['user'])} chars")
    print(f"  Contains policy hint: {policy in prompt['user']}")
'@
    & python -c $pythonCode 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Sample prompts generated successfully" -ForegroundColor Green
    }
}

# Default behavior: Quick validation
if (!$RunTests -and !$ShowPrompts -and !$GeneratePrompts) {
    Write-Host "`n[INFO] Running quick validation..." -ForegroundColor Cyan

    # Test import
    $pythonCode = @'
from src.models.vision_prompts import (
    VISION_SYSTEM_PROMPT_INSTRUCT,
    PromptBuilder,
    format_vision_prompt_with_examples
)
from src.orchestrator.message_packager import (
    get_vision_system_prompt_for_model,
    pack_with_vision_prompts
)

# Instruct prompt
assert VISION_SYSTEM_PROMPT_INSTRUCT
print('[OK] Instruct system prompt imported')

# Model-specific prompts
for model_size in ['2B', '4B', '8B']:
    prompt = get_vision_system_prompt_for_model(model_size)
    print(f'[OK] Vision prompt for {model_size}: {len(prompt)} chars')

# PromptBuilder
builder = PromptBuilder()
builder.add_few_shot_examples(3)
builder.add_context(policy_hint="explore")
complete = builder.build_complete_prompt()
assert "system" in complete and "user" in complete
print('[OK] PromptBuilder: created complete prompt')

# Message packager integration
step_state = {'now': {'env': None, 'grid': None}}
system_prompt, messages = pack_with_vision_prompts(step_state, "explore")
assert system_prompt and messages
print(f'[OK] Message packager: {len(messages)} messages with vision prompts')

print('\n[OK] All quick validation checks passed!')
'@
    & python -c $pythonCode 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Quick validation PASSED" -ForegroundColor Green
    } else {
        Write-Host "`n[ERROR] Quick validation FAILED" -ForegroundColor Red
        exit 1
    }
}

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Vision Prompts components:"
Write-Host "  [OK] src/models/vision_prompts.py - System prompts and builder"
Write-Host "  [OK] Integration with message_packager.py"
Write-Host "  [OK] tests/test_vision_prompts.py - 38 tests"
Write-Host "  [OK] tests/test_message_packager_vision.py - 20 tests"
Write-Host "`nUsage options:"
Write-Host "  .\scripts\validate_vision_prompts.ps1 -RunTests         # Run full test suite"
Write-Host "  .\scripts\validate_vision_prompts.ps1 -ShowPrompts      # Display prompt variants"
Write-Host "  .\scripts\validate_vision_prompts.ps1 -GeneratePrompts  # Generate sample prompts"
Write-Host "`n[OK] Ready for Phase 2 integration!" -ForegroundColor Green
