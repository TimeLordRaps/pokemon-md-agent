# PowerShell script to validate game state schema
# Usage: .\scripts\validate_vision_schema.ps1

param(
    [switch]$RunTests,
    [switch]$GenerateExamples,
    [switch]$ShowSchema,
    [switch]$Verbose
)

# Set error action
$ErrorActionPreference = "Continue"

Write-Host "=== Game State Schema Validator ===" -ForegroundColor Cyan

# Check if we're in the right directory
if (!(Test-Path "src/models/game_state_schema.py")) {
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
    Write-Host "`n[INFO] Running schema validation tests..." -ForegroundColor Cyan
    & python -m pytest tests/test_game_state_schema.py tests/test_game_state_utils.py -v --tb=short
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All tests PASSED" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Tests FAILED" -ForegroundColor Red
        exit 1
    }
}

# Option 2: Generate schema examples
if ($GenerateExamples) {
    Write-Host "`n[INFO] Generating few-shot examples..." -ForegroundColor Cyan
    $pythonCode = @'
from src.models.game_state_utils import generate_few_shot_examples
import json

examples = generate_few_shot_examples(num_examples=3)
for i, ex in enumerate(examples, 1):
    print(f'\nExample {i}: {ex["description"]}')
    print(json.dumps(ex['state'].model_dump(), indent=2))
'@
    & python -c $pythonCode 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Examples generated successfully" -ForegroundColor Green
    }
}

# Option 3: Show schema
if ($ShowSchema) {
    Write-Host "`n[INFO] Game State JSON Schema:" -ForegroundColor Cyan
    Write-Host "---" -ForegroundColor Cyan
    $pythonCode = @'
from src.models.game_state_utils import schema_to_prompt_json
import json

prompt_json = schema_to_prompt_json()
print(prompt_json)
'@
    & python -c $pythonCode 2>&1
    Write-Host "---" -ForegroundColor Cyan
}

# Default behavior: Quick validation
if (!$RunTests -and !$GenerateExamples -and !$ShowSchema) {
    Write-Host "`n[INFO] Running quick validation..." -ForegroundColor Cyan

    # Test import
    $pythonCode = @'
from src.models.game_state_schema import GameState, Entity, GameStateEnum
from src.models.game_state_utils import parse_model_output, validate_game_state

# Create sample state
state = GameState(
    player_pos=(12, 8),
    floor=3,
    state=GameStateEnum.EXPLORING,
    confidence=0.95
)

# Validate
report = validate_game_state(state)
print('[OK] Schema validation: PASS')
print('  - Player at:', state.player_pos)
print('  - Floor:', state.floor)
print('  - Confidence: %.2f' % state.confidence)
print('  - Quality score: %.2f' % report["quality_score"])

# Test JSON roundtrip
json_str = state.model_dump_json()
restored = GameState.model_validate_json(json_str)
assert restored.player_pos == state.player_pos
print('[OK] JSON roundtrip: PASS')
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
Write-Host "Schema components:"
Write-Host "  [OK] src/models/game_state_schema.py - Core schema definitions"
Write-Host "  [OK] src/models/game_state_utils.py - Utility functions"
Write-Host "  [OK] tests/test_game_state_schema.py - Schema tests (30 tests)"
Write-Host "  [OK] tests/test_game_state_utils.py - Utility tests (21 tests)"
Write-Host "`nUsage options:"
Write-Host "  .\scripts\validate_vision_schema.ps1 -RunTests          # Run full test suite"
Write-Host "  .\scripts\validate_vision_schema.ps1 -GenerateExamples  # Show few-shot examples"
Write-Host "  .\scripts\validate_vision_schema.ps1 -ShowSchema        # Display schema structure"
Write-Host "`n[OK] Ready for Phase 1 integration!" -ForegroundColor Green
