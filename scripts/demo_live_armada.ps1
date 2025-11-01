# demo_live_armada.ps1 - Run live armada demo with mGBA setup
# Calls start_mgba.ps1 then runs the multi-model inference loop

param(
    [int]$MaxSteps = 30,
    [switch]$DryRun = $false,
    [string]$DashboardDir = "docs/current"
)

# Set default environment variables if not already set
if (-not $env:PMD_ROM) {
    $env:PMD_ROM = "C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba"
}
if (-not $env:PMD_SAVE) {
    $env:PMD_SAVE = "C:\Homework\agent_hackathon\pokemon-md-agent\config\save_files\game_start_save.ss0"
}
if (-not $env:MGBALUA) {
    $env:MGBALUA = "C:\Homework\agent_hackathon\pokemon-md-agent\src\mgba-harness\mgba-http\mGBASocketServer.lua"
}
if (-not $env:MGBAX) {
    $env:MGBAX = "C:\Program Files\mGBA\mGBA.exe"
}

# Change to project root directory
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Ensure start_mgba.ps1 is called first (unless dry-run)
if (-not $DryRun) {
    Write-Host "Setting up mGBA environment..."
    & "$PSScriptRoot\start_mgba.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start mGBA - aborting demo"
        exit 1
    }
} else {
    Write-Host "Running in DRY-RUN mode (no emulator required)"
}

# Run the live armada demo
Write-Host "Starting live armada (max_steps=$MaxSteps)..."
Write-Host "ROM: $env:PMD_ROM"
Write-Host "Save: $env:PMD_SAVE"
Write-Host "Dashboard: $DashboardDir"

$dryRunFlag = if ($DryRun) { "--dry-run" } else { "" }

mamba info --envs && python --version && mamba activate agent-hackathon && cd pokemon-md-agent && python -m src.runners.live_armada --rom "$env:PMD_ROM" --save "$env:PMD_SAVE" --lua "$env:MGBALUA" --mgba-exe "$env:MGBAX" --dashboard-dir "$DashboardDir" --capture-fps 6 --mem-hz 10 --rate-limit 30 --max-steps $MaxSteps $dryRunFlag --verbose

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Live armada demo failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "✓ Demo complete!"
Write-Host "Outputs written to: $DashboardDir"
Write-Host "  - Images: $DashboardDir\quad_*.png"
Write-Host "  - Traces: $DashboardDir\traces\latest.jsonl"
Write-Host "  - Keyframes: $DashboardDir\keyframes\"
exit 0