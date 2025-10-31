# Sync profiling directories - consolidate root profiling into pokemon-md-agent/profiling
mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;

# Copy from root profiling to project profiling (idempotent)
if (Test-Path "..\profiling") {
    Write-Host "Syncing profiling directories..."
    Copy-Item "..\profiling\*" ".\profiling\" -Recurse -Force -Exclude "__pycache__"
    Write-Host "Sync complete"
} else {
    Write-Host "No root profiling directory found"
}