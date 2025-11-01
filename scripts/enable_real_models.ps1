<# Enable real model backend for PowerShell session #>
Write-Host "Enabling real model backend (PowerShell session)"
$env:MODEL_BACKEND = 'hf'
if (-not $env:HF_TOKEN) {
    Write-Warning "HF_TOKEN is not set. Set it in the environment before running heavy downloads."
}
$env:REAL_MODELS_DRYRUN = '1'
Write-Host "MODEL_BACKEND=$($env:MODEL_BACKEND)"
Write-Host "REAL_MODELS_DRYRUN=$($env:REAL_MODELS_DRYRUN)"
Write-Host "To test credentials run: python -m src.models.real_loader --list"
