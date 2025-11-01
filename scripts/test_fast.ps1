Write-Host "Running: mamba info --envs; python --version; mamba activate agent-hackathon; if (-not (Test-Path 'C:\Homework\agent_hackathon\pokemon-md-agent\pyproject.toml')) { Write-Error 'Not at repo root'; exit 2 }; Set-Location -Path 'C:\Homework\agent_hackathon\pokemon-md-agent'; `$env:PYTHONPATH='C:\Homework\agent_hackathon\pokemon-md-agent\src'; python -m pytest -q --maxfail=1 -m 'not slow and not network and not bench and not longctx'"

mamba info --envs; python --version; mamba activate agent-hackathon;
if (-not (Test-Path 'C:\Homework\agent_hackathon\pokemon-md-agent\pyproject.toml')) { Write-Error 'Not at repo root'; exit 2 }
Set-Location -Path 'C:\Homework\agent_hackathon\pokemon-md-agent';
$env:PYTHONPATH='C:\Homework\agent_hackathon\pokemon-md-agent\src';
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"