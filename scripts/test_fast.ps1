mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;
$env:FAST="1";
$env:PYTEST_FDUMP_S="45";
$env:PYTHONPATH="$(pwd)\src";
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"