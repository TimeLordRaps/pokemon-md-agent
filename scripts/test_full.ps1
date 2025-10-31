mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;
cd "C:\Homework\agent_hackathon\pokemon-md-agent";
Remove-Item Env:FAST -ErrorAction SilentlyContinue; $env:PYTEST_FDUMP_S="90";     $env:PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src";
python -m pytest -q