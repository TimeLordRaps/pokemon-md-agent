#!/bin/bash
mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls
cd "C:\Homework\agent_hackathon\pokemon-md-agent"
unset FAST
export PYTEST_FDUMP_S="90"
export PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src"
python -m pytest -q