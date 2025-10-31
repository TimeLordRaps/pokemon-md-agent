#!/bin/bash
mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls
cd "C:\Homework\agent_hackathon\pokemon-md-agent"
export FAST="1"
export PYTEST_FDUMP_S="45"
export PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src"
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"