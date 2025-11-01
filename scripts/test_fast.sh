#!/bin/bash
mamba info --envs && python --version && mamba activate agent-hackathon && \
[ -f /c/Homework/agent_hackathon/pokemon-md-agent/pyproject.toml ] || { echo "Not at repo root"; exit 2; } && \
cd /c/Homework/agent_hackathon/pokemon-md-agent && pwd && ls -la && \
export PYTHONPATH=/c/Homework/agent_hackathon/pokemon-md-agent/src && \
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"