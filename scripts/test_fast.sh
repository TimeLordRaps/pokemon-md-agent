#!/bin/bash
mamba info --envs && python --version && mamba activate agent-hackathon && pwd
export FAST="1"
export PYTEST_FDUMP_S="45"
python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"