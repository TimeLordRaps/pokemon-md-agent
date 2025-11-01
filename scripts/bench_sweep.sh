#!/bin/bash

# Default values
TIME_BUDGET_S=${TIME_BUDGET_S:-180}
FULL=${FULL:-false}
PLOT=${PLOT:-false}
CONTEXTS=${CONTEXTS:-"1024,2048,4096,8192,16384,32768"}
BATCHES=${BATCHES:-"1,2,4,8"}
IMAGE_TEXT_RATIOS=${IMAGE_TEXT_RATIOS:-"0,1,2"}

mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls -la && \
export PYTHONPATH="$(pwd)/src" && \

# Default values
TIME_BUDGET_S=${TIME_BUDGET_S:-180}
FULL=${FULL:-false}
CREATE_PLOTS=${CREATE_PLOTS:-false}
CONTEXTS=${CONTEXTS:-"1024,2048,4096,8192,16384,32768"}
BATCHES=${BATCHES:-"1,2,4,8"}
IMAGE_TEXT_RATIOS=${IMAGE_TEXT_RATIOS:-"0,1,2"}

mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls -la && \
export PYTHONPATH="$(pwd)/src" && \

ARGS="--models all --time-budget-s $TIME_BUDGET_S --contexts $CONTEXTS --batches $BATCHES --image-text-ratios $IMAGE_TEXT_RATIOS"

if [ "$FULL" = "true" ]; then
    ARGS="$ARGS --full"
fi

if [ "$CREATE_PLOTS" = "true" ]; then
    ARGS="$ARGS --create-plots"
fi

python profiling/bench_qwen_vl.py $ARGS