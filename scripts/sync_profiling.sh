#!/bin/bash
mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls
# Copy from root profiling to project profiling (idempotent)
if [ -d "../profiling" ]; then
    echo "Syncing profiling directories..."
    cp -r ../profiling/* ./profiling/ 2>/dev/null || true
    echo "Sync complete"
else
    echo "No root profiling directory found"
fi