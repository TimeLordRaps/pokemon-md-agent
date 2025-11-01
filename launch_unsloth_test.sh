#!/bin/bash
echo "Starting Unsloth 6-model armada test with GPU backend..."
echo "Log: test_run_unsloth_armada.log"
echo "This may take 5-15 minutes for initial model downloads"
echo ""

mamba run -n agent-hackathon python launch_armada_test.py 2>&1 | tee test_run_unsloth_armada.log &

sleep 2
echo "Background process started"
echo "Monitor progress with: tail -f test_run_unsloth_armada.log"
