#!/bin/bash

echo "=== Integration Validation ==="

# Run 10-step demo (should complete in <2 min)
python prototypes/mgba_live_test/run_agent_core.py \
  --max-steps 10 \
  --timeout 120

# Check for success markers
if grep -q "perception_success.*true" runs/*/trajectory_*.jsonl; then
  echo "✓ Perception cycles working"
else
  echo "✗ Perception still failing"
  exit 1
fi

if [ $(ls -1 runs/*/screenshots/*.png 2>/dev/null | wc -l) -ge 10 ]; then
  echo "✓ Screenshots captured"
else
  echo "✗ Screenshot capture failed"
  exit 1
fi

echo "=== Integration validation PASSED ==="