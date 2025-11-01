#!/bin/bash
# Monitor test progression and run analysis when complete

LOGFILE="test_runs.log"
RESULTS_DIR="docs/test_runs"
ANALYSIS_SCRIPT="analyze_test_results.py"

echo "=== Test Progression Monitoring Started ==="
echo "Log: $LOGFILE"
echo "Results: $RESULTS_DIR"
echo "$(date)"
echo ""

# Function to check if test is complete
check_completion() {
    if [ ! -f "$LOGFILE" ]; then
        return 1
    fi
    
    # Check if we have all 4 test run completion messages
    completed=$(grep -c "TEST PROGRESSION SUMMARY" "$LOGFILE" 2>/dev/null || echo 0)
    return $((4 - completed))
}

# Monitor until completion
while true; do
    # Get current step count
    if [ -f "$LOGFILE" ]; then
        latest_step=$(grep "Step [0-9]" "$LOGFILE" 2>/dev/null | tail -1)
        if [ ! -z "$latest_step" ]; then
            timestamp=$(date "+%Y-%m-%d %H:%M:%S")
            echo "[$timestamp] $latest_step"
        fi
    fi
    
    # Check if tests completed
    check_completion
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== All Tests Completed ==="
        echo "$(date)"
        break
    fi
    
    # Check every 30 seconds
    sleep 30
done

# Run analysis
if [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "=== Running Analysis ==="
    python "$ANALYSIS_SCRIPT"
fi
