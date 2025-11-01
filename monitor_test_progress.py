#!/usr/bin/env python3
"""Monitor test progression without user interaction."""

import time
import subprocess
import sys
from pathlib import Path

log_file = Path("test_run_armada_6models.log")

print("=" * 70)
print("POKEMON MD AGENT - 6-MODEL ARMADA TEST PROGRESSION MONITOR")
print("=" * 70)
print(f"Monitoring log: {log_file.absolute()}")
print(f"Test phases: 15min, 30min, 1hr, 5hr (total ~6.25 hours)")
print(f"Status: Running continuously - no user interaction needed")
print("=" * 70)

last_pos = 0
check_interval = 30  # Check every 30 seconds

while True:
    if log_file.exists():
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
                
            # Look for key progress indicators
            completed_tests = content.count("completed successfully")
            total_steps = len([l for l in content.split('\n') if 'Step' in l])
            
            # Show last few lines
            lines = content.split('\n')
            recent = [l for l in lines[-20:] if l.strip()]
            
            if recent:
                print(f"\n[{time.strftime('%H:%M:%S')}] Progress:")
                print(f"  Tests completed: {completed_tests}/4")
                for line in recent[-5:]:
                    if 'Starting' in line or 'completed' in line or 'Step' in line or 'Skills' in line:
                        print(f"  {line}")
                        
        except Exception as e:
            print(f"  Error reading log: {e}")
    
    time.sleep(check_interval)
