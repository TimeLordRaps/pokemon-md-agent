#!/usr/bin/env python3
"""Analyze completed test progression results."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_test_results():
    """Analyze all completed test runs."""
    test_runs_dir = Path("docs/test_runs")
    
    if not test_runs_dir.exists():
        print("No test_runs directory found")
        return
    
    print("\n" + "="*70)
    print("TEST PROGRESSION ANALYSIS")
    print("="*70)
    
    results = {}
    
    for run_dir in sorted(test_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
            
        run_name = run_dir.name
        print(f"\n[{run_name}]")
        print("-" * 70)
        
        # Count keyframes
        keyframes_dir = run_dir / "keyframes"
        if keyframes_dir.exists():
            keyframe_count = len(list(keyframes_dir.glob("keyframe_*.png")))
            print(f"  Keyframes: {keyframe_count}")
        
        # Analyze JSONL traces
        traces_file = run_dir / "traces" / "latest.jsonl"
        if traces_file.exists():
            traces = []
            with open(traces_file) as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
            
            print(f"  Total steps: {len(traces)}")
            
            if traces:
                # Analyze actions
                actions = defaultdict(int)
                confidences = []
                
                for trace in traces:
                    action = trace.get('action', 'unknown')
                    actions[action] += 1
                    conf = trace.get('confidence', 0)
                    confidences.append(conf)
                
                print(f"  Unique actions: {len(actions)}")
                print(f"  Action breakdown:")
                for action, count in sorted(actions.items(), key=lambda x: -x[1])[:10]:
                    pct = (count / len(traces)) * 100
                    print(f"    - {action}: {count} ({pct:.1f}%)")
                
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    print(f"  Avg confidence: {avg_conf:.2f}")
                
                # Check game progression
                first_trace = traces[0]
                last_trace = traces[-1]
                
                print(f"  Game title: {first_trace.get('game_title', 'Unknown')}")
                print(f"  Model: {first_trace.get('model_id', 'Unknown')}")
                
                duration = last_trace.get('timestamp', 0) - first_trace.get('timestamp', 0)
                if duration > 0:
                    fps = len(traces) / duration if duration > 0 else 0
                    print(f"  Duration: {duration:.1f}s")
                    print(f"  Actual FPS: {fps:.2f}")
        
        results[run_name] = {
            "keyframes": keyframe_count if keyframes_dir.exists() else 0,
            "traces": len(traces) if traces_file.exists() else 0
        }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_frames = sum(r["traces"] for r in results.values())
    total_keyframes = sum(r["keyframes"] for r in results.values())
    
    print(f"\nTotal frames captured: {total_frames}")
    print(f"Total keyframes: {total_keyframes}")
    print(f"Test runs completed: {len(results)}")
    
    for run_name, data in sorted(results.items()):
        print(f"  - {run_name}: {data['traces']} steps, {data['keyframes']} keyframes")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_test_results()
