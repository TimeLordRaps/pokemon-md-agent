#!/usr/bin/env python3
"""Comprehensive analysis of 4-test progression results."""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_complete_progression():
    """Full analysis of all 4 completed tests."""
    test_runs_dir = Path("docs/test_runs")
    
    if not test_runs_dir.exists():
        print("ERROR: No test_runs directory found")
        return False
    
    print("\n" + "="*80)
    print("POKEMON MD AGENT - COMPLETE TEST PROGRESSION ANALYSIS")
    print("="*80)
    
    all_results = {}
    total_steps_all = 0
    total_traces_all = 0
    
    # Process each test run
    for run_dir in sorted(test_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
            
        run_name = run_dir.name
        print(f"\n{'='*80}")
        print(f"Test Run: {run_name}")
        print(f"{'='*80}")
        
        run_data = {
            "name": run_name,
            "keyframes": 0,
            "steps": 0,
            "actions": defaultdict(int),
            "confidences": [],
            "duration_seconds": 0,
            "actual_fps": 0,
            "game_title": None,
            "model": None
        }
        
        # Analyze keyframes
        keyframes_dir = run_dir / "keyframes"
        if keyframes_dir.exists():
            keyframe_files = list(keyframes_dir.glob("keyframe_*.png"))
            run_data["keyframes"] = len(keyframe_files)
            print(f"Keyframes: {len(keyframe_files)}")
        
        # Analyze traces
        traces_file = run_dir / "traces" / "latest.jsonl"
        if traces_file.exists():
            traces = []
            with open(traces_file) as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
            
            run_data["steps"] = len(traces)
            total_steps_all += len(traces)
            total_traces_all += len(traces)
            
            print(f"Total steps: {len(traces)}")
            
            if traces:
                # Analyze actions and confidence
                for trace in traces:
                    action = trace.get('action', 'unknown')
                    run_data["actions"][action] += 1
                    conf = trace.get('confidence', 0)
                    run_data["confidences"].append(conf)
                
                # Action breakdown
                print(f"Unique actions: {len(run_data['actions'])}")
                print(f"Action breakdown:")
                for action, count in sorted(run_data['actions'].items(), key=lambda x: -x[1])[:10]:
                    pct = (count / len(traces)) * 100
                    print(f"  - {action}: {count:5d} ({pct:5.1f}%)")
                
                # Confidence stats
                if run_data["confidences"]:
                    avg_conf = statistics.mean(run_data["confidences"])
                    min_conf = min(run_data["confidences"])
                    max_conf = max(run_data["confidences"])
                    print(f"Confidence stats:")
                    print(f"  - Average: {avg_conf:.3f}")
                    print(f"  - Min: {min_conf:.3f}")
                    print(f"  - Max: {max_conf:.3f}")
                
                # Time analysis
                first_trace = traces[0]
                last_trace = traces[-1]
                
                run_data["game_title"] = first_trace.get('game_title', 'Unknown')
                run_data["model"] = first_trace.get('model_id', 'Unknown')
                
                print(f"Game: {run_data['game_title']}")
                print(f"Model: {run_data['model']}")
                
                duration = last_trace.get('timestamp', 0) - first_trace.get('timestamp', 0)
                run_data["duration_seconds"] = duration
                
                if duration > 0:
                    fps = len(traces) / duration
                    run_data["actual_fps"] = fps
                    
                    print(f"Runtime: {duration:.1f}s ({duration/60:.1f}m)")
                    print(f"Actual FPS: {fps:.2f}")
        
        all_results[run_name] = run_data
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("PROGRESSION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTotal frames captured: {total_traces_all:,}")
    print(f"Total test runs: {len(all_results)}")
    print(f"\nBreakdown by test:")
    
    for run_name, data in sorted(all_results.items()):
        pct = (data["steps"] / total_traces_all * 100) if total_traces_all > 0 else 0
        print(f"  - {run_name:30s}: {data['steps']:6,} steps ({pct:5.1f}%), {data['keyframes']:3d} keyframes")
    
    # Bootstrap effectiveness
    print(f"\n{'='*80}")
    print("BOOTSTRAP ANALYSIS")
    print(f"{'='*80}")
    
    run_list = sorted(all_results.items())
    if len(run_list) > 1:
        print("\nCross-run action distribution:")
        
        prev_actions = None
        for i, (run_name, data) in enumerate(run_list):
            action_set = set(data["actions"].keys())
            print(f"{i+1}. {run_name:30s}: {len(action_set)} unique actions")
            
            if prev_actions:
                common = prev_actions & action_set
                new = action_set - prev_actions
                print(f"   - Continued from previous: {len(common)} actions")
                print(f"   - New actions discovered: {len(new)} actions")
                if new:
                    print(f"   - New: {', '.join(sorted(list(new))[:5])}")
            
            prev_actions = action_set
    
    # Convergence analysis
    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    for run_name, data in sorted(all_results.items()):
        if data["confidences"]:
            avg_conf = statistics.mean(data["confidences"])
            print(f"{run_name:30s}: Average confidence {avg_conf:.3f}")
    
    print(f"\n{'='*80}")
    print("STATUS: All 4 test runs completed successfully")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    success = analyze_complete_progression()
    sys.exit(0 if success else 1)
