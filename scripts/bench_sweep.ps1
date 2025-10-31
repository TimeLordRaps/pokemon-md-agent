mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;
$env:PYTHONPATH="$(pwd)\src";
python profiling/bench_qwen_vl.py --models all --time-budget-s 180 --full