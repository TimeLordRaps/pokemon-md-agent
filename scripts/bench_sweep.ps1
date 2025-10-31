mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;
cd "C:\Homework\agent_hackathon\pokemon-md-agent";
$env:PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src";
python profiling/bench_qwen_vl.py --time-budget-s 30 --contexts 1024,2048,4096,8192,16384,32768 --batches 1,2,4,8 --best-of-n 1,2,4 --use-cache on,off --use-pipeline on,off