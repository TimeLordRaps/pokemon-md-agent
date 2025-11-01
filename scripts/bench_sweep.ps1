param(
    [int]$time_budget_s = 180,
    [switch]$full,
    [switch]$create_plots,
    [string]$contexts = "1024,2048,4096,8192,16384,32768",
    [string]$batches = "1,2,4,8",
    [string]$image_text_ratios = "0,1,2"
)

mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls;
$env:PYTHONPATH="$(pwd)\src";

$args = @(
    "--models", "all",
    "--time-budget-s", $time_budget_s,
    "--contexts", $contexts,
    "--batches", $batches,
    "--image-text-ratios", $image_text_ratios
)

if ($full) { $args += "--full" }
if ($create_plots) { $args += "--create-plots" }

python profiling/bench_qwen_vl.py @args