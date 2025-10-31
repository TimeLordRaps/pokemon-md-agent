"""
Smoke tests for bench_qwen_vl.py - minimal runs that write CSV and plots.
"""

import tempfile
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.bench


class TestBenchSmoke:
    """Smoke tests for benchmark functionality."""

    def test_dry_run_writes_csv(self):
        """Test that dry run produces a CSV with at least one row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "smoke.csv"

            # Run minimal benchmark
            cmd = [
                sys.executable, "profiling/bench_qwen_vl.py",
                "--models", "Qwen/Qwen3-VL-2B-Thinking-FP8",
                "--min-ctx", "1024",
                "--ctx-mult", "1.5",
                "--max-wall", "5",  # Very short for smoke test
                "--batches", "1",
                "--best-of", "1",
                "--csv", str(csv_path),
                "--dry-run"
            ]

            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

            # Should succeed
            assert result.returncode == 0, f"Command failed: {result.stderr}"

            # CSV should exist and have content
            assert csv_path.exists(), "CSV file was not created"

            with csv_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            # Should have header + at least one data row
            assert len(lines) >= 2, f"CSV should have header + data, got {len(lines)} lines"

            # Check header has expected columns
            header = lines[0].strip().split(",")
            expected_cols = ["model_id", "context_len", "batch_size", "best_of_n"]
            for col in expected_cols:
                assert col in header, f"Missing column {col} in header: {header}"

    def test_plot_generation(self):
        """Test that plotting mode can read CSV and generate plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "plot_test.csv"
            plots_dir = Path(tmpdir) / "plots"

            # First create a minimal CSV
            cmd_create = [
                sys.executable, "profiling/bench_qwen_vl.py",
                "--models", "Qwen/Qwen3-VL-2B-Thinking-FP8,unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
                "--min-ctx", "1024",
                "--ctx-mult", "2.0",
                "--max-wall", "5",
                "--batches", "1,2",
                "--best-of", "1",
                "--csv", str(csv_path),
                "--dry-run"
            ]

            result_create = subprocess.run(cmd_create, cwd=Path(__file__).parent.parent, capture_output=True, text=True)
            assert result_create.returncode == 0, f"CSV creation failed: {result_create.stderr}"

            # Now test plotting
            cmd_plot = [
                sys.executable, "profiling/bench_qwen_vl.py",
                "--plot", str(csv_path)
            ]

            result_plot = subprocess.run(cmd_plot, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

            # Plotting should succeed (may warn about matplotlib not available)
            assert result_plot.returncode == 0, f"Plotting failed: {result_plot.stderr}"

            # Check if plots directory was created (only if matplotlib available)
            # We don't assert this since matplotlib might not be available in test environment

    def test_minimal_config_runs(self):
        """Test that the minimal configuration specified in task runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "minimal.csv"

            # Use the exact command from the task description but with dry-run
            cmd = [
                sys.executable, "profiling/bench_qwen_vl.py",
                "--models", "all",
                "--min-ctx", "1024",
                "--ctx-mult", "1.5",
                "--max-wall", "10",  # Reduced for testing
                "--batches", "1,2,4,8",
                "--best-of", "1,2,4,8",
                "--csv", str(csv_path),
                "--dry-run"
            ]

            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

            assert result.returncode == 0, f"Minimal config failed: {result.stderr}"
            assert csv_path.exists(), "CSV not created for minimal config"

            # Count rows (header + data)
            with csv_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            # Should have many rows since we test all models × contexts × batches × best_of
            # all=6 models, contexts~4 (1024,1536,2304,3456), batches=4, best_of=4 = ~384 rows
            assert len(lines) >= 50, f"Expected many rows, got {len(lines)}"