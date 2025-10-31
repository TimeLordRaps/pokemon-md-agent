"""
Tests for bench_qwen_vl.py CLI argument parsing and CSV schema validation.
"""

import argparse
import csv
import tempfile
from pathlib import Path
import pytest

from profiling.bench_qwen_vl import parse_args, BenchmarkResult, write_csv, get_selected_models, build_context_grid

pytestmark = pytest.mark.bench


class TestCLIArgs:
    """Test CLI argument parsing."""

    def test_parse_args_default(self):
        """Test parsing with minimal arguments."""
        args = parse_args(["--csv", "test.csv"])
        assert args.models == "all"
        assert args.min_ctx == 1024
        assert args.ctx_mult == 1.5
        assert args.max_wall == 60
        assert args.batches == "1,2,4,8"
        assert args.best_of == "1,2,4,8"
        assert args.csv == Path("test.csv")
        assert args.plot is None

    def test_parse_args_custom(self):
        """Test parsing with custom arguments."""
        args = parse_args([
            "--models", "Qwen/Qwen3-VL-2B-Thinking-FP8,unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
            "--min-ctx", "512",
            "--ctx-mult", "2.0",
            "--max-wall", "30",
            "--batches", "1,4",
            "--best-of", "1,4",
            "--csv", "output.csv",
            "--plot", "input.csv"
        ])
        assert args.models == "Qwen/Qwen3-VL-2B-Thinking-FP8,unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
        assert args.min_ctx == 512
        assert args.ctx_mult == 2.0
        assert args.max_wall == 30
        assert args.batches == "1,4"
        assert args.best_of == "1,4"
        assert args.csv == Path("output.csv")
        assert args.plot == Path("input.csv")

    def test_get_selected_models_all(self):
        """Test selecting all models."""
        models = get_selected_models("all")
        assert len(models) == 6
        assert "Qwen/Qwen3-VL-2B-Thinking-FP8" in models

    def test_get_selected_models_specific(self):
        """Test selecting specific models."""
        models = get_selected_models("Qwen/Qwen3-VL-2B-Thinking-FP8,unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")
        assert len(models) == 2
        assert "Qwen/Qwen3-VL-2B-Thinking-FP8" in models
        assert "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit" in models

    def test_build_context_grid(self):
        """Test building geometric context grid."""
        contexts = build_context_grid(1024, 1.5, 10000)
        assert contexts[0] == 1024
        assert contexts[1] == 1536  # 1024 * 1.5
        assert contexts[2] == 2304  # 1536 * 1.5
        assert all(c <= 10000 for c in contexts)


class TestCSVSchema:
    """Test CSV output schema and validation."""

    def test_benchmark_result_fields(self):
        """Test BenchmarkResult dataclass has required fields."""
        result = BenchmarkResult(
            model_id="test_model",
            context_len=1024,
            batch_size=1,
            best_of_n=1,
            input_tok_s=100.0,
            output_tok_s=50.0,
            total_tok_s=150.0,
            ttft=0.1,
            wall_clock=5.0,
            cache_hit_rate=0.8,
            used_prompt_cache=True,
            oom=False,
            timeout=False,
            performance_avg=0.75,
            task_max_perf=0.85
        )

        # Check all expected fields exist
        expected_fields = [
            "model_id", "context_len", "batch_size", "best_of_n",
            "input_tok_s", "output_tok_s", "total_tok_s", "ttft",
            "wall_clock", "cache_hit_rate", "used_prompt_cache",
            "oom", "timeout", "performance_avg", "task_max_perf"
        ]

        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_write_csv_creates_file(self):
        """Test that write_csv creates a valid CSV file."""
        results = [
            BenchmarkResult(
                model_id="test_model",
                context_len=1024,
                batch_size=1,
                best_of_n=1,
                input_tok_s=100.0,
                output_tok_s=50.0,
                total_tok_s=150.0,
                ttft=0.1,
                wall_clock=5.0,
                cache_hit_rate=0.8,
                used_prompt_cache=True,
                oom=False,
                timeout=False,
                performance_avg=0.75,
                task_max_perf=0.85
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            write_csv(results, csv_path)

            assert csv_path.exists()

            # Verify CSV content
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]
            assert row["model_id"] == "test_model"
            assert row["context_len"] == "1024"
            assert row["batch_size"] == "1"
            assert row["best_of_n"] == "1"
            assert abs(float(row["input_tok_s"]) - 100.0) < 1e-6
            assert abs(float(row["output_tok_s"]) - 50.0) < 1e-6
            assert abs(float(row["total_tok_s"]) - 150.0) < 1e-6
            assert abs(float(row["performance_avg"]) - 0.75) < 1e-6
            assert abs(float(row["task_max_perf"]) - 0.85) < 1e-6
            assert row["oom"] == "False"
            assert row["timeout"] == "False"

    def test_csv_header_complete(self):
        """Test that CSV has all required columns."""
        results = [
            BenchmarkResult(
                model_id="test",
                context_len=1024,
                batch_size=1,
                best_of_n=1,
                input_tok_s=0.0,
                output_tok_s=0.0,
                total_tok_s=0.0,
                ttft=0.0,
                wall_clock=0.0,
                cache_hit_rate=0.0,
                used_prompt_cache=False,
                oom=False,
                timeout=False,
                performance_avg=0.0,
                task_max_perf=0.0
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            write_csv(results, csv_path)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames

            expected_columns = [
                "model_id", "context_len", "batch_size", "best_of_n",
                "input_tok_s", "output_tok_s", "total_tok_s", "ttft",
                "wall_clock", "cache_hit_rate", "used_prompt_cache",
                "oom", "timeout", "performance_avg", "task_max_perf"
            ]

            for col in expected_columns:
                assert col in header, f"Missing column: {col}"