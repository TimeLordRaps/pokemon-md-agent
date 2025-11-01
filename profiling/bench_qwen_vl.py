#!/usr/bin/env python3
"""Comprehensive Qwen3-VL benchmarking harness with 3D performance analysis."""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
import math
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw
import random
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:  # pragma: no cover - seaborn optional
    sns = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import plot as plotly_offline_plot
except Exception:  # pragma: no cover - plotly optional
    go = None

# Optional imports for skill schema export (may not be available during development)
try:
    from skills.prompting import compose_system_prompt, build_guidance_schema
    from skills.spec import SKILL_SCHEMA
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    compose_system_prompt = None
    build_guidance_schema = None
    SKILL_SCHEMA = None

# Import QwenController for benchmarking
try:
    from src.agent.qwen_controller import QwenController
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    QwenController = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TinyWoodsFrame:
    """Single Tiny Woods frame entry loaded from manifest."""

    frame_id: int
    action: str
    buttons: Tuple[str, ...]
    hold_ms: int
    wait_ms: int
    seed: int
    captured_at: str
    frame_path: Path
    notes: str

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize frame metadata for logging/results."""
        return {
            "frame_id": self.frame_id,
            "action": self.action,
            "buttons": list(self.buttons),
            "hold_ms": self.hold_ms,
            "wait_ms": self.wait_ms,
            "seed": self.seed,
            "captured_at": self.captured_at,
            "frame_path": str(self.frame_path),
            "notes": self.notes,
        }


class TinyWoodsDataset:
    """Manifest-backed Tiny Woods frame dataset."""

    def __init__(self, root: Path, *, shuffle: bool = True, seed: Optional[int] = None):
        self.root = root
        self.frames_dir = root / "frames"
        self.manifest_path = root / "manifest.jsonl"
        self._records = self._load_manifest()
        if not self._records:
            raise ValueError(f"No Tiny Woods frames found in {self.manifest_path}")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._records)

        self._cursor = 0
        self._validate_counts()

    def _load_manifest(self) -> List[TinyWoodsFrame]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Tiny Woods manifest missing: {self.manifest_path}")

        records: List[TinyWoodsFrame] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on line {line_no} of {self.manifest_path}") from exc

                relative_path = Path(payload["frame_path"])
                if not relative_path.is_absolute():
                    frame_path = (self.root / relative_path).resolve(strict=False)
                else:
                    frame_path = relative_path

                record = TinyWoodsFrame(
                    frame_id=int(payload["frame_id"]),
                    action=str(payload.get("action", "")),
                    buttons=tuple(payload.get("buttons", [])),
                    hold_ms=int(payload.get("hold_ms", 0)),
                    wait_ms=int(payload.get("wait_ms", 0)),
                    seed=int(payload.get("seed", 0)),
                    captured_at=str(payload.get("captured_at", "")),
                    frame_path=frame_path,
                    notes=str(payload.get("notes", "")),
                )
                records.append(record)

        return records

    def _validate_counts(self) -> None:
        missing = [record for record in self._records if not record.frame_path.exists()]
        if missing:
            missing_paths = ", ".join(str(record.frame_path) for record in missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} Tiny Woods frames referenced in manifest but missing on disk "
                f"(e.g., {missing_paths})"
            )

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Tiny Woods frames directory missing: {self.frames_dir}")

        total = len(self._records)
        if total < 50 or total > 200:
            logger.warning(
                "Tiny Woods dataset contains %d frames (recommended range 50-200).", total
            )
        else:
            logger.info("Loaded Tiny Woods dataset with %d frames from %s", total, self.manifest_path)

    def next_batch(self, count: int = 1) -> List[TinyWoodsFrame]:
        """Return the next `count` frames, cycling through the dataset."""
        if count <= 0:
            return []

        selection: List[TinyWoodsFrame] = []
        for _ in range(count):
            selection.append(self._records[self._cursor])
            self._cursor = (self._cursor + 1) % len(self._records)
        return selection

    def load_images(self, frames: List[TinyWoodsFrame]) -> List[Image.Image]:
        """Load PIL images for the provided frames."""
        images: List[Image.Image] = []
        for frame in frames:
            with Image.open(frame.frame_path) as img:
                images.append(img.convert("RGB"))
        return images

    def stats(self) -> Dict[str, Any]:
        """Return simple dataset statistics."""
        return {
            "frame_count": len(self._records),
            "manifest_path": str(self.manifest_path),
            "frames_dir": str(self.frames_dir),
        }

class ComprehensiveQwenVLBenchmark:
    """Comprehensive benchmark harness for Qwen3-VL models with 3D analysis."""

    SUPPORTED_MODELS = [
        "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen3-VL-2B-Thinking-FP8",
        "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    ]

    # Context lengths from 1024 to 262144 (256k) in powers of 2
    CONTEXT_LENGTHS = [1024 * (2 ** i) for i in range(9)]  # 1024, 2048, 4096, ..., 262144

    # Batch sizes to test
    BATCH_SIZES = [1, 2, 4, 8]

    # Micro-benchmark tasks
    TASKS = {
        "text_only": "Summarize this text in one sentence.",
        "vision_simple": "What do you see in this image?",
        "vision_complex": "Analyze this image and describe the tactical situation for a Pokemon Mystery Dungeon game.",
        "mixed_reasoning": "Based on this image, what strategy would you recommend for the Pokemon team?",
    }

    def __init__(
        self,
        output_dir: Path,
        tinywoods_dir: Optional[Path] = None,
        *,
        tinywoods_seed: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir = output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        default_dataset_root = Path(__file__).parent / "data" / "tinywoods"
        dataset_root = tinywoods_dir or default_dataset_root
        self.tinywoods_dataset: Optional[TinyWoodsDataset] = None
        self._tinywoods_stats: Optional[Dict[str, Any]] = None

        try:
            if dataset_root.exists():
                self.tinywoods_dataset = TinyWoodsDataset(dataset_root, seed=tinywoods_seed)
                self._tinywoods_stats = self.tinywoods_dataset.stats()
            else:
                logger.warning(
                    "Tiny Woods dataset directory %s not found; falling back to synthetic images.",
                    dataset_root,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to initialize Tiny Woods dataset from %s (%s). "
                "Vision benchmarks will use synthetic frames.",
                dataset_root,
                exc,
            )
            self.tinywoods_dataset = None

        # Initialize QwenController if available
        if not QWEN_AVAILABLE:
            logger.warning("QwenController not available; benchmarks will use dry-run mode only.")
            self.controller = None
        else:
            try:
                self.controller = QwenController(use_pipeline=False)  # Disable pipeline for now
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to initialize QwenController (%s). "
                    "Benchmarks will use dry-run mode only.",
                    exc,
                )
                self.controller = None

        self.results = []

    def generate_dummy_image(self, width: int = 480, height: int = 320) -> Image.Image:
        """Generate a 480×320 dummy image for vision benchmarks."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw some shapes
        draw.rectangle([50, 50, 200, 150], fill='blue', outline='black')
        draw.ellipse([300, 100, 400, 200], fill='red', outline='black')
        draw.text((width//2 - 50, height//2), "Dummy Image", fill='black')

        return img

    def sample_vision_frames(self, count: int = 1) -> Tuple[List[Image.Image], List[TinyWoodsFrame]]:
        """Fetch Tiny Woods frames (fallback to synthetic if dataset unavailable)."""
        if count <= 0:
            return [], []

        if not self.tinywoods_dataset:
            return [self.generate_dummy_image() for _ in range(count)], []

        try:
            frames = self.tinywoods_dataset.next_batch(count)
            images = self.tinywoods_dataset.load_images(frames)
            return images, frames
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Tiny Woods frame sampling failed (%s); using synthetic imagery.", exc)
            return [self.generate_dummy_image() for _ in range(count)], []

    def get_model_max_context(self, model_name: str) -> int:
        """Get maximum context length for a model."""
        if "2B" in model_name:
            return 32768
        elif "4B" in model_name:
            return 65536
        else:  # 8B
            return 131072

    def get_valid_context_lengths(self, model_name: str) -> List[int]:
        """Get valid context lengths for a model."""
        max_context = self.get_model_max_context(model_name)
        return [cl for cl in self.CONTEXT_LENGTHS if cl <= max_context]

    def get_valid_batch_sizes(self, model_name: str, context_length: int) -> List[int]:
        """Get valid batch sizes for a model and context length."""
        # Estimate based on VRAM requirements
        if "2B" in model_name:
            max_batch = 8
        elif "4B" in model_name:
            max_batch = 4
        else:  # 8B
            max_batch = 2

        # Reduce batch size for longer contexts
        if context_length > 32768:
            max_batch = max(1, max_batch // 2)
        if context_length > 65536:
            max_batch = 1

        return [bs for bs in self.BATCH_SIZES if bs <= max_batch]

    def create_test_prompt(
        self,
        task: str,
        context_length: int,
        frame_meta: Optional[TinyWoodsFrame] = None,
    ) -> str:
        """Create a test prompt of specified length for a task."""
        base_prompt = self.TASKS[task]

        frame_context = ""
        if frame_meta:
            button_text = ", ".join(frame_meta.buttons) if frame_meta.buttons else "none"
            note_text = frame_meta.notes if frame_meta.notes else "no additional notes"
            frame_context = (
                f"\nFrame metadata: action={frame_meta.action}, "
                f"buttons={button_text}, hold_ms={frame_meta.hold_ms}, "
                f"wait_ms={frame_meta.wait_ms}, note={note_text}."
            )

        prompt_prefix = f"{base_prompt}{frame_context}"
        words_needed = max(1, (context_length - len(prompt_prefix)) // 6)  # Rough word count

        # Generate filler text
        filler_words = [
            "pokemon", "mystery", "dungeon", "adventure", "explore", "battle", "team",
            "rescue", "friend", "area", "floor", "stairs", "item", "treasure", "monster",
            "strategy", "tactical", "position", "movement", "action", "decision"
        ]

        filler = " ".join(random.choice(filler_words) for _ in range(words_needed))
        return f"{prompt_prefix} {filler}".strip()

    async def benchmark_single_config(
        self,
        model_name: str,
        context_length: int,
        batch_size: int,
        task: str,
        use_vision: bool,
        max_new_tokens: int = 128,
        num_runs: int = 3,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Benchmark a single configuration with multiple runs."""
        logger.info(f"Benchmarking {model_name} | Context: {context_length} | Batch: {batch_size} | Task: {task}")

        if dry_run:
            # Return mock results for dry run
            return {
                "model": model_name,
                "context_length": context_length,
                "batch_size": batch_size,
                "task": task,
                "vision": use_vision,
                "latency_mean": 1.5 + random.random(),
                "latency_std": 0.1,
                "throughput_mean": 50 + random.random() * 20,
                "throughput_std": 5.0,
                "performance_mean": 0.7 + random.random() * 0.3,
                "performance_std": 0.05,
                "runs": num_runs,
            }

        # Force dry run if controller is not available
        if self.controller is None:
            logger.warning("QwenController not available; using dry-run mode for this configuration.")
            return {
                "model": model_name,
                "context_length": context_length,
                "batch_size": batch_size,
                "task": task,
                "vision": use_vision,
                "latency_mean": 1.5 + random.random(),
                "latency_std": 0.1,
                "throughput_mean": 50 + random.random() * 20,
                "throughput_std": 5.0,
                "performance_mean": 0.7 + random.random() * 0.3,
                "performance_std": 0.05,
                "runs": num_runs,
            }

        # Load model
        variant = "thinking" if "Thinking" in model_name else "instruct"
        handle = self.controller.load_model(model_name, variant)

        # Temporarily set batch size in model router
        original_batch_sizes = {
            '2B': self.controller.model_router.batch_size_2b,
            '4B': self.controller.model_router.batch_size_4b,
            '8B': self.controller.model_router.batch_size_8b,
        }

        # Set test batch size
        if "2B" in model_name:
            self.controller.model_router.batch_size_2b = batch_size
        elif "4B" in model_name:
            self.controller.model_router.batch_size_4b = batch_size
        else:
            self.controller.model_router.batch_size_8b = batch_size

        try:
            latencies = []
            throughputs = []
            performances = []

            used_frames: List[TinyWoodsFrame] = []

            for run in range(num_runs):
                frame_batch: List[TinyWoodsFrame] = []
                images: Optional[List[Image.Image]] = None

                if use_vision:
                    images, frame_batch = self.sample_vision_frames()
                    used_frames.extend(frame_batch)

                prompt = self.create_test_prompt(
                    task,
                    context_length,
                    frame_batch[0] if frame_batch else None,
                )

                # Benchmark
                start_time = time.time()
                try:
                    response, scores = await self.controller.generate_async(
                        prompt,
                        images=images,
                        max_tokens=max_new_tokens,
                        temperature=0.1,
                    )
                    latency = time.time() - start_time
                    tokens_generated = len(response.split())
                    throughput = tokens_generated / latency if latency > 0 else 0

                    # Calculate performance score (simple heuristic)
                    performance = self.calculate_performance_score(response, task)

                    latencies.append(latency)
                    throughputs.append(throughput)
                    performances.append(performance)

                except Exception as e:
                    logger.error(f"Failed run {run}: {e}")
                    latencies.append(0)
                    throughputs.append(0)
                    performances.append(0)
                finally:
                    if images:
                        for image in images:
                            try:
                                image.close()
                            except Exception:
                                pass

            # Calculate statistics
            result = {
                "model": model_name,
                "context_length": context_length,
                "batch_size": batch_size,
                "task": task,
                "vision": use_vision,
                "latency_mean": np.mean(latencies),
                "latency_std": np.std(latencies),
                "throughput_mean": np.mean(throughputs),
                "throughput_std": np.std(throughputs),
                "performance_mean": np.mean(performances),
                "performance_std": np.std(performances),
                "runs": num_runs,
                "tinywoods_frames": json.dumps(
                    [frame.to_metadata() for frame in used_frames]
                ) if used_frames else "[]",
                "tinywoods_frame_count": len(used_frames),
                "tinywoods_dataset": json.dumps(self._tinywoods_stats or {}),
                "vision_image_source": (
                    "tinywoods" if used_frames else ("synthetic" if use_vision else "none")
                ),
            }

            logger.info(".2f")
            return result

        finally:
            # Restore original batch sizes
            self.controller.model_router.batch_size_2b = original_batch_sizes['2B']
            self.controller.model_router.batch_size_4b = original_batch_sizes['4B']
            self.controller.model_router.batch_size_8b = original_batch_sizes['8B']

    def calculate_performance_score(self, response: str, task: str) -> float:
        """Calculate a performance score for the response."""
        if not response or len(response.strip()) == 0:
            return 0.0

        # Simple heuristic scoring based on response characteristics
        score = 0.0

        if task == "text_only":
            # Reward concise summaries
            word_count = len(response.split())
            if 5 <= word_count <= 20:
                score += 0.8
            elif word_count > 50:
                score += 0.3
            else:
                score += 0.5

        elif task in ["vision_simple", "vision_complex"]:
            # Reward descriptive responses
            word_count = len(response.split())
            if word_count >= 10:
                score += 0.6
            if "image" in response.lower() or "see" in response.lower():
                score += 0.4

        elif task == "mixed_reasoning":
            # Reward strategic thinking
            strategic_words = ["strategy", "recommend", "approach", "plan", "tactical"]
            found_words = sum(1 for word in strategic_words if word in response.lower())
            score += min(0.8, found_words * 0.2)

        # Length bonus/penalty
        response_len = len(response)
        if response_len > 10:
            score += 0.2

        return min(1.0, score)

    async def run_comprehensive_benchmarks(
        self,
        models: List[str],
        tasks: List[str],
        max_new_tokens: int = 128,
        num_runs: int = 3,
        dry_run: bool = False,
        max_ctx: int = 262144,
        time_budget_s: int = 30,
        plot: bool = False,
    ) -> None:
        """Run comprehensive benchmarks across all configurations."""
        all_results = []

        for model_name in models:
            if model_name not in self.SUPPORTED_MODELS:
                logger.warning(f"Skipping unsupported model: {model_name}")
                continue

            context_lengths = self.get_valid_context_lengths(model_name)
            # Limit context lengths based on max_ctx
            context_lengths = [c for c in context_lengths if c <= max_ctx]

            for context_length in context_lengths:
                batch_sizes = self.get_valid_batch_sizes(model_name, context_length)

                for batch_size in batch_sizes:
                    for task in tasks:
                        for use_vision in [False, True]:
                            # Skip vision for text-only tasks if not relevant
                            if task == "text_only" and use_vision:
                                continue

                            result = await self.benchmark_single_config(
                                model_name, context_length, batch_size, task,
                                use_vision, max_new_tokens, num_runs, dry_run
                            )
                            all_results.append(result)

        # Save results
        self.save_csv(all_results, "comprehensive_benchmark_results.csv")
        self.save_jsonl(all_results, "comprehensive_benchmark_results.jsonl")

        # Create visualizations only if requested
        if plot:
            self.create_visualizations(all_results)

        logger.info(f"Comprehensive benchmark complete. Results in {self.output_dir}")

    def save_csv(self, results: List[Dict[str, Any]], filename: str) -> None:
        """Save results to CSV."""
        if not results:
            return

        csv_path = self.data_dir / filename
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"CSV written to {csv_path}")

    def save_jsonl(self, results: List[Dict[str, Any]], filename: str) -> None:
        """Save results to JSONL."""
        if not results:
            return

        jsonl_path = self.data_dir / filename
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"JSONL written to {jsonl_path}")

    def create_visualizations(self, results: List[Dict[str, Any]]) -> None:
        """Create comprehensive visualizations."""
        if not results:
            return

        models = sorted({r['model'] for r in results})
        tasks = sorted({r['task'] for r in results})

        # Create 3D throughput plots (interactive when possible)
        self.create_3d_throughput_plot(results, models, tasks)

        # Create performance landscapes with smoothing and annotations
        self.create_performance_landscape(results, models, tasks)

        # Create context-length error bar plots
        self.create_log_context_plots(results, models)

        # Create batch optimization summary
        self.create_batch_optimization_plots(results, models)

        logger.info(f"Visualizations saved to {self.plots_dir}")

    def create_3d_throughput_plot(self, results: List[Dict[str, Any]], models: List[str], tasks: List[str]) -> None:
        """Create 3D throughput plots, interactive when Plotly is available."""
        if go is not None:
            for model in models:
                fig = make_subplots(
                    rows=1,
                    cols=len(tasks),
                    specs=[[{"type": "scene"} for _ in tasks]],
                    subplot_titles=[task.replace("_", " ").title() for task in tasks],
                )

                for col, task in enumerate(tasks, start=1):
                    subset = [r for r in results if r['model'] == model and r['task'] == task]
                    if not subset:
                        continue

                    x_vals = [math.log2(r['context_length']) for r in subset]
                    y_vals = [r['batch_size'] for r in subset]
                    z_vals = [r['throughput_mean'] for r in subset]
                    colors = z_vals
                    hover = [
                        f"context={r['context_length']:,} tokens<br>"
                        f"batch={r['batch_size']}<br>"
                        f"throughput={r['throughput_mean']:.1f} tok/s<br>"
                        f"task={task}<br>"
                        f"vision={'yes' if r['vision'] else 'no'}"
                        for r in subset
                    ]

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_vals,
                            y=y_vals,
                            z=z_vals,
                            mode="markers+lines",
                            marker=dict(size=5, color=colors, colorscale="Viridis", opacity=0.9),
                            line=dict(width=1, color="rgba(80,80,80,0.3)"),
                            hovertext=hover,
                            hoverinfo="text",
                            name=task,
                            showlegend=False,
                        ),
                        row=1,
                        col=col,
                    )

                    fig.update_scenes(
                        xaxis_title="log2(context)",
                        yaxis_title="batch size",
                        zaxis_title="throughput (tok/s)",
                        xaxis=go.layout.scene.XAxis(nticks=5),
                        yaxis=go.layout.scene.YAxis(nticks=6),
                        row=1,
                        col=col,
                    )

                fig.update_layout(
                    title=f"Interactive Throughput - {model}",
                    height=600,
                    width=480 * max(1, len(tasks)),
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                output_file = self.plots_dir / f"{self._slugify(model)}_throughput_interactive.html"
                plotly_offline_plot(fig, filename=str(output_file), auto_open=False, include_plotlyjs="cdn")
        else:
            # Static fallback with improved styling
            fig = plt.figure(figsize=(18, 12))
            subplot_idx = 1
            for model in models:
                for task in tasks:
                    subset = [r for r in results if r['model'] == model and r['task'] == task]
                    if not subset:
                        continue

                    ax = fig.add_subplot(len(models), len(tasks), subplot_idx, projection='3d')
                    subplot_idx += 1

                    x_vals = np.array([math.log2(r['context_length']) for r in subset])
                    y_vals = np.array([r['batch_size'] for r in subset])
                    z_vals = np.array([r['throughput_mean'] for r in subset])

                    sc = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=40, depthshade=True)
                    ax.set_xlabel('log2(context)')
                    ax.set_ylabel('batch size')
                    ax.set_zlabel('throughput (tok/s)')
                    ax.set_title(f"{model.split('/')[-1]}\n{task.replace('_', ' ').title()}")
                    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.05, label='throughput')

            fig.tight_layout()
            fig.savefig(self.plots_dir / "3d_throughput_surfaces.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

    def create_performance_landscape(self, results: List[Dict[str, Any]], models: List[str], tasks: List[str]) -> None:
        """Create smoothed heatmap landscapes for throughput and performance."""
        for model in models:
            fig, axes = plt.subplots(len(tasks), 2, figsize=(14, 4 * len(tasks)), constrained_layout=True)
            if len(tasks) == 1:
                axes = np.array([axes])

            for row, task in enumerate(tasks):
                subset = [r for r in results if r['model'] == model and r['task'] == task]
                if not subset:
                    axes[row, 0].axis('off')
                    axes[row, 1].axis('off')
                    continue

                contexts = sorted({r['context_length'] for r in subset})
                batches = sorted({r['batch_size'] for r in subset})

                throughput_grid = self._build_grid(subset, contexts, batches, "throughput_mean")
                performance_grid = self._build_grid(subset, contexts, batches, "performance_mean")

                throughput_smoothed = self._smooth_grid(throughput_grid)
                performance_smoothed = self._smooth_grid(performance_grid)

                self._plot_heatmap(
                    axes[row, 0],
                    throughput_smoothed,
                    contexts,
                    batches,
                    title=f"{task.replace('_', ' ').title()} - Throughput",
                    cmap="magma",
                    value_fmt="{:.0f}",
                )
                self._plot_heatmap(
                    axes[row, 1],
                    performance_smoothed,
                    contexts,
                    batches,
                    title=f"{task.replace('_', ' ').title()} - Performance",
                    cmap="viridis",
                    value_fmt="{:.2f}",
                )

            fig.suptitle(f"Context/Batch Landscape - {model.split('/')[-1]}", fontsize=16)
            fig.savefig(self.plots_dir / f"{self._slugify(model)}_performance_landscape.png", dpi=300)
            plt.close(fig)

    def create_log_context_plots(self, results: List[Dict[str, Any]], models: List[str]) -> None:
        """Create context length plots with error bars and separate vision/text runs."""
        for model in models:
            model_results = [r for r in results if r['model'] == model]
            model_tasks = sorted({r['task'] for r in model_results})

            fig, axes = plt.subplots(len(model_tasks), 1, figsize=(10, 4 * len(model_tasks)), sharex=True)
            if len(model_tasks) == 1:
                axes = [axes]

            for ax, task in zip(axes, model_tasks):
                task_results = [r for r in model_results if r['task'] == task]
                if not task_results:
                    ax.axis('off')
                    continue

                for vision_flag in sorted({r['vision'] for r in task_results}):
                    vision_results = [r for r in task_results if r['vision'] == vision_flag]
                    label_suffix = "vision" if vision_flag else "text"

                    for batch in sorted({r['batch_size'] for r in vision_results}):
                        batch_results = [r for r in vision_results if r['batch_size'] == batch]
                        batch_results.sort(key=lambda r: r['context_length'])

                        x_vals = [math.log2(r['context_length']) for r in batch_results]
                        y_vals = [r['throughput_mean'] for r in batch_results]
                        y_err = [r.get('throughput_std', 0.0) for r in batch_results]

                        if not x_vals:
                            continue

                        ax.errorbar(
                            x_vals,
                            y_vals,
                            yerr=y_err,
                            label=f"bs {batch} | {label_suffix}",
                            marker='o',
                            capsize=3,
                            linewidth=1.8,
                        )

                ax.set_ylabel("Throughput (tok/s)")
                ax.set_title(task.replace("_", " ").title())
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, ncol=2, loc="upper right")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            axes[-1].set_xlabel("Context Length (log2)")
            fig.suptitle(f"Throughput vs Context - {model.split('/')[-1]}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(self.plots_dir / f"{self._slugify(model)}_throughput_errorbars.png", dpi=300)
            plt.close(fig)

    def create_batch_optimization_plots(self, results: List[Dict[str, Any]], models: List[str]) -> None:
        """Create batch optimization plots highlighting peak throughput."""
        fig, axes = plt.subplots(2, len(models), figsize=(18, 10), squeeze=False)

        for col, model in enumerate(models):
            model_results = [r for r in results if r['model'] == model]
            if not model_results:
                axes[0, col].axis('off')
                axes[1, col].axis('off')
                continue

            context_lengths = sorted({r['context_length'] for r in model_results})
            optimal_batches = []
            optimal_throughputs = []
            optimal_throughput_std = []
            context_used = []

            for context in context_lengths:
                context_results = [r for r in model_results if r['context_length'] == context]
                if not context_results:
                    continue
                best = max(context_results, key=lambda r: r['throughput_mean'])
                optimal_batches.append(best['batch_size'])
                optimal_throughputs.append(best['throughput_mean'])
                optimal_throughput_std.append(best.get('throughput_std', 0.0))
                context_used.append(context)

            if not optimal_batches:
                axes[0, col].axis('off')
                axes[1, col].axis('off')
                continue

            x_vals = [math.log2(cl) for cl in context_used]

            axes[0, col].step(x_vals, optimal_batches, where='mid', linewidth=2, color='steelblue')
            axes[0, col].set_title(f"{model.split('/')[-1]}\nOptimal Batch Size")
            axes[0, col].set_ylabel("Batch Size")
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].set_yticks(sorted({r['batch_size'] for r in model_results}))

            axes[1, col].errorbar(
                x_vals,
                optimal_throughputs,
                yerr=optimal_throughput_std,
                marker='o',
                color='crimson',
                linewidth=2,
                capsize=3,
            )
            axes[1, col].set_xlabel("Context Length (log2)")
            axes[1, col].set_ylabel("Throughput (tok/s)")
            axes[1, col].set_title("Peak Throughput")
            axes[1, col].grid(True, alpha=0.3)

            max_idx = int(np.argmax(optimal_throughputs))
            axes[1, col].annotate(
                f"{optimal_throughputs[max_idx]:.0f} tok/s",
                (x_vals[max_idx], optimal_throughputs[max_idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                color='crimson',
            )

        fig.tight_layout()
        fig.savefig(self.plots_dir / "batch_optimization.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _build_grid(
        self,
        records: List[Dict[str, Any]],
        contexts: List[int],
        batches: List[int],
        key: str,
    ) -> np.ndarray:
        """Build a context/batch grid for the requested metric."""
        grid = np.full((len(batches), len(contexts)), np.nan, dtype=float)
        context_index = {value: idx for idx, value in enumerate(contexts)}
        batch_index = {value: idx for idx, value in enumerate(batches)}

        for record in records:
            c_idx = context_index.get(record['context_length'])
            b_idx = batch_index.get(record['batch_size'])
            if c_idx is None or b_idx is None:
                continue
            metric_value = record.get(key)
            if metric_value is None:
                continue
            grid[b_idx, c_idx] = float(metric_value)

        return grid

    def _smooth_grid(self, grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing while respecting missing values."""
        if not np.any(~np.isnan(grid)):
            return grid

        mask = np.isnan(grid)
        filled = np.where(mask, 0.0, grid)
        weights = np.where(mask, 0.0, 1.0)

        smoothed_values = gaussian_filter(filled, sigma=sigma, mode='nearest')
        smoothed_weights = gaussian_filter(weights, sigma=sigma, mode='nearest')

        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = smoothed_values / smoothed_weights
        smoothed[smoothed_weights == 0] = np.nan
        return smoothed

    def _plot_heatmap(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        contexts: List[int],
        batches: List[int],
        title: str,
        cmap: str,
        value_fmt: str,
    ) -> None:
        """Plot a heatmap with annotations and highlight the peak cell."""
        if not np.any(~np.isnan(grid)):
            ax.axis('off')
            return

        heatmap = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Context Length (log2)")
        ax.set_ylabel("Batch Size")
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels([f"{int(math.log2(cl))}" for cl in contexts], rotation=45, ha='right')
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels([str(b) for b in batches])
        ax.grid(False)

        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.set_ylabel(title.split(" - ")[-1], rotation=-90, va='bottom')

        mean_value = np.nanmean(grid)
        for row_idx, batch in enumerate(batches):
            for col_idx, context in enumerate(contexts):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                color = "white" if value > mean_value else "black"
                ax.text(col_idx, row_idx, value_fmt.format(value), ha='center', va='center', fontsize=8, color=color)

        try:
            max_flat_idx = np.nanargmax(grid)
            max_row, max_col = divmod(max_flat_idx, grid.shape[1])
            ax.scatter([max_col], [max_row], marker='*', s=150, edgecolors='black', linewidths=0.6, color='gold')
        except ValueError:
            # All values nan
            pass

    def _slugify(self, text: str) -> str:
        """Create a filesystem-friendly slug."""
        return "".join(ch if ch.isalnum() else "_" for ch in text)

    def find_inflection_points(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find inflection points in performance curves."""
        inflection_points = {}

        # Group by model
        models = set(r['model'] for r in results)

        for model in models:
            model_results = [r for r in results if r['model'] == model]

            # Find throughput vs context length for batch_size=1
            batch_1_results = [r for r in model_results if r['batch_size'] == 1]
            if len(batch_1_results) >= 3:
                # Sort by context length
                batch_1_results.sort(key=lambda x: x['context_length'])

                context_lengths = [r['context_length'] for r in batch_1_results]
                throughputs = [r['throughput_mean'] for r in batch_1_results]

                # Calculate second derivatives to find inflection points
                if len(throughputs) >= 3:
                    # Simple finite difference approximation
                    second_deriv = []
                    for i in range(1, len(throughputs) - 1):
                        d2 = throughputs[i+1] - 2*throughputs[i] + throughputs[i-1]
                        second_deriv.append((context_lengths[i], d2))

                    # Find points where second derivative changes sign
                    inflections = []
                    for i in range(1, len(second_deriv)):
                        if second_deriv[i-1][1] * second_deriv[i][1] < 0:
                            inflections.append(second_deriv[i][0])

                    inflection_points[model] = inflections

        return inflection_points


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to benchmark: 'all' or comma-separated list (e.g., 'Qwen/Qwen3-VL-2B-Thinking-FP8,unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit')"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Tasks to benchmark: 'all' or comma-separated list matching ComprehensiveQwenVLBenchmark.TASKS."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling/output"),
        help="Directory to store benchmark outputs (CSV, plots, trajectories)."
    )

    parser.add_argument(
        "--tinywoods-dir",
        type=Path,
        default=None,
        help="Tiny Woods dataset root containing manifest.jsonl and frames/. Defaults to profiling/data/tinywoods.",
    )

    parser.add_argument(
        "--tinywoods-seed",
        type=int,
        default=None,
        help="Optional shuffle seed for Tiny Woods frame ordering.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate during inference benchmarks."
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of repeated runs per configuration."
    )

    parser.add_argument(
        "--min-ctx",
        type=int,
        default=1024,
        help="Minimum context length (default: 1024)"
    )

    parser.add_argument(
        "--ctx-mult",
        type=float,
        choices=[1.5, 2.0],
        default=1.5,
        help="Context length multiplier (default: 1.5)"
    )

    parser.add_argument(
        "--max-wall",
        type=int,
        default=60,
        help="Maximum wall clock time per benchmark (seconds, default: 60)"
    )

    parser.add_argument(
        "--time-budget-s",
        type=int,
        default=30,
        help="Time budget for benchmark runs (seconds, default: 30)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite including heavy sweeps"
    )

    parser.add_argument(
        "--contexts",
        type=str,
        default="4096,8192,16384",
        help="Context lengths to test (comma-separated, default: '4096,8192,16384')"
    )

    parser.add_argument(
        "--batches",
        type=str,
        default="1,2,4",
        help="Batch sizes to test (comma-separated, default: '1,2,4')"
    )

    parser.add_argument(
        "--best-of-n",
        type=str,
        default="1,2,4",
        help="Best-of-n values to test (comma-separated, default: '1,2,4')"
    )

    parser.add_argument(
        "--use-cache",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable/disable prompt caching (default: on)"
    )

    parser.add_argument(
        "--use-pipeline",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable/disable request pipelining (default: on)"
    )

    parser.add_argument(
        "--image-text-ratios",
        type=str,
        default="0.1,0.25,0.5,0.75",
        help="Image to text ratios to test (comma-separated floats, default: '0.1,0.25,0.5,0.75')"
    )

    parser.add_argument(
        "--reasoning-budgets",
        type=str,
        default=None,
        help="Reasoning budgets to test (comma-separated: low,med,high). Maps to internal chain params if available, else noop."
    )

    parser.add_argument(
        "--plot",
        type=Path,
        help="CSV file to plot from (generates plots in profiling/plots/)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic timings instead of real inference"
    )

    parser.add_argument(
        "--export-skill-schema",
        type=Path,
        help="Write the Python skill JSON schema to the given path."
    )

    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )

    parser.add_argument(
        "--telemetry-log",
        type=Path,
        help="JSONL telemetry log file path"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create visualization plots during benchmarking (slower)"
    )

    return parser.parse_args(argv)


class BenchmarkResult:
    """Result of a single benchmark run."""
    def __init__(self, model_id: str, context_len: int, batch_size: int, best_of_n: int,
                 input_tok_s: float, output_tok_s: float, total_tok_s: float,
                 ttft: float, wall_clock: float, cache_hit_rate: float,
                 used_prompt_cache: bool, oom: bool, timeout: bool,
                 performance_avg: float, task_max_perf: float):
        self.model_id = model_id
        self.context_len = context_len
        self.batch_size = batch_size
        self.best_of_n = best_of_n
        self.input_tok_s = input_tok_s
        self.output_tok_s = output_tok_s
        self.total_tok_s = total_tok_s
        self.ttft = ttft
        self.wall_clock = wall_clock
        self.cache_hit_rate = cache_hit_rate
        self.used_prompt_cache = used_prompt_cache
        self.oom = oom
        self.timeout = timeout
        self.performance_avg = performance_avg
        self.task_max_perf = task_max_perf


def write_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Write benchmark results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ['model_id', 'context_len', 'batch_size', 'best_of_n', 'input_tok_s', 'output_tok_s', 'total_tok_s', 'ttft', 'wall_clock', 'cache_hit_rate', 'used_prompt_cache', 'oom', 'timeout', 'performance_avg', 'task_max_perf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'model_id': result.model_id,
                'context_len': result.context_len,
                'batch_size': result.batch_size,
                'best_of_n': result.best_of_n,
                'input_tok_s': result.input_tok_s,
                'output_tok_s': result.output_tok_s,
                'total_tok_s': result.total_tok_s,
                'ttft': result.ttft,
                'wall_clock': result.wall_clock,
                'cache_hit_rate': result.cache_hit_rate,
                'used_prompt_cache': result.used_prompt_cache,
                'oom': result.oom,
                'timeout': result.timeout,
                'performance_avg': result.performance_avg,
                'task_max_perf': result.task_max_perf
            })


def get_selected_models(models_arg: str) -> List[str]:
    """Parse model selection argument."""
    if models_arg == "all":
        return ComprehensiveQwenVLBenchmark.SUPPORTED_MODELS
    return [model.strip() for model in models_arg.split(",") if model.strip() in ComprehensiveQwenVLBenchmark.SUPPORTED_MODELS]


def build_context_grid(min_ctx: int = 1024, multiplier: float = 2.0, max_ctx: int = 262144) -> List[int]:
    """Build geometric context length grid."""
    contexts = []
    current = min_ctx
    while current <= max_ctx:
        contexts.append(current)
        current = int(current * multiplier)
    return contexts


def load_models() -> List[str]:
    """Load model list from configs/qwen_vl_models.txt armada list."""
    models_file = Path(__file__).parent.parent / "configs" / "qwen_vl_models.txt"
    if not models_file.exists():
        logger.warning(f"Model list file {models_file} not found; using default models.")
        return ComprehensiveQwenVLBenchmark.SUPPORTED_MODELS

    try:
        with open(models_file, 'r', encoding='utf-8') as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if not models:
            logger.warning(f"No models found in {models_file}; using default models.")
            return ComprehensiveQwenVLBenchmark.SUPPORTED_MODELS
        logger.info(f"Loaded {len(models)} models from {models_file}")
        return models
    except Exception as exc:
        logger.warning(f"Failed to load models from {models_file} ({exc}); using default models.")
        return ComprehensiveQwenVLBenchmark.SUPPORTED_MODELS


def main():
    """CLI entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.export_skill_schema:
        if not SKILLS_AVAILABLE:
            logger.error("Skill schema export requires skills DSL to be implemented")
            return
        schema_payload = build_guidance_schema()
        args.export_skill_schema.parent.mkdir(parents=True, exist_ok=True)
        args.export_skill_schema.write_text(schema_payload, encoding="utf-8")
        prompt_path = args.export_skill_schema.with_suffix(".prompt.txt")
        prompt_path.write_text(compose_system_prompt(), encoding="utf-8")
        logger.info("Skill schema exported to %s (prompt scaffold -> %s)", args.export_skill_schema, prompt_path)

    # Parse models
    if args.models == "all":
        models = load_models()
    else:
        models = [m.strip() for m in args.models.split(",")]

    # Parse tasks
    if args.tasks == "all":
        tasks = list(ComprehensiveQwenVLBenchmark.TASKS.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Limit heavy sweeps unless --full is specified
    max_ctx = 32768 if not args.full else 262144  # Cap at 32k unless full mode
    time_budget_s = args.time_budget_s

    # Generate UTC ISO timestamp for output directory
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    utc_iso = utc_now.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"profiling/results/{utc_iso}")

    # Run comprehensive benchmarks
    benchmark = ComprehensiveQwenVLBenchmark(
        output_dir,
        tinywoods_dir=args.tinywoods_dir,
        tinywoods_seed=args.tinywoods_seed,
    )

    asyncio.run(benchmark.run_comprehensive_benchmarks(
        models, tasks, args.max_new_tokens, args.num_runs, args.dry_run,
        max_ctx=max_ctx, time_budget_s=time_budget_s, plot=args.plot
    ))


if __name__ == "__main__":
    main()
