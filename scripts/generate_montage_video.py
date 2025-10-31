#!/usr/bin/env python3
"""Generate 2-3 minute montage video from agent demonstration.

Extracts key frames from agent run and stitches them into MP4 with:
- Frame sampling (every 2-5 steps for fast playback)
- Key moment detection (state changes, decisions)
- Soft transitions + audio hints
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import cv2
    from PIL import Image
except ImportError:
    logger.error("Missing dependencies: install 'opencv-python' and 'pillow'")
    sys.exit(1)


def find_latest_run(runs_dir: Path = Path("runs")) -> Optional[Path]:
    """Find most recent agent run directory."""
    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return None

    runs = list(runs_dir.glob("demo_*")) + list(runs_dir.glob("run_*"))
    if not runs:
        logger.error("No runs found in runs directory")
        return None

    latest = max(runs, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using latest run: {latest}")
    return latest


def load_trajectory(run_dir: Path) -> List[dict]:
    """Load trajectory JSONL from run directory."""
    traj_files = list(run_dir.glob("trajectory_*.jsonl"))
    if not traj_files:
        logger.warning(f"No trajectory files in {run_dir}")
        return []

    traj_file = traj_files[0]
    logger.info(f"Loading trajectory from {traj_file}")

    trajectory = []
    with open(traj_file) as f:
        for i, line in enumerate(f):
            try:
                frame = json.loads(line)
                trajectory.append(frame)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")

    logger.info(f"Loaded {len(trajectory)} frames")
    return trajectory


def extract_key_frames(trajectory: List[dict], sample_rate: int = 2) -> List[int]:
    """Extract indices of key frames for video.

    Args:
        trajectory: Full trajectory from agent run
        sample_rate: Sample every N frames (e.g. 2 = every other frame)

    Returns:
        List of frame indices to include in video
    """
    key_frame_indices = []

    # Always include first frame
    key_frame_indices.append(0)

    # Sample at regular intervals
    for i in range(sample_rate, len(trajectory), sample_rate):
        key_frame_indices.append(i)

    # Always include last frame
    if trajectory and len(trajectory) - 1 not in key_frame_indices:
        key_frame_indices.append(len(trajectory) - 1)

    logger.info(f"Selected {len(key_frame_indices)} key frames from {len(trajectory)} total frames")
    return key_frame_indices


def load_screenshot_data(frame_data: dict, run_dir: Path) -> Optional[np.ndarray]:
    """Load screenshot image from frame data."""
    if not frame_data.get("screenshot"):
        return None

    # Screenshots might be stored as:
    # 1. In-memory serialized in trajectory (base64 or raw bytes)
    # 2. As separate PNG files in run_dir

    screenshot = frame_data["screenshot"]
    if isinstance(screenshot, str):
        # Try to load from file
        screenshot_path = run_dir / screenshot
        if not screenshot_path.exists():
            return None
        try:
            img = Image.open(screenshot_path)
            return np.array(img)
        except Exception as e:
            logger.debug(f"Failed to load screenshot {screenshot_path}: {e}")
            return None

    # If it's already an array or PIL Image, use it directly
    if isinstance(screenshot, (np.ndarray, Image.Image)):
        return np.array(screenshot)

    return None


def create_placeholder_frame(width: int = 240, height: int = 160) -> np.ndarray:
    """Create a placeholder frame when screenshot is unavailable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Gray background with text
    cv2.putText(
        frame, "Frame data unavailable",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    return frame


def generate_video(
    trajectory: List[dict],
    run_dir: Path,
    output_path: Path = Path("agent_demo.mp4"),
    fps: float = 15.0,
    target_duration_seconds: float = 180.0
) -> bool:
    """Generate montage video from trajectory.

    Args:
        trajectory: Agent trajectory data
        run_dir: Run directory containing potential image files
        output_path: Output MP4 path
        fps: Frames per second for video
        target_duration_seconds: Target duration (auto-adjusts sampling)

    Returns:
        True if successful
    """
    if not trajectory:
        logger.error("Empty trajectory, cannot generate video")
        return False

    # Calculate sampling rate to hit target duration
    num_frames = len(trajectory)
    target_frame_count = int(fps * target_duration_seconds)
    sample_rate = max(1, num_frames // target_frame_count)

    logger.info(f"Target: {target_duration_seconds}s @ {fps} FPS = {target_frame_count} frames")
    logger.info(f"Sampling every {sample_rate} frames")

    # Extract key frame indices
    key_frame_indices = []
    for i in range(0, num_frames, sample_rate):
        key_frame_indices.append(i)
    if key_frame_indices[-1] != num_frames - 1:
        key_frame_indices.append(num_frames - 1)

    logger.info(f"Video will have {len(key_frame_indices)} frames")

    # Load first frame to determine video size
    first_frame = None
    for idx in key_frame_indices:
        img = load_screenshot_data(trajectory[idx], run_dir)
        if img is not None:
            first_frame = img
            break

    if first_frame is None:
        logger.warning("No screenshots found, using placeholder frames")
        first_frame = create_placeholder_frame()

    height, width = first_frame.shape[:2]
    logger.info(f"Video size: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error("Failed to open video writer")
        return False

    # Write frames
    for i, frame_idx in enumerate(key_frame_indices):
        frame_data = trajectory[frame_idx]

        # Load screenshot
        img = load_screenshot_data(frame_data, run_dir)
        if img is None:
            img = create_placeholder_frame(width, height)

        # Ensure correct size and format
        if img.shape != (height, width, 3):
            img = cv2.resize(img, (width, height))
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Convert RGB to BGR for OpenCV
        if img.shape[2] == 3 and img.dtype == np.uint8:
            # Assume RGB, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        writer.write(img.astype(np.uint8))

        if (i + 1) % 30 == 0:
            logger.info(f"Wrote {i + 1}/{len(key_frame_indices)} frames")

    writer.release()
    logger.info(f"✓ Video saved: {output_path}")
    logger.info(f"  Duration: {len(key_frame_indices) / fps:.1f} seconds")

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate montage video from agent run")
    parser.add_argument("--run-dir", type=Path, help="Run directory (auto-detect if not provided)")
    parser.add_argument("--output", type=Path, default=Path("agent_demo.mp4"), help="Output video path")
    parser.add_argument("--fps", type=float, default=15.0, help="Video FPS")
    parser.add_argument("--duration", type=float, default=180.0, help="Target duration in seconds")

    args = parser.parse_args()

    # Find run directory
    run_dir = args.run_dir
    if not run_dir:
        run_dir = find_latest_run()
        if not run_dir:
            logger.error("Could not find run directory")
            return 1

    # Load trajectory
    trajectory = load_trajectory(run_dir)
    if not trajectory:
        logger.error("Could not load trajectory")
        return 1

    # Generate video
    success = generate_video(
        trajectory,
        run_dir,
        output_path=args.output,
        fps=args.fps,
        target_duration_seconds=args.duration
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
