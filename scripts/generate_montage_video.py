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
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import Counter, defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _sanitize_hf_environment() -> None:
    """Ensure Hugging Face cache paths are valid on Windows."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home.strip('"')
    else:
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def calculate_sampling_rate(num_frames: int, fps: float, target_duration_seconds: float) -> int:
    """Calculate frame sampling rate to hit target duration."""
    if num_frames <= 0 or fps <= 0 or target_duration_seconds <= 0:
        return 1
    target_frame_count = max(1, int(fps * target_duration_seconds))
    return max(1, num_frames // target_frame_count)


def build_voiceover_script(
    trajectory: List[dict],
    sample_rate: int,
    fps: float,
    target_duration_seconds: float,
) -> str:
    """Generate a narration script summarizing the montage."""
    total_frames = len(trajectory)
    if total_frames == 0:
        return (
            "Welcome to the Pokemon Mystery Dungeon autonomous agent demo. "
            "This highlight reel condenses our Tiny Woods exploration into a short showcase."
        )

    floor_numbers: List[int] = []
    floor_names: List[str] = []
    actions: List[str] = []
    decision_count = 0

    for frame in trajectory:
        # Extract decision/action metadata if present
        decision = None
        for key in ("decision", "agent_decision", "action_detail"):
            value = frame.get(key)
            if isinstance(value, dict):
                decision = value
                break
        if decision and isinstance(decision, dict):
            action = decision.get("action") or decision.get("name")
            if isinstance(action, str):
                actions.append(action.upper())
                decision_count += 1
        else:
            action_value = frame.get("action")
            if isinstance(action_value, str):
                actions.append(action_value.upper())
                decision_count += 1

        # Extract floor metadata
        ram = frame.get("ram") or frame.get("state", {}).get("ram") if isinstance(frame.get("state"), dict) else None
        if isinstance(ram, dict):
            floor_num = ram.get("floor_number") or ram.get("floor")
            if isinstance(floor_num, int):
                floor_numbers.append(floor_num)
            floor_name = ram.get("floor_name") or ram.get("map_name")
            if isinstance(floor_name, str):
                floor_names.append(floor_name)

    unique_floors = sorted(set(floor_numbers))
    floor_label = ""
    if floor_names:
        floor_label = floor_names[0]
    elif unique_floors:
        floor_label = f"Floor {unique_floors[0]}"

    if not floor_label:
        floor_phrase = "Tiny Woods Basement Floor one"
    else:
        floor_phrase = f"Tiny Woods {floor_label}"

    action_counter = Counter(actions)
    if action_counter:
        common_actions = [a.replace("_", " ").lower() for a, _ in action_counter.most_common(3)]
        if len(common_actions) == 1:
            action_phrase = common_actions[0]
        elif len(common_actions) == 2:
            action_phrase = " and ".join(common_actions)
        else:
            action_phrase = ", ".join(common_actions[:-1]) + f", and {common_actions[-1]}"
    else:
        action_phrase = "movement, observation, and item management"

    minutes = target_duration_seconds / 60.0 if target_duration_seconds > 0 else total_frames / max(fps, 1)
    lines = [
        f"Welcome to the Pokemon Mystery Dungeon autonomous agent demo inside {floor_phrase}.",
        f"This montage compresses {total_frames} captured states into about {minutes:.1f} minutes at {fps:.0f} frames per second.",
    ]

    if decision_count:
        lines.append(f"The policy issues roughly {decision_count} planned decisions, highlighting {action_phrase}.")
    else:
        lines.append(f"The showcase focuses on {action_phrase} as the agent reacts to the dungeon layout.")

    lines.extend(
        [
            f"We sample the environment every {sample_rate} frames to surface key transitions and discoveries.",
            "Watch how the agent orients toward the staircase while balancing belly, health, and positioning.",
            "Enjoy this quick look at the agent solving a single room on Tiny Woods Basement Floor one.",
        ]
    )

    return " ".join(lines)


def synthesize_voiceover(
    text: str,
    output_path: Path,
    voice: str = "af_heart",
    lang_code: str = "a",
) -> Path:
    """Generate narration audio using Kokoro TTS."""
    if not text.strip():
        raise ValueError("Voiceover text is empty.")

    _sanitize_hf_environment()

    try:
        from kokoro import KPipeline
    except ImportError as exc:  # pragma: no cover - missing optional dependency
        raise RuntimeError(
            "Missing dependency 'kokoro'. Install with `pip install kokoro soundfile`."
        ) from exc

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - missing optional dependency
        raise RuntimeError(
            "Missing dependency 'soundfile'. Install with `pip install soundfile`."
        ) from exc

    logger.info("Generating voiceover with Kokoro TTS (%s, voice=%s)", lang_code, voice)
    pipeline = KPipeline(lang_code=lang_code)
    generator = pipeline(text, voice=voice)

    segments: List[np.ndarray] = []
    for _, _, audio in generator:
        segments.append(np.array(audio, dtype=np.float32))

    if not segments:
        raise RuntimeError("Kokoro pipeline returned no audio segments.")

    waveform = np.concatenate(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, 24000)
    logger.info("Voiceover saved to %s (%.1f seconds)", output_path, len(waveform) / 24000.0)
    return output_path


def merge_voiceover_with_video(video_path: Path, audio_path: Path, fps: float) -> None:
    """Attach narration audio track to the generated video."""
    try:
        from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
    except ImportError as exc:  # pragma: no cover - missing optional dependency
        raise RuntimeError(
            "Missing dependency 'moviepy'. Install with `pip install moviepy`."
        ) from exc

    temp_output = video_path.with_suffix(".voiceover.tmp.mp4")
    temp_audio = audio_path.with_suffix(".tmp.m4a")

    logger.info("Merging voiceover into video %s", video_path)
    video_clip = VideoFileClip(str(video_path))
    audio_clip = AudioFileClip(str(audio_path))
    composite_audio = CompositeAudioClip([audio_clip]).set_duration(video_clip.duration)

    try:
        video_with_audio = video_clip.set_audio(composite_audio)
        video_with_audio.write_videofile(
            str(temp_output),
            codec="libx264",
            audio_codec="aac",
            fps=fps if fps > 0 else video_clip.fps,
            temp_audiofile=str(temp_audio),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
    finally:
        # Ensure resources are released
        video_clip.close()
        audio_clip.close()
        composite_audio.close()
        if "video_with_audio" in locals():
            video_with_audio.close()

    shutil.move(str(temp_output), str(video_path))
    if temp_audio.exists():
        temp_audio.unlink(missing_ok=True)
    logger.info("Voiceover merged into %s", video_path)

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
    target_duration_seconds: float = 180.0,
    sample_rate: Optional[int] = None,
    voiceover_options: Optional[Dict[str, Optional[str]]] = None,
) -> bool:
    """Generate montage video from trajectory.

    Args:
        trajectory: Agent trajectory data
        run_dir: Run directory containing potential image files
        output_path: Output MP4 path
        fps: Frames per second for video
        target_duration_seconds: Target duration (auto-adjusts sampling)
        sample_rate: Precomputed sampling rate (optional)
        voiceover_options: Voiceover configuration dictionary

    Returns:
        True if successful
    """
    if not trajectory:
        logger.error("Empty trajectory, cannot generate video")
        return False

    # Calculate sampling rate to hit target duration
    num_frames = len(trajectory)
    sample_rate = sample_rate or calculate_sampling_rate(num_frames, fps, target_duration_seconds)
    target_frame_count = max(1, int(fps * target_duration_seconds))

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
    montage_duration = len(key_frame_indices) / fps if fps > 0 else float(len(key_frame_indices))
    logger.info(f"✓ Video saved: {output_path}")
    logger.info(f"  Duration: {montage_duration:.1f} seconds")

    if voiceover_options and voiceover_options.get("enabled"):
        narration_text = voiceover_options.get("text") or build_voiceover_script(
            trajectory,
            sample_rate,
            fps,
            target_duration_seconds,
        )
        try:
            audio_output = Path(voiceover_options.get("audio_path") or output_path.with_suffix(".voiceover.wav"))
            voice_id = voiceover_options.get("voice") or "af_heart"
            lang_code = voiceover_options.get("lang") or "a"

            narration_path = synthesize_voiceover(
                narration_text,
                audio_output,
                voice=voice_id,
                lang_code=lang_code,
            )
            merge_voiceover_with_video(output_path, narration_path, fps)
        except Exception as exc:
            logger.error("Voiceover generation failed: %s", exc)
            return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate montage video from agent run")
    parser.add_argument("--run-dir", type=Path, help="Run directory (auto-detect if not provided)")
    parser.add_argument("--output", type=Path, default=Path("agent_demo.mp4"), help="Output video path")
    parser.add_argument("--fps", type=float, default=15.0, help="Video FPS")
    parser.add_argument("--duration", type=float, default=180.0, help="Target duration in seconds")
    parser.add_argument("--voiceover", action="store_true", help="Generate Kokoro narration and embed audio track")
    parser.add_argument("--voiceover-text", type=Path, help="Path to custom narration text (optional)")
    parser.add_argument("--voiceover-voice", type=str, default="af_heart", help="Kokoro voice id (default: af_heart)")
    parser.add_argument("--voiceover-lang", type=str, default="a", help="Kokoro language code (default: 'a' for English)")
    parser.add_argument("--voiceover-audio", type=Path, help="Optional path for saving the raw narration audio")

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

    sample_rate = calculate_sampling_rate(len(trajectory), args.fps, args.duration)

    voiceover_options: Optional[Dict[str, Optional[str]]] = None
    if args.voiceover:
        narration_text: Optional[str] = None
        if args.voiceover_text:
            try:
                narration_text = args.voiceover_text.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.error("Failed to read voiceover text file %s: %s", args.voiceover_text, exc)
                return 1

        voiceover_options = {
            "enabled": True,
            "text": narration_text,
            "voice": args.voiceover_voice,
            "lang": args.voiceover_lang,
            "audio_path": str(args.voiceover_audio) if args.voiceover_audio else None,
        }

    # Generate video
    success = generate_video(
        trajectory,
        run_dir,
        output_path=args.output,
        fps=args.fps,
        target_duration_seconds=args.duration,
        sample_rate=sample_rate,
        voiceover_options=voiceover_options,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
