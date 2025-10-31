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


def detect_ram_events(trajectory: List[dict]) -> List[Event]:
    """Detect key RAM-based events from trajectory."""
    events = []

    prev_ram = None
    prev_hp = None
    prev_belly = None
    prev_inventory = None

    for i, frame in enumerate(trajectory):
        ram = frame.get("ram") or frame.get("state", {}).get("ram") or {}
        if not ram:
            continue

        timestamp = frame.get("timestamp", 0.0)

        # Floor changes
        current_floor = ram.get("floor_number") or ram.get("floor")
        if prev_ram and current_floor != prev_ram.get("floor_number"):
            events.append(Event(
                timestamp=timestamp,
                event_type="floor_change",
                score=9.0,
                metadata={"new_floor": current_floor, "prev_floor": prev_ram.get("floor_number")},
                frame_idx=i
            ))

        # Low HP detection
        hp_current = ram.get("leader", {}).get("hp") or ram.get("hp_current")
        hp_max = ram.get("leader", {}).get("hp_max") or ram.get("hp_max")
        if hp_current and hp_max and prev_hp:
            hp_ratio = hp_current / hp_max if hp_max > 0 else 0
            if hp_ratio < 0.25 and hp_current < prev_hp:  # Low HP and decreasing
                events.append(Event(
                    timestamp=timestamp,
                    event_type="low_hp",
                    score=8.0,
                    metadata={"hp_current": hp_current, "hp_max": hp_max, "ratio": hp_ratio},
                    frame_idx=i
                ))

        # Low belly detection
        belly_current = ram.get("leader", {}).get("belly") or ram.get("belly_current")
        if belly_current and prev_belly and belly_current < 50 and belly_current < prev_belly:
            events.append(Event(
                timestamp=timestamp,
                event_type="low_belly",
                score=7.0,
                metadata={"belly_current": belly_current},
                frame_idx=i
            ))

        # Item pickup detection (simplified - would need full inventory comparison)
        if prev_inventory and ram.get("items") != prev_inventory.get("items"):
            events.append(Event(
                timestamp=timestamp,
                event_type="item_pickup",
                score=6.0,
                metadata={"items": ram.get("items")},
                frame_idx=i
            ))

        # Mission complete detection
        if ram.get("mission_complete") or ram.get("victory"):
            events.append(Event(
                timestamp=timestamp,
                event_type="mission_complete",
                score=10.0,
                metadata={"victory": True},
                frame_idx=i
            ))

        prev_ram = ram
        prev_hp = hp_current
        prev_belly = belly_current
        prev_inventory = ram.get("items")

    return events


def detect_vision_events(trajectory: List[dict]) -> List[Event]:
    """Detect vision-based events using Qwen-VL analysis."""
    events = []

    # Placeholder for Qwen-VL integration - would analyze screenshots
    # For now, detect basic room type changes from RAM
    prev_room_type = None

    for i, frame in enumerate(trajectory):
        ram = frame.get("ram") or frame.get("state", {}).get("ram") or {}
        timestamp = frame.get("timestamp", 0.0)

        # Room type detection (simplified from RAM)
        current_room_type = ram.get("room_type", "unknown")
        if prev_room_type and current_room_type != prev_room_type:
            score = 5.0 if "staircase" in current_room_type.lower() else 4.0
            events.append(Event(
                timestamp=timestamp,
                event_type="room_type_change",
                score=score,
                metadata={"room_type": current_room_type, "prev_type": prev_room_type},
                frame_idx=i
            ))

        # Enemy proximity detection
        entities = ram.get("monsters", [])
        close_enemies = [e for e in entities if e.get("distance", 10) < 3]
        if close_enemies:
            events.append(Event(
                timestamp=timestamp,
                event_type="enemy_proximity",
                score=6.0,
                metadata={"close_enemies": len(close_enemies), "enemies": close_enemies},
                frame_idx=i
            ))

        prev_room_type = current_room_type

    return events


def detect_skill_triggers(trajectory: List[dict]) -> List[Event]:
    """Detect first-time execution of meta-skills."""
    events = []
    seen_skills = set()

    for i, frame in enumerate(trajectory):
        decision = frame.get("decision")
        if not decision:
            continue

        # Extract skill/action name
        skill_name = None
        if isinstance(decision, dict):
            skill_name = decision.get("action") or decision.get("name")
        elif isinstance(decision, str):
            skill_name = decision

        if skill_name and skill_name not in seen_skills:
            seen_skills.add(skill_name)
            events.append(Event(
                timestamp=frame.get("timestamp", 0.0),
                event_type="first_skill_execution",
                score=9.0,
                metadata={"skill_name": skill_name, "first_time": True},
                frame_idx=i
            ))

    return events


def detect_trajectory_deltas(trajectory: List[dict]) -> List[Event]:
    """Detect significant trajectory changes and progress."""
    events = []

    for i in range(1, len(trajectory)):
        current = trajectory[i]
        prev = trajectory[i-1]

        # Position changes (rapid progress)
        current_pos = current.get("ram", {}).get("player_tile_x", 0), current.get("ram", {}).get("player_tile_y", 0)
        prev_pos = prev.get("ram", {}).get("player_tile_x", 0), prev.get("ram", {}).get("player_tile_y", 0)

        if current_pos != prev_pos:
            distance = abs(current_pos[0] - prev_pos[0]) + abs(current_pos[1] - prev_pos[1])
            if distance > 3:  # Significant movement
                events.append(Event(
                    timestamp=current.get("timestamp", 0.0),
                    event_type="rapid_progress",
                    score=5.0,
                    metadata={"distance": distance, "from": prev_pos, "to": current_pos},
                    frame_idx=i
                ))

        # Reward changes (if reward data available)
        current_reward = current.get("reward", 0)
        prev_reward = prev.get("reward", 0)
        if current_reward > prev_reward + 10:  # Significant positive reward
            events.append(Event(
                timestamp=current.get("timestamp", 0.0),
                event_type="big_reward",
                score=7.0,
                metadata={"reward": current_reward, "delta": current_reward - prev_reward},
                frame_idx=i
            ))

    return events


def collect_key_events(trajectory: List[dict]) -> List[Event]:
    """Collect all key events from multi-signal detection."""
    all_events = []

    # RAM events
    ram_events = detect_ram_events(trajectory)
    all_events.extend(ram_events)
    logger.info(f"Detected {len(ram_events)} RAM events")

    # Vision events
    vision_events = detect_vision_events(trajectory)
    all_events.extend(vision_events)
    logger.info(f"Detected {len(vision_events)} vision events")

    # Skill triggers
    skill_events = detect_skill_triggers(trajectory)
    all_events.extend(skill_events)
    logger.info(f"Detected {len(skill_events)} skill trigger events")

    # Trajectory deltas
    delta_events = detect_trajectory_deltas(trajectory)
    all_events.extend(delta_events)
    logger.info(f"Detected {len(delta_events)} trajectory delta events")

    # Sort by timestamp
    all_events.sort(key=lambda e: e.timestamp)

    # Save events to JSONL
    events_path = Path("events.jsonl")
    with open(events_path, 'w', encoding='utf-8') as f:
        for event in all_events:
            json.dump({
                "t": event.timestamp,
                "type": event.event_type,
                "score": event.score,
                "metadata": event.metadata,
                "frame_idx": event.frame_idx
            }, f)
            f.write('\n')

    logger.info(f"Saved {len(all_events)} events to {events_path}")
    return all_events


def select_key_frames_with_events(trajectory: List[dict], events: List[Event],
                                target_duration_seconds: float = 180.0,
                                fps: float = 15.0) -> List[int]:
    """Select key frames covering early/mid/late game with event prioritization."""
    total_frames = len(trajectory)
    if total_frames == 0:
        return []

    # Calculate target frame count
    target_frame_count = max(1, int(fps * target_duration_seconds))

    # Sort events by score descending
    sorted_events = sorted(events, key=lambda e: e.score, reverse=True)

    # Select top-K events covering different game phases
    selected_events = []
    early_threshold = total_frames * 0.33
    mid_threshold = total_frames * 0.67

    early_events = [e for e in sorted_events if e.frame_idx < early_threshold]
    mid_events = [e for e in sorted_events if early_threshold <= e.frame_idx < mid_threshold]
    late_events = [e for e in sorted_events if e.frame_idx >= mid_threshold]

    # Take top events from each phase
    events_per_phase = max(1, target_frame_count // 10)  # ~10% of frames from events
    selected_events.extend(early_events[:events_per_phase])
    selected_events.extend(mid_events[:events_per_phase])
    selected_events.extend(late_events[:events_per_phase])

    # Add regular sampling for remaining frames
    key_frame_indices = set(e.frame_idx for e in selected_events)
    remaining_frames = target_frame_count - len(key_frame_indices)

    if remaining_frames > 0:
        sample_rate = max(1, total_frames // remaining_frames)
        for i in range(0, total_frames, sample_rate):
            if i not in key_frame_indices:
                key_frame_indices.add(i)
                if len(key_frame_indices) >= target_frame_count:
                    break

    # Ensure first and last frames
    key_frame_indices.add(0)
    if total_frames > 1:
        key_frame_indices.add(total_frames - 1)

    return sorted(list(key_frame_indices))


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
    """Generate narration audio using Kokoro TTS with pyttsx3 fallback."""
    if not text.strip():
        raise ValueError("Voiceover text is empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Kokoro first
    try:
        _sanitize_hf_environment()

        from kokoro import KPipeline
        import soundfile as sf

        logger.info("Generating voiceover with Kokoro TTS (%s, voice=%s)", lang_code, voice)
        pipeline = KPipeline(lang_code=lang_code)
        generator = pipeline(text, voice=voice)

        segments: List[np.ndarray] = []
        for _, _, audio in generator:
            segments.append(np.array(audio, dtype=np.float32))

        if not segments:
            raise RuntimeError("Kokoro pipeline returned no audio segments.")

        waveform = np.concatenate(segments)
        sf.write(str(output_path), waveform, 24000)
        logger.info("Voiceover saved to %s (%.1f seconds)", output_path, len(waveform) / 24000.0)
        return output_path

    except (ImportError, RuntimeError, Exception) as exc:
        logger.warning("Kokoro TTS failed (%s), falling back to pyttsx3", exc)

    # Fallback to pyttsx3
    try:
        import pyttsx3

        engine = pyttsx3.init()
        # Configure voice if available
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a female voice
            female_voice = None
            for v in voices:
                if hasattr(v, 'gender') and v.gender and 'female' in v.gender.lower():
                    female_voice = v
                    break
            if female_voice:
                engine.setProperty('voice', female_voice.id)

        engine.setProperty('rate', 180)  # Speed up speech
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        logger.info("Voiceover saved to %s (pyttsx3 fallback)", output_path)
        return output_path

    except ImportError as exc:
        raise RuntimeError(
            "Missing TTS dependencies. Install 'kokoro soundfile' or 'pyttsx3'."
        ) from exc


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


def validate_frames_black_check(frames: List[np.ndarray], eps: float = 0.05) -> List[np.ndarray]:
    """Validate frames, skipping those that are too dark/black."""
    validated = []

    for img in frames:
        if img is None or img.size == 0:
            continue

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
        else:
            gray = img

        # Calculate mean and std dev of luminance
        mean_lum = np.mean(gray) / 255.0
        std_lum = np.std(gray) / 255.0

        # Skip if too dark/flat (likely black/dead frames)
        if mean_lum < eps or std_lum < eps:
            logger.debug(f"Skipping black/dead frame (mean={mean_lum:.3f}, std={std_lum:.3f})")
            continue

        validated.append(img)

    logger.info(f"Validated {len(validated)}/{len(frames)} frames (skipped {len(frames) - len(validated)} black)")
    return validated


def create_quad_montage_layout(
    screenshot: Optional[np.ndarray],
    minimap: Optional[np.ndarray],
    grid_overlay: Optional[np.ndarray],
    reasoning_text: str,
    timeline_events: List[Event],
    frame_idx: int,
    total_frames: int
) -> np.ndarray:
    """Create 2x2 montage layout for video frame."""
    # Target dimensions
    base_width, base_height = 480, 320  # Scaled GBA resolution
    half_width = base_width // 2
    half_height = base_height // 2

    # Create base frame
    frame = np.zeros((base_height, base_width, 3), dtype=np.uint8)

    # Top-left: Raw gameplay screenshot
    if screenshot is not None:
        resized = cv2.resize(screenshot, (half_width, half_height))
        if len(resized.shape) == 2:  # Grayscale
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        frame[0:half_height, 0:half_width] = resized
    else:
        cv2.putText(frame, "NO SCREENSHOT", (10, half_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Top-right: Mini-map/semantic overlay
    if minimap is not None:
        resized = cv2.resize(minimap, (half_width, half_height))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        frame[0:half_height, half_width:base_width] = resized
    else:
        # Create placeholder minimap
        cv2.rectangle(frame, (half_width, 0), (base_width, half_height), (50, 50, 50), -1)
        cv2.putText(frame, "MINIMAP", (half_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bottom-left: Grid overlay
    if grid_overlay is not None:
        resized = cv2.resize(grid_overlay, (half_width, half_height))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        frame[half_height:base_height, 0:half_width] = resized
    else:
        cv2.rectangle(frame, (0, half_height), (half_width, base_height), (30, 30, 30), -1)
        cv2.putText(frame, "GRID OVERLAY", (10, half_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bottom-right: Reasoning subtitle + timeline
    cv2.rectangle(frame, (half_width, half_height), (base_width, base_height), (20, 20, 20), -1)

    # Add reasoning text (short excerpt)
    if reasoning_text:
        lines = reasoning_text[:100].split('\n')[:3]  # Max 3 lines, 100 chars
        y_offset = half_height + 20
        for line in lines:
            cv2.putText(frame, line[:30], (half_width + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15

    # Add timeline scrub with event markers
    timeline_y = base_height - 30
    timeline_width = base_width - half_width - 20
    timeline_x = half_width + 10

    # Timeline background
    cv2.rectangle(frame, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + 10), (100, 100, 100), -1)

    # Event markers
    for event in timeline_events:
        event_pos = int(timeline_x + (event.frame_idx / max(1, total_frames)) * timeline_width)
        marker_color = (0, 255, 0) if event.score >= 8 else (255, 255, 0) if event.score >= 6 else (255, 165, 0)
        cv2.circle(frame, (event_pos, timeline_y + 5), 2, marker_color, -1)

    # Current position marker
    current_pos = int(timeline_x + (frame_idx / max(1, total_frames)) * timeline_width)
    cv2.circle(frame, (current_pos, timeline_y + 5), 3, (255, 0, 0), -1)

    # Labels
    labels = [
        ("GAMEPLAY", 10, 15),
        ("MAP", half_width + 10, 15),
        ("GRID", 10, half_height + 15),
        ("REASONING", half_width + 10, half_height + 15)
    ]
    for label, x, y in labels:
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return frame


def calculate_adaptive_speedup(trajectory: List[dict], events: List[Event],
                              frame_idx: int, window_size: int = 10) -> float:
    """Calculate adaptive speedup based on event density and activity."""
    # Check for nearby events (high activity = slower playback)
    nearby_events = [e for e in events if abs(e.frame_idx - frame_idx) <= window_size]
    event_density = len(nearby_events) / max(1, window_size * 2)

    # Base speedup
    base_speedup = 4.0

    # Reduce speedup near events
    if event_density > 0.5:
        return 1.0  # Normal speed for critical moments
    elif event_density > 0.2:
        return 2.0  # 2x speed for moderate activity
    else:
        return min(12.0, base_speedup)  # 4-12x speed for idle periods


def generate_video(
    trajectory: List[dict],
    run_dir: Path,
    output_path: Path = Path("docs/assets/agent_demo.mp4"),
    fps: float = 30.0,
    target_duration_seconds: float = 180.0,
    events: Optional[List[Event]] = None,
    voiceover_options: Optional[Dict[str, Optional[str]]] = None,
) -> bool:
    """Generate 2x2 montage video from trajectory with adaptive playback.

    Args:
        trajectory: Agent trajectory data
        run_dir: Run directory containing potential image files
        output_path: Output MP4 path (docs/assets/agent_demo.mp4)
        fps: Base frames per second for video
        target_duration_seconds: Target duration (120-180s)
        events: Pre-detected key events
        voiceover_options: Voiceover configuration dictionary

    Returns:
        True if successful
    """
    if not trajectory:
        logger.error("Empty trajectory, cannot generate video")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect events if not provided
    if events is None:
        events = collect_key_events(trajectory)

    # Select key frames with event prioritization
    key_frame_indices = select_key_frames_with_events(trajectory, events, target_duration_seconds, fps)
    logger.info(f"Selected {len(key_frame_indices)} key frames from {len(trajectory)} total")

    # Validate frames and skip black ones
    valid_frames = []
    valid_indices = []

    for idx in key_frame_indices:
        img = load_screenshot_data(trajectory[idx], run_dir)
        if img is None:
            img = create_placeholder_frame(240, 160)  # GBA resolution

        # Convert to BGR for OpenCV
        if len(img.shape) == 3 and img.dtype == np.uint8:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        valid_frames.append(img)
        valid_indices.append(idx)

    # Anti-black check
    valid_frames = validate_frames_black_check(valid_frames)
    if len(valid_frames) < 150:
        logger.error(f"Insufficient valid frames: {len(valid_frames)} < 150")
        return False

    logger.info(f"Proceeding with {len(valid_frames)} validated frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for h264 if available
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (480, 320))  # 2x GBA resolution

    if not writer.isOpened():
        logger.error("Failed to open video writer")
        return False

    # Write frames with 2x2 layout and adaptive speedup
    for i, (img, frame_idx) in enumerate(zip(valid_frames, valid_indices)):
        # Create quad layout
        reasoning_text = trajectory[frame_idx].get("decision", {}).get("rationale", "")
        quad_frame = create_quad_montage_layout(
            img, None, None, reasoning_text, events, frame_idx, len(trajectory)
        )

        # Adaptive speedup (write multiple frames for slower playback)
        speedup = calculate_adaptive_speedup(trajectory, events, frame_idx)
        frames_to_write = max(1, int(fps / speedup))  # More frames = slower playback

        for _ in range(frames_to_write):
            writer.write(quad_frame)

        if (i + 1) % 30 == 0:
            logger.info(f"Wrote {i + 1}/{len(valid_frames)} frames (speedup: {speedup:.1f}x)")

    writer.release()

    # Validate with ffprobe
    if not validate_video_output(output_path, target_duration_seconds):
        logger.error("Video validation failed")
        return False

    montage_duration = len(valid_frames) / fps  # Approximate
    logger.info(f"✓ Video saved: {output_path}")
    logger.info(f"  Duration: {montage_duration:.1f} seconds")
    logger.info(f"  Events highlighted: {len(events)}")

    # Add voiceover if requested
    if voiceover_options and voiceover_options.get("enabled"):
        narration_text = voiceover_options.get("text") or build_voiceover_script(
            trajectory,
            len(valid_frames) // max(1, int(fps * target_duration_seconds)),
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


def validate_video_output(video_path: Path, min_duration: float = 120.0) -> bool:
    """Validate video output with ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.error("ffprobe failed: %s", result.stderr)
            return False

        import json
        probe_data = json.loads(result.stdout)

        # Check duration
        duration = float(probe_data.get("format", {}).get("duration", 0))
        if duration < min_duration:
            logger.error(f"Video too short: {duration:.1f}s < {min_duration}s")
            return False

        # Check for video stream
        video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            logger.error("No video stream found")
            return False

        logger.info(f"✓ Video validated: {duration:.1f}s duration, {len(video_streams)} video stream(s)")
        return True

    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning("ffprobe validation skipped: %s", e)
        return True  # Allow if ffprobe not available


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2x2 montage video from agent run")
    parser.add_argument("--run-dir", type=Path, help="Run directory (auto-detect if not provided)")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/agent_demo.mp4"), help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--duration", type=float, default=150.0, help="Target duration in seconds (120-180)")
    parser.add_argument("--voiceover", action="store_true", help="Generate narration and embed audio track")
    parser.add_argument("--voiceover-text", type=Path, help="Path to custom narration text (optional)")
    parser.add_argument("--voiceover-voice", type=str, default="af_heart", help="Voice id (default: af_heart)")
    parser.add_argument("--voiceover-lang", type=str, default="a", help="Language code (default: 'a' for English)")
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

    logger.info(f"Loaded {len(trajectory)} frames from {run_dir}")

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

    # Generate video with event detection
    success = generate_video(
        trajectory,
        run_dir,
        output_path=args.output,
        fps=args.fps,
        target_duration_seconds=args.duration,
        voiceover_options=voiceover_options,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
