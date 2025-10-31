#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.action_executor import Button
from src.environment.mgba_controller import MGBAController


@dataclass(frozen=True)
class CaptureStep:
    """Description of a single controller interaction to snapshot."""

    name: str
    buttons: Sequence[Button]
    hold_ms: int = 150
    wait_ms: int = 220
    seed: int = 0
    notes: str | None = None


DEFAULT_STEPS: Sequence[CaptureStep] = (
    CaptureStep(
        "close_dialog",
        (Button.B,),
        hold_ms=70,
        wait_ms=240,
        seed=101,
        notes="Dismiss lingering text banners so HUD is visible.",
    ),
    CaptureStep(
        "move_up",
        (Button.UP,),
        hold_ms=210,
        wait_ms=200,
        seed=102,
        notes="Advance toward north corridor.",
    ),
    CaptureStep(
        "move_right",
        (Button.RIGHT,),
        hold_ms=200,
        wait_ms=180,
        seed=103,
        notes="Sidestep to align with item tile.",
    ),
    CaptureStep(
        "wait_enemy",
        (),
        hold_ms=0,
        wait_ms=420,
        seed=104,
        notes="Idle to let nearby enemy enter frame.",
    ),
    CaptureStep(
        "open_menu",
        (Button.B,),
        hold_ms=90,
        wait_ms=260,
        seed=105,
        notes="Toggle action grid for inventory parse shots.",
    ),
    CaptureStep(
        "navigate_menu",
        (Button.DOWN,),
        hold_ms=130,
        wait_ms=180,
        seed=106,
        notes="Highlight a different inventory slot.",
    ),
    CaptureStep(
        "confirm_menu",
        (Button.A,),
        hold_ms=120,
        wait_ms=250,
        seed=107,
        notes="Show detailed item tooltip.",
    ),
    CaptureStep(
        "exit_menu",
        (Button.B,),
        hold_ms=60,
        wait_ms=200,
        seed=108,
        notes="Return to exploration HUD.",
    ),
    CaptureStep(
        "move_left",
        (Button.LEFT,),
        hold_ms=210,
        wait_ms=190,
        seed=109,
        notes="Strafe toward detected stairs tile.",
    ),
    CaptureStep(
        "scan_room",
        (Button.START,),
        hold_ms=140,
        wait_ms=260,
        seed=110,
        notes="Toggle map overlay for overview capture.",
    ),
    CaptureStep(
        "move_down",
        (Button.DOWN,),
        hold_ms=220,
        wait_ms=190,
        seed=111,
        notes="Retreat to show enemy pursuit.",
    ),
    CaptureStep(
        "attack",
        (Button.A,),
        hold_ms=150,
        wait_ms=240,
        seed=112,
        notes="Standard attack pose for combat proxy.",
    ),
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_manifest(manifest_path: Path, frames_dir: Path) -> Dict[str, object]:
    """Ensure manifest JSONL references frames that exist on disk."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = 0
    missing: List[str] = []
    unique_actions = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_no} of {manifest_path}: {exc}"
                ) from exc

            entries += 1
            action = payload.get("action", "")
            if action:
                unique_actions.add(action)

            rel_path = Path(str(payload.get("frame_path", "")).replace("\\", "/"))
            frame_path = manifest_path.parent / rel_path
            if not frame_path.exists():
                missing.append(str(frame_path))

    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} frame(s) referenced in manifest missing on disk; e.g., {sample}"
        )

    if entries < 50 or entries > 200:
        print(f"[warn] Captured {entries} frames (recommended range is 50-200).")

    return {
        "entries": entries,
        "unique_actions": len(unique_actions),
        "frames_dir": str(frames_dir),
    }


def press_sequence(
    controller: MGBAController,
    buttons: Sequence[Button],
    hold_ms: int,
    *,
    dry_run: bool,
) -> bool:
    if not buttons:
        return True

    key_list = [btn.value for btn in buttons]

    if dry_run:
        time.sleep(max(hold_ms, 0) / 1000.0)
        return True

    ok = True
    if len(key_list) == 1:
        ok = controller.button_hold(key_list[0], max(1, int(hold_ms)))
    else:
        ok = controller.press(key_list)
        if ok:
            time.sleep(max(hold_ms, 0) / 1000.0)
            controller.button_clear_many(key_list)

    if ok:
        controller.sync_after_input(key_list, sync_frames=6)
    return ok


def wait_for_stability(
    controller: MGBAController,
    wait_ms: int,
    *,
    dry_run: bool,
) -> None:
    if wait_ms <= 0:
        return
    frame_budget = max(1, wait_ms // 16)
    if dry_run:
        time.sleep(wait_ms / 1000.0)
    else:
        controller.await_frames(frame_budget)


def capture_dataset(
    controller: MGBAController,
    steps: Sequence[CaptureStep],
    frames_dir: Path,
    manifest_path: Path,
    target_frames: int,
    *,
    dry_run: bool,
    throttle_ms: int,
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    frame_index = 0

    while frame_index < target_frames:
        for step in steps:
            if frame_index >= target_frames:
                break

            random.seed(step.seed + frame_index)
            success = press_sequence(
                controller,
                step.buttons,
                step.hold_ms,
                dry_run=dry_run,
            )
            if not success:
                raise RuntimeError(f"Input dispatch failed for step {step.name}")

            wait_for_stability(controller, step.wait_ms, dry_run=dry_run)

            timestamp = datetime.now(timezone.utc).isoformat()
            filename = f"{frame_index:04d}_{step.name}.png"
            frame_path = frames_dir / filename

            if not dry_run:
                ok = controller.screenshot(str(frame_path))
                if not ok:
                    raise RuntimeError(f"Screenshot command failed for {frame_path}")
            else:
                frame_path.touch(exist_ok=True)

            manifest.append(
                {
                    "frame_id": frame_index,
                    "action": step.name,
                    "buttons": [btn.value for btn in step.buttons],
                    "hold_ms": step.hold_ms,
                    "wait_ms": step.wait_ms,
                    "seed": step.seed,
                    "captured_at": timestamp,
                    "frame_path": str(frame_path.relative_to(frames_dir.parent)),
                    "notes": step.notes,
                }
            )

            frame_index += 1
            if throttle_ms > 0:
                time.sleep(throttle_ms / 1000.0)

    manifest_path.write_text("\n".join(json.dumps(entry) for entry in manifest) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Tiny Woods gameplay frames via mGBA HTTP controller."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling/data/tinywoods"),
        help="Destination directory for frames and manifest.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=72,
        help="Number of frames to capture (50-200 recommended).",
    )
    parser.add_argument(
        "--throttle-ms",
        type=int,
        default=120,
        help="Sleep duration between captures to avoid overruns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip mgba commands and create placeholder files only.",
    )
    parser.add_argument(
        "--loop-steps",
        action="store_true",
        help="Loop the default step plan if more frames than steps are requested (default).",
    )
    return parser.parse_args()


class DummyController:
    """Fallback controller for --dry-run flows."""

    def button_hold(self, button: str, duration_ms: int) -> bool:
        return True

    def press(self, keys: List[str]) -> bool:
        return True

    def button_clear_many(self, buttons: List[str]) -> bool:
        return True

    def sync_after_input(self, input_keys: List[str], sync_frames: int = 5) -> bool:
        return True

    def await_frames(self, n: int) -> bool:
        time.sleep(max(n, 1) * 0.016)
        return True

    def screenshot(self, path: str) -> bool:
        Path(path).touch(exist_ok=True)
        return True

    def connect_with_retry(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None


class DummyContext:
    def __enter__(self) -> DummyController:
        return DummyController()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be positive")

    output_root = args.output_dir
    frames_dir = output_root / "frames"
    ensure_dir(frames_dir)
    manifest_path = output_root / "manifest.jsonl"

    steps = list(DEFAULT_STEPS)
    if args.frames > len(steps) and not args.loop_steps:
        raise SystemExit(
            "Requested frames exceed available steps; pass --loop-steps to reuse the sequence."
        )

    controller_context = DummyContext() if args.dry_run else MGBAController()

    try:
        with controller_context as controller:
            if not args.dry_run:
                controller.connect_with_retry()

            manifest = capture_dataset(
                controller,
                steps,
                frames_dir,
                manifest_path,
                args.frames,
                dry_run=args.dry_run,
                throttle_ms=max(0, args.throttle_ms),
            )
    finally:
        if not args.dry_run:
            controller.disconnect()

    stats = validate_manifest(manifest_path, frames_dir)
    print(f"Captured {len(manifest)} frames to {frames_dir}")
    print(f"Manifest written to {manifest_path}")
    print(
        f"Dataset check: {stats['entries']} entries, {stats['unique_actions']} unique actions."
    )


if __name__ == "__main__":
    main()
