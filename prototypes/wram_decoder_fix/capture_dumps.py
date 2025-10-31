"""Capture raw WRAM dumps from a running PMD-Red session via mGBA."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure the project sources are importable.
sys.path.insert(0, str(REPO_ROOT))

from src.environment.mgba_controller import MGBAController  # type: ignore  # pylint: disable=wrong-import-position


def capture_wram_dump(
    output_path: Path,
    *,
    address: int = 0x02000000,
    size: int = 2048,
    controller: Optional[MGBAController] = None,
) -> None:
    """Capture raw WRAM data and save it to ``output_path``."""
    local_controller = controller or MGBAController()
    owns_controller = controller is None

    try:
        if owns_controller:
            local_controller.connect()

        raw_hex = local_controller._send_command(  # pylint: disable=protected-access
            f"memoryDomain.readRange,wram,{address},{size}"
        )
        raw_bytes = bytes.fromhex(raw_hex.replace(",", ""))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_bytes)

        print(f"✓ Captured {len(raw_bytes)} bytes to {output_path}")
    finally:
        if owns_controller:
            try:
                local_controller.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Failed to disconnect controller cleanly: {exc}")


def main() -> None:
    sandbox_root = Path(__file__).resolve().parent
    dumps_dir = sandbox_root / "sample_wram_dumps"
    dumps_dir.mkdir(exist_ok=True)

    print("=== WRAM Dump Capture ===")
    print("Prerequisites:")
    print("  • mGBA must be running with PMD-Red (US v1.0) loaded.")
    print("  • The IPC bridge script should already be attached.")
    print()

    sequences = [
        ("Floor 1 Room 1", "dump_floor1_room1.bin"),
        ("Floor 1 Room 2", "dump_floor1_room2.bin"),
        ("Combat Encounter", "dump_combat.bin"),
    ]

    for description, filename in sequences:
        input(f"Press Enter to capture {description}...")
        capture_wram_dump(dumps_dir / filename)
        print()

    print("✓ All dumps captured!")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print("⚠️  Could not import MGBAController. Ensure project dependencies are installed.")
        raise SystemExit(1) from exc
