"""ROM version gating for PMD decoders."""

import hashlib
from pathlib import Path
from typing import Optional


class ROMValidationError(Exception):
    """Raised when ROM validation fails."""


def validate_rom_sha1(expected_sha1: str, rom_path: Optional[Path] = None) -> None:
    """Validate ROM SHA-1 hash against expected value.

    Args:
        expected_sha1: Expected SHA-1 hash string
        rom_path: Path to ROM file (optional, searches in default locations)

    Raises:
        ROMValidationError: If ROM validation fails
    """
    if rom_path is None:
        # Search in default ROM directory (parent of pokemon-md-agent)
        rom_dir = Path(__file__).parent.parent.parent.parent / "rom"
        candidates = list(rom_dir.glob("*.gba"))

        if not candidates:
            raise ROMValidationError("No ROM files found in rom/ directory")

        if len(candidates) > 1:
            raise ROMValidationError(f"Multiple ROM files found: {candidates}")

        rom_path = candidates[0]

    if not rom_path.exists():
        raise ROMValidationError(f"ROM file not found: {rom_path}")

    # Compute SHA-1
    sha1 = hashlib.sha1()
    with open(rom_path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)

    computed_sha1 = sha1.hexdigest()

    if computed_sha1 != expected_sha1:
        raise ROMValidationError(
            f"ROM SHA-1 mismatch. Expected: {expected_sha1}, Got: {computed_sha1}"
        )