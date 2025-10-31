"""ROM version gating for PMD decoders."""

import hashlib
from pathlib import Path
from typing import List, Optional


class ROMValidationError(Exception):
    """Raised when ROM validation fails."""


def find_rom_files(rom_dir: Optional[Path] = None) -> List[Path]:
    """Find all .gba ROM files in the rom directory.
    
    Args:
        rom_dir: Directory to search for ROMs (defaults to ../rom from src/)
        
    Returns:
        List of .gba files found
        
    Raises:
        ROMValidationError: If rom_dir doesn't exist
    """
    if rom_dir is None:
        # Search in default ROM directory (parent of pokemon-md-agent)
        rom_dir = Path(__file__).parent.parent.parent.parent / "rom"
    
    if not rom_dir.exists():
        raise ROMValidationError(
            f"ROM directory not found: {rom_dir}\n"
            f"Please ensure the ROM directory exists and contains .gba files.\n"
            f"Expected path: {rom_dir.absolute()}"
        )
    
    if not rom_dir.is_dir():
        raise ROMValidationError(
            f"ROM path is not a directory: {rom_dir}\n"
            f"Please ensure this path points to a directory containing ROM files."
        )
    
    candidates = list(rom_dir.glob("*.gba"))
    return sorted(candidates)


def get_rom_info(rom_path: Path) -> dict:
    """Get information about a ROM file.
    
    Args:
        rom_path: Path to ROM file
        
    Returns:
        Dictionary with ROM information
        
    Raises:
        ROMValidationError: If ROM file is invalid
    """
    if not rom_path.exists():
        raise ROMValidationError(f"ROM file not found: {rom_path}")
    
    if not rom_path.is_file():
        raise ROMValidationError(f"ROM path is not a file: {rom_path}")
    
    if rom_path.suffix.lower() != ".gba":
        raise ROMValidationError(f"Invalid ROM file type: {rom_path} (expected .gba)")
    
    try:
        # Compute SHA-1 hash
        sha1_hash = hashlib.sha1()
        file_size = rom_path.stat().st_size
        
        with open(rom_path, 'rb') as f:
            while chunk := f.read(8192):
                sha1_hash.update(chunk)
        
        return {
            "path": rom_path,
            "name": rom_path.name,
            "size": file_size,
            "sha1": sha1_hash.hexdigest(),
            "exists": True
        }
    except Exception as e:
        raise ROMValidationError(f"Failed to read ROM file {rom_path}: {e}") from e


def validate_rom_sha1(expected_sha1: str, rom_path: Optional[Path] = None) -> Path:
    """Validate ROM SHA-1 hash against expected value and return the ROM path.

    Args:
        expected_sha1: Expected SHA-1 hash string
        rom_path: Path to ROM file (optional, searches in default locations)

    Returns:
        Validated ROM Path

    Raises:
        ROMValidationError: If ROM validation fails
    """
    if rom_path is None:
        # Search for ROM files in default locations
        rom_files = find_rom_files()
        
        if not rom_files:
            rom_dir = Path(__file__).parent.parent.parent.parent / "rom"
            raise ROMValidationError(
                f"No .gba ROM files found in rom/ directory\n"
                f"Searched in: {rom_dir.absolute()}\n"
                f"Please ensure you have a Pokemon Mystery Dungeon .gba file in this directory."
            )
        
        if len(rom_files) > 1:
            rom_names = [f.name for f in rom_files]
            raise ROMValidationError(
                "Multiple ROM files found. Please specify one:\n" +
                "\n".join(f"  - {name}" for name in rom_names) +
                "\n\nExpected files: Pokemon Mystery Dungeon - Red Rescue Team.gba"
            )
        
        rom_path = rom_files[0]
    
    # Get ROM info and validate
    rom_info = get_rom_info(rom_path)
    computed_sha1 = rom_info["sha1"]
    
    if computed_sha1 != expected_sha1:
        raise ROMValidationError(
            f"ROM SHA-1 mismatch for {rom_info['name']}\n"
            f"Expected: {expected_sha1}\n"
            f"Found:    {computed_sha1}\n"
            f"This may not be the correct Pokemon Mystery Dungeon ROM."
        )
    
    logger.info(f"ROM validated: {rom_info['name']} ({file_size_str(rom_info['size'])})")
    return rom_path


def file_size_str(size_bytes: int) -> str:
    """Convert file size in bytes to human readable string."""
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def detect_pm_red_rom() -> Optional[Path]:
    """Detect Pokemon Mystery Dungeon Red Rescue Team ROM.
    
    Returns:
        Path to detected ROM or None if not found
    """
    try:
        rom_files = find_rom_files()
        if not rom_files:
            return None
        
        # Look for PMD Red Rescue Team patterns
        for rom_path in rom_files:
            name_lower = rom_path.name.lower()
            if "pokemon mystery dungeon" in name_lower and "red" in name_lower:
                return rom_path
        
        # If no exact match, return the first ROM found
        return rom_files[0] if rom_files else None
    except ROMValidationError:
        return None


def validate_pm_red_rom(rom_path: Optional[Path] = None) -> Path:
    """Validate Pokemon Mystery Dungeon Red Rescue Team ROM.
    
    Args:
        rom_path: Path to ROM file (optional, will auto-detect)
        
    Returns:
        Validated ROM Path
        
    Raises:
        ROMValidationError: If ROM validation fails
    """
    # Known good SHA-1 for Pokemon Mystery Dungeon - Red Rescue Team (USA)
    PMD_RED_SHA1 = "a386c752b9c6d8e91a4e16e7e58b7c5f2a4d8e9c"  # Example hash
    
    return validate_rom_sha1(PMD_RED_SHA1, rom_path)


# Import logger
import logging
logger = logging.getLogger(__name__)