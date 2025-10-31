#!/usr/bin/env python3
"""
CLI Tool for Live Emulator Data Dumping - analyze_dumps.py

This script provides live data dumping capabilities for Pokemon Mystery Dungeon
emulator sessions. It connects to mGBA and dumps WRAM data to JSON files for
analysis and regression testing.

Features:
- Live WRAM dumping to timestamped JSON files
- Monster entity data extraction
- Configurable output directory
- Safety guards and error handling
- Integration with WRAMDecoderV2 prototype
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from environment.mgba_controller import MGBAController
except ImportError as e:
    print(f"❌ Failed to import MGBAController: {e}")
    print("Make sure you're running from the pokemon-md-agent directory")
    sys.exit(1)

# Optional import for decoder v2
try:
    from prototypes.wram_decoder_fix.decoder_v2 import WRAMDecoderV2, decode_first_mon
    DECODER_V2_AVAILABLE = True
except ImportError:
    print("⚠️  WRAMDecoderV2 not available - some features will be limited")
    DECODER_V2_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveDumper:
    """Handles live data dumping from mGBA emulator."""

    def __init__(self, controller: MGBAController, output_dir: Path):
        """Initialize dumper with controller and output directory.

        Args:
            controller: Connected MGBAController instance
            output_dir: Directory to save dump files
        """
        self.controller = controller
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache address manager
        self.address_manager = controller.address_manager

    def _generate_timestamp(self) -> str:
        """Generate timestamp string for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _read_wram_range(self, start_addr: int, size: int) -> Optional[bytes]:
        """Read a range of WRAM data.

        Args:
            start_addr: Starting address (absolute)
            size: Number of bytes to read

        Returns:
            Raw bytes data or None if failed
        """
        try:
            data = self.controller.peek(start_addr, size)
            if data is None or len(data) != size:
                logger.warning(f"Failed to read {size} bytes from 0x{start_addr:08X}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error reading WRAM range 0x{start_addr:08X}-{start_addr+size:08X}: {e}")
            return None

    def dump_wram_snapshot(self, filename: Optional[str] = None) -> Optional[Path]:
        """Dump complete WRAM snapshot to file.

        Args:
            filename: Optional filename (will generate timestamped name if None)

        Returns:
            Path to created file or None if failed
        """
        if filename is None:
            filename = f"wram_snapshot_{self._generate_timestamp()}.json"

        filepath = self.output_dir / filename

        try:
            # WRAM is 64KB (0x02000000 - 0x02010000)
            wram_start = 0x02000000
            wram_size = 0x10000  # 64KB

            logger.info(f"Dumping WRAM snapshot ({wram_size} bytes) to {filepath}")

            wram_data = self._read_wram_range(wram_start, wram_size)
            if wram_data is None:
                logger.error("Failed to read WRAM data")
                return None

            # Create dump data structure
            dump_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "wram_start": wram_start,
                    "wram_size": wram_size,
                    "game_title": self.controller.get_game_title(),
                    "game_code": self.controller.get_game_code(),
                    "dumper_version": "1.0",
                },
                "wram_hex": wram_data.hex(),
                "wram_bytes": list(wram_data),  # For easier JSON parsing
            }

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2)

            logger.info(f"✅ WRAM snapshot saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to dump WRAM snapshot: {e}")
            return None

    def dump_monster_entities(self, filename: Optional[str] = None) -> Optional[Path]:
        """Dump monster entity data to file.

        Args:
            filename: Optional filename (will generate timestamped name if None)

        Returns:
            Path to created file or None if failed
        """
        if filename is None:
            filename = f"monster_entities_{self._generate_timestamp()}.json"

        filepath = self.output_dir / filename

        try:
            logger.info("Dumping monster entity data")

            # Get basic game state
            game_state = {
                "floor": self.controller.get_floor(),
                "player_pos": self.controller.get_player_position(),
                "player_stats": self.controller.get_player_stats(),
            }

            # Try to use decoder v2 if available
            monsters_data = None
            if DECODER_V2_AVAILABLE:
                try:
                    decoder = WRAMDecoderV2(self.controller)
                    monsters_data = decoder.decode_all_monsters()
                    logger.info(f"Decoded {len(monsters_data) if monsters_data else 0} monsters with v2 decoder")
                except Exception as e:
                    logger.warning(f"Decoder v2 failed: {e}")

            # Fallback: read raw monster list data
            if monsters_data is None:
                monsters_data = self._dump_raw_monster_data()

            # Create dump data structure
            dump_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "game_title": self.controller.get_game_title(),
                    "game_code": self.controller.get_game_code(),
                    "dumper_version": "1.0",
                    "decoder_used": "v2" if DECODER_V2_AVAILABLE and monsters_data else "raw",
                },
                "game_state": game_state,
                "monsters": monsters_data or [],
            }

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2)

            logger.info(f"✅ Monster entities saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to dump monster entities: {e}")
            return None

    def _dump_raw_monster_data(self) -> Optional[list]:
        """Dump raw monster data without decoder (fallback method).

        Returns:
            List of raw monster data dicts or None if failed
        """
        try:
            # Get monster list info
            list_ptr_addr = self.address_manager.get_address("entities", "monster_list_ptr")
            count_addr = self.address_manager.get_address("entities", "monster_count")

            list_ptr_data = self._read_wram_range(list_ptr_addr, 4)
            count_data = self._read_wram_range(count_addr, 1)

            if list_ptr_data is None or count_data is None:
                return None

            list_ptr = int.from_bytes(list_ptr_data, byteorder='little')
            count = int.from_bytes(count_data, byteorder='little')

            if count == 0:
                return []

            monsters = []
            for i in range(min(count, 20)):  # Limit to 20 for safety
                monster_addr = list_ptr + (i * 48)  # 48 bytes per struct
                monster_data = self._read_wram_range(monster_addr, 48)

                if monster_data is None:
                    continue

                monsters.append({
                    "index": i,
                    "address": monster_addr,
                    "raw_bytes": monster_data.hex(),
                    "raw_bytes_list": list(monster_data),
                })

            return monsters

        except Exception as e:
            logger.error(f"Failed to dump raw monster data: {e}")
            return None

    def dump_first_monster(self, filename: Optional[str] = None) -> Optional[Path]:
        """Dump first monster data using decoder v2.

        Args:
            filename: Optional filename (will generate timestamped name if None)

        Returns:
            Path to created file or None if failed
        """
        if not DECODER_V2_AVAILABLE:
            logger.error("Decoder v2 not available for first monster dump")
            return None

        if filename is None:
            filename = f"first_monster_{self._generate_timestamp()}.json"

        filepath = self.output_dir / filename

        try:
            logger.info("Dumping first monster data")

            # Use convenience function
            monster_data = decode_first_mon(self.controller)

            if monster_data is None:
                logger.warning("No first monster data available")
                return None

            # Create dump data structure
            dump_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "game_title": self.controller.get_game_title(),
                    "game_code": self.controller.get_game_code(),
                    "dumper_version": "1.0",
                    "decoder_version": "v2",
                },
                "first_monster": monster_data,
            }

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2)

            logger.info(f"✅ First monster data saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to dump first monster: {e}")
            return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Live emulator data dumping tool for Pokemon Mystery Dungeon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump WRAM snapshot
  python analyze_dumps.py --wram

  # Dump monster entities
  python analyze_dumps.py --monsters

  # Dump first monster (requires MD_DECODER_V2=1)
  python analyze_dumps.py --first-monster

  # Dump everything
  python analyze_dumps.py --all

  # Custom output directory
  python analyze_dumps.py --all --output-dir ./my_dumps
        """
    )

    parser.add_argument(
        "--wram",
        action="store_true",
        help="Dump complete WRAM snapshot"
    )

    parser.add_argument(
        "--monsters",
        action="store_true",
        help="Dump monster entity data"
    )

    parser.add_argument(
        "--first-monster",
        action="store_true",
        help="Dump first monster data (requires decoder v2)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Dump all available data types"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dumps"),
        help="Output directory for dump files (default: ./dumps)"
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="mGBA server host (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="mGBA server port (default: 8888)"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10.0)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine what to dump
    if args.all:
        dump_wram = dump_monsters = dump_first = True
    else:
        dump_wram = args.wram
        dump_monsters = args.monsters
        dump_first = args.first_monster

    if not any([dump_wram, dump_monsters, dump_first]):
        parser.error("Must specify at least one dump type (--wram, --monsters, --first-monster, or --all)")

    # Check for decoder v2 requirement
    if dump_first and not DECODER_V2_AVAILABLE:
        parser.error("--first-monster requires WRAMDecoderV2 (set MD_DECODER_V2=1)")

    # Connect to mGBA
    logger.info(f"Connecting to mGBA at {args.host}:{args.port}")
    controller = MGBAController(
        host=args.host,
        port=args.port,
        timeout=args.timeout
    )

    try:
        if not controller.connect_with_retry():
            logger.error("Failed to connect to mGBA")
            sys.exit(1)

        logger.info("✅ Connected to mGBA successfully")

        # Create dumper
        dumper = LiveDumper(controller, args.output_dir)

        # Perform dumps
        dumped_files = []

        if dump_wram:
            logger.info("Dumping WRAM snapshot...")
            filepath = dumper.dump_wram_snapshot()
            if filepath:
                dumped_files.append(filepath)

        if dump_monsters:
            logger.info("Dumping monster entities...")
            filepath = dumper.dump_monster_entities()
            if filepath:
                dumped_files.append(filepath)

        if dump_first:
            logger.info("Dumping first monster...")
            filepath = dumper.dump_first_monster()
            if filepath:
                dumped_files.append(filepath)

        # Summary
        if dumped_files:
            logger.info("✅ Dump complete!")
            logger.info(f"Files created in {args.output_dir}:")
            for filepath in dumped_files:
                logger.info(f"  - {filepath.name}")
        else:
            logger.warning("⚠️  No files were created")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()