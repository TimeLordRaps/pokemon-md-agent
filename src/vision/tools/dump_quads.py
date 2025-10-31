#!/usr/bin/env python3
"""Quad View Dataset Dumper - Extract 4-up captures from game runs.

This tool extracts quad-view captures (environment, map, grid, meta) from 
Pokemon MD game runs and saves them with CSV manifests for analysis.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageDraw = None

from ..quad_capture import CaptureMetadata


logger = logging.getLogger(__name__)


@dataclass
class QuadCaptureEntry:
    """Entry for quad capture data."""
    capture_id: int
    timecode: float
    frame: int
    floor: int
    dungeon_id: int
    room_kind: str
    player_pos: Tuple[int, int]
    entities_count: int
    items_count: int
    # Paths to quad images
    env_image: Optional[Path] = None
    map_image: Optional[Path] = None
    grid_image: Optional[Path] = None
    meta_image: Optional[Path] = None
    ascii_available: bool = False


class QuadDatasetDumper:
    """Dump quad-view capture data from game runs for dataset creation."""
    
    def __init__(self, output_dir: Path):
        """Initialize the quad dumper.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different view types
        self.views_dir = self.output_dir / "quad_views"
        self.views_dir.mkdir(exist_ok=True)
        
        # Initialize CSV manifest
        self.manifest_path = self.output_dir / "quad_manifest.csv"
        self.manifest_file = open(self.manifest_path, 'w', newline='')
        self.manifest_writer = csv.writer(self.manifest_file)
        
        # Write header
        self.manifest_writer.writerow([
            'capture_id', 'timecode', 'frame', 'floor', 'dungeon_id',
            'room_kind', 'player_x', 'player_y', 'entities_count', 'items_count',
            'env_image', 'map_image', 'grid_image', 'meta_image', 'ascii_available'
        ])
        
        self.capture_count = 0
        
    def dump_quad_capture(self, metadata: CaptureMetadata, quad_images: Dict[str, Any]) -> int:
        """Dump a single quad capture.
        
        Args:
            metadata: Capture metadata
            quad_images: Dict mapping view names to PIL Images
            
        Returns:
            1 if capture was dumped, 0 otherwise
        """
        self.capture_count += 1
        
        # Generate capture filename base
        capture_base = f"quad_{self.capture_count:06d}_frame_{metadata.frame:06d}"
        
        image_paths = {}
        
        # Save each quad image
        for view_name, image in quad_images.items():
            if image is None:
                continue
                
            filename = f"{capture_base}_{view_name}.png"
            image_path = self.views_dir / filename
            
            try:
                image.save(image_path)
                image_paths[view_name] = image_path
                logger.debug(f"Saved {view_name} image to {image_path}")
            except Exception as e:
                logger.error(f"Failed to save {view_name} image: {e}")
                continue
        
        # Write manifest entry
        self.manifest_writer.writerow([
            self.capture_count,
            metadata.timestamp,
            metadata.frame,
            metadata.floor,
            metadata.dungeon_id,
            metadata.room_kind,
            metadata.player_pos[0],
            metadata.player_pos[1],
            metadata.entities_count,
            metadata.items_count,
            image_paths.get('environment', ''),
            image_paths.get('map', ''),
            image_paths.get('grid', ''),
            image_paths.get('meta', ''),
            metadata.ascii_available
        ])
        
        return 1 if image_paths else 0
        
    def close(self):
        """Close the manifest file."""
        if self.manifest_file:
            self.manifest_file.close()
            self.manifest_file = None
            
        logger.info(f"Dumped {self.capture_count} quad captures to {self.output_dir}")
        logger.info(f"Manifest saved to {self.manifest_path}")


def create_synthetic_quad_capture(frame_num: int, width: int = 480, height: int = 320) -> Dict[str, Any]:
    """Create synthetic quad images for testing.
    
    Args:
        frame_num: Frame number for variation
        width: Image width
        height: Image height
        
    Returns:
        Dictionary mapping view names to PIL Images
    """
    if not HAS_PIL or not Image:
        logger.warning("PIL not available, returning empty dict")
        return {}
        
    images = {}
    
    # Create different synthetic patterns for each view
    for view_name in ['environment', 'map', 'grid', 'meta']:
        img = Image.new('RGB', (width, height), color='black')
        if ImageDraw:
            draw = ImageDraw.Draw(img)
        else:
            draw = None
        
        # Add view-specific content
        if view_name == 'environment':
            # Environment: Game-like scene with player
            if draw:
                draw.rectangle([50, 50, 200, 200], fill='brown')  # Ground
                draw.ellipse([120, 80, 140, 120], fill='blue')    # Player
                draw.text((10, 10), f"Env Frame {frame_num}", fill='white')
            
        elif view_name == 'map':
            # Map: Top-down view
            if draw:
                draw.rectangle([10, 10, 100, 100], fill='darkgreen')  # Room
                draw.rectangle([20, 20, 90, 90], fill='lightgreen')   # Floor
                draw.text((10, height-20), f"Map Frame {frame_num}", fill='white')
            
        elif view_name == 'grid':
            # Grid: ASCII-like grid
            if draw:
                for i in range(0, width, 20):
                    draw.line([(i, 0), (i, height)], fill='gray')
                for i in range(0, height, 20):
                    draw.line([(0, i), (width, i)], fill='gray')
                draw.text((10, 10), f"Grid Frame {frame_num}", fill='white')
            
        elif view_name == 'meta':
            # Meta: HUD information
            if draw:
                draw.rectangle([0, 0, width, 50], fill='darkblue')  # HUD bar
                draw.text((10, 10), f"HP: {100 - frame_num % 20}", fill='white')
                draw.text((10, 25), f"Level: {1 + frame_num % 10}", fill='white')
                draw.text((10, height-20), f"Meta Frame {frame_num}", fill='white')
        
        images[view_name] = img
    
    return images


def create_synthetic_metadata(frame_num: int, floor: int = 1) -> CaptureMetadata:
    """Create synthetic metadata for testing.
    
    Args:
        frame_num: Frame number
        floor: Floor number
        
    Returns:
        Synthetic CaptureMetadata
    """
    return CaptureMetadata(
        timestamp=frame_num * (1/30),  # 30 FPS
        frame=frame_num,
        floor=floor,
        dungeon_id=1,
        room_kind="normal",
        player_pos=(120 + frame_num % 10, 80 + frame_num % 5),
        entities_count=5 + frame_num % 3,
        items_count=frame_num % 4,
        ascii_available=True
    )


def main():
    """Main entry point for quad dataset dumper."""
    parser = argparse.ArgumentParser(
        description="Dump quad-view dataset from Pokemon MD game runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic quad dataset for testing
  python dump_quads.py --synthetic --output ./quad_dataset --count 50
  
  # Dump from run directory (when real captures are available)
  python dump_quads.py /path/to/run/dir --output ./quad_dataset
        """
    )
    
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs='?',
        help="Directory containing game run data with quad captures"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for dumped quads and manifest"
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic quad data for testing"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of synthetic captures to generate (default: 100)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Width of synthetic images (default: 480)"
    )
    
    parser.add(
        "--height",
        type=int,
        default=320,
        help="Height of synthetic images (default: 320)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th capture (default: 1)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N captures"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not HAS_PIL:
        logger.error("PIL not available. Install Pillow to use quad dumper.")
        return 1
    
    # Validate input
    if not args.synthetic and not args.run_dir:
        logger.error("Either provide run_dir or use --synthetic flag")
        return 1
        
    if args.synthetic and args.run_dir:
        logger.warning("Both --synthetic and run_dir provided, using synthetic mode")
    
    # Initialize dumper
    dumper = QuadDatasetDumper(args.output)
    
    # Process captures
    total_captures = 0
    
    try:
        if args.synthetic:
            logger.info(f"Generating {args.count} synthetic quad captures...")
            
            for i in range(args.count):
                # Apply stride
                if args.stride > 1 and i % args.stride != 0:
                    continue
                    
                # Apply limit
                if args.limit and i >= args.limit:
                    break
                
                # Create synthetic data
                metadata = create_synthetic_metadata(i)
                quad_images = create_synthetic_quad_capture(i, args.width, args.height)
                
                # Dump capture
                captures_dumped = dumper.dump_quad_capture(metadata, quad_images)
                total_captures += captures_dumped
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i+1}/{args.count} synthetic captures")
                    
        else:
            # Process real captures from run directory
            logger.info(f"Scanning for quad captures in {args.run_dir}...")
            # TODO: Implement real capture processing when available
            
            logger.warning("Real capture processing not yet implemented")
            logger.info("Use --synthetic flag for testing")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error processing captures: {e}")
        return 1
    finally:
        dumper.close()
    
    logger.info(f"Completed! Dumped {total_captures} quad captures")
    logger.info(f"Output directory: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())