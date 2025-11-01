#!/usr/bin/env python3
"""Sprite Dataset Dumper - Extract labeled sprites from game runs.

This tool extracts sprite data from Pokemon MD game runs and saves them
as PNG files with corresponding CSV manifests for dataset creation.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from ..sprite_detector import DetectionResult
from ..sprite_phash import compute_phash


logger = logging.getLogger(__name__)


class SpriteDatasetDumper:
    """Dump sprite data from game runs for dataset creation."""
    
    def __init__(self, output_dir: Path):
        """Initialize the dumper.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sprites_dir = self.output_dir / "sprites"
        self.sprites_dir.mkdir(exist_ok=True)
        
        # Initialize CSV manifest
        self.manifest_path = self.output_dir / "sprite_manifest.csv"
        self.manifest_file = open(self.manifest_path, 'w', newline='')
        self.manifest_writer = csv.writer(self.manifest_file)
        
        # Write header
        self.manifest_writer.writerow([
            'sprite_id', 'timecode', 'label', 'confidence', 
            'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'phash', 'source_frame', 'category'
        ])
        
        self.sprite_count = 0
        
    def dump_frame_sprites(self, image_path: Path, frame_id: str, 
                          timecode: float, detections: List[DetectionResult]) -> int:
        """Dump sprites from a single frame.
        
        Args:
            image_path: Path to the source frame image
            frame_id: Unique identifier for the frame
            timecode: Timestamp for this frame
            detections: List of sprite detections
            
        Returns:
            Number of sprites dumped
        """
        # Load source image
        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return 0
            
        dumped_count = 0
        
        for detection in detections:
            # Skip low confidence detections
            if detection.confidence < 0.7:
                continue
                
            # Extract sprite region
            x, y, w, h = detection.bbox
            sprite_region = image.crop((x, y, x + w, y + h))
            
            # Generate sprite filename
            self.sprite_count += 1
            sprite_filename = f"sprite_{self.sprite_count:06d}_{detection.label}.png"
            sprite_path = self.sprites_dir / sprite_filename
            
            # Save sprite
            sprite_region.save(sprite_path)
            
            # Compute pHash for the sprite
            sprite_array = np.array(sprite_region.convert('L'))  # Convert to grayscale
            phash_array = compute_phash(sprite_array)
            phash_str = ''.join(map(str, phash_array.astype(int)))
            
            # Write manifest entry
            self.manifest_writer.writerow([
                self.sprite_count,
                timecode,
                detection.label,
                detection.confidence,
                x, y, w, h,
                phash_str,
                frame_id,
                detection.metadata.get('category', 'unknown')
            ])
            
            dumped_count += 1
            
        return dumped_count
        
    def close(self):
        """Close the manifest file."""
        if self.manifest_file:
            self.manifest_file.close()
            self.manifest_file = None
            
        logger.info(f"Dumped {self.sprite_count} sprites to {self.output_dir}")
        logger.info(f"Manifest saved to {self.manifest_path}")


def find_frame_files(run_dir: Path) -> List[Path]:
    """Find frame files in a run directory.
    
    Args:
        run_dir: Directory containing run data
        
    Returns:
        List of frame image files sorted by name
    """
    # Common patterns for frame files
    patterns = ["*.png", "*.jpg", "*.jpeg", "frame_*.png", "screenshot_*.png"]
    
    frame_files = []
    for pattern in patterns:
        frame_files.extend(run_dir.glob(pattern))
        
    # Sort by filename to maintain temporal order
    frame_files.sort()
    
    return frame_files


def main():
    """Main entry point for sprite dataset dumper."""
    parser = argparse.ArgumentParser(
        description="Dump sprite dataset from Pokemon MD game runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump all sprites from a run directory
  python dump_sprites.py /path/to/run/dir --output ./sprites_dataset
  
  # Process every 10th frame, limit to 100 frames
  python dump_sprites.py /path/to/run/dir --output ./sprites_dataset --stride 10 --limit 100
        """
    )
    
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing game run data with frame images"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for dumped sprites and manifest"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1, process all frames)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N frames"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for sprite detection (default: 0.7)"
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
    
    # Validate input directory
    if not args.run_dir.exists():
        logger.error(f"Run directory does not exist: {args.run_dir}")
        return 1
        
    if not args.run_dir.is_dir():
        logger.error(f"Path is not a directory: {args.run_dir}")
        return 1
    
    # Find frame files
    logger.info(f"Scanning for frame files in {args.run_dir}...")
    frame_files = find_frame_files(args.run_dir)
    
    if not frame_files:
        logger.error("No frame files found in run directory")
        return 1
        
    logger.info(f"Found {len(frame_files)} frame files")
    
    # Apply stride and limit
    if args.stride > 1:
        frame_files = frame_files[::args.stride]
        logger.info(f"After stride={args.stride}: {len(frame_files)} frames")
        
    if args.limit:
        frame_files = frame_files[:args.limit]
        logger.info(f"After limit={args.limit}: {len(frame_files)} frames")
    
    # Initialize dumper
    dumper = SpriteDatasetDumper(args.output)
    
    # Process frames
    total_sprites = 0
    
    try:
        for i, frame_path in enumerate(frame_files):
            logger.debug(f"Processing frame {i+1}/{len(frame_files)}: {frame_path.name}")
            
            # Generate frame ID and timecode
            frame_id = frame_path.stem
            timecode = i * (args.stride / 30.0)  # Assume 30 FPS if unknown
            
            # For now, use mock detections since sprite detector requires game state
            # In a real implementation, this would integrate with the actual detector
            detections = []
            
            # TODO: Integrate with actual sprite detector when game state is available
            logger.warning("Using mock detections - integrate with real detector for production")
            
            # Dump sprites from this frame
            sprites_dumped = dumper.dump_frame_sprites(frame_path, frame_id, timecode, detections)
            total_sprites += sprites_dumped
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(frame_files)} frames, {total_sprites} sprites dumped")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error processing frames: {e}")
        return 1
    finally:
        dumper.close()
    
    logger.info(f"Completed! Dumped {total_sprites} sprites from {len(frame_files)} frames")
    logger.info(f"Output directory: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())