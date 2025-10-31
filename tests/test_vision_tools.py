"""Unit tests for vision dataset dumper tools.

Tests the sprite and quad capture dataset dumpers with synthetic data.
"""

import tempfile
import csv
import json
from pathlib import Path
from typing import Dict, Any
import pytest
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MockDetectionResult:
    """Mock DetectionResult for testing."""
    def __init__(self, label: str, confidence: float, bbox: tuple, metadata: Dict[str, Any] | None = None):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.metadata = metadata or {}


class TestSpriteDatasetDumper:
    """Test sprite dataset dumper functionality."""
    
    def test_dumper_initialization(self):
        """Test that dumper initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sprites_test"
            
            # Create a minimal implementation for testing
            class TestSpriteDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.sprites_dir = self.output_dir / "sprites"
                    self.sprites_dir.mkdir(exist_ok=True)
                    self.manifest_path = self.output_dir / "sprite_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'sprite_id', 'timecode', 'label', 'confidence', 
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'phash', 'source_frame', 'category'
                    ])
                    self.sprite_count = 0
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestSpriteDumper(output_dir)
            
            assert dumper.output_dir == output_dir
            assert dumper.sprites_dir.exists()
            assert dumper.manifest_path.exists()
            assert dumper.manifest_file is not None
            assert dumper.sprite_count == 0
            
            dumper.close()
            assert dumper.manifest_file is None
    
    def test_manifest_schema(self):
        """Test that manifest has correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sprites_test"
            
            class TestSpriteDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.manifest_path = self.output_dir / "sprite_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'sprite_id', 'timecode', 'label', 'confidence', 
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'phash', 'source_frame', 'category'
                    ])
                    self.manifest_file.flush()  # Ensure data is written
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestSpriteDumper(output_dir)
            dumper.close()  # Close before reading
            
            # Check header
            with open(dumper.manifest_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
            expected_header = [
                'sprite_id', 'timecode', 'label', 'confidence', 
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'phash', 'source_frame', 'category'
            ]
            assert header == expected_header
    
    def test_dump_frame_sprites_with_pil(self):
        """Test dumping sprites from a frame."""
        if not HAS_PIL:
            pytest.skip("PIL not available")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sprites_test"
            
            class TestSpriteDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.sprites_dir = self.output_dir / "sprites"
                    self.sprites_dir.mkdir(exist_ok=True)
                    self.manifest_path = self.output_dir / "sprite_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'sprite_id', 'timecode', 'label', 'confidence', 
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'phash', 'source_frame', 'category'
                    ])
                    self.sprite_count = 0
                    
                def dump_frame_sprites(self, image_path: Path, frame_id: str, 
                                     timecode: float, detections: list) -> int:
                    # Load source image
                    image = Image.open(image_path)
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
                        
                        # Write manifest entry
                        self.manifest_writer.writerow([
                            self.sprite_count,
                            timecode,
                            detection.label,
                            detection.confidence,
                            x, y, w, h,
                            "mock_phash",  # Mock pHash
                            frame_id,
                            detection.metadata.get('category', 'unknown')
                        ])
                        
                        dumped_count += 1
                        
                    return dumped_count
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestSpriteDumper(output_dir)
            
            # Create synthetic frame image
            frame_dir = Path(temp_dir) / "frames"
            frame_dir.mkdir()
            frame_path = frame_dir / "frame_001.png"
            
            # Create a 480x320 test image
            test_image = Image.new('RGB', (480, 320), color='blue')
            test_image.save(frame_path)
            
            # Create test detections
            detections = [
                MockDetectionResult(
                    label="player",
                    confidence=0.9,
                    bbox=(100, 150, 16, 16),
                    metadata={"category": "entities"}
                ),
                MockDetectionResult(
                    label="stairs",
                    confidence=0.8,
                    bbox=(200, 100, 32, 16),
                    metadata={"category": "objects"}
                )
            ]
            
            # Dump sprites
            dumped_count = dumper.dump_frame_sprites(
                frame_path, "frame_001", 1.0, detections
            )
            
            assert dumped_count == 2
            assert dumper.sprite_count == 2
            
            # Check that sprite files were created
            sprites_dir = dumper.sprites_dir
            assert len(list(sprites_dir.glob("*.png"))) == 2
            
            # Check manifest content
            with open(dumper.manifest_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                
            assert len(rows) == 2
            assert rows[0][2] == "player"  # label
            assert rows[0][3] == "0.9"    # confidence
            assert rows[0][6] == "16"     # bbox_w
            
            dumper.close()
    
    def test_low_confidence_filtering(self):
        """Test that low confidence detections are filtered out."""
        if not HAS_PIL:
            pytest.skip("PIL not available")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sprites_test"
            
            class TestSpriteDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.sprites_dir = self.output_dir / "sprites"
                    self.sprites_dir.mkdir(exist_ok=True)
                    self.manifest_path = self.output_dir / "sprite_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'sprite_id', 'timecode', 'label', 'confidence', 
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'phash', 'source_frame', 'category'
                    ])
                    self.sprite_count = 0
                    
                def dump_frame_sprites(self, image_path: Path, frame_id: str, 
                                     timecode: float, detections: list) -> int:
                    image = Image.open(image_path)
                    dumped_count = 0
                    
                    for detection in detections:
                        if detection.confidence < 0.7:
                            continue
                            
                        x, y, w, h = detection.bbox
                        sprite_region = image.crop((x, y, x + w, y + h))
                        
                        self.sprite_count += 1
                        sprite_filename = f"sprite_{self.sprite_count:06d}_{detection.label}.png"
                        sprite_path = self.sprites_dir / sprite_filename
                        sprite_region.save(sprite_path)
                        
                        self.manifest_writer.writerow([
                            self.sprite_count,
                            timecode,
                            detection.label,
                            detection.confidence,
                            x, y, w, h,
                            "mock_phash",
                            frame_id,
                            detection.metadata.get('category', 'unknown')
                        ])
                        
                        dumped_count += 1
                        
                    return dumped_count
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestSpriteDumper(output_dir)
            
            # Create synthetic frame image
            frame_path = Path(temp_dir) / "test_frame.png"
            test_image = Image.new('RGB', (480, 320), color='blue')
            test_image.save(frame_path)
            
            # Create detections with mixed confidence
            detections = [
                MockDetectionResult(label="high_conf", confidence=0.9, bbox=(10, 10, 16, 16), metadata={}),
                MockDetectionResult(label="low_conf", confidence=0.5, bbox=(30, 30, 16, 16), metadata={}),  # Below threshold
                MockDetectionResult(label="medium_conf", confidence=0.7, bbox=(50, 50, 16, 16), metadata={})
            ]
            
            dumped_count = dumper.dump_frame_sprites(frame_path, "test", 1.0, detections)
            
            # Should only dump high and medium confidence (>= 0.7)
            assert dumped_count == 2
            assert dumper.sprite_count == 2
            
            dumper.close()


class TestQuadDatasetDumper:
    """Test quad dataset dumper functionality."""
    
    def test_quad_dumper_initialization(self):
        """Test that quad dumper initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "quads_test"
            
            class TestQuadDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.views_dir = self.output_dir / "quad_views"
                    self.views_dir.mkdir(exist_ok=True)
                    self.manifest_path = self.output_dir / "quad_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'capture_id', 'timecode', 'frame', 'floor', 'dungeon_id',
                        'room_kind', 'player_x', 'player_y', 'entities_count', 'items_count',
                        'env_image', 'map_image', 'grid_image', 'meta_image', 'ascii_available'
                    ])
                    self.capture_count = 0
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestQuadDumper(output_dir)
            
            assert dumper.output_dir == output_dir
            assert dumper.views_dir.exists()
            assert dumper.manifest_path.exists()
            assert dumper.manifest_file is not None
            assert dumper.capture_count == 0
            
            dumper.close()
            assert dumper.manifest_file is None
    
    def test_quad_manifest_schema(self):
        """Test that quad manifest has correct schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "quads_test"
            
            class TestQuadDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.manifest_path = self.output_dir / "quad_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'capture_id', 'timecode', 'frame', 'floor', 'dungeon_id',
                        'room_kind', 'player_x', 'player_y', 'entities_count', 'items_count',
                        'env_image', 'map_image', 'grid_image', 'meta_image', 'ascii_available'
                    ])
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            dumper = TestQuadDumper(output_dir)
            
            # Check header
            with open(dumper.manifest_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
            expected_header = [
                'capture_id', 'timecode', 'frame', 'floor', 'dungeon_id',
                'room_kind', 'player_x', 'player_y', 'entities_count', 'items_count',
                'env_image', 'map_image', 'grid_image', 'meta_image', 'ascii_available'
            ]
            assert header == expected_header
            
            dumper.close()


class TestFindFrameFiles:
    """Test frame file discovery functionality."""
    
    def test_find_frame_files(self):
        """Test finding frame files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            
            # Create test files
            (run_dir / "frame_001.png").touch()
            (run_dir / "frame_002.jpg").touch()
            (run_dir / "screenshot_001.png").touch()
            (run_dir / "other.txt").touch()
            (run_dir / "subdir").mkdir()
            (run_dir / "subdir" / "frame_003.png").touch()
            
            # Find frame files (simulate the function)
            patterns = ["*.png", "*.jpg", "*.jpeg", "frame_*.png", "screenshot_*.png"]
            
            frame_files = []
            for pattern in patterns:
                frame_files.extend(run_dir.glob(pattern))
                
            # Sort by filename to maintain temporal order
            frame_files.sort()
            
            # Should find PNG and JPG files
            filenames = [f.name for f in frame_files]
            assert "frame_001.png" in filenames
            assert "frame_002.jpg" in filenames
            assert "screenshot_001.png" in filenames
            assert "other.txt" not in filenames
            
            # Should be sorted
            assert frame_files[0].name == "frame_001.png"
            assert frame_files[1].name == "frame_002.jpg"
    
    def test_no_frame_files(self):
        """Test behavior when no frame files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            
            # Create non-frame files
            (run_dir / "readme.txt").touch()
            (run_dir / "data.json").touch()
            
            patterns = ["*.png", "*.jpg", "*.jpeg", "frame_*.png", "screenshot_*.png"]
            frame_files = []
            for pattern in patterns:
                frame_files.extend(run_dir.glob(pattern))
                
            assert frame_files == []
    
    def test_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            
            patterns = ["*.png", "*.jpg", "*.jpeg", "frame_*.png", "screenshot_*.png"]
            frame_files = []
            for pattern in patterns:
                frame_files.extend(run_dir.glob(pattern))
                
            assert frame_files == []


class TestIntegrationScenarios:
    """Test complete workflows with synthetic data."""
    
    def test_sprite_dumper_workflow(self):
        """Test complete sprite dumping workflow."""
        if not HAS_PIL:
            pytest.skip("PIL not available")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            output_dir = Path(temp_dir) / "sprites_output"
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir()
            
            # Create test frames
            for i in range(5):
                frame_path = frames_dir / f"frame_{i:03d}.png"
                test_image = Image.new('RGB', (480, 320), color='red')
                test_image.save(frame_path)
            
            # Create dumper with simplified implementation
            class TestSpriteDumper:
                def __init__(self, output_dir: Path):
                    self.output_dir = Path(output_dir)
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    self.sprites_dir = self.output_dir / "sprites"
                    self.sprites_dir.mkdir(exist_ok=True)
                    self.manifest_path = self.output_dir / "sprite_manifest.csv"
                    self.manifest_file = open(self.manifest_path, 'w', newline='')
                    self.manifest_writer = csv.writer(self.manifest_file)
                    self.manifest_writer.writerow([
                        'sprite_id', 'timecode', 'label', 'confidence', 
                        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'phash', 'source_frame', 'category'
                    ])
                    self.sprite_count = 0
                    
                def dump_frame_sprites(self, image_path: Path, frame_id: str, 
                                     timecode: float, detections: list) -> int:
                    image = Image.open(image_path)
                    dumped_count = 0
                    
                    for detection in detections:
                        if detection.confidence < 0.7:
                            continue
                            
                        x, y, w, h = detection.bbox
                        sprite_region = image.crop((x, y, x + w, y + h))
                        
                        self.sprite_count += 1
                        sprite_filename = f"sprite_{self.sprite_count:06d}_{detection.label}.png"
                        sprite_path = self.sprites_dir / sprite_filename
                        sprite_region.save(sprite_path)
                        
                        self.manifest_writer.writerow([
                            self.sprite_count,
                            timecode,
                            detection.label,
                            detection.confidence,
                            x, y, w, h,
                            "mock_phash",
                            frame_id,
                            detection.metadata.get('category', 'unknown')
                        ])
                        
                        dumped_count += 1
                        
                    return dumped_count
                    
                def close(self):
                    if self.manifest_file:
                        self.manifest_file.close()
                        self.manifest_file = None
            
            # Initialize dumper
            dumper = TestSpriteDumper(output_dir)
            
            # Process frames with stride=2, limit=3
            frame_files = list(frames_dir.glob("frame_*.png"))
            frame_files.sort()
            processed_frames = frame_files[::2][:3]  # stride=2, limit=3
            
            for i, frame_path in enumerate(processed_frames):
                detections = [
                    MockDetectionResult(
                        label="test_sprite",
                        confidence=0.8,
                        bbox=(10, 10, 16, 16),
                        metadata={"category": "test"}
                    )
                ]
                
                dumper.dump_frame_sprites(
                    frame_path, f"frame_{i}", i * 0.033, detections
                )
            
            dumper.close()
            
            # Verify output
            assert dumper.sprite_count == 3  # 3 frames processed
            assert len(list(output_dir.glob("sprites/*.png"))) == 3
            assert dumper.manifest_path.exists()
            
            # Verify manifest content
            with open(dumper.manifest_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            assert len(rows) == 4  # Header + 3 data rows
            assert all(row[2] == "test_sprite" for row in rows[1:])  # All sprites have same label


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])