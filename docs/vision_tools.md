# Vision Tools - Dataset Dumpers

## Overview

The vision tools package provides non-runtime helper utilities for extracting and analyzing sprite and quad-view data from Pokemon MD game runs. These tools are designed for dataset creation, analysis, and debugging without interfering with core runtime performance.

## Architecture

```
src/vision/tools/
├── dump_sprites.py    # Sprite extraction and dataset creation
├── dump_quads.py      # Quad-view capture extraction
└── __init__.py        # Package initialization
```

## Features

### Sprite Dataset Dumper (`dump_sprites.py`)

Extracts labeled sprites from game runs and creates a structured dataset with:

- **PNG sprite files**: Individual extracted sprites with standardized naming
- **CSV manifest**: Comprehensive metadata including:
  - Sprite ID and timecode
  - Label and confidence scores  
  - Bounding box coordinates (x, y, w, h)
  - Perceptual hash (pHash) for similarity matching
  - Source frame reference
  - Category classification

#### CLI Usage

```bash
# Basic sprite extraction from run directory
python -m vision.tools.dump_sprites /path/to/run/dir --output ./sprites_dataset

# Process with sampling (every 10th frame, max 100 frames)
python -m vision.tools.dump_sprites /path/to/run/dir --output ./sprites_dataset --stride 10 --limit 100

# Adjust confidence threshold for sprite filtering
python -m vision.tools.dump_sprites /path/to/run/dir --output ./sprites_dataset --confidence-threshold 0.8

# Enable verbose logging
python -m vision.tools.dump_sprites /path/to/run/dir --output ./sprites_dataset --verbose
```

#### Key Parameters

- `run_dir`: Directory containing game run frame images
- `--output, -o`: Output directory for sprites and manifest
- `--stride`: Process every N-th frame (default: 1)
- `--limit`: Maximum number of frames to process
- `--confidence-threshold`: Minimum detection confidence (default: 0.7)
- `--verbose, -v`: Enable detailed logging

### Quad Dataset Dumper (`dump_quads.py`)

Extracts 4-up capture data (environment, map, grid, meta views) for comprehensive analysis:

- **Quad image files**: Separate PNG files for each view type
- **CSV manifest**: Capture metadata including:
  - Capture ID and timecode
  - Frame, floor, and dungeon information
  - Player position and entity counts
  - ASCII availability flags
  - Reference to all quad view images

#### CLI Usage

```bash
# Generate synthetic dataset for testing
python -m vision.tools.dump_quads --synthetic --output ./quad_dataset --count 50

# Specify image dimensions for synthetic data
python -m vision.tools.dump_quads --synthetic --output ./quad_dataset --width 640 --height 480

# Process real captures (when available)
python -m vision.tools.dump_quads /path/to/run/dir --output ./quad_dataset
```

#### Key Parameters

- `run_dir`: Directory containing real quad capture data (optional)
- `--synthetic`: Generate synthetic test data
- `--count`: Number of synthetic captures (default: 100)
- `--width, --height`: Image dimensions for synthetic data
- `--stride`: Process every N-th capture
- `--limit`: Maximum captures to process
- `--verbose, -v`: Enable detailed logging

## Dataset Structure

### Sprite Dataset Layout

```
sprites_dataset/
├── sprites/                    # Extracted sprite images
│   ├── sprite_000001_player.png
│   ├── sprite_000002_stairs.png
│   └── ...
└── sprite_manifest.csv        # Comprehensive metadata
```

### Quad Dataset Layout

```
quad_dataset/
├── quad_views/                # Quad capture images
│   ├── quad_000001_frame_000000_environment.png
│   ├── quad_000001_frame_000000_map.png
│   ├── quad_000001_frame_000000_grid.png
│   ├── quad_000001_frame_000000_meta.png
│   └── ...
└── quad_manifest.csv          # Capture metadata
```

## Manifest Schema

### Sprite Manifest (`sprite_manifest.csv`)

| Column | Type | Description |
|--------|------|-------------|
| sprite_id | int | Unique sprite identifier |
| timecode | float | Timestamp in seconds |
| label | string | Sprite classification |
| confidence | float | Detection confidence score |
| bbox_x | int | Bounding box x coordinate |
| bbox_y | int | Bounding box y coordinate |
| bbox_w | int | Bounding box width |
| bbox_h | int | Bounding box height |
| phash | string | 64-bit perceptual hash |
| source_frame | string | Source frame identifier |
| category | string | Sprite category classification |

### Quad Manifest (`quad_manifest.csv`)

| Column | Type | Description |
|--------|------|-------------|
| capture_id | int | Unique capture identifier |
| timecode | float | Timestamp in seconds |
| frame | int | Frame number |
| floor | int | Dungeon floor number |
| dungeon_id | int | Dungeon identifier |
| room_kind | string | Room type classification |
| player_x | int | Player x position |
| player_y | int | Player y position |
| entities_count | int | Number of entities in frame |
| items_count | int | Number of items in frame |
| env_image | path | Environment view image path |
| map_image | path | Map view image path |
| grid_image | path | Grid view image path |
| meta_image | path | Meta view image path |
| ascii_available | bool | ASCII representation available |

## Integration with Core Systems

### Perceptual Hashing

Both tools leverage the existing `sprite_phash.py` module for:

- **Deterministic hashing**: Fixed 32x32 downsampling with DCT
- **Similarity matching**: Hamming distance thresholding (≤8 bits)
- **Golden hash validation**: Synthetic sprite testing with known patterns

### Sprite Detection Pipeline

The sprite dumper integrates with the existing detection infrastructure:

- **YAML-based labeling**: Compatible with `SpriteLabels` configuration
- **Confidence filtering**: Configurable thresholds for quality control
- **Category mapping**: Structured classification system

### Frame Processing

Supports multiple input patterns for flexibility:

- **PNG/JPG images**: Standard image formats
- **Sequential naming**: `frame_001.png`, `frame_002.png`, etc.
- **Timestamp naming**: `screenshot_20231201_143022.png`
- **Stride sampling**: Efficient large dataset processing
- **Temporal ordering**: Automatic file sorting

## Testing and Validation

### Unit Tests (`tests/test_vision_tools.py`)

Comprehensive test coverage includes:

- **Dumper initialization**: Directory structure and file creation
- **Manifest schema validation**: Column headers and data types
- **Sprite extraction**: Image cropping and metadata generation
- **Confidence filtering**: Quality threshold enforcement
- **Frame file discovery**: Pattern matching and sorting
- **Integration workflows**: End-to-end processing scenarios
- **Synthetic data generation**: Quad view creation and metadata

#### Running Tests

```bash
# Run all vision tools tests
python -m pytest tests/test_vision_tools.py -v

# Run specific test class
python -m pytest tests/test_vision_tools.py::TestSpriteDatasetDumper -v

# Run with coverage
python -m pytest tests/test_vision_tools.py --cov=src.vision.tools
```

### Synthetic Data Validation

For testing without real game data:

- **Golden sprites**: Synthetic patterns with known pHash values
- **Boundary testing**: Exact threshold validation (8 bits)
- **Error handling**: Invalid input and edge case coverage
- **Performance metrics**: Processing speed and memory usage

## Performance Considerations

### Memory Management

- **Streaming processing**: Large datasets without full load
- **Tempfile cleanup**: Automatic file descriptor management
- **CSV buffering**: Efficient manifest writing
- **Image optimization**: Configurable quality settings

### Processing Speed

- **Stride sampling**: Reduced processing for large runs
- **Confidence filtering**: Early rejection of low-quality detections
- **Parallel potential**: Independent frame processing capability
- **Progress logging**: Real-time processing feedback

### Storage Efficiency

- **Standardized naming**: Predictable file organization
- **Metadata consolidation**: Single CSV per dataset type
- **Format optimization**: PNG compression for sprites
- **Incremental processing**: Resume interrupted operations

## Future Enhancements

### Planned Features

- **Real-time capture integration**: Live game state processing
- **Batch processing multiple runs**: Automated workflow support
- **Cloud storage integration**: S3/Blob storage backends
- **Advanced filtering**: ML-based quality assessment
- **Visualization tools**: Dataset exploration interfaces

### API Extensions

- **Library integration**: Import as Python modules
- **Plugin system**: Custom processing pipelines
- **Configuration files**: YAML/JSON parameter files
- **Webhook notifications**: Processing completion alerts

## Troubleshooting

### Common Issues

**No frame files found**
- Check directory contains PNG/JPG images
- Verify file naming patterns match expected formats
- Enable `--verbose` for detailed scanning output

**Permission errors on Windows**
- Ensure write permissions for output directory
- Close any open files in target directories
- Run with administrator privileges if needed

**Memory issues with large datasets**
- Increase `--stride` to reduce memory usage
- Use `--limit` to process smaller batches
- Enable streaming mode for very large datasets

**Low sprite counts**
- Lower `--confidence-threshold` (try 0.5)
- Check source image quality and resolution
- Verify sprite detection model is properly loaded

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python -m vision.tools.dump_sprites /path/to/run --output ./output --verbose --debug
```

## Contributing

### Adding New Features

1. Follow existing code patterns and naming conventions
2. Add comprehensive unit tests
3. Update documentation for new parameters
4. Ensure backward compatibility
5. Test with both synthetic and real data

### Code Quality

- **Type hints**: All function parameters and returns
- **Docstrings**: Complete function documentation  
- **Error handling**: Graceful failure with informative messages
- **Logging**: Appropriate level for different operations
- **Testing**: >90% coverage for new functionality

## License

These tools follow the same license as the Pokemon MD agent project. See the main project LICENSE file for details.