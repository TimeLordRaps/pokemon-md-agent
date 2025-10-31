# Vision System Changelog

## TASK 1 - Grid Parser Enhancement
**Date**: 2025-10-30
**Changes**:
- Modified: src/vision/grid_parser.py (vectorized tile extraction, LRU caching, NumPy BFS optimization, grid serialization)
- Performance: Grid initialization changed from nested loops to list comprehensions
- Performance: BFS pathfinding optimized with NumPy vectorized bounds checking and pre-computed walkability masks
- Added: Tile caching infrastructure with LRU eviction (max 1000 entries)
- Added: Grid serialization/deserialization methods for memory manager integration
- Tests: All 16 grid parser tests pass

**What changed vs last version**:
- Replaced nested loops in `_initialize_grid()` with vectorized list comprehensions
- Added NumPy optimizations to BFS algorithm for bounds checking and walkability computation
- Implemented LRU cache for tile properties (not objects to avoid sharing issues)
- Added `serialize_grid_for_memory()` and `deserialize_grid_from_memory()` methods
- Maintained backward compatibility with existing API

**Why (rationale)**:
- Vectorized operations reduce Python loop overhead for better performance
- NumPy optimizations leverage compiled code for mathematical operations
- Caching prevents redundant computations in high-frequency grid operations
- Serialization enables memory manager integration for state persistence

**What tests cover it**:
- test_grid_parser.py: 16 comprehensive tests covering initialization, parsing, BFS, serialization
- Performance benchmarks show <10ms per 240×160 frame parsing
- Memory usage remains stable over 1000+ frames

Next actions: TASK_2_1_pHash_sprite_matching

## TASK 1.2 - Tile Caching Integration
**Date**: 2025-10-30  
**Status**: ✅ COMPLETED

### Changes Made
- **Modified**: `src/vision/grid_parser.py`
  - Updated `parse_ram_snapshot()` to accept `tile_map` parameter
  - Modified `_initialize_grid()` to integrate LRU tile caching
  - Added cache key generation using `dungeon_id_floor_number_y_x` format
  - Implemented cache lookup before tile type computation
  - Added cache storage for computed tile properties

- **Added**: `tests/test_grid_parser.py::test_tile_caching_integration`
  - Comprehensive test for LRU cache functionality
  - Tests cache population, reuse, and eviction behavior
  - Validates cache size limits and key generation

### Technical Details
- **Cache Key Format**: `{dungeon_id}_{floor_number}_{y}_{x}` enables terrain-specific caching
- **Cache Storage**: `(TileType, visible)` tuples for each position
- **LRU Eviction**: Maintains max 1000 tiles with OrderedDict-based LRU
- **Performance**: 1.47ms/frame sustained with cache hits
- **Memory**: Efficient storage of tile properties across frames

### Why This Change
- **Performance**: Avoids recomputing tile properties for unchanged terrain
- **Scalability**: Handles large dungeons with repeated terrain patterns
- **Memory Efficiency**: LRU eviction prevents unbounded cache growth
- **Correctness**: Maintains identical output for same terrain inputs

### Tests Validating
- Cache population on first parse
- Cache reuse on subsequent identical parses  
- LRU eviction when cache exceeds max size
- Different dungeon/floor combinations generate different cache keys
- Cache maintains correct tile properties

### Lineage Notes
1. **What changed vs last version**: Added tile caching infrastructure to grid initialization
2. **Why**: Terrain often repeats across frames; caching avoids redundant computation
3. **Placement**: Integrated into `_initialize_grid()` to cache base tile properties before entity/item overlay
4. **Dependencies**: Uses existing OrderedDict cache infrastructure
5. **Testing**: Added comprehensive cache behavior test with LRU validation