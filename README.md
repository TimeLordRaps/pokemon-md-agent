# Pokemon Mystery Dungeon Red - Autonomous Agent

Multi-model Qwen3-VL agent with hierarchical RAG system, dynamic temporal resolution, and live dashboard for autonomous Pokemon Mystery Dungeon Red gameplay.

## 🎮 Project Overview

**Goal**: Build an autonomous agent that can play Pokemon Mystery Dungeon Red using:
- Multi-scale visual reasoning (Qwen3-VL 2B/4B/8B)
- Hierarchical RAG with 7 temporal resolution silos
- Dynamic FPS adjustment (30fps → 1fps) and frame multipliers
- Live searchable dashboard (GitHub Pages + You.com Content API)
- Cost-aware model routing and vision optimization

**Tech Stack**:
- **Emulator**: mgba + mgba-http (960x640 @ 30fps)
- **Vision Models**: Qwen3-VL-2B/4B/8B (Thinking + Instruct variants)
- **Vector DB**: ChromaDB or FAISS (multi-scale temporal embeddings)
- **Dashboard**: GitHub Pages (static) + You.com Content API (retrieval)
- **Control**: Python + mgba-http API

---

## 🎬 Demo Video

**[Watch the 3-minute agent demo (MP4)](docs/assets/agent_demo.mp4)** — 180 seconds of autonomous gameplay with Kokoro TTS narration, automatically generated from agent trajectory and You.com knowledge retrieval.

**Submission snapshot:**
- Branch: `deadline-2025-10-30-2355-PT` (frozen @ 23:55 UTC-7)
- Tag: `deadline-2025-10-30-2359-PT` (final submission timestamp)

---

## ⚡ Quick Start (5 minutes)

### Prerequisites
- **mGBA emulator** (version 0.8.0+) with Lua socket server running on port `8888`
- **Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia)** ROM file
- **Python 3.11+** with conda/mamba environment
- **Save file** with game in Tiny Woods or similar dungeon (starter floor)

### Setup

1. **Place ROM & Save in `rom/` directory:**
   ```bash
   # From project root (pokemon-md-agent/)
   cp "Pokemon Mystery Dungeon - Red Rescue Team.gba" ./rom/
   cp "Pokemon Mystery Dungeon - Red Rescue Team.sav" ./rom/
   ```

2. **Create & activate conda environment (installs Kokoro TTS + MoviePy):**
   ```bash
   mamba create -n agent-hackathon python=3.11 -y
   mamba activate agent-hackathon
   pip install -r requirements.txt
   ```

3. **Configure You.com Content API (optional but recommended):**
   ```powershell
   # Persist YOUR key in PowerShell profile or current session
   $Env:YOU_API_KEY = "<your-you-api-key>"

   # Smoke test (live mode) - replace URL with a domain you expect the agent to use
   python -m scripts.check_you_api --url https://www.serebii.net/dungeon/redblue/d001.shtml --live
   ```

   ```bash
   # macOS/Linux example
   export YOU_API_KEY="<your-you-api-key>"
   python -m scripts.check_you_api --url https://www.serebii.net/dungeon/redblue/d001.shtml --live
   ```

   - Success prints `• https://... -> OK | ...`
   - If you skip this step (or the key is invalid) the agent falls back to placeholder content.

4. **Start mGBA with Lua socket server** (Windows PowerShell example):
   ```powershell
   # Ensure mGBA-http is loaded (Lua console > File > Load script > mGBASocketServer.lua)
   # Server defaults to port 8888
   # Verify with: python .temp_check_ram.py
   ```

5. **Run final demo (50-step agent + 3-min video + Kokoro voiceover):**
   ```bash
   mamba activate agent-hackathon
   cd pokemon-md-agent
   python scripts/final_demo_runner.py
   ```

   **Output:**
   - `runs/demo_*/trajectory_*.jsonl` - Full trajectory data
   - `agent_demo.mp4` - 3-minute montage video (key frames + Kokoro TTS narration)
   - Console logs show real-time progress

### Expected Timeline
- **Initialization**: ~5s
- **Agent execution** (50 steps): ~30-60s
- **Video generation**: ~10-20s
- **Total**: ~1-2 minutes

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `Failed to connect to mGBA` | Verify mGBA is running, socket server active, port 8888 |
| `No ROM files found` | Check ROM + SAV files are in `./rom/` directory |
| `Perception failed: unpack requires...` | Stale mGBA connection; restart emulator |
| `Video generation failed` | Ensure `opencv-python` installed: `pip install opencv-python` |

---

## 📊 Dashboard & Monitoring

The agent includes a comprehensive dashboard system for monitoring gameplay and retrieving external knowledge:

### Dashboard Features
- **Live Updates**: Real-time trajectory logging and meta-view generation
- **Searchable Content**: Client-side FAISS indexes for fast similarity search
- **Rate Limiting**: Token bucket rate limiting (30 files/min, 300/hour) with exponential backoff
- **Build Budget**: Coalesces commits to ≤10/hour to avoid GitHub Actions limits
- **LFS Avoidance**: Keeps artifacts under 8MB; no Git LFS unless required
- **Resolution Modes**: 2× (480×320) default for dashboard, 1× (240×160) for Qwen-VL benchmarking

### Upload Modes
1. **Git Push**: Direct push to `pages` branch (recommended for development)
2. **GitHub API**: REST API calls (fallback when git unavailable)
3. **No-op**: Dashboard disabled, retains local cache only

### Content API Integration
- **You.com Wrapper**: Multi-URL batch fetching with budget management
- **Monthly Budget**: 1,000 calls/month default, persisted to `~/.cache/pmd-red/youcom_budget.json`
- **Gate Policy**: Requires ≥3 on-device shallow hits before issuing gate burst (max 2 content calls)
- **Cool-down**: Per-gate invocation permits 2 calls max (bulk defaults + focused deep-dive)
- **Environment**: Set `YOU_API_KEY` (and optionally `YOU_API_BASE`) before running the agent
- **Smoke Test**: `python -m scripts.check_you_api --url https://example.com --live` validates credentials before demos

### Configuration
```python
config = AgentConfig(
    # Skill triggers
    enable_skill_triggers=True,       # Enable automatic skill triggers
    skill_belly_threshold=0.3,        # Trigger when belly < 30%
    skill_hp_threshold=0.25,          # Trigger when HP < 25%
    skill_backoff_seconds=5.0,        # Backoff after failures

    # Dashboard
    dashboard_enabled=True,           # Toggle dashboard uploads
    dashboard_branch="pages",         # Git branch for Pages
    dashboard_site_root="docs",       # Site root directory
    dashboard_flush_seconds=30.0,     # Batch flush interval
    dashboard_max_batch_bytes=8*1024*1024,  # 8MB batch limit
    dashboard_max_files_per_minute=30 # Rate limit
)
```

### Costs & Limits
- **Pages Bandwidth**: 100GB/month free, then $0.008/GB
- **LFS Storage**: 1GB free, then $0.008/GB/month
- **LFS Bandwidth**: $0.008/GB
- **Recommendation**: Disable dashboard or use external storage (Cloudflare R2) when approaching limits

---

## �📁 Project Structure

```
pokemon-md-agent/
├── README.md                          # This file
├── AGENTS.md                          # Instructions for code agents (Copilot/Claude Code)
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── docs/                             # Architecture & design documents
│   ├── pokemon-md-rag-system.md     # RAG system architecture
│   ├── pokemon-md-dashboard.md      # Dashboard design
│   ├── pokemon-md-agent-scaffold.md # Agent scaffold & environment
│   └── embedding-types.md           # Detailed embedding strategy
│
├── src/                              # Source code
│   ├── agent/                       # Agent core
│   │   ├── __init__.py
│   │   ├── qwen_controller.py       # Multi-model Qwen3-VL orchestration
│   │   ├── model_router.py          # 2B/4B/8B routing logic
│   │   └── memory_manager.py        # Scratchpad & persistent memory
│   │
│   ├── orchestrator/                # Message orchestration
│   │   ├── __init__.py
│   │   └── message_packager.py      # Three-message protocol with model presets
│   │
│   ├── embeddings/                  # Embedding generation & storage
│   │   ├── __init__.py
│   │   ├── extractor.py             # Extract embeddings from Qwen3-VL
│   │   ├── temporal_silo.py         # 7 temporal resolution managers
│   │   └── vector_store.py          # ChromaDB wrapper
│   │
│   ├── vision/                      # Screenshot processing
│   │   ├── __init__.py
│   │   ├── sprite_detector.py       # Qwen3-VL sprite detection
│   │   ├── grid_parser.py           # Convert to tile grid for pathfinding
│   │   └── ascii_renderer.py        # ASCII state for blind LLMs
│   │
│   ├── environment/                 # mgba integration
│   │   ├── __init__.py
│   │   ├── mgba_controller.py       # mgba-http API wrapper
│   │   ├── fps_adjuster.py          # Dynamic FPS & frame multiplier
│   │   └── action_executor.py       # Button press execution
│   │
│   ├── retrieval/                   # RAG system
│   │   ├── __init__.py
│   │   ├── auto_retrieve.py         # Automatic trajectory retrieval
│   │   ├── circular_buffer.py       # On-device circular buffer (60-min window)
│   │   ├── cross_silo_search.py     # Multi-scale search
│   │   ├── deduplicator.py          # pHash/sprite-hash deduplication
│   │   ├── embedding_generator.py   # Text/image embedding generation
│   │   ├── keyframe_policy.py       # Keyframe selection (SSIM/floor/combat triggers)
│   │   ├── local_ann_index.py       # SQLite ANN index for KNN search
│   │   ├── meta_view_writer.py      # 2×2 meta-view generation
│   │   ├── on_device_buffer.py      # Orchestrates all buffer components
│   │   └── stuckness_detector.py    # Loop detection
│   │
│   └── dashboard/                   # Live dashboard
│       ├── __init__.py
│       ├── uploader.py              # Batch upload to GitHub Pages
│       ├── content_api.py           # You.com Content API wrapper
│       └── similarity_precompute.py # Pre-compute comparison pages
│
├── tests/                           # Unit tests
│   ├── test_mgba_connection.py
│   └── test_on_device_buffer.py
│
├── demos/                           # Visual demonstrations
│   └── embedding_visualization.py
│
├── examples/                        # Example usage
│   └── quickstart.py
│
├── research/                        # Related papers & inspirations
│   └── qwen3-vl-summary.md
│
└── config/                          # Configuration files
    ├── agent_config.yaml            # Agent behavior settings
    ├── embedding_config.yaml        # Embedding strategy config
    └── mgba_config.ini              # mgba settings
```

---

## 🚀 Quick Start (Post-Fix)

1. Start mGBA with ROM + Lua script:
   ```
   C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba
   C:\Homework\agent_hackathon\pokemon-md-agent\config\save_files\game_start_save.ss0
   C:\Homework\agent_hackathon\pokemon-md-agent\src\mgba-harness\mgba-http\mGBASocketServer.lua
   ```

2. Run demo:
   ```bash
   cd pokemon-md-agent
   python demo_agent.py --max-steps 50
   ```

3. View results:
   ```bash
   ls -lt runs/  # Latest run folder
   ```

## Troubleshooting

- **Screenshot locked:** Fixed in v1.1 (auto-retry with exponential backoff)
- **Socket error:** Fixed in v1.1 (proper cleanup on disconnect)
- **WRAM defaults:** Check `config/addresses/pmd_red_us_v1.json` offsets

### Prerequisites (Original Setup)

- **Python 3.11+** (with CUDA support for GPU acceleration)
- **mgba** with mgba-http enabled (Lua-only setup)
- **Pokemon Mystery Dungeon Red ROM** (you provide)
- **GPU**: NVIDIA GPU with CUDA support recommended (RTX 30-series or newer)

### Installation (Original)

```bash
# Clone or extract this repo
cd pokemon-md-agent

# Install PyTorch with CUDA support first (required for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install as editable package
pip install -e .
```

> ℹ️ **New dependency**: the benchmark now leverages `nano-graphrag` for
> retrieval-augmented prompt scaffolding.  It is included in
> `requirements.txt` and will be installed automatically with the editable
> package command above.

**Note**: The installation automatically detects your CUDA version and GPU architecture to install the correct PyTorch and Unsloth versions. If you encounter CUDA detection issues, you can manually run Unsloth's auto-install script first:

```bash
# Optional: Run Unsloth's auto-detection script
python -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py').read())"
```

**Verified**: This installation method has been tested and confirmed to work with:
- PyTorch 2.9.0+cu128 (CUDA 12.8, compatible with CUDA 12.9)
- Unsloth v2025.10.10 with Qwen3-VL support
- RTX 4090 GPU with Ampere architecture

### Configure mgba (Lua-Only Setup)

**Important**: This project uses mgba-http with Lua socket server. No Python socket server needed.

1. Download mgba v0.10.5+ from [mgba.io](https://mgba.io/)
2. Place your Pokemon Mystery Dungeon Red ROM in the `rom/` directory
3. Start mgba and load the game:
   - Load the ROM: `File → Load ROM` → select `rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba`
   - Load the save file: `File → Load State File` → select `config/game_start.sav`
   - Load the Lua script: `Tools → Scripting` → `Load script` → select `src/mgba-harness/mgba-http/mGBASocketServer.lua`
4. The Lua script will start the HTTP server automatically on port 8888

**Save Slot Advice**:
- Slot 0: Title screen (for reset)
- Slot 1: Floor ready (for benchmark loops) - agent loads this automatically
- Slot 2: Last autosave
- Slots 3-98: Manual saves
- Slot 99: Final save on agent shutdown

The agent will automatically load slot 1 on startup for consistent benchmarking.

### Run Agent (Original)

```bash
python examples/quickstart.py
```

---

## 📊 Benchmarking

### Comprehensive 3D Performance Analysis

The project includes a comprehensive benchmark harness for measuring Qwen3-VL model performance across context lengths, batch sizes, and task types:

```bash
# Run comprehensive benchmark with 3D analysis
python profiling/bench_qwen_vl.py --models all --tasks all --num-runs 3

# Dry run for testing (no actual model inference)
python profiling/bench_qwen_vl.py --dry-run --models "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit" --tasks "text_only"

# Custom configuration
python profiling/bench_qwen_vl.py --models "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit,unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit" --max-new-tokens 256
```

### Benchmark Features

**Context Length Scaling**: Tests from 1024 to 256k tokens (262k max for Qwen3-VL) on log2 scale
- 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 tokens

**Batch Size Optimization**: Tests batch sizes 1, 2, 4, 8 with automatic model-aware limits
- 2B models: up to batch size 8
- 4B models: up to batch size 4  
- 8B models: up to batch size 2

**Task Performance Analysis**: Four micro-benchmark tasks
- `text_only`: Text summarization
- `vision_simple`: Basic image description
- `vision_complex`: Tactical situation analysis
- `mixed_reasoning`: Strategic decision making

**3D Visualizations**: Interactive performance landscapes
- Throughput surfaces (context × batch size × tokens/sec)
- Performance contour maps
- Optimal batch size curves
- Log-scale context length plots

### Expected Outputs

- **CSV Data**: `profiling/data/comprehensive_benchmark_results.csv` with all measurements
- **3D Surface Plots**: `profiling/plots/3d_throughput_surfaces.png`
- **Performance Landscapes**: `profiling/plots/performance_landscapes.png`
- **Optimization Curves**: `profiling/plots/batch_optimization.png`
- **Context Scaling**: `profiling/plots/log_context_throughput.png`

### Interpreting Results

**Throughput Analysis**:
- Higher values indicate faster inference
- Look for inflection points where performance degrades
- Compare batching vs non-batching efficiency

**Performance Scores**:
- 0.0-1.0 scale based on response quality heuristics
- Task-specific scoring (conciseness, descriptiveness, strategy)

**Optimal Configurations**:
- Batch size curves show sweet spots for each context length
- 3D surfaces reveal performance saddle points
- Contour maps highlight efficient operating regions

### Model-Specific Limits

| Model | Max Context | Max Batch | Typical Throughput |
|-------|-------------|-----------|-------------------|
| Qwen3-VL-2B | 32,768 | 8 | 60-80 tokens/sec |
| Qwen3-VL-4B | 65,536 | 4 | 40-60 tokens/sec |
| Qwen3-VL-8B | 131,072 | 2 | 20-40 tokens/sec |

The benchmark automatically respects these limits and provides consistent comparison across all supported Qwen3-VL variants.

---

### Text-Speed Guarantee

The agent implements a text-speed guarantee feature to ensure OCR capture of dialogue frames:

- **Menu Profile**: `src/mgba-harness/profiles/set_text_speed_slow.json` navigates Options → Text Speed → Slow on boot
- **RAM Fallback**: Direct memory poke to text-speed setting when `allow_memory_write` enabled and ROM hash safe
- **Input Pacing**: A button taps throttled to ≥1 second intervals during textboxes for reliable OCR capture

### Multi-Scale Temporal Embeddings

7 temporal resolution silos with dynamic FPS adjustment:

| Silo | Base Sample Rate | Agent-Adjustable FPS | Context Span |
|------|-----------------|---------------------|--------------|
| temporal_1frame | Every frame | 30→10→5→3→1 fps | 0-4 sec |
| temporal_2frame | Every 2nd | - | 0-8 sec |
| temporal_4frame | Every 4th | - | 0-16 sec |
| temporal_8frame | Every 8th | - | 0-32 sec |
| temporal_16frame | Every 16th | - | 0-64 sec |
| temporal_32frame | Every 32nd | - | 0-128 sec |
| temporal_64frame | Every 64th | - | 2+ min |

**Agent can dynamically**:
- Adjust base FPS (30→1fps) to "zoom out" temporally
- Change frame multipliers (4x→8x→16x) for finer resolution
- Allocate memory budget across silos (e.g., 3/4 for last 5 min)

### Embedding Types (Corrected)

**Input embeddings**:
- `input`: Hidden states of what was sent to the model

**Thinking models** (reasoning-aware):
- `think_input`: Hidden state at/before `</think>` + input
- `think_full`: Hidden state before `</s>` (full input+output)
- `think_only`: Embedding of only `<think>...</think>` block
- `think_image_input`: Like `think_input` but image-only input
- `think_image_full`: Like `think_full` but image-only input
- `think_image_only`: Image-only reasoning (experimental)

**Instruct models** (fast, no reasoning overhead):
- `instruct_eos`: Hidden state at `</s>` token
- `instruct_image_only`: Image tokens only

### Model Routing

```
Qwen3-VL-2B-Instruct → Fast compression, simple navigation
         ↓
Qwen3-VL-4B-Thinking → Routing, retrieval, stuck detection
         ↓
Qwen3-VL-8B-Thinking-FP8 → Strategic decisions, dashboard queries
```

**Escalation triggers**:
- Confidence < 0.8 → 2B→4B
- Confidence < 0.6 OR stuck > 5 → 4B→8B
- 8B can call You.com Content API (cooldown: 5 min, budget: 100 calls)

### Inference Batching & KV Caching

The agent implements micro-batching for improved throughput:
- **Batch sizes**: 8 for 2B, 4 for 4B, 2 for 8B models
- **Timeout**: 50ms default for batch accumulation
- **KV cache**: On-disk memmap for long prefixes (HF_HOME/pmd_kv_cache)
- **Async processing**: asyncio.gather for parallel inference

---

## 🎯 Key Features

### 1. Dynamic Temporal Resolution

Agent can adjust how it perceives time:

```python
# Zoom out (see longer time span with less detail)
agent.adjust_fps(target_fps=5)  # 30fps → 5fps
agent.adjust_frame_multiplier(multiplier=16)  # 4x → 16x

# Zoom in (see recent moments with more detail)
agent.adjust_fps(target_fps=30)  # Back to 30fps
agent.adjust_frame_multiplier(multiplier=2)  # 16x → 2x
```

### 2. Memory Split Control

Agent can allocate context budget across temporal ranges:

```python
# Example: 3/4 for last 5 min, 1/4 for storyline/missions
agent.allocate_memory({
    "last_5_minutes": 0.75,
    "storyline": 0.15,
    "active_missions": 0.10
})
```

### 3. Persistent Scratchpad

Agent has a "sticky note" that persists across environment interactions:

```python
agent.scratchpad.write("Floor 7: stairs are usually in NE corner")
# This will be visible to agent in next inference
```

### 4. Stuckness Detection

Cross-temporal divergence metric:
- High short-term similarity (repeating micro-actions)
- Low long-term similarity (no macro progress)
→ Triggers escalation to 8B + dashboard fetch

### 5. Live Searchable Dashboard

- GitHub Pages hosted (updated every 5 minutes)
- Pre-computed similarity comparisons
- Accessible via You.com Content API (agent-only secret URLs)
- Judge message wall for hackathon feedback

---

## 📋 Usage Examples

### Grid Parser

The grid parser produces a uniform tile grid and screen mapping from game screen data, enabling pathfinding and spatial reasoning for the agent.

```python
from src.vision.grid_parser import GridParser
from src.environment.ram_decoders import RAMSnapshot

# Initialize parser
parser = GridParser()

# Parse RAM data into grid
grid_frame = parser.parse_ram_snapshot(ram_snapshot)

# Access grid properties
print(f"Grid size: {grid_frame.width}x{grid_frame.height}")
print(f"Tile size: {grid_frame.tile_size_px}px")

# Get tile at position
tile = grid_frame.tiles[y][x]
print(f"Tile type: {tile.tile_type}")

# Compute pathfinding distances
bfs_result = parser.compute_bfs_distances(grid_frame, start=(x, y))
distance_to_target = bfs_result.distances[target_y][target_x]
```

---

## 🛠️ Development Workflow

### For Code Agents (Copilot/Claude Code/Roo-Coder)

See [AGENTS.md](AGENTS.md) for detailed instructions on:
- How to structure code changes
- Testing procedures
- Integration patterns
- Prompt templates

### Manual Development

1. **Make changes** in `src/` directory
2. **Test fast lane** with `.\scripts\test_fast.ps1` (Windows) or `bash scripts/test_fast.sh` (Linux/Mac)
3. **Test full suite** with `.\scripts\test_full.ps1` (Windows) or `bash scripts/test_full.sh` (Linux/Mac)
4. **Run demos** in `demos/` to visualize changes
5. **Commit** with descriptive messages

#### Test Markers & Scripts

**Fast Lane** (`scripts/test_fast.ps1`):
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; $env:FAST="1"; $env:PYTEST_FDUMP_S="45"; $env:PYTHONPATH="$(pwd)\src"; python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"`
- **Expected Runtime**: <3 minutes
- **Purpose**: Quick validation excluding slow/network/bench/longctx tests

**Full Lane** (`scripts/test_full.ps1`):
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; Remove-Item Env:FAST -ErrorAction SilentlyContinue; $env:PYTEST_FDUMP_S="90"; $env:PYTHONPATH="$(pwd)\src"; python -m pytest -q`
- **Expected Runtime**: 10-15 minutes
- **Purpose**: Complete test suite with all markers

**CI Lane** (`scripts/test_ci.ps1`):
- **Command**: Calls `scripts/test_fast.ps1`
- **Expected Runtime**: <3 minutes
- **Purpose**: Minimal CI validation

**Bench Sweep** (`scripts/bench_sweep.ps1`):
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; $env:PYTHONPATH="$(pwd)\src"; python profiling/bench_qwen_vl.py --models all --csv bench_results.csv --time-budget-s 180 --full --plot bench_results.csv`
- **Expected Runtime**: 5-10 minutes per configuration
- **Purpose**: Performance benchmarking with parameter sweeps, saves CSV + JSONL + PNG plots to `profiling/results/<UTC_ISO>/`

**Sync Profiling** (`scripts/sync_profiling.ps1`):
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; Copy-Item "..\profiling\*" ".\profiling\" -Recurse -Force -Exclude "__pycache__"`
- **Expected Runtime**: <1 minute
- **Purpose**: Consolidate profiling data from root directory

**Markers**:
- `@pytest.mark.slow`: Long-running tests (model training, heavy parametrization)
- `@pytest.mark.network`: Tests requiring emulator/web connections
- `@pytest.mark.bench`: Performance benchmarking and plotting
- `@pytest.mark.longctx`: Tests with ≥64k context

**Environment Variables**:
- `FAST=1`: Reduces test parameters for faster execution
- `PYTEST_FDUMP_S=45`: Session timeout for deadlock detection (default 60s)

**Flags**:
- `--maxfail=1`: Stop after first failure
- `--timeout=30 --timeout-method=thread`: 30s timeout per test with thread method
- `-m "not slow and not network and not bench and not longctx"`: Exclude marked tests
- `filterwarnings = ["ignore::DeprecationWarning"]`: Suppress deprecation warnings

#### Troubleshooting

**Test Failures**:
- **Timeout errors**: Increase `PYTEST_FDUMP_S` environment variable or check for infinite loops
- **Import errors**: Ensure `PYTHONPATH` includes `src/` directory
- **mGBA connection failures**: Verify emulator is running with Lua script on port 8888
- **CUDA out of memory**: Reduce batch sizes or use smaller models for testing

**Benchmark Issues**:
- **Long runtimes**: Use `--time-budget-s` to limit entire benchmark duration (default 180s)
- **Time budget exceeded**: Benchmark suite ran longer than `--time-budget-s` limit - check summary.json
- **OOM during bench**: Reduce `--batches` or `--contexts` parameters, or use smaller models
- **No plots generated**: Ensure matplotlib is installed and CSV file exists
- **Output directory errors**: Check write permissions for `profiling/results/<UTC_TIMESTAMP>/`
- **Fast lane limitations**: Use `--full` flag to run comprehensive benchmarks

**Common Runtime Issues**:
- **SyntaxError in qwen_controller.py**: See `agent_mailbox/copilot2codex.md` for core team fix
- **faulthandler timeout**: Tests hanging - check for blocking I/O operations
- **Top slow tests**: Review session output for slowest tests to optimize

**Expected Runtimes**:
- Fast lane: 2-3 minutes
- Full lane: 10-15 minutes  
- Bench sweep: 5-10 minutes per config
- CI lane: <3 minutes

#### Profiling Consolidation

Run `.\scripts\sync_profiling.ps1` to consolidate profiling data from legacy root `profiling/` directory into `pokemon-md-agent/profiling/`.

#### Current Test Status

⚠️ **Tests currently blocked by runtime bug**: SyntaxError in `src/agent/qwen_controller.py` (await outside async function). See `agent_mailbox/copilot2codex.md` for details. Core team fix required before test suite can run.

#### Benchmarking & Profiling

Run performance benchmarks with `.\scripts\bench_sweep.ps1` (Windows) or equivalent bash script.

**Bench Flags**:
- `--time-budget-s`: Time budget for entire benchmark suite (seconds, default: 180)
- `--full`: Run full benchmark suite (longer, more comprehensive)
- `--contexts`: Exact context lengths to test (comma-separated, overrides --min-ctx/--ctx-mult)
- `--image-text-ratios`: Image-to-text content ratios to test (comma-separated floats, default: '0.5')
- `--models`: Models to benchmark ('all' or comma-separated list)
- `--min-ctx`: Minimum context length (default: 1024)
- `--ctx-mult`: Context length multiplier (default: 1.5)
- `--max-wall`: Maximum wall clock time per benchmark (seconds, default: 60)
- `--batches`: Batch sizes to test (comma-separated, default: '1,2,4,8')
- `--best-of`: Best-of values to test (comma-separated, default: '1,2,4,8')
- `--csv`: Output CSV path (required for benchmarking)
- `--plot`: CSV file to plot from (generates plots in profiling/plots/)
- `--dry-run`: Use synthetic timings instead of real inference

**Example Commands**:
```bash
# Fast lane benchmark (default)
python profiling/bench_qwen_vl.py --csv results.csv --dry-run

# Full benchmark with time budget
python profiling/bench_qwen_vl.py --full --time-budget-s 300 --csv results.csv

# Custom contexts and image-text ratios
python profiling/bench_qwen_vl.py --contexts 1024,2048,4096,8192 --image-text-ratios 0.3,0.5,0.7 --csv results.csv

# Plot existing results
python profiling/bench_qwen_vl.py --plot results.csv
```

Results saved to `profiling/results/<UTC_TIMESTAMP>/` with CSV metrics, JSON summary, and interactive plots.

---

## 📈 Performance Targets

- **Inference speed**: <2 sec per decision (2B), <5 sec (4B), <10 sec (8B)
- **Token efficiency**: <200k tokens/inference (2B/4B), <64k (8B)
- **Memory footprint**: <50GB local cache (<1 hour history)
- **API budget**: <100 Content API calls total (for stuck situations only)

---

## 🔗 Related Resources

- [Qwen3-VL Models](https://huggingface.co/Qwen)
- [mgba-http Documentation](https://mgba.io/)
- [You.com Content API](https://documentation.you.com/)
- [Pokemon Mystery Dungeon Red Wiki](https://bulbapedia.bulbagarden.net/)

---

## 📝 Next Actions

1. ✅ Extract this zip to `C:\Homework\agent_hackathon`
2. Install dependencies: `pip install -r requirements.txt` (includes `imagehash` for retrieval deduplication tests)
3. Configure mgba (see `config/mgba_config.ini`)
4. Test mgba connection: `python tests/test_mgba_connection.py`
5. **Run fast test suite**: `.\scripts\test_fast.ps1` (Windows) or `bash scripts/test_fast.sh` (Linux/Mac)
6. Run quickstart: `python examples/quickstart.py`
7. Read architecture docs in `docs/` folder
8. Review `AGENTS.md` for code agent instructions

**Current Status**: ⚙️ Seed project structure - ready for implementation

---

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a hackathon project. Contributions welcome via pull requests!
