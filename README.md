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

## � Dashboard & Monitoring

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

### Configuration
```python
config = AgentConfig(
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

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+** (with CUDA for GPU acceleration)
- **mgba** with mgba-http enabled (Lua-only setup)
- **Pokemon Mystery Dungeon Red ROM** (you provide)
- **GPU**: 24GB+ VRAM recommended (for Qwen3-VL-8B-FP8)

### 2. Installation

```bash
# Clone or extract this repo
cd C:\Homework\agent_hackathon\pokemon-md-agent

# Install as editable package (recommended for development)
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt

# Download Qwen3-VL models (will auto-download on first run)
# Models used: 2B/4B/8B in both Thinking and Instruct variants
```

### 3. Configure mgba (Lua-Only Setup)

**Important**: This project uses mgba-http with Lua socket server. No Python socket server needed.

1. Download mgba v0.8.0+ from [mgba.io](https://mgba.io/)
2. Place your Pokemon Mystery Dungeon Red ROM in the `rom/` directory
3. Start mgba with HTTP server enabled:

```bash
# Windows
mgba.exe --http-server --port 8888 rom/Pokemon\ Mystery\ Dungeon\ -\ Red\ Rescue\ Team\ \(USA\,\ Australia\).gba

# Linux/Mac
mgba --http-server --port 8888 rom/Pokemon\ Mystery\ Dungeon\ -\ Red\ Rescue\ Team\ \(USA\,\ Australia\).gba
```

**Save Slot Advice**:
- Slot 0: Title screen (for reset)
- Slot 1: Floor ready (for benchmark loops) - agent loads this automatically
- Slot 2: Last autosave
- Slots 3-98: Manual saves
- Slot 99: Final save on agent shutdown

The agent will automatically load slot 1 on startup for consistent benchmarking.

### 4. Run Agent

```bash
python examples/quickstart.py
```

---

## 📊 Architecture Highlights

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

## 🛠️ Development Workflow

### For Code Agents (Copilot/Claude Code/Roo-Coder)

See [AGENTS.md](AGENTS.md) for detailed instructions on:
- How to structure code changes
- Testing procedures
- Integration patterns
- Prompt templates

### Manual Development

1. **Make changes** in `src/` directory
2. **Test** with `pytest tests/`
3. **Run demos** in `demos/` to visualize changes
4. **Commit** with descriptive messages

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
2. Install dependencies: `pip install -r requirements.txt`
3. Configure mgba (see `config/mgba_config.ini`)
4. Test mgba connection: `python tests/test_mgba_connection.py`
5. Run quickstart: `python examples/quickstart.py`
6. Read architecture docs in `docs/` folder
7. Review `AGENTS.md` for code agent instructions

**Current Status**: ⚙️ Seed project structure - ready for implementation

---

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a hackathon project. Contributions welcome via pull requests!
