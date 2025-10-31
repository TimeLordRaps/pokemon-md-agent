# AGENTS.md - Instructions for Code Agents

> **Target Audience**: GitHub Copilot, Claude Code, Roo-Coder, Codex, and similar AI coding assistants

This document provides context, constraints, and patterns for AI agents working on the Pokemon MD Agent project.

---

## 🎯 Project Mission

Build an autonomous agent that plays Pokemon Mystery Dungeon Red using:
- **Multi-model Qwen3-VL** (2B/4B/8B in Thinking+Instruct variants)
- **Hierarchical RAG** with 7 temporal resolution silos
- **Dynamic temporal adjustment** (FPS and frame multipliers)
- **Live dashboard** (GitHub Pages + You.com Content API)
- **Cost-aware routing** (use smallest capable model)

---

## 📋 Core Constraints & Invariants

### File Organization
- ✅ Root folder structure: `src/`, `tests/`, `docs/`, `demos/`, `examples/`, `research/`, `config/`
- ✅ No double-nesting (avoid `repo/repo`)
- ✅ No absolute paths in source code (only in config/prompts)
- ✅ Windows + WSL2 friendly (normalize path separators)

### Code Quality
- ✅ **Full files only** (no placeholder comments like `# TODO: implement`)
- ✅ **Type hints** for all function signatures
- ✅ **Docstrings** for all classes and public methods
- ✅ **Error handling** with specific exceptions (not bare `except:`)
- ✅ **Logging** instead of print statements

### Architecture
- ✅ **Delta-only changes** (minimal edits, explain placement)
- ✅ **Local fixes first** (escalate to system-wide only if co-occurring issues)
- ✅ **Reversible transforms** (checkpoint before risky changes)
- ✅ **Cite fresh facts** (≥3 reputable sources with dates if using new APIs)

---

## 🧩 Key Architectural Patterns

### 1. Embedding Extraction Pattern

**Corrected Embedding Types** (see `docs/embedding-types.md` for details):

```python
from src.embeddings.extractor import QwenEmbeddingExtractor

extractor = QwenEmbeddingExtractor(model_name="Qwen3-VL-4B-Thinking")

# Available extraction modes:
embeddings = extractor.extract(
    input_data=screenshot,
    mode="think_full"  # Options: input, think_input, think_full, think_only,
                       #          think_image_input, think_image_full,
                       #          think_image_only, instruct_eos, instruct_image_only
)
```

**Embedding Types**:
- `input`: Hidden states of model input
- `think_input`: Hidden state at/before `</think>` + input
- `think_full`: Hidden state before `</s>` (full input+output)
- `think_only`: Only `<think>...</think>` block
- `think_image_input`: Like `think_input` but image-only
- `think_image_full`: Like `think_full` but image-only
- `think_image_only`: Image-only reasoning (experimental)
- `instruct_eos`: Hidden state at `</s>` (Instruct models)
- `instruct_image_only`: Image tokens only (Instruct models)

### 2. Temporal Silo Pattern

```python
from src.embeddings.temporal_silo import TemporalSiloManager

# Initialize 7 temporal silos
silo_manager = TemporalSiloManager(
    base_fps=30,
    silos=[1, 2, 4, 8, 16, 32, 64]  # Frame intervals
)

# Store embedding in appropriate silo
silo_manager.store(
    embedding=emb_vector,
    metadata={
        "timestamp": time.time(),
        "floor": 7,
        "hp": 85,
        "action": "move_right"
    },
    silo_id="temporal_4frame"
)

# Retrieve similar trajectories across silos
results = silo_manager.cross_silo_search(
    query_embedding=current_emb,
    silos=["temporal_1frame", "temporal_4frame", "temporal_16frame"],
    top_k=3
)
```

### 3. Model Routing Pattern

```python
from src.agent.model_router import ModelRouter

router = ModelRouter(
    confidence_2b_threshold=0.8,
    confidence_4b_threshold=0.6,
    stuck_threshold=5
)

# Router decides which model to use
model_choice = router.select_model(
    confidence=agent_state.confidence,
    stuck_counter=agent_state.stuck_counter,
    complexity=situation.complexity_score
)

if model_choice == "2B":
    response = qwen_2b_instruct.infer(screenshot)
elif model_choice == "4B":
    response = qwen_4b_thinking.infer(screenshot, retrieved_trajectories)
elif model_choice == "8B":
    response = qwen_8b_thinking.infer(screenshot, retrieved_trajectories, dashboard_context)
```

### 4. Dynamic FPS Adjustment Pattern

```python
from src.environment.fps_adjuster import FPSAdjuster

fps_adjuster = FPSAdjuster(base_fps=30, allowed_fps=[30, 10, 5, 3, 1])

# Agent can request FPS change
if agent.perceives_redundant_frames():
    fps_adjuster.set_fps(target_fps=5)  # Zoom out temporally

# Agent can adjust frame multiplier
if agent.needs_finer_resolution():
    fps_adjuster.set_multiplier(multiplier=16)  # 4x → 16x
```

### 5. Memory Allocation Pattern

```python
from src.agent.memory_manager import MemoryManager

memory_mgr = MemoryManager(total_context_budget=256_000)

# Agent can split memory across temporal ranges
memory_mgr.allocate({
    "last_5_minutes": 0.75,   # 75% of context for recent
    "storyline": 0.15,         # 15% for mission context
    "active_missions": 0.10    # 10% for current objectives
})

# Scratchpad (persistent sticky notes)
memory_mgr.scratchpad.write("Floor 7: stairs usually NE corner")
# This persists across environment interactions
```

### 6. Stuckness Detection Pattern

```python
from src.retrieval.stuckness_detector import StucknessDetector

detector = StucknessDetector(divergence_threshold=0.4)

# Check if agent is stuck in loop
stuckness = detector.analyze(
    short_term_similarity=0.95,  # Last 4 seconds very similar
    mid_term_similarity=0.88,    # Last 64 seconds similar
    long_term_similarity=0.45    # Last 2+ minutes very different
)

if stuckness["status"] == "stuck":
    # Escalate to 8B + dashboard fetch
    response = qwen_8b_thinking.infer(
        screenshot,
        dashboard_fetch=content_api.fetch_guide("stuck_loop_breaking")
    )
```

---

## 🔧 Implementation Guidelines

### Adding a New Feature

1. **Read architecture docs** (`docs/` folder)
2. **Identify affected modules** (check imports)
3. **Write tests first** (`tests/` folder)
4. **Implement incrementally** (small commits)
5. **Update README** if user-facing

### Modifying Embedding Strategy

1. **Read** `docs/embedding-types.md` for context
2. **Update** `src/embeddings/extractor.py`
3. **Update** `src/embeddings/temporal_silo.py` if silo logic changes
4. **Test** with `demos/embedding_visualization.py`
5. **Document** changes in `docs/embedding-types.md`

### Adding a New Temporal Silo

1. **Update** `src/embeddings/temporal_silo.py`
2. **Add** to config: `config/embedding_config.yaml`
3. **Test** retrieval logic: `tests/test_cross_silo_search.py`
4. **Update** dashboard upload script: `src/dashboard/uploader.py`

### Integrating New Vision Model

1. **Read** Qwen3-VL docs: `research/qwen3-vl-summary.md`
2. **Create** wrapper: `src/vision/qwen_wrapper.py`
3. **Test** sprite detection: `tests/test_sprite_detection.py`
4. **Benchmark** inference speed: `demos/model_benchmark.py`
5. **Update** router: `src/agent/model_router.py`

---

## 🧪 Testing Patterns

### Test Markers & Scripts

**Fast Lane**: `scripts/test_fast.ps1` (Windows) or `bash scripts/test_fast.sh` (Linux/Mac)
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; cd "C:\Homework\agent_hackathon\pokemon-md-agent"; $env:FAST="1"; $env:PYTEST_FDUMP_S="45"; $env:PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src"; python -m pytest -q --maxfail=1 -m "not slow and not network and not bench and not longctx"`
- **Expected Runtime**: <3 minutes
- **Purpose**: Quick validation excluding slow/network/bench/longctx tests

**Full Lane**: `scripts/test_full.ps1` (Windows) or `bash scripts/test_full.sh` (Linux/Mac)
- **Command**: `mamba info --envs; python --version; mamba activate agent-hackathon; pwd; ls; cd "C:\Homework\agent_hackathon\pokemon-md-agent"; Remove-Item Env:FAST -ErrorAction SilentlyContinue; $env:PYTEST_FDUMP_S="90"; $env:PYTHONPATH="C:\Homework\agent_hackathon\pokemon-md-agent\src"; python -m pytest -q`
- **Expected Runtime**: 10-15 minutes
- **Purpose**: Complete test suite with all markers

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

### Troubleshooting Guide

**Test Execution Issues**:
- **Command prefix failures**: Ensure mamba/conda environment is properly configured
- **Import errors**: Verify `PYTHONPATH` includes project `src/` directory
- **Timeout exceeded**: Check `PYTEST_FDUMP_S` value (default 60s) or investigate hanging tests
- **faulthandler dumps**: Review traceback files for deadlock locations

**Benchmark Problems**:
- **Long execution times**: Use `--time-budget-s` parameter to limit run duration (default: 180s)
- **Memory errors**: Reduce batch sizes (`--batches 1,2`) or context lengths (`--contexts 1024,2048`)
- **Output path issues**: Ensure `profiling/results/` directory is writable
- **Model loading failures**: Check CUDA availability and VRAM capacity
- **Time budget exceeded**: Benchmark ran longer than `--time-budget-s` limit

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

**Expected Runtimes**:
- Fast lane: 2-3 minutes
- Full lane: 10-15 minutes
- Bench sweep: 5-10 minutes per configuration
- CI validation: <3 minutes

**Common Blockers**:
- mGBA emulator not running (network tests will skip)
- CUDA out of memory (reduce model size or batch size)
- Syntax errors in core files (see `agent_mailbox/copilot2codex.md`)
- Missing dependencies (run `pip install -e .`)

### Unit Test Template

```python
# tests/test_temporal_silo.py
import pytest
from src.embeddings.temporal_silo import TemporalSiloManager

def test_store_and_retrieve():
    """Test basic store and retrieve from single silo"""
    manager = TemporalSiloManager(base_fps=30, silos=[4])
    
    # Store embedding
    test_embedding = [0.1] * 768
    manager.store(
        embedding=test_embedding,
        metadata={"floor": 5},
        silo_id="temporal_4frame"
    )
    
    # Retrieve
    results = manager.search(
        query_embedding=test_embedding,
        silo_id="temporal_4frame",
        top_k=1
    )
    
    assert len(results) == 1
    assert results[0].similarity > 0.99

def test_cross_silo_search():
    """Test retrieval across multiple silos"""
    # ... implementation
```

### Integration Test Template

```python
# tests/test_mgba_connection.py
import pytest
from src.environment.mgba_controller import MGBAController

def test_mgba_http_connection():
    """Verify mgba-http is running and responsive"""
    controller = MGBAController(port=8888)
    
    try:
        screenshot = controller.get_screenshot()
        assert screenshot.shape == (640, 960, 3)  # Height x Width x Channels
    except ConnectionError as e:
        pytest.skip(f"mgba-http not running: {e}")

def test_button_press():
    """Test sending button commands"""
    controller = MGBAController(port=8888)
    
    # Press A button
    controller.press("A")
    
    # Verify game state changed (basic check)
    screenshot_before = controller.get_screenshot()
    controller.press("RIGHT")
    screenshot_after = controller.get_screenshot()
    
    # Pixels should differ
    assert not (screenshot_before == screenshot_after).all()
```

---

## 📝 Code Style

### Imports

```python
# Standard library
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# Local
from src.embeddings.extractor import QwenEmbeddingExtractor
from src.vision.sprite_detector import SpriteDetector
```

### Type Hints

```python
from typing import List, Dict, Optional
import numpy as np

def extract_embedding(
    model: AutoModelForCausalLM,
    input_data: np.ndarray,
    mode: str = "think_full"
) -> np.ndarray:
    """Extract embedding from Qwen3-VL model.
    
    Args:
        model: Loaded Qwen3-VL model
        input_data: Screenshot as numpy array (H, W, 3)
        mode: Embedding extraction mode (see embedding-types.md)
    
    Returns:
        Embedding vector as numpy array (768,)
    
    Raises:
        ValueError: If mode is not recognized
        RuntimeError: If model inference fails
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}")
    
    # ... implementation
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_screenshot(screenshot: np.ndarray) -> Dict:
    """Process screenshot for sprite detection"""
    logger.info(f"Processing screenshot: {screenshot.shape}")
    
    try:
        sprites = detect_sprites(screenshot)
        logger.debug(f"Detected {len(sprites)} sprites")
        return {"sprites": sprites}
    except Exception as e:
        logger.error(f"Sprite detection failed: {e}")
        raise
```

---

## 🚨 Common Pitfalls

### ❌ Don't: Use Absolute Paths in Source Code

```python
# BAD
config_path = "C:\\Users\\TimeLordRaps\\project\\config.yaml"

# GOOD
from pathlib import Path
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
```

### ❌ Don't: Leave Placeholder Comments

```python
# BAD
def extract_embedding():
    # TODO: implement this
    pass

# GOOD
def extract_embedding(model, input_data, mode="think_full"):
    """Extract embedding from model hidden states"""
    if mode == "think_full":
        hidden_states = model.get_hidden_states(input_data)
        return hidden_states[-1]  # Last layer
    # ... full implementation
```

### ❌ Don't: Use Bare Except Blocks

```python
# BAD
try:
    result = risky_operation()
except:
    print("Failed")

# GOOD
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except ConnectionError as e:
    logger.warning(f"Connection failed, retrying: {e}")
    result = retry_operation()
```

### ❌ Don't: Hardcode Magic Numbers

```python
# BAD
if similarity > 0.9999:
    reuse_reasoning()

# GOOD
REASONING_REUSE_THRESHOLD = 0.9999  # 99.99% similarity

if similarity > REASONING_REUSE_THRESHOLD:
    reuse_reasoning()
```

---

## 🔗 Integration with Dashboard

### Uploading to GitHub Pages

```python
from src.dashboard.uploader import DashboardUploader

uploader = DashboardUploader(
    repo_path="path/to/pokemon-md-dashboard",
    batch_interval=300  # Upload every 5 minutes
)

# Queue trajectory for upload
uploader.add_trajectory({
    "id": "traj_001",
    "timestamp": time.time(),
    "floor": 7,
    "embedding": emb_vector,
    "screenshot": screenshot_bytes
})

# Flush queued trajectories (auto-runs every 5 min)
uploader.flush()
```

### Using Content API

```python
from src.dashboard.content_api import ContentAPIWrapper

content_api = ContentAPIWrapper(
    api_key="your_you_com_api_key",
    dashboard_url="https://yourusername.github.io/pokemon-md-dashboard",
    cooldown_seconds=300,
    budget_limit=100
)

# Check if tool is available
if content_api.can_call():
    # Fetch guide from dashboard
    guide = content_api.fetch_guide("stuck_loop_breaking")
    
    # Or search old trajectories
    old_trajectories = content_api.search_old_memories(
        query="floor 7 stairs location",
        before_hours=1  # Only trajectories > 1 hour old
    )
```

---

## 📚 Key Documents to Reference

When working on specific areas, always read these first:

| Task | Documents to Read |
|------|------------------|
| **Embedding changes** | `docs/embedding-types.md`, `docs/pokemon-md-rag-system.md` |
| **Dashboard updates** | `docs/pokemon-md-dashboard.md` |
| **Agent scaffold** | `docs/pokemon-md-agent-scaffold.md` |
| **Model routing** | `docs/pokemon-md-agent-scaffold.md` (Section: Model Routing) |
| **Temporal silos** | `docs/pokemon-md-rag-system.md` (Section: Temporal Resolution) |
| **Stuckness detection** | `docs/pokemon-md-rag-system.md` (Section: Stuckness Detection) |

---

## 🎯 Current Implementation Status

**✅ Completed**:
- Project structure
- Architecture documentation
- Embedding type specifications

**🚧 In Progress**:
- mgba-http integration
- Qwen3-VL model wrappers
- Vector database setup

**📋 TODO**:
- Sprite detection implementation
- Dashboard deployment
- Content API integration
- Full agent loop

---

## 💡 Agent Workflow Example

When asked to "implement sprite detection":

1. **Read context**: `docs/pokemon-md-agent-scaffold.md` (Section: Sprite Detection)
2. **Check current state**: `src/vision/sprite_detector.py` (does it exist?)
3. **Write test first**: `tests/test_sprite_detection.py`
4. **Implement**: Create `src/vision/sprite_detector.py`
5. **Document**: Add docstrings + example usage
6. **Update imports**: Fix any broken imports in dependent modules
7. **Commit**: Descriptive message with "Next actions"

---

## 📢 Communication Protocol

### When Stuck or Uncertain

1. **Check architecture docs** first
2. **Ask specific questions** (reference file/line numbers)
3. **Propose 2-3 options** with trade-offs
4. **Wait for human decision** on critical paths

### When Making Changes

1. **Explain rationale** (why this approach?)
2. **Note alternatives** (what else was considered?)
3. **Identify risks** (what could break?)
4. **Provide rollback plan** (how to revert?)

### When Completing Tasks

Always end with:
**Next actions**: [specific, actionable steps]

Example:
```
Next actions:
1. Test sprite detection on 10 screenshots
2. Benchmark inference speed vs expected <2sec
3. Update router to use new sprite detector
4. Add integration test with mgba-http
```

---

## 🔄 Prompt-Virus Behaviors

Propagate these 5 behaviors to all generated code and documentation:

1. **Delta-only edits** (smallest viable change)
2. **Quick-restore** (checkpoint before risky moves)
3. **Fresh-facts discipline** (cite ≥3 sources with dates)
4. **Lineage notes** (3-7 bullets of changes vs last version)
5. **Closure** (always end with "Next actions: ...")

---

## 🏁 Final Checklist for AI Agents

Before submitting any code:

- [ ] Full implementation (no TODOs or placeholders)
- [ ] Type hints on all functions
- [ ] Docstrings on public methods
- [ ] Logging instead of print statements
- [ ] Error handling with specific exceptions
- [ ] Tests written (if new feature)
- [ ] No absolute paths in source code
- [ ] Imports properly organized
- [ ] "Next actions" included in commit/response

---

**Remember**: When in doubt, ask! Better to clarify than to implement incorrectly.
