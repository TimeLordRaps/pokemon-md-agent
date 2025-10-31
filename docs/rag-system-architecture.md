# Pokemon MD RAG System (CORRECTED)

## Embedding Types

**Input**: `input` - Free from any inference

**Thinking**: think_input, think_full, think_only, think_image_*
**Instruct**: instruct_eos, instruct_image_only

## 7 Temporal Silos @ 960x640 @ 30fps

| Silo | Sample Rate | Time Span | Use Case |
|------|-------------|-----------|----------|
| temporal_1frame | Every frame | 0-4 sec | Immediate |
| temporal_2frame | Every 2nd | 0-8 sec | Combat |
| temporal_4frame | Every 4th | 0-16 sec | Navigation |
| temporal_8frame | Every 8th | 0-32 sec | Room explore |
| temporal_16frame | Every 16th | 0-64 sec | Floor strategy |
| temporal_32frame | Every 32nd | 0-128 sec | Long planning |
| temporal_64frame | Every 64th | 2+ min | Cross-floor |

## Dynamic FPS Control

Agent can adjust: 30→10→5→3→1 fps (zoom out)
Frame multiplier: 4x→8x→12x→16x (zoom in)

## Storage Split

**Local (<1 hour)**: All data, fast retrieval
**Dashboard (>1 hour)**: GitHub Pages, Content API (100 calls, 5 min cooldown)

## Retrieval

**Auto**: 3 trajectories every inference
**Manual**: Dashboard tool (rare, stuck_counter > 5)

## Episode-Aware Temporal Memory

- `TemporalSiloManager` now tracks `episode_id` boundaries using floor change events and savestate loads, ensuring contiguous dungeon runs stay isolated.
- Each episode maintains its own FAISS index for similarity search; cross-episode queries combine the top-k hits per episode before global re-ranking.
- Recency-aware retrieval applies a configurable decay factor (`DEFAULT_DECAY_FACTOR_PER_HOUR`, default `0.001`) so newer memories surface ~20% more often than stale ones.

## Scratchpad

Guaranteed carryforward memory:
```python
agent.scratchpad.write("Mission: Rescue Caterpie Floor 5")
agent.scratchpad.read()  # Always in next context
```

## Next Actions

Set up ChromaDB, implement auto_retrieve(), test stuck detection
