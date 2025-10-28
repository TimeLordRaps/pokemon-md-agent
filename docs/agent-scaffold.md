# Pokemon MD Agent Scaffold

## mgba Settings

Resolution: 960x640 (4x)
Framerate: 30fps
Filters: None
Color: 24-bit RGB

## Architecture

```
mgba (960x640@30fps)
  ↓
Sprite Detector (Qwen3-VL-4B-Instruct)
  ↓
Vector DB (7 silos + scratchpad)
  ↓
Router (2B/4B/8B)
  ↓
mgba Controller
```

## Control Loop

1. Screenshot from mgba
2. Detect sprites
3. Read scratchpad
4. Auto-retrieve 3 trajectories
5. Route to model (2B/4B/8B)
6. Get action decision
7. Press button
8. Store trajectory
9. Update dashboard (every 5 min)

## Model Routing

- 2B: confidence ≥ 0.8
- 4B: 0.6 ≤ confidence < 0.8
- 8B: confidence < 0.6 OR stuck > 5

## Next Actions

Install mgba-http, test Qwen3-VL-4B sprite detection, build loop
