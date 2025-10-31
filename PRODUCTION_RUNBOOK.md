# Production Runbook: Pokemon MD Agent Demo (3-Minute Video)

**Objective**: Generate a 3-minute autonomous gameplay video from a single mGBA session.

**Time Budget**: 20 minutes total
**Output**: `agent_demo.mp4` (ready for presentation)

---

## Phase 1: Pre-Flight Check (2 minutes)

### Step 1.1: Verify Environment
```bash
mamba info --envs && python --version && mamba activate agent-hackathon && pwd && ls -la
```

**Expected Output:**
- `agent-hackathon` environment active
- Python 3.11+ or 3.12
- Current dir: `pokemon-md-agent`

**Troubleshooting:**
- If no `agent-hackathon` env: `mamba create -n agent-hackathon python=3.11 -y && mamba activate agent-hackathon && pip install -r requirements.txt`

### Step 1.2: Verify ROM & SAV Files
```bash
ls -la rom/
```

**Expected Output:**
```
-rw-r--r-- ... Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba
-rw-r--r-- ... Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).sav
```

**Troubleshooting:**
- If missing: Copy ROM & SAV files to `./rom/` directory before proceeding

### Step 1.3: Verify mGBA Socket Server
```bash
python .temp_check_ram.py
```

**Expected Output:**
```
--- Player State ---
floor_number: 1       # Non-zero if game is active
dungeon_id: 0         # Should match loaded dungeon
...
```

**If connection fails:**
1. **Start mGBA** (if not already running)
2. **Load Pokemon Mystery Dungeon ROM** (File → Open ROM)
3. **Load the SAV file** (File → Load Save → `rom/Pokemon...sav`)
4. **Ensure Lua socket server is running**:
   - Lua Console → File → Load script → `mGBASocketServer.lua`
   - You should see: `Listening on port 8888`
5. **Retry `.temp_check_ram.py`**

**If socket server missing:**
- The Lua socket server should be bundled with mGBA v0.8.0+
- If not: Download `mGBASocketServer.lua` from the mGBA-http repository
- Load it via Lua Console → File → Load script

### Step 1.4: Validate You.com Content API (optional but recommended)
```bash
# Windows PowerShell (expects YOU_API_KEY already configured)
python -m scripts.check_you_api --url https://www.serebii.net/dungeon/redblue/d001.shtml --live

# macOS/Linux
python -m scripts.check_you_api --url https://www.serebii.net/dungeon/redblue/d001.shtml --live
```

**Expected Output:**
```
YOU_API_KEY configured: yes
Mock mode: False
• https://www.serebii.net/dungeon/redblue/d001.shtml -> OK | <first line preview>
```

**Troubleshooting:**
- If you see `Mock mode: True`, the key is missing — export `YOU_API_KEY` and rerun.
- `ERROR (You.com API rejected the request (401)...)` → key invalid or expired.
- To skip live calls (offline demos), omit `--live`; the system will generate placeholder content.

---

## Phase 2: Execute Demo (2-3 minutes)

### Step 2.1: Run Final Demo Pipeline
```bash
cd pokemon-md-agent
python scripts/final_demo_runner.py
```

**Pipeline stages** (auto-runs):
1. **Agent Execution** (~1-2 min)
   - Agent navigates for 50 steps
   - Outputs: `runs/demo_*/trajectory_*.jsonl`

2. **Validation** (~5 sec)
   - Checks trajectory contains ≥10 frames

3. **Video + Voiceover Generation** (~10-20 sec)
   - Samples frames at 15 FPS
   - Target duration: 180 seconds (3 minutes)
   - Runs Kokoro TTS (hexgrad/Kokoro-82M) for narration
   - Output: `agent_demo.mp4`

**Console Output:**
```
============================================================
PHASE 1: AGENT AUTONOMOUS DEMO
============================================================
Starting agent demo (50 steps)...
✓ Agent demo completed successfully

============================================================
PHASE 2: VALIDATION
============================================================
✓ Trajectory: 45 frames logged

============================================================
PHASE 3: VIDEO GENERATION
============================================================
✓ Video saved: agent_demo.mp4
  Duration: 180.5 seconds

✓ DEMO COMPLETE!
```

### Step 2.2: Verify Output
```bash
ls -la agent_demo.mp4 && ffprobe agent_demo.mp4
```

**Expected:**
- File size: 10-50 MB (typical)
- Duration: ~180 seconds
- Codec: h264 or h265, AAC audio track (Kokoro voiceover)

---

## Phase 3: Post-Production (Optional)

### Option A: Direct Use
```bash
# Play directly
ffplay agent_demo.mp4

# Copy to clipboard (Windows)
copy agent_demo.mp4 ..\..\demos\
```

### Option B: Transcoding for Platform
```bash
# YouTube (8 Mbps)
ffmpeg -i agent_demo.mp4 -vb 8M -ab 128k -c:v libx264 -preset slow agent_demo_youtube.mp4

# Twitter (15 Mbps, square)
ffmpeg -i agent_demo.mp4 -vb 15M -vf "scale=1080:1080" agent_demo_square.mp4

# Discord (embed, 25 MB limit)
ffmpeg -i agent_demo.mp4 -vf "scale=1280:720" -c:v libx264 -crf 23 agent_demo_discord.mp4
```

---

## Troubleshooting Reference

### Agent Fails to Initialize
**Error:** `Failed to connect to mGBA`

**Solutions:**
1. Check mGBA is running: `netstat -an | grep 8888` (should show listening)
2. Restart mGBA Lua socket server (Lua Console → File → Load script)
3. Try restarting mGBA entirely
4. Check if another process is using port 8888: Change port in `config/addresses/pmd_red_us_v1.json`

### Agent Crashes During Execution
**Error:** `Perception failed: unpack requires a buffer of 1 bytes`

**Solution:** This is now handled gracefully (returns 0 for missing fields). If persists:
1. Restart mGBA + Lua socket server
2. Reload the ROM + SAV file
3. Check RAM addresses in config file

### Video Generation Fails
**Error:** `Video generation failed`

**Solutions:**
1. Install dependencies: `pip install opencv-python pillow`
2. Check trajectory file exists: `ls runs/demo_*/trajectory_*.jsonl`
3. Check write permissions: `ls -ld .` (should show `d..w..w`)
4. Check disk space: `df -h .` (need ~100 MB free)

### Video Has Wrong Dimensions
**Issue:** Video is distorted or wrong aspect ratio

**Solution:**
- Confirm mGBA resolution: 960×640 (emulator settings)
- Re-run video generation with explicit dims:
```bash
python scripts/generate_montage_video.py --run-dir runs/demo_XXXXX --output agent_demo_fixed.mp4
```

---

## Success Criteria

✅ **All phases complete without error**
✅ **agent_demo.mp4 exists and plays**
✅ **Video duration ~2-3 minutes**
✅ **Trajectory has ≥10 frames**
✅ **Console shows "DEMO COMPLETE!" message**

---

## Quick Reference: Full Command Sequence

```bash
# Activate environment
mamba activate agent-hackathon

# Navigate to project
cd /path/to/pokemon-md-agent

# Verify setup
python .temp_check_ram.py
python -m scripts.check_you_api --url https://www.serebii.net/dungeon/redblue/d001.shtml --live

# Run demo
python scripts/final_demo_runner.py

# Verify output
ls -la agent_demo.mp4
ffplay agent_demo.mp4
```

---

## Contact / Debug Logs

If demo fails:
1. Capture full console output: `python scripts/final_demo_runner.py > demo_log.txt 2>&1`
2. Check agent logs: `cat logs/agent_*.log | tail -100`
3. Include: `demo_log.txt`, `logs/`, and error screenshot

---

**Last Updated:** 2025-10-30
**Agent Version:** v6.1 (lean edit-semantics, aligned with Copilot Instructions v1.1)
