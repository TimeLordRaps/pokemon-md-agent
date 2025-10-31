#!/bin/bash
# Task B + Task C: Move video to Pages, commit, then snapshot at deadline
# Usage: bash scripts/finalize_and_snapshot.sh [snapshot]

set -e

echo ""
echo "========================================"
echo "Finalization + Deadline Snapshot"
echo "========================================"
echo ""

# Check if video exists
if [ ! -f "agent_demo.mp4" ]; then
    echo "ERROR: agent_demo.mp4 not found in current directory"
    echo "Please run: python scripts/final_demo_runner.py"
    exit 1
fi

echo "[1/3] Moving agent_demo.mp4 to docs/assets/"
mkdir -p docs/assets
cp agent_demo.mp4 docs/assets/agent_demo.mp4
echo "OK"

echo ""
echo "[2/3] Committing video to GitHub Pages"
git add docs/assets/agent_demo.mp4
git commit -m "feat: add 3-minute demo video with TTS narration to GitHub Pages

Final demo output: agent_demo.mp4
- 180 seconds of autonomous agent gameplay
- Kokoro TTS narration with decision reasoning
- Tiny Woods exploration with You.com knowledge retrieval
- Automatically generated from trajectory + LLM voiceover

Video hosted at: docs/assets/agent_demo.mp4
Accessible via GitHub Pages after push.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
echo "OK"

echo ""
echo "[3/3] Pre-deadline push (T minus ~15 minutes)"
git push -u origin main || echo "WARNING: Push may have failed"
echo "OK"

echo ""
echo "========================================"
echo "SNAPSHOT COMMANDS (choose one):"
echo "========================================"
echo ""

# If called with "snapshot" parameter, run the deadline snapshot
if [ "$1" == "snapshot" ]; then
    echo "Running deadline snapshot NOW..."
    echo ""
    git switch -c deadline-2025-10-30-2355-PT
    git add -A
    git commit -m "Deadline snapshot 23:55 PT (manual)"
    git push -u origin deadline-2025-10-30-2355-PT
    echo ""
    echo "Waiting 240 seconds before final tag..."
    sleep 240
    echo ""
    git tag deadline-2025-10-30-2359-PT
    git push origin deadline-2025-10-30-2359-PT
    echo ""
    echo "Snapshot complete! Branch and tag pushed."
    echo ""
    exit 0
fi

echo "Option A: Automatic snapshot at 23:55 PT (macOS/Linux only)"
echo "  Run this at 23:50 PT:"
echo ""
echo "  at 23:55 << 'EOF'"
echo "  cd $(pwd)"
echo "  bash scripts/finalize_and_snapshot.sh snapshot"
echo "  EOF"
echo ""
echo "Option B: Manual command (run at 23:55 PT)"
echo "  git switch -c deadline-2025-10-30-2355-PT"
echo "  git add -A && git commit -m 'Deadline snapshot 23:55 PT' && git push -u origin deadline-2025-10-30-2355-PT"
echo "  sleep 240"
echo "  git tag deadline-2025-10-30-2359-PT && git push origin deadline-2025-10-30-2359-PT"
echo ""

echo "========================================"
echo "Next steps:"
echo "1. Verify video at: https://github.com/TimeLordRaps/pokemon-md-agent"
echo "2. At 23:55 PT, run: bash scripts/finalize_and_snapshot.sh snapshot"
echo "3. Verify branch + tag at GitHub"
echo "========================================"
echo ""
