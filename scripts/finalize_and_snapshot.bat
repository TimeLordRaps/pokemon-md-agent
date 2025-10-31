@echo off
REM Task B + Task C: Move video to Pages, commit, then snapshot at deadline
REM Usage: finalize_and_snapshot.bat

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Finalization + Deadline Snapshot
echo ========================================
echo.

REM Check if video exists in root
if not exist "agent_demo.mp4" (
    echo ERROR: agent_demo.mp4 not found in current directory
    echo Please run: python scripts/final_demo_runner.py
    exit /b 1
)

echo [1/3] Moving agent_demo.mp4 to docs/assets/
copy "agent_demo.mp4" "docs\assets\agent_demo.mp4"
if errorlevel 1 (
    echo ERROR: Failed to copy video to docs/assets/
    exit /b 1
)
echo OK

echo.
echo [2/3] Committing video to GitHub Pages
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
if errorlevel 1 (
    echo ERROR: Failed to commit video
    exit /b 1
)
echo OK

echo.
echo [3/3] Pre-deadline push (T minus ~15 minutes)
git push -u origin main
if errorlevel 1 (
    echo WARNING: Push failed, but continuing to snapshot commands
)
echo OK

echo.
echo ========================================
echo SNAPSHOT COMMANDS (choose one):
echo ========================================
echo.
echo Option A: Schedule automatic snapshots (Windows Task Scheduler)
echo   Run this PowerShell command at 23:50 PT to schedule:
echo.
powershell -NoProfile -Command ^
  "$repo='%cd%'; " ^
  "$cmdline='powershell -NoProfile -Command cd `\"' + $repo + '\"; ' ^
  "git switch -c deadline-2025-10-30-2355-PT; ' ^
  "git add -A; ' ^
  "git commit -m \" ^
  "\"Deadline snapshot 23:55 PT\"; ' ^
  "git push -u origin deadline-2025-10-30-2355-PT; ' ^
  "Start-Sleep -Seconds 240; ' ^
  "git tag deadline-2025-10-30-2359-PT; ' ^
  "git push origin deadline-2025-10-30-2359-PT; ' ^
  "Write-Host \"Snapshot complete - submitted!\" -ForegroundColor Green'; " ^
  "schtasks /Create /SC ONCE /TN PMD-Deadline-Snapshot /TR $cmdline /ST 23:55 /F"
echo.
echo   OR manually run this script again at 23:55 PT with parameter:
echo   finalize_and_snapshot.bat snapshot
echo.
echo Option B: Manual command (run at 23:55 PT)
echo   git switch -c deadline-2025-10-30-2355-PT
echo   git add -A ^&^& git commit -m "Deadline snapshot 23:55 PT" ^&^& git push -u origin deadline-2025-10-30-2355-PT
echo   timeout /t 240 /nobreak
echo   git tag deadline-2025-10-30-2359-PT ^&^& git push origin deadline-2025-10-30-2359-PT
echo.

REM If called with "snapshot" parameter, run the deadline snapshot now
if "%1"=="snapshot" (
    echo.
    echo Running deadline snapshot NOW...
    git switch -c deadline-2025-10-30-2355-PT
    git add -A
    git commit -m "Deadline snapshot 23:55 PT (manual)"
    git push -u origin deadline-2025-10-30-2355-PT
    timeout /t 5 /nobreak
    git tag deadline-2025-10-30-2359-PT
    git push origin deadline-2025-10-30-2359-PT
    echo.
    echo Snapshot complete!
    goto end
)

echo.
echo ========================================
echo Next steps:
echo 1. Verify video at: https://github.com/TimeLordRaps/pokemon-md-agent
echo 2. At 23:55 PT, run: finalize_and_snapshot.bat snapshot
echo    (or use PowerShell scheduler command above)
echo 3. Verify branch + tag at GitHub
echo ========================================
echo.

:end
pause
