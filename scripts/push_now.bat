@echo off
REM One-click push to GitHub
REM Usage: push_now.bat

echo.
echo ========================================
echo Quick Push to GitHub
echo ========================================
echo.

REM Verify we're in the right directory
if not exist "README.md" (
    echo ERROR: Not in pokemon-md-agent directory
    exit /b 1
)

REM Add all changes
echo Adding all changes...
git add -A

REM Check if there are changes to commit
git status --porcelain | findstr /r /c:".*" >nul
if errorlevel 1 (
    echo No changes to commit
    exit /b 0
)

REM Commit
echo Committing changes...
git commit -m "chore: publish current demo + docs"

REM Push
echo Pushing to GitHub...
git push origin main

echo.
echo ========================================
echo SUCCESS! Pushed to GitHub
echo ========================================
echo.
echo View at: https://github.com/TimeLordRaps/pokemon-md-agent
echo.