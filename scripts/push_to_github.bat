@echo off
REM Quick push to GitHub after demo
REM Usage: push_to_github.bat

setlocal enabledelayedexpansion

echo.
echo ========================================
echo GitHub Push Automation
echo ========================================
echo.

REM Verify we're in the right directory
if not exist "README.md" (
    echo ERROR: Not in pokemon-md-agent directory
    exit /b 1
)

REM Check git status
echo.
echo Current git status:
git status --short | findstr /v /c:"" | head -5
echo.

REM Confirm
set /p confirm="Ready to push to GitHub? (y/n): "
if /i not "!confirm!"=="y" (
    echo Cancelled
    exit /b 0
)

REM Configure remote if not already done
echo.
echo Configuring remote...
git remote | findstr "origin" >nul
if errorlevel 1 (
    echo Adding remote: https://github.com/TimeLordRaps/pokemon-md-agent.git
    git remote add origin https://github.com/TimeLordRaps/pokemon-md-agent.git
) else (
    echo Remote already configured
    git remote -v | findstr "origin"
)

REM Ensure main branch
echo.
echo Switching to main branch...
git branch -M main

REM Push
echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo SUCCESS! Repository pushed to GitHub
echo ========================================
echo.
echo View at: https://github.com/TimeLordRaps/pokemon-md-agent
echo.

REM Show recent commits
echo Latest commits:
git log --oneline -3
echo.
echo Done!
pause
