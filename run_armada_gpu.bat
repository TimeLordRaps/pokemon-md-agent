@echo off
REM Launch armada test with proper GPU environment
echo Activating agent-hackathon GPU environment...
call activate agent-hackathon

echo Configuring environment variables...
set PMD_ROM=C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba
set PMD_SAVE=C:\Homework\agent_hackathon\rom\Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).sav
set MGBALUA=C:\Homework\agent_hackathon\pokemon-md-agent\src\mgba-harness\mgba-http\mGBASocketServer.lua
set MGBAX=C:\Program Files\mGBA\mGBA.exe

echo Starting 6-model armada test with GPU...
cd /d "C:\Homework\agent_hackathon\pokemon-md-agent"
python launch_armada_test.py 2>&1 | tee test_run_armada_gpu.log

pause
