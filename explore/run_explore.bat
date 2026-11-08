@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" run_explore.py
echo.
echo Finished. See results\progress.txt
pause
