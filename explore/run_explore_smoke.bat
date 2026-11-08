@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" run_explore.py --smoke
echo.
echo Smoke test finished. See results_smoke\progress.txt
pause
