@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" run_formal.py
echo.
echo Finished. See results\progress.txt and results\verdicts.md
pause
