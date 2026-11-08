@echo off
rem Re-run the timing only, with the environment in .venv\ made by setup_venv_and_bench.bat.
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0bench_laptop.py"
pause
