@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" -I test_verdicts.py > tests_log.txt 2>&1
"%~dp0..\.venv\Scripts\python.exe" -I test_stamp.py >> tests_log.txt 2>&1
type tests_log.txt
"%~dp0..\.venv\Scripts\python.exe" run_formal.py --smoke
echo.
echo Smoke test finished. See results_smoke\progress.txt and tests_log.txt
pause
