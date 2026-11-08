@echo off
rem Self-contained environment in this folder, no Anaconda:
rem   1. download uv 0.13.7 (the official Windows package on PyPI), check its SHA-256, unpack uv.exe into _uv\
rem   2. uv downloads Python 3.13 into _uv\python and creates .venv\ here
rem   3. install the packages into .venv\, check that they import, run the timing
rem Double-click this file, or run its full path in PowerShell.
setlocal
cd /d "%~dp0"
if not exist bench_results mkdir bench_results
if not exist _uv mkdir _uv
set "LOG=%~dp0bench_results\setup_venv_log.txt"
set "WHL=%~dp0_uv\uv-0.13.7-py3-none-win_amd64.whl"
set "UV=%~dp0_uv\uv-0.13.7.data\scripts\uv.exe"
set "SHA=2835e2ab397d0db86323b9975dce75ecd76aa9cfa0d6a2aea84734aae7f8c9b3"
set "UV_PYTHON_INSTALL_DIR=%~dp0_uv\python"
set "UV_CACHE_DIR=%~dp0_uv\cache"
set "UV_PYTHON_PREFERENCE=only-managed"
echo [1/5] Getting uv 0.13.7 from PyPI and checking its SHA-256...
if exist "%UV%" goto :have_uv
curl.exe -sSL -o "%WHL%" https://files.pythonhosted.org/packages/8e/4f/041b59d2a38807a3e31ef41064cf3e45d2752fe0409d89040d52778049ae/uv-0.13.7-py3-none-win_amd64.whl >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 -LiteralPath '%WHL%').Hash -ne '%SHA%') { exit 1 }"
if errorlevel 1 (echo SHA-256 of the uv download does not match PyPI. >> "%LOG%" & goto :fail)
tar -xf "%WHL%" -C "%~dp0_uv" uv-0.13.7.data/scripts/uv.exe >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
:have_uv
echo [2/5] Creating .venv with Python 3.13, downloaded into this folder...
"%UV%" venv --python 3.13 --allow-existing "%~dp0.venv" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo [3/5] Installing jax, numpy, scipy, scikit-fem, triangle, psutil...
"%UV%" pip install --python "%~dp0.venv\Scripts\python.exe" jax numpy scipy scikit-fem triangle psutil >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo [4/5] Checking the installation...
"%~dp0.venv\Scripts\python.exe" -c "import jax, numpy, scipy, skfem, triangle, psutil; print('jax', jax.__version__, jax.devices())" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo [5/5] Timing, about 5 to 8 minutes. Keep the laptop plugged in with Game Shift on; the fans will be loud.
"%~dp0.venv\Scripts\python.exe" "%~dp0bench_laptop.py"
if errorlevel 1 goto :fail
echo.
echo Done. The results are in the bench_results folder; Claude can read them there.
pause
exit /b 0
:fail
echo.
echo Something failed. The log is bench_results\setup_venv_log.txt; Claude can read it.
pause
exit /b 1
