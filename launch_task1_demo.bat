@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "task1_demo.py"
) else (
  echo Install dependencies with: python -m pip install -r requirements-inference.txt
  python task1_demo.py
  pause
)
