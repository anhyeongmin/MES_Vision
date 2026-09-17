@echo off
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Python environment is missing.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\manual_inspection.py"
