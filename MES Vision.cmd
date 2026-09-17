@echo off
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Python environment not found. Run scripts\setup.ps1 first.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\desktop.py"
