@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_clone.ps1"
if errorlevel 1 (
  echo Setup failed. Read the error above.
  pause
  exit /b 1
)
echo Ready. Open MES Vision.cmd or MES Vision Manual.cmd.
pause
