param([switch]$SkipGpuCheck)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install GitHub CLI (winget install GitHub.cli), then run gh auth login.' }
& gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Sign in with gh auth login using an account allowed to read this private repository.' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 (winget install Python.Python.3.12), reopen this window and retry.' }
}
$projectPython = Join-Path $PWD '.venv/Scripts/python.exe'
& $projectPython -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 required'"
if ($LASTEXITCODE -ne 0) { throw 'This virtual environment must use Python 3.12.' }
& $projectPython -m pip install --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $projectPython -m pip install --disable-pip-version-check -r requirements-lock.txt --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $projectPython -B -X utf8 scripts/restore_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'Runtime asset restore failed. Re-run Setup.cmd to retry.' }
if ($SkipGpuCheck) {
    & $projectPython -B -X utf8 scripts/check_clone.py
} else {
    & $projectPython -B -X utf8 scripts/check_clone.py --gpu
}
if ($LASTEXITCODE -ne 0) { throw 'Startup verification failed.' }
Write-Output 'Setup completed. Camera selection and robot coordinate calibration remain hardware-specific.'
