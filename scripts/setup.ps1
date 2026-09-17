param([switch]$SkipRuntimeChecks)
& (Join-Path $PSScriptRoot 'setup_clone.ps1') -SkipGpuCheck:$SkipRuntimeChecks
