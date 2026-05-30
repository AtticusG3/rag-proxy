# Run offline pytest suite from repo root.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$DevReq = Join-Path $RepoRoot "requirements-dev.txt"
if (Test-Path $DevReq) {
    & $Python -m pip install -q -r $DevReq
}

& $Python -m pytest tests/ -q --tb=short
exit $LASTEXITCODE
