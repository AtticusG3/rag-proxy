# Remove local build/test caches from the working tree (safe; gitignored).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$names = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$removed = 0
Get-ChildItem -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object {
        ($names -contains $_.Name) -and
        ($_.FullName -notmatch '[\\/]\.venv[\\/]') -and
        ($_.FullName -notmatch '[\\/]venv[\\/]') -and
        ($_.FullName -notmatch '[\\/]\.git[\\/]')
    } |
    ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName
        $removed++
        Write-Host "removed $($_.FullName)"
    }
Write-Host "[OK] removed $removed cache director(ies)"
