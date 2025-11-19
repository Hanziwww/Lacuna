#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "[1/5] Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "[2/5] Installing Python tooling (maturin, pytest, ruff)..." -ForegroundColor Cyan
pip install maturin pytest ruff

Write-Host "[3/5] Building and installing extension with maturin develop..." -ForegroundColor Cyan
maturin develop --release

Write-Host "[4/5] Running pytest..." -ForegroundColor Cyan
pytest -q python/tests

Write-Host "[5/5] Running ruff checks (format + import sorting)..." -ForegroundColor Cyan
ruff format
ruff check --select I --fix

Write-Host "All Python checks completed." -ForegroundColor Green
