#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Upgrading pip..."
python -m pip install --upgrade pip

echo "[2/5] Installing Python tooling (maturin, pytest, ruff)..."
pip install maturin pytest ruff

echo "[3/5] Building and installing extension with maturin develop..."
maturin develop -m crates/lacuna-py/Cargo.toml

echo "[4/5] Running pytest..."
pytest -q python/tests

echo "[5/5] Running ruff checks (format + import sorting)..."
ruff format --check
ruff check --select I --diff

echo "All Python checks completed."
