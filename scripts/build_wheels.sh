#!/usr/bin/env bash
set -euo pipefail
maturin build --release -m pyproject.toml
