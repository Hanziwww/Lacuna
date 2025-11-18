---
html_theme.sidebar_primary.remove: true
---
# Development Guide

## Prerequisites

- **Python**: 3.10–3.13
- **Rust**: Stable toolchain with `cargo`
- **maturin**: `pip install maturin`
- **Virtual environment**: `python -m venv .venv` (recommended)
- **Platforms**: Windows, Linux, macOS

## Quickstart (Windows PowerShell)

```powershell
.venv/Scripts/Activate.ps1
maturin develop -m crates/lacuna-py/Cargo.toml
cargo test -q
pytest -q python/tests
```

## Quickstart (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop -m crates/lacuna-py/Cargo.toml
cargo test -q
pytest -q python/tests
```

## Development build

- `maturin develop -m crates/lacuna-py/Cargo.toml`
  - Compiles Rust crates and installs the Python module into the active venv for iterative dev.

## Testing

- **Rust tests**: `cargo test -q`
- **Python tests**: `pytest -q python/tests`

## Formatting and linting

- **Rust format**: `cargo fmt --all`
- **Rust lint (Clippy)**: `cargo clippy -q --all-targets --all-features`

## Benchmarks

- Run benchmark scripts directly, e.g.:
  - `python python/benchmarks/benchmark_spmv.py`
  - `python python/benchmarks/benchmark_spmm.py`

## Building wheels

- Windows: `./scripts/build_wheels.ps1`
- macOS/Linux: `bash ./scripts/build_wheels.sh`

## Repository layout (high level)

- `crates/` Rust crates
  - `lacuna-core/` core data structures and reference kernels
  - `lacuna-kernels/` optimized parallel/SIMD kernels
  - `lacuna-io/` IO utilities
  - `lacuna-py/` PyO3 bindings
- `python/` Python package `lacun/` and tests/benchmarks
- `docs/` Sphinx docs (MyST Markdown)
- `scripts/` helper scripts (wheels, release, datasets)

## Contribution workflow

1. Create a feature branch.
2. Develop with `maturin develop` and an active venv.
3. Run `cargo test -q` and `pytest -q python/tests`.
4. Ensure `cargo fmt` and `cargo clippy` pass.
5. Open a PR with a short description of changes and benchmarks if relevant.
