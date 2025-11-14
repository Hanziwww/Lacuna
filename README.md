<p align="center">
  <img src="docs/logo.png" alt="Lacuna logo" width="300">

</p>

<h1 align="center">Lacuna</h1>

<p align="center">High-performance sparse arrays/matrices for Python with a pure Rust core.</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.90%2B-93450a?logo=rust&logoColor=white" alt="Rust 1.90+" /></a>
  <a href="https://pyo3.rs/"><img src="https://img.shields.io/badge/PyO3-enabled-orange" alt="PyO3 enabled" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white" alt="Python 3.10–3.13" /></a>
  <a href="https://crates.io/crates/wide"><img src="https://img.shields.io/badge/SIMD-wide-0A7BBB" alt="SIMD wide" /></a>
  <a href="https://crates.io/crates/rayon"><img src="https://img.shields.io/badge/Rayon-parallelism-5C6BC0" alt="Rayon parallelism" /></a>
  <img src="https://img.shields.io/badge/Status-Dev-green" alt="Status: Dev" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-purple" alt="License: Apache 2.0" />
</p>

 ⚠ Status: Work in progress. APIs and performance characteristics may change without notice.

## Quick start (development install)

Requirements:

- Python 3.10–3.13
- Rust toolchain (stable)
- pip

Install with maturin (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U maturin
maturin develop -m crates/lacuna-py/Cargo.toml --release
```

Verify:

```powershell
python -c "import lacuna as la; print('threads:', la.get_num_threads())"
```

## Basic usage

```python
import numpy as np
from lacuna.sparse.csr import CSR

# A = [[1, 0, 2],
#      [0, 3, 0]]
indptr = np.array([0, 2, 3], dtype=np.int64)
indices = np.array([0, 2, 1], dtype=np.int64)
data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
A = CSR(indptr, indices, data, (2, 3), check=False)

# SpMV
x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
y = A @ x  # -> array([1*10 + 2*30, 3*20])

# SpMM
B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
Y = A @ B  # shape (2, 2)

# Reductions
total = A.sum()            # 6.0
row_sums = A.sum(axis=1)   # [3.0, 3.0]
col_sums = A.sum(axis=0)   # [1.0, 3.0, 2.0]

# Transpose
AT = A.T  # (3, 2)

# Arithmetic
C = A + A
Z = A - A
H = A.multiply(A)  # Hadamard (elementwise)

# Cleanup
Az = A.eliminate_zeros()
Ap = A.prune(1e-6)

# Indexing
a00 = A[0, 0]      # 1.0
row0 = A[0, :]     # dense row (numpy)
col1 = A[:, 1]     # dense col (numpy)
```

## Build the documentation

```powershell
python -m pip install -U sphinx pydata-sphinx-theme myst-parser
python -m sphinx -b html docs docs/_build/html
Start-Process docs\_build\html\index.html
```

## Tiny Benchmark

## Contributing

We welcome contributors! If you're interested in sparse matrix, feel free to open an issue or pull request. The project is under active development, so expect rapid iteration and occasional breaking changes.

## License

Apache-2.0
