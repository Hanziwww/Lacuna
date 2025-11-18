<p align="center">
  <img src="docs/logo.png" alt="Lacuna logo" width="300">

</p>
<p align="center">High-performance Sparse Matrices for Python.</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.90%2B-93450a?logo=rust&logoColor=white" alt="Rust 1.90+" /></a>
  <a href="https://pyo3.rs/"><img src="https://img.shields.io/badge/PyO3-enabled-orange" alt="PyO3 enabled" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white" alt="Python 3.10–3.13" /></a>
  <img src="https://img.shields.io/badge/Status-Dev-green" alt="Status: Dev" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-purple" alt="License: Apache 2.0" />
  <br/>
  <a href="https://crates.io/crates/wide"><img src="https://img.shields.io/badge/SIMD-wide-0A7BBB" alt="SIMD wide" /></a>
  <a href="https://crates.io/crates/rayon"><img src="https://img.shields.io/badge/Rayon-parallelism-5C6BC0" alt="Rayon parallelism" /></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-compatible-013243?logo=numpy&logoColor=white" alt="NumPy compatible" /></a>
</p>

> ⚠ Status: Work in progress. APIs and performance characteristics may change without notice. **Contact me if you'd like to join the development**! View [docs](https://lacuna.hanziwww.me).

## Table of Contents

- [Features](#features)
- [Quick start (development install)](#quick-start-development-install)
- [Basic usage](#basic-usage)
  - [CSC and COO quick examples](#csc-and-coo-quick-examples)
  - [ND COO (COOND) examples](#nd-coo-coond-examples)
- [Why Lacuna](#why-lacuna)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)

## Features

- Formats

  - CSR, CSC, COO (2D)
  - COOND (N-D COO tensors)
- Kernels (f64 values, i64 indices)

  - SpMV, SpMM
  - Reductions: sum, row/col sums
  - Transpose
  - Arithmetic: add, sub, Hadamard (elementwise)
  - Cleanup: prune(eps), eliminate_zeros
  - More comming soon...
- ND COO operations

  - sum, mean
  - reduce_sum_axes / reduce_mean_axes
  - permute_axes, reshape
  - hadamard_broadcast
  - unfold to CSR/CSC (mode and grouped axes)
- Python API

  - Simple, NumPy-friendly classes with zero-copy reads of input buffers when safe

## Quick start (development install)

Requirements:

- Python 3.10–3.13
- Rust toolchain (stable)
- pip

Install with maturin:

```powershell
python -m venv .venv
# then activate venv

python -m pip install -U maturin
maturin develop -m crates/lacuna-py/Cargo.toml --release
```

Verify:

```powershell
python -c "import lacuna as la; print('threads:', la.get_num_threads())"
```

Thread control:

```python
import lacuna as la

# Set number of Rayon threads used by Rust kernels
la.set_num_threads(8)
print("threads:", la.get_num_threads())
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

### CSC and COO quick examples

```python
import numpy as np
from lacuna.sparse import CSC, COO

# CSC
indptr = np.array([0, 1, 2, 3], dtype=np.int64)
indices = np.array([0, 1, 1], dtype=np.int64)
data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
C = CSC(indptr, indices, data, (2, 3), check=False)
_ = C.sum()            # total sum
_ = C.sum(axis=0)      # column sums
_ = C.sum(axis=1)      # row sums
CT = C.T               # transpose (still CSC)

# COO
row = np.array([0, 1, 1], dtype=np.int64)
col = np.array([0, 0, 2], dtype=np.int64)
val = np.array([1.0, 2.0, 3.0], dtype=np.float64)
O = COO(row, col, val, (2, 3), check=False)
_ = O.sum()            # total sum
OT = O.T               # transpose (COO)
```

### ND COO (COOND) examples

```python
import numpy as np
from lacuna.sparse import COOND

shape = (2, 3, 4)
# indices can be flattened (nnz * ndim)
indices = np.array([
    0, 1, 2,
    1, 2, 3,
], dtype=np.int64)
data = np.array([1.0, 3.0], dtype=np.float64)
A = COOND(shape, indices, data, check=False)

total = A.sum()                # 4.0
avg = A.mean()                 # mean over full dense shape
B = A.permute_axes([2, 1, 0])  # shape -> (4, 3, 2)
R = A.reduce_sum_axes([2])     # sum over last axis -> shape (2, 3)
M = A.reduce_mean_axes([0, 2]) # mean over axes 0 and 2 -> shape (3,)
S = A.reshape((3, 2, 4))
H = A.hadamard_broadcast(S)    # broadcasting elementwise product

# Unfold to 2D sparse matrices
CSR0 = A.mode_unfold_to_csr(axis=0)  # rows = shape[0], cols = prod(shape[1:])
CSC1 = A.mode_unfold_to_csc(axis=1)
CSRrg = A.axes_unfold_to_csr([0, 2]) # group axes as rows
CSCrg = A.axes_unfold_to_csc([1])    # group axes as rows (CSC)
```

## Why Lacuna

Lacuna bridges matrix-first sparse libraries and array-first ND sparse arrays: high-performance Rust kernels with a simple, NumPy-friendly Python API and first-class N-dimensional COO tensors.

Compared with existing options:

- **SciPy.sparse**
  - Two-dimensional, linear-algebra–first. Mature and battle-tested.
  - Not natively N-dimensional; integrating array/tensor semantics (broadcasting, axis-wise ops) often requires reshaping or densifying.
- **PyData/Sparse (`sparse`)**
  - Modern, N-dimensional `COO` with NumPy-like semantics and broadcasting.
  - Performance is often below SciPy for core kernels; API coverage is a subset of NumPy; ecosystem is smaller.

What Lacuna offers today:

- **N-D and 2-D in one library**
  - COOND tensors plus CSR/CSC/COO matrices, with unfolding from ND to 2D (mode and grouped axes).
- **High-performance native kernels**
  - Pure Rust with Rayon parallelism and wide SIMD (f64x4), built with `-C target-cpu=native`.
  - SpMV/SpMM, reductions, and transforms use nnz-aware work partitioning, per-thread accumulators, and stripe-local buffers to improve cache locality and reduce contention.
- **Practical 2D operations**
  - CSR/CSC/COO: SpMV, SpMM, add/sub/Hadamard, transpose, prune(eps), eliminate_zeros, row/col sums, total sum.
- **ND operations that feel like NumPy**
  - COOND: sum/mean, reduce over axes, permute axes, reshape, broadcasting Hadamard, unfold to CSR/CSC.
- **Simple Python ergonomics**
  - NumPy-friendly classes, zero-copy buffer reads when safe, explicit float64 values and int64 indices, thread control via `set_num_threads`.

## Benchmarks

- Linear solve (CG)

  ![Linear solve (CG)](python/benchmarks/plots/solve_times.png)
- Block power iteration

  ![Block power iteration (SpMM)](python/benchmarks/plots/spmm_power_times.png)
- Block PageRank

  ![Block PageRank](python/benchmarks/plots/pagerank_block.png)

## Contributing

We welcome contributors! If you're interested in sparse matrix, feel free to open an issue or pull request. The project is under active development, so expect rapid iteration and occasional breaking changes.
