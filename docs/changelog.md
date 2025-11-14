# Changelog

This project adheres to Keep a Changelog and Semantic Versioning.

## v0.1.0

- Added

  - Core CSR sparse matrix type with `f64` values and `i64` indices.
  - Python bindings via PyO3; development builds via `maturin develop`.
  - Core operations (CSR basics):
    - Shape/nnz/meta, transpose, dtype/index casting.
    - Reductions: global sum, row/column sums.
    - Element access (`A[i, j]` read) and row/column slicing.
    - Arithmetic: `A @ x` (SpMV), `A @ B_dense` (SpMM), `A + B`, `A - B`, Hadamard `A.multiply(B)`, scalar ops.
    - Cleanup utilities: `eliminate_zeros`, `prune(eps)`.
  - High-performance kernels (rayon parallelism + `std::simd`) for `f64/i64` in `lacuna-kernels`.
  - Repository structure with `lacuna-core`, `lacuna-kernels`, `lacuna-io`, `lacuna-py`, and Python package `lacun`.
  - Tests: Rust unit tests and Python `pytest` suite; basic benchmark scripts.
- Packaging

  - Build and install with `maturin`; Python 3.10–3.13.
  - Windows, Linux (manylinux), and macOS targets in plan; wheel scripts provided.
- Documentation

  - Initial developer guide and project overview.

