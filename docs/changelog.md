# Changelog

This project adheres to Keep a Changelog and Semantic Versioning.

## v0.2.0

- Added

  - CSC/COO Python classes and bindings parity with CSR (`Csc64`, `Coo64`).
  - Format conversions exposed via Python: CSR<->CSC, CSR<->COO, CSC<->COO.
  - ND COO baseline:
    - Kernels in `lacuna-kernels`: `sum_coond_f64`, `permute_axes_coond_f64_i64`, `reduce_sum_axes_coond_f64_i64`, `spmv_coond_f64_i64`, `spmm_coond_f64_i64`.
    - ND→2D conversions: `coond_mode_to_{csr,csc}_f64_i64`, `coond_axes_to_{csr,csc}_f64_i64`.
  - PyO3 exports for ND wrappers in `lacuna-py` (`coond_*_from_parts`).
  - High-level Python `COOND` class in `lacuna.sparse`; added `python/tests/test_nd.py`.
  - ND COO new ops:
    - Mean: `mean_coond_f64`; Axis-wise mean: `reduce_mean_axes_coond_f64_i64`.
    - Reshape: `reshape_coond_f64_i64`.
    - Broadcasted Hadamard product: `hadamard_broadcast_coond_f64_i64`.
  - PyO3 exports for new ND ops:
    - `coond_mean_from_parts`, `coond_reduce_mean_axes_from_parts`, `coond_reshape_from_parts`, `coond_hadamard_broadcast_from_parts`.
  - Python `COOND` methods: `mean`, `reduce_mean_axes`, `reshape`, `hadamard_broadcast`.
  - Python tests: added coverage for the above ND ops in `python/tests/test_nd.py`.
- Changed

  - Split monolithic `lacuna-py/src/lib.rs` into modules: `csr.rs`, `csc.rs`, `coo.rs`, `functions.rs`.
    - No Python API changes; `lib.rs` now only aggregates and registers symbols.
  - Centralized kernel utilities in `lacuna-kernels/src/util.rs` (constants, helpers, `UsizeF64Map`).
  - Benchmarks now import `lacuna` only from the installed environment (removed local `sys.path` injection).
- Performance

  - Replaced `HashMap`-based sparse accumulators with a custom linear-probing `UsizeF64Map` in `reduce.rs` and `spmv.rs`.
  - Parallelized small-dimension reduction paths; SIMD-accelerated stripe merge in column-sum kernels.
  - Parallel linearization in `coond_axes_to_coo_f64_i64` using slice access + unchecked indexing to reduce overhead.
  - Parallel index normalization/accumulation in ND broadcasted Hadamard and parallel linearize/delinearize in ND reshape.
- Fixed

  - Balanced unmatched braces in `reduce.rs` causing "unclosed delimiter" compile errors; added missing closers.
  - Fixed Rayon `Sync` error in `convert.rs` by avoiding capturing raw pointers in parallel closures (use slice + `get_unchecked`).
  - Fixed `i64_to_usize` missing brace in `arith.rs`.
  - Resolved `E0596` borrow errors in parallel closures in `arith.rs` and `transform.rs` by using raw pointer writes for disjoint parallel writes.
  - Removed stray closing brace in `arith.rs`.
  - Corrected `product_checked` to return the accumulator and moved `mean_coond_f64`/`reduce_mean_axes_coond_f64_i64` to top-level scope in `reduce.rs`.

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
