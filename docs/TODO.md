# TODO

Project-wide checklist derived from PLAN.md. Items are grouped by milestones; all are initially unchecked.

## Implementation Milestones

- **M0: Scaffolding**

  - [X] Repository layout: crates, Python package skeleton, CI with maturin
  - [X] Basic CSR struct in Rust and initial PyO3 binding stub
  - [X] Docs site scaffold (Sphinx + MyST)
- **M1: CSR Core**

  - [X] Kernels: SpMV, SpMM, reductions (sum, row/col sums), transpose
  - [X] Indexing/slicing: rows/cols (read-only)
  - [X] Cleanup ops: prune(eps), eliminate_zeros
  - [X] Python OOP façade; NumPy interop; release GIL for compute
  - [X] Parallel + SIMD implementations (Rayon + std::simd) for all v0.1 kernels
  - [X] Implement and optimize the basic arithmetic kernels
- **M2: Conversions and Formats**

  - [X] Public COO and CSC types
  - [X] Conversions: CSR <-> COO <-> CSC
  - [X] Arithmetic: A + B, Hadamard A.multiply(B)
  - [ ] IO: Matrix Market (.mtx) and NPZ save/load
  - [ ] Dtype/index casting
- **M3: Performance and Stability**

  - [ ] Blocked/tiling strategies; cache/NUMA tuning
  - [ ] Benchmark suite and performance regression gates
  - [ ] API polish and error taxonomy
  - [ ] Documentation/tutorials pass
- **M4: Advanced**

  - [ ] BSR format
  - [ ] Iterative solvers (CG/GMRES) and basic preconditioners
  - [ ] Plug-in kernel strategy
  - [ ] Optional f32 kernels and dtype growth

## Cross-cutting

- **Formats & Data Model**

  - [X] CSR default: values f64, indices i64; plan for f32/i32 as feature flags
  - [ ] ND baseline: COO-ND representation and invariants (v0.2)
    - [X] COO-ND storage and invariants
    - [X] Axis reductions: sum over axes
    - [X] Axis permutation
    - [X] ND→2D conversions: mode/axes unfold to CSR/CSC
    - [X] Broadcasting elementwise ops (Hadamard)
    - [X] mean and reshape
  - [ ] Future CSF for ND advanced ops (v0.4+)
- **Python API**

  - [X] `lacuna.sparse` classes: SparseArray/SparseMatrix, CSR/CSC/COO/COOND
  - [ ] Construction and conversion APIs; SciPy/NumPy bridges
  - [X] Ops surface: matmul, add, multiply, transpose, sum; slicing semantics
- **Rust Design**

  - [ ] Core traits: SparseFormat, SparseND, and op traits (SpMV/SpMM/Add/MulElem/Transpose/...)
  - [ ] Deterministic reductions where required; careful `unsafe` only in hot paths

## Kernel Implementation Checklist (by crate)

- **crates/lacuna-kernels** (Rust optimized kernels, f64/i64, parallel+SIMD)

  - SpMV
    - [X] Feature done
    - [X] Tests done
  - SpMM
    - [X] Feature done
    - [X] Tests done
  - Reductions: sum, row_sums, col_sums
    - [X] Feature done
    - [X] Tests done
  - Transpose (CSR -> CSR)
    - [X] Feature done
    - [X] Tests done
  - Cleanup: eliminate_zeros, prune(eps)
    - [X] Feature done
    - [X] Tests done
  - Arithmetic: add_csr (A+B), mul_scalar (alpha*A)
    - [X] Feature done
    - [X] Tests done
  - Utilities / Refactors
    - [X] Centralize reusable kernel utilities in `util.rs` (constants, helpers, `UsizeF64Map`)
    - [X] Replace HashMap-based sparse accumulators with `UsizeF64Map` in `reduce.rs` and `spmv.rs`
    - [X] Improve reduction paths: parallel small-dimension branches; SIMD stripe merge for column sums
  - ND COO
    - [X] Sum / Permute axes / Reduce sum over axes (COO-ND kernels)
    - [X] SpMV/SpMM along mode axis
    - [X] ND→2D conversions (mode/axes unfold) in `convert.rs`
- **crates/lacuna-py** (PyO3 bindings)

  - SpMV / SpMM
    - [X] Feature done (`Csr64.spmv/spmm` and `*_from_parts`)
    - [X] Tests done (indirectly covered via Python tests)
  - Reductions / Transpose / Cleanup / Arithmetic
    - [X] Feature done (`sum/row_sums/col_sums`, `transpose`, `prune/eliminate_zeros`, `add/mul_scalar` bindings)
    - [X] Tests done (indirectly covered via Python tests)
  - Bindings structure
    - [X] Split monolithic `src/lib.rs` into modules: `csr.rs`, `csc.rs`, `coo.rs`, `functions.rs`; keep `lib.rs` as aggregator (no Python API changes)
  - ND bindings
    - [X] Export ND wrappers: `coond_sum_from_parts`, `coond_mean_from_parts`, `coond_reduce_sum_axes_from_parts`, `coond_reduce_mean_axes_from_parts`, `coond_permute_axes_from_parts`, `coond_reshape_from_parts`, `coond_hadamard_broadcast_from_parts`, `coond_mode_to_{csr,csc}_from_parts`, `coond_axes_to_{csr,csc}_from_parts`
    - [X] Registered in `lib.rs`
- **python/lacuna** (High-level Python API: CSR facade)

  - SpMV
    - [X] Feature done (`__matmul__` 1D)
    - [X] Tests done (`python/tests/test_ops.py`)
    - [X] Benchmarks done (`python/benchmarks/benchmark_spmv.py`)
    - [X] Docs done
  - SpMM
    - [X] Feature done (`__matmul__` 2D)
    - [X] Tests done (`python/tests/test_ops.py`)
    - [X] Benchmarks done (`python/benchmarks/benchmark_spmm.py`)
    - [X] Docs done
  - Reductions: sum, row_sums, col_sums
    - [X] Feature done (`CSR.sum` supports `None/0/1`)
    - [X] Tests done (`python/tests/test_ops.py`)
    - [X] Benchmarks done
    - [X] Docs done
  - Transpose
    - [X] Feature done (`CSR.T`)
    - [X] Tests done (`python/tests/test_ops.py`/`test_more_ops.py`)
    - [X] Benchmarks done
    - [X] Docs done
  - Cleanup: prune, eliminate_zeros
    - [X] Feature done
    - [X] Tests done (`python/tests/test_ops.py`)
    - [X] Benchmarks done
    - [X] Docs done
  - Arithmetic: add, mul_scalar, sub, hadamard
    - [X] Feature done (`__add__`, `__mul__/__rmul__`)
    - [X] Tests done (`python/tests/test_ops.py`/`test_more_ops.py`)
    - [X] Benchmarks done
    - [X] Docs done
 - **python/lacuna** (High-level Python API: ND COO facade)
  
  - COOND
    - [X] Feature done (`sum`, `mean`, `reduce_sum_axes`, `reduce_mean_axes`, `permute_axes`, `reshape`, `hadamard_broadcast`, `mode_unfold_to_{csr,csc}`, `axes_unfold_to_{csr,csc}`)
    - [X] Tests done (`python/tests/test_nd.py`)
- **Planned (per PLAN.md milestones)**

  - [X] Arithmetic: subtraction (A - B)
  - [X] Arithmetic: Hadamard elementwise multiply `A.multiply(B)`
  - [X] Format conversions: CSR <-> CSC, CSR <-> COO
  - [X] ND baseline (COO-ND): elementwise ops with broadcasting (Hadamard), `sum/mean` over axes, `transpose/permute`, `reshape`
  - [ ] Reordering: CSR reorder (cache locality)
  - [ ] Cache-aware/blocked SpMM improvements
  - [ ] Block formats: BSR kernels
  - [ ] Dtype/index variants: f32 values, i32 indices (feature-gated)
  - [ ] ND advanced (CSF): kernels for `tensordot`/mode-n product; masked ops
- **Packaging & CI**

  - [ ] Build wheels with maturin for Win/macOS/Linux and Python 3.10–3.13
  - [ ] GitHub Actions matrix for wheels + sdist
  - [X] Versioning (SemVer) and licensing check (Apache-2.0)
- **Testing & Benchmarks**

  - [X] Rust unit/property tests
  - [X] Python pytest parity tests vs NumPy/SciPy; randomized matrices
  - [X] pytest-benchmark scenarios; SuiteSparse/synthetic datasets
  - [X] Benchmarks import `lacuna` only from installed environment (no local path injection)
- **Documentation**

  - [ ] User guides (MyST), API docs (autodoc/napoleon), and design notes
  - [ ] Tutorials: build CSR from COO; SpMV at scale; convert to SciPy
