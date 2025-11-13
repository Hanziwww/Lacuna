# Lacuna: Architecture and Implementation Plan

## Vision and Goals

- **Purpose**: A high-performance, extensible Python library for sparse matrix operations with a Rust core.
- **Design**: Object-oriented Python API, format-agnostic operations, pluggable formats/kernels over time.
- **Performance**: Competitive with or faster than SciPy for core ops; predictable memory and thread scalability.
- **Dimensionality**: Support 2D and higher (N-dimensional sparse arrays/tensors) with a consistent API.
- **Compatibility**: Python 3.10–3.13. Windows, Linux (manylinux), macOS (x86_64/arm64). NumPy interop first-class.

## Non-Goals (Initial)

- **GPU** in v0.x (prepare backend abstraction for future CUDA/Metal).
- **Distributed** operations (out of scope for v0.x).
- **All formats at once** (start with CSR; add CSC/COO later).
- **Full linear algebra suite** (factorizations/solvers later milestones).

## Core Principles

- **Clean separation**: Rust core without Python deps; thin PyO3 bindings; Pythonic OOP API on top.
- **Extensibility**: New formats/kernels as independent modules behind traits and registries.
- **Interoperability**: Seamless conversion to/from SciPy `scipy.sparse` and NumPy arrays.
- **Safety and speed**: Safe Rust by default; `unsafe` only in hot kernels with tests/benchmarks.

## High-Level Architecture

- **lacuna-core (Rust crate)**: Data structures (formats), traits, core kernels, no Python.
- **lacuna-kernels (Rust crate)**: Optimized algorithms, parallelism, dtype specializations, feature-gated.
- **lacuna-io (Rust crate)**: MatrixMarket/NPZ readers/writers (optional).
- **lacuna-py (Rust crate + PyO3)**: Python bindings exposing `PyClass` wrappers; releases GIL during compute.
- **python/lacun (Python package)**: OOP façade, format registry, dtype/index dispatch, NumPy/SciPy bridges.

## Data Model and Formats

- **Initial dtype/index**: values `float64` (default), indices `int64`. Later enable `float32`/`int32`.
- **CSR (v0.1)**:
  - `indptr: Vec<Idx>` length `n_rows + 1`
  - `indices: Vec<Idx>` length `nnz`
  - `data: Vec<T>` length `nnz`
  - Invariants: row-major; column indices sorted within each row; duplicates aggregated.
- **ND arrays (2D+)**:
  - v0.2 baseline: COO-ND representation with `indices` of shape `(nnz, ndim)` (or per-axis index arrays), `data` length `nnz`, and `shape: Vec<Idx>` length `ndim`.
  - Invariants: indices within bounds; duplicates aggregated by default; optional internal lexicographic sort for canonical form.
  - v0.4+: CSF (Compressed Sparse Fiber) for high-performance ND operations (reductions, contractions).
- **Next formats**:
  - v0.2: COO (construction, IO) and CSC (fast SpMV^T).
  - v0.4+: BSR, DIA, ELL, DOK, LIL, triangular/symmetric flags.
- **Construction**:
  - From triples `(i, j, v)` with duplicate policy: `sum|last|error` (default: `sum`).
  - Input indices may be unsorted; sorted internally per row to maintain invariants.
  - Out-of-range indices: error.
  - From SciPy/NumPy buffers without copying when safe (read-only views to `data/indices/indptr`).
  - ND: from tuple of index arrays `(i0, i1, ..., ik)` and `v`, or a 2D indices array `(nnz, ndim)` plus `v` and `shape`.

## Operations (by Milestone)

- **v0.1 (CSR basics)**:
  - Shape/nnz/meta, transpose, dtype/index casting.
  - Reductions: sum; row/col sums.
  - Element access: `A[i, j]` read; row/column slicing.
  - Arithmetic: `A @ x` (SpMV), `A @ B_dense` (SpMM), `A + B`, scalar ops.
  - Cleanup: `eliminate_zeros`, `prune(eps)`.
- **v0.2 (COO/CSC)**:
  - Public COO/CSC types; conversions among CSR/CSC/COO.
  - Faster transpose via CSC; improved construction pipelines.
  - Arithmetic: `A - B`, Hadamard `A.multiply(B)`.
  - ND baseline (COO-ND): elementwise ops with NumPy-style broadcasting, `sum/mean` reductions over specified axes, `transpose/permute` axes, `reshape` (no copy when shapes compatible).
- **v0.3 (Dense interop and IO)**:
  - `toarray()`, `astype()`.
  - IO: Matrix Market `.mtx`, compressed `.npz`.
- **v0.4+ (Advanced)**:
  - Reordering (CSR reorder), block formats (BSR), cache-aware SpMM.
  - Linalg: iterative solvers (CG/GMRES), basic preconditioners. Factorizations later.
  - ND advanced: CSF kernels; `tensordot`/mode-n product; masked operations.

## API Design (Python)

- **Module layout**:
  - `lacun.sparse`: `CSR`, `CSC`, `COO`, `Format` registry.
  - `lacun.linalg`: iterative solvers (future).
  - `lacun.io`: `mmread/mmwrite`, `load_npz/save_npz`.
- **Key classes/protocols**:
  - `class SparseArray`: abstract ND base with `shape`, `ndim`, `nnz`, `dtype`; exposes storage views when applicable.
  - `class SparseMatrix(SparseArray)`: 2D specialization.
  - `class CSR(SparseMatrix)`, `CSC`.
  - `class COO(SparseArray)`: ND (2D+) COO; also used internally for conversions.
- **Core methods**:
  - Construction: `CSR.from_coo(i, j, v, shape, duplicate='sum')`, `from_scipy(x)`, `from_numpy(x, ...)`.
  - Conversion: `toformat('csr'|'csc'|'coo'|'csf')`, `to_scipy()`, `toarray()`, `astype(dtype)`.
  - Ops: `A @ x`, `A @ B`, `A + B`, `A.multiply(B)` (Hadamard), `A.T`/`A.transpose(axes=None)`, `A.sum(axis=...)`.
  - Slicing: ND tuple indexing and slicing `A[i0, i1, ..., ik]`; row/col convenience for 2D; advanced gather. Read-only in v0.1; mutation via dedicated APIs in v0.3+.
- **NumPy interop**:
  - Implement `__array_priority__`, `__array_function__` (NEP-18) and `__array_ufunc__` where sensible.
  - Return NumPy arrays for dense results; accept array-likes; zero-copy where safe.

## Rust Design

- **Traits**:
  - `SparseFormat<T, I>`: 2D formats; shape/meta, iteration, index invariants.
  - `SparseND<T, I>`: ND formats (COO-ND/CSF) with dimension-aware iteration and axis-wise operations.
  - `SpMV`, `SpMM`, `Add`, `MulElem`, `Transpose`, `ConvertTo`, `ReduceAxes`, `PermuteAxes`, `Reshape`.
- **Type strategy**:
  - Start with `f64/i64`; design for generic `T: Num + Zero` and `I: Index` using `num-traits`.
  - Use enum dispatch for runtime dtype in Python layer; macros generate monomorphized kernels.
- **Performance**:
  - Parallel kernels with `rayon` enabled by default; automatically detect available hardware threads; cache-friendly loop ordering.
  - GIL released around kernels via `Python::allow_threads`.
  - Native SIMD via `std::simd` (stable) enabled by default.
- **Error handling**:
  - `thiserror` in Rust; map to `LacunaError` (Python Exception).
  - Strict invariants checked at boundaries; debug assertions inside kernels.

## Extensibility Strategy

- **Format registry (Python)**:
  - Map format name -> Python class and Rust type ID.
  - Uniform conversion interface; new formats add minimal glue.
- **Kernel registration (Rust)**:
  - Trait-based kernels; new formats implement traits to gain operations.
  - Feature flags enable optional formats and dtypes.
- **Dtype/index growth**:
  - v0.1: `f64/i64`.
  - v0.2: `f32` opt-in.
  - v0.3: `int32` indices for compact matrices.

## Packaging and Distribution

- **Build**: `maturin` with `pyproject.toml`, `pyo3`/`numpy` crates.
- **CI**: GitHub Actions to build wheels for:
  - Windows (MSVC), macOS (x86_64, arm64 universal2), Linux manylinux.
  - Python 3.10–3.13 wheels + sdist.
- **Runtime deps**: Python `numpy>=1.22`. SciPy optional for convenience converters.
- **Versioning**: SemVer starting at v0.1. ABI-stable Python API, careful Rust refactors under the hood.
- **Distribution (PyPI)**: `lucuna-sparse`; import package: `lacun`.
- **License**: Apache-2.0.

## Testing and Benchmarking

- **Rust**: unit tests per crate; property tests with `proptest` for conversions/ops.
- **Python**: `pytest` with NumPy/SciPy parity tests; randomized matrices.
- **Benchmarks**:
  - Rust: `criterion` microbench (SpMV/SpMM/add/convert).
  - Python: `pytest-benchmark`; datasets: SuiteSparse small/medium, synthetic RNG.
- **Performance gates**: PRs must not regress more than a threshold.

## Documentation

- **User docs**: Sphinx (with MyST Markdown, napoleon, autodoc). Tutorials: “build CSR from COO”, “SpMV at scale”, “convert to SciPy”.
- **API docs**: Auto-generated from Python docstrings and Rust docs where appropriate.
- **Design docs**: Format invariants, kernel notes, contribution guide.

## Repository Structure

```
lacuna/
  Cargo.toml
  pyproject.toml
  README.md
  LICENSE
  CHANGELOG.md
  .gitignore
  rust-toolchain.toml
  rustfmt.toml

  crates/
    lacuna-core/
      src/
      benches/
      tests/
      Cargo.toml
    lacuna-kernels/
      src/
      benches/
      tests/
      Cargo.toml
    lacuna-io/
      src/
      tests/
      Cargo.toml
    lacuna-py/
      src/lib.rs
      Cargo.toml

  python/
    lacun/
      __init__.py
      _runtime.py
      _types.pyi
      sparse/
        __init__.py
        base.py
        csr.py
        csc.py
        coo.py
      io/
        __init__.py
        mtx.py
        npz.py
    tests/
      test_csr.py
      test_convert.py
    benchmarks/
      benchmark_spmv.py
      benchmark_spmm.py

  docs/
    conf.py
    index.md
    api/
    guides/
    tutorials/

  scripts/
    build_wheels.ps1
    build_wheels.sh
    bench_matrix_download.py
    release_notes.py

  data/
    samples/
      tiny_csr.npz
      tiny_coo.mtx
```

## Implementation Milestones and Timeline

- **M0: Scaffolding**
  - Repo + crates, Python package skeleton, CI with `maturin`.
  - Basic `CSR` struct and PyO3 binding stub; docs site scaffold.
- **M1: CSR Core**
  - CSR invariants, constructors (from COO, from buffers).
  - SpMV/SpMM, reductions, transpose, slicing (rows/cols), prune/eliminate zeros.
  - Python OOP façade; NumPy interop; release GIL.
- **M2: Conversions and Formats**
  - Public COO/CSC; CSR<->COO<->CSC conversions; arithmetic `A+B`, `A*B` (Hadamard).
  - Matrix Market IO; NPZ save/load; dtype/index casting.
- **M3: Performance and Stability**
  - Parallel/blocked kernels, SIMD, benchmark suite, regression gates.
  - API polish, error taxonomy, docs/tutorials.
- **M4: Advanced**
  - BSR; iterative solvers; preconditioners; plug-in kernel strategy; optional f32.

## Key Technical Choices

- **Bindings**: PyO3 + `numpy` crate; GIL released around kernels.
- **Kernels**: Fully in-house, pure Rust implementation; no `sprs` dependency. Self-developed numeric/coefficients kernels for sparse formats; may adopt algorithmic ideas without linking to external crates.
- **Parallelism**: `rayon` task-based; deterministic reductions where needed.
- **Memory**: Avoid intermediate allocations; reuse buffers; careful with aliasing/`unsafe`.
- **Interop**: Zero-copy read-only views for `data/indices/indptr` as a first-class feature; no Python-level mutable views in v0.1.
- **Indexing**: Default 64-bit; optional 32-bit for memory savings; conversions explicit.

## Risks and Mitigations

- **Generic dtype explosion**: Start with `f64` only; macro-driven expansion later.
- **API surface creep**: Gate by milestones; deprecate via wrappers, not breaking changes.
- **Wheel complexity**: Use `maturin-action` templates; test nightly and release guards.
- **Performance parity**: Benchmark early; adopt known algorithmic best practices (CSR-vector, cache-aware tiling).
