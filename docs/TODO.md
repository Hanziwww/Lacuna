# Array API TODO Roadmap (Sparse-First)

This document enumerates the remaining Array API functions to implement in Lacuna, organized by namespace, and provides detailed development paths for each category. The policy remains: sparse-first dispatch, no implicit densification, explicit capability declarations, and comprehensive tests.

## 1) Not-yet-implemented by namespace (to be added)

### 1.1 linalg

- [X] **tensordot**(x, y, *, axes=...)
- [X] **vecdot**(x, y, *, axis=-1)

Notes: Sparse@sparse remains intentionally unimplemented for matmul.

### 1.2 reductions (beyond sum/mean/count_nonzero)

- [X] **prod**(x, axis=None, keepdims=False)
- [X] **min**(x, axis=None, keepdims=False)
- [X] **max**(x, axis=None, keepdims=False)
- [X] **var**(x, axis=None, correction=0, keepdims=False)
- [X] **std**(x, axis=None, correction=0, keepdims=False)
- [X] **cumulative_prod**(x, axis=None)
- [X] **cumulative_sum**(x, axis=None)
- [X] **all**(x, axis=None, keepdims=False)
- [X] **any**(x, axis=None, keepdims=False)
- [X] **diff**(x, *, n=1, axis=-1)  (sparse-first variant)

Target first for CSR/CSC 2D; COOND support where feasible.

### 1.3 elementwise (arithmetic/compare/logical/unary/math)

- Arithmetic:
  - [X] **divide**(x1: array | int | float | complex, x2: array | int | float | complex, /)
  - [X] **floor_divide**(x1: array | int | float, x2: array | int | float, /)
  - [ ] **remainder**(x1: array | int | float, x2: array | int | float, /)
  - [ ] **pow**(x1: array | int | float | complex, x2: array | int | float | complex, /)
  - [ ] **negative**(x: array, /)
  - [ ] **abs**(x: array, /)
  - [ ] **sign**(x: array, /)
- Compare:
  - [ ] **equal**(x1: array | int | float | complex | bool, x2: array | int | float | complex | bool, /)
  - [ ] **not_equal**(x1: array | int | float | complex | bool, x2: array | int | float | complex | bool, /)
  - [ ] **less**(x1: array | int | float, x2: array | int | float, /)
  - [ ] **less_equal**(x1: array | int | float, x2: array | int | float, /)
  - [ ] **greater**(x1: array | int | float, x2: array | int | float, /)
  - [ ] **greater_equal**(x1: array | int | float, x2: array | int | float, /)
- Logical (boolean COOND/CSR upon availability):
  - [ ] logical_and, logical_or, logical_xor, logical_not
- Math (real):
  - [ ] exp, log, log1p, sqrt
  - [ ] sin, cos, tan, asin, acos, atan
  - [ ] sinh, cosh, tanh, asinh, acosh, atanh
  - [ ] clip, maximum, minimum
  - [ ] isnan, isinf, isfinite (when dtype support expands)

Notes:

- COOND add/sub kernels exist; currently not routed via array_api for COOND. Expose through dispatch.
- maximum/minimum on sparse require union-of-support semantics; see dev path below.

### 1.4 manipulation (beyond COOND permute_dims/reshape)

- [ ] **squeeze**(x, axis=None)
- [ ] **expand_dims**(x, axis)
- [ ] **moveaxis**(x, source, destination)
- [ ] **stack**(arrays, axis=0)
- [ ] **concat**(arrays, axis=0)
- [ ] **unstack**(x, axis=0)
- [ ] **broadcast_to**(x, shape)
- [ ] **broadcast_arrays**(*arrays)
- [ ] **flip**(x, axis=None)
- [ ] **roll**(x, shift, axis=None)
- [ ] **repeat**(x, repeats, axis=None)
- [ ] **tile**(x, reps)

COOND-first where applicable; 2D formats as feasible with guarantees to not densify.

### 1.5 searching / indexing / sets / sorting (minimum viable)

- [ ] **nonzero**(x)
- [ ] **where**(condition, x, y)
- [ ] **argmax**(x, axis=None, keepdims=False)
- [ ] **argmin**(x, axis=None, keepdims=False)
- [ ] **take**(x, indices, *, axis=None)
- [ ] **take_along_axis**(x, indices, axis)
- [ ] **searchsorted**(x, v, *, side="left")
- [ ] **argsort**(x, axis=-1)
- [ ] **sort**(x, axis=-1)
- [ ] **unique**(x)
- [ ] **unique_all**(x)
- [ ] **unique_counts**(x)
- [ ] **unique_inverse**(x)

Provide sparse-first versions or explicit NotImplemented with no densify.

### 1.6 dtypes / devices

- [ ] astype for COO/CSC/COOND and for additional dtype targets
- [ ] can_cast/isdtype/finfo/iinfo/result_type — currently routed to NumPy; keep unless moving to Rust
- [ ] device abstraction exposure (if/when relevant)

## 2) Development Path (per feature)

Use this repeatable pipeline for each function or group:

1) Kernel implementation (Rust: `crates/lacuna-kernels`)

- Choose module: `linalg`, `statistical`, `elementwise`, `manipulation`, `search_sort`, `setops`, `indexing`.
- Implement kernel over target sparse formats (CSR/CSC/COO/COOND). Prefer format-native algorithms (e.g., CSR row ops, CSC col ops, COOND axis-aware iteration).
- Add unit tests in Rust (where helpful) and re-export symbols in `crates/lacuna-kernels/src/lib.rs`.
- Performance: leverage iterators over compressed structures; avoid allocating dense temporaries.

2) Bindings (Rust→Python: `crates/lacuna-py`)

- Create/extend appropriate `src/array_api/*.rs` module.
- Expose `#[pyfunction]` wrappers with shape/dtype/axis validations mirroring Array API.
- Map from Python sparse objects to Rust types; uphold error messages consistent with Python layer.

3) Python dispatch wiring (`python/lacuna/array_api`)

- Add routing in `_dispatch.py` with:
  - Sparse-first detection (`_is_sparse`) and axis normalization helpers (`_normalize_axes_2d`, `_normalize_axes_nd`).
  - Keepdims reinsertion for COOND (`_coond_with_keepdims`).
  - Dense guard: raise `NotImplementedError` when only sparse inputs are provided but kernel not available; never densify.
- Add thin wrappers in module files (e.g., `reductions.py`, `linalg.py`, `elementwise/arithmetic.py`, `manipulation.py`, `searching.py`) that call `_dispatch`.

4) Capabilities declaration

- Update `__array_namespace_info__().capabilities` to advertise newly available ops under their namespaces.

5) Testing (Python: `python/tests/array_api`)

- Positive cases: correctness against NumPy semantics where defined (sparse×dense comparisons), shape and keepdims expectations, broadcasting and batch matmul behavior.
- Negative cases: axis errors (out-of-range, duplicates, empty `axis=()` if unsupported), dtype mismatches, forbidden densify paths.
- Coverage across CSR/CSC/COO; COOND for ND behaviors and broadcasting.

6) Documentation

- Update this TODO and `docs/api/*.md` with coverage matrices and examples.
- Mention limitations (e.g., sparse@sparse matmul not yet supported; dtype limited to float64).

7) Benchmarks (optional)

- Add micro-benchmarks under `python/benchmarks` for representative shapes and sparsity.

## 3) Category-specific design notes

### 3.1 linalg: tensordot / vecdot

- vecdot: for 1D sparse×sparse, compute overlap of indices (COO/COOND 1D); for 2D, permit axis selection mapping to row/col vector dot, or reduce to CSR/CSC slices. Start with sparse×dense and COOND×dense.
- tensordot: for sparse×dense, implement contraction by batching along remaining axes; for COOND×dense, index-match along contracted axes using coalesced coords; add axis normalization and shape inference; maintain batch broadcasting rules.

### 3.2 reductions: min/max/prod/var/std/cum*

- min/max: union-of-support consideration — treat missing entries as zeros; for max/min with zeros, ensure semantics match dense reference (e.g., max assumes zeros where not explicitly stored). Implement per-axis via CSR row scans / CSC col scans; COOND via grouped reduce over coordinates.
- prod: be careful with implicit zeros → product becomes 0 unless axis has all stored and implied ones; document semantics.
- var/std: reuse sum and sum of squares; correction handling; ensure numeric stability.
- cumsum/cumprod: CSR row-wise prefix; CSC column-wise prefix; COOND along axis by sorting/grouping by that axis.
- all/any: logical view of non-zeros; support boolean sparse once dtype available.

### 3.3 elementwise: divide/remainder/pow/maximum/minimum/logic/compare

- scalar variants first; then sparse×sparse where semantics are pointwise on intersection/union as appropriate:
  - hadamard-like ops (e.g., divide) typically intersection of support (zeros outside stay zeros).
  - max/min require union; build efficient merge of two sorted index streams per row/col (CSR/CSC) or via hash map for COORD/COOND small stripes.
- comparisons produce boolean sparse; until boolean dtype is supported, return NotImplemented or float64 with 0/1 documented.

### 3.4 manipulation: broadcasting and stacking

- broadcast_to/broadcast_arrays: COOND-first by repeating strides virtually (no data copy) and emitting broadcasted indices lazily or via computed tiles; ensure memory safety and complexity bounds.
- stack/concat/unstack: operate on parts without densify; for CSR/CSC, update indptr/indices blocks; for COOND, offset coordinates.
- moveaxis/squeeze/expand_dims: COOND is primary; 2D formats expose limited and well-defined transformations only.

### 3.5 searching/indexing/sets/sorting

- nonzero: return sparse-native nonzero indices; for COOND return all coordinate tuples; for CSR/CSC build from structure.
- where: when `condition` is sparse, implement masked selection against `x/y`; start with dense `x/y` + sparse mask.
- take/take_along_axis: slicing along a given axis using gather; CSR row gather and CSC col gather primitives; COOND general index remap.
- argsort/sort: support along an axis within each row/col; COOND per-slice sort.
- unique*: operate on data or along axis; leverage hash-based de-dup on coordinates/data as applicable.

## 4) Prioritization

- High: linalg.tensordot/vecdot; reductions {min,max,prod,var,std}; elementwise {divide, maximum/minimum}; searching {nonzero}; manipulation {stack, concat, broadcast_to}.
- Medium: cumsum/cumprod/all/any/diff; comparisons; logical ops; moveaxis/squeeze/expand_dims.
- Low: sorting and full sets; unique* variants; advanced indexing.

## 5) Definition of Done (per op)

- Rust kernel implemented and re-exported in `lacuna-kernels`.
- Bound in `lacuna-py` under the Array API submodule with ergonomic signatures and errors.
- Routed through Python `_dispatch` with sparse-first guard; module wrapper provided.
- Capability advertised in `__array_namespace_info__`.
- Tests cover CSR/CSC/COO and COOND where relevant; include axis/keepdims/broadcast and error branches.
- Docs updated here and in API/tutorials.

## 6) Code map (for contributors)

- Kernels: `crates/lacuna-kernels/src/{linalg,statistical,elementwise,manipulation,search_sort,setops}`
- Rust bindings: `crates/lacuna-py/src/array_api/{linalg.rs,reduce.rs,elementwise.rs,manipulation.rs,...}`
- Python dispatch: `python/lacuna/array_api/_dispatch.py`
- Python API modules: `python/lacuna/array_api/{linalg.py,reductions.py,elementwise/,manipulation.py,searching.py,dtypes.py,_namespace.py}`
- Tests: `python/tests/array_api` (unit/feature tests)
