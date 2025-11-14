import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# Ensure we can import lacuna from source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------- Builders ----------


def build_scipy_csr(
    m: int, n: int, density: float, seed: int, dtype: np.dtype
) -> Tuple[sp.csr_matrix, int]:
    rs = np.random.RandomState(seed)
    data_rvs = lambda s: rs.standard_normal(s).astype(dtype)
    A_coo = sp.random(m, n, density=density, format="coo", random_state=rs, data_rvs=data_rvs)
    A_csr = A_coo.tocsr()
    return A_csr, int(A_coo.nnz)


def build_scipy_csr_pair(
    m: int, n: int, density: float, seed: int, dtype: np.dtype
) -> Tuple[sp.csr_matrix, sp.csr_matrix, int, int]:
    A, nnzA = build_scipy_csr(m, n, density, seed, dtype)
    B, nnzB = build_scipy_csr(m, n, density, seed + 101, dtype)
    return A, B, nnzA, nnzB


def build_dense_vector(n: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rs = np.random.RandomState(seed)
    return rs.standard_normal(n).astype(dtype)


def build_dense_matrix(k: int, n: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rs = np.random.RandomState(seed)
    return rs.standard_normal((k, n)).astype(dtype)


def build_lacuna_csr_from_scipy(A_scipy: sp.csr_matrix):
    try:
        from lacuna.sparse.csr import CSR
    except Exception:
        return None
    csr = A_scipy.tocsr()
    data64 = csr.data.astype(np.float64, copy=False)
    try:
        return CSR(csr.indptr, csr.indices, data64, csr.shape, check=False)
    except Exception:
        return None


def build_pydata_sparse_from_scipy(A_scipy: sp.coo_matrix):
    try:
        import sparse as psparse  # pydata/sparse
    except Exception:
        return None
    try:
        A_coo = A_scipy if sp.isspmatrix_coo(A_scipy) else A_scipy.tocoo()
        return psparse.COO.from_scipy_sparse(A_coo)
    except Exception:
        try:
            A_coo = A_scipy if sp.isspmatrix_coo(A_scipy) else A_scipy.tocoo()
            coords = np.vstack([A_coo.row, A_coo.col])
            return psparse.COO(coords, A_coo.data, shape=A_coo.shape)
        except Exception:
            return None


# ---------- Timing helpers ----------


def time_op(fn: Callable[[], Any], warmup: int, repeat: int) -> List[float]:
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def summarize(name: str, times: List[float], ops: float) -> Optional[Dict[str, float]]:
    if not times:
        return None
    arr = np.array(times, dtype=np.float64)
    return {
        "name": name,
        "min_ms": float(arr.min() * 1e3),
        "median_ms": float(np.median(arr) * 1e3),
        "mean_ms": float(arr.mean() * 1e3),
        "gops": float((ops / arr.min()) / 1e9) if ops > 0 else 0.0,
    }


# ---------- Ops registry (per backend) ----------


class Backend:
    SCIPY = "scipy"
    PYDATA_SPARSE = "pydata.sparse"
    LACUNA = "lacuna"


def run_spmv_scipy(A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    return (A @ x).astype(np.float64, copy=False)


def run_spmv_sparse(A_sparse: Any, x: np.ndarray) -> Optional[np.ndarray]:
    try:
        y = A_sparse @ x
        return np.asarray(y, dtype=np.float64)
    except Exception:
        return None


def run_spmv_lacuna(A_lacuna: Any, x64: np.ndarray) -> np.ndarray:
    y = A_lacuna @ x64
    return np.asarray(y, dtype=np.float64)


def run_spmm_scipy(A: sp.csr_matrix, B: np.ndarray) -> np.ndarray:
    return (A @ B).astype(np.float64, copy=False)


def run_spmm_sparse(A_sparse: Any, B: np.ndarray) -> Optional[np.ndarray]:
    try:
        import sparse as psparse
    except Exception:
        return None
    try:
        out = A_sparse @ B
    except Exception:
        try:
            out = psparse.tensordot(A_sparse, B, axes=1)
        except Exception:
            return None
    return np.asarray(out, dtype=np.float64)


def run_spmm_lacuna(A_lacuna: Any, B64: np.ndarray) -> np.ndarray:
    C = A_lacuna @ B64
    return np.asarray(C, dtype=np.float64)


def run_sum_scipy(A: sp.csr_matrix) -> float:
    return float(A.sum())


def run_sum_axis_scipy(A: sp.csr_matrix, axis: int) -> np.ndarray:
    # SciPy returns matrix; convert to 1D array
    arr = np.asarray(A.sum(axis=axis)).ravel()
    return arr.astype(np.float64, copy=False)


def run_sum_lacuna(A_lacuna: Any, axis: Optional[int] = None) -> Any:
    return A_lacuna.sum(axis=axis)


def run_sum_sparse(A_sparse: Any, axis: Optional[int] = None) -> Optional[Any]:
    try:
        if axis is None:
            return float(A_sparse.sum())
        else:
            out = A_sparse.sum(axis=axis)
            return np.asarray(out).ravel()
    except Exception:
        return None


def run_transpose_scipy(A: sp.csr_matrix) -> sp.csr_matrix:
    return A.transpose().tocsr()


def run_transpose_lacuna(A_lacuna: Any) -> Any:
    return A_lacuna.T


def run_transpose_sparse(A_sparse: Any) -> Optional[Any]:
    try:
        return A_sparse.transpose()
    except Exception:
        try:
            return A_sparse.T
        except Exception:
            return None


def run_prune_scipy(A: sp.csr_matrix, eps: float) -> sp.csr_matrix:
    # Create pruned copy: zero out small values then eliminate zeros
    B = A.copy().tocsr()
    if B.data.size:
        mask = np.abs(B.data) > eps
        B.data = B.data * mask
    B.eliminate_zeros()
    return B


def run_prune_lacuna(A_lacuna: Any, eps: float) -> Any:
    return A_lacuna.prune(float(eps))


def run_prune_sparse(A_sparse: Any, eps: float) -> Optional[Any]:
    # Placeholder: attempt boolean mask; fallback to not implemented
    try:
        data = A_sparse.data
        coords = A_sparse.coords
        mask = np.abs(data) > eps
        if mask.ndim == 1:
            import sparse as psparse

            return psparse.COO(coords[:, mask], data[mask], shape=A_sparse.shape)
    except Exception:
        return None
    return None


def run_eliminate_scipy(A: sp.csr_matrix) -> sp.csr_matrix:
    B = A.copy().tocsr()
    B.eliminate_zeros()
    return B


def run_eliminate_lacuna(A_lacuna: Any) -> Any:
    return A_lacuna.eliminate_zeros()


def run_eliminate_sparse(A_sparse: Any) -> Optional[Any]:
    # Not directly available; emulate by pruning eps=0
    return run_prune_sparse(A_sparse, 0.0)


def run_add_scipy(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    return (A + B).tocsr()


def run_add_lacuna(A_lacuna: Any, B_lacuna: Any) -> Any:
    return A_lacuna + B_lacuna


def run_add_sparse(A_sparse: Any, B_sparse: Any) -> Optional[Any]:
    try:
        return A_sparse + B_sparse
    except Exception:
        return None


def run_mul_scipy(A: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    return (alpha * A).tocsr()


def run_mul_lacuna(A_lacuna: Any, alpha: float) -> Any:
    return alpha * A_lacuna


def run_mul_sparse(A_sparse: Any, alpha: float) -> Optional[Any]:
    try:
        return alpha * A_sparse
    except Exception:
        return None


def run_sub_scipy(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    return (A - B).tocsr()


def run_sub_lacuna(A_lacuna: Any, B_lacuna: Any) -> Optional[Any]:
    try:
        return A_lacuna - B_lacuna
    except Exception:
        return None


def run_sub_sparse(A_sparse: Any, B_sparse: Any) -> Optional[Any]:
    try:
        return A_sparse - B_sparse
    except Exception:
        return None


def run_hadamard_scipy(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    return A.multiply(B).tocsr()


def run_hadamard_lacuna(A_lacuna: Any, B_lacuna: Any) -> Optional[Any]:
    try:
        C = A_lacuna.multiply(B_lacuna)
        return C
    except Exception:
        return None


def run_hadamard_sparse(A_sparse: Any, B_sparse: Any) -> Optional[Any]:
    try:
        return A_sparse * B_sparse
    except Exception:
        return None


# ---------- Main ----------


def main():
    p = argparse.ArgumentParser(description="CSR benchmarks across backends for multiple kernels")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096, help="Inner dimension for SpMM")
    p.add_argument("--density", type=float, default=0.001)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_scipy", action="store_true")
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--no_lacuna", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument(
        "--ops",
        type=str,
        default="all",
        help=(
            "Comma-separated ops: "
            "spmv, spmm, sum, row_sums, col_sums, transpose, prune, eliminate, add, sub, mul, hadamard"
        ),
    )
    p.add_argument("--alpha", type=float, default=2.0, help="Scalar for mul")
    p.add_argument("--eps", type=float, default=1e-9, help="Threshold for prune")

    args = p.parse_args()
    dtype = np.float64 if args.dtype == "float64" else np.float32

    # Build baseline SciPy matrices
    A_scipy, nnz = build_scipy_csr(args.m, args.n, args.density, args.seed, dtype)
    A_scipy2, _ = build_scipy_csr(args.m, args.n, args.density, args.seed + 7, dtype)

    # Optional backends
    A_sparse = None if args.no_sparse else build_pydata_sparse_from_scipy(A_scipy)
    A_sparse2 = (
        None if (args.no_sparse or A_sparse is None) else build_pydata_sparse_from_scipy(A_scipy2)
    )
    A_lacuna = None if args.no_lacuna else build_lacuna_csr_from_scipy(A_scipy)
    B_lacuna = None if args.no_lacuna else build_lacuna_csr_from_scipy(A_scipy2)

    # Dense operands
    x = build_dense_vector(args.n, args.seed + 1, dtype)
    # For SpMM we need B shape (n, k) given A is (m, n)
    B = build_dense_matrix(args.n, args.k, args.seed + 2, dtype)

    # For lacuna, ensure float64 inputs
    x64 = x if x.dtype == np.float64 else x.astype(np.float64)
    B64 = B if B.dtype == np.float64 else B.astype(np.float64)

    # Ops selection
    wanted = {op.strip().lower() for op in (args.ops.split(",") if args.ops else [])}
    if "all" in wanted or not wanted:
        wanted = {
            "spmv",
            "spmm",
            "sum",
            "row_sums",
            "col_sums",
            "transpose",
            "prune",
            "eliminate",
            "add",
            "sub",
            "mul",
            "hadamard",
        }

    results: List[Dict[str, float]] = []

    # ---- SpMV ----
    if "spmv" in wanted:
        flops = 2.0 * nnz
        y_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_spmv_scipy(A_scipy, x), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":spmv", times, flops)
            if stats:
                results.append(stats)
            y_ref = run_spmv_scipy(A_scipy, x)
        if not args.no_sparse and A_sparse is not None:
            out = run_spmv_sparse(A_sparse, x)
            if out is not None:
                times = time_op(lambda: run_spmv_sparse(A_sparse, x), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":spmv", times, flops)
                if stats:
                    results.append(stats)
                if args.validate and y_ref is not None:
                    rtol = 1e-4 if dtype == np.float32 else 1e-7
                    atol = 1e-6 if dtype == np.float32 else 1e-9
                    if not np.allclose(out, y_ref, rtol=rtol, atol=atol):
                        raise AssertionError("Validation failed: pydata.sparse spmv vs scipy")
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_spmv_lacuna(A_lacuna, x64), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":spmv", times, flops)
            if stats:
                results.append(stats)
            if args.validate and y_ref is not None:
                out = run_spmv_lacuna(A_lacuna, x64)
                rtol = 1e-4 if dtype == np.float32 else 1e-7
                atol = 1e-6 if dtype == np.float32 else 1e-9
                if not np.allclose(out, y_ref, rtol=rtol, atol=atol):
                    raise AssertionError("Validation failed: lacuna spmv vs scipy")

    # ---- SpMM ----
    if "spmm" in wanted:
        flops = 2.0 * nnz * args.k
        C_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_spmm_scipy(A_scipy, B), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":spmm", times, flops)
            if stats:
                results.append(stats)
            C_ref = run_spmm_scipy(A_scipy, B)
        if not args.no_sparse and A_sparse is not None:
            out = run_spmm_sparse(A_sparse, B)
            if out is not None:
                times = time_op(lambda: run_spmm_sparse(A_sparse, B), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":spmm", times, flops)
                if stats:
                    results.append(stats)
                if args.validate and C_ref is not None:
                    rtol = 1e-4 if dtype == np.float32 else 1e-7
                    atol = 1e-6 if dtype == np.float32 else 1e-9
                    if not np.allclose(out, C_ref, rtol=rtol, atol=atol):
                        raise AssertionError("Validation failed: pydata.sparse spmm vs scipy")
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_spmm_lacuna(A_lacuna, B64), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":spmm", times, flops)
            if stats:
                results.append(stats)
            if args.validate and C_ref is not None:
                out = run_spmm_lacuna(A_lacuna, B64)
                rtol = 1e-4 if dtype == np.float32 else 1e-7
                atol = 1e-6 if dtype == np.float32 else 1e-9
                if not np.allclose(out, C_ref, rtol=rtol, atol=atol):
                    raise AssertionError("Validation failed: lacuna spmm vs scipy")

    # ---- sum / row_sums / col_sums ----
    if "sum" in wanted:
        ops = float(nnz)
        s_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_sum_scipy(A_scipy), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":sum", times, ops)
            if stats:
                results.append(stats)
            s_ref = run_sum_scipy(A_scipy)
        if not args.no_sparse and A_sparse is not None:
            out = run_sum_sparse(A_sparse, None)
            if out is not None:
                times = time_op(lambda: run_sum_sparse(A_sparse, None), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":sum", times, ops)
                if stats:
                    results.append(stats)
                if args.validate and s_ref is not None and isinstance(out, float):
                    rtol = 1e-4 if dtype == np.float32 else 1e-7
                    atol = 1e-6 if dtype == np.float32 else 1e-9
                    if not np.isclose(out, s_ref, rtol=rtol, atol=atol):
                        raise AssertionError("Validation failed: pydata.sparse sum vs scipy")
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_sum_lacuna(A_lacuna, None), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":sum", times, ops)
            if stats:
                results.append(stats)
            if args.validate and s_ref is not None:
                out = run_sum_lacuna(A_lacuna, None)
                rtol = 1e-4 if dtype == np.float32 else 1e-7
                atol = 1e-6 if dtype == np.float32 else 1e-9
                if not np.isclose(out, s_ref, rtol=rtol, atol=atol):
                    raise AssertionError("Validation failed: lacuna sum vs scipy")

    if "row_sums" in wanted:
        ops = float(nnz)
        rs_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_sum_axis_scipy(A_scipy, axis=1), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":row_sums", times, ops)
            if stats:
                results.append(stats)
            rs_ref = run_sum_axis_scipy(A_scipy, axis=1)
        if not args.no_sparse and A_sparse is not None:
            out = run_sum_sparse(A_sparse, 1)
            if out is not None:
                times = time_op(lambda: run_sum_sparse(A_sparse, 1), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":row_sums", times, ops)
                if stats:
                    results.append(stats)
                if args.validate and rs_ref is not None:
                    rtol = 1e-4 if dtype == np.float32 else 1e-7
                    atol = 1e-6 if dtype == np.float32 else 1e-9
                    if not np.allclose(out, rs_ref, rtol=rtol, atol=atol):
                        raise AssertionError("Validation failed: pydata.sparse row_sums vs scipy")
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_sum_lacuna(A_lacuna, 1), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":row_sums", times, ops)
            if stats:
                results.append(stats)
            if args.validate and rs_ref is not None:
                out = run_sum_lacuna(A_lacuna, 1)
                rtol = 1e-4 if dtype == np.float32 else 1e-7
                atol = 1e-6 if dtype == np.float32 else 1e-9
                if not np.allclose(out, rs_ref, rtol=rtol, atol=atol):
                    raise AssertionError("Validation failed: lacuna row_sums vs scipy")

    if "col_sums" in wanted:
        ops = float(nnz)
        cs_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_sum_axis_scipy(A_scipy, axis=0), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":col_sums", times, ops)
            if stats:
                results.append(stats)
            cs_ref = run_sum_axis_scipy(A_scipy, axis=0)
        if not args.no_sparse and A_sparse is not None:
            out = run_sum_sparse(A_sparse, 0)
            if out is not None:
                times = time_op(lambda: run_sum_sparse(A_sparse, 0), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":col_sums", times, ops)
                if stats:
                    results.append(stats)
                if args.validate and cs_ref is not None:
                    rtol = 1e-4 if dtype == np.float32 else 1e-7
                    atol = 1e-6 if dtype == np.float32 else 1e-9
                    if not np.allclose(out, cs_ref, rtol=rtol, atol=atol):
                        raise AssertionError("Validation failed: pydata.sparse col_sums vs scipy")
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_sum_lacuna(A_lacuna, 0), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":col_sums", times, ops)
            if stats:
                results.append(stats)
            if args.validate and cs_ref is not None:
                out = run_sum_lacuna(A_lacuna, 0)
                rtol = 1e-4 if dtype == np.float32 else 1e-7
                atol = 1e-6 if dtype == np.float32 else 1e-9
                if not np.allclose(out, cs_ref, rtol=rtol, atol=atol):
                    raise AssertionError("Validation failed: lacuna col_sums vs scipy")

    # ---- transpose ----
    if "transpose" in wanted:
        ops = float(nnz)
        T_ref = None
        if not args.no_scipy:
            times = time_op(lambda: run_transpose_scipy(A_scipy), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":transpose", times, ops)
            if stats:
                results.append(stats)
            T_ref = run_transpose_scipy(A_scipy)
        if not args.no_sparse and A_sparse is not None:
            out = run_transpose_sparse(A_sparse)
            if out is not None:
                times = time_op(lambda: run_transpose_sparse(A_sparse), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":transpose", times, ops)
                if stats:
                    results.append(stats)
                if args.validate and T_ref is not None:
                    out_dense = (
                        np.asarray(out.todense())
                        if hasattr(out, "todense")
                        else np.asarray(out.todense())
                        if hasattr(out, "todense")
                        else None
                    )
                    # If cannot validate, skip silently
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_transpose_lacuna(A_lacuna), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":transpose", times, ops)
            if stats:
                results.append(stats)

    # ---- prune / eliminate ----
    if "prune" in wanted:
        ops = float(nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_prune_scipy(A_scipy, args.eps), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":prune", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None:
            out = run_prune_sparse(A_sparse, args.eps)
            if out is not None:
                times = time_op(
                    lambda: run_prune_sparse(A_sparse, args.eps), args.warmup, args.repeat
                )
                stats = summarize(Backend.PYDATA_SPARSE + ":prune", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_prune_lacuna(A_lacuna, args.eps), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":prune", times, ops)
            if stats:
                results.append(stats)

    if "eliminate" in wanted:
        ops = float(nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_eliminate_scipy(A_scipy), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":eliminate", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None:
            out = run_eliminate_sparse(A_sparse)
            if out is not None:
                times = time_op(lambda: run_eliminate_sparse(A_sparse), args.warmup, args.repeat)
                stats = summarize(Backend.PYDATA_SPARSE + ":eliminate", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_eliminate_lacuna(A_lacuna), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":eliminate", times, ops)
            if stats:
                results.append(stats)

    # ---- add / mul ----
    if "add" in wanted:
        ops = float(A_scipy.nnz + A_scipy2.nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_add_scipy(A_scipy, A_scipy2), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":add", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None and A_sparse2 is not None:
            out = run_add_sparse(A_sparse, A_sparse2)
            if out is not None:
                times = time_op(
                    lambda: run_add_sparse(A_sparse, A_sparse2), args.warmup, args.repeat
                )
                stats = summarize(Backend.PYDATA_SPARSE + ":add", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None and B_lacuna is not None:
            times = time_op(lambda: run_add_lacuna(A_lacuna, B_lacuna), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":add", times, ops)
            if stats:
                results.append(stats)

    if "mul" in wanted:
        ops = float(nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_mul_scipy(A_scipy, args.alpha), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":mul", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None:
            out = run_mul_sparse(A_sparse, args.alpha)
            if out is not None:
                times = time_op(
                    lambda: run_mul_sparse(A_sparse, args.alpha), args.warmup, args.repeat
                )
                stats = summarize(Backend.PYDATA_SPARSE + ":mul", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(lambda: run_mul_lacuna(A_lacuna, args.alpha), args.warmup, args.repeat)
            stats = summarize(Backend.LACUNA + ":mul", times, ops)
            if stats:
                results.append(stats)

    # ---- subtraction (A - B) ----
    if "sub" in wanted:
        ops = float(A_scipy.nnz + A_scipy2.nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_sub_scipy(A_scipy, A_scipy2), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":sub", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None and A_sparse2 is not None:
            out = run_sub_sparse(A_sparse, A_sparse2)
            if out is not None:
                times = time_op(
                    lambda: run_sub_sparse(A_sparse, A_sparse2), args.warmup, args.repeat
                )
                stats = summarize(Backend.PYDATA_SPARSE + ":sub", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None and B_lacuna is not None:
            out = run_sub_lacuna(A_lacuna, B_lacuna)
            if out is not None:
                times = time_op(
                    lambda: run_sub_lacuna(A_lacuna, B_lacuna), args.warmup, args.repeat
                )
                stats = summarize(Backend.LACUNA + ":sub", times, ops)
                if stats:
                    results.append(stats)

    # ---- hadamard (elementwise multiply) ----
    if "hadamard" in wanted:
        ops = float(nnz)
        if not args.no_scipy:
            times = time_op(lambda: run_hadamard_scipy(A_scipy, A_scipy2), args.warmup, args.repeat)
            stats = summarize(Backend.SCIPY + ":hadamard", times, ops)
            if stats:
                results.append(stats)
        if not args.no_sparse and A_sparse is not None and A_sparse2 is not None:
            out = run_hadamard_sparse(A_sparse, A_sparse2)
            if out is not None:
                times = time_op(
                    lambda: run_hadamard_sparse(A_sparse, A_sparse2), args.warmup, args.repeat
                )
                stats = summarize(Backend.PYDATA_SPARSE + ":hadamard", times, ops)
                if stats:
                    results.append(stats)
        if not args.no_lacuna and A_lacuna is not None and B_lacuna is not None:
            out = run_hadamard_lacuna(A_lacuna, B_lacuna)
            if out is not None:
                times = time_op(
                    lambda: run_hadamard_lacuna(A_lacuna, B_lacuna), args.warmup, args.repeat
                )
                stats = summarize(Backend.LACUNA + ":hadamard", times, ops)
                if stats:
                    results.append(stats)

    # ---- print summary ----
    print(
        f"CSR Benchmarks: m={args.m} n={args.n} k={args.k} density={args.density} dtype={args.dtype} nnz={nnz}"
    )
    for r in results:
        if not r:
            continue
        print(
            f"{r['name']:>20}: min {r['min_ms']:.3f} ms | median {r['median_ms']:.3f} ms | mean {r['mean_ms']:.3f} ms | {r['gops']:.2f} GOps/s"
        )


if __name__ == "__main__":
    main()
