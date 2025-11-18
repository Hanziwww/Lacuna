import argparse
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as _e:
    sp = None
    spla = None


def build_spd_csr(n: int, density: float, seed: int, dtype: np.dtype) -> Tuple[Any, int]:
    if sp is None:
        raise RuntimeError("SciPy is required to build test matrices")
    rs = np.random.RandomState(seed)
    nnz = max(n, int(n * n * float(density)))
    row = rs.randint(0, n, size=nnz, dtype=np.int64)
    col = rs.randint(0, n, size=nnz, dtype=np.int64)
    data = rs.standard_normal(nnz).astype(dtype, copy=False)
    A0 = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A = (A0 + A0.transpose()).tocsr()
    abs_row_sum = np.asarray(np.abs(A).sum(axis=1)).ravel()
    shift = 1e-3
    A.setdiag(abs_row_sum + shift)
    A.eliminate_zeros()
    return A.tocsr(), int(A.nnz)


def build_lacuna_csr_from_scipy(A_scipy: Any) -> Optional[Any]:
    try:
        from lacuna.sparse.csr import CSR
    except Exception:
        return None
    try:
        csr = A_scipy.tocsr()
        data64 = csr.data.astype(np.float64, copy=False)
        return CSR(csr.indptr, csr.indices, data64, csr.shape, check=False)
    except Exception:
        return None


def build_pydata_sparse_from_scipy(A_scipy: Any) -> Optional[Any]:
    try:
        import sparse as psparse
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


def cg_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray],
    tol: float,
    maxiter: int,
) -> Tuple[np.ndarray, int, float]:
    n = b.size
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    r = b - matvec(x)
    p = r.copy()
    rr = float(np.dot(r, r))
    nb = float(np.linalg.norm(b))
    nb = nb if nb > 0 else 1.0
    k = 0
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if pAp == 0.0:
            break
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(np.dot(r, r))
        if (rr_new**0.5) / nb <= tol:
            rr = rr_new
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    rel_res = (rr**0.5) / nb
    return x, k, rel_res


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


def summarize(name: str, times: List[float]) -> Optional[Dict[str, float]]:
    if not times:
        return None
    arr = np.array(times, dtype=np.float64)
    return {
        "name": name,
        "min_ms": float(arr.min() * 1e3),
        "median_ms": float(np.median(arr) * 1e3),
        "mean_ms": float(arr.mean() * 1e3),
    }


def sizes_from_args(args) -> List[int]:
    if not getattr(args, "sweep", False):
        return [int(args.n)]
    n_min = int(args.n_min)
    n_max = int(args.n_max)
    k = int(args.num_points)
    if args.scale == "log":
        arr = np.geomspace(max(1, n_min), max(1, n_max), k)
    else:
        arr = np.linspace(max(1, n_min), max(1, n_max), k)
    sizes = sorted({int(max(1, round(v))) for v in arr})
    return sizes


def run_sweep(args) -> None:
    dtype = np.float64 if args.dtype == "float64" else np.float32
    if sp is None:
        raise SystemExit("SciPy is required for this benchmark (to build matrices)")
    sizes = sizes_from_args(args)
    series: Dict[str, List[float]] = {}
    for n in sizes:
        A_scipy, _ = build_spd_csr(n, args.density, args.seed, dtype)
        rs = np.random.RandomState(args.seed + 123)
        x_true = rs.standard_normal(n).astype(np.float64)
        b = np.asarray(A_scipy @ x_true, dtype=np.float64)
        A_sparse = None if args.no_sparse else build_pydata_sparse_from_scipy(A_scipy)
        A_lacuna = None if args.no_lacuna else build_lacuna_csr_from_scipy(A_scipy)

        def matvec_scipy(v: np.ndarray) -> np.ndarray:
            return np.asarray(A_scipy @ v, dtype=np.float64)

        def matvec_sparse(v: np.ndarray) -> Optional[np.ndarray]:
            try:
                return np.asarray(A_sparse @ v, dtype=np.float64)
            except Exception:
                return None

        def matvec_lacuna(v: np.ndarray) -> Optional[np.ndarray]:
            try:
                out = A_lacuna @ v
                return np.asarray(out, dtype=np.float64)
            except Exception:
                return None

        if not args.no_scipy:
            times = time_op(
                lambda: cg_solve(matvec_scipy, b, None, args.tol, args.maxiter),
                args.warmup,
                args.repeat,
            )
            lab = "scipy:cg(common)"
            series.setdefault(lab, []).append(float(np.min(times) * 1e3))

        if not args.no_sparse and A_sparse is not None:
            times = time_op(
                lambda: cg_solve(lambda v: matvec_sparse(v), b, None, args.tol, args.maxiter),
                args.warmup,
                args.repeat,
            )
            lab = "pydata.sparse:cg(common)"
            series.setdefault(lab, []).append(float(np.min(times) * 1e3))

        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(
                lambda: cg_solve(lambda v: matvec_lacuna(v), b, None, args.tol, args.maxiter),
                args.warmup,
                args.repeat,
            )
            lab = "lacuna:cg(common)"
            series.setdefault(lab, []).append(float(np.min(times) * 1e3))

        if args.scipy_native_cg and not args.no_scipy and spla is not None:

            def solve_scipy_native():
                x, info = spla.cg(A_scipy, b, tol=args.tol, maxiter=args.maxiter)
                return x

            times = time_op(solve_scipy_native, args.warmup, args.repeat)
            lab = "scipy:cg(native)"
            series.setdefault(lab, []).append(float(np.min(times) * 1e3))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required for plotting. Install via: pip install matplotlib")
        for name, vals in series.items():
            print(name, vals)
        return

    plt.figure(figsize=(8, 5))
    for name, vals in series.items():
        plt.plot(sizes, vals, marker="o", label=name)
    plt.xscale("log" if args.scale == "log" else "linear")
    plt.yscale("log")
    plt.xlabel("Problem size n")
    plt.ylabel("Time (ms) per solve (min over repeats)")
    plt.title(
        f"CG solve time vs n (SPD, density={args.density}, tol={args.tol}, maxiter={args.maxiter})"
    )
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    out = getattr(args, "plot_file", "linear_solve_cg_times.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved plot to {out}")


def main():
    p = argparse.ArgumentParser(
        description="Linear solve (CG) benchmark across SciPy, PyData/Sparse, Lacuna"
    )
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--density", type=float, default=2e-4)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_scipy", action="store_true")
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--no_lacuna", action="store_true")
    p.add_argument(
        "--scipy_native_cg", action="store_true", help="Also time scipy.sparse.linalg.cg"
    )
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n_min", type=int, default=1000)
    p.add_argument("--n_max", type=int, default=100000)
    p.add_argument("--num_points", type=int, default=10)
    p.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--plot_file", type=str, default="linear_solve_cg_times.png")

    args = p.parse_args()
    dtype = np.float64 if args.dtype == "float64" else np.float32

    if sp is None:
        raise SystemExit("SciPy is required for this benchmark (to build matrices)")

    if args.sweep:
        run_sweep(args)
        return

    A_scipy, nnz = build_spd_csr(args.n, args.density, args.seed, dtype)

    rs = np.random.RandomState(args.seed + 123)
    x_true = rs.standard_normal(args.n).astype(np.float64)
    b = np.asarray(A_scipy @ x_true, dtype=np.float64)

    A_sparse = None if args.no_sparse else build_pydata_sparse_from_scipy(A_scipy)
    A_lacuna = None if args.no_lacuna else build_lacuna_csr_from_scipy(A_scipy)

    results: List[Dict[str, float]] = []
    notes: List[str] = []

    def matvec_scipy(v: np.ndarray) -> np.ndarray:
        return np.asarray(A_scipy @ v, dtype=np.float64)

    def matvec_sparse(v: np.ndarray) -> Optional[np.ndarray]:
        try:
            return np.asarray(A_sparse @ v, dtype=np.float64)
        except Exception:
            return None

    def matvec_lacuna(v: np.ndarray) -> Optional[np.ndarray]:
        try:
            out = A_lacuna @ v
            return np.asarray(out, dtype=np.float64)
        except Exception:
            return None

    if not args.no_scipy:
        times = time_op(
            lambda: cg_solve(matvec_scipy, b, None, args.tol, args.maxiter),
            args.warmup,
            args.repeat,
        )
        stats = summarize("scipy:cg(common)", times)
        if stats:
            results.append(stats)
        _x, iters, rel = cg_solve(matvec_scipy, b, None, args.tol, args.maxiter)
        notes.append(f"scipy: iters={iters} rel_resid={rel:.2e}")

    if not args.no_sparse and A_sparse is not None:
        times = time_op(
            lambda: cg_solve(lambda v: matvec_sparse(v), b, None, args.tol, args.maxiter),
            args.warmup,
            args.repeat,
        )
        stats = summarize("pydata.sparse:cg(common)", times)
        if stats:
            results.append(stats)
        _x, iters, rel = cg_solve(lambda v: matvec_sparse(v), b, None, args.tol, args.maxiter)
        notes.append(f"pydata.sparse: iters={iters} rel_resid={rel:.2e}")

    if not args.no_lacuna and A_lacuna is not None:
        times = time_op(
            lambda: cg_solve(lambda v: matvec_lacuna(v), b, None, args.tol, args.maxiter),
            args.warmup,
            args.repeat,
        )
        stats = summarize("lacuna:cg(common)", times)
        if stats:
            results.append(stats)
        _x, iters, rel = cg_solve(lambda v: matvec_lacuna(v), b, None, args.tol, args.maxiter)
        notes.append(f"lacuna: iters={iters} rel_resid={rel:.2e}")

    if args.scipy_native_cg and not args.no_scipy and spla is not None:

        def solve_scipy_native():
            x, info = spla.cg(A_scipy, b, tol=args.tol, maxiter=args.maxiter)
            return x

        times = time_op(solve_scipy_native, args.warmup, args.repeat)
        stats = summarize("scipy:cg(native)", times)
        if stats:
            results.append(stats)
        x, info = spla.cg(A_scipy, b, tol=args.tol, maxiter=args.maxiter)
        r = b - np.asarray(A_scipy @ x, dtype=np.float64)
        rel = float(np.linalg.norm(r) / (np.linalg.norm(b) if np.linalg.norm(b) > 0 else 1.0))
        iters = -1 if info is None else (int(info) if isinstance(info, (int, np.integer)) else -1)
        notes.append(f"scipy(native): iters={iters} rel_resid={rel:.2e}")

    print(
        f"Linear Solve (CG) Benchmark: n={args.n} density={args.density} dtype={args.dtype} nnz={nnz} tol={args.tol} maxiter={args.maxiter}"
    )
    for r in results:
        if not r:
            continue
        print(
            f"{r['name']:>24}: min {r['min_ms']:.3f} ms | median {r['median_ms']:.3f} ms | mean {r['mean_ms']:.3f} ms"
        )
    for line in notes:
        print("  - " + line)


if __name__ == "__main__":
    main()
