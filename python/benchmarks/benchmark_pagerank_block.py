import argparse
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# ---------- Graph builder (row-stochastic) ----------


def build_row_stochastic_csr(n: int, density: float, seed: int, dtype: np.dtype) -> Any:
    """Build a random row-stochastic transition matrix P (CSR, shape n x n).

    - Start from a random adjacency with expected nnz ~ n^2 * density.
    - Add self-loops for dangling nodes (zero out-degree).
    - Row-normalize to make each row sum to 1.
    """
    if sp is None:
        raise RuntimeError("SciPy is required to build test matrices")
    rs = np.random.RandomState(seed)
    nnz = max(n, int(n * n * float(density)))
    row = rs.randint(0, n, size=nnz, dtype=np.int64)
    col = rs.randint(0, n, size=nnz, dtype=np.int64)
    data = np.ones(nnz, dtype=dtype)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A.sum_duplicates()

    # Handle dangling rows: add self-loop
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    idx = np.where(deg == 0)[0]
    if idx.size > 0:
        D = sp.csr_matrix((np.ones(idx.size, dtype=dtype), (idx, idx)), shape=(n, n))
        A = (A + D).tocsr()
        A.sum_duplicates()
        deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)

    # Row-normalize: P = D^{-1} A
    invdeg = 1.0 / deg
    P = sp.diags(invdeg, format="csr") @ A
    return P.astype(np.float64)


# ---------- Backends: SpMM wrappers ----------


def spmm_scipy(P: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(P @ X, dtype=np.float64)


def spmm_sparse(P_sparse: Any, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        out = P_sparse @ X
    except Exception:
        try:
            import sparse as psparse

            out = psparse.tensordot(P_sparse, X, axes=1)
        except Exception:
            return None
    return np.asarray(out, dtype=np.float64)


def spmm_lacuna(P_lacuna: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(P_lacuna @ X, dtype=np.float64)


# ---------- Block PageRank ----------


def block_pagerank(
    matmul: Callable[[np.ndarray], np.ndarray],
    X0: np.ndarray,
    alpha: float,
    tol: float,
    maxiter: int,
) -> Tuple[np.ndarray, int, float]:
    X = X0.copy()
    nb = float(np.linalg.norm(X0))
    nb = nb if nb > 0 else 1.0
    k = 0
    for k in range(1, maxiter + 1):
        PX = matmul(X)
        X_new = alpha * PX + (1.0 - alpha) * X0
        # Normalize columns to sum to 1 (probability mass)
        colsum = X_new.sum(axis=0)
        colsum[colsum == 0.0] = 1.0
        X_new = X_new / colsum
        r = X_new - X
        rel = float(
            np.linalg.norm(r) / (np.linalg.norm(X_new) if np.linalg.norm(X_new) > 0 else 1.0)
        )
        X = X_new
        if rel <= tol:
            break
    return X, k, rel


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


# ---------- Sweep/plot helpers ----------


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
        P_scipy = build_row_stochastic_csr(n, args.density, args.seed, dtype)
        rs = np.random.RandomState(args.seed + 123)
        # Personalized seeds: one-hot per column
        kcols = int(args.k)
        X0 = np.zeros((n, kcols), dtype=np.float64)
        seeds = rs.randint(0, n, size=kcols)
        X0[seeds, np.arange(kcols)] = 1.0

        # Build backends
        try:
            from lacuna.sparse.csr import CSR
        except Exception:
            CSR = None
        P_lacuna = (
            None
            if (args.no_lacuna or CSR is None)
            else CSR(
                P_scipy.indptr,
                P_scipy.indices,
                P_scipy.data.astype(np.float64, copy=False),
                P_scipy.shape,
                check=False,
            )
        )
        P_sparse = None
        if not args.no_sparse:
            try:
                import sparse as psparse  # noqa: F401

                P_sparse = psparse.COO.from_scipy_sparse(P_scipy.tocoo())
            except Exception:
                P_sparse = None

        if not args.no_scipy:
            times = time_op(
                lambda: block_pagerank(
                    lambda X: spmm_scipy(P_scipy, X), X0, args.alpha, args.tol, args.maxiter
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("scipy:pagerank", []).append(float(np.min(times) * 1e3))

        if not args.no_sparse and P_sparse is not None:
            times = time_op(
                lambda: block_pagerank(
                    lambda X: spmm_sparse(P_sparse, X), X0, args.alpha, args.tol, args.maxiter
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("pydata.sparse:pagerank", []).append(float(np.min(times) * 1e3))

        if not args.no_lacuna and P_lacuna is not None:
            times = time_op(
                lambda: block_pagerank(
                    lambda X: spmm_lacuna(P_lacuna, X.astype(np.float64, copy=False)),
                    X0,
                    args.alpha,
                    args.tol,
                    args.maxiter,
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("lacuna:pagerank", []).append(float(np.min(times) * 1e3))

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
    plt.ylabel("Time (ms) per run (min over repeats)")
    plt.title(
        f"Block PageRank time vs n (alpha={args.alpha}, k={args.k}, density={args.density}, maxiter={args.maxiter})"
    )
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    out = getattr(args, "plot_file", "pagerank_block_times.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved plot to {out}")


# ---------- Main ----------


def main():
    p = argparse.ArgumentParser(
        description="Block PageRank benchmark across SciPy, PyData/Sparse, Lacuna"
    )
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--density", type=float, default=2e-4)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--k", type=int, default=16, help="Number of personalized PageRank columns")
    p.add_argument("--alpha", type=float, default=0.85, help="Damping factor")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_scipy", action="store_true")
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--no_lacuna", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n_min", type=int, default=1000)
    p.add_argument("--n_max", type=int, default=100000)
    p.add_argument("--num_points", type=int, default=10)
    p.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--plot_file", type=str, default="pagerank_block_times.png")

    args = p.parse_args()
    dtype = np.float64 if args.dtype == "float64" else np.float32

    if sp is None:
        raise SystemExit("SciPy is required for this benchmark (to build matrices)")

    if args.sweep:
        run_sweep(args)
        return

    P_scipy = build_row_stochastic_csr(args.n, args.density, args.seed, dtype)

    # Build X0
    rs = np.random.RandomState(args.seed + 123)
    X0 = np.zeros((args.n, int(args.k)), dtype=np.float64)
    seeds = rs.randint(0, args.n, size=int(args.k))
    X0[seeds, np.arange(int(args.k))] = 1.0

    # Backends
    try:
        from lacuna.sparse.csr import CSR
    except Exception:
        CSR = None
    P_lacuna = (
        None
        if (args.no_lacuna or CSR is None)
        else CSR(
            P_scipy.indptr,
            P_scipy.indices,
            P_scipy.data.astype(np.float64, copy=False),
            P_scipy.shape,
            check=False,
        )
    )

    P_sparse = None
    if not args.no_sparse:
        try:
            import sparse as psparse  # noqa: F401

            P_sparse = psparse.COO.from_scipy_sparse(P_scipy.tocoo())
        except Exception:
            P_sparse = None

    results: List[Dict[str, float]] = []
    notes: List[str] = []

    # SciPy
    if not args.no_scipy:
        times = time_op(
            lambda: block_pagerank(
                lambda X: spmm_scipy(P_scipy, X), X0, args.alpha, args.tol, args.maxiter
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("scipy:pagerank", times)
        if stats:
            results.append(stats)
        _, iters, rel = block_pagerank(
            lambda X: spmm_scipy(P_scipy, X), X0, args.alpha, args.tol, args.maxiter
        )
        notes.append(f"scipy: iters={iters} rel_resid={rel:.2e}")

    # PyData/Sparse
    if not args.no_sparse and P_sparse is not None:
        times = time_op(
            lambda: block_pagerank(
                lambda X: spmm_sparse(P_sparse, X), X0, args.alpha, args.tol, args.maxiter
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("pydata.sparse:pagerank", times)
        if stats:
            results.append(stats)
        _, iters, rel = block_pagerank(
            lambda X: spmm_sparse(P_sparse, X), X0, args.alpha, args.tol, args.maxiter
        )
        notes.append(f"pydata.sparse: iters={iters} rel_resid={rel:.2e}")

    # Lacuna
    if not args.no_lacuna and P_lacuna is not None:
        times = time_op(
            lambda: block_pagerank(
                lambda X: spmm_lacuna(P_lacuna, X.astype(np.float64, copy=False)),
                X0,
                args.alpha,
                args.tol,
                args.maxiter,
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("lacuna:pagerank", times)
        if stats:
            results.append(stats)
        _, iters, rel = block_pagerank(
            lambda X: spmm_lacuna(P_lacuna, X.astype(np.float64, copy=False)),
            X0,
            args.alpha,
            args.tol,
            args.maxiter,
        )
        notes.append(f"lacuna: iters={iters} rel_resid={rel:.2e}")

    print(
        f"Block PageRank: n={args.n} k={args.k} alpha={args.alpha} maxiter={args.maxiter} density={args.density} dtype={args.dtype}"
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
