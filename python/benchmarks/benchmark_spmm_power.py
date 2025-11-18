import argparse
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# ---------- Matrix builder ----------


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


# ---------- Backends: SpMM wrappers ----------


def spmm_scipy(A: Any, Q: np.ndarray) -> np.ndarray:
    return np.asarray(A @ Q, dtype=np.float64)


def spmm_sparse(A_sparse: Any, Q: np.ndarray) -> Optional[np.ndarray]:
    try:
        out = A_sparse @ Q
    except Exception:
        try:
            import sparse as psparse

            out = psparse.tensordot(A_sparse, Q, axes=1)
        except Exception:
            return None
    return np.asarray(out, dtype=np.float64)


def spmm_lacuna(A_lacuna: Any, Q: np.ndarray) -> np.ndarray:
    return np.asarray(A_lacuna @ Q, dtype=np.float64)


# ---------- Block power iteration ----------


def block_power(
    spmm: Callable[[np.ndarray], np.ndarray], n: int, k: int, iters: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed + 999)
    Q = rs.standard_normal((n, k)).astype(np.float64)
    Q, _ = np.linalg.qr(Q, mode="reduced")
    for _ in range(iters):
        Z = spmm(Q)
        Q, _ = np.linalg.qr(Z, mode="reduced")
    AQ = spmm(Q)
    B = Q.T @ AQ
    evals = np.linalg.eigvalsh(B)[::-1]
    return Q, evals


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
        A_scipy, _ = build_spd_csr(n, args.density, args.seed, dtype)
        try:
            from lacuna.sparse.csr import CSR
        except Exception:
            CSR = None
        A_lacuna = (
            None
            if (args.no_lacuna or CSR is None)
            else CSR(
                A_scipy.indptr,
                A_scipy.indices,
                A_scipy.data.astype(np.float64, copy=False),
                A_scipy.shape,
                check=False,
            )
        )
        A_sparse = None
        if not args.no_sparse:
            try:
                import sparse as psparse  # noqa: F401

                A_sparse = A_scipy.tocoo()
                A_sparse = psparse.COO.from_scipy_sparse(A_sparse)
            except Exception:
                A_sparse = None

        if not args.no_scipy:
            times = time_op(
                lambda: block_power(
                    lambda Q: spmm_scipy(A_scipy, Q), n, args.k, args.iters, args.seed
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("scipy:power", []).append(float(np.min(times) * 1e3))

        if not args.no_sparse and A_sparse is not None:
            times = time_op(
                lambda: block_power(
                    lambda Q: spmm_sparse(A_sparse, Q), n, args.k, args.iters, args.seed
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("pydata.sparse:power", []).append(float(np.min(times) * 1e3))

        if not args.no_lacuna and A_lacuna is not None:
            times = time_op(
                lambda: block_power(
                    lambda Q: spmm_lacuna(A_lacuna, Q.astype(np.float64, copy=False)),
                    n,
                    args.k,
                    args.iters,
                    args.seed,
                ),
                args.warmup,
                args.repeat,
            )
            series.setdefault("lacuna:power", []).append(float(np.min(times) * 1e3))

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
        f"Block power iteration (k={args.k}, iters={args.iters}) vs n | SPD density={args.density}"
    )
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    out = getattr(args, "plot_file", "spmm_power_times.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved plot to {out}")


# ---------- Main ----------


def main():
    p = argparse.ArgumentParser(
        description="SpMM benchmark via Block Power Iteration across SciPy, PyData/Sparse, Lacuna"
    )
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--density", type=float, default=2e-4)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--k", type=int, default=32, help="Block size (number of eigenvectors)")
    p.add_argument("--iters", type=int, default=10, help="Number of power iterations")
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
    p.add_argument("--plot_file", type=str, default="spmm_power_times.png")

    args = p.parse_args()
    dtype = np.float64 if args.dtype == "float64" else np.float32

    if sp is None:
        raise SystemExit("SciPy is required for this benchmark (to build matrices)")

    if args.sweep:
        run_sweep(args)
        return

    A_scipy, _ = build_spd_csr(args.n, args.density, args.seed, dtype)

    try:
        from lacuna.sparse.csr import CSR
    except Exception:
        CSR = None
    A_lacuna = (
        None
        if (args.no_lacuna or CSR is None)
        else CSR(
            A_scipy.indptr,
            A_scipy.indices,
            A_scipy.data.astype(np.float64, copy=False),
            A_scipy.shape,
            check=False,
        )
    )

    A_sparse = None
    if not args.no_sparse:
        try:
            import sparse as psparse  # noqa: F401

            A_sparse = A_scipy.tocoo()
            A_sparse = psparse.COO.from_scipy_sparse(A_sparse)
        except Exception:
            A_sparse = None

    results: List[Dict[str, float]] = []
    notes: List[str] = []

    # SciPy
    if not args.no_scipy:
        times = time_op(
            lambda: block_power(
                lambda Q: spmm_scipy(A_scipy, Q), args.n, args.k, args.iters, args.seed
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("scipy:power", times)
        if stats:
            results.append(stats)
        _, evals = block_power(
            lambda Q: spmm_scipy(A_scipy, Q), args.n, args.k, args.iters, args.seed
        )
        notes.append(f"scipy: lambda_max~{float(evals[0]):.4e}")

    # PyData/Sparse
    if not args.no_sparse and A_sparse is not None:
        times = time_op(
            lambda: block_power(
                lambda Q: spmm_sparse(A_sparse, Q), args.n, args.k, args.iters, args.seed
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("pydata.sparse:power", times)
        if stats:
            results.append(stats)
        _, evals = block_power(
            lambda Q: spmm_sparse(A_sparse, Q), args.n, args.k, args.iters, args.seed
        )
        notes.append(f"pydata.sparse: lambda_max~{float(evals[0]):.4e}")

    # Lacuna
    if not args.no_lacuna and A_lacuna is not None:
        times = time_op(
            lambda: block_power(
                lambda Q: spmm_lacuna(A_lacuna, Q.astype(np.float64, copy=False)),
                args.n,
                args.k,
                args.iters,
                args.seed,
            ),
            args.warmup,
            args.repeat,
        )
        stats = summarize("lacuna:power", times)
        if stats:
            results.append(stats)
        _, evals = block_power(
            lambda Q: spmm_lacuna(A_lacuna, Q.astype(np.float64, copy=False)),
            args.n,
            args.k,
            args.iters,
            args.seed,
        )
        notes.append(f"lacuna: lambda_max~{float(evals[0]):.4e}")

    print(
        f"Block Power Iteration: n={args.n} k={args.k} iters={args.iters} density={args.density} dtype={args.dtype}"
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
