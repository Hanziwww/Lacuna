# Linear solve with Conjugate Gradient (CG)

This tutorial shows how to implement a simple linear solver using Lacuna’s CSR and a common Conjugate Gradient loop. It is fully SciPy-free and constructs the test matrix directly with NumPy.

## Problem

Solve Ax = b for a symmetric positive definite (SPD) sparse matrix A.

## Build an SPD test matrix (SciPy-free)

```python
import numpy as np

n = 4096
rs = np.random.RandomState(0)
density = 2e-4
nnz = max(n, int(n*n*density))

# Accumulate random COO into a dict to coalesce duplicates
coo = {}
ri = rs.randint(0, n, size=nnz, dtype=np.int64)
cj = rs.randint(0, n, size=nnz, dtype=np.int64)
val = rs.standard_normal(nnz)
for r, c, v in zip(ri, cj, val):
    coo[(int(r), int(c))] = coo.get((int(r), int(c)), 0.0) + float(v)

# Symmetrize: A = A + A.T (coalesce on the fly)
sym = {}
for (i, j), v in coo.items():
    sym[(i, j)] = sym.get((i, j), 0.0) + v
    sym[(j, i)] = sym.get((j, i), 0.0) + v

# Make strictly diagonally dominant => SPD
row_sum_abs = np.zeros(n, dtype=np.float64)
for (i, j), v in sym.items():
    if i != j:
        row_sum_abs[i] += abs(v)
eps = 1e-3
for i in range(n):
    sym[(i, i)] = sym.get((i, i), 0.0) + row_sum_abs[i] + eps

# Convert dict to CSR arrays
rows = [[] for _ in range(n)]
vals = [[] for _ in range(n)]
for (i, j), v in sym.items():
    rows[i].append(int(j))
    vals[i].append(float(v))
for i in range(n):
    if rows[i]:
        order = np.argsort(rows[i])
        rows[i] = list(np.asarray(rows[i])[order])
        vals[i] = list(np.asarray(vals[i])[order])

indptr = np.zeros(n+1, dtype=np.int64)
for i in range(n):
    indptr[i+1] = indptr[i] + len(rows[i])
indices = np.zeros(indptr[-1], dtype=np.int64)
data = np.zeros(indptr[-1], dtype=np.float64)
pos = 0
for i in range(n):
    k = len(rows[i])
    if k:
        indices[pos:pos+k] = np.asarray(rows[i], dtype=np.int64)
        data[pos:pos+k] = np.asarray(vals[i], dtype=np.float64)
        pos += k
shape = (n, n)
```

## Wrap in Lacuna CSR

```python
from lacuna.sparse.csr import CSR

A_lac = CSR(indptr, indices, data, shape, check=False)
```

## Conjugate Gradient (CG)

```python
def cg(matvec, b, tol=1e-8, maxiter=1000, x0=None):
    import numpy as np
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    r = b - matvec(x)
    p = r.copy()
    rr = float(np.dot(r, r))
    nb = float(np.linalg.norm(b)) or 1.0
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if pAp == 0.0:
            break
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(np.dot(r, r))
        if (rr_new ** 0.5) / nb <= tol:
            rr = rr_new
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    rel = (rr ** 0.5) / nb
    return x, k, rel
```

## Run

```python
rs = np.random.RandomState(123)
x_true = rs.standard_normal(n)
b = np.asarray(A_lac @ x_true, dtype=np.float64)

# Lacuna matvec
matvec_lac = lambda v: np.asarray(A_lac @ v, dtype=np.float64)

x, iters, rel = cg(matvec_lac, b, tol=1e-8, maxiter=1000)
print(iters, rel)
```

## Notes

- Lacuna uses Rayon threads internally; control with:
  ```python
  import lacuna as la
  la.set_num_threads(8)
  ```
- For full benchmark and plotting, see `python/benchmarks/benchmark_linear_solve.py`.
