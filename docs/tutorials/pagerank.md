# Block PageRank

This tutorial implements block Personalized PageRank with Lacuna’s CSR matrices. It is fully SciPy-free and builds the row-stochastic matrix directly with NumPy.

## Problem

Given a row-stochastic transition matrix P (n×n) and k seed columns X0 (n×k), iterate

X_{t+1} = α · P @ X_t + (1−α) · X0,

column-normalizing each step until convergence.

## Build a row-stochastic matrix (SciPy-free)

```python
import numpy as np

n = 10000
rs = np.random.RandomState(0)
density = 2e-4
nnz = max(n, int(n*n*density))

# Accumulate random unweighted adjacency in dict to coalesce duplicates
coo = {}
ri = rs.randint(0, n, size=nnz, dtype=np.int64)
cj = rs.randint(0, n, size=nnz, dtype=np.int64)
for r, c in zip(ri, cj):
    coo[(int(r), int(c))] = 1.0

# Build CSR arrays (A)
rows = [[] for _ in range(n)]
vals = [[] for _ in range(n)]
for (i, j), v in coo.items():
    rows[i].append(int(j))
    vals[i].append(float(v))

# Add self-loops for dangling rows
outdeg = np.array([len(rows[i]) for i in range(n)], dtype=np.int64)
for i in range(n):
    if outdeg[i] == 0:
        rows[i].append(i)
        vals[i].append(1.0)

# Sort columns per row and compute row-normalization
indptr = np.zeros(n+1, dtype=np.int64)
for i in range(n):
    if rows[i]:
        order = np.argsort(rows[i])
        rows[i] = list(np.asarray(rows[i])[order])
        vals[i] = list(np.asarray(vals[i])[order])
    indptr[i+1] = indptr[i] + len(rows[i])
indices = np.zeros(indptr[-1], dtype=np.int64)
data = np.zeros(indptr[-1], dtype=np.float64)
pos = 0
for i in range(n):
    k = len(rows[i])
    if k:
        indices[pos:pos+k] = np.asarray(rows[i], dtype=np.int64)
        # row-stochastic => divide by out-degree (k) here
        data[pos:pos+k] = np.asarray(vals[i], dtype=np.float64) / float(k)
        pos += k
shape = (n, n)
```

## Wrap in Lacuna CSR

```python
from lacuna.sparse.csr import CSR
P_lac = CSR(indptr, indices, data, shape, check=False)
```

## Block PageRank loop

```python
def block_pagerank(matmul, X0, alpha=0.85, tol=1e-8, maxiter=100):
    import numpy as np
    X = X0.copy()
    for k in range(1, maxiter + 1):
        PX = matmul(X)
        X_new = alpha * PX + (1.0 - alpha) * X0
        # column-normalize to keep probability mass
        s = X_new.sum(axis=0)
        s[s == 0.0] = 1.0
        X_new = X_new / s
        rel = float(np.linalg.norm(X_new - X) / (np.linalg.norm(X_new) or 1.0))
        X = X_new
        if rel <= tol:
            return X, k, rel
    return X, maxiter, rel
```

## Run

```python
k = 16
rs = np.random.RandomState(123)
X0 = np.zeros((n, k), dtype=np.float64)
seeds = rs.randint(0, n, size=k)
X0[seeds, np.arange(k)] = 1.0

# Lacuna matmul
matmul_lac = lambda X: (P_lac @ X).astype(np.float64)
X, iters, rel = block_pagerank(matmul_lac, X0, alpha=0.85, tol=1e-8, maxiter=100)
print(iters, rel)
```

## Notes

- Control threads for Lacuna kernels:
  ```python
  import lacuna as la
  la.set_num_threads(8)
  ```
- For end-to-end timing and plots, see `python/benchmarks/benchmark_pagerank_block.py`.
