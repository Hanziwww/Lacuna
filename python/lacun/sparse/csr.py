from .base import SparseMatrix
import numpy as np

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None

class CSR(SparseMatrix):
    def __init__(self, indptr, indices, data, shape, dtype=np.float64, check=True):
        super().__init__(shape=shape, dtype=dtype)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        if check and _core is not None:
            # Validate by constructing once through core (errors if invalid)
            _ = _core.sum_from_parts(self.shape[0], self.shape[1], self.indptr, self.indices, self.data, True)

    @classmethod
    def from_arrays(cls, indptr, indices, data, shape, check=True):
        return cls(indptr, indices, data, shape, check=check)

    @property
    def nnz(self):
        return int(self.data.size)

    def _parts(self):
        return (self.shape[0], self.shape[1], self.indptr, self.indices, self.data)

    def __matmul__(self, other):
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        arr = np.asarray(other)
        if arr.ndim == 1:
            if arr.shape[0] != ncols:
                raise ValueError("vector length must equal ncols")
            return _core.spmv_from_parts(nrows, ncols, indptr, indices, data, arr, False)
        elif arr.ndim == 2:
            if arr.shape[0] != ncols:
                raise ValueError("matrix rows must equal ncols")
            return _core.spmm_from_parts(nrows, ncols, indptr, indices, data, arr, False)
        else:
            raise ValueError("right operand must be 1D or 2D")

    def sum(self, axis=None):
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        if axis is None:
            return float(_core.sum_from_parts(nrows, ncols, indptr, indices, data, False))
        if axis == 0:
            return _core.col_sums_from_parts(nrows, ncols, indptr, indices, data, False)
        if axis == 1:
            return _core.row_sums_from_parts(nrows, ncols, indptr, indices, data, False)
        raise ValueError("axis must be None, 0, or 1")

    @property
    def T(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        ti, tj, tv, tr, tc = _core.transpose_from_parts(nrows, ncols, indptr, indices, data, False)
        return CSR(ti, tj, tv, (tr, tc), check=False)

    def prune(self, eps):
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        pi, pj, pv, pr, pc = _core.prune_from_parts(nrows, ncols, indptr, indices, data, float(eps), False)
        return CSR(pi, pj, pv, (pr, pc), check=False)

    def eliminate_zeros(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        pi, pj, pv, pr, pc = _core.eliminate_zeros_from_parts(nrows, ncols, indptr, indices, data, False)
        return CSR(pi, pj, pv, (pr, pc), check=False)

    def __mul__(self, alpha):
        if _core is None:
            raise RuntimeError("native core is not available")
        alpha = float(alpha)
        nrows, ncols, indptr, indices, data = self._parts()
        oi, oj, ov, orr, occ = _core.mul_scalar_from_parts(nrows, ncols, indptr, indices, data, alpha, False)
        return CSR(oi, oj, ov, (orr, occ), check=False)

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, CSR):
            if _core is None:
                raise RuntimeError("native core is not available")
            a = self._parts()
            b = other._parts()
            ci, cj, cv, cr, cc = _core.add_from_parts(
                a[0], a[1], a[2], a[3], a[4],
                b[0], b[1], b[2], b[3], b[4],
                False,
            )
            return CSR(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    # Minimal read-only indexing
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                # scalar lookup
                i = int(i); j = int(j)
                if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                    raise IndexError("index out of bounds")
                s = int(self.indptr[i]); e = int(self.indptr[i+1])
                row_idx = self.indices[s:e]
                pos = np.searchsorted(row_idx, j)
                if pos < row_idx.size and row_idx[pos] == j:
                    return float(self.data[s+pos])
                return 0.0
            if isinstance(i, int) and j == slice(None):
                i = int(i)
                s = int(self.indptr[i]); e = int(self.indptr[i+1])
                out = np.zeros((self.shape[1],), dtype=self.data.dtype)
                out[self.indices[s:e]] = self.data[s:e]
                return out
            if i == slice(None) and isinstance(j, int):
                j = int(j)
                # slow column gather
                out = np.zeros((self.shape[0],), dtype=self.data.dtype)
                for r in range(self.shape[0]):
                    s = int(self.indptr[r]); e = int(self.indptr[r+1])
                    row_idx = self.indices[s:e]
                    pos = np.searchsorted(row_idx, j)
                    if pos < row_idx.size and row_idx[pos] == j:
                        out[r] = self.data[s+pos]
                return out
        raise NotImplementedError("advanced indexing is not implemented in v0.1")

    def toarray(self):
        nrows, ncols = self.shape
        out = np.zeros((nrows, ncols), dtype=self.data.dtype)
        # fill row by row
        for i in range(nrows):
            s = int(self.indptr[i]); e = int(self.indptr[i+1])
            if s < e:
                out[i, self.indices[s:e]] = self.data[s:e]
        return out

    def astype(self, dtype):
        """Return a copy with the given dtype. v0.1 supports float64 only."""
        dtype = np.dtype(dtype)
        if dtype == np.float64:
            if self.data.dtype == np.float64:
                return CSR(self.indptr.copy(), self.indices.copy(), self.data.copy(), self.shape, dtype=np.float64, check=False)
            else:
                return CSR(self.indptr.copy(), self.indices.copy(), self.data.astype(np.float64, copy=True), self.shape, dtype=np.float64, check=False)
        raise NotImplementedError("astype supports only float64 in v0.1")
