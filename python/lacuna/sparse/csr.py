"""Sparse CSR matrix implementation backed by a Rust core.

This module exposes a minimal, NumPy-friendly CSR matrix class with
high-performance kernels delegated to a PyO3-bound Rust extension.

Notes
-----
- Index arrays (`indptr`, `indices`) are stored as int64 and validated (when
  `check=True`) against structural invariants expected by the Rust core.
- Data is stored as float64 in v0.1.
- Most computational methods call into the Rust core when available; if the
  core is not available, a RuntimeError is raised.
"""

import numpy as np

from .base import SparseMatrix

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None


class CSR(SparseMatrix):
    """Compressed Sparse Row (CSR) matrix.

    Parameters
    ----------
    indptr : array-like of int64, shape (n_rows + 1,)
        Row pointer array. Must be non-decreasing, start at 0, end at `nnz`.
    indices : array-like of int64, shape (nnz,)
        Column indices for each non-zero. Must be strictly increasing within each row.
    data : array-like of float64, shape (nnz,)
        Non-zero values.
    shape : tuple[int, int]
        Matrix shape (n_rows, n_cols).
    dtype : numpy.dtype, optional (default: np.float64)
        Data dtype (v0.1 supports float64 only).
    check : bool, optional (default: True)
        When True, validate structural invariants using the Rust core.

    Notes
    -----
    The constructor may raise if the structure is invalid when `check=True`.

    Examples
    --------
    Construct a small CSR and run basic ops::

        >>> import numpy as np
        >>> from lacuna.sparse import CSR
        >>> indptr = np.array([0, 2, 3])  # 2 rows, 3 nnz
        >>> indices = np.array([0, 2, 1])
        >>> data = np.array([1.0, 3.0, 2.0])
        >>> a = CSR(indptr, indices, data, shape=(2, 3))
        >>> a.nnz
        3
        >>> (a @ np.array([1.0, 0.0, 1.0])).tolist()  # SpMV
        [4.0, 2.0]
        >>> a.sum()
        6.0
    """

    def __init__(self, indptr, indices, data, shape, dtype=np.float64, check=True):
        super().__init__(shape=shape, dtype=dtype)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        if check and _core is not None:
            # Validate by constructing once through core (errors if invalid)
            _ = _core.sum_from_parts(
                self.shape[0], self.shape[1], self.indptr, self.indices, self.data, True
            )
        if _core is not None:
            try:
                self._handle = _core.Csr64(
                    self.shape[0], self.shape[1], self.indptr, self.indices, self.data, False
                )
            except Exception:
                self._handle = None
        else:
            self._handle = None

    @classmethod
    def from_arrays(cls, indptr, indices, data, shape, check=True):
        """Construct a CSR from raw arrays.

        Parameters
        ----------
        indptr, indices, data, shape, check
            See `CSR.__init__`.

        Returns
        -------
        CSR
            New CSR instance.
        """
        return cls(indptr, indices, data, shape, check=check)

    @property
    def nnz(self):
        """Number of stored non-zero entries (int)."""
        return int(self.data.size)

    def _parts(self):
        return (self.shape[0], self.shape[1], self.indptr, self.indices, self.data)

    def __matmul__(self, other):
        """Matrix multiplication: CSR @ vector or CSR @ dense-2D.

        Parameters
        ----------
        other : array_like
            A 1D vector (length = ncols) or a 2D dense matrix of shape (ncols, k).

        Returns
        -------
        numpy.ndarray
            1D result for vector input, 2D result for matrix input.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        ValueError
            If the input shape is incompatible or has ndim not in {1, 2}.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        nrows, ncols, indptr, indices, data = self._parts()
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != ncols:
                raise ValueError("vector length must equal ncols")
            h = getattr(self, "_handle", None)
            if h is not None:
                return h.spmv(arr)
            return _core.spmv_from_parts(nrows, ncols, indptr, indices, data, arr, False)
        elif arr.ndim == 2:
            if arr.shape[0] != ncols:
                raise ValueError("matrix rows must equal ncols")
            h = getattr(self, "_handle", None)
            if h is not None:
                return h.spmm(arr)
            return _core.spmm_from_parts(nrows, ncols, indptr, indices, data, arr, False)
        else:
            raise ValueError("right operand must be 1D or 2D")

    def sum(self, axis=None):
        """Sum elements along the given axis using the Rust core.

        Parameters
        ----------
        axis : {None, 0, 1}, optional
            None for total sum (scalar), 0 for column sums (length = ncols),
            1 for row sums (length = nrows).

        Returns
        -------
        float or numpy.ndarray
            Scalar for total sum, 1D array for axis-specific sums.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        ValueError
            If `axis` is not one of {None, 0, 1}.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is not None:
            if axis is None:
                return float(h.sum())
            if axis == 0:
                return h.col_sums()
            if axis == 1:
                return h.row_sums()
            raise ValueError("axis must be None, 0, or 1")
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
        """Transpose of the matrix (CSR).

        Returns
        -------
        CSR
            Transposed matrix as a new CSR instance.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is not None:
            ti, tj, tv, tr, tc = h.transpose()
            return CSR(ti, tj, tv, (tr, tc), check=False)
        nrows, ncols, indptr, indices, data = self._parts()
        ti, tj, tv, tr, tc = _core.transpose_from_parts(nrows, ncols, indptr, indices, data, False)
        return CSR(ti, tj, tv, (tr, tc), check=False)

    def prune(self, eps):
        """Remove entries with absolute value <= `eps`.

        Parameters
        ----------
        eps : float
            Threshold for pruning.

        Returns
        -------
        CSR
            New CSR with pruned entries.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is not None:
            pi, pj, pv, pr, pc = h.prune(float(eps))
            return CSR(pi, pj, pv, (pr, pc), check=False)
        nrows, ncols, indptr, indices, data = self._parts()
        pi, pj, pv, pr, pc = _core.prune_from_parts(
            nrows, ncols, indptr, indices, data, float(eps), False
        )
        return CSR(pi, pj, pv, (pr, pc), check=False)

    def eliminate_zeros(self):
        """Remove exact zeros from the matrix structure.

        Returns
        -------
        CSR
            New CSR with all zero entries removed.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is not None:
            pi, pj, pv, pr, pc = h.eliminate_zeros()
            return CSR(pi, pj, pv, (pr, pc), check=False)
        nrows, ncols, indptr, indices, data = self._parts()
        pi, pj, pv, pr, pc = _core.eliminate_zeros_from_parts(
            nrows, ncols, indptr, indices, data, False
        )
        return CSR(pi, pj, pv, (pr, pc), check=False)

    def __mul__(self, alpha):
        """Scalar multiplication (right or left): CSR * alpha.

        Parameters
        ----------
        alpha : float
            Scalar multiplier.

        Returns
        -------
        CSR
            New CSR with data scaled by `alpha`.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        alpha = float(alpha)
        h = getattr(self, "_handle", None)
        if h is not None:
            oi, oj, ov, orr, occ = h.mul_scalar(alpha)
            return CSR(oi, oj, ov, (orr, occ), check=False)
        nrows, ncols, indptr, indices, data = self._parts()
        oi, oj, ov, orr, occ = _core.mul_scalar_from_parts(
            nrows, ncols, indptr, indices, data, alpha, False
        )
        return CSR(oi, oj, ov, (orr, occ), check=False)

    __rmul__ = __mul__

    def __add__(self, other):
        """Elementwise addition with another CSR (same shape).

        Parameters
        ----------
        other : CSR
            The right-hand operand.

        Returns
        -------
        CSR
            New CSR representing `self + other`.

        Raises
        ------
        RuntimeError
            If the native core is not available.
        NotImplementedError
            If `other` is not a CSR.
        """
        if isinstance(other, CSR):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.add(hb)
                return CSR(ci, cj, cv, (cr, cc), check=False)
            a = self._parts()
            b = other._parts()
            ci, cj, cv, cr, cc = _core.add_from_parts(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                False,
            )
            return CSR(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, CSR):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.sub(hb)
                return CSR(ci, cj, cv, (cr, cc), check=False)
            a = self._parts()
            b = other._parts()
            ci, cj, cv, cr, cc = _core.sub_from_parts(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                False,
            )
            return CSR(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def multiply(self, other):
        if isinstance(other, CSR):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.hadamard(hb)
                return CSR(ci, cj, cv, (cr, cc), check=False)
            a = self._parts()
            b = other._parts()
            ci, cj, cv, cr, cc = _core.hadamard_from_parts(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                False,
            )
            return CSR(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    # Minimal read-only indexing
    def __getitem__(self, key):
        """Read-only indexing.

        Supported forms
        ---------------
        (i, j) : int, int
            Return the scalar at (i, j).
        (i, :) : int, slice(None)
            Return a dense row as a 1D numpy array.
        (:, j) : slice(None), int
            Return a dense column as a 1D numpy array (slow).

        Raises
        ------
        IndexError
            If indices are out of bounds.
        NotImplementedError
            For advanced indexing.
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                # scalar lookup
                i = int(i)
                j = int(j)
                if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                    raise IndexError("index out of bounds")
                s = int(self.indptr[i])
                e = int(self.indptr[i + 1])
                row_idx = self.indices[s:e]
                pos = np.searchsorted(row_idx, j)
                if pos < row_idx.size and row_idx[pos] == j:
                    return float(self.data[s + pos])
                return 0.0
            if isinstance(i, int) and j == slice(None):
                i = int(i)
                s = int(self.indptr[i])
                e = int(self.indptr[i + 1])
                out = np.zeros((self.shape[1],), dtype=self.data.dtype)
                out[self.indices[s:e]] = self.data[s:e]
                return out
            if i == slice(None) and isinstance(j, int):
                j = int(j)
                # slow column gather
                out = np.zeros((self.shape[0],), dtype=self.data.dtype)
                for r in range(self.shape[0]):
                    s = int(self.indptr[r])
                    e = int(self.indptr[r + 1])
                    row_idx = self.indices[s:e]
                    pos = np.searchsorted(row_idx, j)
                    if pos < row_idx.size and row_idx[pos] == j:
                        out[r] = self.data[s + pos]
                return out
        raise NotImplementedError("advanced indexing is not implemented in v0.1")

    def toarray(self):
        """Materialize the sparse matrix as a dense numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            Dense array of shape `self.shape` with the same dtype as `data`.
        """
        nrows, ncols = self.shape
        out = np.zeros((nrows, ncols), dtype=self.data.dtype)
        # fill row by row
        for i in range(nrows):
            s = int(self.indptr[i])
            e = int(self.indptr[i + 1])
            if s < e:
                out[i, self.indices[s:e]] = self.data[s:e]
        return out

    def astype(self, dtype):
        """Return a copy converted to the given dtype.

        Parameters
        ----------
        dtype : numpy.dtype or str
            Target dtype. v0.1 supports float64 only.

        Returns
        -------
        CSR
            A new CSR with data cast to the requested dtype.

        Raises
        ------
        NotImplementedError
            If `dtype` is not float64 in v0.1.
        """
        dtype = np.dtype(dtype)
        if dtype == np.float64:
            if self.data.dtype == np.float64:
                return CSR(
                    self.indptr.copy(),
                    self.indices.copy(),
                    self.data.copy(),
                    self.shape,
                    dtype=np.float64,
                    check=False,
                )
            else:
                return CSR(
                    self.indptr.copy(),
                    self.indices.copy(),
                    self.data.astype(np.float64, copy=True),
                    self.shape,
                    dtype=np.float64,
                    check=False,
                )
        raise NotImplementedError("astype supports only float64 in v0.1")
