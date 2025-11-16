import numpy as np
from .base import SparseArray

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None


class COO(SparseArray):
    """Coordinate (COO) sparse matrix.

    Parameters
    ----------
    row : array_like of int64
        Row indices for nonzero entries, length ``nnz``.
    col : array_like of int64
        Column indices for nonzero entries, length ``nnz``.
    data : array_like of float64
        Nonzero values, length ``nnz``.
    shape : tuple of int
        Matrix shape ``(nrows, ncols)``.
    dtype : numpy.dtype, optional
        Value dtype, defaults to ``np.float64``.
    check : bool, optional
        If True, validate invariants in the native layer (may be slower).

    Attributes
    ----------
    row, col, data : numpy.ndarray
        Storage arrays for indices and values.
    shape : tuple[int, int]
        Matrix dimensions.
    dtype : numpy.dtype
        Value dtype.
    nnz : int
        Number of stored elements (with duplicates allowed).

    Notes
    -----
    Backed by Rust kernels through ``lacuna._core.Coo64``; operations release the GIL.

    Examples
    --------
    Construct a small COO and run basic ops::

        >>> import numpy as np
        >>> from lacuna.sparse import COO
        >>> row = np.array([0, 1, 1])
        >>> col = np.array([0, 0, 2])
        >>> val = np.array([1.0, 2.0, 3.0])
        >>> a = COO(row, col, val, shape=(2, 3))
        >>> a.nnz
        3
        >>> (a @ np.array([1.0, 0.0, 1.0])).tolist()  # SpMV
        [1.0, 3.0]
        >>> a.sum()
        6.0
    """

    def __init__(self, row, col, data, shape, dtype=np.float64, check=True):
        super().__init__(shape=shape, dtype=dtype)
        self.row = np.asarray(row, dtype=np.int64)
        self.col = np.asarray(col, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        if _core is not None:
            try:
                self._handle = _core.Coo64(
                    self.shape[0], self.shape[1], self.row, self.col, self.data, check
                )
            except Exception:
                self._handle = None
        else:
            self._handle = None

    @classmethod
    def from_arrays(cls, row, col, data, shape, check=True):
        """Construct from index/value arrays.

        Parameters
        ----------
        row, col, data : array_like
            Coordinate indices and values.
        shape : tuple[int, int]
            Matrix shape.
        check : bool, optional
            Validate invariants in the native layer.
        """
        return cls(row, col, data, shape, check=check)

    @property
    def nnz(self):
        """Number of stored values (including duplicates)."""
        return int(self.data.size)

    def __matmul__(self, other):
        """Matrix product with a dense vector or dense 2D array.

        - If ``other`` is 1D, returns ``(nrows,)``.
        - If ``other`` is 2D of shape ``(ncols, k)``, returns ``(nrows, k)``.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != self.shape[1]:
                raise ValueError("vector length must equal ncols")
            h = getattr(self, "_handle", None)
            if h is not None:
                return h.spmv(arr)
            raise RuntimeError("native handle is not available")
        elif arr.ndim == 2:
            if arr.shape[0] != self.shape[1]:
                raise ValueError("matrix rows must equal ncols")
            h = getattr(self, "_handle", None)
            if h is not None:
                return h.spmm(arr)
            raise RuntimeError("native handle is not available")
        else:
            raise ValueError("right operand must be 1D or 2D")

    def sum(self, axis=None):
        """Sum of entries.

        Parameters
        ----------
        axis : {None, 0, 1}, optional
            ``None`` for global sum; ``0`` for column sums; ``1`` for row sums.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        if axis is None:
            return float(h.sum())
        if axis == 0:
            return h.col_sums()
        if axis == 1:
            return h.row_sums()
        raise ValueError("axis must be None, 0, or 1")

    def prune(self, eps):
        """Drop entries with ``abs(value) <= eps``.

        Returns a new :class:`COO`.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        pr, pc, pv, nr, nc = h.prune(float(eps))
        return COO(pr, pc, pv, (nr, nc), check=False)

    def eliminate_zeros(self):
        """Remove explicit zeros. Returns a new :class:`COO`."""
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        pr, pc, pv, nr, nc = h.eliminate_zeros()
        return COO(pr, pc, pv, (nr, nc), check=False)

    def __mul__(self, alpha):
        """Scalar multiplication: returns ``alpha * self`` as :class:`COO`."""
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        alpha = float(alpha)
        rr, cc, vv, nr, nc = h.mul_scalar(alpha)
        return COO(rr, cc, vv, (nr, nc), check=False)

    __rmul__ = __mul__

    def toarray(self):
        """Convert to a dense NumPy ``ndarray`` of shape ``(nrows, ncols)``."""
        nrows, ncols = self.shape
        out = np.zeros((nrows, ncols), dtype=self.data.dtype)
        if self.data.size == 0:
            return out
        # accumulate duplicates
        r = self.row.astype(np.intp, copy=False)
        c = self.col.astype(np.intp, copy=False)
        for k in range(self.data.size):
            out[r[k], c[k]] += self.data[k]
        return out
