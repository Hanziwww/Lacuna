import numpy as np

from .base import SparseMatrix

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None


class CSC(SparseMatrix):
    """Compressed Sparse Column (CSC) matrix.

    Parameters
    ----------
    indptr : array_like of int64, shape ``(ncols + 1,)``
        Column pointer array.
    indices : array_like of int64, shape ``(nnz,)``
        Row indices of nonzero values.
    data : array_like of float64, shape ``(nnz,)``
        Nonzero values.
    shape : tuple of int
        Matrix shape ``(nrows, ncols)``.
    dtype : numpy.dtype, optional
        Value dtype, defaults to ``np.float64``.
    check : bool, optional
        If True, validate invariants in the native layer (may be slower).

    Attributes
    ----------
    indptr, indices, data : numpy.ndarray
        Storage arrays for CSC structure and values.
    shape : tuple[int, int]
        Matrix dimensions.
    dtype : numpy.dtype
        Value dtype.
    nnz : int
        Number of stored elements.

    Notes
    -----
    Backed by Rust kernels through ``lacuna._core.Csc64``; operations release the GIL.

    Examples
    --------
    Construct a small CSC and run basic ops::

        >>> import numpy as np
        >>> from lacuna.sparse import CSC
        >>> indptr = np.array([0, 1, 2, 3])
        >>> indices = np.array([0, 1, 1])
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> a = CSC(indptr, indices, data, shape=(2, 3))
        >>> a.nnz
        3
        >>> (a @ np.array([1.0, 0.0, 1.0])).tolist()  # SpMV
        [1.0, 3.0]
        >>> a.sum()
        6.0
    """

    def __init__(self, indptr, indices, data, shape, dtype=np.float64, check=True):
        super().__init__(shape=shape, dtype=dtype)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        if _core is not None:
            try:
                self._handle = _core.Csc64(
                    self.shape[0], self.shape[1], self.indptr, self.indices, self.data, check
                )
            except Exception:
                self._handle = None
        else:
            self._handle = None

    @classmethod
    def from_arrays(cls, indptr, indices, data, shape, check=True):
        """Construct from CSC arrays.

        Parameters
        ----------
        indptr, indices, data : array_like
            CSC structure and values.
        shape : tuple[int, int]
            Matrix shape.
        check : bool, optional
            Validate invariants in the native layer.
        """
        return cls(indptr, indices, data, shape, check=check)

    @property
    def nnz(self):
        """Number of stored values."""
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
        """Drop entries with ``abs(value) <= eps``. Returns a new :class:`CSC`."""
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        pi, pj, pv, pr, pc = h.prune(float(eps))
        return CSC(pi, pj, pv, (pr, pc), check=False)

    def eliminate_zeros(self):
        """Remove explicit zeros. Returns a new :class:`CSC`."""
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        pi, pj, pv, pr, pc = h.eliminate_zeros()
        return CSC(pi, pj, pv, (pr, pc), check=False)

    def __mul__(self, alpha):
        """Scalar multiplication: returns ``alpha * self`` as :class:`CSC`."""
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is None:
            raise RuntimeError("native handle is not available")
        alpha = float(alpha)
        oi, oj, ov, orr, occ = h.mul_scalar(alpha)
        return CSC(oi, oj, ov, (orr, occ), check=False)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __abs__(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        oi, oj, ov, orr, occ = _core.abs_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, False
        )
        return CSC(oi, oj, ov, (orr, occ), check=False)

    def sign(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        oi, oj, ov, orr, occ = _core.sign_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, False
        )
        return CSC(oi, oj, ov, (orr, occ), check=False)

    def __add__(self, other):
        """Elementwise addition with another :class:`CSC`."""
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.add(hb)
                return CSC(ci, cj, cv, (cr, cc), check=False)
            raise RuntimeError("native handle is not available")
        return NotImplemented

    def __sub__(self, other):
        """Elementwise subtraction with another :class:`CSC`."""
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.sub(hb)
                return CSC(ci, cj, cv, (cr, cc), check=False)
            raise RuntimeError("native handle is not available")
        return NotImplemented

    def multiply(self, other):
        """Hadamard (elementwise) product with another :class:`CSC`."""
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.hadamard(hb)
                return CSC(ci, cj, cv, (cr, cc), check=False)
            raise RuntimeError("native handle is not available")
        return NotImplemented

    def divide(self, other):
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.div(hb)
                return CSC(ci, cj, cv, (cr, cc), check=False)
            # from_parts fallback
            ci, cj, cv, cr, cc = _core.div_csc_from_parts(
                self.shape[0],
                self.shape[1],
                self.indptr,
                self.indices,
                self.data,
                other.shape[0],
                other.shape[1],
                other.indptr,
                other.indices,
                other.data,
                False,
            )
            return CSC(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def __truediv__(self, alpha):
        alpha = float(alpha)
        if alpha == 0.0:
            raise ZeroDivisionError("division by zero")
        return self * (1.0 / alpha)

    def remainder(self, other):
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ci, cj, cv, cr, cc = _core.remainder_csc_from_parts(
                self.shape[0],
                self.shape[1],
                self.indptr,
                self.indices,
                self.data,
                other.shape[0],
                other.shape[1],
                other.indptr,
                other.indices,
                other.data,
                False,
            )
            return CSC(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def power(self, other):
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ci, cj, cv, cr, cc = _core.pow_csc_from_parts(
                self.shape[0],
                self.shape[1],
                self.indptr,
                self.indices,
                self.data,
                other.shape[0],
                other.shape[1],
                other.indptr,
                other.indices,
                other.data,
                False,
            )
            return CSC(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def floor_divide(self, other):
        if isinstance(other, CSC):
            if _core is None:
                raise RuntimeError("native core is not available")
            ha = getattr(self, "_handle", None)
            hb = getattr(other, "_handle", None)
            if ha is not None and hb is not None:
                ci, cj, cv, cr, cc = ha.floordiv(hb)
                return CSC(ci, cj, cv, (cr, cc), check=False)
            # from_parts fallback
            ci, cj, cv, cr, cc = _core.floordiv_csc_from_parts(
                self.shape[0],
                self.shape[1],
                self.indptr,
                self.indices,
                self.data,
                other.shape[0],
                other.shape[1],
                other.indptr,
                other.indices,
                other.data,
                False,
            )
            return CSC(ci, cj, cv, (cr, cc), check=False)
        return NotImplemented

    def __floordiv__(self, alpha):
        alpha = float(alpha)
        if alpha == 0.0:
            raise ZeroDivisionError("division by zero")
        if _core is None:
            raise RuntimeError("native core is not available")
        # from_parts call
        oi, oj, ov, orr, occ = _core.floordiv_scalar_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, alpha, False
        )
        return CSC(oi, oj, ov, (orr, occ), check=False)

    def __mod__(self, alpha):
        alpha = float(alpha)
        if alpha == 0.0:
            raise ZeroDivisionError("integer modulo by zero")
        if _core is None:
            raise RuntimeError("native core is not available")
        oi, oj, ov, orr, occ = _core.remainder_scalar_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, alpha, False
        )
        return CSC(oi, oj, ov, (orr, occ), check=False)

    def __pow__(self, alpha):
        alpha = float(alpha)
        if _core is None:
            raise RuntimeError("native core is not available")
        oi, oj, ov, orr, occ = _core.pow_scalar_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, alpha, False
        )
        return CSC(oi, oj, ov, (orr, occ), check=False)

    def toarray(self):
        """Convert to a dense NumPy ``ndarray`` of shape ``(nrows, ncols)``."""
        nrows, ncols = self.shape
        out = np.zeros((nrows, ncols), dtype=self.data.dtype)
        for j in range(ncols):
            s = int(self.indptr[j])
            e = int(self.indptr[j + 1])
            if s < e:
                out[self.indices[s:e], j] = self.data[s:e]
        return out

    @property
    def T(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        h = getattr(self, "_handle", None)
        if h is not None:
            ti, tj, tv, tr, tc = h.transpose()
            return CSC(ti, tj, tv, (tr, tc), check=False)
        # fallback to from_parts path
        ti, tj, tv, tr, tc = _core.transpose_csc_from_parts(
            self.shape[0], self.shape[1], self.indptr, self.indices, self.data, False
        )
        return CSC(ti, tj, tv, (tr, tc), check=False)

    def __repr__(self):
        return f"CSC(shape={self.shape}, nnz={self.nnz}, dtype={self.data.dtype.name})"

    def __str__(self):
        return self.__repr__()
