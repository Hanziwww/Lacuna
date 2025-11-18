"""N-dimensional COO sparse array (COOND) backed by a Rust core.

This module exposes the `COOND` class, a Pythonic façade over ND-COO kernels
implemented in Rust and bound via PyO3. Indices are stored as int64 and
flattened to length ``nnz * ndim`` (or accepted as a 2D ``(nnz, ndim)`` array),
and values are float64 in v0.1.

Notes
-----
- Operations release the GIL and call from-parts bindings in ``lacuna._core``.
- Duplicates in indices are allowed; aggregation is performed by operations
  (e.g., reductions or toarray materialization).
- When ``check=True``, basic invariants are validated via the native layer.

Contents
--------
- `COOND`: construction, `sum`/`mean`, axis reductions/permute, `reshape`,
  broadcasting Hadamard, unfolding to CSR/CSC.
"""

import numpy as np

from .base import SparseArray
from .csc import CSC
from .csr import CSR

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None


class COOND(SparseArray):
    """N-dimensional sparse array in COO format.

    Parameters
    ----------
    shape : tuple[int, ...]
        Overall tensor shape of length ``ndim``.
    indices : array_like of int64, shape ``(nnz * ndim,)`` or ``(nnz, ndim)``
        Concatenated per-nnz indices or a 2D array; values must be within
        bounds of ``shape``. Duplicates are allowed (aggregated by ops).
    data : array_like of float64, shape ``(nnz,)``
        Nonzero values.
    dtype : numpy.dtype, optional
        Value dtype, defaults to ``np.float64``.
    check : bool, optional
        If True, validate invariants in the native layer where applicable.

    Attributes
    ----------
    shape : tuple[int, ...]
        Tensor dimensions.
    ndim : int
        Number of dimensions (``len(shape)``).
    indices : numpy.ndarray (int64)
        Flattened indices of length ``nnz * ndim``.
    data : numpy.ndarray (float64)
        Nonzero values of length ``nnz``.
    nnz : int
        Number of stored elements.

    Notes
    -----
    Backed by Rust ND-COO kernels via ``lacuna._core``. Operations release the GIL.
    Key operations include ``sum``, ``mean``, axis reductions/permutations,
    ``reshape``, broadcasting Hadamard multiply, and unfolding to CSR/CSC.

    Examples
    --------
    Construct a small 3D tensor and run basic ops::

        >>> import numpy as np
        >>> from lacuna.sparse import COOND
        >>> shape = (2, 3, 4)
        >>> # 2 nonzeros at positions (0,1,2) and (1,2,3)
        >>> idx = np.array([
        ...     0, 1, 2,
        ...     1, 2, 3,
        ... ], dtype=np.int64)
        >>> val = np.array([1.0, 3.0])
        >>> a = COOND(shape, idx, val)
        >>> a.nnz
        2
        >>> a.sum()
        4.0
        >>> a.reduce_sum_axes([2]).shape  # sum over last axis
        (2, 3)
        >>> a.permute_axes([2, 1, 0]).shape
        (4, 3, 2)
        >>> b = a.reshape((3, 2, 4))
        >>> (a.hadamard_broadcast(b)).nnz
        2
        >>> a.mode_unfold_to_csr(axis=0).shape
        (2, 12)
    """

    def __init__(self, shape, indices, data, dtype=np.float64, check=True):
        super().__init__(shape=shape, dtype=dtype)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        if self.ndim < 1:
            raise ValueError("shape must have ndim >= 1")
        if self.indices.size % self.ndim != 0:
            raise ValueError("indices length must be a multiple of ndim")
        if check and _core is None:
            raise RuntimeError("native core is not available")
        # No persistent native handle for ND; we call _core from-parts APIs per op.

    @classmethod
    def from_arrays(cls, shape, indices, data, check=True):
        """Construct COOND from raw arrays.

        Parameters
        ----------
        shape : tuple[int, ...]
            Tensor shape.
        indices : array_like of int64
            Flattened or 2D indices (nnz x ndim).
        data : array_like of float64
            Nonzero values.
        check : bool, optional
            Validate invariants via native layer when possible.
        """
        return cls(shape, indices, data, check=check)

    @property
    def nnz(self):
        """Number of stored values."""
        return int(self.data.size)

    def _shape_i64(self):
        return np.asarray(self.shape, dtype=np.int64)

    def sum(self):
        """Sum of all entries.

        Returns
        -------
        float
            Total sum of nonzero values.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        return float(_core.coond_sum_from_parts(self._shape_i64(), self.indices, self.data, False))

    def mean(self):
        """Mean of all entries (sum / total elements implied by shape)."""
        if _core is None:
            raise RuntimeError("native core is not available")
        return float(_core.coond_mean_from_parts(self._shape_i64(), self.indices, self.data, False))

    def reduce_sum_axes(self, axes):
        """Sum over specified axes.

        Parameters
        ----------
        axes : array_like of int
            Axes to reduce. Order-insensitive, must be valid axes of ``shape``.

        Returns
        -------
        COOND
            A new tensor with the given axes summed out.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        axes = np.asarray(axes, dtype=np.int64)
        nshape, nidx, ndata = _core.coond_reduce_sum_axes_from_parts(
            self._shape_i64(), self.indices, self.data, axes, False
        )
        return COOND(
            tuple(int(x) for x in np.asarray(nshape, dtype=np.int64)), nidx, ndata, check=False
        )

    def permute_axes(self, perm):
        """Permute axes by the given permutation.

        Parameters
        ----------
        perm : array_like of int
            A permutation of ``range(ndim)``.

        Returns
        -------
        COOND
            Tensor with permuted axes.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        perm = np.asarray(perm, dtype=np.int64)
        nshape, nidx, ndata = _core.coond_permute_axes_from_parts(
            self._shape_i64(), self.indices, self.data, perm, False
        )
        return COOND(
            tuple(int(x) for x in np.asarray(nshape, dtype=np.int64)), nidx, ndata, check=False
        )

    def reduce_mean_axes(self, axes):
        """Mean over specified axes.

        Parameters
        ----------
        axes : array_like of int
            Axes to average over.

        Returns
        -------
        COOND
            A new tensor with the given axes averaged out.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        axes = np.asarray(axes, dtype=np.int64)
        nshape, nidx, ndata = _core.coond_reduce_mean_axes_from_parts(
            self._shape_i64(), self.indices, self.data, axes, False
        )
        return COOND(
            tuple(int(x) for x in np.asarray(nshape, dtype=np.int64)), nidx, ndata, check=False
        )

    def reshape(self, new_shape):
        """Return a new tensor with the same data but a different shape.

        Parameters
        ----------
        new_shape : tuple[int, ...]
            Target shape; must be compatible with ``shape``.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        new_shape = np.asarray(tuple(int(x) for x in new_shape), dtype=np.int64)
        nshape, nidx, ndata = _core.coond_reshape_from_parts(
            self._shape_i64(), self.indices, self.data, new_shape, False
        )
        return COOND(
            tuple(int(x) for x in np.asarray(nshape, dtype=np.int64)), nidx, ndata, check=False
        )

    def hadamard_broadcast(self, other):
        """Elementwise product with broadcasting against another COOND.

        Parameters
        ----------
        other : COOND
            Right-hand operand.

        Returns
        -------
        COOND
            Result of broadcasting Hadamard multiplication.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        if not isinstance(other, COOND):
            raise TypeError("other must be COOND")
        oshape, oidx, odata = _core.coond_hadamard_broadcast_from_parts(
            self._shape_i64(),
            self.indices,
            self.data,
            np.asarray(other.shape, dtype=np.int64),
            other.indices,
            other.data,
            False,
        )
        return COOND(
            tuple(int(x) for x in np.asarray(oshape, dtype=np.int64)), oidx, odata, check=False
        )

    def mode_unfold_to_csr(self, axis):
        """Unfold along a single mode into a 2D CSR matrix.

        Parameters
        ----------
        axis : int
            Mode (axis) to place along the rows; remaining modes are flattened as columns.

        Returns
        -------
        CSR
            A CSR matrix of shape ``(shape[axis], prod(shape[~axis]))``.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        axis = int(axis)
        indptr, cols, vals, nr, nc = _core.coond_mode_to_csr_from_parts(
            self._shape_i64(), self.indices, self.data, axis, False
        )
        return CSR(indptr, cols, vals, shape=(int(nr), int(nc)), check=False)

    def mode_unfold_to_csc(self, axis):
        """Unfold along a single mode into a 2D CSC matrix.

        Parameters
        ----------
        axis : int
            Mode (axis) to place along the columns; remaining modes are flattened as rows.

        Returns
        -------
        CSC
            A CSC matrix of shape ``(prod(shape[~axis]), shape[axis])``.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        axis = int(axis)
        indptr, rows, vals, nr, nc = _core.coond_mode_to_csc_from_parts(
            self._shape_i64(), self.indices, self.data, axis, False
        )
        return CSC(indptr, rows, vals, shape=(int(nr), int(nc)), check=False)

    def axes_unfold_to_csr(self, row_axes):
        """Unfold by grouping selected axes as CSR rows and the rest as columns.

        Parameters
        ----------
        row_axes : array_like of int
            Axes that form the row index in the unfolded matrix.

        Returns
        -------
        CSR
            A CSR matrix.
        """
        if _core is None:
            raise RuntimeError("native core is not available")
        row_axes = np.asarray(row_axes, dtype=np.int64)
        indptr, cols, vals, nr, nc = _core.coond_axes_to_csr_from_parts(
            self._shape_i64(), self.indices, self.data, row_axes, False
        )
        return CSR(indptr, cols, vals, shape=(int(nr), int(nc)), check=False)

    def axes_unfold_to_csc(self, row_axes):
        """Unfold by grouping selected axes as CSC rows (the rest become columns)."""
        if _core is None:
            raise RuntimeError("native core is not available")
        row_axes = np.asarray(row_axes, dtype=np.int64)
        indptr, rows, vals, nr, nc = _core.coond_axes_to_csc_from_parts(
            self._shape_i64(), self.indices, self.data, row_axes, False
        )
        return CSC(indptr, rows, vals, shape=(int(nr), int(nc)), check=False)

    def __repr__(self):
        return f"COOND(shape={self.shape}, nnz={self.nnz}, dtype={self.data.dtype.name})"

    def __str__(self):
        return self.__repr__()
