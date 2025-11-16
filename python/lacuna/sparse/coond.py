import numpy as np

from .base import SparseArray
from .csc import CSC
from .csr import CSR

try:
    from .. import _core
except Exception:  # pragma: no cover
    _core = None


class COOND(SparseArray):
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
        return cls(shape, indices, data, check=check)

    @property
    def nnz(self):
        return int(self.data.size)

    def _shape_i64(self):
        return np.asarray(self.shape, dtype=np.int64)

    def sum(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        return float(_core.coond_sum_from_parts(self._shape_i64(), self.indices, self.data, False))

    def mean(self):
        if _core is None:
            raise RuntimeError("native core is not available")
        return float(_core.coond_mean_from_parts(self._shape_i64(), self.indices, self.data, False))

    def reduce_sum_axes(self, axes):
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
        if _core is None:
            raise RuntimeError("native core is not available")
        axis = int(axis)
        indptr, cols, vals, nr, nc = _core.coond_mode_to_csr_from_parts(
            self._shape_i64(), self.indices, self.data, axis, False
        )
        return CSR(indptr, cols, vals, shape=(int(nr), int(nc)), check=False)

    def mode_unfold_to_csc(self, axis):
        if _core is None:
            raise RuntimeError("native core is not available")
        axis = int(axis)
        indptr, rows, vals, nr, nc = _core.coond_mode_to_csc_from_parts(
            self._shape_i64(), self.indices, self.data, axis, False
        )
        return CSC(indptr, rows, vals, shape=(int(nr), int(nc)), check=False)

    def axes_unfold_to_csr(self, row_axes):
        if _core is None:
            raise RuntimeError("native core is not available")
        row_axes = np.asarray(row_axes, dtype=np.int64)
        indptr, cols, vals, nr, nc = _core.coond_axes_to_csr_from_parts(
            self._shape_i64(), self.indices, self.data, row_axes, False
        )
        return CSR(indptr, cols, vals, shape=(int(nr), int(nc)), check=False)

    def axes_unfold_to_csc(self, row_axes):
        if _core is None:
            raise RuntimeError("native core is not available")
        row_axes = np.asarray(row_axes, dtype=np.int64)
        indptr, rows, vals, nr, nc = _core.coond_axes_to_csc_from_parts(
            self._shape_i64(), self.indices, self.data, row_axes, False
        )
        return CSC(indptr, rows, vals, shape=(int(nr), int(nc)), check=False)
