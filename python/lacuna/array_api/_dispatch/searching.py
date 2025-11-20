import numpy as np

from ...sparse import COO, CSC, CSR
from .._namespace import _numpy_xp


def _is_sparse(x) -> bool:
    return isinstance(x, (CSR, CSC, COO))


def count_nonzero(x, axis=None, keepdims=False):
    if isinstance(x, CSR):
        nrows, ncols = x.shape
        norm = _normalize_axes_2d(axis)
        if norm is None or norm == (0, 1):
            c = int(x.nnz)
            return np.array(c).reshape((1, 1)) if keepdims else c
        if norm == ():
            raise NotImplementedError(
                "count_nonzero with axis=() (no reduction) not implemented for sparse inputs"
            )
        if norm == (1,):
            counts = np.diff(x.indptr).astype(np.int64, copy=False)
            return counts.reshape((nrows, 1)) if keepdims else counts
        if norm == (0,):
            counts = np.bincount(x.indices, minlength=ncols).astype(np.int64, copy=False)
            return counts.reshape((1, ncols)) if keepdims else counts
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, CSC):
        nrows, ncols = x.shape
        norm = _normalize_axes_2d(axis)
        if norm is None or norm == (0, 1):
            c = int(x.nnz)
            return np.array(c).reshape((1, 1)) if keepdims else c
        if norm == ():
            raise NotImplementedError(
                "count_nonzero with axis=() (no reduction) not implemented for sparse inputs"
            )
        if norm == (0,):
            counts = np.diff(x.indptr).astype(np.int64, copy=False)
            return counts.reshape((1, ncols)) if keepdims else counts
        if norm == (1,):
            counts = np.bincount(x.indices, minlength=nrows).astype(np.int64, copy=False)
            return counts.reshape((nrows, 1)) if keepdims else counts
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COO):
        nrows, ncols = x.shape
        norm = _normalize_axes_2d(axis)
        if norm is None or norm == (0, 1):
            c = int(x.nnz)
            return np.array(c).reshape((1, 1)) if keepdims else c
        if norm == ():
            raise NotImplementedError(
                "count_nonzero with axis=() (no reduction) not implemented for sparse inputs"
            )
        if norm == (1,):
            counts = np.bincount(x.row, minlength=nrows).astype(np.int64, copy=False)
            return counts.reshape((nrows, 1)) if keepdims else counts
        if norm == (0,):
            counts = np.bincount(x.col, minlength=ncols).astype(np.int64, copy=False)
            return counts.reshape((1, ncols)) if keepdims else counts
        raise ValueError("invalid axis for 2D input")

    xp = _numpy_xp()
    return xp.count_nonzero(x, axis=axis, keepdims=keepdims)


def _normalize_axes_2d(axis):
    if axis is None:
        return None
    if isinstance(axis, (list, tuple)):
        if len(axis) == 0:
            return tuple()
        axes = tuple(int(a) for a in axis)
    else:
        axes = (int(axis),)
    norm = []
    for a in axes:
        if a < 0:
            a = 2 + a
        if a not in (0, 1):
            raise ValueError("axis must be in {-2,-1,0,1} or tuple thereof for 2D inputs")
        if a not in norm:
            norm.append(a)
    return tuple(sorted(norm))
