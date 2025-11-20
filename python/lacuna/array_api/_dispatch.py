import numpy as np

from ..sparse import COO, COOND, CSC, CSR
from ._namespace import _numpy_xp


def _is_sparse(x) -> bool:
    return isinstance(x, (CSR, CSC, COO, COOND))


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


def _normalize_axes_nd(axis, ndim: int):
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
            a = ndim + a
        if not (0 <= a < ndim):
            raise ValueError("axis out of range")
        if a not in norm:
            norm.append(a)
    return tuple(sorted(norm))


def _coond_with_keepdims(x: COOND, reduced: COOND, reduced_axes: tuple[int, ...]) -> COOND:
    # Reinsert size-1 dimensions at reduced_axes positions; adjust indices accordingly.
    orig_ndim = x.ndim
    rem_axes = [i for i in range(orig_ndim) if i not in reduced_axes]
    nnz = reduced.nnz
    rem_ndim = len(rem_axes)
    if rem_ndim == 0:
        # reduced is scalar-like sparse (should have no indices); just build shape of all ones
        new_shape = tuple(1 for _ in range(orig_ndim))
        return COOND(new_shape, np.zeros((0,), dtype=np.int64), reduced.data, check=False)
    idx = reduced.indices.reshape(nnz, rem_ndim)
    new_idx = np.zeros((nnz, orig_ndim), dtype=np.int64)
    new_idx[:, rem_axes] = idx
    new_indices_flat = new_idx.reshape(-1)
    # Build new shape: 1 at reduced axes, keep dims of remaining axes in original order
    reduced_shape_iter = iter(reduced.shape)
    new_shape_list = []
    for d in range(orig_ndim):
        if d in reduced_axes:
            new_shape_list.append(1)
        else:
            new_shape_list.append(int(next(reduced_shape_iter)))
    return COOND(tuple(new_shape_list), new_indices_flat, reduced.data, check=False)


# ===== Linalg =====
def matmul(x, y):
    if _is_sparse(x) or _is_sparse(y):
        # Sparse-sparse matmul not supported yet (avoid densify)
        if _is_sparse(x) and _is_sparse(y):
            raise NotImplementedError(
                "matmul for sparse@sparse is not implemented in lacuna.array_api"
            )

        # Sparse @ dense (supported for CSR/CSC/COO)
        if isinstance(x, (CSR, CSC, COO)) and isinstance(y, np.ndarray):
            if y.ndim <= 2:
                return x @ y
            # batched right operand: (..., k, n), with x shape (m, k)
            if y.shape[-2] != x.shape[1]:
                raise ValueError("Inner dimensions must match for matmul")
            batch_shape = y.shape[:-2]
            b_flat = int(np.prod(batch_shape))
            k = x.shape[1]
            n = y.shape[-1]
            y_bkn = y.reshape(b_flat, k, n)  # (B,k,n)
            y_kbn = np.transpose(y_bkn, (1, 0, 2))  # (k,B,n)
            y2 = np.ascontiguousarray(y_kbn.reshape(k, b_flat * n))
            out2 = x @ y2  # (m, B*n)
            out_mbn = out2.reshape(x.shape[0], b_flat, n)  # (m,B,n)
            out_bmn = np.transpose(out_mbn, (1, 0, 2))  # (B,m,n)
            out = out_bmn.reshape(*batch_shape, x.shape[0], n)
            return out

        # Dense @ sparse (not yet supported)
        if isinstance(y, (CSR, CSC)) and isinstance(x, np.ndarray):
            raise NotImplementedError(
                "dense @ sparse matmul is not implemented in lacuna.array_api"
            )

        # Other sparse types (e.g., COO/COOND) are not supported for matmul
        raise NotImplementedError(
            "matmul is not implemented for these sparse inputs in lacuna.array_api"
        )

    xp = _numpy_xp()
    return xp.matmul(x, y)


def matrix_transpose(x):
    if isinstance(x, (CSR, CSC, COO)):
        return x.T
    xp = _numpy_xp()
    return xp.matrix_transpose(x)


# ===== Reductions =====
def sum(x, axis=None, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            s = x.sum(None)
            return np.array(s).reshape((1, 1)) if keepdims else s
        if norm == ():
            raise NotImplementedError(
                "sum with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            s = np.asarray(x.sum(0))
            return s.reshape((1, ncols)) if keepdims else s
        if norm == (1,):
            s = np.asarray(x.sum(1))
            return s.reshape((nrows, 1)) if keepdims else s
        # unreachable for 2D
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        norm = _normalize_axes_nd(axis, x.ndim)
        if norm is None:
            # global reduction
            s = x.sum()
            if keepdims:
                return np.array(s).reshape(tuple(1 for _ in range(x.ndim)))
            return s
        if norm == ():
            raise NotImplementedError(
                "sum with axis=() (no reduction) is not implemented for COOND"
            )
        reduced = x.reduce_sum_axes(list(norm))
        if keepdims:
            return _coond_with_keepdims(x, reduced, norm)
        return reduced

    xp = _numpy_xp()
    return xp.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            m = float(x.sum(None)) / float(nrows * ncols)
            return np.array(m).reshape((1, 1)) if keepdims else m
        if norm == ():
            raise NotImplementedError(
                "mean with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            s = np.asarray(x.sum(0)) / float(nrows)
            return s.reshape((1, ncols)) if keepdims else s
        if norm == (1,):
            s = np.asarray(x.sum(1)) / float(ncols)
            return s.reshape((nrows, 1)) if keepdims else s
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        norm = _normalize_axes_nd(axis, x.ndim)
        if norm is None:
            m = x.mean()
            if keepdims:
                return np.array(m).reshape(tuple(1 for _ in range(x.ndim)))
            return m
        if norm == ():
            raise NotImplementedError(
                "mean with axis=() (no reduction) is not implemented for COOND"
            )
        reduced = x.reduce_mean_axes(list(norm))
        if keepdims:
            return _coond_with_keepdims(x, reduced, norm)
        return reduced

    xp = _numpy_xp()
    return xp.mean(x, axis=axis, keepdims=keepdims)


# ===== Searching =====
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


# ===== Elementwise =====
def add(x, y):
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x + y
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x + y
    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "add for sparse inputs is only implemented for CSR+CSR and CSC+CSC"
        )
    xp = _numpy_xp()
    return xp.add(x, y)


def subtract(x, y):
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x - y
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x - y
    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "subtract for sparse inputs is only implemented for CSR-CSR and CSC-CSC"
        )
    xp = _numpy_xp()
    return xp.subtract(x, y)


def multiply(x, y):
    # scalar * sparse or sparse * scalar
    if isinstance(x, (int, float)) and isinstance(y, (CSR, CSC, COO)):
        return y * float(x)
    if isinstance(y, (int, float)) and isinstance(x, (CSR, CSC, COO)):
        return x * float(y)

    # sparse * sparse (Hadamard)
    if isinstance(x, COOND) and isinstance(y, COOND):
        return x.hadamard_broadcast(y)
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x.multiply(y)
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x.multiply(y)

    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "multiply for sparse inputs is only implemented for scalar*sparse and CSR*CSR/CSC*CSC"
        )

    xp = _numpy_xp()
    return xp.multiply(x, y)


# ===== Manipulation =====
def permute_dims(x, axes):
    if isinstance(x, COOND):
        return x.permute_axes(axes)
    if _is_sparse(x):
        raise NotImplementedError(
            "permute_dims is only implemented for COOND sparse arrays in lacuna.array_api"
        )
    xp = _numpy_xp()
    return xp.permute_dims(x, axes)


def reshape(x, newshape):
    if isinstance(x, COOND):
        return x.reshape(newshape)
    if _is_sparse(x):
        raise NotImplementedError(
            "reshape is only implemented for COOND sparse arrays in lacuna.array_api"
        )
    xp = _numpy_xp()
    return xp.reshape(x, newshape)
