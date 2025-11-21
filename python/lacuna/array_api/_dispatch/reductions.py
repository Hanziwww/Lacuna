import numpy as np

from ...sparse import COO, COOND, CSC, CSR
from .._namespace import _numpy_xp

try:
    from ... import _core
except Exception:  # pragma: no cover
    _core = None


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
    orig_ndim = x.ndim
    rem_axes = [i for i in range(orig_ndim) if i not in reduced_axes]
    nnz = reduced.nnz
    rem_ndim = len(rem_axes)
    if rem_ndim == 0:
        new_shape = tuple(1 for _ in range(orig_ndim))
        return COOND(new_shape, np.zeros((0,), dtype=np.int64), reduced.data, check=False)
    idx = reduced.indices.reshape(nnz, rem_ndim)
    new_idx = np.zeros((nnz, orig_ndim), dtype=np.int64)
    new_idx[:, rem_axes] = idx
    new_indices_flat = new_idx.reshape(-1)
    reduced_shape_iter = iter(reduced.shape)
    new_shape_list = []
    for d in range(orig_ndim):
        if d in reduced_axes:
            new_shape_list.append(1)
        else:
            new_shape_list.append(int(next(reduced_shape_iter)))
    return COOND(tuple(new_shape_list), new_indices_flat, reduced.data, check=False)


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
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        norm = _normalize_axes_nd(axis, x.ndim)
        if norm is None:
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


def prod(x, axis=None, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        if _core is None:
            raise RuntimeError("native core is not available")
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            if isinstance(x, CSR):
                v = _core.prod_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.prod_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.prod_coo_from_parts(x.shape[0], x.shape[1], x.row, x.col, x.data, False)
            return np.array(v).reshape((1, 1)) if keepdims else float(v)
        if norm == ():
            raise NotImplementedError(
                "prod with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            if isinstance(x, CSR):
                v = _core.col_prods_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.col_prods_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.col_prods_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((1, ncols)) if keepdims else v
        if norm == (1,):
            if isinstance(x, CSR):
                v = _core.row_prods_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.row_prods_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.row_prods_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((nrows, 1)) if keepdims else v
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        if axis is None:
            v = _core.coond_prod_from_parts(x.shape, x.indices, x.data, False)
            if keepdims:
                return np.array(v).reshape(tuple(1 for _ in range(x.ndim)))
            return v
        if axis == ():
            raise NotImplementedError(
                "prod with axis=() (no reduction) is not implemented for COOND"
            )
        raise NotImplementedError("prod with axis for COOND is not yet implemented")

    xp = _numpy_xp()
    return xp.prod(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, correction=0.0, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        if _core is None:
            raise RuntimeError("native core is not available")
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            if isinstance(x, CSR):
                v = _core.var_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.var_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.var_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            return np.array(v).reshape((1, 1)) if keepdims else float(v)
        if norm == ():
            raise NotImplementedError(
                "var with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            if isinstance(x, CSR):
                v = _core.col_vars_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.col_vars_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.col_vars_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            v = np.asarray(v)
            return v.reshape((1, ncols)) if keepdims else v
        if norm == (1,):
            if isinstance(x, CSR):
                v = _core.row_vars_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.row_vars_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.row_vars_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            v = np.asarray(v)
            return v.reshape((nrows, 1)) if keepdims else v
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        if axis is None:
            v = _core.coond_var_from_parts(x.shape, x.indices, x.data, float(correction), False)
            if keepdims:
                return np.array(v).reshape(tuple(1 for _ in range(x.ndim)))
            return v
        if axis == ():
            raise NotImplementedError(
                "var with axis=() (no reduction) is not implemented for COOND"
            )
        raise NotImplementedError("var with axis for COOND is not yet implemented")

    xp = _numpy_xp()
    return xp.var(x, axis=axis, ddof=correction, keepdims=keepdims)


def std(x, axis=None, correction=0.0, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        if _core is None:
            raise RuntimeError("native core is not available")
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            if isinstance(x, CSR):
                v = _core.std_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.std_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.std_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            return np.array(v).reshape((1, 1)) if keepdims else float(v)
        if norm == ():
            raise NotImplementedError(
                "std with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            if isinstance(x, CSR):
                v = _core.col_stds_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.col_stds_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.col_stds_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            v = np.asarray(v)
            return v.reshape((1, ncols)) if keepdims else v
        if norm == (1,):
            if isinstance(x, CSR):
                v = _core.row_stds_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            elif isinstance(x, CSC):
                v = _core.row_stds_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, float(correction), False
                )
            else:
                v = _core.row_stds_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, float(correction), False
                )
            v = np.asarray(v)
            return v.reshape((nrows, 1)) if keepdims else v
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        if axis is None:
            v = _core.coond_std_from_parts(x.shape, x.indices, x.data, float(correction), False)
            if keepdims:
                return np.array(v).reshape(tuple(1 for _ in range(x.ndim)))
            return v
        if axis == ():
            raise NotImplementedError(
                "std with axis=() (no reduction) is not implemented for COOND"
            )
        raise NotImplementedError("std with axis for COOND is not yet implemented")

    xp = _numpy_xp()
    return xp.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        if _core is None:
            raise RuntimeError("native core is not available")
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            if isinstance(x, CSR):
                m = _core.min_from_parts(x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False)
            elif isinstance(x, CSC):
                m = _core.min_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                m = _core.min_coo_from_parts(x.shape[0], x.shape[1], x.row, x.col, x.data, False)
            return np.array(m).reshape((1, 1)) if keepdims else float(m)
        if norm == ():
            raise NotImplementedError(
                "min with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            if isinstance(x, CSR):
                v = _core.col_mins_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.col_mins_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.col_mins_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((1, ncols)) if keepdims else v
        if norm == (1,):
            if isinstance(x, CSR):
                v = _core.row_mins_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.row_mins_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.row_mins_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((nrows, 1)) if keepdims else v
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        raise NotImplementedError("min for COO/COOND is not yet implemented")

    xp = _numpy_xp()
    return xp.min(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    if isinstance(x, (CSR, CSC, COO)):
        if _core is None:
            raise RuntimeError("native core is not available")
        norm = _normalize_axes_2d(axis)
        nrows, ncols = x.shape
        if norm is None or norm == (0, 1):
            if isinstance(x, CSR):
                m = _core.max_from_parts(x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False)
            elif isinstance(x, CSC):
                m = _core.max_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                m = _core.max_coo_from_parts(x.shape[0], x.shape[1], x.row, x.col, x.data, False)
            return np.array(m).reshape((1, 1)) if keepdims else float(m)
        if norm == ():
            raise NotImplementedError(
                "max with axis=() (no reduction) is not implemented for sparse inputs"
            )
        if norm == (0,):
            if isinstance(x, CSR):
                v = _core.col_maxs_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.col_maxs_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.col_maxs_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((1, ncols)) if keepdims else v
        if norm == (1,):
            if isinstance(x, CSR):
                v = _core.row_maxs_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            elif isinstance(x, CSC):
                v = _core.row_maxs_csc_from_parts(
                    x.shape[0], x.shape[1], x.indptr, x.indices, x.data, False
                )
            else:
                v = _core.row_maxs_coo_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, False
                )
            v = np.asarray(v)
            return v.reshape((nrows, 1)) if keepdims else v
        raise ValueError("invalid axis for 2D input")

    if isinstance(x, COOND):
        raise NotImplementedError("max for COO/COOND is not yet implemented")

    xp = _numpy_xp()
    return xp.max(x, axis=axis, keepdims=keepdims)
