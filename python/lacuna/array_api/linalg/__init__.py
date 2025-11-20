import numpy as np

from ...sparse import COO, COOND, CSC, CSR
from .. import _namespace as _ns
from .._namespace import _numpy_xp

try:
    from ... import _core
except Exception:  # pragma: no cover
    _core = None


def _is_sparse(x) -> bool:
    return isinstance(x, (CSR, CSC, COO, COOND))


def matmul(x, y):
    if _is_sparse(x) or _is_sparse(y):
        if _is_sparse(x) and _is_sparse(y):
            raise NotImplementedError(
                "matmul for sparse@sparse is not implemented in lacuna.array_api"
            )

        if isinstance(x, (CSR, CSC, COO)) and isinstance(y, np.ndarray):
            if y.ndim <= 2:
                return x @ y
            if y.shape[-2] != x.shape[1]:
                raise ValueError("Inner dimensions must match for matmul")
            batch_shape = y.shape[:-2]
            b_flat = int(np.prod(batch_shape))
            k = x.shape[1]
            n = y.shape[-1]
            y_bkn = y.reshape(b_flat, k, n)
            y_kbn = np.transpose(y_bkn, (1, 0, 2))
            y2 = np.ascontiguousarray(y_kbn.reshape(k, b_flat * n))
            out2 = x @ y2
            out_mbn = out2.reshape(x.shape[0], b_flat, n)
            out_bmn = np.transpose(out_mbn, (1, 0, 2))
            out = out_bmn.reshape(*batch_shape, x.shape[0], n)
            return out

        if isinstance(y, (CSR, CSC)) and isinstance(x, np.ndarray):
            raise NotImplementedError(
                "dense @ sparse matmul is not implemented in lacuna.array_api"
            )

        raise NotImplementedError(
            "matmul is not implemented for these sparse inputs in lacuna.array_api"
        )

    xp = _numpy_xp()
    return xp.matmul(x, y)


def tensordot(x, y, *, axes=2):
    if _is_sparse(x) or _is_sparse(y):
        if _core is None:
            raise RuntimeError("native core is not available")
        if _is_sparse(x) and _is_sparse(y):
            raise NotImplementedError("tensordot for sparse@sparse is not implemented")

        if isinstance(x, (CSR, CSC, COO)) and isinstance(y, np.ndarray):
            if isinstance(axes, int):
                if axes != 1:
                    raise NotImplementedError("tensordot supports axes=1 for sparse × dense")
                ax_x, ax_y = 1, 0
            else:
                axx, axy = axes
                axx = int(axx) if not isinstance(axx, (list, tuple)) else int(axx[0])
                axy = int(axy) if not isinstance(axy, (list, tuple)) else int(axy[0])
                if axx != 1 or axy != 0:
                    raise NotImplementedError(
                        "tensordot supports axes=([1],[0]) for sparse × dense"
                    )
                ax_x, ax_y = axx, axy

            if y.ndim == 1:
                if y.shape[0] != x.shape[1]:
                    raise ValueError("vector length must equal ncols")
                return x @ y
            if y.shape[0] != x.shape[1]:
                raise ValueError("Inner dimensions must match")
            b_shape = np.asarray(y.shape, dtype=np.int64)
            if isinstance(x, CSR):
                out2 = _core.tensordot_csr_dense_axes1x0_from_parts(
                    x.shape[0],
                    x.shape[1],
                    x.indptr,
                    x.indices,
                    x.data,
                    y.reshape(-1),
                    b_shape,
                    False,
                )
            elif isinstance(x, CSC):
                out2 = _core.tensordot_csc_dense_axes1x0_from_parts(
                    x.shape[0],
                    x.shape[1],
                    x.indptr,
                    x.indices,
                    x.data,
                    y.reshape(-1),
                    b_shape,
                    False,
                )
            else:
                out2 = _core.tensordot_coo_dense_axes1x0_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, y.reshape(-1), b_shape, False
                )
            return np.asarray(out2).reshape((x.shape[0],) + tuple(y.shape[1:]))

        if isinstance(x, COOND) and isinstance(y, np.ndarray):
            if isinstance(axes, int):
                if axes != 1:
                    raise NotImplementedError("tensordot supports axes=1 for COOND × dense")
                axis = x.ndim - 1
            else:
                axx, axy = axes
                axx = int(axx) if not isinstance(axx, (list, tuple)) else int(axx[0])
                axy = int(axy) if not isinstance(axy, (list, tuple)) else int(axy[0])
                if axy != 0:
                    raise NotImplementedError("right operand axis must be 0 for COOND × dense")
                axis = axx
            if y.shape[0] != x.shape[axis]:
                raise ValueError("Inner dimensions must match")
            oshape, oidx, odata = _core.coond_tensordot_dense_axis_from_parts(
                np.asarray(x.shape, dtype=np.int64),
                x.indices,
                x.data,
                int(axis),
                y.reshape(-1),
                np.asarray(y.shape, dtype=np.int64),
                False,
            )
            return COOND(
                tuple(int(t) for t in np.asarray(oshape, dtype=np.int64)), oidx, odata, check=False
            )

        raise NotImplementedError("tensordot is not implemented for these inputs")
    xp = _numpy_xp()
    return xp.tensordot(x, y, axes=axes)


def vecdot(x, y, *, axis=None):
    if _is_sparse(x) or _is_sparse(y):
        if _core is None:
            raise RuntimeError("native core is not available")
        if _is_sparse(x) and _is_sparse(y):
            raise NotImplementedError("vecdot for sparse@sparse is not implemented")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise NotImplementedError(
                "vecdot requires dense 1D vector as second operand for sparse × dense"
            )

        if isinstance(x, (CSR, CSC, COO)):
            ax = axis
            if ax is None or ax == -1:
                ax = 1
            ax = int(ax)
            if ax == 1:
                if y.shape[0] != x.shape[1]:
                    raise ValueError("vector length must equal ncols")
                return x @ y
            if ax == 0:
                if y.shape[0] != x.shape[0]:
                    raise ValueError("vector length must equal nrows")
                if isinstance(x, CSR):
                    return _core.vecdot_csr_axis0_from_parts(
                        x.shape[0], x.shape[1], x.indptr, x.indices, x.data, y, False
                    )
                if isinstance(x, CSC):
                    return _core.vecdot_csc_axis0_from_parts(
                        x.shape[0], x.shape[1], x.indptr, x.indices, x.data, y, False
                    )
                return _core.vecdot_coo_axis0_from_parts(
                    x.shape[0], x.shape[1], x.row, x.col, x.data, y, False
                )
            raise NotImplementedError("vecdot supports axis in {0,1} for 2D sparse × dense")

        if isinstance(x, COOND):
            ax = axis
            if ax is None or ax == -1:
                ax = x.ndim - 1
            ax = int(ax)
            if y.shape[0] != x.shape[ax]:
                raise ValueError("vector length must equal shape[axis]")
            oshape, oidx, odata = _core.coond_vecdot_axis_from_parts(
                np.asarray(x.shape, dtype=np.int64), x.indices, x.data, int(ax), y, False
            )
            return COOND(
                tuple(int(t) for t in np.asarray(oshape, dtype=np.int64)), oidx, odata, check=False
            )

        raise NotImplementedError("vecdot is not implemented for these inputs")
    xp = _numpy_xp()
    return xp.vecdot(x, y, axis=axis)


def matrix_transpose(x):
    if isinstance(x, (CSR, CSC, COO)):
        return x.T
    xp = _numpy_xp()
    return xp.matrix_transpose(x)
