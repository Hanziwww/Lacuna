import numpy as np

from ...sparse import COO, COOND, CSC, CSR
from .. import _namespace as _ns
from .._namespace import _numpy_xp


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
    return getattr(_ns, "tensordot")(x, y, axes=axes)


def vecdot(x, y, *, axis=None):
    return getattr(_ns, "vecdot")(x, y, axis=axis)


def matrix_transpose(x):
    if isinstance(x, (CSR, CSC, COO)):
        return x.T
    xp = _numpy_xp()
    return xp.matrix_transpose(x)
