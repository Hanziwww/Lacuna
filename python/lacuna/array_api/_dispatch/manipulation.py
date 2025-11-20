from ...sparse import COO, COOND, CSC, CSR
from .._namespace import _numpy_xp


def _is_sparse(x) -> bool:
    return isinstance(x, (CSR, CSC, COO, COOND))


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
