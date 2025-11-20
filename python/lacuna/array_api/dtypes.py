from ..sparse import CSR
from ..sparse.base import SparseArray, SparseMatrix
from . import _namespace as _ns


def astype(x, dtype, /, *, copy=True):
    # Provide sparse path for CSR; other sparse types not yet supported here.
    if isinstance(x, CSR):
        return x.astype(dtype)
    if isinstance(x, (SparseArray, SparseMatrix)):
        raise NotImplementedError("astype is only implemented for CSR in lacuna.array_api")
    return getattr(_ns, "astype")(x, dtype, copy=copy)


def can_cast(from_dtype, to_dtype, /):
    return getattr(_ns, "can_cast")(from_dtype, to_dtype)


def isdtype(dtype, kind):
    return getattr(_ns, "isdtype")(dtype, kind)


def result_type(*arrays_and_dtypes):
    return getattr(_ns, "result_type")(*arrays_and_dtypes)


def finfo(dtype):
    return getattr(_ns, "finfo")(dtype)


def iinfo(dtype):
    return getattr(_ns, "iinfo")(dtype)
