from .. import _dispatch as _dp
from .. import _namespace as _ns


def argmax(x, axis=None, keepdims=False):
    return getattr(_ns, "argmax")(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    return getattr(_ns, "argmin")(x, axis=axis, keepdims=keepdims)


def nonzero(x):
    return getattr(_ns, "nonzero")(x)


def count_nonzero(x, axis=None, keepdims=False):
    return _dp.count_nonzero(x, axis=axis, keepdims=keepdims)


def where(cond, x, y):
    return getattr(_ns, "where")(cond, x, y)


def searchsorted(x, v, *, side="left"):
    return getattr(_ns, "searchsorted")(x, v, side=side)
