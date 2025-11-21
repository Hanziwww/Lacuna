from .. import _dispatch as _dp
from .. import _namespace as _ns


def sum(x, axis=None, keepdims=False):
    return _dp.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    return _dp.prod(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return _dp.min(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return _dp.max(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return _dp.mean(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, correction=0.0, keepdims=False):
    return _dp.var(x, axis=axis, correction=correction, keepdims=keepdims)


def std(x, axis=None, correction=0.0, keepdims=False):
    return _dp.std(x, axis=axis, correction=correction, keepdims=keepdims)


def cumulative_sum(x, axis=None):
    return getattr(_ns, "cumulative_sum")(x, axis=axis)


def cumulative_prod(x, axis=None):
    return getattr(_ns, "cumulative_prod")(x, axis=axis)


def all(x, axis=None, keepdims=False):
    return getattr(_ns, "all")(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    return getattr(_ns, "any")(x, axis=axis, keepdims=keepdims)


def diff(x, *, n=1, axis=-1):
    return getattr(_ns, "diff")(x, n=n, axis=axis)
