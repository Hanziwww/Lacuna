from .. import _dispatch as _dp
from .. import _namespace as _ns


def add(x, y):
    return _dp.add(x, y)


def subtract(x, y):
    return _dp.subtract(x, y)


def multiply(x, y):
    return _dp.multiply(x, y)


def divide(x, y):
    return _dp.divide(x, y)


def floor_divide(x, y):
    return _dp.floor_divide(x, y)


def remainder(x, y):
    return _dp.remainder(x, y)


def pow(x, y):
    return _dp.pow(x, y)


def maximum(x, y):
    return getattr(_ns, "maximum")(x, y)


def minimum(x, y):
    return getattr(_ns, "minimum")(x, y)


def equal(x, y):
    return _dp.equal(x, y)


def not_equal(x, y):
    return _dp.not_equal(x, y)


def greater(x, y):
    return _dp.greater(x, y)


def greater_equal(x, y):
    return _dp.greater_equal(x, y)


def less(x, y):
    return _dp.less(x, y)


def less_equal(x, y):
    return _dp.less_equal(x, y)
