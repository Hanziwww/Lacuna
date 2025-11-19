from . import _namespace as _ns


def matmul(x, y):
    return getattr(_ns, "matmul")(x, y)


def tensordot(x, y, *, axes=2):
    return getattr(_ns, "tensordot")(x, y, axes=axes)


def vecdot(x, y, *, axis=None):
    return getattr(_ns, "vecdot")(x, y, axis=axis)


def matrix_transpose(x):
    return getattr(_ns, "matrix_transpose")(x)

