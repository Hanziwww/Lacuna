from . import _namespace as _ns


def sort(x, axis=-1):
    return getattr(_ns, "sort")(x, axis=axis)


def argsort(x, axis=-1):
    return getattr(_ns, "argsort")(x, axis=axis)
