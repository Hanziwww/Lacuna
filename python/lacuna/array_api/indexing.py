from . import _namespace as _ns


def take(x, indices, /, *, axis=None):
    return getattr(_ns, "take")(x, indices, axis=axis)


def take_along_axis(x, indices, axis):
    return getattr(_ns, "take_along_axis")(x, indices, axis)

