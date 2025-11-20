from .. import _dispatch as _dp
from .. import _namespace as _ns


def reshape(x, newshape):
    return _dp.reshape(x, newshape)


def squeeze(x, axis=None):
    return getattr(_ns, "squeeze")(x, axis=axis)


def expand_dims(x, axis):
    return getattr(_ns, "expand_dims")(x, axis)


def moveaxis(x, source, destination):
    return getattr(_ns, "moveaxis")(x, source, destination)


def permute_dims(x, axes):
    return _dp.permute_dims(x, axes)


def stack(arrays, axis=0):
    return getattr(_ns, "stack")(arrays, axis=axis)


def concat(arrays, axis=0):
    return getattr(_ns, "concat")(arrays, axis=axis)


def unstack(x, axis=0):
    return getattr(_ns, "unstack")(x, axis=axis)


def broadcast_to(x, shape):
    return getattr(_ns, "broadcast_to")(x, shape)


def broadcast_arrays(*arrays):
    return getattr(_ns, "broadcast_arrays")(*arrays)


def flip(x, axis=None):
    return getattr(_ns, "flip")(x, axis=axis)


def roll(x, shift, axis=None):
    return getattr(_ns, "roll")(x, shift, axis=axis)


def repeat(x, repeats, axis=None):
    return getattr(_ns, "repeat")(x, repeats, axis=axis)


def tile(x, reps):
    return getattr(_ns, "tile")(x, reps)
