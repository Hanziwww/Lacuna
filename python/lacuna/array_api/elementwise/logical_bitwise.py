from .. import _namespace as _ns


def logical_and(x, y):
    return getattr(_ns, "logical_and")(x, y)


def logical_or(x, y):
    return getattr(_ns, "logical_or")(x, y)


def logical_not(x):
    return getattr(_ns, "logical_not")(x)


def logical_xor(x, y):
    return getattr(_ns, "logical_xor")(x, y)


def bitwise_and(x, y):
    return getattr(_ns, "bitwise_and")(x, y)


def bitwise_or(x, y):
    return getattr(_ns, "bitwise_or")(x, y)


def bitwise_xor(x, y):
    return getattr(_ns, "bitwise_xor")(x, y)


def bitwise_invert(x):
    return getattr(_ns, "bitwise_invert")(x)


def bitwise_left_shift(x, y):
    return getattr(_ns, "bitwise_left_shift")(x, y)


def bitwise_right_shift(x, y):
    return getattr(_ns, "bitwise_right_shift")(x, y)

