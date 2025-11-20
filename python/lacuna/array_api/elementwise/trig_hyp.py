from .. import _namespace as _ns


def sin(x):
    return getattr(_ns, "sin")(x)


def cos(x):
    return getattr(_ns, "cos")(x)


def tan(x):
    return getattr(_ns, "tan")(x)


def asin(x):
    return getattr(_ns, "asin")(x)


def acos(x):
    return getattr(_ns, "acos")(x)


def atan(x):
    return getattr(_ns, "atan")(x)


def atan2(y, x):
    return getattr(_ns, "atan2")(y, x)


def sinh(x):
    return getattr(_ns, "sinh")(x)


def cosh(x):
    return getattr(_ns, "cosh")(x)


def tanh(x):
    return getattr(_ns, "tanh")(x)


def asinh(x):
    return getattr(_ns, "asinh")(x)


def acosh(x):
    return getattr(_ns, "acosh")(x)


def atanh(x):
    return getattr(_ns, "atanh")(x)
