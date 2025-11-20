from .. import _namespace as _ns


def abs(x):
    return getattr(_ns, "abs")(x)


def negative(x):
    return getattr(_ns, "negative")(x)


def positive(x):
    return getattr(_ns, "positive")(x)


def sign(x):
    return getattr(_ns, "sign")(x)


def signbit(x):
    return getattr(_ns, "signbit")(x)


def sqrt(x):
    return getattr(_ns, "sqrt")(x)


def square(x):
    return getattr(_ns, "square")(x)


def floor(x):
    return getattr(_ns, "floor")(x)


def ceil(x):
    return getattr(_ns, "ceil")(x)


def trunc(x):
    return getattr(_ns, "trunc")(x)


def round(x):
    return getattr(_ns, "round")(x)
