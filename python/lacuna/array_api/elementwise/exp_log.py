from .. import _namespace as _ns


def exp(x):
    return getattr(_ns, "exp")(x)


def expm1(x):
    return getattr(_ns, "expm1")(x)


def log(x):
    return getattr(_ns, "log")(x)


def log1p(x):
    return getattr(_ns, "log1p")(x)


def log2(x):
    return getattr(_ns, "log2")(x)


def log10(x):
    return getattr(_ns, "log10")(x)


def logaddexp(x, y):
    return getattr(_ns, "logaddexp")(x, y)
