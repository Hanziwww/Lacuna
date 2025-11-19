from .. import _namespace as _ns


def isfinite(x):
    return getattr(_ns, "isfinite")(x)


def isinf(x):
    return getattr(_ns, "isinf")(x)


def isnan(x):
    return getattr(_ns, "isnan")(x)


def copysign(x, y):
    return getattr(_ns, "copysign")(x, y)


def nextafter(x, y):
    return getattr(_ns, "nextafter")(x, y)

