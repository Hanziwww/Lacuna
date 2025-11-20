from .. import _namespace as _ns


def real(x):
    return getattr(_ns, "real")(x)


def imag(x):
    return getattr(_ns, "imag")(x)


def conj(x):
    return getattr(_ns, "conj")(x)
