from . import _namespace as _ns


def unique_values(x):
    return getattr(_ns, "unique_values")(x)


def unique_counts(x):
    return getattr(_ns, "unique_counts")(x)


def unique_inverse(x):
    return getattr(_ns, "unique_inverse")(x)


def unique_all(x):
    return getattr(_ns, "unique_all")(x)
