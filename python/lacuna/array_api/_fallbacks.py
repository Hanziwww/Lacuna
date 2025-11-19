def numpy_fallback(name: str):
    import numpy.array_api as _xp  # type: ignore

    return getattr(_xp, name)
