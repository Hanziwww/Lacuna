from typing import Any


def finfo(dtype: Any):
    import numpy as _np

    return _np.finfo(dtype)


def iinfo(dtype: Any):
    import numpy as _np

    return _np.iinfo(dtype)
