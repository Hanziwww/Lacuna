from typing import Any, Dict

from ..sparse.base import SparseArray, SparseMatrix


def _numpy_xp():
    try:
        import numpy.array_api as xp  # type: ignore
    except Exception:  # ModuleNotFoundError or other import issues
        import numpy as xp  # type: ignore
    return xp


def __array_namespace_info__() -> Dict[str, Any]:
    return {
        "devices": ["cpu"],
        "default_device": "cpu",
        "dtypes": ["bool", "int64", "float64"],
        "default_dtypes": {
            "floating": "float64",
            "integral": "int64",
            "boolean": "bool",
        },
        "capabilities": {
            # Global
            "sparse": True,
            # Implemented sparse-first ops in this release
            "linalg": ["matmul", "matrix_transpose"],
            "reductions": ["sum", "mean", "count_nonzero"],
            "elementwise": ["add", "subtract", "multiply"],  # multiply includes COOND broadcast
            # Creation helpers routed to sparse types
            "creation": ["zeros", "eye"],
            # Manipulation for COOND
            "manipulation": ["permute_dims", "reshape"],
        },
    }


def __getattr__(name: str):
    xp = _numpy_xp()
    attr = getattr(xp, name)

    if callable(attr):

        def guarded(*args, **kwargs):
            def _is_sparse(x: Any) -> bool:
                return isinstance(x, (SparseArray, SparseMatrix))

            for v in args:
                if _is_sparse(v):
                    raise NotImplementedError(
                        f"{name!s} is not implemented for sparse inputs in lacuna.array_api"
                    )
            for v in kwargs.values():
                if _is_sparse(v):
                    raise NotImplementedError(
                        f"{name!s} is not implemented for sparse inputs in lacuna.array_api"
                    )
            return attr(*args, **kwargs)

        return guarded

    return attr
