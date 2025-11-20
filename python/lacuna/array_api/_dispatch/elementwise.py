import numpy as np

from ...sparse import COO, COOND, CSC, CSR
from .._namespace import _numpy_xp


def _is_sparse(x) -> bool:
    return isinstance(x, (CSR, CSC, COO, COOND))


def add(x, y):
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x + y
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x + y
    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "add for sparse inputs is only implemented for CSR+CSR and CSC+CSC"
        )
    xp = _numpy_xp()
    return xp.add(x, y)


def subtract(x, y):
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x - y
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x - y
    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "subtract for sparse inputs is only implemented for CSR-CSR and CSC-CSC"
        )
    xp = _numpy_xp()
    return xp.subtract(x, y)


def multiply(x, y):
    # scalar * sparse or sparse * scalar
    if isinstance(x, (int, float)) and isinstance(y, (CSR, CSC, COO)):
        return y * float(x)
    if isinstance(y, (int, float)) and isinstance(x, (CSR, CSC, COO)):
        return x * float(y)

    # sparse * sparse (Hadamard)
    if isinstance(x, COOND) and isinstance(y, COOND):
        return x.hadamard_broadcast(y)
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x.multiply(y)
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x.multiply(y)

    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "multiply for sparse inputs is only implemented for scalar*sparse and CSR*CSR/CSC*CSC"
        )

    xp = _numpy_xp()
    return xp.multiply(x, y)
