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


def divide(x, y):
    # sparse / scalar
    if isinstance(y, (int, float)) and isinstance(x, (CSR, CSC, COO)):
        return x / float(y)

    # scalar / sparse would densify (implicit zeros -> inf), not supported
    if isinstance(x, (int, float)) and _is_sparse(y):
        raise NotImplementedError("scalar / sparse would densify; not supported")

    # sparse / sparse
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x.divide(y)
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x.divide(y)
    if isinstance(x, COOND) and isinstance(y, COOND):
        raise NotImplementedError("COOND divide is not implemented yet")

    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "divide for sparse inputs is only implemented for CSR/CSC pairs and sparse/scalar"
        )

    xp = _numpy_xp()
    return xp.divide(x, y)


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


def floor_divide(x, y):
    # sparse // scalar
    if isinstance(y, (int, float)) and isinstance(x, (CSR, CSC, COO, COOND)):
        return x // float(y)

    # scalar // sparse would densify (implicit zeros -> inf), not supported
    if isinstance(x, (int, float)) and _is_sparse(y):
        raise NotImplementedError("scalar // sparse would densify; not supported")

    # sparse // sparse
    if isinstance(x, CSR) and isinstance(y, CSR):
        return x.floor_divide(y)
    if isinstance(x, CSC) and isinstance(y, CSC):
        return x.floor_divide(y)
    if isinstance(x, COOND) and isinstance(y, COOND):
        raise NotImplementedError("COOND floor_divide is not implemented yet")

    if _is_sparse(x) or _is_sparse(y):
        raise NotImplementedError(
            "floor_divide for sparse inputs is only implemented for CSR/CSC pairs and sparse//scalar"
        )

    xp = _numpy_xp()
    return xp.floor_divide(x, y)
