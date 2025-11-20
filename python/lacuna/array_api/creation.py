from . import _namespace as _ns


def asarray(obj, /, *, dtype=None, device=None, copy=None):
    return getattr(_ns, "asarray")(obj, dtype=dtype, device=device, copy=copy)


def zeros(shape, *, dtype=None, device=None):
    """Create a sparse zero matrix (Array API compliant).

    Parameters
    ----------
    shape : tuple of int
        Matrix shape (nrows, ncols).
    dtype : dtype, optional
        Data type (default: float64).
    device : str, optional
        Device (default: "cpu").

    Returns
    -------
    CSR
        Sparse zero matrix using Rust backend.

    Examples
    --------
    >>> import lacuna.array_api as xp
    >>> A = xp.zeros((10, 20))
    >>> A.nnz
    0
    """
    import numpy as np

    from ..sparse import CSR

    if dtype is None:
        dtype = np.float64
    if device is not None and device != "cpu":
        raise ValueError("Only 'cpu' device is currently supported")

    return CSR.zeros(shape, dtype=dtype)


def ones(shape, *, dtype=None, device=None):
    return getattr(_ns, "ones")(shape, dtype=dtype, device=device)


def full(shape, fill_value, *, dtype=None, device=None):
    return getattr(_ns, "full")(shape, fill_value, dtype=dtype, device=device)


def empty(shape, *, dtype=None, device=None):
    return getattr(_ns, "empty")(shape, dtype=dtype, device=device)


def zeros_like(x, *, dtype=None, device=None):
    return getattr(_ns, "zeros_like")(x, dtype=dtype, device=device)


def ones_like(x, *, dtype=None, device=None):
    return getattr(_ns, "ones_like")(x, dtype=dtype, device=device)


def full_like(x, fill_value, *, dtype=None, device=None):
    return getattr(_ns, "full_like")(x, fill_value, dtype=dtype, device=device)


def empty_like(x, *, dtype=None, device=None):
    return getattr(_ns, "empty_like")(x, dtype=dtype, device=device)


def arange(start, stop=None, step=1, *, dtype=None, device=None):
    return getattr(_ns, "arange")(start, stop, step, dtype=dtype, device=device)


def linspace(start, stop, num, *, dtype=None, device=None):
    return getattr(_ns, "linspace")(start, stop, num, dtype=dtype, device=device)


def meshgrid(*arrays, indexing="xy"):
    return getattr(_ns, "meshgrid")(*arrays, indexing=indexing)


def eye(n_rows, n_cols=None, k=0, *, dtype=None, device=None):
    """Create an identity matrix (Array API compliant).

    Parameters
    ----------
    n_rows : int
        Number of rows (for square matrix, this is the size).
    n_cols : int, optional
        Number of columns (if None, creates square matrix).
    k : int, optional
        Diagonal offset (default: 0 for main diagonal).
    dtype : dtype, optional
        Data type (default: float64).
    device : str, optional
        Device (default: "cpu").

    Returns
    -------
    CSR
        Identity matrix using Rust backend.

    Notes
    -----
    Currently only supports k=0 (main diagonal) and square matrices.

    Examples
    --------
    >>> import lacuna.array_api as xp
    >>> I = xp.eye(5)
    >>> I.sum()
    5.0
    """
    import numpy as np

    from ..sparse import CSR

    if dtype is None:
        dtype = np.float64
    if device is not None and device != "cpu":
        raise ValueError("Only 'cpu' device is currently supported")
    if n_cols is not None and n_cols != n_rows:
        raise NotImplementedError("Non-square identity matrices not yet supported")
    if k != 0:
        raise NotImplementedError("Off-diagonal identity (k != 0) not yet supported")

    return CSR.eye(n_rows, dtype=dtype)


def tril(x, k=0):
    return getattr(_ns, "tril")(x, k)


def triu(x, k=0):
    return getattr(_ns, "triu")(x, k)


def from_dlpack(dlpack_obj):
    return getattr(_ns, "from_dlpack")(dlpack_obj)
