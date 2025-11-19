"""Base classes for sparse arrays and matrices.

These classes define the minimal interface shared by concrete sparse types in
`lacuna.sparse`, including shape/dtype bookkeeping and basic materialization.
"""


class SparseArray:
    """Abstract base class for sparse N-dimensional arrays.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape. Stored as a tuple.
    dtype : Any, optional
        Element dtype metadata (informational for base class).

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape.
    ndim : int
        Number of dimensions, equal to ``len(shape)``.
    dtype : Any
        Element type metadata.
    """

    def __init__(self, shape, dtype=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    def __array_namespace__(self):
        import lacuna.array_api as xp
        return xp


class SparseMatrix(SparseArray):
    """Abstract base class for 2D sparse matrices.

    Parameters
    ----------
    shape : tuple[int, int]
        Matrix shape. Must be two-dimensional.
    dtype : Any, optional
        Element dtype metadata.

    Raises
    ------
    ValueError
        If ``shape`` is not 2D.
    """

    def __init__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError("SparseMatrix requires 2D shape")
        super().__init__(shape, dtype=dtype)

    def toarray(self):
        """Return a dense numpy.ndarray with the same shape and dtype.

        Notes
        -----
        The base implementation returns an all-zeros array. Concrete sparse
        matrix types should override this to materialize actual data.
        """
        import numpy as np

        return np.zeros(self.shape, dtype=self.dtype)
