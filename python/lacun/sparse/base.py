class SparseArray:
    def __init__(self, shape, dtype=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

class SparseMatrix(SparseArray):
    def __init__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError("SparseMatrix requires 2D shape")
        super().__init__(shape, dtype=dtype)
