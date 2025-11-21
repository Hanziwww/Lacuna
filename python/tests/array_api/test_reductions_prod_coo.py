import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import COO


def sample_coo():
    # [[1.0, 0.0, 3.0],
    #  [0.0, -2.0, 0.0]]
    row = np.array([0, 0, 1], dtype=np.int64)
    col = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, -2.0])
    return COO(row, col, data, shape=(2, 3))


def test_prod_coo_global_and_axes():
    A = sample_coo()
    # global product: implicit zeros make product 0
    assert xp.reductions.prod(A) == 0.0

    # axis=0 (columns)
    p0 = xp.reductions.prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([0.0, 0.0, 0.0]))

    # axis=1 (rows), keepdims
    p1k = xp.reductions.prod(A, axis=1, keepdims=True)
    np.testing.assert_allclose(p1k, np.array([[0.0], [0.0]]))
