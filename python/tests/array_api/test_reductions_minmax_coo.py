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


def test_minmax_coo_global_and_axes():
    A = sample_coo()
    # global
    assert xp.reductions.min(A) == -2.0
    assert xp.reductions.max(A) == 3.0

    # axis=0
    mn0 = xp.reductions.min(A, axis=0)
    mx0 = xp.reductions.max(A, axis=0)
    np.testing.assert_allclose(mn0, np.array([0.0, -2.0, 0.0]))
    np.testing.assert_allclose(mx0, np.array([1.0, 0.0, 3.0]))

    # axis=1, keepdims
    mn1_k = xp.reductions.min(A, axis=1, keepdims=True)
    mx1_k = xp.reductions.max(A, axis=1, keepdims=True)
    np.testing.assert_allclose(mn1_k, np.array([[0.0], [-2.0]]))
    np.testing.assert_allclose(mx1_k, np.array([[3.0], [0.0]]))
