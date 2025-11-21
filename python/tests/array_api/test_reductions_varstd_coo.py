import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import COO


def sample_dense():
    return np.array([[1.0, 0.0, 3.0], [0.0, -2.0, 0.0]])


def sample_coo():
    row = np.array([0, 0, 1], dtype=np.int64)
    col = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, -2.0])
    return COO(row, col, data, shape=(2, 3))


def test_var_coo_global_and_axes():
    A = sample_coo()
    D = sample_dense()

    # global
    v = xp.reductions.var(A)
    assert np.isclose(v, np.var(D, ddof=0))

    # axis=0
    v0 = xp.reductions.var(A, axis=0)
    np.testing.assert_allclose(v0, np.var(D, axis=0, ddof=0))

    # axis=1, keepdims
    v1k = xp.reductions.var(A, axis=1, keepdims=True)
    np.testing.assert_allclose(v1k, np.var(D, axis=1, ddof=0).reshape(2, 1))

    # correction=1
    v_ddof1 = xp.reductions.var(A, correction=1.0)
    assert np.isclose(v_ddof1, np.var(D, ddof=1))


def test_std_coo_global_and_axes():
    A = sample_coo()
    D = sample_dense()

    # global
    s = xp.reductions.std(A)
    assert np.isclose(s, np.std(D, ddof=0))

    # axis=0
    s0 = xp.reductions.std(A, axis=0)
    np.testing.assert_allclose(s0, np.std(D, axis=0, ddof=0))

    # axis=1, keepdims
    s1k = xp.reductions.std(A, axis=1, keepdims=True)
    np.testing.assert_allclose(s1k, np.std(D, axis=1, ddof=0).reshape(2, 1))

    # correction=1
    s_ddof1 = xp.reductions.std(A, correction=1.0)
    assert np.isclose(s_ddof1, np.std(D, ddof=1))
