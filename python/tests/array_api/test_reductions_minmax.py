import numpy as np
import lacuna.array_api as xp
from lacuna.sparse import CSR, CSC


def sample_csr():
    # [[1.0, 0.0, 3.0],
    #  [0.0, -2.0, 0.0]]
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, -2.0])
    return CSR(indptr, indices, data, shape=(2, 3))


def sample_csc():
    # transpose of above
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, -2.0, 3.0])
    return CSC(indptr, indices, data, shape=(2, 3))


def test_minmax_csr_global_and_axes():
    A = sample_csr()
    # global
    assert xp.reductions.min(A) == -2.0
    assert xp.reductions.max(A) == 3.0

    # axis=0/1 with keepdims
    mn0 = xp.reductions.min(A, axis=0)
    mx0 = xp.reductions.max(A, axis=0)
    np.testing.assert_allclose(mn0, np.array([0.0, -2.0, 0.0]))
    np.testing.assert_allclose(mx0, np.array([1.0, 0.0, 3.0]))

    mn1_k = xp.reductions.min(A, axis=1, keepdims=True)
    mx1_k = xp.reductions.max(A, axis=1, keepdims=True)
    np.testing.assert_allclose(mn1_k, np.array([[0.0], [-2.0]]))
    np.testing.assert_allclose(mx1_k, np.array([[3.0], [0.0]]))


def test_minmax_csc_global_and_axes():
    A = sample_csc()
    assert xp.reductions.min(A) == -2.0
    assert xp.reductions.max(A) == 3.0

    mn0 = xp.reductions.min(A, axis=0)
    mx0 = xp.reductions.max(A, axis=0)
    np.testing.assert_allclose(mn0, np.array([0.0, -2.0, 0.0]))
    np.testing.assert_allclose(mx0, np.array([1.0, 0.0, 3.0]))

    mn1 = xp.reductions.min(A, axis=1)
    mx1 = xp.reductions.max(A, axis=1)
    np.testing.assert_allclose(mn1, np.array([0.0, -2.0]))
    np.testing.assert_allclose(mx1, np.array([3.0, 0.0]))
