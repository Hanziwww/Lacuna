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
    # transpose of above CSR
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, -2.0, 3.0])
    return CSC(indptr, indices, data, shape=(2, 3))


def test_prod_csr_global_and_axes():
    A = sample_csr()
    # global product: implicit zeros => 0
    assert xp.reductions.prod(A) == 0.0

    # axis=0
    p0 = xp.reductions.prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([0.0, 0.0, 0.0]))

    # axis=1, keepdims
    p1k = xp.reductions.prod(A, axis=1, keepdims=True)
    np.testing.assert_allclose(p1k, np.array([[0.0], [0.0]]))


def test_prod_csc_global_and_axes():
    A = sample_csc()
    assert xp.reductions.prod(A) == 0.0
    p0 = xp.reductions.prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([0.0, 0.0, 0.0]))
    p1 = xp.reductions.prod(A, axis=1)
    np.testing.assert_allclose(p1, np.array([0.0, 0.0]))
