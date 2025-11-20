import numpy as np
import pytest

import lacuna.array_api as xp
from lacuna.sparse import COOND, CSR


def make_csr():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    return CSR(indptr, indices, data, shape=(2, 3))


def test_csr_axes_negative_and_tuple_sum_mean_count():
    A = make_csr()

    # sum
    np.testing.assert_allclose(xp.sum(A, axis=-1), xp.sum(A, axis=1))
    np.testing.assert_allclose(xp.sum(A, axis=(-1,)), xp.sum(A, axis=1))
    assert xp.sum(A, axis=(-2, -1)) == xp.sum(A)
    assert xp.sum(A, axis=(0, 1)) == xp.sum(A)
    with pytest.raises(NotImplementedError):
        xp.sum(A, axis=())

    # mean
    np.testing.assert_allclose(xp.mean(A, axis=-1), xp.mean(A, axis=1))
    np.testing.assert_allclose(xp.mean(A, axis=(-1,)), xp.mean(A, axis=1))
    assert xp.mean(A, axis=(-2, -1)) == xp.mean(A)
    assert xp.mean(A, axis=(0, 1)) == xp.mean(A)
    with pytest.raises(NotImplementedError):
        xp.mean(A, axis=())

    # count_nonzero
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=-1), xp.count_nonzero(A, axis=1))
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=(-1,)), xp.count_nonzero(A, axis=1))
    assert xp.count_nonzero(A, axis=(-2, -1)) == xp.count_nonzero(A)
    assert xp.count_nonzero(A, axis=(0, 1)) == xp.count_nonzero(A)
    with pytest.raises(NotImplementedError):
        xp.count_nonzero(A, axis=())


def test_batched_matmul_sparse_dense():
    A = make_csr()  # (2,3)
    # B has shape (batch=4, k=3, n=5)
    B = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    Y = xp.matmul(A, B)
    assert Y.shape == (4, 2, 5)
    A_dense = A.toarray()
    expected = np.matmul(np.broadcast_to(A_dense, (4, 2, 3)), B)
    np.testing.assert_allclose(Y, expected)


def test_coond_axis_and_keepdims():
    # shape (2,3,4), indices for (0,1,2)=1.0 and (1,2,3)=3.0
    shape = (2, 3, 4)
    indices = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)
    data = np.array([1.0, 3.0], dtype=np.float64)
    X = COOND(shape, indices, data)

    # global
    assert xp.sum(X) == 4.0
    assert xp.mean(X) == 4.0 / np.prod(shape)

    # axis reduce and keepdims
    Y = xp.sum(X, axis=1)
    assert Y.shape == (2, 4)
    Yk = xp.sum(X, axis=1, keepdims=True)
    assert Yk.shape == (2, 1, 4)

    Z = xp.mean(X, axis=(0, 2))
    assert Z.shape == (3,)
    Zk = xp.mean(X, axis=(0, 2), keepdims=True)
    assert Zk.shape == (1, 3, 1)
