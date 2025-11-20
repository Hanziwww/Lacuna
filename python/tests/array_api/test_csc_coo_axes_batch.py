import numpy as np
import pytest

import lacuna.array_api as xp
from lacuna.sparse import COO, CSC

# Build a small 2x3 matrix with values:
# [[1, 0, 0],
#  [0, 2, 3]]


def make_csc():
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 1], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return CSC(indptr, indices, data, shape=(2, 3))


def make_coo():
    row = np.array([0, 1, 1], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return COO(row, col, data, shape=(2, 3))


@pytest.mark.parametrize("maker", [make_csc, make_coo])
def test_axes_2d_sum_mean_count_basic(maker):
    A = maker()
    # sum total
    assert xp.sum(A) == pytest.approx(6.0)
    # sum along axes (accept negative and tuple axes)
    np.testing.assert_allclose(xp.sum(A, axis=0), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(xp.sum(A, axis=1), np.array([1.0, 5.0]))
    np.testing.assert_allclose(xp.sum(A, axis=-1), xp.sum(A, axis=1))
    assert xp.sum(A, axis=(0, 1)) == xp.sum(A)
    assert xp.sum(A, axis=(-2, -1)) == xp.sum(A)

    # mean
    np.testing.assert_allclose(xp.mean(A, axis=0), np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose(xp.mean(A, axis=1), np.array([1.0 / 3.0, 5.0 / 3.0]))
    np.testing.assert_allclose(xp.mean(A, axis=-1), xp.mean(A, axis=1))
    assert xp.mean(A, axis=(0, 1)) == xp.mean(A)

    # count_nonzero
    np.testing.assert_array_equal(xp.count_nonzero(A), 3)
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=0), np.array([1, 1, 1]))
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=1), np.array([1, 2]))

    # keepdims shapes
    np.testing.assert_allclose(xp.sum(A, axis=0, keepdims=True), np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(xp.sum(A, axis=1, keepdims=True), np.array([[1.0], [5.0]]))


@pytest.mark.parametrize("maker", [make_csc, make_coo])
def test_axes_2d_errors(maker):
    A = maker()
    with pytest.raises(NotImplementedError):
        xp.sum(A, axis=())
    with pytest.raises(NotImplementedError):
        xp.mean(A, axis=())
    with pytest.raises(NotImplementedError):
        xp.count_nonzero(A, axis=())
    with pytest.raises(ValueError):
        xp.sum(A, axis=2)
    with pytest.raises(ValueError):
        xp.mean(A, axis=2)
    with pytest.raises(ValueError):
        xp.count_nonzero(A, axis=2)


def test_batched_matmul_coo_dense():
    A = make_coo()  # (2,3)
    B = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)  # (batch=4, k=3, n=5)
    Y = xp.matmul(A, B)
    assert Y.shape == (4, 2, 5)

    A_dense = np.zeros((2, 3), dtype=np.float64)
    A_dense[0, 0] = 1.0
    A_dense[1, 1] = 2.0
    A_dense[1, 2] = 3.0
    expected = np.matmul(np.broadcast_to(A_dense, (4, 2, 3)), B)
    np.testing.assert_allclose(Y, expected)
