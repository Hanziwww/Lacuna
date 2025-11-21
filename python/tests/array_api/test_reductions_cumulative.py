import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import COO, CSC, CSR


def sample_csr():
    # [[1.0, 0.0, 3.0],
    #  [0.0, -2.0, 0.0]]
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, -2.0])
    return CSR(indptr, indices, data, shape=(2, 3))


def sample_csc():
    # transpose of above CSR as CSC representation (same dense content)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, -2.0, 3.0])
    return CSC(indptr, indices, data, shape=(2, 3))


def sample_coo():
    # same dense content
    row = np.array([0, 0, 1], dtype=np.int64)
    col = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, -2.0])
    return COO(row, col, data, shape=(2, 3))


def test_cumulative_sum_csr():
    A = sample_csr()
    # axis=1 (row-wise prefix)
    s1 = xp.reductions.cumulative_sum(A, axis=1)
    np.testing.assert_allclose(s1, np.array([[1.0, 1.0, 4.0], [0.0, -2.0, -2.0]]))
    # axis=0 (col-wise prefix)
    s0 = xp.reductions.cumulative_sum(A, axis=0)
    np.testing.assert_allclose(s0, np.array([[1.0, 0.0, 3.0], [1.0, -2.0, 3.0]]))


def test_cumulative_prod_csr():
    A = sample_csr()
    # axis=1 (row-wise prefix)
    p1 = xp.reductions.cumulative_prod(A, axis=1)
    np.testing.assert_allclose(p1, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    # axis=0 (col-wise prefix)
    p0 = xp.reductions.cumulative_prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0]]))


def test_cumulative_sum_csc():
    A = sample_csc()
    s1 = xp.reductions.cumulative_sum(A, axis=1)
    np.testing.assert_allclose(s1, np.array([[1.0, 1.0, 4.0], [0.0, -2.0, -2.0]]))
    s0 = xp.reductions.cumulative_sum(A, axis=0)
    np.testing.assert_allclose(s0, np.array([[1.0, 0.0, 3.0], [1.0, -2.0, 3.0]]))


def test_cumulative_prod_csc():
    A = sample_csc()
    p1 = xp.reductions.cumulative_prod(A, axis=1)
    np.testing.assert_allclose(p1, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    p0 = xp.reductions.cumulative_prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0]]))


def test_cumulative_sum_coo():
    A = sample_coo()
    s1 = xp.reductions.cumulative_sum(A, axis=1)
    np.testing.assert_allclose(s1, np.array([[1.0, 1.0, 4.0], [0.0, -2.0, -2.0]]))
    s0 = xp.reductions.cumulative_sum(A, axis=0)
    np.testing.assert_allclose(s0, np.array([[1.0, 0.0, 3.0], [1.0, -2.0, 3.0]]))


def test_cumulative_prod_coo():
    A = sample_coo()
    p1 = xp.reductions.cumulative_prod(A, axis=1)
    np.testing.assert_allclose(p1, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    p0 = xp.reductions.cumulative_prod(A, axis=0)
    np.testing.assert_allclose(p0, np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0]]))


def test_cumulative_axis_errors():
    A = sample_csr()
    # axis=None not supported for sparse in first phase
    try:
        xp.reductions.cumulative_sum(A)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    try:
        xp.reductions.cumulative_prod(A)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    # invalid axis
    try:
        xp.reductions.cumulative_sum(A, axis=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        xp.reductions.cumulative_prod(A, axis=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
