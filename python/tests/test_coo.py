import numpy as np
import pytest

from lacuna.sparse.coo import COO


def make_simple_coo():
    # A = [[1,0,2],[0,3,0]] in COO
    row = np.array([0, 1, 0], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    A = COO(row, col, data, (2, 3), check=False)
    return A


def test_coo_spmv():
    A = make_simple_coo()
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = A @ x
    np.testing.assert_allclose(y, np.array([1 * 10 + 2 * 30, 3 * 20], dtype=np.float64))


def test_coo_sum_and_axes():
    A = make_simple_coo()
    assert A.sum() == pytest.approx(6.0)
    np.testing.assert_allclose(A.sum(axis=1), np.array([3.0, 3.0]))
    np.testing.assert_allclose(A.sum(axis=0), np.array([1.0, 3.0, 2.0]))


def test_coo_prune_and_eliminate():
    row = np.array([0, 0, 0], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, 0.0, 1e-9], dtype=np.float64)
    A = COO(row, col, data, (1, 3), check=False)
    Az = A.eliminate_zeros()
    assert Az.nnz == 2
    Ap = A.prune(1e-6)
    assert Ap.nnz == 1


def test_coo_scalar_mul_and_toarray():
    A = make_simple_coo()
    M = 2.0 * A
    np.testing.assert_allclose(M.toarray(), 2.0 * A.toarray())

    # duplicates accumulation check
    row = np.array([0, 0], dtype=np.int64)
    col = np.array([1, 1], dtype=np.int64)
    data = np.array([2.0, 3.0], dtype=np.float64)
    D = COO(row, col, data, (1, 3), check=False)
    arr = D.toarray()
    np.testing.assert_allclose(arr, np.array([[0.0, 5.0, 0.0]], dtype=np.float64))


def test_coo_scalar_div():
    A = make_simple_coo()
    S = A / 2.0
    np.testing.assert_allclose(S.toarray(), A.toarray() / 2.0)


def test_coo_spmm():
    A = make_simple_coo()
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)  # (3,2)
    Y = A @ B
    np.testing.assert_allclose(Y, np.array([[11.0, 14.0], [9.0, 12.0]], dtype=np.float64))
