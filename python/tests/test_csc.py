import numpy as np
import pytest

from lacuna.sparse.csc import CSC


def make_simple_csc():
    # A = [[1,0,2],[0,3,0]] in CSC
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    A = CSC(indptr, indices, data, (2, 3), check=False)
    return A


def test_csc_spmv():
    A = make_simple_csc()
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = A @ x
    np.testing.assert_allclose(y, np.array([1 * 10 + 2 * 30, 3 * 20], dtype=np.float64))


def test_csc_sum_and_axes():
    A = make_simple_csc()
    assert A.sum() == pytest.approx(6.0)
    np.testing.assert_allclose(A.sum(axis=1), np.array([3.0, 3.0]))
    np.testing.assert_allclose(A.sum(axis=0), np.array([1.0, 3.0, 2.0]))


def test_csc_prune_and_eliminate():
    # column 0 -> [0]=1.0, column1 -> [1]=0.0, column2 -> [0]=1e-9
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 0.0, 1e-9], dtype=np.float64)
    A = CSC(indptr, indices, data, (2, 3), check=False)
    Az = A.eliminate_zeros()
    assert Az.nnz == 2
    Ap = A.prune(1e-6)
    assert Ap.nnz == 1


def test_csc_scalar_add_sub_hadamard():
    A = make_simple_csc()
    C = A + A
    assert C.sum() == pytest.approx(12.0)
    S = A - A
    np.testing.assert_allclose(S.toarray(), np.zeros_like(A.toarray()))
    H = A.multiply(A)
    np.testing.assert_allclose(H.toarray(), A.toarray() * A.toarray())


def test_csc_scalar_mul():
    A = make_simple_csc()
    M = 2.0 * A
    np.testing.assert_allclose(M.toarray(), 2.0 * A.toarray())


def test_csc_spmm():
    A = make_simple_csc()
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)  # (3,2)
    Y = A @ B
    np.testing.assert_allclose(Y, np.array([[11.0, 14.0], [9.0, 12.0]], dtype=np.float64))
