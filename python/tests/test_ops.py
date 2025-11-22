import numpy as np
import pytest

from lacuna.sparse.csr import CSR


def make_simple():
    # A = [[1,0,2],[0,3,0]]
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    A = CSR(indptr, indices, data, (2, 3), check=False)
    return A


def test_spmv():
    A = make_simple()
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = A @ x
    np.testing.assert_allclose(y, np.array([1 * 10 + 2 * 30, 3 * 20], dtype=np.float64))


def test_spmm():
    A = make_simple()
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)  # (3,2)
    Y = A @ B
    np.testing.assert_allclose(Y, np.array([[11.0, 14.0], [9.0, 12.0]], dtype=np.float64))


def test_sum_and_axes():
    A = make_simple()
    assert A.sum() == pytest.approx(6.0)
    np.testing.assert_allclose(A.sum(axis=1), np.array([3.0, 3.0]))
    np.testing.assert_allclose(A.sum(axis=0), np.array([1.0, 3.0, 2.0]))


def test_transpose_and_add():
    A = make_simple()
    AT = A.T
    assert AT.shape == (3, 2)
    C = A + A
    assert C.sum() == pytest.approx(12.0)


def test_prune_and_eliminate():
    indptr = np.array([0, 3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, 0.0, 1e-9], dtype=np.float64)
    A = CSR(indptr, indices, data, (1, 3), check=False)
    Az = A.eliminate_zeros()
    assert Az.nnz == 2
    Ap = A.prune(1e-6)
    assert Ap.nnz == 1


def test_subtraction():
    A = make_simple()
    Z = A - A
    assert Z.nnz == 0
    np.testing.assert_allclose(Z.toarray(), np.zeros_like(A.toarray()))


def test_hadamard_multiply():
    A = make_simple()
    H = A.multiply(A)
    np.testing.assert_allclose(H.toarray(), A.toarray() * A.toarray())


def test_indexing():
    A = make_simple()
    assert A[0, 0] == pytest.approx(1.0)
    assert A[0, 1] == 0.0
    row0 = A[0, :]
    np.testing.assert_allclose(row0, np.array([1.0, 0.0, 2.0]))
    col1 = A[:, 1]
    np.testing.assert_allclose(col1, np.array([0.0, 3.0]))


def test_divide_elementwise_csr():
    A = make_simple()
    D = A.divide(A)
    exp = (A.toarray() != 0.0).astype(float)
    np.testing.assert_allclose(D.toarray(), exp)


def test_scalar_division_csr():
    A = make_simple()
    S = A / 2.0
    np.testing.assert_allclose(S.toarray(), A.toarray() / 2.0)
