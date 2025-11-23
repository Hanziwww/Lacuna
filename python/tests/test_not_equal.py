import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import COO, COOND, CSC, CSR


def make_csr():
    indptr = np.array([0, 2, 4], dtype=np.int64)
    indices = np.array([0, 2, 0, 1], dtype=np.int64)
    data = np.array([1.0, -2.0, 0.0, np.nan], dtype=np.float64)
    return CSR(indptr, indices, data, (2, 3), check=False)


def make_csc():
    indptr = np.array([0, 2, 3, 4], dtype=np.int64)
    indices = np.array([0, 1, 1, 0], dtype=np.int64)
    data = np.array([1.0, -0.0, -3.0, np.nan], dtype=np.float64)
    return CSC(indptr, indices, data, (2, 3), check=False)


def make_coo():
    row = np.array([0, 1, 0, 1], dtype=np.int64)
    col = np.array([0, 0, 2, 1], dtype=np.int64)
    data = np.array([1.0, -2.0, 0.0, np.nan], dtype=np.float64)
    return COO(row, col, data, (2, 3), check=False)


def test_not_equal_csr_dense():
    A = make_csr()
    Bd = make_csr().toarray()
    Bd[1, 1] = -2.0
    R = xp.not_equal(A, Bd)
    exp = np.not_equal(A.toarray(), Bd)
    assert R.dtype == exp.dtype
    np.testing.assert_array_equal(R, exp)


essential_zero = 0.0


def test_not_equal_csr_scalar_both_orders():
    A = make_csr()
    R1 = xp.not_equal(A, essential_zero)
    R2 = xp.not_equal(essential_zero, A)
    exp = np.not_equal(A.toarray(), essential_zero)
    np.testing.assert_array_equal(R1, exp)
    np.testing.assert_array_equal(R2, exp)


def test_not_equal_coo_dense():
    A = make_coo()
    D = np.array([[1.0, 0.0, 2.0], [0.0, -2.0, 0.0]], dtype=np.float64)
    R = xp.not_equal(A, D)
    exp = np.not_equal(A.toarray(), D)
    np.testing.assert_array_equal(R, exp)


def test_not_equal_coond_scalar():
    shape = (2, 2, 2)
    idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    data = np.array([1.0, -2.0], dtype=np.float64)
    X = COOND(shape, idx, data)
    R = xp.not_equal(X, 0.0)
    # Densify COOND for expected
    arr = np.zeros(shape, dtype=np.float64)
    nnz = data.size
    ndim = len(shape)
    idm = idx.reshape(nnz, ndim)
    np.add.at(arr, tuple(idm.T), data)
    exp = np.not_equal(arr, 0.0)
    np.testing.assert_array_equal(R, exp)


def test_namespace_capability_has_not_equal():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert "not_equal" in caps.get("elementwise", [])
