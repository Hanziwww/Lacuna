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


def test_equal_csr_csr():
    A = make_csr()
    # B equal to A except change one entry to match equality expectations
    Bd = make_csr().toarray()
    # Make Bd different at (1,1) to ensure False where A is NaN vs finite
    Bd[1, 1] = -2.0
    # Compare CSR vs dense
    R = xp.equal(A, Bd)
    exp = np.equal(A.toarray(), Bd)
    assert R.dtype == exp.dtype
    np.testing.assert_array_equal(R, exp)


essential_zero = 0.0


def test_equal_csr_scalar_both_orders():
    A = make_csr()
    R1 = xp.equal(A, essential_zero)
    R2 = xp.equal(essential_zero, A)
    exp = np.equal(A.toarray(), essential_zero)
    np.testing.assert_array_equal(R1, exp)
    np.testing.assert_array_equal(R2, exp)


def test_equal_coo_dense():
    A = make_coo()
    D = np.array([[1.0, 0.0, 2.0], [0.0, -2.0, 0.0]], dtype=np.float64)
    R = xp.equal(A, D)
    exp = np.equal(A.toarray(), D)
    np.testing.assert_array_equal(R, exp)


def test_equal_coond_scalar():
    shape = (2, 2, 2)
    idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    data = np.array([1.0, -2.0], dtype=np.float64)
    X = COOND(shape, idx, data)
    R = xp.equal(X, 0.0)
    # Densify COOND for expected
    arr = np.zeros(shape, dtype=np.float64)
    nnz = data.size
    ndim = len(shape)
    idm = idx.reshape(nnz, ndim)
    np.add.at(arr, tuple(idm.T), data)
    exp = np.equal(arr, 0.0)
    np.testing.assert_array_equal(R, exp)


def test_namespace_capability_has_equal():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert "equal" in caps.get("elementwise", [])
