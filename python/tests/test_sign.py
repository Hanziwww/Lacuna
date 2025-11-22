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


def test_sign_csr_toarray_semantics():
    A = make_csr()
    S = A.sign()
    Ad = A.toarray()
    Sd = S.toarray()
    # Expected with numpy semantics
    exp = np.sign(Ad)
    # Compare NaN locations and finite values separately
    np.testing.assert_array_equal(np.isnan(Sd), np.isnan(exp))
    np.testing.assert_allclose(Sd[~np.isnan(exp)], exp[~np.isnan(exp)])


def test_sign_csc_toarray_semantics():
    A = make_csc()
    S = A.sign()
    Ad = A.toarray()
    Sd = S.toarray()
    exp = np.sign(Ad)
    np.testing.assert_array_equal(np.isnan(Sd), np.isnan(exp))
    np.testing.assert_allclose(Sd[~np.isnan(exp)], exp[~np.isnan(exp)])


def test_sign_coo_toarray_semantics():
    A = make_coo()
    S = A.sign()
    Ad = A.toarray()
    Sd = S.toarray()
    exp = np.sign(Ad)
    np.testing.assert_array_equal(np.isnan(Sd), np.isnan(exp))
    np.testing.assert_allclose(Sd[~np.isnan(exp)], exp[~np.isnan(exp)])


def test_sign_coond_data_semantics():
    shape = (2, 2, 2)
    idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    data = np.array([1.0, -2.0], dtype=np.float64)
    X = COOND(shape, idx, data)
    Y = X.sign()
    np.testing.assert_allclose(Y.data, np.sign(X.data))
    assert Y.shape == X.shape


def test_array_api_sign_dispatch():
    A = make_csr()
    S = xp.sign(A)
    np.testing.assert_array_equal(np.isnan(S.toarray()), np.isnan(np.sign(A.toarray())))


def test_namespace_capability_has_sign():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert "sign" in caps.get("elementwise", [])
