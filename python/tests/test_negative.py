import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import COO, COOND, CSC, CSR


def make_csr():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    return CSR(indptr, indices, data, (2, 3), check=False)


def make_csc():
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, -3.0, 2.5], dtype=np.float64)
    return CSC(indptr, indices, data, (2, 3), check=False)


def make_coo():
    row = np.array([0, 1, 0], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, -3.0, 2.0], dtype=np.float64)
    return COO(row, col, data, (2, 3), check=False)


def test_negative_csr():
    A = make_csr()
    R = -A
    np.testing.assert_allclose(R.toarray(), -A.toarray())
    np.testing.assert_allclose((-R).toarray(), A.toarray())


def test_negative_csc():
    A = make_csc()
    R = -A
    np.testing.assert_allclose(R.toarray(), -A.toarray())
    np.testing.assert_allclose((-R).toarray(), A.toarray())


def test_negative_coo():
    A = make_coo()
    R = -A
    np.testing.assert_allclose(R.toarray(), -A.toarray())
    np.testing.assert_allclose((-R).toarray(), A.toarray())


def test_negative_coond():
    shape = (2, 3, 2)
    idx = np.array([0, 0, 0, 0, 2, 1, 1, 1, 0], dtype=np.int64)
    data = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    X = COOND(shape, idx, data)
    Y = -X
    np.testing.assert_allclose(Y.data, -X.data)
    assert Y.shape == X.shape


def test_array_api_negative_dispatch():
    A = make_csr()
    R = xp.negative(A)
    np.testing.assert_allclose(R.toarray(), (-A).toarray())


def test_namespace_capability_has_negative():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert "negative" in caps.get("elementwise", [])
