import numpy as np
import pytest

import lacuna.array_api as xp
from lacuna.sparse import COO, COOND, CSC, CSR


def make_csr():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return CSR(indptr, indices, data, (2, 3), check=False)


def make_csc():
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    return CSC(indptr, indices, data, (2, 3), check=False)


def make_coo():
    row = np.array([0, 1, 0], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    return COO(row, col, data, (2, 3), check=False)


def test_csr_scalar_pow():
    A = make_csr()
    R = A**2.0
    np.testing.assert_allclose(R.toarray(), A.toarray() ** 2.0)


def test_csr_pair_pow_masked():
    A = make_csr()
    B = CSR(A.indptr, A.indices, np.array([2.0, 2.0, 1.0]), A.shape, check=False)
    C = A.power(B)
    Ad = A.toarray()
    Bd = B.toarray()
    mask = (Ad != 0.0) & (Bd != 0.0)
    exp = np.zeros_like(Ad)
    exp[mask] = np.power(Ad[mask], Bd[mask])
    np.testing.assert_allclose(C.toarray(), exp)


def test_csc_scalar_pow():
    A = make_csc()
    R = A**2.0
    np.testing.assert_allclose(R.toarray(), A.toarray() ** 2.0)


def test_csc_pair_pow_masked():
    A = make_csc()
    B = CSC(
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([2.0, 1.0, 2.0], dtype=np.float64),
        (2, 3),
        check=False,
    )
    C = A.power(B)
    Ad = A.toarray()
    Bd = B.toarray()
    mask = (Ad != 0.0) & (Bd != 0.0)
    exp = np.zeros_like(Ad)
    exp[mask] = np.power(Ad[mask], Bd[mask])
    np.testing.assert_allclose(C.toarray(), exp)


def test_coo_scalar_pow():
    A = make_coo()
    R = A**3.0
    np.testing.assert_allclose(R.toarray(), A.toarray() ** 3.0)


def test_coond_scalar_pow():
    shape = (2, 3, 2)
    # entries: (0,0,0)=1, (0,2,1)=2, (1,1,0)=3
    idx = np.array([0, 0, 0, 0, 2, 1, 1, 1, 0], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    X = COOND(shape, idx, data)
    Y = X**2.0
    np.testing.assert_allclose(Y.data, X.data**2.0)
    assert Y.shape == X.shape


def test_array_api_pow_scalar_and_pair():
    A = make_csr()
    B = CSR(A.indptr, A.indices, np.array([2.0, 2.0, 1.0]), A.shape, check=False)
    R1 = xp.pow(A, 2.0)
    R2 = xp.pow(A, B)
    np.testing.assert_allclose(R1.toarray(), (A**2.0).toarray())
    np.testing.assert_allclose(R2.toarray(), A.power(B).toarray())


def test_array_api_pow_errors():
    A = make_csr()
    with pytest.raises(NotImplementedError):
        _ = xp.pow(2.0, A)


def test_namespace_capability_has_pow():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert "pow" in caps.get("elementwise", [])
