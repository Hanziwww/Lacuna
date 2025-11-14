import os

import numpy as np
import pytest

from lacuna import get_num_threads, set_num_threads
from lacuna.sparse.csr import CSR


def make_simple():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return CSR(indptr, indices, data, (2, 3), check=False)


def test_mul_scalar():
    A = make_simple()
    B = 2.0 * A
    C = A * 2.0
    np.testing.assert_allclose(B.data, 2.0 * A.data)
    np.testing.assert_allclose(C.data, 2.0 * A.data)


def test_astype_float64():
    A = make_simple()
    B = A.astype(np.float64)
    assert B.data.dtype == np.float64
    np.testing.assert_allclose(B.toarray(), A.toarray())
    with pytest.raises(NotImplementedError):
        _ = A.astype(np.float32)


def test_spmv_shape_error():
    A = make_simple()
    x = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _ = A @ x


def test_spmm_shape_error():
    A = make_simple()
    B = np.ones((A.shape[1] + 1, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        _ = A @ B


def test_index_out_of_bounds_and_advanced_index():
    A = make_simple()
    with pytest.raises(IndexError):
        _ = A[-1, 0]
    with pytest.raises(NotImplementedError):
        _ = A[[0, 1], :]


def test_transpose_values():
    A = make_simple()
    np.testing.assert_allclose(A.T.toarray(), A.toarray().T)


def test_duplicates_add_coalesce():
    indptr = np.array([0, 3], dtype=np.int64)
    indices = np.array([0, 0, 2], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    A = CSR(indptr, indices, data, (1, 3), check=False)
    C = A + A
    np.testing.assert_array_equal(C.indptr, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(C.indices, np.array([0, 2], dtype=np.int64))
    np.testing.assert_allclose(C.data, np.array([8.0, 4.0]))


def test_empty_and_zero_dim():
    A = CSR(
        np.array([0, 0, 0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        (2, 3),
        check=True,
    )
    assert A.nnz == 0
    np.testing.assert_allclose(A.toarray(), np.zeros((2, 3)))
    AT = A.T
    assert AT.shape == (3, 2)

    Zr = CSR(
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        (0, 3),
        check=True,
    )
    np.testing.assert_allclose(Zr.toarray(), np.zeros((0, 3)))

    Zc = CSR(
        np.array([0, 0, 0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        (2, 0),
        check=True,
    )
    np.testing.assert_allclose(Zc.toarray(), np.zeros((2, 0)))


def test_threads_runtime():
    set_num_threads(1)
    assert get_num_threads() == 1
    assert os.environ.get("RAYON_NUM_THREADS") == "1"
