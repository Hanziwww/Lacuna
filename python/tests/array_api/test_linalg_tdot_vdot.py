import numpy as np
import pytest

from lacuna.array_api import tensordot, vecdot
from lacuna.sparse import COOND, CSR


def test_csr_tensordot_vector_and_tensor():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0])
    A = CSR(indptr, indices, data, shape=(2, 3))

    v = np.array([1.0, 2.0, 3.0])
    out_v = tensordot(A, v, axes=1)
    np.testing.assert_allclose(out_v, A @ v)

    B = np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4)
    out = tensordot(A, B, axes=([1], [0]))
    expect = np.tensordot(A.toarray(), B, axes=([1], [0]))
    np.testing.assert_allclose(out, expect)


def test_csr_vecdot_axes():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0])
    A = CSR(indptr, indices, data, shape=(2, 3))

    vcols = np.array([1.0, 2.0, 3.0])
    out_row = vecdot(A, vcols, axis=1)
    np.testing.assert_allclose(out_row, A @ vcols)

    vrows = np.array([4.0, 5.0])
    out_col = vecdot(A, vrows, axis=0)
    expect = A.toarray().T @ vrows
    np.testing.assert_allclose(out_col, expect)


def _dense_from_coond(shape, indices, data):
    arr = np.zeros(shape, dtype=np.float64)
    ndim = len(shape)
    idx = indices.reshape(-1, ndim)
    for k, pos in enumerate(idx):
        arr[tuple(int(i) for i in pos)] += data[k]
    return arr


def test_coond_tensordot_and_vecdot():
    shape = (2, 3, 4)
    idx = np.array(
        [
            0,
            1,
            2,
            1,
            2,
            3,
            1,
            0,
            0,
        ],
        dtype=np.int64,
    )
    val = np.array([1.0, 3.0, 2.5])
    A = COOND(shape, idx, val)

    # tensordot along axis 1 with dense (3, 5)
    Y = np.arange(3 * 5, dtype=np.float64).reshape(3, 5)
    T = tensordot(A, Y, axes=([1], [0]))
    dense_A = _dense_from_coond(shape, A.indices, A.data)
    expect_T = np.tensordot(dense_A, Y, axes=([1], [0]))
    # densify T
    dense_T = _dense_from_coond(T.shape, T.indices, T.data)
    np.testing.assert_allclose(dense_T, expect_T)

    # vecdot along axis 2 with dense vector length 4
    v = np.array([1.0, 0.5, -1.0, 2.0])
    V = vecdot(A, v, axis=2)
    expect_V = np.tensordot(dense_A, v, axes=([2], [0]))
    dense_V = _dense_from_coond(V.shape, V.indices, V.data)
    np.testing.assert_allclose(dense_V, expect_V)
