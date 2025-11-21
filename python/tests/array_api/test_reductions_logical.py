import numpy as np
import pytest

from lacuna.array_api import reductions as xp
from lacuna.sparse import COO, CSC, CSR


def _mk_csr():
    # [[1, 0, 2],
    #  [0, 0, 3]]
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 2], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return CSR(indptr, indices, data, shape=(2, 3), check=False)


def _mk_csc():
    # same matrix in CSC
    indptr = np.array([0, 1, 1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    data = np.array([1.0, 3.0], dtype=np.float64)
    return CSC(indptr, indices, data, shape=(2, 3), check=False)


def _mk_coo():
    # same matrix in COO
    row = np.array([0, 0, 1], dtype=np.int64)
    col = np.array([0, 2, 2], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return COO(row, col, data, shape=(2, 3), check=False)


@pytest.mark.parametrize("maker", [_mk_csr, _mk_csc, _mk_coo])
@pytest.mark.parametrize("keepdims", [False, True])
def test_all_any_global_and_axes(maker, keepdims):
    a = maker()
    # Dense reference
    A = a.toarray()

    # global
    res_any = xp.any(a, axis=None, keepdims=keepdims)
    res_all = xp.all(a, axis=None, keepdims=keepdims)
    if keepdims:
        np.testing.assert_array_equal(
            np.asarray(res_any), np.array(np.array(A).any()).reshape(1, 1)
        )
        np.testing.assert_array_equal(
            np.asarray(res_all), np.array(np.array(A).all()).reshape(1, 1)
        )
    else:
        assert bool(res_any) == bool(np.array(A).any())
        assert bool(res_all) == bool(np.array(A).all())

    # axis=0 and axis=1
    col_any = xp.any(a, axis=0, keepdims=keepdims)
    row_any = xp.any(a, axis=1, keepdims=keepdims)
    np.testing.assert_array_equal(
        np.asarray(col_any), np.asarray(A).any(axis=0, keepdims=keepdims)
    )
    np.testing.assert_array_equal(
        np.asarray(row_any), np.asarray(A).any(axis=1, keepdims=keepdims)
    )

    col_all = xp.all(a, axis=0, keepdims=keepdims)
    row_all = xp.all(a, axis=1, keepdims=keepdims)
    np.testing.assert_array_equal(
        np.asarray(col_all), np.asarray(A).all(axis=0, keepdims=keepdims)
    )
    np.testing.assert_array_equal(
        np.asarray(row_all), np.asarray(A).all(axis=1, keepdims=keepdims)
    )


@pytest.mark.parametrize("maker", [_mk_csr, _mk_csc, _mk_coo])
def test_all_any_shapes_keepdims(maker):
    a = maker()
    # keepdims shapes
    assert np.asarray(xp.any(a, axis=0, keepdims=True)).shape == (1, a.shape[1])
    assert np.asarray(xp.any(a, axis=1, keepdims=True)).shape == (a.shape[0], 1)
    assert np.asarray(xp.all(a, axis=0, keepdims=True)).shape == (1, a.shape[1])
    assert np.asarray(xp.all(a, axis=1, keepdims=True)).shape == (a.shape[0], 1)


@pytest.mark.parametrize("maker", [_mk_csr, _mk_csc, _mk_coo])
def test_all_any_edge_empty_axis(maker):
    # ncols = 0 -> all over axis=1 should be True, any should be False
    a = maker()
    if a.shape[1] == 0:
        pytest.skip("maker does not support zero columns")
    # Build empty ncols=0 of same rows
    if isinstance(a, CSR):
        indptr = np.zeros(a.shape[0] + 1, dtype=np.int64)
        b = CSR(indptr, np.array([], np.int64), np.array([], np.float64), shape=(a.shape[0], 0))
    elif isinstance(a, CSC):
        indptr = np.zeros(1, dtype=np.int64)
        b = CSC(indptr, np.array([], np.int64), np.array([], np.float64), shape=(a.shape[0], 0))
    else:
        b = COO(
            np.array([], np.int64),
            np.array([], np.int64),
            np.array([], np.float64),
            shape=(a.shape[0], 0),
        )
    np.testing.assert_array_equal(xp.all(b, axis=1), np.ones((a.shape[0],), dtype=bool))
    np.testing.assert_array_equal(xp.any(b, axis=1), np.zeros((a.shape[0],), dtype=bool))
