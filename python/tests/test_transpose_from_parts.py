import numpy as np
import pytest

from lacuna import _core as core

pytestmark = pytest.mark.skipif(core is None, reason="native core is not available")
from lacuna.sparse.coo import COO
from lacuna.sparse.csc import CSC


def test_core_transpose_csc_from_parts_basic():
    # A = [[1,0,2],[0,3,0]] in CSC
    nrows, ncols = 2, 3
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    ti, tj, tv, tr, tc = core.transpose_csc_from_parts(nrows, ncols, indptr, indices, data, True)
    assert (tr, tc) == (3, 2)
    np.testing.assert_array_equal(ti, np.array([0, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(tj, np.array([0, 2, 1], dtype=np.int64))
    np.testing.assert_allclose(tv, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_core_transpose_csc_from_parts_empty():
    nrows, ncols = 2, 3
    indptr = np.array([0, 0, 0, 0], dtype=np.int64)
    indices = np.array([], dtype=np.int64)
    data = np.array([], dtype=np.float64)
    ti, tj, tv, tr, tc = core.transpose_csc_from_parts(nrows, ncols, indptr, indices, data, True)
    assert (tr, tc) == (3, 2)
    np.testing.assert_array_equal(ti, np.array([0, 0, 0], dtype=np.int64))
    assert tj.size == 0 and tv.size == 0


def test_core_transpose_coo_from_parts_basic():
    # A = [[1,0,2],[0,3,0]] in COO
    nrows, ncols = 2, 3
    row = np.array([0, 1, 1], dtype=np.int64)
    col = np.array([0, 0, 2], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rr, cc, vv, nr, nc = core.transpose_coo_from_parts(nrows, ncols, row, col, data, True)
    assert (nr, nc) == (3, 2)
    np.testing.assert_array_equal(rr, np.array([0, 0, 2], dtype=np.int64))
    np.testing.assert_array_equal(cc, np.array([0, 1, 1], dtype=np.int64))
    np.testing.assert_allclose(vv, data)


def test_csc_T_fallback_matches_dense():
    # Build CSC and force fallback by clearing handle
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    A = CSC(indptr, indices, data, (2, 3), check=True)
    A._handle = None  # force from_parts path
    AT = A.T
    np.testing.assert_allclose(AT.toarray(), A.toarray().T)


essentially_empty = pytest.mark.parametrize(
    "shape,indptr,indices,data",
    [
        # CSC requires indptr length == ncols + 1
        (
            (0, 0),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        ),
        (
            (0, 3),
            np.array([0, 0, 0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        ),
        (
            (3, 0),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        ),
    ],
)


@essentially_empty
def test_csc_T_fallback_empty_cases(shape, indptr, indices, data):
    A = CSC(indptr, indices, data, shape, check=True)
    A._handle = None
    AT = A.T
    np.testing.assert_allclose(AT.toarray(), A.toarray().T)


def test_coo_T_fallback_matches_dense():
    row = np.array([0, 1, 1], dtype=np.int64)
    col = np.array([0, 0, 2], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    A = COO(row, col, data, (2, 3), check=True)
    A._handle = None
    AT = A.T
    np.testing.assert_allclose(AT.toarray(), A.toarray().T)
