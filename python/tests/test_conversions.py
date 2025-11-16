import numpy as np

from lacuna import _core as core


def mk_csr_parts():
    # A = [[1,0,2],[0,3,0]] CSR
    nrows, ncols = 2, 3
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return nrows, ncols, indptr, indices, data


def test_csr_csc_roundtrip():
    nrows, ncols, indptr, indices, data = mk_csr_parts()
    ci, cj, cv, cr, cc = core.csr_to_csc_from_parts(nrows, ncols, indptr, indices, data, True)
    # Back to CSR
    ri, rj, rv, rr, rc = core.csc_to_csr_from_parts(cr, cc, ci, cj, cv, True)
    assert rr == nrows and rc == ncols
    assert np.array_equal(ri, indptr)
    assert np.array_equal(rj, indices)
    np.testing.assert_allclose(rv, data)


def test_csr_coo_roundtrip():
    nrows, ncols, indptr, indices, data = mk_csr_parts()
    r, c, v, nr, nc = core.csr_to_coo_from_parts(nrows, ncols, indptr, indices, data, True)
    # Back to CSR
    ri, rj, rv, rr, rc = core.coo_to_csr_from_parts(nr, nc, r, c, v, True)
    assert rr == nrows and rc == ncols
    assert np.array_equal(ri, indptr)
    assert np.array_equal(rj, indices)
    np.testing.assert_allclose(rv, data)


def test_csc_coo_roundtrip():
    # Build CSC of the same A
    ci = np.array([0, 1, 2, 3], dtype=np.int64)
    cj = np.array([0, 1, 0], dtype=np.int64)
    cv = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    cr, cc = 2, 3
    r, c, v, nr, nc = core.csc_to_coo_from_parts(cr, cc, ci, cj, cv, True)
    # Back to CSC
    oi, oj, ov, orr, occ = core.coo_to_csc_from_parts(nr, nc, r, c, v, True)
    assert orr == cr and occ == cc
    assert np.array_equal(oi, ci)
    assert np.array_equal(oj, cj)
    np.testing.assert_allclose(ov, cv)
