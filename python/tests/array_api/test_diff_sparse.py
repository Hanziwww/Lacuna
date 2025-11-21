import numpy as np
import pytest

from lacuna.array_api import reductions as xp
from lacuna.sparse import COO, CSC, CSR


def make_mats():
    # 3x4 matrix with some zeros implicitly and explicitly
    # [[1, 0, 2, -1],
    #  [0, 0, 0,  3],
    #  [4, 5, 0,  0]]
    # CSR
    csr = CSR(
        np.array([0, 3, 4, 6], dtype=np.int64),
        np.array([0, 2, 3, 3, 0, 1], dtype=np.int64),
        np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0], dtype=np.float64),
        shape=(3, 4),
        check=False,
    )
    # CSC
    csc = CSC(
        np.array([0, 2, 3, 4, 6], dtype=np.int64),
        np.array([0, 2, 0, 0, 1, 2], dtype=np.int64),
        np.array([1.0, 4.0, 5.0, 2.0, -1.0, 3.0], dtype=np.float64),
        shape=(3, 4),
        check=False,
    )
    # COO
    coo = COO(
        np.array([0, 0, 0, 1, 2, 2], dtype=np.int64),
        np.array([0, 2, 3, 3, 0, 1], dtype=np.int64),
        np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0], dtype=np.float64),
        shape=(3, 4),
        check=False,
    )
    return csr, csc, coo


@pytest.mark.parametrize("which", ["csr", "csc", "coo"])
def test_diff_axis1_matches_numpy(which):
    csr, csc, coo = make_mats()
    a = {"csr": csr, "csc": csc, "coo": coo}[which]
    A = a.toarray()
    # n=1
    s1 = xp.diff(a, n=1, axis=1)
    np.testing.assert_allclose(s1.toarray(), np.diff(A, n=1, axis=1))
    # n=2
    s2 = xp.diff(a, n=2, axis=1)
    np.testing.assert_allclose(s2.toarray(), np.diff(A, n=2, axis=1))


@pytest.mark.parametrize("which", ["csr", "csc", "coo"])
def test_diff_axis0_matches_numpy(which):
    csr, csc, coo = make_mats()
    a = {"csr": csr, "csc": csc, "coo": coo}[which]
    A = a.toarray()
    # n=1
    s1 = xp.diff(a, n=1, axis=0)
    np.testing.assert_allclose(s1.toarray(), np.diff(A, n=1, axis=0))
    # n=2
    s2 = xp.diff(a, n=2, axis=0)
    np.testing.assert_allclose(s2.toarray(), np.diff(A, n=2, axis=0))


@pytest.mark.parametrize("which", ["csr", "csc", "coo"])
def test_diff_negative_axis(which):
    csr, csc, coo = make_mats()
    a = {"csr": csr, "csc": csc, "coo": coo}[which]
    A = a.toarray()
    s1 = xp.diff(a, n=1, axis=-1)
    np.testing.assert_allclose(s1.toarray(), np.diff(A, n=1, axis=-1))
    s0 = xp.diff(a, n=1, axis=-2)
    np.testing.assert_allclose(s0.toarray(), np.diff(A, n=1, axis=-2))


@pytest.mark.parametrize("which", ["csr", "csc", "coo"])
def test_diff_n_ge_dim(which):
    csr, csc, coo = make_mats()
    a = {"csr": csr, "csc": csc, "coo": coo}[which]
    A = a.toarray()
    s = xp.diff(a, n=a.shape[1], axis=1)
    assert s.shape == (a.shape[0], 0)
    assert np.diff(A, n=a.shape[1], axis=1).shape == (a.shape[0], 0)
    s = xp.diff(a, n=a.shape[0], axis=0)
    assert s.shape == (0, a.shape[1])
    assert np.diff(A, n=a.shape[0], axis=0).shape == (0, a.shape[1])


def test_diff_invalid_axis_raises():
    a = make_mats()[0]
    with pytest.raises(ValueError):
        xp.diff(a, n=1, axis=2)
    with pytest.raises(ValueError):
        xp.diff(a, n=1, axis=-3)
