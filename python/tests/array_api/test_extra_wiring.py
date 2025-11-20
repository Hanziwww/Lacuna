import numpy as np
import pytest

import lacuna.array_api as xp
from lacuna.sparse import COOND, CSR


def make_csr():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    return CSR(indptr, indices, data, shape=(2, 3))


def test_coond_permute_dims_and_reshape():
    shape = (2, 3, 4)
    idx = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)  # two entries: (0,1,2)=1, (1,2,3)=3
    data = np.array([1.0, 3.0], dtype=np.float64)
    X = COOND(shape, idx, data)

    # permute dims
    Y = xp.permute_dims(X, [2, 1, 0])
    assert Y.shape == (4, 3, 2)
    # sum invariant under permutation
    assert xp.sum(Y) == xp.sum(X)

    # reshape (compatible)
    Z = xp.reshape(X, (3, 2, 4))
    assert Z.shape == (3, 2, 4)
    assert xp.sum(Z) == xp.sum(X)


def test_coond_broadcast_multiply():
    # A shape (2,3,1), only last dim is 1; nonzero at (1,2,0)=2.0
    A_shape = (2, 3, 1)
    A_idx = np.array([1, 2, 0], dtype=np.int64)
    A_val = np.array([2.0], dtype=np.float64)
    A = COOND(A_shape, A_idx, A_val)

    # B shape (1,3,4), nonzeros at (0,2,1)=3.0 and (0,2,3)=5.0
    B_shape = (1, 3, 4)
    B_idx = np.array([0, 2, 1, 0, 2, 3], dtype=np.int64)
    B_val = np.array([3.0, 5.0], dtype=np.float64)
    B = COOND(B_shape, B_idx, B_val)

    C = xp.multiply(A, B)
    assert C.shape == (2, 3, 4)
    # Expect two results at (1,2,1)=6 and (1,2,3)=10
    assert C.nnz == 2
    # Build pairs to check values independent of ordering
    c_idx = C.indices.reshape(C.nnz, C.ndim)
    c_val = C.data
    pairs = {(int(i), int(j), int(k)): float(v) for (i, j, k), v in zip(c_idx, c_val)}
    assert pairs.get((1, 2, 1)) == pytest.approx(6.0)
    assert pairs.get((1, 2, 3)) == pytest.approx(10.0)


def test_csr_astype():
    A = make_csr()
    B = xp.astype(A, np.float64)
    assert isinstance(B, CSR)
    np.testing.assert_allclose(B.toarray(), A.toarray())

    with pytest.raises(NotImplementedError):
        xp.astype(A, np.float32)
