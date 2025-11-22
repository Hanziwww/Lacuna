import numpy as np

import lacuna.array_api as xp
from lacuna.sparse import CSR


def make_sample_csr():
    # 2x3 matrix with nnz=3
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    data = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    return CSR(indptr, indices, data, shape=(2, 3))


def test_reductions_sum_mean_keepdims():
    A = make_sample_csr()

    # sum
    assert xp.sum(A) == 6.0
    np.testing.assert_allclose(xp.sum(A, axis=0), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(xp.sum(A, axis=1), np.array([4.0, 2.0]))
    np.testing.assert_allclose(xp.sum(A, axis=0, keepdims=True), np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(xp.sum(A, axis=1, keepdims=True), np.array([[4.0], [2.0]]))

    # mean
    assert xp.mean(A) == 6.0 / (2 * 3)
    np.testing.assert_allclose(xp.mean(A, axis=0), np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose(xp.mean(A, axis=1), np.array([4.0 / 3.0, 2.0 / 3.0]))


def test_searching_count_nonzero():
    A = make_sample_csr()
    assert xp.count_nonzero(A) == 3
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=0), np.array([1, 1, 1]))
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=1), np.array([2, 1]))
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=0, keepdims=True), np.array([[1, 1, 1]]))
    np.testing.assert_array_equal(xp.count_nonzero(A, axis=1, keepdims=True), np.array([[2], [1]]))


def test_linalg_matmul_and_transpose():
    A = make_sample_csr()

    # SpMV
    x = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    y = xp.matmul(A, x)
    np.testing.assert_allclose(y, np.array([4.0, 0.0]))

    # SpMM
    B = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    C = xp.matmul(A, B)
    np.testing.assert_allclose(C, np.array([[4.0, 11.0], [0.0, 0.0]]))

    # Transpose
    AT = xp.matrix_transpose(A)
    assert AT.shape == (3, 2)


def test_elementwise_add_sub_mul():
    A = make_sample_csr()

    # A + A
    C = xp.add(A, A)
    np.testing.assert_allclose(C.toarray(), 2.0 * A.toarray())

    # A - A
    Z = xp.subtract(A, A)
    np.testing.assert_allclose(Z.toarray(), np.zeros_like(A.toarray()))

    # Hadamard A * A
    H = xp.multiply(A, A)
    np.testing.assert_allclose(H.toarray(), A.toarray() * A.toarray())

    # scalar * A, A * scalar
    S1 = xp.multiply(2.0, A)
    S2 = xp.multiply(A, 2.0)
    np.testing.assert_allclose(S1.toarray(), 2.0 * A.toarray())
    np.testing.assert_allclose(S2.toarray(), 2.0 * A.toarray())


def test_elementwise_divide_and_guards():
    A = make_sample_csr()
    # CSR / CSR works
    D = xp.divide(A, A)
    exp = (A.toarray() != 0.0).astype(float)
    np.testing.assert_allclose(D.toarray(), exp)

    # sparse / scalar works
    S = xp.divide(A, 2.0)
    np.testing.assert_allclose(S.toarray(), A.toarray() / 2.0)

    # scalar / sparse not supported
    try:
        _ = xp.divide(2.0, A)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("scalar / sparse should raise NotImplementedError")


def test_elementwise_floor_divide_and_guards():
    A = make_sample_csr()
    F = xp.floor_divide(A, A)
    exp = (A.toarray() != 0.0).astype(float)
    np.testing.assert_allclose(F.toarray(), exp)

    S = xp.floor_divide(A, 2.0)
    np.testing.assert_allclose(S.toarray(), np.floor(A.toarray() / 2.0))

    try:
        _ = xp.floor_divide(2.0, A)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("scalar // sparse should raise NotImplementedError")

    from lacuna.sparse import CSC

    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    B = CSC(indptr, indices, data, shape=(2, 3))
    FB = xp.floor_divide(B, B)
    np.testing.assert_allclose(FB.toarray(), (B.toarray() != 0.0).astype(float))
