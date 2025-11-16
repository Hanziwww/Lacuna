import numpy as np

from lacuna.sparse import COOND


def sample_coond():
    # shape [2,3,2], entries: (0,0,0)=1, (0,2,1)=2, (1,1,0)=3
    shape = np.array([2, 3, 2], dtype=np.int64)
    indices = np.array(
        [
            0,
            0,
            0,
            0,
            2,
            1,
            1,
            1,
            0,
        ],
        dtype=np.int64,
    )
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return shape, indices, data


def test_nd_sum_reduce_permute():
    shape, indices, data = sample_coond()
    a = COOND(shape, indices, data, check=False)
    s = a.sum()
    assert abs(s - 6.0) < 1e-12

    # reduce axes=[2] -> shape [2,3]
    r = a.reduce_sum_axes([2])
    assert r.shape == (2, 3)
    assert abs(r.data.sum() - s) < 1e-12

    # permute [1,0,2] -> shape [3,2,2], sum preserved
    p = a.permute_axes([1, 0, 2])
    assert p.shape == (3, 2, 2)
    assert abs(p.data.sum() - s) < 1e-12


def test_nd_convert_mode_and_axes():
    shape, indices, data = sample_coond()
    a = COOND(shape, indices, data, check=False)
    s = a.sum()

    # mode-0 unfolding -> CSR: rows=2, cols=3*2=6
    csr0 = a.mode_unfold_to_csr(0)
    assert csr0.shape == (2, 6)
    assert abs(csr0.data.sum() - s) < 1e-12

    # mode-1 unfolding -> CSC: rows=3, cols=2*2=4
    csc1 = a.mode_unfold_to_csc(1)
    assert csc1.shape == (3, 4)
    assert abs(csc1.data.sum() - s) < 1e-12

    # axes unfolding rows=[0,2], cols=[1]
    csr_axes = a.axes_unfold_to_csr([0, 2])
    assert csr_axes.shape == (2 * 2, 3)
    assert abs(csr_axes.data.sum() - s) < 1e-12


def test_nd_mean_and_reduce_mean():
    shape, indices, data = sample_coond()
    a = COOND(shape, indices, data, check=False)
    s = a.sum()
    total = int(np.prod(shape))
    m = a.mean()
    assert abs(m - s / float(total)) < 1e-12

    r = a.reduce_mean_axes([2])
    # sum of reduced mean equals sum/size_of_axis
    assert r.shape == (2, 3)
    assert abs(r.data.sum() - s / float(shape[2])) < 1e-12


def test_nd_reshape():
    shape, indices, data = sample_coond()
    a = COOND(shape, indices, data, check=False)
    s = a.sum()
    b = a.reshape((3, 2, 2))
    assert b.shape == (3, 2, 2)
    assert abs(b.data.sum() - s) < 1e-12


def test_nd_hadamard_broadcast():
    # a: from sample
    shape, indices, data = sample_coond()
    a = COOND(shape, indices, data, check=False)
    # b: [1,3,1], indices along j dimension, values [10,20,30]
    b = COOND(
        (1, 3, 1),
        np.array([0, 0, 0, 0, 1, 0, 0, 2, 0], dtype=np.int64),
        np.array([10.0, 20.0, 30.0], dtype=np.float64),
        check=False,
    )
    h = a.hadamard_broadcast(b)
    assert h.shape == (2, 3, 2)
    # expected: 1*10 + 2*30 + 3*20 = 130
    assert abs(h.data.sum() - 130.0) < 1e-12
