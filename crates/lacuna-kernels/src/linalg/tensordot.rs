use crate::linalg::matmul::{spmm_coo_f64_i64, spmm_coond_f64_i64, spmm_csc_f64_i64, spmm_f64_i64};
use lacuna_core::{Coo, CooNd, Csc, Csr};

#[inline]
fn product(xs: &[usize]) -> usize {
    xs.iter().copied().fold(1usize, |acc, x| {
        acc.checked_mul(x).expect("shape product overflow")
    })
}

#[must_use]
pub fn tensordot_csr_dense_axes1x0_f64_i64(
    a: &Csr<f64, i64>,
    b: &[f64],
    b_shape: &[usize],
) -> Vec<f64> {
    assert!(!b_shape.is_empty(), "b_shape must be non-empty");
    let expected = product(b_shape);
    assert_eq!(b.len(), expected, "b length must equal product(b_shape)");
    assert_eq!(b_shape[0], a.ncols, "contracted dim must equal a.ncols");
    let k = product(&b_shape[1..]);
    spmm_f64_i64(a, b, k)
}

#[must_use]
pub fn tensordot_csc_dense_axes1x0_f64_i64(
    a: &Csc<f64, i64>,
    b: &[f64],
    b_shape: &[usize],
) -> Vec<f64> {
    assert!(!b_shape.is_empty(), "b_shape must be non-empty");
    let expected = product(b_shape);
    assert_eq!(b.len(), expected, "b length must equal product(b_shape)");
    assert_eq!(b_shape[0], a.ncols, "contracted dim must equal a.ncols");
    let k = product(&b_shape[1..]);
    spmm_csc_f64_i64(a, b, k)
}

#[must_use]
pub fn tensordot_coo_dense_axes1x0_f64_i64(
    a: &Coo<f64, i64>,
    b: &[f64],
    b_shape: &[usize],
) -> Vec<f64> {
    assert!(!b_shape.is_empty(), "b_shape must be non-empty");
    let expected = product(b_shape);
    assert_eq!(b.len(), expected, "b length must equal product(b_shape)");
    assert_eq!(b_shape[0], a.ncols, "contracted dim must equal a.ncols");
    let k = product(&b_shape[1..]);
    spmm_coo_f64_i64(a, b, k)
}

#[must_use]
pub fn tensordot_coond_dense_axis_f64_i64(
    a: &CooNd<f64, i64>,
    axis_a: usize,
    b: &[f64],
    b_shape: &[usize],
) -> CooNd<f64, i64> {
    assert!(axis_a < a.shape.len(), "axis out of bounds");
    assert!(!b_shape.is_empty(), "b_shape must be non-empty");
    assert_eq!(
        b.len(),
        product(b_shape),
        "b length must equal product(b_shape)"
    );
    assert_eq!(
        b_shape[0], a.shape[axis_a],
        "contracted dim must equal a.shape[axis]"
    );
    let k = product(&b_shape[1..]);
    spmm_coond_f64_i64(a, axis_a, b, k)
}
