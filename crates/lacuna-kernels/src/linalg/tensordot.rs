//! Tensor dot product (contraction) between sparse and dense arrays.
//!
//! This module wraps SpMM kernels to provide tensor contraction operations:
//! - **CSR/CSC/COO sparse tensors** contracted with dense tensors along specified axes
//! - Validates dimension compatibility and delegates to appropriate SpMM kernel
//! - Supports N-dimensional sparse arrays with arbitrary axis selection

use crate::linalg::matmul::{spmm_coo_f64_i64, spmm_coond_f64_i64, spmm_csc_f64_i64, spmm_f64_i64};
use lacuna_core::{Coo, CooNd, Csc, Csr};

/// Computes the product of all elements in a slice with overflow checking.
/// Panics if overflow occurs (product exceeds usize::MAX).
#[inline]
fn product(xs: &[usize]) -> usize {
    xs.iter().copied().fold(1usize, |acc, x| {
        acc.checked_mul(x).expect("shape product overflow")
    })
}

/// Tensor dot product: CSR matrix × dense tensor along axes (1, 0).
///
/// Computes the contraction of a 2D CSR sparse matrix A with a dense tensor B,
/// contracting A's axis 1 (columns) with B's axis 0 (first dimension).
/// This is equivalent to matrix-matrix multiplication: Y = A @ B'.
///
/// # Mathematical Operation
/// - A: (nrows × ncols) sparse matrix in CSR format
/// - B: (ndim_0 × d_1 × d_2 × ... × d_n) dense tensor (flattened row-major)
/// - Contraction: A.axis[1] (ncols) with B.axis[0] (ndim_0)
/// - Result: (nrows × d_1 × d_2 × ... × d_n) dense tensor (flattened)
///
/// # Algorithm
/// 1. Reshape B's dimensions: (ndim_0, k) where k = d_1 * d_2 * ... * d_n
/// 2. Delegate to SpMM: spmm_f64_i64(A, B, k)
/// 3. Output is nrows × k (row-major dense)
///
/// # Arguments
/// * `a` - CSR matrix (nrows × ncols)
/// * `b` - Dense tensor flattened to 1D (ndim_0 × k elements)
/// * `b_shape` - Shape of B: [ndim_0, d_1, d_2, ..., d_n]
///
/// # Returns
/// Dense result (nrows × k) in row-major order
///
/// # Panics
/// - If b_shape is empty
/// - If b.len() != product(b_shape)
/// - If b_shape[0] != a.ncols (dimension mismatch)
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

/// Tensor dot product: CSC matrix × dense tensor along axes (1, 0).
///
/// Computes the contraction of a 2D CSC sparse matrix A with a dense tensor B,
/// contracting A's axis 1 (columns) with B's axis 0 (first dimension).
/// This is equivalent to matrix-matrix multiplication: Y = A @ B'.
///
/// # Mathematical Operation
/// - A: (nrows × ncols) sparse matrix in CSC format
/// - B: (ndim_0 × d_1 × d_2 × ... × d_n) dense tensor (flattened row-major)
/// - Contraction: A.axis[1] (ncols) with B.axis[0] (ndim_0)
/// - Result: (nrows × d_1 × d_2 × ... × d_n) dense tensor (flattened)
///
/// # Algorithm
/// 1. Reshape B's dimensions: (ndim_0, k) where k = d_1 * d_2 * ... * d_n
/// 2. Delegate to SpMM: spmm_csc_f64_i64(A, B, k)
/// 3. Output is nrows × k (row-major dense)
///
/// # Arguments
/// * `a` - CSC matrix (nrows × ncols)
/// * `b` - Dense tensor flattened to 1D (ndim_0 × k elements)
/// * `b_shape` - Shape of B: [ndim_0, d_1, d_2, ..., d_n]
///
/// # Returns
/// Dense result (nrows × k) in row-major order
///
/// # Panics
/// - If b_shape is empty
/// - If b.len() != product(b_shape)
/// - If b_shape[0] != a.ncols (dimension mismatch)
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

/// Tensor dot product: COO matrix × dense tensor along axes (1, 0).
///
/// Computes the contraction of a 2D COO sparse matrix A with a dense tensor B,
/// contracting A's axis 1 (columns) with B's axis 0 (first dimension).
/// This is equivalent to matrix-matrix multiplication: Y = A @ B'.
///
/// # Mathematical Operation
/// - A: (nrows × ncols) sparse matrix in COO format
/// - B: (ndim_0 × d_1 × d_2 × ... × d_n) dense tensor (flattened row-major)
/// - Contraction: A.axis[1] (ncols) with B.axis[0] (ndim_0)
/// - Result: (nrows × d_1 × d_2 × ... × d_n) dense tensor (flattened)
///
/// # Algorithm
/// 1. Reshape B's dimensions: (ndim_0, k) where k = d_1 * d_2 * ... * d_n
/// 2. Delegate to SpMM: spmm_coo_f64_i64(A, B, k)
/// 3. Output is nrows × k (row-major dense)
///
/// # Arguments
/// * `a` - COO matrix (nrows × ncols)
/// * `b` - Dense tensor flattened to 1D (ndim_0 × k elements)
/// * `b_shape` - Shape of B: [ndim_0, d_1, d_2, ..., d_n]
///
/// # Returns
/// Dense result (nrows × k) in row-major order
///
/// # Panics
/// - If b_shape is empty
/// - If b.len() != product(b_shape)
/// - If b_shape[0] != a.ncols (dimension mismatch)
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

/// Tensor dot product: N-D COO sparse array × dense tensor along specified axes.
///
/// Computes the contraction of an N-dimensional sparse array A with a dense tensor B,
/// contracting A's specified axis with B's axis 0 (first dimension).
/// This is a generalization of tensordot for sparse N-D arrays.
///
/// # Mathematical Operation
/// - A: (d_0 × d_1 × ... × d_n) sparse array in CooNd format
/// - B: (d_axis × b_1 × b_2 × ... × b_m) dense tensor (flattened row-major)
/// - Contraction: A.axis[axis_a] (d_axis) with B.axis[0]
/// - Result: (d_0 × ... × d_{axis-1} × d_{axis+1} × ... × d_n × b_1 × b_2 × ... × b_m) sparse array
///
/// # Algorithm
/// 1. Validate axis_a is within bounds
/// 2. Reshape B's remaining dimensions: k = b_1 * b_2 * ... * b_m
/// 3. Delegate to SpMM: spmm_coond_f64_i64(A, axis_a, B, k)
/// 4. Output is CooNd with (N-1+1=N) dimensions (axis replaced by k dimensions)
///
/// # Arguments
/// * `a` - CooNd sparse array
/// * `axis_a` - Axis of A to contract (0 <= axis_a < ndim)
/// * `b` - Dense tensor flattened to 1D (a.shape[axis_a] × k elements)
/// * `b_shape` - Shape of B: [a.shape[axis_a], b_1, b_2, ..., b_m]
///
/// # Returns
/// CooNd sparse array with dimensions: (non-contracted A axes, k)
///
/// # Panics
/// - If axis_a >= a.ndim
/// - If b_shape is empty
/// - If b.len() != product(b_shape)
/// - If b_shape[0] != a.shape[axis_a] (dimension mismatch)
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
