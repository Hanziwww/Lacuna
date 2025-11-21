//! Mean reductions
//
// This module implements mean reduction operations for N-dimensional sparse arrays (COOND).
// Functions include global mean and mean reduction along specified axes. All reductions
// handle empty shapes and avoid division by zero. The implementation leverages sum reductions
// and multiplies by the reciprocal of the number of elements for efficiency.

use lacuna_core::CooNd;

/// Compute the product of dimensions, checking for overflow.
///
/// Used to determine the total number of elements in an N-dimensional array.
/// Panics if the product overflows usize.
#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

/// Compute the mean of all elements in an N-dimensional COO array.
///
/// Returns 0.0 for empty shapes or zero total size to avoid division by zero.
/// The mean is computed as the sum of all elements divided by the total number of elements.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_coond_f64(a: &CooNd<f64, i64>) -> f64 {
    if a.shape.is_empty() {
        return 0.0; // convention: empty shape means zero length; avoid div by zero
    }
    let denom = product_checked(&a.shape) as f64;
    if denom == 0.0 {
        return 0.0;
    }
    // Use sum reduction and divide by total number of elements
    crate::statistical::sum::sum_coond_f64(a) / denom
}

/// Compute the mean along specified axes for an N-dimensional COO array.
///
/// This function first reduces the array by summing along the given axes, then divides
/// the result by the product of the sizes of the reduced axes. Handles overflow and avoids
/// division by zero. Returns the reduced COO array with means along the specified axes.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn reduce_mean_axes_coond_f64_i64(a: &CooNd<f64, i64>, axes: &[usize]) -> CooNd<f64, i64> {
    // First, sum along the specified axes
    let reduced = crate::statistical::sum::reduce_sum_axes_coond_f64_i64(a, axes);
    // Track which axes are being reduced
    let mut reduce = vec![false; a.shape.len()];
    for &ax in axes {
        reduce[ax] = true;
    }
    // Compute denominator: product of sizes of reduced axes
    let mut denom_us: usize = 1;
    for (d, &sz) in a.shape.iter().enumerate() {
        if reduce[d] {
            denom_us = denom_us.checked_mul(sz).expect("shape product overflow");
        }
    }
    // If denominator is 1, no reduction occurred; return sum result
    if denom_us == 1 {
        return reduced;
    }
    // Otherwise, scale the sum by the reciprocal of the denominator to get the mean
    let factor = 1.0f64 / (denom_us as f64);
    crate::elementwise::multiply::mul_scalar_coond_f64(&reduced, factor)
}
