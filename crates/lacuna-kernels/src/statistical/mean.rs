//! Mean reductions

use lacuna_core::CooNd;

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_coond_f64(a: &CooNd<f64, i64>) -> f64 {
    if a.shape.is_empty() {
        return 0.0; // conventionally empty shape -> 0 length; avoid div by zero
    }
    let denom = product_checked(&a.shape) as f64;
    if denom == 0.0 {
        return 0.0;
    }
    crate::statistical::sum::sum_coond_f64(a) / denom
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn reduce_mean_axes_coond_f64_i64(a: &CooNd<f64, i64>, axes: &[usize]) -> CooNd<f64, i64> {
    let reduced = crate::statistical::sum::reduce_sum_axes_coond_f64_i64(a, axes);
    let mut reduce = vec![false; a.shape.len()];
    for &ax in axes {
        reduce[ax] = true;
    }
    let mut denom_us: usize = 1;
    for (d, &sz) in a.shape.iter().enumerate() {
        if reduce[d] {
            denom_us = denom_us.checked_mul(sz).expect("shape product overflow");
        }
    }
    if denom_us == 1 {
        return reduced;
    }
    let factor = 1.0f64 / (denom_us as f64);
    crate::elementwise::multiply::mul_scalar_coond_f64(&reduced, factor)
}
