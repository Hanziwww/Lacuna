//! Reshape for COOND

use lacuna_core::CooNd;
use rayon::prelude::*;

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0, "value must be non-negative");
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok(), "value must fit in i64");
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

#[inline]
fn build_strides_row_major(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let n = dims.len();
    let mut strides = vec![0usize; n];
    strides[n - 1] = 1;
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1])
            .expect("shape product overflow");
    }
    strides
}

#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn reshape_coond_f64_i64(a: &CooNd<f64, i64>, new_shape: &[usize]) -> CooNd<f64, i64> {
    let old_elems = product_checked(&a.shape);
    let new_elems = product_checked(new_shape);
    assert_eq!(
        old_elems, new_elems,
        "reshape requires same number of elements"
    );
    let ndim_old = a.shape.len();
    let ndim_new = new_shape.len();
    let nnz = a.data.len();
    if nnz == 0 {
        return CooNd::from_parts_unchecked(new_shape.to_vec(), Vec::new(), Vec::new());
    }
    let old_strides = build_strides_row_major(&a.shape);
    let new_strides = build_strides_row_major(new_shape);
    let mut out_indices = vec![0i64; nnz * ndim_new];
    let out_ptr = out_indices.as_mut_ptr() as usize;
    (0..nnz).into_par_iter().for_each(|k| {
        let base_old = k * ndim_old;
        let mut lin: usize = 0;
        for d in 0..ndim_old {
            let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base_old + d) });
            let s = old_strides[d];
            lin = lin
                .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        // de-linearize into new shape
        let base_new = k * ndim_new;
        let mut rem = lin;
        for d in 0..ndim_new {
            let s = new_strides[d];
            let idx = if s == 0 { 0 } else { rem / s };
            rem -= idx * s;
            unsafe {
                let p = out_ptr as *mut i64;
                std::ptr::write(p.add(base_new + d), usize_to_i64(idx));
            }
        }
    });
    CooNd::from_parts_unchecked(new_shape.to_vec(), out_indices, a.data.clone())
}
