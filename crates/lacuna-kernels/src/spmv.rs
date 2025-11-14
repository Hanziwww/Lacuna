#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p to denote indices and pointers"
)]
use lacuna_core::Csr;
use rayon::prelude::*;

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

#[inline]
fn spmv_row_f64_i64(a: &Csr<f64, i64>, x: &[f64], i: usize) -> f64 {
    let start = i64_to_usize(a.indptr[i]);
    let end = i64_to_usize(a.indptr[i + 1]);
    let len = end - start;

    if len == 0 {
        return 0.0;
    }

    let mut acc = 0.0f64;

    unsafe {
        let idx_ptr = a.indices.as_ptr().add(start);
        let val_ptr = a.data.as_ptr().add(start);

        let mut t = 0usize;
        let limit4 = len & !3;

        while t < limit4 {
            let j0 = i64_to_usize(*idx_ptr.add(t));
            let j1 = i64_to_usize(*idx_ptr.add(t + 1));
            let j2 = i64_to_usize(*idx_ptr.add(t + 2));
            let j3 = i64_to_usize(*idx_ptr.add(t + 3));

            acc = (*val_ptr.add(t + 3)).mul_add(
                *x.get_unchecked(j3),
                (*val_ptr.add(t + 2)).mul_add(
                    *x.get_unchecked(j2),
                    (*val_ptr.add(t + 1)).mul_add(
                        *x.get_unchecked(j1),
                        (*val_ptr.add(t)).mul_add(*x.get_unchecked(j0), acc),
                    ),
                ),
            );

            t += 4;
        }

        while t < len {
            let j = i64_to_usize(*idx_ptr.add(t));
            acc = (*val_ptr.add(t)).mul_add(*x.get_unchecked(j), acc);
            t += 1;
        }
    }

    acc
}

/// y = A @ x
#[must_use]
pub fn spmv_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    let nrows = a.nrows;
    let nnz = a.data.len();
    let mut y = vec![0.0f64; nrows];

    // For small problems, avoid rayon overhead and compute sequentially.
    let small = nrows < 2048 || nnz < 32768;
    if small {
        for (i, yi) in y.iter_mut().enumerate().take(nrows) {
            *yi = spmv_row_f64_i64(a, x, i);
        }
        return y;
    }

    y.par_iter_mut()
        .enumerate()
        .for_each(|(i, yi)| *yi = spmv_row_f64_i64(a, x, i));

    y
}
