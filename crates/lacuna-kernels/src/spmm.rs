#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
use lacuna_core::Csr;
use rayon::prelude::*;
use wide::f64x4;

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// Y = A @ B, where B is (ncols, k) row-major; returns Y as (nrows, k) row-major
#[must_use]
pub fn spmm_f64_i64(a: &Csr<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut y = vec![0.0f64; nrows * k];

    // Process per row in parallel; within row, use SIMD across k.
    y.par_chunks_mut(k).enumerate().for_each(|(i, yi)| {
        let start = i64_to_usize(a.indptr[i]);
        let end = i64_to_usize(a.indptr[i + 1]);
        let _ = ncols;

        let limit4 = k.saturating_sub(k % 4);

        let mut c = 0usize;
        while c < limit4 {
            let mut acc = unsafe {
                let p = yi.as_ptr().add(c).cast::<[f64; 4]>();
                f64x4::new(core::ptr::read_unaligned(p))
            };

            for p in start..end {
                let j = i64_to_usize(a.indices[p]);
                let aijv = f64x4::splat(a.data[p]);
                let base = j * k + c;
                let vb = unsafe {
                    let p = b.as_ptr().add(base).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                acc += vb * aijv;
            }

            unsafe {
                let p = yi.as_mut_ptr().add(c).cast::<[f64; 4]>();
                core::ptr::write_unaligned(p, acc.to_array());
            }
            c += 4;
        }

        // Remainder columns
        while c < k {
            let mut acc = yi[c];
            for p in start..end {
                let j = i64_to_usize(a.indices[p]);
                let aij = a.data[p];
                let base = j * k + c;
                acc += aij * b[base];
            }
            yi[c] = acc;
            c += 1;
        }
    });
    y
}
