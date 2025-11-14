#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k for indices"
)]
use lacuna_core::Csr;
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use wide::f64x4;

const SMALL_NNZ_REDUCE: usize = 16 * 1024;

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// sum of all data
#[must_use]
pub fn sum_f64(a: &Csr<f64, i64>) -> f64 {
    // Parallel reduce with SIMD inside chunks
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = f64x4::from([chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]]);
                accv += v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

/// row sums
#[must_use]
pub fn row_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let mut out = vec![0.0f64; a.nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let row = &a.data[s..e];
        let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
        let mut k = 0usize;
        let limit4 = row.len() & !3;
        while k < limit4 {
            let v = f64x4::from([row[k], row[k + 1], row[k + 2], row[k + 3]]);
            accv += v;
            k += 4;
        }
        let arr = accv.to_array();
        let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
        while k < row.len() {
            acc += row[k];
            k += 1;
        }
        *oi = acc;
    });
    out
}

/// column sums (parallelized and vectorized)
#[must_use]
pub fn col_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }
    let nnz = a.data.len();
    if nnz < SMALL_NNZ_REDUCE {
        let mut out = vec![0.0f64; ncols];
        for i in 0..a.nrows {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                out[j] += a.data[p];
            }
        }
        return out;
    }

    let tls: ThreadLocal<RefCell<Vec<f64>>> = ThreadLocal::new();
    (0..a.nrows).into_par_iter().for_each(|i| {
        let cell = tls.get_or(|| RefCell::new(vec![0.0f64; ncols]));
        let mut acc = cell.borrow_mut();
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        for p in s..e {
            let j = i64_to_usize(a.indices[p]);
            acc[j] += a.data[p];
        }
    });
    let mut out = vec![0.0f64; ncols];
    for cell in tls {
        let acc = cell.into_inner();
        let mut c = 0usize;
        let limit4 = ncols & !3;
        while c < limit4 {
            let v1 = f64x4::from([out[c], out[c + 1], out[c + 2], out[c + 3]]);
            let v2 = f64x4::from([acc[c], acc[c + 1], acc[c + 2], acc[c + 3]]);
            let r = v1 + v2;
            let arr = r.to_array();
            out[c] = arr[0];
            out[c + 1] = arr[1];
            out[c + 2] = arr[2];
            out[c + 3] = arr[3];
            c += 4;
        }
        while c < ncols {
            out[c] += acc[c];
            c += 1;
        }
    }
    out
}
