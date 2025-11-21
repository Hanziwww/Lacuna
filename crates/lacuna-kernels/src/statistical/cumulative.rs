//! Cumulative operations (cumsum, cumprod) producing dense outputs for CSR

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices; dense write is O(nrows*ncols)"
)]

use crate::linalg::matrix_transpose::transpose_f64_i64;
use crate::utility::util::i64_to_usize;
use lacuna_core::Csr;
use rayon::prelude::*;

#[inline]
fn transpose_dense_row_major(input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // input is row-major with shape (rows, cols)
    // output is row-major with shape (cols, rows)
    let mut out = vec![0.0f64; rows * cols];
    // SAFETY: bounds are checked by loops
    for r in 0..rows {
        let base_in = r * cols;
        for c in 0..cols {
            let v = unsafe { *input.get_unchecked(base_in + c) };
            let base_out = c * rows;
            unsafe {
                *out.get_unchecked_mut(base_out + r) = v;
            }
        }
    }
    out
}

/// CSR row-wise cumulative sum (axis=1). Returns dense row-major (nrows * ncols)
#[must_use]
pub fn csr_cumsum_dense_axis1_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; nrows * ncols];
    out.par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(i, row_out)| {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut prev = 0usize;
            let mut acc = 0.0f64;
            let mut p = s;
            while p < e {
                let col = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                if col > prev {
                    row_out[prev..col].fill(acc);
                }
                acc += unsafe { *a.data.get_unchecked(p) };
                unsafe {
                    *row_out.get_unchecked_mut(col) = acc;
                }
                prev = col + 1;
                p += 1;
            }
            if prev < ncols {
                row_out[prev..].fill(acc);
            }
        });
    out
}

/// CSR column-wise cumulative sum (axis=0). Returns dense row-major (nrows * ncols)
#[must_use]
pub fn csr_cumsum_dense_axis0_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    // Compute on transpose row-wise, then transpose dense back
    let t = transpose_f64_i64(a);
    let out_t = csr_cumsum_dense_axis1_f64(&t); // shape (ncols, nrows)
    transpose_dense_row_major(&out_t, t.nrows, t.ncols) // becomes (nrows, ncols)
}

/// CSR row-wise cumulative product (axis=1). Returns dense row-major (nrows * ncols)
#[must_use]
pub fn csr_cumprod_dense_axis1_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; nrows * ncols];
    out.par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(i, row_out)| {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut prev = 0usize;
            let mut acc = 1.0f64;
            let mut p = s;
            while p < e {
                let col = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                if col > prev {
                    // Encountered first implicit zero -> remainder stays zero
                    // row_out already zero-initialized; nothing more to do
                    return;
                }
                acc *= unsafe { *a.data.get_unchecked(p) };
                unsafe {
                    *row_out.get_unchecked_mut(col) = acc;
                }
                prev = col + 1;
                p += 1;
            }
            // After the last explicitly stored element, the next position is implicit zero,
            // thus the remainder stays zero (already initialized)
        });
    out
}

/// CSR column-wise cumulative product (axis=0). Returns dense row-major (nrows * ncols)
#[must_use]
pub fn csr_cumprod_dense_axis0_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    // Compute on transpose row-wise, then transpose dense back
    let t = transpose_f64_i64(a);
    let out_t = csr_cumprod_dense_axis1_f64(&t); // shape (ncols, nrows)
    transpose_dense_row_major(&out_t, t.nrows, t.ncols) // becomes (nrows, ncols)
}
