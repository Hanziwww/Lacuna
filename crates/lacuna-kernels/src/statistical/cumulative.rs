//! Cumulative operations (cumsum, cumprod) producing dense outputs for CSR
//
// This module implements cumulative sum and cumulative product operations for sparse matrices
// in CSR format. The results are always returned as dense row-major arrays. Both row-wise (axis=1)
// and column-wise (axis=0) operations are supported. For column-wise, the matrix is transposed,
// the row-wise operation is performed, and the result is transposed back. Parallelism is used for performance.

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
/// Transpose a dense row-major matrix (represented as a flat array).
/// Input shape: (rows, cols), output shape: (cols, rows), both row-major.
/// Used to convert between axis-0 and axis-1 cumulative operations.
fn transpose_dense_row_major(input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    // For each element, place it in the transposed position
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

/// Compute the row-wise cumulative sum (axis=1) for a CSR matrix.
/// Returns a dense row-major array of shape (nrows, ncols).
/// For each row, the cumulative sum is computed left-to-right, filling in zeros for missing columns.
#[must_use]
pub fn csr_cumsum_dense_axis1_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; nrows * ncols];
    // Parallelize over rows for performance
    out.par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(i, row_out)| {
            // For each row, walk through nonzero entries
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut prev = 0usize;
            let mut acc = 0.0f64;
            let mut p = s;
            while p < e {
                let col = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                // Fill zeros between previous and current column with accumulated sum
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
            // Fill remaining columns after last nonzero
            if prev < ncols {
                row_out[prev..].fill(acc);
            }
        });
    out
}

/// Compute the column-wise cumulative sum (axis=0) for a CSR matrix.
/// Returns a dense row-major array of shape (nrows, ncols).
/// This is implemented by transposing the matrix, applying row-wise cumsum, and transposing back.
#[must_use]
pub fn csr_cumsum_dense_axis0_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    // Transpose, compute row-wise cumsum, then transpose back to get column-wise result
    let t = transpose_f64_i64(a);
    let out_t = csr_cumsum_dense_axis1_f64(&t); // shape (ncols, nrows)
    transpose_dense_row_major(&out_t, t.nrows, t.ncols) // becomes (nrows, ncols)
}

/// Compute the row-wise cumulative product (axis=1) for a CSR matrix.
/// Returns a dense row-major array of shape (nrows, ncols).
/// If an implicit zero is encountered, the remainder of the row stays zero.
#[must_use]
pub fn csr_cumprod_dense_axis1_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; nrows * ncols];
    // Parallelize over rows for performance
    out.par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(i, row_out)| {
            // For each row, walk through nonzero entries
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut prev = 0usize;
            let mut acc = 1.0f64;
            let mut p = s;
            while p < e {
                let col = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                // If there is a gap (implicit zero), remainder of row stays zero
                if col > prev {
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
            // After last nonzero, remainder stays zero (already initialized)
        });
    out
}

/// Compute the column-wise cumulative product (axis=0) for a CSR matrix.
/// Returns a dense row-major array of shape (nrows, ncols).
/// This is implemented by transposing the matrix, applying row-wise cumprod, and transposing back.
#[must_use]
pub fn csr_cumprod_dense_axis0_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    // Transpose, compute row-wise cumprod, then transpose back to get column-wise result
    let t = transpose_f64_i64(a);
    let out_t = csr_cumprod_dense_axis1_f64(&t); // shape (ncols, nrows)
    transpose_dense_row_major(&out_t, t.nrows, t.ncols) // becomes (nrows, ncols)
}
