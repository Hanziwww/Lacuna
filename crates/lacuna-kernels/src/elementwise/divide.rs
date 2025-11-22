//! Element-wise division kernels for sparse matrices (CSR and CSC formats).
//
// This module implements division operations for sparse matrices in CSR and CSC formats.
// Division results in zero when the denominator is zero or the result is non-finite.
// The implementation uses two-pass algorithms: count pass to determine output nnz,
// then fill pass to compute the actual quotient values.

#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::comparison_chain
)]

use core::cmp::Ordering;
use lacuna_core::{Csc, Csr};
use rayon::prelude::*;

/// Convert i64 to usize, asserting non-negativity.
#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// Convert usize to i64, asserting the value fits within i64 range.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Count non-zero elements in the quotient of two rows (a / b).
/// Merges indices from both rows using two-pointer technique.
/// Quotient element is zero if denominator is zero or result is non-finite.
///
/// # Safety
/// Requires:
/// - `ai`, `av` point to valid arrays of length `alen`
/// - `bi`, `bv` point to valid arrays of length `blen`
/// - Arrays must be sorted by column index
#[inline]
unsafe fn div_row_count(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
) -> usize {
    unsafe {
        let mut pa = 0usize; // Pointer into array a
        let mut pb = 0usize; // Pointer into array b
        let mut cnt = 0usize; // Count of non-zero quotients

        // Merge-like traversal of both sparse rows
        while pa < alen && pb < blen {
            let ja = *ai.add(pa); // Current column index in a
            let jb = *bi.add(pb); // Current column index in b

            match ja.cmp(&jb) {
                Ordering::Equal => {
                    // Column indices match: aggregate values and compute quotient
                    let j = ja;
                    let mut sa = 0.0f64; // Accumulated value from a
                    while pa < alen && *ai.add(pa) == j {
                        sa += *av.add(pa);
                        pa += 1;
                    }
                    let mut sb = 0.0f64; // Accumulated value from b
                    while pb < blen && *bi.add(pb) == j {
                        sb += *bv.add(pb);
                        pb += 1;
                    }
                    // Only count if denominator is non-zero and quotient is finite
                    if sb != 0.0 {
                        let v = sa / sb;
                        if v != 0.0 && v.is_finite() {
                            cnt += 1;
                        }
                    }
                }
                Ordering::Less => {
                    // Column index in a is smaller: skip all entries for this column in a
                    let j = ja;
                    while pa < alen && *ai.add(pa) == j {
                        pa += 1;
                    }
                }
                Ordering::Greater => {
                    // Column index in b is smaller: skip all entries for this column in b
                    let j = jb;
                    while pb < blen && *bi.add(pb) == j {
                        pb += 1;
                    }
                }
            }
        }
        cnt
    }
}

/// Fill quotient of two rows into output arrays.
/// Computes a / b and stores non-zero results in the output arrays.
/// Returns the number of elements written.
///
/// # Safety
/// Requires:
/// - `ai`, `av` point to valid arrays of length `alen`
/// - `bi`, `bv` point to valid arrays of length `blen`
/// - `out_i`, `out_v` point to valid writable arrays
/// - Arrays must be sorted by column index
/// - Output arrays must have sufficient capacity
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn div_row_fill(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
    out_i: *mut i64,
    out_v: *mut f64,
) -> usize {
    unsafe {
        let mut pa = 0usize; // Pointer into array a
        let mut pb = 0usize; // Pointer into array b
        let mut dst = 0usize; // Write position in output

        // Merge-like traversal of both sparse rows
        while pa < alen && pb < blen {
            let ja = *ai.add(pa); // Current column index in a
            let jb = *bi.add(pb); // Current column index in b

            match ja.cmp(&jb) {
                Ordering::Equal => {
                    // Column indices match: aggregate values and compute quotient
                    let j = ja;
                    let mut sa = 0.0f64; // Accumulated value from a
                    while pa < alen && *ai.add(pa) == j {
                        sa += *av.add(pa);
                        pa += 1;
                    }
                    let mut sb = 0.0f64; // Accumulated value from b
                    while pb < blen && *bi.add(pb) == j {
                        sb += *bv.add(pb);
                        pb += 1;
                    }
                    // Write result if denominator is non-zero and quotient is finite and non-zero
                    if sb != 0.0 {
                        let v = sa / sb;
                        if v != 0.0 && v.is_finite() {
                            std::ptr::write(out_i.add(dst), j);
                            std::ptr::write(out_v.add(dst), v);
                            dst += 1;
                        }
                    }
                }
                Ordering::Less => {
                    // Column index in a is smaller: skip all entries for this column in a
                    let j = ja;
                    while pa < alen && *ai.add(pa) == j {
                        pa += 1;
                    }
                }
                Ordering::Greater => {
                    // Column index in b is smaller: skip all entries for this column in b
                    let j = jb;
                    while pb < blen && *bi.add(pb) == j {
                        pb += 1;
                    }
                }
            }
        }
        dst
    }
}

/// Divide two CSR matrices element-wise (a / b).
/// Returns a new CSR matrix with the quotient values.
/// Matrices must have the same dimensions.
#[must_use]
pub fn div_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;

    // Phase 1: Count non-zeros per row in parallel
    let counts: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let sa = i64_to_usize(a.indptr[i]);
            let ea = i64_to_usize(a.indptr[i + 1]);
            let sb = i64_to_usize(b.indptr[i]);
            let eb = i64_to_usize(b.indptr[i + 1]);
            let alen = ea - sa;
            let blen = eb - sb;
            unsafe {
                div_row_count(
                    a.indices.as_ptr().add(sa),
                    a.data.as_ptr().add(sa),
                    alen,
                    b.indices.as_ptr().add(sb),
                    b.data.as_ptr().add(sb),
                    blen,
                )
            }
        })
        .collect();

    // Build row pointer array from row counts
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows {
        indptr[i + 1] = indptr[i] + usize_to_i64(counts[i]);
    }
    let nnz = i64_to_usize(indptr[nrows]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];

    // Store raw pointers for thread-safe access in parallel phase
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    // Phase 2: Fill rows in parallel using raw pointer addresses
    (0..nrows).into_par_iter().for_each(move |i| {
        let sa = i64_to_usize(a.indptr[i]);
        let ea = i64_to_usize(a.indptr[i + 1]);
        let sb = i64_to_usize(b.indptr[i]);
        let eb = i64_to_usize(b.indptr[i + 1]);
        let alen = ea - sa;
        let blen = eb - sb;
        let row_start = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(i) });
        unsafe {
            let pi = (pi_addr as *mut i64).add(row_start);
            let pv = (pv_addr as *mut f64).add(row_start);
            let written = div_row_fill(
                a.indices.as_ptr().add(sa),
                a.data.as_ptr().add(sa),
                alen,
                b.indices.as_ptr().add(sb),
                b.data.as_ptr().add(sb),
                blen,
                pi,
                pv,
            );
            let expected = i64_to_usize(*(indptr_addr as *const i64).add(i + 1)) - row_start;
            debug_assert_eq!(written, expected);
        }
    });
    Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data)
}

/// Divide two CSC matrices element-wise (a / b).
/// Returns a new CSC matrix with the quotient values.
/// Matrices must have the same dimensions.
#[must_use]
pub fn div_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let ncols = a.ncols;

    // Phase 1: Count non-zeros per column in parallel
    let counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let sa = i64_to_usize(a.indptr[j]);
            let ea = i64_to_usize(a.indptr[j + 1]);
            let sb = i64_to_usize(b.indptr[j]);
            let eb = i64_to_usize(b.indptr[j + 1]);
            let alen = ea - sa;
            let blen = eb - sb;
            unsafe {
                div_row_count(
                    a.indices.as_ptr().add(sa),
                    a.data.as_ptr().add(sa),
                    alen,
                    b.indices.as_ptr().add(sb),
                    b.data.as_ptr().add(sb),
                    blen,
                )
            }
        })
        .collect();

    // Build column pointer array from column counts
    let mut indptr = vec![0i64; ncols + 1];
    for j in 0..ncols {
        indptr[j + 1] = indptr[j] + usize_to_i64(counts[j]);
    }
    let nnz = i64_to_usize(indptr[ncols]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];

    // Store raw pointers for thread-safe access in parallel phase
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    // Phase 2: Fill columns in parallel using raw pointer addresses
    (0..ncols).into_par_iter().for_each(move |j| {
        let sa = i64_to_usize(a.indptr[j]);
        let ea = i64_to_usize(a.indptr[j + 1]);
        let sb = i64_to_usize(b.indptr[j]);
        let eb = i64_to_usize(b.indptr[j + 1]);
        let alen = ea - sa;
        let blen = eb - sb;
        let col_start = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j) });
        unsafe {
            let pi = (pi_addr as *mut i64).add(col_start);
            let pv = (pv_addr as *mut f64).add(col_start);
            let written = div_row_fill(
                a.indices.as_ptr().add(sa),
                a.data.as_ptr().add(sa),
                alen,
                b.indices.as_ptr().add(sb),
                b.data.as_ptr().add(sb),
                blen,
                pi,
                pv,
            );
            let expected = i64_to_usize(*(indptr_addr as *const i64).add(j + 1)) - col_start;
            debug_assert_eq!(written, expected);
        }
    });
    Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data)
}
