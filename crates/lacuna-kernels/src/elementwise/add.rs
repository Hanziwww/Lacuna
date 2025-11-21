//! Elementwise addition for all sparse formats (Array API: `add`).
//!
//! Implements sparse + sparse addition for:
//! - CSR (Compressed Sparse Row): merges rows element-wise
//! - CSC (Compressed Sparse Column): merges columns element-wise
//! - CooND (N-dimensional COO): merges via linearized coordinate keys
//!
//! All operations coalesce duplicate coordinates and filter out exact zeros.

#![allow(
    clippy::similar_names,
    reason = "Pointer/address aliases (pi/pv, ai/bi) are intentional in low-level kernels"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]

use crate::utility::util::UsizeF64Map;
use lacuna_core::{CooNd, Csc, Csr};
use rayon::prelude::*;

/// Converts i64 to usize with debug assertions for non-negative values.
#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// Converts usize to i64 with debug assertions for range validity.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Counts nonzero entries resulting from merging two sorted sparse rows.
///
/// Performs a two-pointer merge of column indices from rows of matrices A and B,
/// accumulating values at matching column positions and filtering exact zeros.
/// Both input rows are assumed to have strictly increasing column indices with no duplicates.
///
/// # Arguments
/// * `ai` - Pointer to column indices of matrix A's row segment
/// * `av` - Pointer to values of matrix A's row segment
/// * `alen` - Length of A's row segment
/// * `bi` - Pointer to column indices of matrix B's row segment
/// * `bv` - Pointer to values of matrix B's row segment
/// * `blen` - Length of B's row segment
///
/// # Returns
/// Count of nonzero entries in the merged result (zeros are filtered out)
///
/// # Safety
/// Caller must ensure:
/// - All pointers are valid for their respective lengths
/// - Column indices are strictly increasing within each row segment
#[inline]
unsafe fn add_row_count(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
) -> usize {
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut cnt = 0usize;
    while pa < alen || pb < blen {
        let ja = if pa < alen {
            unsafe { *ai.add(pa) }
        } else {
            i64::MAX
        };
        let jb = if pb < blen {
            unsafe { *bi.add(pb) }
        } else {
            i64::MAX
        };
        let j = if ja <= jb { ja } else { jb };
        let mut v = 0.0f64;
        while pa < alen && unsafe { *ai.add(pa) } == j {
            v += unsafe { *av.add(pa) };
            pa += 1;
        }
        while pb < blen && unsafe { *bi.add(pb) } == j {
            v += unsafe { *bv.add(pb) };
            pb += 1;
        }
        if v != 0.0 {
            cnt += 1;
        }
    }
    cnt
}

/// Merges two sorted sparse rows and writes output, filtering zeros.
///
/// Performs the same merge operation as `add_row_count` but also writes the
/// resulting column indices and accumulated values to output arrays.
/// Both input rows are assumed to have strictly increasing column indices with no duplicates.
///
/// # Arguments
/// * `ai`, `av`, `alen` - Column indices and values of matrix A's row segment
/// * `bi`, `bv`, `blen` - Column indices and values of matrix B's row segment
/// * `out_i` - Output buffer for merged column indices
/// * `out_v` - Output buffer for merged values
///
/// # Returns
/// Number of elements written to output (zeros filtered out)
///
/// # Safety
/// Caller must ensure:
/// - All input pointers are valid for their respective lengths
/// - Output pointers are valid for at least alen+blen elements
/// - Column indices are strictly increasing within each input row
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn add_row_fill(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
    out_i: *mut i64,
    out_v: *mut f64,
) -> usize {
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut dst = 0usize;
    while pa < alen || pb < blen {
        let ja = if pa < alen {
            unsafe { *ai.add(pa) }
        } else {
            i64::MAX
        };
        let jb = if pb < blen {
            unsafe { *bi.add(pb) }
        } else {
            i64::MAX
        };
        let j = if ja <= jb { ja } else { jb };
        let mut v = 0.0f64;
        while pa < alen && unsafe { *ai.add(pa) } == j {
            v += unsafe { *av.add(pa) };
            pa += 1;
        }
        while pb < blen && unsafe { *bi.add(pb) } == j {
            v += unsafe { *bv.add(pb) };
            pb += 1;
        }
        if v != 0.0 {
            unsafe {
                std::ptr::write(out_i.add(dst), j);
                std::ptr::write(out_v.add(dst), v);
            }
            dst += 1;
        }
    }
    dst
}

/// Adds two CSR matrices: A + B → Result (in CSR format).
///
/// Performs element-wise addition by merging rows in parallel. Duplicate
/// coordinates (same row and column) are coalesced by summing their values.
/// Exact zeros in the result are automatically filtered out.
///
/// # Algorithm
/// **Pass 1: Count Phase** (parallel over rows)
/// - For each row i, count nonzeros in the merged result of A[i] and B[i]
/// - Use two-pointer merge algorithm on sorted column indices
///
/// **Pass 2: Fill Phase** (parallel over rows)
/// - Compute row pointers (indptr) via prefix sum of counts
/// - Allocate output indices and data arrays
/// - For each row, merge and write results to output
///
/// # Complexity
/// - Time: O((nnz_A + nnz_B) + nrows * log(nrows)) in practice
/// - Space: O(nnz_output)
///
/// # Panics
/// - If input matrices have different shapes
#[must_use]
pub fn add_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;

    // Pass 1: count output nnz per row in parallel
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
                add_row_count(
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

    // Prefix sum -> indptr
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows {
        indptr[i + 1] = indptr[i] + usize_to_i64(counts[i]);
    }
    let nnz = i64_to_usize(indptr[nrows]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    // Pass 2: fill rows in parallel
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
            let written = add_row_fill(
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

/// Adds two CSC matrices: A + B → Result (in CSC format).
///
/// Performs element-wise addition by merging columns in parallel. Duplicate
/// coordinates (same row and column) are coalesced by summing their values.
/// Exact zeros in the result are automatically filtered out.
///
/// # Algorithm
/// **Pass 1: Count Phase** (parallel over columns)
/// - For each column j, count nonzeros in the merged result of A[j] and B[j]
/// - Use two-pointer merge algorithm on sorted row indices
///
/// **Pass 2: Fill Phase** (parallel over columns)
/// - Compute column pointers (indptr) via prefix sum of counts
/// - Allocate output indices and data arrays
/// - For each column, merge and write results to output
///
/// # Complexity
/// - Time: O((nnz_A + nnz_B) + ncols * log(ncols)) in practice
/// - Space: O(nnz_output)
///
/// # Panics
/// - If input matrices have different shapes
#[must_use]
pub fn add_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let ncols = a.ncols;

    // Pass 1: count per column
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
                add_row_count(
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

    let mut indptr = vec![0i64; ncols + 1];
    for j in 0..ncols {
        indptr[j + 1] = indptr[j] + usize_to_i64(counts[j]);
    }
    let nnz = i64_to_usize(indptr[ncols]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

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
            let written = add_row_fill(
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

/// Adds two N-dimensional COO arrays: A + B → Result (in COO format).
///
/// Performs element-wise addition on N-D sparse arrays. Duplicate coordinates
/// are coalesced by summing their values. Exact zeros are filtered out.
///
/// # Algorithm
/// 1. **Linearization**: Compute row-major strides from shape
/// 2. **Accumulation**: Convert all coordinates to linearized indices (keys)
///    - Insert A's entries into hash map with accumulation
///    - Insert B's entries into hash map with accumulation
/// 3. **Filtering & Sorting**: Filter out exact zeros, sort by linearized key
/// 4. **Reconstruction**: Convert linearized indices back to N-D coordinates
///    - For each (linearized_key, value) pair, reconstruct N-D coordinates
///    - Build output indices array and data array
///
/// # Complexity
/// - Time: O((nnz_A + nnz_B) * ndim) for linearization/reconstruction
/// - Space: O(nnz_output + ndim)
///
/// # Panics
/// - If input arrays have different shapes or ndim
/// - If shape product or linear index computation overflows
#[must_use]
pub fn add_coond_f64_i64(a: &CooNd<f64, i64>, b: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    assert_eq!(a.shape.len(), b.shape.len());
    assert_eq!(a.shape, b.shape);
    let ndim = a.shape.len();
    let nnz_a = a.data.len();
    let nnz_b = b.data.len();

    // Build row-major strides for linearization
    let mut strides = vec![0usize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(a.shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }

    // Accumulate entries from A and B via linearized keys
    let mut acc = UsizeF64Map::with_capacity(nnz_a + nnz_b);
    for k in 0..nnz_a {
        let mut lin = 0usize;
        let base = k * ndim;
        for (axis, stride) in strides.iter().enumerate() {
            let idx = i64_to_usize(a.indices[base + axis]);
            lin = lin
                .checked_add(idx.checked_mul(*stride).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc.insert_add(lin, a.data[k]);
    }
    for k in 0..nnz_b {
        let mut lin = 0usize;
        let base = k * ndim;
        for (axis, stride) in strides.iter().enumerate() {
            let idx = i64_to_usize(b.indices[base + axis]);
            lin = lin
                .checked_add(idx.checked_mul(*stride).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc.insert_add(lin, b.data[k]);
    }

    // Sort by linearized key and reconstruct indices
    let mut pairs = acc.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);
    let mut out_pairs: Vec<(usize, f64)> = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        if v != 0.0 {
            out_pairs.push((k, v));
        }
    }
    let out_nnz = out_pairs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * ndim];
    for (pos, (mut lin, v)) in out_pairs.into_iter().enumerate() {
        let base = pos * ndim;
        for (axis, stride) in strides.iter().enumerate() {
            let idx = lin / *stride;
            lin -= idx * *stride;
            out_indices[base + axis] = usize_to_i64(idx);
        }
        out_data.push(v);
    }
    CooNd::from_parts_unchecked(a.shape.clone(), out_indices, out_data)
}
