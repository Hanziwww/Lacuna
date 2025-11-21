//! Sparse matrix format conversions (astype in Array API sense).
//!
//! This module provides functions to convert between different sparse matrix formats:
//! - CSR <-> CSC (between row-major and column-major compressed formats)
//! - CSR <-> COO (between compressed row format and coordinate format)
//! - CSC <-> COO (between compressed column format and coordinate format)
//! - N-dimensional COO to 2D CSR/CSC (using specified axes for row/column projection)
//!
//! All conversions preserve the nonzero values and handle proper index reordering.

#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]

use crate::utility::util::{SMALL_NNZ_LIMIT, i64_to_usize};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

/// Converts a usize to i64 with debug assertions for validity.
/// Assumes the input is small enough to fit in i64 range.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Converts CSR (Compressed Sparse Row) to CSC (Compressed Sparse Column) format.
///
/// This conversion transposition-like operation reorders the matrix from row-major
/// to column-major storage while maintaining all nonzero values.
///
/// # Algorithm
/// 1. Count nonzeros per column (from row indices)
/// 2. Compute cumulative column pointers
/// 3. Use atomic operations to place each element in its correct column position
/// 4. Row indices are extracted from the CSR row ranges
///
/// # Parallelization
/// The conversion processes each row in parallel using atomic operations to
/// safely write column data without synchronization overhead.
#[must_use]
pub fn csr_to_csc_f64_i64(a: &Csr<f64, i64>) -> Csc<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();
    let mut indptr = vec![0i64; ncols + 1];
    for &j in &a.indices {
        indptr[i64_to_usize(j) + 1] += 1;
    }
    for j in 0..ncols {
        indptr[j + 1] += indptr[j];
    }
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();
    let indices_addr = indices.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        for p in s..e {
            let j = i64_to_usize(a.indices[p]);
            let dst = i64_to_usize(next[j].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = indices_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(i));
                std::ptr::write(pv.add(dst), a.data[p]);
            }
        }
    });
    Csc::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Converts CSC (Compressed Sparse Column) to CSR (Compressed Sparse Row) format.
///
/// This operation reorders the matrix from column-major to row-major storage
/// while maintaining all nonzero values.
///
/// # Algorithm
/// 1. Count nonzeros per row (from column indices)
/// 2. Compute cumulative row pointers
/// 3. Use atomic operations to place each element in its correct row position
/// 4. Column indices are extracted from the CSC column ranges
///
/// # Parallelization
/// The conversion processes each column in parallel using atomic operations to
/// safely write row data without synchronization overhead.
#[must_use]
pub fn csc_to_csr_f64_i64(a: &Csc<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();
    let mut indptr = vec![0i64; nrows + 1];
    for &i in &a.indices {
        indptr[i64_to_usize(i) + 1] += 1;
    }
    for r in 0..nrows {
        indptr[r + 1] += indptr[r];
    }
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();
    let indices_addr = indices.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        for p in s..e {
            let i = i64_to_usize(a.indices[p]);
            let dst = i64_to_usize(next[i].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = indices_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(j));
                std::ptr::write(pv.add(dst), a.data[p]);
            }
        }
    });
    Csr::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Converts CSR (Compressed Sparse Row) to COO (Coordinate) format.
///
/// Expands the compressed row format into explicit row and column coordinates
/// for each nonzero element.
///
/// # Algorithm
/// For each row, use the row pointers to determine which elements belong to that row,
/// then write the row index along with the column index and value to the output arrays.
///
/// # Parallelization
/// Each row is processed independently in parallel without data races since each
/// row writes to its own disjoint portion of the output arrays.
#[must_use]
pub fn csr_to_coo_f64_i64(a: &Csr<f64, i64>) -> Coo<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();
    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let row_addr = row.as_mut_ptr() as usize;
    let col_addr = col.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let mut dst = s;
        for p in s..e {
            unsafe {
                let pr = row_addr as *mut i64;
                let pc = col_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pr.add(dst), usize_to_i64(i));
                std::ptr::write(pc.add(dst), a.indices[p]);
                std::ptr::write(pv.add(dst), a.data[p]);
            }
            dst += 1;
        }
    });
    Coo::from_parts_unchecked(nrows, ncols, row, col, data)
}

/// Converts CSC (Compressed Sparse Column) to COO (Coordinate) format.
///
/// Expands the compressed column format into explicit row and column coordinates
/// for each nonzero element.
///
/// # Algorithm
/// For each column, use the column pointers to determine which elements belong to that column,
/// then write the row index (from indices) and column index to the output arrays.
///
/// # Parallelization
/// Each column is processed independently in parallel without data races since each
/// column writes to its own disjoint portion of the output arrays.
#[must_use]
pub fn csc_to_coo_f64_i64(a: &Csc<f64, i64>) -> Coo<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();
    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let row_addr = row.as_mut_ptr() as usize;
    let col_addr = col.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        for p in s..e {
            unsafe {
                let pr = row_addr as *mut i64;
                let pc = col_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pr.add(p), a.indices[p]);
                std::ptr::write(pc.add(p), usize_to_i64(j));
                std::ptr::write(pv.add(p), a.data[p]);
            }
        }
    });
    Coo::from_parts_unchecked(nrows, ncols, row, col, data)
}

/// Converts COO (Coordinate) to CSR (Compressed Sparse Row) format.
///
/// Compresses the coordinate format into efficient row-major storage with automatic
/// duplicate handling (sums values with identical coordinates).
///
/// # Algorithm
/// 1. Create (row, col, value) tuples and sort by (row, col) order
/// 2. Iterate through sorted triples, accumulating values for duplicate (row, col) pairs
/// 3. Build row pointers by tracking row transitions
/// 4. Collect unique (col, accumulated_value) pairs into CSR format
///
/// # Properties
/// - Automatically sums duplicate entries
/// - Ensures column indices within each row are strictly increasing
/// - Handles empty rows correctly
#[must_use]
pub fn coo_to_csr_f64_i64(a: &Coo<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();
    let mut triples: Vec<(i64, i64, f64)> =
        (0..nnz).map(|k| (a.row[k], a.col[k], a.data[k])).collect();
    triples.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));
    let mut indptr = vec![0i64; nrows + 1];
    let mut indices: Vec<i64> = Vec::with_capacity(nnz);
    let mut data: Vec<f64> = Vec::with_capacity(nnz);
    let mut cur_row: i64 = -1;
    let mut last_col: i64 = -1;
    let mut acc: f64 = 0.0;
    for (r, c, v) in triples {
        if r != cur_row {
            if cur_row >= 0 && last_col >= 0 {
                indices.push(last_col);
                data.push(acc);
            }
            let r_us = i64_to_usize(r);
            let prev_us = i64_to_usize((cur_row + 1).max(0));
            let len_i64 = i64::try_from(indices.len()).unwrap_or(i64::MAX);
            for ptr in indptr.iter_mut().take(r_us + 1).skip(prev_us) {
                *ptr = len_i64;
            }
            cur_row = r;
            last_col = c;
            acc = v;
            continue;
        }
        if c == last_col {
            acc += v;
        } else {
            indices.push(last_col);
            data.push(acc);
            last_col = c;
            acc = v;
        }
    }
    if cur_row >= 0 && last_col >= 0 {
        indices.push(last_col);
        data.push(acc);
    }
    let start_row = if cur_row < 0 {
        0
    } else {
        i64_to_usize(cur_row + 1)
    };
    let len_i64 = i64::try_from(indices.len()).unwrap_or(i64::MAX);
    for ptr in indptr.iter_mut().take(nrows + 1).skip(start_row) {
        *ptr = len_i64;
    }
    Csr::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Converts COO (Coordinate) to CSC (Compressed Sparse Column) format.
///
/// Converts to CSC by leveraging the COO->CSR conversion with transposed inputs/outputs.
/// This approach reuses the CSR conversion logic while maintaining correctness.
///
/// # Algorithm
/// 1. Create a transposed COO (swap rows and cols)
/// 2. Convert transposed COO to CSR (which becomes CSC after transposition back)
/// 3. Return with appropriate dimension ordering
#[must_use]
pub fn coo_to_csc_f64_i64(a: &Coo<f64, i64>) -> Csc<f64, i64> {
    let coo_t = Coo::from_parts_unchecked(
        a.ncols,
        a.nrows,
        a.col.clone(),
        a.row.clone(),
        a.data.clone(),
    );
    let csr_t = coo_to_csr_f64_i64(&coo_t);
    Csc::from_parts_unchecked(a.nrows, a.ncols, csr_t.indptr, csr_t.indices, csr_t.data)
}

/// Computes the product of dimensions, panicking on overflow.
/// Used to validate and calculate total size of multi-dimensional arrays.
#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

/// Builds strides for row-major (C-style) ordering.
/// For shape [d0, d1, ..., dn], computes strides where stride[i] = d[i+1] * d[i+2] * ...
/// This allows converting multi-dimensional indices to linear indices.
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

/// Converts an N-dimensional COO array to a 2D COO matrix by projecting specified axes.
///
/// Reduces an N-D sparse array to a 2D sparse matrix by selecting which dimensions
/// form the row indices and which form the column indices.
///
/// # Arguments
/// * `a` - The N-dimensional COO array
/// * `row_axes` - Indices of dimensions that form the rows (must be non-duplicate and in bounds)
///
/// # Algorithm
/// 1. Validate row_axes (no duplicates, in bounds)
/// 2. Compute col_axes as remaining dimensions
/// 3. Compute row shape and column shape from selected axes
/// 4. For each nonzero element, compute linear indices in row-major order:
///    - Row index: linear combination of row dimensions and strides
///    - Column index: linear combination of column dimensions and strides
/// 5. Use parallel processing for large nnz, serial for small nnz
///
/// # Parallelization
/// Uses threshold (SMALL_NNZ_LIMIT) to decide between serial and parallel execution.
fn coond_axes_to_coo_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Coo<f64, i64> {
    let ndim = a.shape.len();
    let mut used = vec![false; ndim];
    for &ax in row_axes {
        assert!(ax < ndim, "row axis out of bounds");
        assert!(!used[ax], "duplicate axis in row_axes");
        used[ax] = true;
    }
    let mut col_axes: Vec<usize> = Vec::with_capacity(ndim - row_axes.len());
    for (d, &flag) in used.iter().enumerate().take(ndim) {
        if !flag {
            col_axes.push(d);
        }
    }
    let row_shape: Vec<usize> = row_axes.iter().map(|&d| a.shape[d]).collect();
    let col_shape: Vec<usize> = col_axes.iter().map(|&d| a.shape[d]).collect();
    let nrows = product_checked(&row_shape);
    let ncols = product_checked(&col_shape);
    let row_strides = build_strides_row_major(&row_shape);
    let col_strides = build_strides_row_major(&col_shape);
    let nnz = a.data.len();
    if nnz == 0 {
        return Coo::from_parts_unchecked(nrows, ncols, Vec::new(), Vec::new(), Vec::new());
    }
    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    if nnz < SMALL_NNZ_LIMIT {
        for k in 0..nnz {
            let base = k * ndim;
            let mut r: usize = 0;
            for (m, &d) in row_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base + d) });
                let s = if row_strides.is_empty() {
                    0
                } else {
                    row_strides[m]
                };
                r = r
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            let mut c2: usize = 0;
            for (m, &d) in col_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base + d) });
                let s = if col_strides.is_empty() {
                    0
                } else {
                    col_strides[m]
                };
                c2 = c2
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            row[k] = usize_to_i64(r);
            col[k] = usize_to_i64(c2);
        }
    } else {
        let row_addr = row.as_mut_ptr() as usize;
        let col_addr = col.as_mut_ptr() as usize;
        let indices = &a.indices;
        (0..nnz).into_par_iter().for_each(|k| {
            let base = k * ndim;
            let mut r: usize = 0;
            for (m, &d) in row_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *indices.get_unchecked(base + d) });
                let s = if row_strides.is_empty() {
                    0
                } else {
                    row_strides[m]
                };
                r = r
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            let mut c2: usize = 0;
            for (m, &d) in col_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *indices.get_unchecked(base + d) });
                let s = if col_strides.is_empty() {
                    0
                } else {
                    col_strides[m]
                };
                c2 = c2
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            unsafe {
                *(row_addr as *mut i64).add(k) = usize_to_i64(r);
                *(col_addr as *mut i64).add(k) = usize_to_i64(c2);
            }
        });
    }
    Coo::from_parts_unchecked(nrows, ncols, row, col, a.data.clone())
}

/// Converts an N-dimensional COO array to CSR by projecting specified axes as row dimensions.
/// Internally converts to COO with the specified row axes, then to CSR.
#[must_use]
pub fn coond_axes_to_csr_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Csr<f64, i64> {
    let coo = coond_axes_to_coo_f64_i64(a, row_axes);
    coo_to_csr_f64_i64(&coo)
}

/// Converts an N-dimensional COO array to CSC by projecting specified axes as row dimensions.
/// Internally converts to COO with the specified row axes, then to CSC.
#[must_use]
pub fn coond_axes_to_csc_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Csc<f64, i64> {
    let coo = coond_axes_to_coo_f64_i64(a, row_axes);
    coo_to_csc_f64_i64(&coo)
}

/// Converts an N-dimensional COO array to CSR using a single dimension as rows.
/// Convenience wrapper that treats one mode (dimension) as the row dimension.
///
/// # Arguments
/// * `a` - The N-dimensional COO array
/// * `axis` - The dimension index to use as row axis (must be in bounds)
#[must_use]
pub fn coond_mode_to_csr_f64_i64(a: &CooNd<f64, i64>, axis: usize) -> Csr<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    let row_axes = [axis];
    coond_axes_to_csr_f64_i64(a, &row_axes)
}

/// Converts an N-dimensional COO array to CSC using a single dimension as rows.
/// Convenience wrapper that treats one mode (dimension) as the row dimension.
///
/// # Arguments
/// * `a` - The N-dimensional COO array
/// * `axis` - The dimension index to use as row axis (must be in bounds)
#[must_use]
pub fn coond_mode_to_csc_f64_i64(a: &CooNd<f64, i64>, axis: usize) -> Csc<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    let row_axes = [axis];
    coond_axes_to_csc_f64_i64(a, &row_axes)
}
