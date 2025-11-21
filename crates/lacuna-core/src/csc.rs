//! Definitions and constructors for CSC (Compressed Sparse Column) format matrices.
//!
//! This file provides:
//! - The Csc struct for 2D sparse matrices in CSC format.
//! - Constructors, validation, and utility methods.
//!
//! CSC format is column-oriented: each column stores its nonzero row indices
//! and values in a contiguous block, with an indptr array marking column boundaries.

/// CSC (Compressed Sparse Column) format for 2D sparse matrices.
///
/// Stores nonzero elements in column-major order with row indices and column pointers.
///
/// - `data`: values of nonzero elements (length = nnz)
/// - `indices`: row indices for nonzeros in column order (length = nnz)
/// - `indptr`: column pointers (length = ncols + 1)
///   - indptr[j] = start index in indices/data for column j
///   - indptr[j+1] = start index for column j+1
///   - indptr[ncols] = nnz
/// - `nrows`: number of rows in the matrix
/// - `ncols`: number of columns in the matrix
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Csc<T, I> {
    pub data: Vec<T>,    // Nonzero values in column-major order
    pub indices: Vec<I>, // Row indices (length = nnz)
    pub indptr: Vec<I>,  // Column pointers (length = ncols + 1)
    pub ncols: usize,    // Number of columns
    pub nrows: usize,    // Number of rows
}

impl<T, I> Csc<T, I> {
    /// Returns the number of nonzero elements (nnz).
    #[inline]
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the shape of the matrix as (nrows, ncols).
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl Csc<f64, i64> {
    /// Constructs a Csc<f64, i64> from parts, with optional bounds and format checking.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `indptr` - Column pointers (length = ncols + 1)
    /// * `indices` - Row indices in column order (length = nnz)
    /// * `data` - Nonzero values in column order (length = nnz)
    /// * `check` - If true, validate CSC format invariants
    ///
    /// # Returns
    /// * `Ok(Csc)` if inputs are valid
    /// * `Err(String)` if validation fails
    #[inline]
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        indptr: Vec<i64>,
        indices: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        // Validate indptr length: must be ncols + 1
        let Some(expected_len) = ncols.checked_add(1) else {
            return Err("ncols overflow when adding 1".into());
        };
        if indptr.len() != expected_len {
            return Err("indptr length must be ncols + 1".into());
        }
        // Validate indices and data have equal length
        if indices.len() != data.len() {
            return Err("indices and data must have equal length".into());
        }
        let nnz = indices.len();
        // Validate indptr boundaries
        if usize::try_from(indptr.last().copied().unwrap_or(0)).ok() != Some(nnz) {
            return Err("indptr last element must equal nnz".into());
        }
        if indptr.first().copied().unwrap_or(0) != 0 {
            return Err("indptr first element must be 0".into());
        }
        if check {
            // Validate indptr is non-decreasing and non-negative
            for (prev_ptr, next_ptr) in indptr.iter().zip(indptr.iter().skip(1)) {
                if prev_ptr > next_ptr {
                    return Err("indptr must be non-decreasing".into());
                }
                if *prev_ptr < 0 || *next_ptr < 0 {
                    return Err("indptr must be non-negative".into());
                }
            }
            // Validate per-column structure: row indices must be strictly increasing
            for (_col, (&start_i, &end_i)) in indptr
                .iter()
                .zip(indptr.iter().skip(1))
                .enumerate()
                .take(ncols)
            {
                // Convert indptr pointers to usizes
                let Ok(start) = usize::try_from(start_i) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                let Ok(end) = usize::try_from(end_i) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                if end < start {
                    return Err("indptr must be non-decreasing per column".into());
                }
                if start > nnz || end > nnz {
                    return Err("indptr elements must be within [0, nnz]".into());
                }
                // Check row indices within the column are strictly increasing and in bounds
                let mut prev_row = -1_i64;
                let Some(col_indices) = indices.get(start..end) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                for &i in col_indices {
                    let out_of_bounds = usize::try_from(i).map_or(true, |row| row >= nrows);
                    if i < 0 || out_of_bounds {
                        return Err("row index out of bounds".into());
                    }
                    if i <= prev_row {
                        return Err(
                            "row indices must be strictly increasing within each column".into()
                        );
                    }
                    prev_row = i;
                }
            }
        }
        Ok(Self {
            data,
            indices,
            indptr,
            ncols,
            nrows,
        })
    }

    /// Constructs a Csc<f64, i64> from parts without any checks.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `indptr.len()` == ncols + 1
    /// - `indices.len()` == `data.len()` == nnz
    /// - `indptr[0] == 0` and `indptr[ncols] == nnz`
    /// - `indptr` is non-decreasing
    /// - For each column: row indices are strictly increasing and within `[0, nrows)`
    #[inline]
    #[must_use]
    pub const fn from_parts_unchecked(
        nrows: usize,
        ncols: usize,
        indptr: Vec<i64>,
        indices: Vec<i64>,
        data: Vec<f64>,
    ) -> Self {
        Self {
            data,
            indices,
            indptr,
            ncols,
            nrows,
        }
    }
}
