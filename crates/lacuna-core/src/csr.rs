//! Definitions and constructors for CSR (Compressed Sparse Row) format matrices.
//!
//! This file provides:
//! - The Csr struct for 2D sparse matrices in CSR format.
//! - Constructors, validation, and utility methods.
//!
//! CSR format is row-oriented: each row stores its nonzero column indices
//! and values in a contiguous block, with an indptr array marking row boundaries.

/// CSR (Compressed Sparse Row) format for 2D sparse matrices.
///
/// Stores nonzero elements in row-major order with column indices and row pointers.
///
/// - `data`: values of nonzero elements (length = nnz)
/// - `indices`: column indices for nonzeros in row order (length = nnz)
/// - `indptr`: row pointers (length = nrows + 1)
///   - indptr[i] = start index in indices/data for row i
///   - indptr[i+1] = start index for row i+1
///   - indptr[nrows] = nnz
/// - `nrows`: number of rows in the matrix
/// - `ncols`: number of columns in the matrix
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Csr<T, I> {
    pub data: Vec<T>,    // Nonzero values in row-major order
    pub indices: Vec<I>, // Column indices (length = nnz)
    pub indptr: Vec<I>,  // Row pointers (length = nrows + 1)
    pub ncols: usize,    // Number of columns
    pub nrows: usize,    // Number of rows
}

impl<T, I> Csr<T, I> {
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

impl Csr<f64, i64> {
    /// Constructs a Csr<f64, i64> from parts, with optional bounds and format checking.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `indptr` - Row pointers (length = nrows + 1)
    /// * `indices` - Column indices in row order (length = nnz)
    /// * `data` - Nonzero values in row order (length = nnz)
    /// * `check` - If true, validate CSR format invariants
    ///
    /// # Returns
    /// * `Ok(Csr)` if inputs are valid
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
        // Validate indptr length: must be nrows + 1
        let Some(expected_len) = nrows.checked_add(1) else {
            return Err("nrows overflow when adding 1".into());
        };
        if indptr.len() != expected_len {
            return Err("indptr length must be nrows + 1".into());
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
            // Validate per-row structure: column indices must be strictly increasing
            for (_row, (&start_i, &end_i)) in indptr
                .iter()
                .zip(indptr.iter().skip(1))
                .enumerate()
                .take(nrows)
            {
                // Convert indptr pointers to usizes
                let Ok(start) = usize::try_from(start_i) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                let Ok(end) = usize::try_from(end_i) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                if end < start {
                    return Err("indptr must be non-decreasing per row".into());
                }
                if start > nnz || end > nnz {
                    return Err("indptr elements must be within [0, nnz]".into());
                }
                // Check column indices within the row are strictly increasing and in bounds
                let mut prev_col = -1_i64;
                let Some(row_indices) = indices.get(start..end) else {
                    return Err("indptr elements must be within [0, nnz]".into());
                };
                for &j in row_indices {
                    let out_of_bounds = usize::try_from(j).map_or(true, |col| col >= ncols);
                    if j < 0 || out_of_bounds {
                        return Err("column index out of bounds".into());
                    }
                    if j <= prev_col {
                        return Err(
                            "column indices must be strictly increasing within each row".into()
                        );
                    }
                    prev_col = j;
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

    /// Constructs a Csr<f64, i64> from parts without any checks.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - indptr.len() == nrows + 1
    /// - indices.len() == data.len() == nnz
    /// - indptr[0] == 0 and indptr[nrows] == nnz
    /// - indptr is non-decreasing
    /// - For each row: column indices are strictly increasing and within [0, ncols)
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
