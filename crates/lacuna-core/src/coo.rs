//! Definitions and constructors for COO (Coordinate) sparse matrix formats (2D and ND).
//
// This file provides:
// - The Coo struct for 2D sparse matrices in COO format.
// - The CooNd struct for N-dimensional sparse arrays in COO format.
// - Constructors, validation, and utility methods for both types.
//
// COO format stores nonzero elements as a list of coordinates and values.
// For 2D: (row, col, value). For ND: (indices..., value).

#[derive(Debug, Clone)]
#[non_exhaustive]
/// COO (Coordinate) format for 2D sparse matrices.
///
/// Stores nonzero elements as lists of row indices, column indices, and values.
///
/// - `data`: values of nonzero elements (length = nnz)
/// - `row`: row indices for each nonzero (length = nnz)
/// - `col`: column indices for each nonzero (length = nnz)
/// - `nrows`: number of rows in the matrix
/// - `ncols`: number of columns in the matrix
pub struct Coo<T, I> {
    pub data: Vec<T>, // Nonzero values
    pub row: Vec<I>,  // Row indices (length = nnz)
    pub col: Vec<I>,  // Column indices (length = nnz)
    pub ncols: usize, // Number of columns
    pub nrows: usize, // Number of rows
}

impl<T, I> Coo<T, I> {
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

impl Coo<f64, i64> {
    /// Constructs a `Coo<f64, i64>` from parts, with optional bounds checking.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `row` - Row indices (length = nnz)
    /// * `col` - Column indices (length = nnz)
    /// * `data` - Nonzero values (length = nnz)
    /// * `check` - If true, validate indices and lengths
    ///
    /// # Returns
    /// * `Ok(Coo)` if inputs are valid
    /// * `Err(String)` if validation fails
    #[inline]
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        row: Vec<i64>,
        col: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        // Check that all input vectors have the same length
        if row.len() != data.len() || col.len() != data.len() {
            return Err("row/col/data must have equal length".into());
        }
        if check {
            let nnz = data.len();
            for k in 0..nnz {
                let i = row[k];
                let j = col[k];
                // Indices must be non-negative
                if i < 0 || j < 0 {
                    return Err("indices must be non-negative".into());
                }
                // Indices must be within bounds
                let ok_i = usize::try_from(i).is_ok_and(|ii| ii < nrows);
                let ok_j = usize::try_from(j).is_ok_and(|jj| jj < ncols);
                if !ok_i || !ok_j {
                    return Err("indices out of bounds".into());
                }
            }
        }
        Ok(Self {
            data,
            row,
            col,
            ncols,
            nrows,
        })
    }

    /// Constructs a `Coo<f64, i64>` from parts without any checks.
    ///
    /// # Safety
    /// Caller must ensure inputs are valid.
    #[inline]
    #[must_use]
    pub const fn from_parts_unchecked(
        nrows: usize,
        ncols: usize,
        row: Vec<i64>,
        col: Vec<i64>,
        data: Vec<f64>,
    ) -> Self {
        Self {
            data,
            row,
            col,
            ncols,
            nrows,
        }
    }
}

/// COO format for N-dimensional sparse arrays.
///
/// Stores nonzero elements as a flat list of indices and values.
///
/// - `data`: values of nonzero elements (length = nnz)
/// - `indices`: flattened indices for each nonzero (length = nnz * ndim)
/// - `shape`: shape of the ND array (length = ndim)
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CooNd<T, I> {
    pub data: Vec<T>,      // Nonzero values
    pub indices: Vec<I>,   // Flattened indices (length = nnz * ndim)
    pub shape: Vec<usize>, // Shape of the ND array
}

impl<T, I> CooNd<T, I> {
    /// Returns the number of nonzero elements (nnz).
    #[inline]
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of dimensions (ndim).
    #[inline]
    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl CooNd<f64, i64> {
    /// Constructs a `CooNd<f64, i64>` from parts, with optional bounds checking.
    ///
    /// # Arguments
    /// * `shape` - Shape of the ND array (length = ndim)
    /// * `indices` - Flattened indices (length = nnz * ndim)
    /// * `data` - Nonzero values (length = nnz)
    /// * `check` - If true, validate indices and lengths
    ///
    /// # Returns
    /// * `Ok(CooNd)` if inputs are valid
    /// * `Err(String)` if validation fails
    #[inline]
    pub fn from_parts(
        shape: Vec<usize>,
        indices: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        let ndim = shape.len();
        // Shape must be non-empty
        if ndim == 0 {
            return Err("shape must be non-empty".into());
        }
        // If data is empty, indices must also be empty
        if data.is_empty() && !indices.is_empty() {
            return Err("indices must be empty when data is empty".into());
        }
        let nnz = data.len();
        // Indices length must be nnz * ndim
        let expected = nnz
            .checked_mul(ndim)
            .ok_or_else(|| "indices length overflow".to_string())?;
        if indices.len() != expected {
            return Err("indices length must be nnz * ndim".into());
        }
        if check {
            for k in 0..nnz {
                for d in 0..ndim {
                    let idx = indices[k * ndim + d];
                    // Indices must be non-negative
                    if idx < 0 {
                        return Err("indices must be non-negative".into());
                    }
                    // Indices must be within bounds for each dimension
                    let ok = usize::try_from(idx).is_ok_and(|ii| ii < shape[d]);
                    if !ok {
                        return Err("index out of bounds".into());
                    }
                }
            }
        }
        Ok(Self {
            data,
            indices,
            shape,
        })
    }

    /// Constructs a `CooNd<f64, i64>` from parts without any checks.
    ///
    /// # Safety
    /// Caller must ensure inputs are valid.
    #[inline]
    #[must_use]
    pub const fn from_parts_unchecked(
        shape: Vec<usize>,
        indices: Vec<i64>,
        data: Vec<f64>,
    ) -> Self {
        Self {
            data,
            indices,
            shape,
        }
    }
}
