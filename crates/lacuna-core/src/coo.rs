//! COO format definitions (2D and ND) and constructors

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Coo<T, I> {
    pub data: Vec<T>,
    pub row: Vec<I>, // length nnz
    pub col: Vec<I>, // length nnz
    pub ncols: usize,
    pub nrows: usize,
}

impl<T, I> Coo<T, I> {
    #[inline]
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.data.len()
    }
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl Coo<f64, i64> {
    #[inline]
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        row: Vec<i64>,
        col: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        if row.len() != data.len() || col.len() != data.len() {
            return Err("row/col/data must have equal length".into());
        }
        if check {
            let nnz = data.len();
            for k in 0..nnz {
                let i = row[k];
                let j = col[k];
                if i < 0 || j < 0 {
                    return Err("indices must be non-negative".into());
                }
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

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CooNd<T, I> {
    pub data: Vec<T>,
    pub indices: Vec<I>, // flattened shape (nnz * ndim)
    pub shape: Vec<usize>,
}

impl<T, I> CooNd<T, I> {
    #[inline]
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.data.len()
    }
    #[inline]
    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl CooNd<f64, i64> {
    #[inline]
    pub fn from_parts(
        shape: Vec<usize>,
        indices: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err("shape must be non-empty".into());
        }
        if data.is_empty() && !indices.is_empty() {
            return Err("indices must be empty when data is empty".into());
        }
        let nnz = data.len();
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
                    if idx < 0 {
                        return Err("indices must be non-negative".into());
                    }
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
