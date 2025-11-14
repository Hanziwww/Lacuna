//! CSR format definitions and constructors

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Csr<T, I> {
    pub data: Vec<T>,
    pub indices: Vec<I>,
    pub indptr: Vec<I>,
    pub ncols: usize,
    pub nrows: usize,
}

impl<T, I> Csr<T, I> {
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

impl Csr<f64, i64> {
    #[inline]
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        indptr: Vec<i64>,
        indices: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        let Some(expected_len) = nrows.checked_add(1) else {
            return Err("nrows overflow when adding 1".into());
        };
        if indptr.len() != expected_len {
            return Err("indptr length must be nrows + 1".into());
        }
        if indices.len() != data.len() {
            return Err("indices and data must have equal length".into());
        }
        let nnz = indices.len();
        if usize::try_from(indptr.last().copied().unwrap_or(0)).ok() != Some(nnz) {
            return Err("indptr last element must equal nnz".into());
        }
        if indptr.first().copied().unwrap_or(0) != 0 {
            return Err("indptr first element must be 0".into());
        }
        if check {
            for (prev_ptr, next_ptr) in indptr.iter().zip(indptr.iter().skip(1)) {
                if prev_ptr > next_ptr {
                    return Err("indptr must be non-decreasing".into());
                }
                if *prev_ptr < 0 || *next_ptr < 0 {
                    return Err("indptr must be non-negative".into());
                }
            }
            for (_row, (&start_i, &end_i)) in indptr
                .iter()
                .zip(indptr.iter().skip(1))
                .enumerate()
                .take(nrows)
            {
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
