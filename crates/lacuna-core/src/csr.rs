//! CSR format definitions and constructors

#[derive(Debug, Clone)]
pub struct Csr<T, I> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<I>,
    pub indices: Vec<I>,
    pub data: Vec<T>,
}

impl<T, I> Csr<T, I> {
    pub fn shape(&self) -> (usize, usize) { (self.nrows, self.ncols) }
    pub fn nnz(&self) -> usize { self.data.len() }
}

impl Csr<f64, i64> {
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        indptr: Vec<i64>,
        indices: Vec<i64>,
        data: Vec<f64>,
        check: bool,
    ) -> Result<Self, String> {
        if indptr.len() != nrows + 1 {
            return Err("indptr length must be nrows + 1".into());
        }
        if indices.len() != data.len() {
            return Err("indices and data must have equal length".into());
        }
        let nnz = indices.len();
        if (indptr.last().copied().unwrap_or(0)) as usize != nnz {
            return Err("indptr last element must equal nnz".into());
        }
        if indptr.first().copied().unwrap_or(0) != 0 {
            return Err("indptr first element must be 0".into());
        }
        if check {
            for w in indptr.windows(2) {
                if w[0] > w[1] { return Err("indptr must be non-decreasing".into()); }
                if w[0] < 0 || w[1] < 0 { return Err("indptr must be non-negative".into()); }
            }
            for i in 0..nrows {
                let start = indptr[i] as usize;
                let end = indptr[i+1] as usize;
                if end < start { return Err("indptr must be non-decreasing per row".into()); }
                if start > nnz || end > nnz { return Err("indptr elements must be within [0, nnz]".into()); }
                let mut prev_col = -1i64;
                for &j in &indices[start..end] {
                    if j < 0 || j as usize >= ncols { return Err("column index out of bounds".into()); }
                    if j <= prev_col { return Err("column indices must be strictly increasing within each row".into()); }
                    prev_col = j;
                }
            }
        }
        Ok(Csr { nrows, ncols, indptr, indices, data })
    }
}
