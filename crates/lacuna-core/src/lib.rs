//! Core data structures and traits for Lacuna (pure Rust)

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
