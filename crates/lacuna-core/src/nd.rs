pub trait SparseNd<T, I> {
    fn nnz(&self) -> usize;
    fn ndim(&self) -> usize;
    fn shape(&self) -> &[usize];
}

use crate::coo;

impl<T, I> SparseNd<T, I> for coo::CooNd<T, I> {
    #[inline]
    fn nnz(&self) -> usize {
        self.nnz()
    }

    #[inline]
    fn ndim(&self) -> usize {
        self.ndim()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}
