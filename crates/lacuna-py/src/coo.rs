use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use lacuna_core::Coo;
use lacuna_kernels::{
    col_sums_coo_f64, eliminate_zeros_coo, mul_scalar_coo_f64, prune_eps_coo, row_sums_coo_f64,
    spmm_coo_f64_i64, spmv_coo_f64_i64, sum_coo_f64,
};

#[pyclass]
pub struct Coo64 {
    pub(crate) inner: Coo<f64, i64>,
}

#[pymethods]
impl Coo64 {
    #[new]
    fn new(
        nrows: usize,
        ncols: usize,
        row: PyReadonlyArray1<'_, i64>,
        col: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
        check: bool,
    ) -> PyResult<Self> {
        let coo_mat = Coo::from_parts(
            nrows,
            ncols,
            row.as_slice()?.to_vec(),
            col.as_slice()?.to_vec(),
            data.as_slice()?.to_vec(),
            check,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner: coo_mat })
    }

    fn shape(&self) -> (usize, usize) {
        (self.inner.nrows, self.inner.ncols)
    }

    fn nnz(&self) -> usize {
        self.inner.data.len()
    }

    fn spmv<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xv: Vec<f64> = x.as_slice()?.to_vec();
        let y = py.detach(|| spmv_coo_f64_i64(&self.inner, &xv));
        Ok(PyArray1::from_vec(py, y))
    }

    fn spmm<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let b_arr = b.as_array();
        let shape = b_arr.shape();
        if shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "B must be 2D",
            ));
        }
        let k = shape[1];
        if shape[0] != self.inner.ncols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "B shape[0] must equal ncols",
            ));
        }
        let b_vec: Vec<f64> = b_arr.iter().copied().collect();
        let y = py.detach(|| spmm_coo_f64_i64(&self.inner, &b_vec, k));

        let arr = Array2::from_shape_vec((self.inner.nrows, k), y).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Output shape mismatch")
        })?;
        Ok(arr.into_pyarray(py))
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<f64> {
        let out = py.detach(|| sum_coo_f64(&self.inner));
        Ok(out)
    }

    fn row_sums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out = py.detach(|| row_sums_coo_f64(&self.inner));
        Ok(PyArray1::from_vec(py, out))
    }

    fn col_sums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out = py.detach(|| col_sums_coo_f64(&self.inner));
        Ok(PyArray1::from_vec(py, out))
    }

    fn prune<'py>(
        &self,
        py: Python<'py>,
        eps: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let p = py.detach(|| prune_eps_coo(&self.inner, eps));
        Ok((
            PyArray1::from_vec(py, p.row),
            PyArray1::from_vec(py, p.col),
            PyArray1::from_vec(py, p.data),
            p.nrows,
            p.ncols,
        ))
    }

    fn eliminate_zeros<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let p = py.detach(|| eliminate_zeros_coo(&self.inner));
        Ok((
            PyArray1::from_vec(py, p.row),
            PyArray1::from_vec(py, p.col),
            PyArray1::from_vec(py, p.data),
            p.nrows,
            p.ncols,
        ))
    }

    fn mul_scalar<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let c = py.detach(|| mul_scalar_coo_f64(&self.inner, alpha));
        Ok((
            PyArray1::from_vec(py, c.row),
            PyArray1::from_vec(py, c.col),
            PyArray1::from_vec(py, c.data),
            c.nrows,
            c.ncols,
        ))
    }
}
