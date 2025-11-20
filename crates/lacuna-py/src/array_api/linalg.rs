//! Linear algebra operations for sparse arrays (Array API aligned)

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use lacuna_core::{Coo, Csc, Csr};
use lacuna_kernels::{
    spmm_f64_i64, spmv_f64_i64, transpose_coo_f64_i64, transpose_csc_f64_i64, transpose_f64_i64,
};

/// Sparse matrix-vector multiply: y = A @ x
#[pyfunction]
pub(crate) fn spmv_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let x_slice = x.as_slice()?;
    let y = py.detach(|| spmv_f64_i64(&a, x_slice));
    Ok(PyArray1::from_vec(py, y))
}

/// Sparse matrix-matrix multiply: C = A @ B
#[pyfunction]
pub(crate) fn spmm_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let b_shape = b.shape();
    if b_shape[0] != ncols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Inner dimensions must match",
        ));
    }
    let k = b_shape[1];
    let b_slice = b.as_slice()?;
    let c_flat = py.detach(|| spmm_f64_i64(&a, b_slice, k));
    let c_arr = numpy::ndarray::Array2::from_shape_vec((nrows, k), c_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(c_arr.into_pyarray(py))
}

/// Transpose CSR matrix
#[pyfunction]
pub(crate) fn transpose_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let t = py.detach(|| transpose_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, t.indptr),
        PyArray1::from_vec(py, t.indices),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

/// Transpose CSC matrix
#[pyfunction]
pub(crate) fn transpose_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let t = py.detach(|| transpose_csc_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, t.indptr),
        PyArray1::from_vec(py, t.indices),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

/// Transpose COO matrix
#[pyfunction]
pub(crate) fn transpose_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let t = py.detach(|| transpose_coo_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, t.row),
        PyArray1::from_vec(py, t.col),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}
