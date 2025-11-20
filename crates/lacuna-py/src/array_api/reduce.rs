//! Statistical reductions for sparse arrays (Array API aligned)

#[allow(unused_imports)]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use lacuna_core::{CooNd, Csr};
use lacuna_kernels::{
    col_sums_f64, mean_coond_f64, reduce_mean_axes_coond_f64_i64, reduce_sum_axes_coond_f64_i64,
    row_sums_f64, sum_coond_f64, sum_f64,
};

use super::helpers::{
    convert_axes_i64_to_usize, convert_shape_i64_to_usize, convert_shape_usize_to_i64,
};

/// Sum all elements in CSR matrix
#[pyfunction]
pub(crate) fn sum_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let s = py.detach(|| sum_f64(&a));
    Ok(s)
}

/// Sum rows of CSR matrix
#[pyfunction]
pub(crate) fn row_sums_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
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
    let sums = py.detach(|| row_sums_f64(&a));
    Ok(PyArray1::from_vec(py, sums))
}

/// Sum columns of CSR matrix
#[pyfunction]
pub(crate) fn col_sums_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
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
    let sums = py.detach(|| col_sums_f64(&a));
    Ok(PyArray1::from_vec(py, sums))
}

/// Sum all elements in COO-ND array
#[pyfunction]
pub(crate) fn coond_sum_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let s = py.detach(|| sum_coond_f64(&a));
    Ok(s)
}

/// Mean of all elements in COO-ND array
#[pyfunction]
pub(crate) fn coond_mean_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let m = py.detach(|| mean_coond_f64(&a));
    Ok(m)
}

/// Sum over specified axes in COO-ND array
#[pyfunction]
pub(crate) fn coond_reduce_sum_axes_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let axes_us = convert_axes_i64_to_usize(axes.as_slice()?, "axes")?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let res = py.detach(|| reduce_sum_axes_coond_f64_i64(&a, &axes_us));

    Ok((
        PyArray1::from_vec(py, convert_shape_usize_to_i64(&res.shape)),
        PyArray1::from_vec(py, res.indices),
        PyArray1::from_vec(py, res.data),
    ))
}

/// Mean over specified axes in COO-ND array
#[pyfunction]
pub(crate) fn coond_reduce_mean_axes_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let axes_us = convert_axes_i64_to_usize(axes.as_slice()?, "axes")?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let res = py.detach(|| reduce_mean_axes_coond_f64_i64(&a, &axes_us));

    Ok((
        PyArray1::from_vec(py, convert_shape_usize_to_i64(&res.shape)),
        PyArray1::from_vec(py, res.indices),
        PyArray1::from_vec(py, res.data),
    ))
}
