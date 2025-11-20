//! Utility functions: format conversion, cleanup (Array API support)

#[allow(unused_imports)]
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use lacuna_core::{Coo, CooNd, Csc, Csr};
use lacuna_kernels::{
    coo_to_csc_f64_i64, coo_to_csr_f64_i64, coond_axes_to_csc_f64_i64, coond_axes_to_csr_f64_i64,
    coond_mode_to_csc_f64_i64, coond_mode_to_csr_f64_i64, csc_to_coo_f64_i64, csc_to_csr_f64_i64,
    csr_to_coo_f64_i64, csr_to_csc_f64_i64, eliminate_zeros, prune_eps,
};

use super::helpers::{convert_axes_i64_to_usize, convert_shape_i64_to_usize};

// ========== Format Conversion Functions ==========

/// Convert CSR to CSC
#[pyfunction]
pub(crate) fn csr_to_csc_from_parts<'py>(
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
    let c = py.detach(|| csr_to_csc_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert CSC to CSR
#[pyfunction]
pub(crate) fn csc_to_csr_from_parts<'py>(
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
    let c = py.detach(|| csc_to_csr_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert CSR to COO
#[pyfunction]
pub(crate) fn csr_to_coo_from_parts<'py>(
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
    let c = py.detach(|| csr_to_coo_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.row),
        PyArray1::from_vec(py, c.col),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert CSC to COO
#[pyfunction]
pub(crate) fn csc_to_coo_from_parts<'py>(
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
    let c = py.detach(|| csc_to_coo_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.row),
        PyArray1::from_vec(py, c.col),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO to CSR
#[pyfunction]
pub(crate) fn coo_to_csr_from_parts<'py>(
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
    let c = py.detach(|| coo_to_csr_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO to CSC
#[pyfunction]
pub(crate) fn coo_to_csc_from_parts<'py>(
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
    let c = py.detach(|| coo_to_csc_f64_i64(&a));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO-ND to CSR by selecting a mode
#[pyfunction]
pub(crate) fn coond_mode_to_csr_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    mode: usize,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| coond_mode_to_csr_f64_i64(&a, mode));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO-ND to CSC by selecting a mode
#[pyfunction]
pub(crate) fn coond_mode_to_csc_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    mode: usize,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| coond_mode_to_csc_f64_i64(&a, mode));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO-ND to CSR by unfold along specified axes
#[pyfunction]
pub(crate) fn coond_axes_to_csr_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    row_axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let row_axes_us = convert_axes_i64_to_usize(row_axes.as_slice()?, "row_axes")?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| coond_axes_to_csr_f64_i64(&a, &row_axes_us));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Convert COO-ND to CSC by unfold along specified axes
#[pyfunction]
pub(crate) fn coond_axes_to_csc_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    row_axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let row_axes_us = convert_axes_i64_to_usize(row_axes.as_slice()?, "row_axes")?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| coond_axes_to_csc_f64_i64(&a, &row_axes_us));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

// ========== Cleanup Functions ==========

/// Remove entries with absolute value below threshold
#[pyfunction]
pub(crate) fn prune_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    threshold: f64,
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
    let b = py.detach(|| prune_eps(&a, threshold));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Remove entries that are exactly zero
#[pyfunction]
pub(crate) fn eliminate_zeros_from_parts<'py>(
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
    let b = py.detach(|| eliminate_zeros(&a));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}
