//! Manipulation operations for sparse arrays (Array API aligned)

#[allow(unused_imports)]
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use lacuna_core::CooNd;
use lacuna_kernels::{permute_axes_coond_f64_i64, reshape_coond_f64_i64};

use super::helpers::{
    convert_axes_i64_to_usize, convert_shape_i64_to_usize, convert_shape_usize_to_i64,
};

/// Permute axes of COO-ND array
#[pyfunction]
pub(crate) fn coond_permute_axes_from_parts<'py>(
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
    // Convert shapes and axes using helpers
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let axes_us = convert_axes_i64_to_usize(axes.as_slice()?, "axes")?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let res = py.detach(|| permute_axes_coond_f64_i64(&a, &axes_us));

    Ok((
        PyArray1::from_vec(py, convert_shape_usize_to_i64(&res.shape)),
        PyArray1::from_vec(py, res.indices),
        PyArray1::from_vec(py, res.data),
    ))
}

/// Reshape COO-ND array
#[pyfunction]
pub(crate) fn coond_reshape_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    new_shape: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    // Convert shapes using helpers
    let shape_us = convert_shape_i64_to_usize(shape.as_slice()?)?;
    let new_shape_us = convert_shape_i64_to_usize(new_shape.as_slice()?)?;

    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let res = py.detach(|| reshape_coond_f64_i64(&a, &new_shape_us));

    Ok((
        PyArray1::from_vec(py, convert_shape_usize_to_i64(&res.shape)),
        PyArray1::from_vec(py, res.indices),
        PyArray1::from_vec(py, res.data),
    ))
}
