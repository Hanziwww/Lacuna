//! Statistical reductions for sparse arrays (Array API aligned)

#[allow(unused_imports)]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use lacuna_core::{Coo, CooNd, Csc, Csr};
use lacuna_kernels::{
    col_maxs_coo_f64,
    col_maxs_csc_f64,
    col_maxs_f64,
    col_mins_coo_f64,
    col_mins_csc_f64,
    col_mins_f64,
    col_prods_coo_f64,
    // prod
    col_prods_csc_f64,
    col_prods_f64,
    col_stds_coo_f64,
    // var/std
    col_stds_csc_f64,
    col_stds_f64,
    col_sums_f64,
    col_vars_coo_f64,
    col_vars_csc_f64,
    col_vars_f64,
    max_coo_f64,
    max_csc_f64,
    max_f64,
    mean_coond_f64,
    min_coo_f64,
    min_csc_f64,
    min_f64,
    prod_coo_f64,
    prod_coond_f64,
    prod_csc_f64,
    prod_f64,
    reduce_mean_axes_coond_f64_i64,
    reduce_sum_axes_coond_f64_i64,
    row_maxs_coo_f64,
    row_maxs_csc_f64,
    row_maxs_f64,
    row_mins_coo_f64,
    row_mins_csc_f64,
    row_mins_f64,
    row_prods_coo_f64,
    row_prods_csc_f64,
    row_prods_f64,
    row_stds_coo_f64,
    row_stds_csc_f64,
    row_stds_f64,
    row_sums_f64,
    row_vars_coo_f64,
    row_vars_csc_f64,
    row_vars_f64,
    std_coo_f64,
    std_coond_f64,
    std_csc_f64,
    std_f64,
    sum_coond_f64,
    sum_f64,
    var_coo_f64,
    var_coond_f64,
    var_csc_f64,
    var_f64,
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

/// Min/Max for COO (global and per-axis)
#[pyfunction]
pub(crate) fn min_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| min_coo_f64(&a)))
}

#[pyfunction]
pub(crate) fn max_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| max_coo_f64(&a)))
}

#[pyfunction]
pub(crate) fn row_mins_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_mins_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn row_maxs_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_maxs_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_mins_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_mins_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_maxs_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_maxs_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

/// Min of all elements in CSR (treating implied zeros)
#[pyfunction]
pub(crate) fn min_from_parts<'py>(
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
    Ok(py.detach(|| min_f64(&a)))
}

/// Max of all elements in CSR (treating implied zeros)
#[pyfunction]
pub(crate) fn max_from_parts<'py>(
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
    Ok(py.detach(|| max_f64(&a)))
}

/// Row-wise mins in CSR (implied zeros lower-bound rows)
#[pyfunction]
pub(crate) fn row_mins_from_parts<'py>(
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
    let v = py.detach(|| row_mins_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

/// Row-wise maxs in CSR (implied zeros upper-bound rows)
#[pyfunction]
pub(crate) fn row_maxs_from_parts<'py>(
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
    let v = py.detach(|| row_maxs_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

/// Column-wise mins in CSR
#[pyfunction]
pub(crate) fn col_mins_from_parts<'py>(
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
    let v = py.detach(|| col_mins_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

/// Column-wise maxs in CSR
#[pyfunction]
pub(crate) fn col_maxs_from_parts<'py>(
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
    let v = py.detach(|| col_maxs_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

/// Min/Max for CSC (global and per-axis)
#[pyfunction]
pub(crate) fn min_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| min_csc_f64(&a)))
}

#[pyfunction]
pub(crate) fn max_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| max_csc_f64(&a)))
}

#[pyfunction]
pub(crate) fn row_mins_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_mins_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn row_maxs_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_maxs_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_mins_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_mins_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_maxs_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_maxs_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
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

// ===== Product (CSR/CSC/COO/COOND) =====

#[pyfunction]
pub(crate) fn prod_from_parts<'py>(
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
    Ok(py.detach(|| prod_f64(&a)))
}

#[pyfunction]
pub(crate) fn row_prods_from_parts<'py>(
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
    let v = py.detach(|| row_prods_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_prods_from_parts<'py>(
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
    let v = py.detach(|| col_prods_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn prod_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| prod_csc_f64(&a)))
}

#[pyfunction]
pub(crate) fn row_prods_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_prods_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_prods_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_prods_csc_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn prod_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| prod_coo_f64(&a)))
}

#[pyfunction]
pub(crate) fn row_prods_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_prods_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_prods_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_prods_coo_f64(&a));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn coond_prod_from_parts<'py>(
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
    Ok(py.detach(|| prod_coond_f64(&a)))
}

// ===== Variance / Standard Deviation =====

#[pyfunction]
pub(crate) fn var_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    Ok(py.detach(|| var_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn std_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    Ok(py.detach(|| std_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn row_vars_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    let v = py.detach(|| row_vars_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn row_stds_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    let v = py.detach(|| row_stds_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_vars_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    let v = py.detach(|| col_vars_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_stds_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    let v = py.detach(|| col_stds_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn var_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<f64> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| var_csc_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn std_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<f64> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| std_csc_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn row_vars_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_vars_csc_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn row_stds_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_stds_csc_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_vars_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_vars_csc_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_stds_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_stds_csc_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn var_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<f64> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| var_coo_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn std_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<f64> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(py.detach(|| std_coo_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn row_vars_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_vars_coo_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn row_stds_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| row_stds_coo_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_vars_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_vars_coo_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn col_stds_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let v = py.detach(|| col_stds_coo_f64(&a, correction));
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
pub(crate) fn coond_var_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    Ok(py.detach(|| var_coond_f64(&a, correction)))
}

#[pyfunction]
pub(crate) fn coond_std_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    correction: f64,
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
    Ok(py.detach(|| std_coond_f64(&a, correction)))
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
