//! Element-wise operations for sparse arrays (Array API aligned)

#[allow(unused_imports)]
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use lacuna_core::{Coo, CooNd, Csc, Csr};
use lacuna_kernels::{
    abs_scalar_coo_f64, abs_scalar_coond_f64, abs_scalar_csc_f64, abs_scalar_f64, add_csr_f64_i64,
    div_csc_f64_i64, div_csr_f64_i64, floordiv_csc_f64_i64, floordiv_csr_f64_i64,
    floordiv_scalar_coo_f64, floordiv_scalar_coond_f64, floordiv_scalar_csc_f64,
    floordiv_scalar_f64, hadamard_broadcast_coond_f64_i64, hadamard_csr_f64_i64,
    mul_scalar_coo_f64, mul_scalar_coond_f64, mul_scalar_csc_f64, mul_scalar_f64, pow_csc_f64_i64,
    pow_csr_f64_i64, pow_scalar_coo_f64, pow_scalar_coond_f64, pow_scalar_csc_f64, pow_scalar_f64,
    rem_csc_f64_i64, rem_csr_f64_i64, rem_scalar_coo_f64, rem_scalar_coond_f64, rem_scalar_csc_f64,
    rem_scalar_f64, sign_scalar_coo_f64, sign_scalar_coond_f64, sign_scalar_csc_f64,
    sign_scalar_f64, sub_csr_f64_i64,
};

/// Add two CSR matrices: C = A + B
#[pyfunction]
pub(crate) fn add_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| add_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Sign for CSR: sign(A)
#[pyfunction]
pub(crate) fn sign_from_parts<'py>(
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
    let b = py.detach(|| sign_scalar_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Sign for CSC: sign(A)
#[pyfunction]
pub(crate) fn sign_csc_from_parts<'py>(
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
    let b = py.detach(|| sign_scalar_csc_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Sign for COO: sign(A)
#[pyfunction]
pub(crate) fn sign_coo_from_parts<'py>(
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
    let b = py.detach(|| sign_scalar_coo_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Sign for COOND: sign(X)
#[pyfunction]
pub(crate) fn sign_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| sign_scalar_coond_f64(&a));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

/// Absolute value for CSR: |A|
#[pyfunction]
pub(crate) fn abs_from_parts<'py>(
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
    let b = py.detach(|| abs_scalar_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Absolute value for CSC: |A|
#[pyfunction]
pub(crate) fn abs_csc_from_parts<'py>(
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
    let b = py.detach(|| abs_scalar_csc_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Absolute value for COO: |A|
#[pyfunction]
pub(crate) fn abs_coo_from_parts<'py>(
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
    let b = py.detach(|| abs_scalar_coo_f64(&a));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Absolute value for COOND: |X|
#[pyfunction]
pub(crate) fn abs_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| abs_scalar_coond_f64(&a));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

#[pyfunction]
pub(crate) fn mul_scalar_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| mul_scalar_csc_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

#[pyfunction]
pub(crate) fn mul_scalar_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| mul_scalar_coo_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

#[pyfunction]
pub(crate) fn mul_scalar_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| mul_scalar_coond_f64(&a, alpha));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

/// Remainder of two CSR matrices element-wise: C = remainder(A, B)
#[pyfunction]
pub(crate) fn remainder_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| rem_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Remainder of two CSC matrices element-wise: C = remainder(A, B)
#[pyfunction]
pub(crate) fn remainder_csc_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csc::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csc::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| rem_csc_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Remainder CSR by scalar: B = remainder(A, alpha)
#[pyfunction]
pub(crate) fn remainder_scalar_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| rem_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Remainder CSC by scalar
#[pyfunction]
pub(crate) fn remainder_scalar_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| rem_scalar_csc_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Remainder COO by scalar
#[pyfunction]
pub(crate) fn remainder_scalar_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| rem_scalar_coo_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Remainder COOND by scalar
#[pyfunction]
pub(crate) fn remainder_scalar_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| rem_scalar_coond_f64(&a, alpha));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

/// Power two CSR matrices element-wise: C = pow(A, B)
#[pyfunction]
pub(crate) fn pow_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| pow_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Power two CSC matrices element-wise: C = pow(A, B)
#[pyfunction]
pub(crate) fn pow_csc_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csc::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csc::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| pow_csc_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Power CSR by scalar: B = A ** alpha
#[pyfunction]
pub(crate) fn pow_scalar_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| pow_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Power CSC by scalar
#[pyfunction]
pub(crate) fn pow_scalar_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| pow_scalar_csc_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Power COO by scalar
#[pyfunction]
pub(crate) fn pow_scalar_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| pow_scalar_coo_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Power COOND by scalar
#[pyfunction]
pub(crate) fn pow_scalar_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| pow_scalar_coond_f64(&a, alpha));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

/// Floor-divide two CSC matrices element-wise: C = floor(A / B)
#[pyfunction]
pub(crate) fn floordiv_csc_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csc::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csc::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| floordiv_csc_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Floor-divide two CSR matrices element-wise: C = floor(A / B)
#[pyfunction]
pub(crate) fn floordiv_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| floordiv_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Divide two CSC matrices element-wise: C = A / B
#[pyfunction]
pub(crate) fn div_csc_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csc::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csc::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| div_csc_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Subtract two CSR matrices: C = A - B
#[pyfunction]
pub(crate) fn sub_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| sub_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Divide two CSR matrices element-wise: C = A / B
#[pyfunction]
pub(crate) fn div_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| div_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Element-wise (Hadamard) product of two CSR matrices: C = A ⊙ B
#[pyfunction]
pub(crate) fn hadamard_from_parts<'py>(
    py: Python<'py>,
    a_nrows: usize,
    a_ncols: usize,
    a_indptr: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_nrows: usize,
    b_ncols: usize,
    b_indptr: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let a = Csr::from_parts(
        a_nrows,
        a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = Csr::from_parts(
        b_nrows,
        b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let c = py.detach(|| hadamard_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

/// Multiply sparse array by scalar: B = alpha * A
#[pyfunction]
pub(crate) fn mul_scalar_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| mul_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Floor-divide CSR by scalar: B = floor(A / alpha)
#[pyfunction]
pub(crate) fn floordiv_scalar_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| floordiv_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Floor-divide CSC by scalar: B = floor(A / alpha)
#[pyfunction]
pub(crate) fn floordiv_scalar_csc_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| floordiv_scalar_csc_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.indptr),
        PyArray1::from_vec(py, b.indices),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Floor-divide COO by scalar: B = floor(A / alpha)
#[pyfunction]
pub(crate) fn floordiv_scalar_coo_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    row: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
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
    let b = py.detach(|| floordiv_scalar_coo_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, b.row),
        PyArray1::from_vec(py, b.col),
        PyArray1::from_vec(py, b.data),
        b.nrows,
        b.ncols,
    ))
}

/// Floor-divide COOND by scalar: B = floor(A / alpha)
#[pyfunction]
pub(crate) fn floordiv_scalar_coond_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| floordiv_scalar_coond_f64(&a, alpha));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}

/// Broadcasting Hadamard product for COO-ND arrays
#[pyfunction]
pub(crate) fn coond_hadamard_broadcast_from_parts<'py>(
    py: Python<'py>,
    a_shape: PyReadonlyArray1<'py, i64>,
    a_indices: PyReadonlyArray1<'py, i64>,
    a_data: PyReadonlyArray1<'py, f64>,
    b_shape: PyReadonlyArray1<'py, i64>,
    b_indices: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a_shape_i64 = a_shape.as_slice()?.to_vec();
    let mut a_shape_us: Vec<usize> = Vec::with_capacity(a_shape_i64.len());
    for &s in &a_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        a_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        a_shape_us,
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let b_shape_i64 = b_shape.as_slice()?.to_vec();
    let mut b_shape_us: Vec<usize> = Vec::with_capacity(b_shape_i64.len());
    for &s in &b_shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        b_shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let b = CooNd::from_parts(
        b_shape_us,
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let c = py.detach(|| hadamard_broadcast_coond_f64_i64(&a, &b));
    let shape_i64: Vec<i64> = c.shape.iter().map(|&s| i64::try_from(s).unwrap()).collect();
    Ok((
        PyArray1::from_vec(py, shape_i64),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
    ))
}
