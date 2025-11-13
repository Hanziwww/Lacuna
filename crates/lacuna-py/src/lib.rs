use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArray2, IntoPyArray};
use numpy::ndarray::Array2;

use lacuna_core::Csr;
use lacuna_kernels::{
    spmv_f64_i64, spmm_f64_i64,
    sum_f64, row_sums_f64, col_sums_f64,
    transpose_f64_i64, prune_eps, eliminate_zeros,
    add_csr_f64_i64, mul_scalar_f64,
};

#[pyfunction]
fn spmv_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let x_slice = x.as_slice()?;
    let y = py.allow_threads(|| spmv_f64_i64(&csr, x_slice));
    Ok(PyArray1::from_vec(py, y))
}

#[pyfunction]
fn spmm_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>, // shape (ncols, k)
    check: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let b_arr = b.as_array();
    let shape = b_arr.shape();
    if shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("B must be 2D"));
    }
    let k = shape[1];
    if shape[0] != ncols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("B shape[0] must equal ncols"));
    }
    let b_vec: Vec<f64> = b_arr.iter().copied().collect();
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let y = py.allow_threads(|| spmm_f64_i64(&csr, &b_vec, k));

    // Build ndarray then convert to PyArray2
    let arr = Array2::from_shape_vec((nrows, k), y)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Output shape mismatch"))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
fn sum_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let out = py.allow_threads(|| sum_f64(&csr));
    Ok(out)
}

#[pyfunction]
fn row_sums_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let out = py.allow_threads(|| row_sums_f64(&csr));
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn col_sums_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let out = py.allow_threads(|| col_sums_f64(&csr));
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn transpose_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let t = py.allow_threads(|| transpose_f64_i64(&csr));
    Ok((
        PyArray1::from_vec(py, t.indptr),
        PyArray1::from_vec(py, t.indices),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

#[pyfunction]
fn prune_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    eps: f64,
    check: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let p = py.allow_threads(|| prune_eps(&csr, eps));
    Ok((
        PyArray1::from_vec(py, p.indptr),
        PyArray1::from_vec(py, p.indices),
        PyArray1::from_vec(py, p.data),
        p.nrows,
        p.ncols,
    ))
}

#[pyfunction]
fn eliminate_zeros_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
    let csr = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let p = py.allow_threads(|| eliminate_zeros(&csr));
    Ok((
        PyArray1::from_vec(py, p.indptr),
        PyArray1::from_vec(py, p.indices),
        PyArray1::from_vec(py, p.data),
        p.nrows,
        p.ncols,
    ))
}

#[pyfunction]
fn add_from_parts<'py>(
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
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
    let a = Csr::from_parts(
        a_nrows, a_ncols,
        a_indptr.as_slice()?.to_vec(),
        a_indices.as_slice()?.to_vec(),
        a_data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let b = Csr::from_parts(
        b_nrows, b_ncols,
        b_indptr.as_slice()?.to_vec(),
        b_indices.as_slice()?.to_vec(),
        b_data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let c = py.allow_threads(|| add_csr_f64_i64(&a, &b));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

#[pyfunction]
fn mul_scalar_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    check: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
    let a = Csr::from_parts(
        nrows, ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let c = py.allow_threads(|| mul_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(spmv_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(spmm_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(sum_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(row_sums_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(col_sums_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(prune_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(eliminate_zeros_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(add_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar_from_parts, m)?)?;
    Ok(())
}
