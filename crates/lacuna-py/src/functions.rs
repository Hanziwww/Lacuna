#![allow(clippy::redundant_pub_crate)]
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use lacuna_core::{Coo, Csc, Csr, CooNd};
use lacuna_kernels::{
    add_csr_f64_i64, col_sums_f64, coo_to_csc_f64_i64, coo_to_csr_f64_i64, csc_to_coo_f64_i64,
    csc_to_csr_f64_i64, csr_to_coo_f64_i64, csr_to_csc_f64_i64, eliminate_zeros,
    coond_mode_to_csr_f64_i64, coond_mode_to_csc_f64_i64, coond_axes_to_csr_f64_i64,
    coond_axes_to_csc_f64_i64, permute_axes_coond_f64_i64, reduce_sum_axes_coond_f64_i64,
    reduce_mean_axes_coond_f64_i64, hadamard_broadcast_coond_f64_i64, reshape_coond_f64_i64,
    hadamard_csr_f64_i64, mul_scalar_f64, prune_eps, row_sums_f64, spmm_f64_i64, spmv_f64_i64,
    sub_csr_f64_i64, sum_coond_f64, mean_coond_f64, sum_f64, transpose_coo_f64_i64,
    transpose_csc_f64_i64, transpose_f64_i64,
};

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

#[pyfunction]
pub(crate) fn coond_mean_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
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

#[pyfunction]
pub(crate) fn coond_sum_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    check: bool,
) -> PyResult<f64> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
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

#[pyfunction]
pub(crate) fn coond_permute_axes_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    perm: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>, // new shape (i64)
    Bound<'py, PyArray1<i64>>, // new indices
    Bound<'py, PyArray1<f64>>, // data
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let perm_i64 = perm.as_slice()?.to_vec();
    let mut perm_us: Vec<usize> = Vec::with_capacity(perm_i64.len());
    for &p in &perm_i64 {
        if p < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "perm must be non-negative",
            ));
        }
        perm_us.push(usize::try_from(p).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("perm value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let p = py.detach(|| permute_axes_coond_f64_i64(&a, &perm_us));
    let new_shape_i64: Vec<i64> = p.shape.iter().map(|&x| x as i64).collect();
    Ok((
        PyArray1::from_vec(py, new_shape_i64),
        PyArray1::from_vec(py, p.indices),
        PyArray1::from_vec(py, p.data),
    ))
}

#[pyfunction]
pub(crate) fn coond_reduce_sum_axes_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>, // new shape (i64)
    Bound<'py, PyArray1<i64>>, // new indices
    Bound<'py, PyArray1<f64>>, // data
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let axes_i64 = axes.as_slice()?.to_vec();
    let mut axes_us: Vec<usize> = Vec::with_capacity(axes_i64.len());
    for &ax in &axes_i64 {
        if ax < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "axes must be non-negative",
            ));
        }
        axes_us.push(usize::try_from(ax).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("axes value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let r = py.detach(|| reduce_sum_axes_coond_f64_i64(&a, &axes_us));
    let new_shape_i64: Vec<i64> = r.shape.iter().map(|&x| x as i64).collect();
    Ok((
        PyArray1::from_vec(py, new_shape_i64),
        PyArray1::from_vec(py, r.indices),
        PyArray1::from_vec(py, r.data),
    ))
}

#[pyfunction]
pub(crate) fn coond_reduce_mean_axes_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axes: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>, // new shape (i64)
    Bound<'py, PyArray1<i64>>, // new indices
    Bound<'py, PyArray1<f64>>, // data
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let axes_i64 = axes.as_slice()?.to_vec();
    let mut axes_us: Vec<usize> = Vec::with_capacity(axes_i64.len());
    for &ax in &axes_i64 {
        if ax < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "axes must be non-negative",
            ));
        }
        axes_us.push(usize::try_from(ax).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("axes value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let r = py.detach(|| reduce_mean_axes_coond_f64_i64(&a, &axes_us));
    let new_shape_i64: Vec<i64> = r.shape.iter().map(|&x| x as i64).collect();
    Ok((
        PyArray1::from_vec(py, new_shape_i64),
        PyArray1::from_vec(py, r.indices),
        PyArray1::from_vec(py, r.data),
    ))
}

#[pyfunction]
pub(crate) fn coond_reshape_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    new_shape: PyReadonlyArray1<'py, i64>,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>, // new shape (i64)
    Bound<'py, PyArray1<i64>>, // new indices
    Bound<'py, PyArray1<f64>>, // data
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let new_i64 = new_shape.as_slice()?.to_vec();
    let mut new_us: Vec<usize> = Vec::with_capacity(new_i64.len());
    for &s in &new_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "new_shape must be non-negative",
            ));
        }
        new_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("new_shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let r = py.detach(|| reshape_coond_f64_i64(&a, &new_us));
    let new_shape_i64: Vec<i64> = r.shape.iter().map(|&x| x as i64).collect();
    Ok((
        PyArray1::from_vec(py, new_shape_i64),
        PyArray1::from_vec(py, r.indices),
        PyArray1::from_vec(py, r.data),
    ))
}

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
    Bound<'py, PyArray1<i64>>, // out shape
    Bound<'py, PyArray1<i64>>, // out indices
    Bound<'py, PyArray1<f64>>, // out data
)> {
    let ash_i64 = a_shape.as_slice()?.to_vec();
    let mut ash: Vec<usize> = Vec::with_capacity(ash_i64.len());
    for &s in &ash_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("a_shape must be non-negative"));
        }
        ash.push(usize::try_from(s).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("a_shape value overflow"))?);
    }
    let bsh_i64 = b_shape.as_slice()?.to_vec();
    let mut bsh: Vec<usize> = Vec::with_capacity(bsh_i64.len());
    for &s in &bsh_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("b_shape must be non-negative"));
        }
        bsh.push(usize::try_from(s).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("b_shape value overflow"))?);
    }
    let a = CooNd::from_parts(ash, a_indices.as_slice()?.to_vec(), a_data.as_slice()?.to_vec(), check)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let b = CooNd::from_parts(bsh, b_indices.as_slice()?.to_vec(), b_data.as_slice()?.to_vec(), check)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let r = py.detach(|| hadamard_broadcast_coond_f64_i64(&a, &b));
    let out_shape_i64: Vec<i64> = r.shape.iter().map(|&x| x as i64).collect();
    Ok((
        PyArray1::from_vec(py, out_shape_i64),
        PyArray1::from_vec(py, r.indices),
        PyArray1::from_vec(py, r.data),
    ))
}

#[pyfunction]
pub(crate) fn coond_mode_to_csr_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axis: usize,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let csr = py.detach(|| coond_mode_to_csr_f64_i64(&a, axis));
    Ok((
        PyArray1::from_vec(py, csr.indptr),
        PyArray1::from_vec(py, csr.indices),
        PyArray1::from_vec(py, csr.data),
        csr.nrows,
        csr.ncols,
    ))
}

#[pyfunction]
pub(crate) fn coond_mode_to_csc_from_parts<'py>(
    py: Python<'py>,
    shape: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    axis: usize,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let csc = py.detach(|| coond_mode_to_csc_f64_i64(&a, axis));
    Ok((
        PyArray1::from_vec(py, csc.indptr),
        PyArray1::from_vec(py, csc.indices),
        PyArray1::from_vec(py, csc.data),
        csc.nrows,
        csc.ncols,
    ))
}

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
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let row_axes_i64 = row_axes.as_slice()?.to_vec();
    let mut row_axes_us: Vec<usize> = Vec::with_capacity(row_axes_i64.len());
    for &ax in &row_axes_i64 {
        if ax < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "row_axes must be non-negative",
            ));
        }
        row_axes_us.push(usize::try_from(ax).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("row_axes value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let csr = py.detach(|| coond_axes_to_csr_f64_i64(&a, &row_axes_us));
    Ok((
        PyArray1::from_vec(py, csr.indptr),
        PyArray1::from_vec(py, csr.indices),
        PyArray1::from_vec(py, csr.data),
        csr.nrows,
        csr.ncols,
    ))
}

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
    let shape_i64 = shape.as_slice()?.to_vec();
    let mut shape_us: Vec<usize> = Vec::with_capacity(shape_i64.len());
    for &s in &shape_i64 {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape must be non-negative",
            ));
        }
        shape_us.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("shape value overflow")
        })?);
    }
    let row_axes_i64 = row_axes.as_slice()?.to_vec();
    let mut row_axes_us: Vec<usize> = Vec::with_capacity(row_axes_i64.len());
    for &ax in &row_axes_i64 {
        if ax < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "row_axes must be non-negative",
            ));
        }
        row_axes_us.push(usize::try_from(ax).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("row_axes value overflow")
        })?);
    }
    let a = CooNd::from_parts(
        shape_us,
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let csc = py.detach(|| coond_axes_to_csc_f64_i64(&a, &row_axes_us));
    Ok((
        PyArray1::from_vec(py, csc.indptr),
        PyArray1::from_vec(py, csc.indices),
        PyArray1::from_vec(py, csc.data),
        csc.nrows,
        csc.ncols,
    ))
}

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
    let csc = Csc::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let t = py.detach(|| transpose_csc_f64_i64(&csc));
    Ok((
        PyArray1::from_vec(py, t.indptr),
        PyArray1::from_vec(py, t.indices),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

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
    Bound<'py, PyArray1<i64>>, // row
    Bound<'py, PyArray1<i64>>, // col
    Bound<'py, PyArray1<f64>>, // data
    usize,
    usize,
)> {
    let coo = Coo::from_parts(
        nrows,
        ncols,
        row.as_slice()?.to_vec(),
        col.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let t = py.detach(|| transpose_coo_f64_i64(&coo));
    Ok((
        PyArray1::from_vec(py, t.row),
        PyArray1::from_vec(py, t.col),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let xv: Vec<f64> = x.as_slice()?.to_vec();
    let y = py.detach(|| spmv_f64_i64(&csr, &xv));
    Ok(PyArray1::from_vec(py, y))
}

#[pyfunction]
pub(crate) fn spmm_from_parts<'py>(
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "B must be 2D",
        ));
    }
    let k = shape[1];
    if shape[0] != ncols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "B shape[0] must equal ncols",
        ));
    }
    let b_vec: Vec<f64> = b_arr.iter().copied().collect();
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let y = py.detach(|| spmm_f64_i64(&csr, &b_vec, k));

    let arr = Array2::from_shape_vec((nrows, k), y)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Output shape mismatch"))?;
    Ok(arr.into_pyarray(py))
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let out = py.detach(|| sum_f64(&csr));
    Ok(out)
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let out = py.detach(|| row_sums_f64(&csr));
    Ok(PyArray1::from_vec(py, out))
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let out = py.detach(|| col_sums_f64(&csr));
    Ok(PyArray1::from_vec(py, out))
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let t = py.detach(|| transpose_f64_i64(&csr));
    Ok((
        PyArray1::from_vec(py, t.indptr),
        PyArray1::from_vec(py, t.indices),
        PyArray1::from_vec(py, t.data),
        t.nrows,
        t.ncols,
    ))
}

#[pyfunction]
pub(crate) fn prune_from_parts<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    eps: f64,
    check: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
)> {
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let p = py.detach(|| prune_eps(&csr, eps));
    Ok((
        PyArray1::from_vec(py, p.indptr),
        PyArray1::from_vec(py, p.indices),
        PyArray1::from_vec(py, p.data),
        p.nrows,
        p.ncols,
    ))
}

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
    let csr = Csr::from_parts(
        nrows,
        ncols,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
        check,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let p = py.detach(|| eliminate_zeros(&csr));
    Ok((
        PyArray1::from_vec(py, p.indptr),
        PyArray1::from_vec(py, p.indices),
        PyArray1::from_vec(py, p.data),
        p.nrows,
        p.ncols,
    ))
}

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
    let c = py.detach(|| mul_scalar_f64(&a, alpha));
    Ok((
        PyArray1::from_vec(py, c.indptr),
        PyArray1::from_vec(py, c.indices),
        PyArray1::from_vec(py, c.data),
        c.nrows,
        c.ncols,
    ))
}
