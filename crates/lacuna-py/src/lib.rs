#![allow(
    clippy::type_complexity,
    reason = "pyo3 functions often return tuples of arrays and sizes"
)]
#![allow(
    clippy::too_many_arguments,
    reason = "Python-exposed constructors/functions map directly to multiple array arguments"
)]
#![allow(
    clippy::needless_pass_by_value,
    reason = "PyReadonlyArray types are thin wrappers passed by value in pyo3 idioms"
)]
#![allow(
    clippy::redundant_closure,
    reason = "map_err closures are acceptable in pyo3 error conversions and simplify readability"
)]
#![allow(
    clippy::missing_const_for_fn,
    reason = "pyo3 #[pymethods] are not const-compatible"
)]
#![allow(
    clippy::unnecessary_wraps,
    reason = "PyO3 methods conventionally return PyResult for Python-facing APIs"
)]
#![allow(
    clippy::elidable_lifetime_names,
    reason = "Explicit 'py lifetimes are idiomatic and clear in PyO3 method signatures"
)]
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use lacuna_core::Csr;
use lacuna_kernels::{
    add_csr_f64_i64, col_sums_f64, eliminate_zeros, hadamard_csr_f64_i64, mul_scalar_f64,
    prune_eps, row_sums_f64, spmm_f64_i64, spmv_f64_i64, sub_csr_f64_i64, sum_f64,
    transpose_f64_i64,
};

#[pyclass]
struct Csr64 {
    inner: Csr<f64, i64>,
}

#[pymethods]
impl Csr64 {
    #[new]
    fn new(
        nrows: usize,
        ncols: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
        check: bool,
    ) -> PyResult<Self> {
        let csr = Csr::from_parts(
            nrows,
            ncols,
            indptr.as_slice()?.to_vec(),
            indices.as_slice()?.to_vec(),
            data.as_slice()?.to_vec(),
            check,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self { inner: csr })
    }

    fn shape(&self) -> (usize, usize) {
        (self.inner.nrows, self.inner.ncols)
    }

    fn spmv<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xv: Vec<f64> = x.as_slice()?.to_vec();
        let y = py.detach(|| spmv_f64_i64(&self.inner, &xv));
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
        let y = py.detach(|| spmm_f64_i64(&self.inner, &b_vec, k));

        // Build ndarray then convert to PyArray2
        let arr = Array2::from_shape_vec((self.inner.nrows, k), y).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Output shape mismatch")
        })?;
        Ok(arr.into_pyarray(py))
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<f64> {
        let out = py.detach(|| sum_f64(&self.inner));
        Ok(out)
    }

    fn row_sums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out = py.detach(|| row_sums_f64(&self.inner));
        Ok(PyArray1::from_vec(py, out))
    }

    fn col_sums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out = py.detach(|| col_sums_f64(&self.inner));
        Ok(PyArray1::from_vec(py, out))
    }

    fn transpose<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let t = py.detach(|| transpose_f64_i64(&self.inner));
        Ok((
            PyArray1::from_vec(py, t.indptr),
            PyArray1::from_vec(py, t.indices),
            PyArray1::from_vec(py, t.data),
            t.nrows,
            t.ncols,
        ))
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
        let p = py.detach(|| prune_eps(&self.inner, eps));
        Ok((
            PyArray1::from_vec(py, p.indptr),
            PyArray1::from_vec(py, p.indices),
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
        let p = py.detach(|| eliminate_zeros(&self.inner));
        Ok((
            PyArray1::from_vec(py, p.indptr),
            PyArray1::from_vec(py, p.indices),
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
        let c = py.detach(|| mul_scalar_f64(&self.inner, alpha));
        Ok((
            PyArray1::from_vec(py, c.indptr),
            PyArray1::from_vec(py, c.indices),
            PyArray1::from_vec(py, c.data),
            c.nrows,
            c.ncols,
        ))
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        other: &Self,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let c = py.detach(|| add_csr_f64_i64(&self.inner, &other.inner));
        Ok((
            PyArray1::from_vec(py, c.indptr),
            PyArray1::from_vec(py, c.indices),
            PyArray1::from_vec(py, c.data),
            c.nrows,
            c.ncols,
        ))
    }

    fn sub<'py>(
        &self,
        py: Python<'py>,
        other: &Self,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let c = py.detach(|| sub_csr_f64_i64(&self.inner, &other.inner));
        Ok((
            PyArray1::from_vec(py, c.indptr),
            PyArray1::from_vec(py, c.indices),
            PyArray1::from_vec(py, c.data),
            c.nrows,
            c.ncols,
        ))
    }

    fn hadamard<'py>(
        &self,
        py: Python<'py>,
        other: &Self,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        usize,
        usize,
    )> {
        let c = py.detach(|| hadamard_csr_f64_i64(&self.inner, &other.inner));
        Ok((
            PyArray1::from_vec(py, c.indptr),
            PyArray1::from_vec(py, c.indices),
            PyArray1::from_vec(py, c.data),
            c.nrows,
            c.ncols,
        ))
    }
}

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
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let xv: Vec<f64> = x.as_slice()?.to_vec();
    let y = py.detach(|| spmv_f64_i64(&csr, &xv));
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
fn transpose_from_parts<'py>(
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
fn prune_from_parts<'py>(
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
fn eliminate_zeros_from_parts<'py>(
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
fn sub_from_parts<'py>(
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
fn hadamard_from_parts<'py>(
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
fn mul_scalar_from_parts<'py>(
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

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Csr64>()?;
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
    m.add_function(wrap_pyfunction!(sub_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(hadamard_from_parts, m)?)?;
    Ok(())
}
