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
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

mod coo;
mod csc;
mod csr;
mod functions;

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<crate::csr::Csr64>()?;
    m.add_class::<crate::csc::Csc64>()?;
    m.add_class::<crate::coo::Coo64>()?;
    m.add_function(wrap_pyfunction!(crate::functions::spmv_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functions::spmm_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functions::sum_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functions::row_sums_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functions::col_sums_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coo_to_csc_from_parts,
        m
    )?)?;
    // ND COO -> CSR/CSC conversions
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_mode_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_mode_to_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_axes_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_axes_to_csc_from_parts,
        m
    )?)?;
    // ND COO ops
    m.add_function(wrap_pyfunction!(crate::functions::coond_sum_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_mean_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_reduce_sum_axes_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_reduce_mean_axes_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_permute_axes_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_reshape_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coond_hadamard_broadcast_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::functions::transpose_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::transpose_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::transpose_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::functions::prune_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::eliminate_zeros_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::functions::add_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::mul_scalar_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::functions::sub_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functions::hadamard_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::csr_to_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::csc_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::csr_to_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::csc_to_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coo_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::functions::coo_to_csc_from_parts,
        m
    )?)?;
    Ok(())
}
