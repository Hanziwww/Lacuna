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

// Sparse array type definitions
mod types;

// Array API aligned module structure
pub mod array_api;

#[allow(
    clippy::too_many_lines,
    reason = "PyO3 module registration is naturally verbose"
)]
#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    // Module metadata
    m.add("version", env!("CARGO_PKG_VERSION"))?;

    // Sparse array types
    m.add_class::<crate::types::Csr64>()?;
    m.add_class::<crate::types::Csc64>()?;
    m.add_class::<crate::types::Coo64>()?;

    // ===== Linear Algebra =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::spmv_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::spmm_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::transpose_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::transpose_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::transpose_coo_from_parts,
        m
    )?)?;

    // Linalg: tensordot (sparse × dense)
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::tensordot_csr_dense_axes1x0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::tensordot_csc_dense_axes1x0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::tensordot_coo_dense_axes1x0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::coond_tensordot_dense_axis_from_parts,
        m
    )?)?;

    // Linalg: vecdot (sparse × dense)
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::vecdot_csr_axis0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::vecdot_csc_axis0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::vecdot_coo_axis0_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::linalg::coond_vecdot_axis_from_parts,
        m
    )?)?;

    // ===== Statistical Reductions =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::sum_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_sums_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_sums_from_parts,
        m
    )?)?;
    // Min/Max (CSR)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::min_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::max_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_mins_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_maxs_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_mins_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_maxs_from_parts,
        m
    )?)?;
    // Min/Max (CSC)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::min_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::max_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_mins_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_maxs_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_mins_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_maxs_csc_from_parts,
        m
    )?)?;
    // Min/Max (COO)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::min_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::max_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_mins_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_maxs_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_mins_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_maxs_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_sum_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_mean_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_reduce_sum_axes_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_reduce_mean_axes_from_parts,
        m
    )?)?;

    // Logical reductions: all / any (CSR)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::all_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::any_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_alls_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_alls_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_anys_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_anys_from_parts,
        m
    )?)?;
    // Logical reductions: all / any (CSC)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::all_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::any_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_alls_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_alls_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_anys_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_anys_csc_from_parts,
        m
    )?)?;
    // Logical reductions: all / any (COO & COOND)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::all_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::any_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_alls_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_alls_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_anys_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_anys_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_all_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_any_from_parts,
        m
    )?)?;

    // Product (CSR/CSC/COO/COOND)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::prod_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_prods_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_prods_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::prod_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_prods_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_prods_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::prod_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_prods_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_prods_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_prod_from_parts,
        m
    )?)?;

    // Cumulative (dense ndarray output)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumsum_from_parts_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumsum_csc_from_parts_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumsum_coo_from_parts_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumprod_from_parts_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumprod_csc_from_parts_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::cumprod_coo_from_parts_dense,
        m
    )?)?;

    // Variance / Standard Deviation (CSR/CSC/COO/COOND)
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::var_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::std_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_vars_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_stds_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_vars_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_stds_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::var_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::std_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_vars_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_stds_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_vars_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_stds_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::var_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::std_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_vars_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::row_stds_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_vars_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::col_stds_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_var_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::reduce::coond_std_from_parts,
        m
    )?)?;

    // ===== Element-wise Operations =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::add_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::sub_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::div_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::div_csc_from_parts,
        m
    )?)?;
    // power (pairwise CSR/CSC)
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_csc_from_parts,
        m
    )?)?;
    // absolute value (unary)
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::abs_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::abs_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::abs_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::abs_coond_from_parts,
        m
    )?)?;
    // sign (unary)
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::sign_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::sign_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::sign_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::sign_coond_from_parts,
        m
    )?)?;
    // remainder (pairwise CSR/CSC)
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_csc_from_parts,
        m
    )?)?;
    // floor_divide
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_csc_from_parts,
        m
    )?)?;
    // scalar floor divide variants
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_scalar_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_scalar_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_scalar_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::floordiv_scalar_coond_from_parts,
        m
    )?)?;
    // power scalar variants
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_scalar_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_scalar_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_scalar_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::pow_scalar_coond_from_parts,
        m
    )?)?;
    // remainder scalar variants
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_scalar_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_scalar_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_scalar_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::remainder_scalar_coond_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::hadamard_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::mul_scalar_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::mul_scalar_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::mul_scalar_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::mul_scalar_coond_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::elementwise::coond_hadamard_broadcast_from_parts,
        m
    )?)?;

    // ===== Manipulation =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::manipulation::coond_permute_axes_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::manipulation::coond_reshape_from_parts,
        m
    )?)?;
    // diff (CSR/CSC/COO)
    m.add_function(wrap_pyfunction!(
        crate::array_api::manipulation::diff_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::manipulation::diff_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::manipulation::diff_coo_from_parts,
        m
    )?)?;

    // ===== Utilities: Format Conversion & Cleanup =====
    // Format conversions
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::csr_to_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::csc_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::csr_to_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::csc_to_coo_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coo_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coo_to_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coond_mode_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coond_mode_to_csc_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coond_axes_to_csr_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::coond_axes_to_csc_from_parts,
        m
    )?)?;
    // Cleanup
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::prune_from_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::utility::eliminate_zeros_from_parts,
        m
    )?)?;

    // ===== Array Creation =====
    m.add_function(wrap_pyfunction!(crate::array_api::creation::zeros_csr, m)?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::creation::zeros_csc, m)?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::creation::zeros_coo, m)?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::creation::eye_csr, m)?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::creation::diag_csr, m)?)?;

    // ===== Data Types =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::dtypes::get_default_float_dtype,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::dtypes::get_default_int_dtype,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::dtypes::is_float_dtype,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::dtypes::is_int_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::dtypes::dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::dtypes::promote_dtypes,
        m
    )?)?;

    // ===== Devices =====
    m.add_function(wrap_pyfunction!(
        crate::array_api::devices::get_default_device,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::devices::is_valid_device,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::array_api::devices::list_devices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::array_api::devices::device_info, m)?)?;

    Ok(())
}
