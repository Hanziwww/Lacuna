//! Optimized kernels for Lacuna (pure Rust, SIMD/parallel ready)
#![allow(
    clippy::blanket_clippy_restriction_lints,
    reason = "User requested enabling the restriction group; allow the blanket lint to use a curated subset"
)]
#![allow(
    clippy::missing_panics_doc,
    reason = "Performance kernels may assert on invalid input; full docs are provided at the Python layer"
)]

pub const fn init_parallel() {
    // Rayon auto-detects threads by default; users may set RAYON_NUM_THREADS.
}

// New Array API-aligned module structure
pub mod data_type_functions;
pub mod elementwise;
pub mod linalg;
pub mod manipulation;
pub mod statistical;
pub mod utility;

// Placeholder modules (empty for now)
pub mod indexing;
pub mod search_sort;
pub mod setops;

// Re-export all public functions from new structure
pub use elementwise::add::{add_coond_f64_i64, add_csc_f64_i64, add_csr_f64_i64};
pub use elementwise::multiply::{
    hadamard_broadcast_coond_f64_i64, hadamard_coond_f64_i64, hadamard_csc_f64_i64,
    hadamard_csr_f64_i64, mul_scalar_coo_f64, mul_scalar_coond_f64, mul_scalar_csc_f64,
    mul_scalar_f64,
};
pub use elementwise::subtract::{sub_coond_f64_i64, sub_csc_f64_i64, sub_csr_f64_i64};

pub use linalg::matmul::{
    spmm_auto_f64_i64, spmm_coo_f64_i64, spmm_coond_f64_i64, spmm_csc_f64_i64, spmm_f64_i64,
    spmv_coo_f64_i64, spmv_coond_f64_i64, spmv_csc_f64_i64, spmv_f64_i64,
};
pub use linalg::matrix_transpose::{
    transpose_coo_f64_i64, transpose_csc_f64_i64, transpose_f64_i64,
};

pub use linalg::tensordot::{
    tensordot_coo_dense_axes1x0_f64_i64, tensordot_coond_dense_axis_f64_i64,
    tensordot_csc_dense_axes1x0_f64_i64, tensordot_csr_dense_axes1x0_f64_i64,
};
pub use linalg::vecdot::{
    vecdot_coo_dense_axis0_f64_i64, vecdot_coo_dense_axis1_f64_i64,
    vecdot_coond_dense_axis_f64_i64, vecdot_csc_dense_axis0_f64_i64,
    vecdot_csc_dense_axis1_f64_i64, vecdot_csr_dense_axis0_f64_i64, vecdot_csr_dense_axis1_f64_i64,
};

pub use statistical::mean::{mean_coond_f64, reduce_mean_axes_coond_f64_i64};
pub use statistical::sum::{
    col_sums_coo_f64, col_sums_csc_f64, col_sums_f64, reduce_sum_axes_coond_f64_i64,
    row_sums_coo_f64, row_sums_csc_f64, row_sums_f64, sum_coo_f64, sum_coond_f64, sum_csc_f64,
    sum_f64,
};

pub use statistical::minmax::{
    col_maxs_coo_f64, col_maxs_csc_f64, col_maxs_f64, col_mins_coo_f64, col_mins_csc_f64,
    col_mins_f64, max_coo_f64, max_csc_f64, max_f64, min_coo_f64, min_csc_f64, min_f64,
    row_maxs_coo_f64, row_maxs_csc_f64, row_maxs_f64, row_mins_coo_f64, row_mins_csc_f64,
    row_mins_f64,
};

pub use statistical::prod::{
    col_prods_coo_f64, col_prods_csc_f64, col_prods_f64, prod_coo_f64, prod_coond_f64,
    prod_csc_f64, prod_f64, row_prods_coo_f64, row_prods_csc_f64, row_prods_f64,
};

// Cumulative (dense outputs)
pub use statistical::cumulative::{
    csr_cumprod_dense_axis0_f64, csr_cumprod_dense_axis1_f64, csr_cumsum_dense_axis0_f64,
    csr_cumsum_dense_axis1_f64,
};

pub use statistical::varstd::{
    col_stds_coo_f64, col_stds_csc_f64, col_stds_f64, col_vars_coo_f64, col_vars_csc_f64,
    col_vars_f64, row_stds_coo_f64, row_stds_csc_f64, row_stds_f64, row_vars_coo_f64,
    row_vars_csc_f64, row_vars_f64, std_coo_f64, std_coond_f64, std_csc_f64, std_f64, var_coo_f64,
    var_coond_f64, var_csc_f64, var_f64,
};

pub use manipulation::diff::{
    diff_coo_axis0_f64_i64, diff_coo_axis1_f64_i64, diff_csc_axis0_f64_i64, diff_csc_axis1_f64_i64,
    diff_csr_axis0_f64_i64, diff_csr_axis1_f64_i64,
};
pub use manipulation::permute_dims::permute_axes_coond_f64_i64;
pub use manipulation::reshape::reshape_coond_f64_i64;

pub use data_type_functions::astype::{
    coo_to_csc_f64_i64, coo_to_csr_f64_i64, coond_axes_to_csc_f64_i64, coond_axes_to_csr_f64_i64,
    coond_mode_to_csc_f64_i64, coond_mode_to_csr_f64_i64, csc_to_coo_f64_i64, csc_to_csr_f64_i64,
    csr_to_coo_f64_i64, csr_to_csc_f64_i64,
};

// Logical reductions: all/any
pub use statistical::logic::{
    all_coo_f64, all_coond_f64, all_csc_f64, all_f64, any_coo_f64, any_coond_f64, any_csc_f64,
    any_f64, col_alls_csc_f64, col_alls_f64, col_anys_csc_f64, col_anys_f64, row_alls_csc_f64,
    row_alls_f64, row_anys_csc_f64, row_anys_f64,
};

pub use utility::eliminate_zeros::{
    eliminate_zeros, eliminate_zeros_coo, eliminate_zeros_coond, eliminate_zeros_csc,
};
pub use utility::prune::{prune_eps, prune_eps_coo, prune_eps_coond, prune_eps_csc};
