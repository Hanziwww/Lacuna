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

pub use statistical::mean::{mean_coond_f64, reduce_mean_axes_coond_f64_i64};
pub use statistical::sum::{
    col_sums_coo_f64, col_sums_csc_f64, col_sums_f64, reduce_sum_axes_coond_f64_i64,
    row_sums_coo_f64, row_sums_csc_f64, row_sums_f64, sum_coo_f64, sum_coond_f64, sum_csc_f64,
    sum_f64,
};

pub use manipulation::permute_dims::permute_axes_coond_f64_i64;
pub use manipulation::reshape::reshape_coond_f64_i64;

pub use data_type_functions::astype::{
    coo_to_csc_f64_i64, coo_to_csr_f64_i64, coond_axes_to_csc_f64_i64, coond_axes_to_csr_f64_i64,
    coond_mode_to_csc_f64_i64, coond_mode_to_csr_f64_i64, csc_to_coo_f64_i64, csc_to_csr_f64_i64,
    csr_to_coo_f64_i64, csr_to_csc_f64_i64,
};

pub use utility::eliminate_zeros::{
    eliminate_zeros, eliminate_zeros_coo, eliminate_zeros_coond, eliminate_zeros_csc,
};
pub use utility::prune::{prune_eps, prune_eps_coo, prune_eps_coond, prune_eps_csc};
