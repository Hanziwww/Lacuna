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

pub mod arith;
pub mod cleanup;
pub mod reduce;
pub mod spmm;
pub mod spmv;
pub mod transform;

pub use arith::{add_csr_f64_i64, hadamard_csr_f64_i64, mul_scalar_f64, sub_csr_f64_i64};
pub use cleanup::{eliminate_zeros, prune_eps};
pub use reduce::{col_sums_f64, row_sums_f64, sum_f64};
pub use spmm::spmm_f64_i64;
pub use spmv::spmv_f64_i64;
pub use transform::transpose_f64_i64;
