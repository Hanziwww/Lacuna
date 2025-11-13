//! Optimized kernels for Lacuna (pure Rust, SIMD/parallel ready)

pub fn init_parallel() {
    // Rayon auto-detects threads by default; users may set RAYON_NUM_THREADS.
}

pub mod arith;
pub mod spmv;
pub mod spmm;
pub mod reduce;
pub mod cleanup;
pub mod transform;

pub use arith::{mul_scalar_f64, add_csr_f64_i64};
pub use spmv::spmv_f64_i64;
pub use spmm::spmm_f64_i64;
pub use reduce::{sum_f64, row_sums_f64, col_sums_f64};
pub use cleanup::{eliminate_zeros, prune_eps};
pub use transform::transpose_f64_i64;
