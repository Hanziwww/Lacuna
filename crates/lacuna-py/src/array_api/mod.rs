//! Array API aligned bindings module structure
//!
//! This module organizes `PyO3` bindings according to the Python Array API Standard categories:
//! - linalg: Linear algebra operations (matmul, transpose, etc.)
//! - reduce: Statistical reductions (sum, mean, etc.)
//! - elementwise: Element-wise operations (add, mul, etc.)
//! - manipulation: Array manipulation (reshape, permute, etc.)
//! - utility: Format conversions and cleanup utilities
//! - dtypes: Data type operations (placeholders for future)
//! - devices: Device management (placeholders for future)
//! - creation: Array creation functions (placeholders for future)
//! - indexing: Indexing operations (placeholders for future)
//! - `search_sort_set`: Search, sort, and set operations (placeholders for future)

// Common helpers (internal use only)
pub mod helpers;

// Core modules implementing Array API aligned functions
pub mod elementwise;
pub mod linalg;
pub mod manipulation;
pub mod reduce;
pub mod utility;

// Placeholder modules for future implementation
pub mod creation;
pub mod devices;
pub mod dtypes;
pub mod indexing;
pub mod search_sort_set;

// Note: Functions are accessed directly via their module paths in lib.rs
// e.g., crate::array_api::linalg::spmv_from_parts
// No re-exports needed here since all functions are exposed via their modules
