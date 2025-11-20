//! Sparse array type definitions
//!
//! This module contains the `#[pyclass]` definitions for sparse array types
//! exposed to Python. Each type is a thin wrapper around the corresponding
//! `lacuna_core` type with `PyO3` bindings.
//!
//! The types provide:
//! - Construction from parts (indptr, indices, data)
//! - Property access (shape, nnz, dtype, etc.)
//! - Methods that delegate to `array_api` module functions
//!
//! ## Design Philosophy
//! - **Separation of Concerns**: Type definitions are separate from operations
//! - **Thin Wrappers**: Minimal logic here, operations in `array_api` modules
//! - **Consistent API**: All types follow the same pattern for construction and access

pub mod coo;
pub mod csc;
pub mod csr;

// Re-export the main types for convenient access
pub use coo::Coo64;
pub use csc::Csc64;
pub use csr::Csr64;
