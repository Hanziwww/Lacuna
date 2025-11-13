//! Core data structures and traits for Lacuna (pure Rust)

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod csr;
pub use csr::Csr;
