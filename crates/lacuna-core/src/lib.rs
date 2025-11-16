//! Core data structures and traits for Lacuna (pure Rust)
#![allow(
    clippy::implicit_return,
    reason = "Prefer expression style; avoids conflict with needless-return under restriction set"
)]
#![allow(
    clippy::missing_errors_doc,
    reason = "Docs are provided in the Python layer; keeping Rust core lean for now"
)]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    reason = "User requested enabling the restriction group; allow the blanket lint to use a curated subset"
)]

pub mod coo;
pub mod csc;
pub mod csr;
pub mod nd;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub type Csr<T, I> = csr::Csr<T, I>;
pub type Csc<T, I> = csc::Csc<T, I>;
pub type Coo<T, I> = coo::Coo<T, I>;
pub type CooNd<T, I> = coo::CooNd<T, I>;
