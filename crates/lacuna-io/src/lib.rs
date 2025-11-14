//! IO helpers for Lacuna
#![allow(
    clippy::implicit_return,
    reason = "Prefer expression style; matches project code style under restriction lints"
)]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    reason = "User requested enabling the restriction group; allow the blanket lint to use a curated subset"
)]

#[inline]
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
