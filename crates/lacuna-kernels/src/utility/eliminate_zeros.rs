//! Eliminate explicit zeros (utility function, not standard Array API)

use crate::utility::prune::{prune_eps, prune_eps_coo, prune_eps_coond, prune_eps_csc};
use lacuna_core::{Coo, CooNd, Csc, Csr};

#[must_use]
pub fn eliminate_zeros(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    prune_eps(a, 0.0)
}

#[must_use]
pub fn eliminate_zeros_coo(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    prune_eps_coo(a, 0.0)
}

#[must_use]
pub fn eliminate_zeros_coond(a: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    prune_eps_coond(a, 0.0)
}

#[must_use]
pub fn eliminate_zeros_csc(a: &Csc<f64, i64>) -> Csc<f64, i64> {
    prune_eps_csc(a, 0.0)
}
