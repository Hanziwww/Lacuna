//! Elementwise sign for sparse formats (CSR/CSC/COO/COOND).
//! Parallelized; vectorization omitted as branching dominates.

#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::comparison_chain
)]

use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;

#[inline]
fn sign_f64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// sign(A) for CSR
#[must_use]
pub fn sign_scalar_f64(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = sign_f64(*v);
        }
    });
    Csr::from_parts_unchecked(a.nrows, a.ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// sign(A) for CSC
#[must_use]
pub fn sign_scalar_csc_f64(a: &Csc<f64, i64>) -> Csc<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = sign_f64(*v);
        }
    });
    Csc::from_parts_unchecked(a.nrows, a.ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// sign(A) for COO
#[must_use]
pub fn sign_scalar_coo_f64(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = sign_f64(*v);
        }
    });
    Coo::from_parts_unchecked(a.nrows, a.ncols, a.row.clone(), a.col.clone(), data)
}

/// sign(X) for COOND
#[must_use]
pub fn sign_scalar_coond_f64(a: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = sign_f64(*v);
        }
    });
    CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data)
}
