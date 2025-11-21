//! Logical reductions: all/any for CSR/CSC/COO and global COOND

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

use crate::data_type_functions::astype::coo_to_csr_f64_i64;
use crate::linalg::matrix_transpose::transpose_f64_i64;
use crate::utility::util::i64_to_usize;
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

// ===== CSR =====

#[must_use]
pub fn all_f64(a: &Csr<f64, i64>) -> bool {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    // all non-zeros must be truthy (!= 0.0). NaN and infinities are non-zero => true.
    !a.data.par_iter().any(|&x| x == 0.0)
}

#[must_use]
pub fn any_f64(a: &Csr<f64, i64>) -> bool {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}

#[must_use]
pub fn row_alls_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    if ncols == 0 {
        // all over empty axis -> true
        return vec![true; nrows];
    }
    let mut out = vec![false; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        if (e - s) < ncols {
            *oi = false;
        } else {
            let row = &a.data[s..e];
            *oi = !row.iter().any(|&x| x == 0.0);
        }
    });
    out
}

#[must_use]
pub fn row_anys_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    if ncols == 0 {
        // any over empty axis -> false
        return vec![false; nrows];
    }
    let mut out = vec![false; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let row = &a.data[s..e];
        *oi = row.iter().any(|&x| x != 0.0);
    });
    out
}

#[must_use]
pub fn col_alls_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    let t = transpose_f64_i64(a);
    // row-wise all of transpose equals column-wise all of original
    row_alls_f64(&t)
}

#[must_use]
pub fn col_anys_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    let t = transpose_f64_i64(a);
    row_anys_f64(&t)
}

// ===== CSC =====

#[must_use]
pub fn all_csc_f64(a: &Csc<f64, i64>) -> bool {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    !a.data.par_iter().any(|&x| x == 0.0)
}

#[must_use]
pub fn any_csc_f64(a: &Csc<f64, i64>) -> bool {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}

#[must_use]
pub fn row_alls_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    // Convert to CSR and reuse CSR row-wise implementation
    let csr = crate::data_type_functions::astype::csc_to_csr_f64_i64(a);
    row_alls_f64(&csr)
}

#[must_use]
pub fn row_anys_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    let csr = crate::data_type_functions::astype::csc_to_csr_f64_i64(a);
    row_anys_f64(&csr)
}

#[must_use]
pub fn col_alls_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    if nrows == 0 {
        return vec![true; ncols];
    }
    let mut out = vec![false; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        if (e - s) < nrows {
            *oj = false;
        } else {
            let col = &a.data[s..e];
            *oj = !col.iter().any(|&x| x == 0.0);
        }
    });
    out
}

#[must_use]
pub fn col_anys_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    if nrows == 0 {
        return vec![false; ncols];
    }
    let mut out = vec![false; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let col = &a.data[s..e];
        *oj = col.iter().any(|&x| x != 0.0);
    });
    out
}

// ===== COO (convert and reuse CSR/CSC logic) =====

#[must_use]
pub fn all_coo_f64(a: &Coo<f64, i64>) -> bool {
    let csr = coo_to_csr_f64_i64(a);
    all_f64(&csr)
}

#[must_use]
pub fn any_coo_f64(a: &Coo<f64, i64>) -> bool {
    let csr = coo_to_csr_f64_i64(a);
    any_f64(&csr)
}

// For axis-wise operations, call through conversions from Python to reuse CSR/CSC kernels.

// ===== COOND (global only for now) =====

#[must_use]
pub fn all_coond_f64(a: &CooNd<f64, i64>) -> bool {
    let full = product_checked(&a.shape);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    !a.data.par_iter().any(|&x| x == 0.0)
}

#[must_use]
pub fn any_coond_f64(a: &CooNd<f64, i64>) -> bool {
    let full = product_checked(&a.shape);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}
