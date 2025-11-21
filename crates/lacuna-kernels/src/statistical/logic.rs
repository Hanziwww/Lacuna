//! Logical reductions: all/any for CSR/CSC/COO and global COOND
//
// This module implements logical reduction operations (all / any) for sparse matrix
// data structures used throughout the project. We provide:
// - scalar reductions that answer whether all/any elements are truthy for whole matrices
// - axis-wise reductions that return row-wise or column-wise boolean vectors
// - conversions for COO and CSC to reuse CSR kernels where appropriate
//
// The functions assume sparse data uses 0.0 to represent logical False and any
// non-zero (including NaN/Inf) represents logical True. All operations preserve
// the semantics of missing elements (implicit zeros) in the sparse formats.

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
/// Compute product of dimensions with overflow check.
///
/// This is used by ND variants to detect if the overall volume of the tensor
/// overflows an usize when we try to compute reductions that require the full number
/// of elements (e.g., to check whether the stored data length matches the full
/// dense size). Panics on overflow with a helpful message.
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

// ===== CSR =====

/// Return true when all elements in the CSR matrix are `truthy` (non-zero).
///
/// Behavior:
/// - Empty matrices (zero total size) return `true` by convention.
/// - If the number of explicitly stored values is less than the full dense size,
///   some entries are implicitly zero and the function returns `false`.
/// - NaN or infinity values are treated as non-zero and therefore `truthy`.
#[must_use]
pub fn all_f64(a: &Csr<f64, i64>) -> bool {
    // Compute full dense size (nrows * ncols) while saturating to avoid overflow
    // in pathological inputs. If the full size is 0 (empty matrix) return true
    // as per convention that all over an empty collection is true.
    let full = a.nrows.saturating_mul(a.ncols);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    // If the number of stored values is less than the full dense size, some
    // elements are implicitly zero and thus `all` must return false.
    // Otherwise, all stored elements must be non-zero. We iterate in parallel
    // and return false if any value equals 0.0. Note: NaN/Inf are considered
    // non-zero and therefore truthy for the purposes of `all`.
    !a.data.par_iter().any(|&x| x == 0.0)
}

/// Return true when any element in the CSR matrix is `truthy` (non-zero).
///
/// Behavior:
/// - Empty matrices (zero total size) return `false`.
/// - The function inspects only explicitly stored values; if any stored
///   value is non-zero, the result is `true`.
#[must_use]
pub fn any_f64(a: &Csr<f64, i64>) -> bool {
    // Any over empty matrix -> false
    let full = a.nrows.saturating_mul(a.ncols);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}

/// Compute row-wise `all` reduction for a CSR matrix.
///
/// Returns a boolean vector with length `nrows`. Each entry indicates whether
/// all values in that row are `truthy` (non-zero). Missing entries (implicit zeros)
/// mean the row is not all-true unless the stored entries explicitly cover every
/// column and none of the stored values equal 0.0.
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
        // If the number of stored entries in the row is less than `ncols`, some
        // entries are implicitly zero -> row cannot be all true.
        if (e - s) < ncols {
            *oi = false;
        } else {
            // If we have a complete set of explicit entries, check if any stored
            // value equals 0.0. If none are zero, the row is all true.
            let row = &a.data[s..e];
            *oi = !row.contains(&0.0);
        }
    });
    out
}

/// Compute row-wise `any` reduction for a CSR matrix.
///
/// Returns a boolean vector with length `nrows`. Each entry is true iff any
/// explicitly stored value in the row is non-zero. Missing implicit zero
/// entries do not cause `any` to be true.
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
        // Any is true if any stored element is non-zero; missing implicit zeros
        // do not change this result since they are zero.
        *oi = row.iter().any(|&x| x != 0.0);
    });
    out
}

/// Compute column-wise `all` reduction for a CSR matrix.
///
/// Implemented by transposing the matrix and performing `row_alls` on the
/// transposed representation. This avoids duplicating column-wise logic.
#[must_use]
pub fn col_alls_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    // Reuse row-wise logic by transposing the matrix: row_alls on transpose
    // corresponds to column-wise all on the original matrix.
    let t = transpose_f64_i64(a);
    row_alls_f64(&t)
}

/// Compute column-wise `any` reduction for a CSR matrix by transposing the
/// matrix and calling `row_anys` on the transposed matrix.
#[must_use]
pub fn col_anys_f64(a: &Csr<f64, i64>) -> Vec<bool> {
    // Similarly, reuse row_anys on the transposed matrix for column-wise any.
    let t = transpose_f64_i64(a);
    row_anys_f64(&t)
}

// ===== CSC =====

/// Return true when all elements in the CSC matrix are `truthy` (non-zero).
///
/// Behavior mirrors `all_f64` for CSR. For performance we directly inspect the
/// underlying storage for missing/explicit entries.
#[must_use]
pub fn all_csc_f64(a: &Csc<f64, i64>) -> bool {
    // Same rules as CSR: empty -> true, fewer stored entries than dense size => false
    let full = a.nrows.saturating_mul(a.ncols);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    !a.data.par_iter().any(|&x| x == 0.0)
}

/// Return true when any element in the CSC matrix is `truthy` (non-zero).
/// Mirrors `any_f64` for CSR.
#[must_use]
pub fn any_csc_f64(a: &Csc<f64, i64>) -> bool {
    let full = a.nrows.saturating_mul(a.ncols);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}

/// Compute row-wise `all` for CSC by converting to CSR and reusing `row_alls_f64`.
#[must_use]
pub fn row_alls_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    // Convert to CSR and reuse the CSR row-wise `row_alls_f64` implementation.
    // This avoids duplicating the row-wise logic for CSC storage format.
    let csr = crate::data_type_functions::astype::csc_to_csr_f64_i64(a);
    row_alls_f64(&csr)
}

/// Compute row-wise `any` for CSC by converting to CSR and reusing `row_anys_f64`.
#[must_use]
pub fn row_anys_csc_f64(a: &Csc<f64, i64>) -> Vec<bool> {
    let csr = crate::data_type_functions::astype::csc_to_csr_f64_i64(a);
    row_anys_f64(&csr)
}

/// Compute column-wise `all` directly on CSC storage.
///
/// This iterates each column slice and checks whether the column is fully
/// represented and contains no zero values.
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
        // For each column in CSC, check whether the number of stored values
        // equals the number of rows. If it's less, some implicit zeros exist
        // and the column is not all-true. Otherwise, check the explicit stored
        // elements for any zeros.
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        if (e - s) < nrows {
            *oj = false;
        } else {
            let col = &a.data[s..e];
            *oj = !col.contains(&0.0);
        }
    });
    out
}

/// Compute column-wise `any` directly on CSC storage.
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
        // Any over a column is true if any stored value is non-zero.
        // Implicit zeros due to missing stored entries do not affect the result
        // since they are equal to zero.
        *oj = col.iter().any(|&x| x != 0.0);
    });
    out
}

// ===== COO (convert and reuse CSR/CSC logic) =====

/// Return true when all elements in a COO matrix are `truthy` (non-zero).
///
/// This uses conversion to CSR and reuses the `all_f64` kernel.
#[must_use]
pub fn all_coo_f64(a: &Coo<f64, i64>) -> bool {
    // Convert COO to CSR and reuse the CSR `all_f64` logic.
    let csr = coo_to_csr_f64_i64(a);
    all_f64(&csr)
}

/// Return true when any element in a COO matrix is `truthy` (non-zero).
///
/// This uses conversion to CSR and reuses the `any_f64` kernel.
#[must_use]
pub fn any_coo_f64(a: &Coo<f64, i64>) -> bool {
    // Convert COO to CSR and reuse the CSR `any_f64` logic.
    let csr = coo_to_csr_f64_i64(a);
    any_f64(&csr)
}

// For axis-wise operations, call through conversions from Python to reuse CSR/CSC kernels.

// ===== COOND (global only for now) =====

/// Global `all` reduction for an N-D COO array.
///
/// Returns true when every element in the full dense shape is non-zero.
/// Uses `product_checked` to safely compute the total number of elements.
#[must_use]
pub fn all_coond_f64(a: &CooNd<f64, i64>) -> bool {
    // For ND COO arrays, compute the overall size and apply the same
    // rules as for 2D sparse matrices: empty -> true, fewer stored items -> false.
    let full = product_checked(&a.shape);
    if full == 0 {
        return true;
    }
    if a.data.len() < full {
        return false;
    }
    !a.data.par_iter().any(|&x| x == 0.0)
}

/// Global `any` reduction for an N-D COO array.
///
/// Returns true when any stored element is non-zero.
#[must_use]
pub fn any_coond_f64(a: &CooNd<f64, i64>) -> bool {
    let full = product_checked(&a.shape);
    if full == 0 {
        return false;
    }
    a.data.par_iter().any(|&x| x != 0.0)
}
