//! Discrete forward differences (n-th order) along a specified axis for sparse matrices.
//! Returns sparse matrices of the same format (CSR/CSC) and COO via conversion.

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

use crate::data_type_functions::astype::{csc_to_csr_f64_i64, csr_to_csc_f64_i64};
use crate::linalg::matrix_transpose::transpose_f64_i64;
use crate::utility::util::i64_to_usize;
use lacuna_core::{Coo, Csc, Csr};
use rayon::prelude::*;

#[inline]
fn combine_sorted_pairs(pairs: &mut Vec<(usize, f64)>) {
    if pairs.is_empty() {
        return;
    }
    pairs.sort_unstable_by_key(|x| x.0);
    let mut w = 0usize;
    let mut last_c = pairs[0].0;
    let mut acc = pairs[0].1;
    for k in 1..pairs.len() {
        let (c, v) = pairs[k];
        if c == last_c {
            acc += v;
        } else {
            if acc != 0.0 {
                pairs[w] = (last_c, acc);
                w += 1;
            }
            last_c = c;
            acc = v;
        }
    }
    if acc != 0.0 {
        pairs[w] = (last_c, acc);
        w += 1;
    }
    pairs.truncate(w);
}

#[inline]
fn diff_row_pairs(mut pairs: Vec<(usize, f64)>, width: usize, n: usize) -> Vec<(usize, f64)> {
    if n == 0 || width == 0 {
        pairs.retain(|(_c, v)| *v != 0.0);
        return pairs;
    }
    let mut cur = pairs;
    let mut cur_w = width;
    for _ in 0..n {
        if cur_w == 0 || cur_w == 1 {
            return Vec::new();
        }
        let mut next: Vec<(usize, f64)> = Vec::with_capacity(cur.len() * 2);
        let limit = cur_w - 1; // output columns are [0..limit-1]
        for &(c, v) in &cur {
            if c < limit {
                next.push((c, -v));
            }
            if c >= 1 {
                next.push((c - 1, v));
            }
        }
        combine_sorted_pairs(&mut next);
        cur = next;
        cur_w -= 1;
        if cur.is_empty() {
            break;
        }
    }
    cur
}

#[must_use]
pub fn diff_csr_axis1_f64_i64(a: &Csr<f64, i64>, n: usize) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let out_cols = ncols.saturating_sub(n);
    if nrows == 0 || out_cols == 0 {
        return Csr::from_parts_unchecked(
            nrows,
            out_cols,
            vec![0; nrows + 1],
            Vec::new(),
            Vec::new(),
        );
    }
    // First pass: count nnz per row
    let counts: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut row_pairs: Vec<(usize, f64)> = Vec::with_capacity(e - s);
            let mut p = s;
            while p < e {
                let c = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                let v = unsafe { *a.data.get_unchecked(p) };
                row_pairs.push((c, v));
                p += 1;
            }
            let out = diff_row_pairs(row_pairs, ncols, n);
            out.len()
        })
        .collect();
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows {
        indptr[i + 1] = indptr[i] + (counts[i] as i64);
    }
    let total = indptr[nrows] as usize;
    let mut indices = vec![0i64; total];
    let mut data = vec![0.0f64; total];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    // Second pass: fill
    (0..nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let mut row_pairs: Vec<(usize, f64)> = Vec::with_capacity(e - s);
        let mut p = s;
        while p < e {
            let c = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
            let v = unsafe { *a.data.get_unchecked(p) };
            row_pairs.push((c, v));
            p += 1;
        }
        let out = diff_row_pairs(row_pairs, ncols, n);
        let row_start = i64_to_usize(indptr[i]);
        for (k, (c, v)) in out.into_iter().enumerate() {
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(row_start + k), c as i64);
                std::ptr::write(pv.add(row_start + k), v);
            }
        }
    });
    Csr::from_parts_unchecked(nrows, out_cols, indptr, indices, data)
}

#[must_use]
pub fn diff_csr_axis0_f64_i64(a: &Csr<f64, i64>, n: usize) -> Csr<f64, i64> {
    if n == 0 {
        return Csr::from_parts_unchecked(
            a.nrows,
            a.ncols,
            a.indptr.clone(),
            a.indices.clone(),
            a.data.clone(),
        );
    }
    let t = transpose_f64_i64(a);
    let td = diff_csr_axis1_f64_i64(&t, n);
    transpose_f64_i64(&td)
}

#[must_use]
pub fn diff_csc_axis1_f64_i64(a: &Csc<f64, i64>, n: usize) -> Csc<f64, i64> {
    let csr = csc_to_csr_f64_i64(a);
    let b = diff_csr_axis1_f64_i64(&csr, n);
    csr_to_csc_f64_i64(&b)
}

#[must_use]
pub fn diff_csc_axis0_f64_i64(a: &Csc<f64, i64>, n: usize) -> Csc<f64, i64> {
    let csr = csc_to_csr_f64_i64(a);
    let b = diff_csr_axis0_f64_i64(&csr, n);
    csr_to_csc_f64_i64(&b)
}

#[must_use]
pub fn diff_coo_axis1_f64_i64(a: &Coo<f64, i64>, n: usize) -> Coo<f64, i64> {
    use crate::data_type_functions::astype::{coo_to_csr_f64_i64, csr_to_coo_f64_i64};
    let csr = coo_to_csr_f64_i64(a);
    let b = diff_csr_axis1_f64_i64(&csr, n);
    csr_to_coo_f64_i64(&b)
}

#[must_use]
pub fn diff_coo_axis0_f64_i64(a: &Coo<f64, i64>, n: usize) -> Coo<f64, i64> {
    use crate::data_type_functions::astype::{coo_to_csr_f64_i64, csr_to_coo_f64_i64};
    let csr = coo_to_csr_f64_i64(a);
    let b = diff_csr_axis0_f64_i64(&csr, n);
    csr_to_coo_f64_i64(&b)
}
