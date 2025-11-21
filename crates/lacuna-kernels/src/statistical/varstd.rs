//! Variance and standard deviation reductions (CSR/CSC/COO/COOND)

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

use crate::linalg::matrix_transpose::{transpose_csc_f64_i64, transpose_f64_i64};
use crate::utility::util::{STRIPE, i64_to_usize};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use wide::f64x4;

// Type aliases to reduce complexity in ThreadLocal accumulator types
type AccEntry = (Vec<f64>, Vec<f64>, Vec<u8>, Vec<usize>);
type AccTls = ThreadLocal<RefCell<Vec<Option<AccEntry>>>>;

#[inline]
fn chunk_sum_sum2(chunk: &[f64]) -> (f64, f64) {
    if chunk.is_empty() {
        return (0.0, 0.0);
    }
    let mut acc = f64x4::from([0.0; 4]);
    let mut acc2 = f64x4::from([0.0; 4]);
    let mut i = 0usize;
    let limit4 = chunk.len() & !3;
    while i < limit4 {
        let v = unsafe {
            let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
            f64x4::new(core::ptr::read_unaligned(p))
        };
        acc += v;
        let vv = v * v;
        acc2 += vv;
        i += 4;
    }
    let arr = acc.to_array();
    let mut s = arr[0] + arr[1] + arr[2] + arr[3];
    let arr2 = acc2.to_array();
    let mut s2 = arr2[0] + arr2[1] + arr2[2] + arr2[3];
    while i < chunk.len() {
        let x = chunk[i];
        s += x;
        s2 += x * x;
        i += 1;
    }
    (s, s2)
}

#[inline]
#[allow(clippy::cast_precision_loss)]
fn variance_from_sums(s: f64, s2: f64, n: usize, correction: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let denom = nf - correction;
    if denom <= 0.0 {
        return 0.0;
    }
    let mean_sq = s * s / nf;
    let var = (s2 - mean_sq) / denom;
    if var < 0.0 { 0.0 } else { var }
}

#[inline]
fn std_from_var(v: f64) -> f64 {
    v.sqrt()
}

// ===== CSR =====

#[must_use]
pub fn var_f64(a: &Csr<f64, i64>, correction: f64) -> f64 {
    let n = a.nrows.saturating_mul(a.ncols);
    if n == 0 {
        return 0.0;
    }
    let (s, s2) = a
        .data
        .par_chunks(4096)
        .map(chunk_sum_sum2)
        .reduce(|| (0.0, 0.0), |(s1, s21), (s2_, s22)| (s1 + s2_, s21 + s22));
    variance_from_sums(s, s2, n, correction)
}

#[must_use]
pub fn std_f64(a: &Csr<f64, i64>, correction: f64) -> f64 {
    std_from_var(var_f64(a, correction))
}

#[must_use]
pub fn row_vars_f64(a: &Csr<f64, i64>, correction: f64) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        if ncols == 0 {
            *oi = 0.0;
            return;
        }
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let (sum, sum2) = chunk_sum_sum2(&a.data[s..e]);
        *oi = variance_from_sums(sum, sum2, ncols, correction);
    });
    out
}

#[must_use]
pub fn row_stds_f64(a: &Csr<f64, i64>, correction: f64) -> Vec<f64> {
    row_vars_f64(a, correction)
        .into_iter()
        .map(std_from_var)
        .collect()
}

#[must_use]
pub fn col_vars_f64(a: &Csr<f64, i64>, correction: f64) -> Vec<f64> {
    let t = transpose_f64_i64(a);
    row_vars_f64(&t, correction)
}

#[must_use]
pub fn col_stds_f64(a: &Csr<f64, i64>, correction: f64) -> Vec<f64> {
    let t = transpose_f64_i64(a);
    row_stds_f64(&t, correction)
}

// ===== CSC =====

#[must_use]
pub fn var_csc_f64(a: &Csc<f64, i64>, correction: f64) -> f64 {
    let n = a.nrows.saturating_mul(a.ncols);
    if n == 0 {
        return 0.0;
    }
    let (s, s2) = a
        .data
        .par_chunks(4096)
        .map(chunk_sum_sum2)
        .reduce(|| (0.0, 0.0), |(s1, s21), (s2_, s22)| (s1 + s2_, s21 + s22));
    variance_from_sums(s, s2, n, correction)
}

#[must_use]
pub fn std_csc_f64(a: &Csc<f64, i64>, correction: f64) -> f64 {
    std_from_var(var_csc_f64(a, correction))
}

#[must_use]
pub fn col_vars_csc_f64(a: &Csc<f64, i64>, correction: f64) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        if nrows == 0 {
            *oj = 0.0;
            return;
        }
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let (sum, sum2) = chunk_sum_sum2(&a.data[s..e]);
        *oj = variance_from_sums(sum, sum2, nrows, correction);
    });
    out
}

#[must_use]
pub fn col_stds_csc_f64(a: &Csc<f64, i64>, correction: f64) -> Vec<f64> {
    col_vars_csc_f64(a, correction)
        .into_iter()
        .map(std_from_var)
        .collect()
}

#[must_use]
pub fn row_vars_csc_f64(a: &Csc<f64, i64>, correction: f64) -> Vec<f64> {
    let t = transpose_csc_f64_i64(a);
    col_vars_csc_f64(&t, correction)
}

#[must_use]
pub fn row_stds_csc_f64(a: &Csc<f64, i64>, correction: f64) -> Vec<f64> {
    let t = transpose_csc_f64_i64(a);
    col_stds_csc_f64(&t, correction)
}

// ===== COO =====

#[must_use]
pub fn var_coo_f64(a: &Coo<f64, i64>, correction: f64) -> f64 {
    let n = a.nrows.saturating_mul(a.ncols);
    if n == 0 {
        return 0.0;
    }
    let (s, s2) = a
        .data
        .par_chunks(4096)
        .map(chunk_sum_sum2)
        .reduce(|| (0.0, 0.0), |(s1, s21), (s2_, s22)| (s1 + s2_, s21 + s22));
    variance_from_sums(s, s2, n, correction)
}

#[must_use]
pub fn std_coo_f64(a: &Coo<f64, i64>, correction: f64) -> f64 {
    std_from_var(var_coo_f64(a, correction))
}

#[must_use]
pub fn row_vars_coo_f64(a: &Coo<f64, i64>, correction: f64) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    if ncols == 0 {
        return vec![0.0; nrows];
    }
    let nnz = a.data.len();
    let nstripes = nrows.div_ceil(STRIPE);
    // acc: (sum, sumsq, seen, touched)
    let tls: AccTls = ThreadLocal::new();
    let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
    (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
        let start = t * chunk;
        let end = (start + chunk).min(nnz);
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
        for k in start..end {
            let i = i64_to_usize(a.row[k]);
            let sid = i / STRIPE;
            let base = sid * STRIPE;
            let off = i - base;
            if accs[sid].is_none() {
                let stripe_len = (nrows - base).min(STRIPE);
                accs[sid] = Some((
                    vec![0.0f64; stripe_len],
                    vec![0.0f64; stripe_len],
                    vec![0u8; stripe_len],
                    Vec::new(),
                ));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.2[off] == 0 {
                acc.2[off] = 1;
                acc.0[off] = a.data[k];
                acc.1[off] = a.data[k] * a.data[k];
                acc.3.push(off);
            } else {
                let v = a.data[k];
                acc.0[off] += v;
                acc.1[off] += v * v;
            }
        }
    });
    let mut sums = vec![0.0f64; nrows];
    let mut sums2 = vec![0.0f64; nrows];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, vals2, _seen, touched)) = stripe_opt {
                let base = sid * STRIPE;
                for &off in &touched {
                    let idx = base + off;
                    sums[idx] += vals[off];
                    sums2[idx] += vals2[off];
                }
            }
        }
    }
    let mut out = vec![0.0f64; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        *oi = variance_from_sums(sums[i], sums2[i], ncols, correction);
    });
    out
}

#[must_use]
pub fn row_stds_coo_f64(a: &Coo<f64, i64>, correction: f64) -> Vec<f64> {
    row_vars_coo_f64(a, correction)
        .into_iter()
        .map(std_from_var)
        .collect()
}

#[must_use]
pub fn col_vars_coo_f64(a: &Coo<f64, i64>, correction: f64) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    if nrows == 0 {
        return vec![0.0; ncols];
    }
    let nnz = a.data.len();
    let nstripes = ncols.div_ceil(STRIPE);
    let tls: AccTls = ThreadLocal::new();
    let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
    (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
        let start = t * chunk;
        let end = (start + chunk).min(nnz);
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
        for k in start..end {
            let j = i64_to_usize(a.col[k]);
            let sid = j / STRIPE;
            let base = sid * STRIPE;
            let off = j - base;
            if accs[sid].is_none() {
                let stripe_len = (ncols - base).min(STRIPE);
                accs[sid] = Some((
                    vec![0.0f64; stripe_len],
                    vec![0.0f64; stripe_len],
                    vec![0u8; stripe_len],
                    Vec::new(),
                ));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.2[off] == 0 {
                acc.2[off] = 1;
                acc.0[off] = a.data[k];
                acc.1[off] = a.data[k] * a.data[k];
                acc.3.push(off);
            } else {
                let v = a.data[k];
                acc.0[off] += v;
                acc.1[off] += v * v;
            }
        }
    });
    let mut sums = vec![0.0f64; ncols];
    let mut sums2 = vec![0.0f64; ncols];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, vals2, _seen, touched)) = stripe_opt {
                let base = sid * STRIPE;
                for &off in &touched {
                    let idx = base + off;
                    sums[idx] += vals[off];
                    sums2[idx] += vals2[off];
                }
            }
        }
    }
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        *oj = variance_from_sums(sums[j], sums2[j], nrows, correction);
    });
    out
}

#[must_use]
pub fn col_stds_coo_f64(a: &Coo<f64, i64>, correction: f64) -> Vec<f64> {
    col_vars_coo_f64(a, correction)
        .into_iter()
        .map(std_from_var)
        .collect()
}

// ===== COOND =====

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

#[must_use]
pub fn var_coond_f64(a: &CooNd<f64, i64>, correction: f64) -> f64 {
    if a.shape.is_empty() {
        return 0.0;
    }
    let n = product_checked(&a.shape);
    if n == 0 {
        return 0.0;
    }
    let (s, s2) = a
        .data
        .par_chunks(4096)
        .map(chunk_sum_sum2)
        .reduce(|| (0.0, 0.0), |(s1, s21), (s2_, s22)| (s1 + s2_, s21 + s22));
    variance_from_sums(s, s2, n, correction)
}

#[must_use]
pub fn std_coond_f64(a: &CooNd<f64, i64>, correction: f64) -> f64 {
    std_from_var(var_coond_f64(a, correction))
}
