//! Product reductions: prod, row_prods, col_prods (CSR/CSC/COO) and global COOND

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

use crate::linalg::matrix_transpose::{transpose_csc_f64_i64, transpose_f64_i64};
use crate::utility::util::{STRIPE, StripeAccs, i64_to_usize};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use wide::f64x4;

#[inline]
fn chunk_prod(chunk: &[f64]) -> f64 {
    if chunk.is_empty() {
        return 1.0;
    }
    let mut accv = f64x4::from([1.0; 4]);
    let mut i = 0usize;
    let limit4 = chunk.len() & !3;
    while i < limit4 {
        let v = unsafe {
            let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
            f64x4::new(core::ptr::read_unaligned(p))
        };
        accv = accv * v;
        i += 4;
    }
    let arr = accv.to_array();
    let mut acc = arr[0] * arr[1] * arr[2] * arr[3];
    while i < chunk.len() {
        acc *= chunk[i];
        i += 1;
    }
    acc
}

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

#[must_use]
pub fn prod_f64(a: &Csr<f64, i64>) -> f64 {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return 1.0;
    }
    if a.data.len() < full {
        return 0.0;
    }
    a.data
        .par_chunks(4096)
        .map(chunk_prod)
        .reduce(|| 1.0, |x, y| x * y)
}

#[must_use]
pub fn row_prods_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    if ncols == 0 {
        return vec![1.0; nrows];
    }
    let mut out = vec![0.0f64; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        if (e - s) < ncols {
            *oi = 0.0;
            return;
        }
        let row = &a.data[s..e];
        *oi = chunk_prod(row);
    });
    out
}

#[must_use]
pub fn col_prods_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let t = transpose_f64_i64(a);
    // row products of transpose equal column products of original
    row_prods_f64(&t)
}

#[must_use]
pub fn prod_csc_f64(a: &Csc<f64, i64>) -> f64 {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return 1.0;
    }
    if a.data.len() < full {
        return 0.0;
    }
    a.data
        .par_chunks(4096)
        .map(chunk_prod)
        .reduce(|| 1.0, |x, y| x * y)
}

#[must_use]
pub fn col_prods_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    if nrows == 0 {
        return vec![1.0; ncols];
    }
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        if (e - s) < nrows {
            *oj = 0.0;
            return;
        }
        let col = &a.data[s..e];
        *oj = chunk_prod(col);
    });
    out
}

#[must_use]
pub fn row_prods_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let t = transpose_csc_f64_i64(a);
    // column products of transposed CSC equal row products of original
    col_prods_csc_f64(&t)
}

#[must_use]
pub fn prod_coo_f64(a: &Coo<f64, i64>) -> f64 {
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if full == 0 {
        return 1.0;
    }
    if a.data.len() < full {
        return 0.0;
    }
    a.data
        .par_chunks(4096)
        .map(chunk_prod)
        .reduce(|| 1.0, |x, y| x * y)
}

#[must_use]
pub fn row_prods_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 {
        return Vec::new();
    }
    if ncols == 0 {
        return vec![1.0; nrows];
    }
    let nnz = a.data.len();
    let mut counts = vec![0usize; nrows];
    for &ri in &a.row {
        let i = i64_to_usize(ri);
        counts[i] += 1;
    }
    let nstripes = nrows.div_ceil(STRIPE);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
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
                accs[sid] = Some((vec![1.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                acc.0[off] *= a.data[k];
            }
        }
    });
    let mut out = vec![1.0f64; nrows];
    let mut seen = vec![0u8; nrows];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, _s, touched)) = stripe_opt {
                let base = sid * STRIPE;
                for &off in &touched {
                    let idx = base + off;
                    if seen[idx] == 0 {
                        seen[idx] = 1;
                        out[idx] = vals[off];
                    } else {
                        out[idx] *= vals[off];
                    }
                }
            }
        }
    }
    for i in 0..nrows {
        if counts[i] < ncols {
            out[i] = 0.0;
        }
    }
    out
}

#[must_use]
pub fn col_prods_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    if ncols == 0 {
        return Vec::new();
    }
    if nrows == 0 {
        return vec![1.0; ncols];
    }
    let nnz = a.data.len();
    let mut counts = vec![0usize; ncols];
    for &cj in &a.col {
        let j = i64_to_usize(cj);
        counts[j] += 1;
    }
    let nstripes = ncols.div_ceil(STRIPE);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
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
                accs[sid] = Some((vec![1.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                acc.0[off] *= a.data[k];
            }
        }
    });
    let mut out = vec![1.0f64; ncols];
    let mut seen = vec![0u8; ncols];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, _s, touched)) = stripe_opt {
                let base = sid * STRIPE;
                for &off in &touched {
                    let idx = base + off;
                    if seen[idx] == 0 {
                        seen[idx] = 1;
                        out[idx] = vals[off];
                    } else {
                        out[idx] *= vals[off];
                    }
                }
            }
        }
    }
    for j in 0..ncols {
        if counts[j] < nrows {
            out[j] = 0.0;
        }
    }
    out
}

#[must_use]
pub fn prod_coond_f64(a: &CooNd<f64, i64>) -> f64 {
    let full = product_checked(&a.shape);
    if full == 0 {
        return 1.0;
    }
    if a.data.len() < full {
        return 0.0;
    }
    a.data
        .par_chunks(4096)
        .map(chunk_prod)
        .reduce(|| 1.0, |x, y| x * y)
}
