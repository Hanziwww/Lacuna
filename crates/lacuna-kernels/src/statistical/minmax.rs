#![allow(clippy::many_single_char_names, clippy::too_many_lines)]
use crate::linalg::matrix_transpose::{transpose_csc_f64_i64, transpose_f64_i64};
use crate::utility::util::{i64_to_usize, StripeAccs, STRIPE};
use lacuna_core::{Coo, Csc, Csr};
use rayon::prelude::*;
use wide::f64x4;
use std::cell::RefCell;
use thread_local::ThreadLocal;

#[inline]
fn chunk_min(chunk: &[f64]) -> f64 {
    if chunk.is_empty() {
        return f64::INFINITY;
    }
    let mut accv = f64x4::from([f64::INFINITY; 4]);
    let mut i = 0usize;
    let limit4 = chunk.len() & !3;
    while i < limit4 {
        let v = unsafe {
            let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
            f64x4::new(core::ptr::read_unaligned(p))
        };
        accv = accv.min(v);
        i += 4;
    }
    let arr = accv.to_array();
    let mut acc = arr[0].min(arr[1]).min(arr[2]).min(arr[3]);
    while i < chunk.len() {
        let x = chunk[i];
        if x < acc {
            acc = x;
        }
        i += 1;
    }
    acc
}

#[inline]
fn chunk_max(chunk: &[f64]) -> f64 {
    if chunk.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut accv = f64x4::from([f64::NEG_INFINITY; 4]);
    let mut i = 0usize;
    let limit4 = chunk.len() & !3;
    while i < limit4 {
        let v = unsafe {
            let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
            f64x4::new(core::ptr::read_unaligned(p))
        };
        accv = accv.max(v);
        i += 4;
    }
    let arr = accv.to_array();
    let mut acc = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
    while i < chunk.len() {
        let x = chunk[i];
        if x > acc {
            acc = x;
        }
        i += 1;
    }
    acc
}

#[must_use]
pub fn min_f64(a: &Csr<f64, i64>) -> f64 {
    let nnz = a.data.len();
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_min = a
        .data
        .par_chunks(4096)
        .map(chunk_min)
        .reduce(|| f64::INFINITY, f64::min);

    let full = a
        .nrows
        .checked_mul(a.ncols)
        .unwrap_or(usize::MAX);
    if nnz < full {
        data_min.min(0.0)
    } else {
        data_min
    }
}

#[must_use]
pub fn max_f64(a: &Csr<f64, i64>) -> f64 {
    let nnz = a.data.len();
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_max = a
        .data
        .par_chunks(4096)
        .map(chunk_max)
        .reduce(|| f64::NEG_INFINITY, f64::max);

    let full = a
        .nrows
        .checked_mul(a.ncols)
        .unwrap_or(usize::MAX);
    if nnz < full {
        data_max.max(0.0)
    } else {
        data_max
    }
}

#[must_use]
pub fn row_mins_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut out = vec![0.0f64; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        if s == e {
            *oi = 0.0;
            return;
        }
        let row = &a.data[s..e];
        let m = chunk_min(row);
        if (e - s) < ncols {
            *oi = m.min(0.0);
        } else {
            *oi = m;
        }
    });
    out
}

#[must_use]
pub fn row_maxs_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut out = vec![0.0f64; nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        if s == e {
            *oi = 0.0;
            return;
        }
        let row = &a.data[s..e];
        let m = chunk_max(row);
        if (e - s) < ncols {
            *oi = m.max(0.0);
        } else {
            *oi = m;
        }
    });
    out
}

#[must_use]
pub fn col_mins_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let t = transpose_f64_i64(a);
    // row mins of transpose equals column mins of original
    row_mins_f64(&t)
}

#[must_use]
pub fn col_maxs_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let t = transpose_f64_i64(a);
    row_maxs_f64(&t)
}

#[must_use]
pub fn min_csc_f64(a: &Csc<f64, i64>) -> f64 {
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_min = a
        .data
        .par_chunks(4096)
        .map(chunk_min)
        .reduce(|| f64::INFINITY, f64::min);
    let full = a
        .nrows
        .checked_mul(a.ncols)
        .unwrap_or(usize::MAX);
    if a.data.len() < full {
        data_min.min(0.0)
    } else {
        data_min
    }
}

#[must_use]
pub fn max_csc_f64(a: &Csc<f64, i64>) -> f64 {
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_max = a
        .data
        .par_chunks(4096)
        .map(chunk_max)
        .reduce(|| f64::NEG_INFINITY, f64::max);
    let full = a
        .nrows
        .checked_mul(a.ncols)
        .unwrap_or(usize::MAX);
    if a.data.len() < full {
        data_max.max(0.0)
    } else {
        data_max
    }
}

#[must_use]
pub fn col_mins_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        if s == e {
            *oj = 0.0;
            return;
        }
        let col = &a.data[s..e];
        let m = chunk_min(col);
        if (e - s) < nrows {
            *oj = m.min(0.0);
        } else {
            *oj = m;
        }
    });
    out
}

#[must_use]
pub fn col_maxs_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    let nrows = a.nrows;
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        if s == e {
            *oj = 0.0;
            return;
        }
        let col = &a.data[s..e];
        let m = chunk_max(col);
        if (e - s) < nrows {
            *oj = m.max(0.0);
        } else {
            *oj = m;
        }
    });
    out
}

#[must_use]
pub fn row_mins_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let t = transpose_csc_f64_i64(a);
    // column mins of transposed CSC equal row mins of original
    col_mins_csc_f64(&t)
}

#[must_use]
pub fn row_maxs_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let t = transpose_csc_f64_i64(a);
    col_maxs_csc_f64(&t)
}

#[must_use]
pub fn min_coo_f64(a: &Coo<f64, i64>) -> f64 {
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_min = a
        .data
        .par_chunks(4096)
        .map(chunk_min)
        .reduce(|| f64::INFINITY, f64::min);
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if a.data.len() < full {
        data_min.min(0.0)
    } else {
        data_min
    }
}

#[must_use]
pub fn max_coo_f64(a: &Coo<f64, i64>) -> f64 {
    if a.nrows == 0 || a.ncols == 0 {
        return 0.0;
    }
    let data_max = a
        .data
        .par_chunks(4096)
        .map(chunk_max)
        .reduce(|| f64::NEG_INFINITY, f64::max);
    let full = a.nrows.checked_mul(a.ncols).unwrap_or(usize::MAX);
    if a.data.len() < full {
        data_max.max(0.0)
    } else {
        data_max
    }
}

#[must_use]
pub fn row_mins_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    if nrows == 0 {
        return Vec::new();
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
                accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                let cur = acc.0[off];
                let v = a.data[k];
                acc.0[off] = if v < cur { v } else { cur };
            }
        }
    });
    let mut out = vec![0.0f64; nrows];
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
                    } else if vals[off] < out[idx] {
                        out[idx] = vals[off];
                    }
                }
            }
        }
    }
    for i in 0..nrows {
        if counts[i] < a.ncols {
            if seen[i] == 1 {
                out[i] = out[i].min(0.0);
            } else {
                out[i] = 0.0;
            }
        }
    }
    out
}

#[must_use]
pub fn row_maxs_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    if nrows == 0 {
        return Vec::new();
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
                accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                let cur = acc.0[off];
                let v = a.data[k];
                acc.0[off] = if v > cur { v } else { cur };
            }
        }
    });
    let mut out = vec![0.0f64; nrows];
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
                    } else if vals[off] > out[idx] {
                        out[idx] = vals[off];
                    }
                }
            }
        }
    }
    for i in 0..nrows {
        if counts[i] < a.ncols {
            if seen[i] == 1 {
                out[i] = out[i].max(0.0);
            } else {
                out[i] = 0.0;
            }
        }
    }
    out
}

#[must_use]
pub fn col_mins_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
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
                accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                let cur = acc.0[off];
                let v = a.data[k];
                acc.0[off] = if v < cur { v } else { cur };
            }
        }
    });
    let mut out = vec![0.0f64; ncols];
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
                    } else if vals[off] < out[idx] {
                        out[idx] = vals[off];
                    }
                }
            }
        }
    }
    for j in 0..ncols {
        if counts[j] < a.nrows {
            if seen[j] == 1 {
                out[j] = out[j].min(0.0);
            } else {
                out[j] = 0.0;
            }
        }
    }
    out
}

#[must_use]
pub fn col_maxs_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
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
                accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
            }
            let acc = accs[sid].as_mut().unwrap();
            if acc.1[off] == 0 {
                acc.1[off] = 1;
                acc.0[off] = a.data[k];
                acc.2.push(off);
            } else {
                let cur = acc.0[off];
                let v = a.data[k];
                acc.0[off] = if v > cur { v } else { cur };
            }
        }
    });
    let mut out = vec![0.0f64; ncols];
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
                    } else if vals[off] > out[idx] {
                        out[idx] = vals[off];
                    }
                }
            }
        }
    }
    for j in 0..ncols {
        if counts[j] < a.nrows {
            if seen[j] == 1 {
                out[j] = out[j].max(0.0);
            } else {
                out[j] = 0.0;
            }
        }
    }
    out
}
