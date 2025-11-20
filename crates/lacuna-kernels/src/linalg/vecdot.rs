#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

use crate::linalg::matmul::{spmv_coo_f64_i64, spmv_coond_f64_i64, spmv_csc_f64_i64, spmv_f64_i64};
use crate::utility::util::{STRIPE, StripeAccs, UsizeF64Map, i64_to_usize};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;

#[must_use]
pub fn vecdot_csr_dense_axis1_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_f64_i64(a, x)
}

#[must_use]
pub fn vecdot_csr_dense_axis0_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }

    // Stripe-accumulator pattern across columns, weighting by row vector x
    let stripe = STRIPE;
    let nstripes = ncols.div_ceil(stripe);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();

    let target: usize = 128 * 1024;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut acc = 0usize;
    let mut r0 = 0usize;
    for i in 0..a.nrows {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let row_nnz = e - s;
        if acc == 0 {
            r0 = i;
        }
        acc += row_nnz;
        if acc >= target {
            ranges.push((r0, i + 1));
            acc = 0;
        }
    }
    if acc > 0 {
        ranges.push((r0, a.nrows));
    }

    ranges.into_par_iter().for_each(|(r0, r1)| {
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
        let xs = &x[r0..r1];
        for (off_row, &w) in xs.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let i = r0 + off_row;
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                let sid = j / stripe;
                let base = sid * stripe;
                let off = j - base;
                if accs[sid].is_none() {
                    let stripe_len = (ncols - base).min(stripe);
                    accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
                }
                let acc = accs[sid].as_mut().unwrap();
                if acc.1[off] == 0 {
                    acc.1[off] = 1;
                    acc.2.push(off);
                }
                acc.0[off] += a.data[p] * w;
            }
        }
    });

    let mut out = vec![0.0f64; ncols];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, mut seen, touched)) = stripe_opt {
                let base = sid * stripe;
                if touched.len() > vals.len() / 2 {
                    for (off, &v) in vals.iter().enumerate() {
                        out[base + off] += v;
                    }
                } else {
                    for &off in &touched {
                        out[base + off] += vals[off];
                        seen[off] = 0;
                    }
                }
            }
        }
    }
    out
}

#[must_use]
pub fn vecdot_csc_dense_axis1_f64_i64(a: &Csc<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_csc_f64_i64(a, x)
}

#[allow(
    clippy::too_many_lines,
    reason = "Kernel opts and unrolled loops make this long"
)]
#[must_use]
pub fn vecdot_csc_dense_axis0_f64_i64(a: &Csc<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let mut acc = 0.0f64;
        let mut p = s;
        let end16 = e - ((e - p) & 15);
        unsafe {
            while p < end16 {
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                let i4 = i64_to_usize(*a.indices.get_unchecked(p + 4));
                let i5 = i64_to_usize(*a.indices.get_unchecked(p + 5));
                let i6 = i64_to_usize(*a.indices.get_unchecked(p + 6));
                let i7 = i64_to_usize(*a.indices.get_unchecked(p + 7));
                let i8 = i64_to_usize(*a.indices.get_unchecked(p + 8));
                let i9 = i64_to_usize(*a.indices.get_unchecked(p + 9));
                let i10 = i64_to_usize(*a.indices.get_unchecked(p + 10));
                let i11 = i64_to_usize(*a.indices.get_unchecked(p + 11));
                let i12 = i64_to_usize(*a.indices.get_unchecked(p + 12));
                let i13 = i64_to_usize(*a.indices.get_unchecked(p + 13));
                let i14 = i64_to_usize(*a.indices.get_unchecked(p + 14));
                let i15 = i64_to_usize(*a.indices.get_unchecked(p + 15));
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                acc = (*a.data.get_unchecked(p + 7)).mul_add(
                    *x.get_unchecked(i7),
                    (*a.data.get_unchecked(p + 6)).mul_add(
                        *x.get_unchecked(i6),
                        (*a.data.get_unchecked(p + 5)).mul_add(
                            *x.get_unchecked(i5),
                            (*a.data.get_unchecked(p + 4)).mul_add(*x.get_unchecked(i4), acc),
                        ),
                    ),
                );
                acc = (*a.data.get_unchecked(p + 11)).mul_add(
                    *x.get_unchecked(i11),
                    (*a.data.get_unchecked(p + 10)).mul_add(
                        *x.get_unchecked(i10),
                        (*a.data.get_unchecked(p + 9)).mul_add(
                            *x.get_unchecked(i9),
                            (*a.data.get_unchecked(p + 8)).mul_add(*x.get_unchecked(i8), acc),
                        ),
                    ),
                );
                acc = (*a.data.get_unchecked(p + 15)).mul_add(
                    *x.get_unchecked(i15),
                    (*a.data.get_unchecked(p + 14)).mul_add(
                        *x.get_unchecked(i14),
                        (*a.data.get_unchecked(p + 13)).mul_add(
                            *x.get_unchecked(i13),
                            (*a.data.get_unchecked(p + 12)).mul_add(*x.get_unchecked(i12), acc),
                        ),
                    ),
                );
                p += 16;
            }
            let end8 = e - ((e - p) & 7);
            while p < end8 {
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                let i4 = i64_to_usize(*a.indices.get_unchecked(p + 4));
                let i5 = i64_to_usize(*a.indices.get_unchecked(p + 5));
                let i6 = i64_to_usize(*a.indices.get_unchecked(p + 6));
                let i7 = i64_to_usize(*a.indices.get_unchecked(p + 7));
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                acc = (*a.data.get_unchecked(p + 7)).mul_add(
                    *x.get_unchecked(i7),
                    (*a.data.get_unchecked(p + 6)).mul_add(
                        *x.get_unchecked(i6),
                        (*a.data.get_unchecked(p + 5)).mul_add(
                            *x.get_unchecked(i5),
                            (*a.data.get_unchecked(p + 4)).mul_add(*x.get_unchecked(i4), acc),
                        ),
                    ),
                );
                p += 8;
            }
            let end4 = e - ((e - p) & 3);
            while p < end4 {
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                p += 4;
            }
            while p < e {
                let i = i64_to_usize(*a.indices.get_unchecked(p));
                acc = (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i), acc);
                p += 1;
            }
        }
        *oj = acc;
    });
    out
}

#[must_use]
pub fn vecdot_coo_dense_axis1_f64_i64(a: &Coo<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_coo_f64_i64(a, x)
}

#[must_use]
pub fn vecdot_coo_dense_axis0_f64_i64(a: &Coo<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    let nnz = a.data.len();
    if ncols == 0 || nnz == 0 {
        return vec![0.0; ncols];
    }
    let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
    let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
    (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
        let start = t * chunk;
        let end = (start + chunk).min(nnz);
        let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
        let mut acc = cell.borrow_mut();
        for k in start..end {
            let i = i64_to_usize(a.row[k]);
            let j = i64_to_usize(a.col[k]);
            let w = x[i];
            if w != 0.0 {
                acc.insert_add(j, a.data[k] * w);
            }
        }
    });
    let mut out = vec![0.0f64; ncols];
    for cell in tls {
        let mut acc = cell.into_inner();
        acc.drain_to(&mut out);
    }
    out
}

#[must_use]
pub fn vecdot_coond_dense_axis_f64_i64(
    a: &CooNd<f64, i64>,
    axis: usize,
    x: &[f64],
) -> CooNd<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    assert_eq!(x.len(), a.shape[axis], "x length must equal shape[axis]");
    spmv_coond_f64_i64(a, axis, x)
}
