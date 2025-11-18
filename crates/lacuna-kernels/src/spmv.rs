#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p to denote indices and pointers"
)]
use crate::util::{
    DenseStripe, SMALL_DIM_LIMIT, SMALL_NNZ_LIMIT, STRIPE_ROWS, StripeAccs, UsizeF64Map,
    i64_to_usize,
};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// y = A @ x for CSC
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn spmv_csc_f64_i64(a: &Csc<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    let nrows = a.nrows;
    let nnz = a.data.len();
    let mut y = vec![0.0f64; nrows];

    let small = a.ncols <= SMALL_DIM_LIMIT || nnz <= SMALL_NNZ_LIMIT;
    if small {
        for (j, &xj) in x.iter().enumerate().take(a.ncols) {
            let s = i64_to_usize(a.indptr[j]);
            let e = i64_to_usize(a.indptr[j + 1]);
            for p in s..e {
                let i = i64_to_usize(a.indices[p]);
                y[i] += a.data[p] * xj;
            }
        }
        return y;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let dense_cutover = (nrows / 2).saturating_mul(nthreads);
    let target: usize = 128 * 1024;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut acc = 0usize;
    let mut c0 = 0usize;
    for j in 0..a.ncols {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let col_nnz = e - s;
        if acc == 0 {
            c0 = j;
        }
        acc += col_nnz;
        if acc >= target {
            ranges.push((c0, j + 1));
            acc = 0;
        }
    }
    if acc > 0 {
        ranges.push((c0, a.ncols));
    }

    if nnz < dense_cutover {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        ranges.into_par_iter().for_each(|(c0, c1)| {
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            for (j, &xj) in x.iter().enumerate().take(c1).skip(c0) {
                let s = i64_to_usize(a.indptr[j]);
                let e = i64_to_usize(a.indptr[j + 1]);
                for p in s..e {
                    let i = i64_to_usize(a.indices[p]);
                    acc.insert_add(i, a.data[p] * xj);
                }
            }
        });
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut y);
        }
        return y;
    }

    if nrows <= STRIPE_ROWS * 2 {
        let tls: ThreadLocal<RefCell<DenseStripe>> = ThreadLocal::new();
        ranges.into_par_iter().for_each(|(c0, c1)| {
            let cell =
                tls.get_or(|| RefCell::new((vec![0.0f64; nrows], vec![0u8; nrows], Vec::new())));
            let mut acc = cell.borrow_mut();
            let (vals, seen, touched) = &mut *acc;
            for (j, &xj) in x.iter().enumerate().take(c1).skip(c0) {
                let s = i64_to_usize(a.indptr[j]);
                let e = i64_to_usize(a.indptr[j + 1]);
                for p in s..e {
                    let i = i64_to_usize(a.indices[p]);
                    if seen[i] == 0 {
                        seen[i] = 1;
                        touched.push(i);
                    }
                    vals[i] += a.data[p] * xj;
                }
            }
        });
        for cell in tls {
            let (vals, mut seen, touched) = cell.into_inner();
            if touched.len() > nrows / 2 {
                for i in 0..nrows {
                    y[i] += vals[i];
                }
            } else {
                for &i in &touched {
                    y[i] += vals[i];
                    seen[i] = 0;
                }
            }
        }
        return y;
    }

    let stripe = STRIPE_ROWS;
    let nstripes = nrows.div_ceil(stripe);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
    ranges.into_par_iter().for_each(|(c0, c1)| {
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
        for (j, &xj) in x.iter().enumerate().take(c1).skip(c0) {
            let s = i64_to_usize(a.indptr[j]);
            let e = i64_to_usize(a.indptr[j + 1]);
            for p in s..e {
                let i = i64_to_usize(a.indices[p]);
                let sid = i / stripe;
                let base = sid * stripe;
                let off = i - base;
                if accs[sid].is_none() {
                    let stripe_len = (nrows - base).min(stripe);
                    accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
                }
                let acc = accs[sid].as_mut().unwrap();
                if acc.1[off] == 0 {
                    acc.1[off] = 1;
                    acc.2.push(off);
                }
                acc.0[off] += a.data[p] * xj;
            }
        }
    });
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, mut seen, touched)) = stripe_opt {
                let stripe_len = vals.len();
                let base = sid * stripe;
                if touched.len() > stripe_len / 2 {
                    for off in 0..stripe_len {
                        y[base + off] += vals[off];
                    }
                } else {
                    for &off in &touched {
                        y[base + off] += vals[off];
                        seen[off] = 0;
                    }
                }
            }
        }
    }
    y
}

/// ND SpMV along a specific axis: out = tensordot(a, x, axes=[axis])
#[allow(clippy::doc_markdown)]
#[must_use]
pub fn spmv_coond_f64_i64(a: &CooNd<f64, i64>, axis: usize, x: &[f64]) -> CooNd<f64, i64> {
    let ndim = a.shape.len();
    assert!(axis < ndim, "axis out of bounds");
    assert_eq!(x.len(), a.shape[axis], "x length must equal shape[axis]");

    let nnz = a.data.len();
    let remain_axes: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();
    assert!(
        !remain_axes.is_empty(),
        "contracting over all axes yields scalar; use sum_coond_f64 instead",
    );
    if nnz == 0 {
        let out_shape: Vec<usize> = remain_axes.iter().map(|&d| a.shape[d]).collect();
        return CooNd::from_parts_unchecked(out_shape, Vec::new(), Vec::new());
    }

    let out_ndim = remain_axes.len();
    let out_shape: Vec<usize> = remain_axes.iter().map(|&d| a.shape[d]).collect();
    let mut strides = vec![0usize; out_ndim];
    strides[out_ndim - 1] = 1;
    for i in (0..out_ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(out_shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }

    let mut acc = UsizeF64Map::with_capacity(nnz);
    for k in 0..nnz {
        let base = k * ndim;
        let ax_idx = i64_to_usize(a.indices[base + axis]);
        let mut lin: usize = 0;
        for (m, &d) in remain_axes.iter().enumerate() {
            let idx = i64_to_usize(a.indices[base + d]);
            lin = lin
                .checked_add(idx.checked_mul(strides[m]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        let contrib = a.data[k] * x[ax_idx];
        if contrib != 0.0 {
            acc.insert_add(lin, contrib);
        }
    }

    let mut pairs = acc.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);
    let out_nnz = pairs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * out_ndim];
    for (pos, (mut lin, v)) in pairs.into_iter().enumerate() {
        let base = pos * out_ndim;
        for m in 0..out_ndim {
            let s = strides[m];
            let idx = lin / s;
            lin -= idx * s;
            out_indices[base + m] = usize_to_i64(idx);
        }
        out_data.push(v);
    }
    CooNd::from_parts_unchecked(out_shape, out_indices, out_data)
}

/// y = A @ x for COO
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn spmv_coo_f64_i64(a: &Coo<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    let nrows = a.nrows;
    let nnz = a.data.len();
    let mut y = vec![0.0f64; nrows];

    let small = a.nrows <= SMALL_DIM_LIMIT || nnz <= SMALL_NNZ_LIMIT;
    if small {
        for k in 0..nnz {
            let i = i64_to_usize(a.row[k]);
            let j = i64_to_usize(a.col[k]);
            y[i] += a.data[k] * x[j];
        }
        return y;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let dense_cutover = (nrows / 2).saturating_mul(nthreads);
    let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
    if nnz < dense_cutover {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            for k in start..end {
                let i = i64_to_usize(a.row[k]);
                let j = i64_to_usize(a.col[k]);
                acc.insert_add(i, a.data[k] * x[j]);
            }
        });
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut y);
        }
        return y;
    }

    if nrows <= STRIPE_ROWS * 2 {
        let tls: ThreadLocal<RefCell<DenseStripe>> = ThreadLocal::new();
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell =
                tls.get_or(|| RefCell::new((vec![0.0f64; nrows], vec![0u8; nrows], Vec::new())));
            let mut acc = cell.borrow_mut();
            let (vals, seen, touched) = &mut *acc;
            for k in start..end {
                let i = i64_to_usize(a.row[k]);
                let j = i64_to_usize(a.col[k]);
                if seen[i] == 0 {
                    seen[i] = 1;
                    touched.push(i);
                }
                vals[i] += a.data[k] * x[j];
            }
        });
        for cell in tls {
            let (vals, mut seen, touched) = cell.into_inner();
            if touched.len() > nrows / 2 {
                for i in 0..nrows {
                    y[i] += vals[i];
                }
            } else {
                for &i in &touched {
                    y[i] += vals[i];
                    seen[i] = 0;
                }
            }
        }
        y
    } else {
        let stripe = STRIPE_ROWS;
        let nstripes = nrows.div_ceil(stripe);
        let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
            let mut accs = cell.borrow_mut();
            for k in start..end {
                let i = i64_to_usize(a.row[k]);
                let j = i64_to_usize(a.col[k]);
                let sid = i / stripe;
                let base = sid * stripe;
                let off = i - base;
                if accs[sid].is_none() {
                    let stripe_len = (nrows - base).min(stripe);
                    accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
                }
                let acc = accs[sid].as_mut().unwrap();
                if acc.1[off] == 0 {
                    acc.1[off] = 1;
                    acc.2.push(off);
                }
                acc.0[off] += a.data[k] * x[j];
            }
        });
        for cell in tls {
            let accs = cell.into_inner();
            for (sid, stripe_opt) in accs.into_iter().enumerate() {
                if let Some((vals, mut seen, touched)) = stripe_opt {
                    let stripe_len = vals.len();
                    let base = sid * stripe;
                    if touched.len() > stripe_len / 2 {
                        for off in 0..stripe_len {
                            y[base + off] += vals[off];
                        }
                    } else {
                        for &off in &touched {
                            y[base + off] += vals[off];
                            seen[off] = 0;
                        }
                    }
                }
            }
        }
        y
    }
}

#[allow(clippy::too_many_lines)]
#[inline]
fn spmv_row_f64_i64(a: &Csr<f64, i64>, x: &[f64], i: usize) -> f64 {
    let start = i64_to_usize(a.indptr[i]);
    let end = i64_to_usize(a.indptr[i + 1]);
    let len = end - start;

    if len == 0 {
        return 0.0;
    }

    let mut acc = 0.0f64;

    unsafe {
        let idx_ptr = a.indices.as_ptr().add(start);
        let val_ptr = a.data.as_ptr().add(start);

        let mut t = 0usize;
        let limit16 = len & !15;
        while t < limit16 {
            let j0 = i64_to_usize(*idx_ptr.add(t));
            let j1 = i64_to_usize(*idx_ptr.add(t + 1));
            let j2 = i64_to_usize(*idx_ptr.add(t + 2));
            let j3 = i64_to_usize(*idx_ptr.add(t + 3));
            let j4 = i64_to_usize(*idx_ptr.add(t + 4));
            let j5 = i64_to_usize(*idx_ptr.add(t + 5));
            let j6 = i64_to_usize(*idx_ptr.add(t + 6));
            let j7 = i64_to_usize(*idx_ptr.add(t + 7));
            let j8 = i64_to_usize(*idx_ptr.add(t + 8));
            let j9 = i64_to_usize(*idx_ptr.add(t + 9));
            let j10 = i64_to_usize(*idx_ptr.add(t + 10));
            let j11 = i64_to_usize(*idx_ptr.add(t + 11));
            let j12 = i64_to_usize(*idx_ptr.add(t + 12));
            let j13 = i64_to_usize(*idx_ptr.add(t + 13));
            let j14 = i64_to_usize(*idx_ptr.add(t + 14));
            let j15 = i64_to_usize(*idx_ptr.add(t + 15));

            acc = (*val_ptr.add(t + 3)).mul_add(
                *x.get_unchecked(j3),
                (*val_ptr.add(t + 2)).mul_add(
                    *x.get_unchecked(j2),
                    (*val_ptr.add(t + 1)).mul_add(
                        *x.get_unchecked(j1),
                        (*val_ptr.add(t)).mul_add(*x.get_unchecked(j0), acc),
                    ),
                ),
            );
            acc = (*val_ptr.add(t + 7)).mul_add(
                *x.get_unchecked(j7),
                (*val_ptr.add(t + 6)).mul_add(
                    *x.get_unchecked(j6),
                    (*val_ptr.add(t + 5)).mul_add(
                        *x.get_unchecked(j5),
                        (*val_ptr.add(t + 4)).mul_add(*x.get_unchecked(j4), acc),
                    ),
                ),
            );
            acc = (*val_ptr.add(t + 11)).mul_add(
                *x.get_unchecked(j11),
                (*val_ptr.add(t + 10)).mul_add(
                    *x.get_unchecked(j10),
                    (*val_ptr.add(t + 9)).mul_add(
                        *x.get_unchecked(j9),
                        (*val_ptr.add(t + 8)).mul_add(*x.get_unchecked(j8), acc),
                    ),
                ),
            );
            acc = (*val_ptr.add(t + 15)).mul_add(
                *x.get_unchecked(j15),
                (*val_ptr.add(t + 14)).mul_add(
                    *x.get_unchecked(j14),
                    (*val_ptr.add(t + 13)).mul_add(
                        *x.get_unchecked(j13),
                        (*val_ptr.add(t + 12)).mul_add(*x.get_unchecked(j12), acc),
                    ),
                ),
            );

            t += 16;
        }

        let limit8 = len & !7;
        while t < limit8 {
            let j0 = i64_to_usize(*idx_ptr.add(t));
            let j1 = i64_to_usize(*idx_ptr.add(t + 1));
            let j2 = i64_to_usize(*idx_ptr.add(t + 2));
            let j3 = i64_to_usize(*idx_ptr.add(t + 3));
            let j4 = i64_to_usize(*idx_ptr.add(t + 4));
            let j5 = i64_to_usize(*idx_ptr.add(t + 5));
            let j6 = i64_to_usize(*idx_ptr.add(t + 6));
            let j7 = i64_to_usize(*idx_ptr.add(t + 7));

            acc = (*val_ptr.add(t + 3)).mul_add(
                *x.get_unchecked(j3),
                (*val_ptr.add(t + 2)).mul_add(
                    *x.get_unchecked(j2),
                    (*val_ptr.add(t + 1)).mul_add(
                        *x.get_unchecked(j1),
                        (*val_ptr.add(t)).mul_add(*x.get_unchecked(j0), acc),
                    ),
                ),
            );
            acc = (*val_ptr.add(t + 7)).mul_add(
                *x.get_unchecked(j7),
                (*val_ptr.add(t + 6)).mul_add(
                    *x.get_unchecked(j6),
                    (*val_ptr.add(t + 5)).mul_add(
                        *x.get_unchecked(j5),
                        (*val_ptr.add(t + 4)).mul_add(*x.get_unchecked(j4), acc),
                    ),
                ),
            );

            t += 8;
        }

        let limit4 = len & !3;
        while t < limit4 {
            let j0 = i64_to_usize(*idx_ptr.add(t));
            let j1 = i64_to_usize(*idx_ptr.add(t + 1));
            let j2 = i64_to_usize(*idx_ptr.add(t + 2));
            let j3 = i64_to_usize(*idx_ptr.add(t + 3));

            acc = (*val_ptr.add(t + 3)).mul_add(
                *x.get_unchecked(j3),
                (*val_ptr.add(t + 2)).mul_add(
                    *x.get_unchecked(j2),
                    (*val_ptr.add(t + 1)).mul_add(
                        *x.get_unchecked(j1),
                        (*val_ptr.add(t)).mul_add(*x.get_unchecked(j0), acc),
                    ),
                ),
            );

            t += 4;
        }

        while t < len {
            let j = i64_to_usize(*idx_ptr.add(t));
            acc = (*val_ptr.add(t)).mul_add(*x.get_unchecked(j), acc);
            t += 1;
        }
    }

    acc
}

/// y = A @ x
#[must_use]
pub fn spmv_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    let nrows = a.nrows;
    let nnz = a.data.len();
    let mut y = vec![0.0f64; nrows];

    // For small problems, avoid rayon overhead and compute sequentially.
    let small = nrows <= SMALL_DIM_LIMIT || nnz <= SMALL_NNZ_LIMIT;
    if small {
        for (i, yi) in y.iter_mut().enumerate().take(nrows) {
            *yi = spmv_row_f64_i64(a, x, i);
        }
        return y;
    }

    let target: usize = 128 * 1024;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut acc = 0usize;
    let mut r0 = 0usize;
    for i in 0..nrows {
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
        ranges.push((r0, nrows));
    }
    let y_addr = y.as_mut_ptr() as usize;
    ranges.into_par_iter().for_each(|(r0, r1)| {
        let y_ptr = y_addr as *mut f64;
        for i in r0..r1 {
            let val = spmv_row_f64_i64(a, x, i);
            unsafe {
                *y_ptr.add(i) = val;
            }
        }
    });
    y
}
