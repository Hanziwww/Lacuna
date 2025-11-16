#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Math kernels conventionally use i/j/k for indices"
)]
use crate::util::{
    SMALL_DIM_LIMIT, SMALL_NNZ_LIMIT, STRIPE, StripeAccs, UsizeF64Map, i64_to_usize,
};
use lacuna_core::{Coo, Csc, Csr, CooNd};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use wide::f64x4;

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
    }
    acc
}

#[must_use]
pub fn mean_coond_f64(a: &CooNd<f64, i64>) -> f64 {
    if a.shape.is_empty() {
        return 0.0; // conventionally empty shape -> 0 length; avoid div by zero
    }
    let denom = product_checked(&a.shape) as f64;
    if denom == 0.0 {
        return 0.0;
    }
    sum_coond_f64(a) / denom
}

#[must_use]
pub fn reduce_mean_axes_coond_f64_i64(a: &CooNd<f64, i64>, axes: &[usize]) -> CooNd<f64, i64> {
    let reduced = reduce_sum_axes_coond_f64_i64(a, axes);
    let mut reduce = vec![false; a.shape.len()];
    for &ax in axes {
        reduce[ax] = true;
    }
    let mut denom_us: usize = 1;
    for (d, &sz) in a.shape.iter().enumerate() {
        if reduce[d] {
            denom_us = denom_us.checked_mul(sz).expect("shape product overflow");
        }
    }
    if denom_us == 1 {
        return reduced;
    }
    let factor = 1.0f64 / (denom_us as f64);
    crate::arith::mul_scalar_coond_f64(&reduced, factor)
}

#[must_use]
pub fn sum_csc_f64(a: &Csc<f64, i64>) -> f64 {
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = unsafe {
                    let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                accv += v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

#[must_use]
pub fn reduce_sum_axes_coond_f64_i64(a: &CooNd<f64, i64>, axes: &[usize]) -> CooNd<f64, i64> {
    let ndim = a.shape.len();
    assert!(!axes.is_empty(), "axes must be non-empty");
    let mut reduce = vec![false; ndim];
    for &ax in axes {
        assert!(ax < ndim, "axis out of bounds");
        assert!(!reduce[ax], "duplicate axis in axes");
        reduce[ax] = true;
    }
    let remain_axes: Vec<usize> = (0..ndim).filter(|&d| !reduce[d]).collect();
    assert!(
        !remain_axes.is_empty(),
        "reducing over all axes would yield a scalar; use sum_coond_f64"
    );

    let remain_ndim = remain_axes.len();
    let remain_shape: Vec<usize> = remain_axes.iter().map(|&d| a.shape[d]).collect();

    let mut strides = vec![0usize; remain_ndim];
    strides[remain_ndim - 1] = 1;
    for i in (0..remain_ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(remain_shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }

    let nnz = a.data.len();
    if nnz == 0 {
        return CooNd::from_parts_unchecked(remain_shape, Vec::new(), Vec::new());
    }

    let mut acc: UsizeF64Map = UsizeF64Map::with_capacity(nnz);
    let ndim_us = ndim; // alias for indexing
    for k in 0..nnz {
        let mut lin: usize = 0;
        let base = k * ndim_us;
        for (m, &d) in remain_axes.iter().enumerate() {
            let idx = i64_to_usize(a.indices[base + d]);
            let s = strides[m];
            lin = lin
                .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc.insert_add(lin, a.data[k]);
    }

    let mut pairs = acc.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);
    let out_nnz = pairs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * remain_ndim];

    for (pos, (lin, sum)) in pairs.into_iter().enumerate() {
        let mut rem = lin;
        let base = pos * remain_ndim;
        for m in 0..remain_ndim {
            let s = strides[m];
            let idx = rem / s;
            rem -= idx * s;
            out_indices[base + m] = idx as i64;
        }
        out_data.push(sum);
    }

    CooNd::from_parts_unchecked(remain_shape, out_indices, out_data)
}

#[must_use]
pub fn sum_coond_f64(a: &CooNd<f64, i64>) -> f64 {
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = unsafe {
                    let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                accv += v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

#[must_use]
pub fn sum_coo_f64(a: &Coo<f64, i64>) -> f64 {
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = unsafe {
                    let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                accv += v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

/// sum of all data
#[must_use]
pub fn sum_f64(a: &Csr<f64, i64>) -> f64 {
    // Parallel reduce with SIMD inside chunks
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = unsafe {
                    let p = chunk.as_ptr().add(i).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                accv += v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

/// row sums
#[must_use]
pub fn row_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    let mut out = vec![0.0f64; nrows];
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
    let out_addr = out.as_mut_ptr() as usize;
    ranges.into_par_iter().for_each(|(r0, r1)| {
        let out_ptr = out_addr as *mut f64;
        for i in r0..r1 {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let row = &a.data[s..e];
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut k = 0usize;
            let limit4 = row.len() & !3;
            while k < limit4 {
                let v = unsafe {
                    let p = row.as_ptr().add(k).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                accv += v;
                k += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while k < row.len() {
                acc += row[k];
                k += 1;
            }
            unsafe {
                *out_ptr.add(i) = acc;
            }
        }
    });
    out
}

/// column sums (parallelized and vectorized)
#[must_use]
pub fn col_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }
    let nnz = a.data.len();
    if ncols <= SMALL_DIM_LIMIT {
        let tls: ThreadLocal<RefCell<Vec<f64>>> = ThreadLocal::new();
        (0..a.nrows).into_par_iter().for_each(|i| {
            let cell = tls.get_or(|| RefCell::new(vec![0.0f64; ncols]));
            let mut local = cell.borrow_mut();
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                local[j] += a.data[p];
            }
        });
        let mut out = vec![0.0f64; ncols];
        let out_addr = out.as_mut_ptr() as usize;
        for cell in tls {
            let local = cell.into_inner();
            let out_ptr = out_addr as *mut f64;
            let mut c = 0usize;
            let limit4 = ncols & !3;
            while c < limit4 {
                unsafe {
                    let p_out = out_ptr.add(c).cast::<[f64; 4]>();
                    let v_out = f64x4::new(core::ptr::read_unaligned(p_out));
                    let p_loc = local.as_ptr().add(c).cast::<[f64; 4]>();
                    let v_loc = f64x4::new(core::ptr::read_unaligned(p_loc));
                    let r = v_out + v_loc;
                    core::ptr::write_unaligned(p_out, r.to_array());
                }
                c += 4;
            }
            while c < ncols {
                unsafe {
                    *out_ptr.add(c) += local[c];
                }
                c += 1;
            }
        }
        return out;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let dense_cutover = (ncols / 2).saturating_mul(nthreads);
    if nnz < dense_cutover {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        (0..a.nrows).into_par_iter().for_each(|i| {
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                acc.insert_add(j, a.data[p]);
            }
        });
        let mut out = vec![0.0f64; ncols];
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut out);
        }
        return out;
    }

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
        for i in r0..r1 {
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
                acc.0[off] += a.data[p];
            }
        }
    });
    let mut out = vec![0.0f64; ncols];
    let mut cells: Vec<StripeAccs> = Vec::new();
    for cell in tls {
        cells.push(cell.into_inner());
    }
    let out_addr = out.as_mut_ptr() as usize;
    (0..nstripes).into_par_iter().for_each(|sid| {
        let base = sid * stripe;
        let stripe_len = (ncols - base).min(stripe);
        let out_ptr = out_addr as *mut f64;
        for accs in &cells {
            if let Some((vals, _seen, touched)) = &accs[sid] {
                if touched.len() > stripe_len / 2 {
                    let mut c = 0usize;
                    let limit4 = stripe_len & !3;
                    while c < limit4 {
                        unsafe {
                            let p_out = out_ptr.add(base + c).cast::<[f64; 4]>();
                            let v_out = f64x4::new(core::ptr::read_unaligned(p_out));
                            let p_vals = vals.as_ptr().add(c).cast::<[f64; 4]>();
                            let v_vals = f64x4::new(core::ptr::read_unaligned(p_vals));
                            let r = v_out + v_vals;
                            core::ptr::write_unaligned(p_out, r.to_array());
                        }
                        c += 4;
                    }
                    while c < stripe_len {
                        unsafe {
                            *out_ptr.add(base + c) += vals[c];
                        }
                        c += 1;
                    }
                } else {
                    for &off in touched {
                        unsafe {
                            *out_ptr.add(base + off) += vals[off];
                        }
                    }
                }
            }
        }
    });
    out
}

#[must_use]
pub fn row_sums_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    if nrows == 0 {
        return Vec::new();
    }
    let nnz = a.data.len();
    if nrows <= SMALL_DIM_LIMIT || nnz <= SMALL_NNZ_LIMIT {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        (0..a.ncols).into_par_iter().for_each(|j| {
            let s = i64_to_usize(a.indptr[j]);
            let e = i64_to_usize(a.indptr[j + 1]);
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            for p in s..e {
                let i = i64_to_usize(a.indices[p]);
                acc.insert_add(i, a.data[p]);
            }
        });
        let mut out = vec![0.0f64; nrows];
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut out);
        }
        return out;
    }
    let stripe = STRIPE;
    let nstripes = nrows.div_ceil(stripe);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
    (0..a.ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
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
            acc.0[off] += a.data[p];
        }
    });
    let mut out = vec![0.0f64; nrows];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, mut seen, touched)) = stripe_opt {
                let stripe_len = vals.len();
                let base = sid * stripe;
                if touched.len() > stripe_len / 2 {
                    let mut c = 0usize;
                    let limit4 = stripe_len & !3;
                    while c < limit4 {
                        unsafe {
                            let out_ptr = out.as_mut_ptr().add(base + c);
                            let vals_ptr = vals.as_ptr().add(c);
                            *out_ptr += *vals_ptr;
                            *out_ptr.add(1) += *vals_ptr.add(1);
                            *out_ptr.add(2) += *vals_ptr.add(2);
                            *out_ptr.add(3) += *vals_ptr.add(3);
                        }
                        c += 4;
                    }
                    while c < stripe_len {
                        out[base + c] += vals[c];
                        c += 1;
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
pub fn col_sums_csc_f64(a: &Csc<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f64; ncols];
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let col = &a.data[s..e];
        let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
        let mut k = 0usize;
        let limit4 = col.len() & !3;
        while k < limit4 {
            let v = unsafe {
                let p = col.as_ptr().add(k).cast::<[f64; 4]>();
                f64x4::new(core::ptr::read_unaligned(p))
            };
            accv += v;
            k += 4;
        }
        let arr = accv.to_array();
        let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
        while k < col.len() {
            acc += col[k];
            k += 1;
        }
        *oj = acc;
    });
    out
}

#[must_use]
pub fn row_sums_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let nrows = a.nrows;
    if nrows == 0 {
        return Vec::new();
    }
    let nnz = a.data.len();
    if nnz == 0 {
        return vec![0.0; nrows];
    }
    if nnz <= SMALL_NNZ_LIMIT || nrows <= SMALL_DIM_LIMIT {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            for k in start..end {
                let i = i64_to_usize(a.row[k]);
                acc.insert_add(i, a.data[k]);
            }
        });
        let mut out = vec![0.0f64; nrows];
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut out);
        }
        out
    } else {
        let stripe = STRIPE;
        let nstripes = nrows.div_ceil(stripe);
        let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
        let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
            let mut accs = cell.borrow_mut();
            for k in start..end {
                let i = i64_to_usize(a.row[k]);
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
                acc.0[off] += a.data[k];
            }
        });
        let mut out = vec![0.0f64; nrows];
        for cell in tls {
            let accs = cell.into_inner();
            for (sid, stripe_opt) in accs.into_iter().enumerate() {
                if let Some((vals, mut seen, touched)) = stripe_opt {
                    let stripe_len = vals.len();
                    let base = sid * stripe;
                    if touched.len() > stripe_len / 2 {
                        for idx in 0..stripe_len {
                            out[base + idx] += vals[idx];
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
}

#[allow(clippy::too_many_lines)]
#[must_use]
pub fn col_sums_coo_f64(a: &Coo<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }
    let nnz = a.data.len();
    if nnz == 0 {
        return vec![0.0; ncols];
    }
    let nthreads = rayon::current_num_threads().max(1);
    let dense_cutover = (ncols / 2).saturating_mul(nthreads);
    if nnz < dense_cutover {
        let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
        let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
            let mut acc = cell.borrow_mut();
            for k in start..end {
                let j = i64_to_usize(a.col[k]);
                acc.insert_add(j, a.data[k]);
            }
        });
        let mut out = vec![0.0f64; ncols];
        for cell in tls {
            let mut acc = cell.into_inner();
            acc.drain_to(&mut out);
        }
        out
    } else {
        let stripe = STRIPE;
        let nstripes = ncols.div_ceil(stripe);
        let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();
        let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
        (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(nnz);
            let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
            let mut accs = cell.borrow_mut();
            for k in start..end {
                let j = i64_to_usize(a.col[k]);
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
                acc.0[off] += a.data[k];
            }
        });
        let mut out = vec![0.0f64; ncols];
        let mut cells: Vec<StripeAccs> = Vec::new();
        for cell in tls {
            cells.push(cell.into_inner());
        }
        let out_addr = out.as_mut_ptr() as usize;
        (0..nstripes).into_par_iter().for_each(|sid| {
            let base = sid * stripe;
            let stripe_len = (ncols - base).min(stripe);
            let out_ptr = out_addr as *mut f64;
            for accs in &cells {
                if let Some((vals, _seen, touched)) = &accs[sid] {
                    if touched.len() > stripe_len / 2 {
                        let mut c = 0usize;
                        let limit4 = stripe_len & !3;
                        while c < limit4 {
                            unsafe {
                                let p_out = out_ptr.add(base + c).cast::<[f64; 4]>();
                                let v_out = f64x4::new(core::ptr::read_unaligned(p_out));
                                let p_vals = vals.as_ptr().add(c).cast::<[f64; 4]>();
                                let v_vals = f64x4::new(core::ptr::read_unaligned(p_vals));
                                let r = v_out + v_vals;
                                core::ptr::write_unaligned(p_out, r.to_array());
                            }
                            c += 4;
                        }
                        while c < stripe_len {
                            unsafe {
                                *out_ptr.add(base + c) += vals[c];
                            }
                            c += 1;
                        }
                    } else {
                        for &off in touched {
                            unsafe {
                                *out_ptr.add(base + off) += vals[off];
                            }
                        }
                    }
                }
            }
        });
        out
    }
}
