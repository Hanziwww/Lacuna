#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
use lacuna_core::{Coo, Csc, Csr, CooNd};
use crate::util::SMALL_NNZ_LIMIT;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Convert CSR -> CSC (f64/i64)
#[must_use]
pub fn csr_to_csc_f64_i64(a: &Csr<f64, i64>) -> Csc<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();

    // Count per column
    let mut indptr = vec![0i64; ncols + 1];
    for &j in &a.indices {
        indptr[i64_to_usize(j) + 1] += 1;
    }
    for j in 0..ncols {
        indptr[j + 1] += indptr[j];
    }

    let mut indices = vec![0i64; nnz]; // row indices
    let mut data = vec![0.0f64; nnz];

    // Atomic next pointers initialized to indptr
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();

    let indices_addr = indices.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        for p in s..e {
            let j = i64_to_usize(a.indices[p]);
            let dst = i64_to_usize(next[j].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = indices_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(i));
                std::ptr::write(pv.add(dst), a.data[p]);
            }
        }
    });

    Csc::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Convert CSC -> CSR (f64/i64)
#[must_use]
pub fn csc_to_csr_f64_i64(a: &Csc<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();

    // Count per row (from CSC row indices)
    let mut indptr = vec![0i64; nrows + 1];
    for &i in &a.indices {
        indptr[i64_to_usize(i) + 1] += 1;
    }
    for r in 0..nrows {
        indptr[r + 1] += indptr[r];
    }

    let mut indices = vec![0i64; nnz]; // col indices
    let mut data = vec![0.0f64; nnz];

    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();

    let indices_addr = indices.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        for p in s..e {
            let i = i64_to_usize(a.indices[p]);
            let dst = i64_to_usize(next[i].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = indices_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(j));
                std::ptr::write(pv.add(dst), a.data[p]);
            }
        }
    });

    Csr::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Convert CSR -> COO (f64/i64)
#[must_use]
pub fn csr_to_coo_f64_i64(a: &Csr<f64, i64>) -> Coo<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();

    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];

    let row_addr = row.as_mut_ptr() as usize;
    let col_addr = col.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let mut dst = s;
        for p in s..e {
            unsafe {
                let pr = row_addr as *mut i64;
                let pc = col_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pr.add(dst), usize_to_i64(i));
                std::ptr::write(pc.add(dst), a.indices[p]);
                std::ptr::write(pv.add(dst), a.data[p]);
            }
            dst += 1;
        }
    });

    Coo::from_parts_unchecked(nrows, ncols, row, col, data)
}

/// Convert CSC -> COO (f64/i64)
#[must_use]
pub fn csc_to_coo_f64_i64(a: &Csc<f64, i64>) -> Coo<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();

    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];

    let row_addr = row.as_mut_ptr() as usize;
    let col_addr = col.as_mut_ptr() as usize;
    let data_addr = data.as_mut_ptr() as usize;
    (0..ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        for p in s..e {
            unsafe {
                let pr = row_addr as *mut i64;
                let pc = col_addr as *mut i64;
                let pv = data_addr as *mut f64;
                std::ptr::write(pr.add(p), a.indices[p]);
                std::ptr::write(pc.add(p), usize_to_i64(j));
                std::ptr::write(pv.add(p), a.data[p]);
            }
        }
    });

    Coo::from_parts_unchecked(nrows, ncols, row, col, data)
}

/// Convert COO -> CSR, summing duplicates and sorting columns within rows.
#[must_use]
pub fn coo_to_csr_f64_i64(a: &Coo<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let nnz = a.data.len();

    // Gather triples and sort by (row, col)
    let mut triples: Vec<(i64, i64, f64)> =
        (0..nnz).map(|k| (a.row[k], a.col[k], a.data[k])).collect();
    triples.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));

    // Build indptr
    let mut indptr = vec![0i64; nrows + 1];
    let mut indices: Vec<i64> = Vec::with_capacity(nnz);
    let mut data: Vec<f64> = Vec::with_capacity(nnz);

    let mut cur_row: i64 = -1;
    let mut last_col: i64 = -1;
    let mut acc: f64 = 0.0;

    for (r, c, v) in triples {
        if r != cur_row {
            // flush previous entry
            if cur_row >= 0 && last_col >= 0 {
                indices.push(last_col);
                data.push(acc);
            }
            // advance indptr up to current row
            let r_us = i64_to_usize(r);
            let prev_us = i64_to_usize((cur_row + 1).max(0));
            let len_i64 = i64::try_from(indices.len()).unwrap_or(i64::MAX);
            for ptr in indptr.iter_mut().take(r_us + 1).skip(prev_us) {
                *ptr = len_i64;
            }
            cur_row = r;
            last_col = c;
            acc = v;
            continue;
        }
        // same row
        if c == last_col {
            acc += v; // duplicate
        } else {
            // flush previous column
            indices.push(last_col);
            data.push(acc);
            last_col = c;
            acc = v;
        }
    }
    // flush final
    if cur_row >= 0 && last_col >= 0 {
        indices.push(last_col);
        data.push(acc);
    }
    // fill remaining indptr
    let start_row = if cur_row < 0 {
        0
    } else {
        i64_to_usize(cur_row + 1)
    };
    let len_i64 = i64::try_from(indices.len()).unwrap_or(i64::MAX);
    for ptr in indptr.iter_mut().take(nrows + 1).skip(start_row) {
        *ptr = len_i64;
    }

    Csr::from_parts_unchecked(nrows, ncols, indptr, indices, data)
}

/// Convert COO -> CSC, summing duplicates and sorting rows within columns.
#[must_use]
pub fn coo_to_csc_f64_i64(a: &Coo<f64, i64>) -> Csc<f64, i64> {
    let coo_t = Coo::from_parts_unchecked(
        a.ncols,
        a.nrows,
        a.col.clone(),
        a.row.clone(),
        a.data.clone(),
    );
    let csr_t = coo_to_csr_f64_i64(&coo_t);
    // reinterpret as CSC of original shape
    Csc::from_parts_unchecked(a.nrows, a.ncols, csr_t.indptr, csr_t.indices, csr_t.data)
}

// ----- CooNd -> 2D conversions -----

#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc
            .checked_mul(x)
            .expect("shape product overflow");
    }
    acc
}

#[inline]
fn build_strides_row_major(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let n = dims.len();
    let mut strides = vec![0usize; n];
    strides[n - 1] = 1;
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1])
            .expect("shape product overflow");
    }
    strides
}

fn coond_axes_to_coo_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Coo<f64, i64> {
    let ndim = a.shape.len();
    // validate axes
    let mut used = vec![false; ndim];
    for &ax in row_axes {
        assert!(ax < ndim, "row axis out of bounds");
        assert!(!used[ax], "duplicate axis in row_axes");
        used[ax] = true;
    }
    let mut col_axes: Vec<usize> = Vec::with_capacity(ndim - row_axes.len());
    for d in 0..ndim {
        if !used[d] {
            col_axes.push(d);
        }
    }

    let row_shape: Vec<usize> = row_axes.iter().map(|&d| a.shape[d]).collect();
    let col_shape: Vec<usize> = col_axes.iter().map(|&d| a.shape[d]).collect();
    let nrows = product_checked(&row_shape);
    let ncols = product_checked(&col_shape);
    let row_strides = build_strides_row_major(&row_shape);
    let col_strides = build_strides_row_major(&col_shape);

    let nnz = a.data.len();
    if nnz == 0 {
        return Coo::from_parts_unchecked(nrows, ncols, Vec::new(), Vec::new(), Vec::new());
    }

    let mut row = vec![0i64; nnz];
    let mut col = vec![0i64; nnz];
    if nnz < SMALL_NNZ_LIMIT {
        for k in 0..nnz {
            let base = k * ndim;
            let mut r: usize = 0;
            for (m, &d) in row_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base + d) });
                let s = if row_strides.is_empty() { 0 } else { row_strides[m] };
                r = r
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            let mut c2: usize = 0;
            for (m, &d) in col_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base + d) });
                let s = if col_strides.is_empty() { 0 } else { col_strides[m] };
                c2 = c2
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            row[k] = usize_to_i64(r);
            col[k] = usize_to_i64(c2);
        }
    } else {
        let row_addr = row.as_mut_ptr() as usize;
        let col_addr = col.as_mut_ptr() as usize;
        let indices = &a.indices;
        (0..nnz).into_par_iter().for_each(|k| {
            let base = k * ndim;
            let mut r: usize = 0;
            for (m, &d) in row_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *indices.get_unchecked(base + d) });
                let s = if row_strides.is_empty() { 0 } else { row_strides[m] };
                r = r
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            let mut c2: usize = 0;
            for (m, &d) in col_axes.iter().enumerate() {
                let idx = i64_to_usize(unsafe { *indices.get_unchecked(base + d) });
                let s = if col_strides.is_empty() { 0 } else { col_strides[m] };
                c2 = c2
                    .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                    .expect("linear index overflow");
            }
            unsafe {
                *(row_addr as *mut i64).add(k) = usize_to_i64(r);
                *(col_addr as *mut i64).add(k) = usize_to_i64(c2);
            }
        });
    }

    Coo::from_parts_unchecked(nrows, ncols, row, col, a.data.clone())
}

#[must_use]
pub fn coond_axes_to_csr_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Csr<f64, i64> {
    let coo = coond_axes_to_coo_f64_i64(a, row_axes);
    coo_to_csr_f64_i64(&coo)
}

#[must_use]
pub fn coond_axes_to_csc_f64_i64(a: &CooNd<f64, i64>, row_axes: &[usize]) -> Csc<f64, i64> {
    let coo = coond_axes_to_coo_f64_i64(a, row_axes);
    coo_to_csc_f64_i64(&coo)
}

#[must_use]
pub fn coond_mode_to_csr_f64_i64(a: &CooNd<f64, i64>, axis: usize) -> Csr<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    let row_axes = [axis];
    coond_axes_to_csr_f64_i64(a, &row_axes)
}

#[must_use]
pub fn coond_mode_to_csc_f64_i64(a: &CooNd<f64, i64>, axis: usize) -> Csc<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    let row_axes = [axis];
    coond_axes_to_csc_f64_i64(a, &row_axes)
}
