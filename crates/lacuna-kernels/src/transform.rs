use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0, "value must be non-negative");
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}
#[inline]
fn product_checked(dims: &[usize]) -> usize {
    let mut acc: usize = 1;
    for &x in dims {
        acc = acc.checked_mul(x).expect("shape product overflow");
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

#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn reshape_coond_f64_i64(a: &CooNd<f64, i64>, new_shape: &[usize]) -> CooNd<f64, i64> {
    let old_elems = product_checked(&a.shape);
    let new_elems = product_checked(new_shape);
    assert_eq!(
        old_elems, new_elems,
        "reshape requires same number of elements"
    );
    let ndim_old = a.shape.len();
    let ndim_new = new_shape.len();
    let nnz = a.data.len();
    if nnz == 0 {
        return CooNd::from_parts_unchecked(new_shape.to_vec(), Vec::new(), Vec::new());
    }
    let old_strides = build_strides_row_major(&a.shape);
    let new_strides = build_strides_row_major(new_shape);
    let mut out_indices = vec![0i64; nnz * ndim_new];
    let out_ptr = out_indices.as_mut_ptr() as usize;
    (0..nnz).into_par_iter().for_each(|k| {
        let base_old = k * ndim_old;
        let mut lin: usize = 0;
        for d in 0..ndim_old {
            let idx = i64_to_usize(unsafe { *a.indices.get_unchecked(base_old + d) });
            let s = old_strides[d];
            lin = lin
                .checked_add(idx.checked_mul(s).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        // de-linearize into new shape
        let base_new = k * ndim_new;
        let mut rem = lin;
        for d in 0..ndim_new {
            let s = new_strides[d];
            let idx = if s == 0 { 0 } else { rem / s };
            rem -= idx * s;
            unsafe {
                let p = out_ptr as *mut i64;
                std::ptr::write(p.add(base_new + d), usize_to_i64(idx));
            }
        }
    });
    CooNd::from_parts_unchecked(new_shape.to_vec(), out_indices, a.data.clone())
}
/// Transpose CSC -> CSC (parallel histogram-based)
#[allow(
    clippy::similar_names,
    reason = "Pointer aliases (pi/pv/pir/pvr etc.) are intentionally similar in low-level kernels"
)]
#[allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use short names like i/j/k/s/e/p/j"
)]
#[allow(
    clippy::manual_div_ceil,
    reason = "Use of (a + b - 1) / b is intentional for broad compiler compatibility"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "Index-based loops keep CSC math clear and efficient"
)]
#[allow(
    clippy::too_many_lines,
    reason = "Transpose kernel is intentionally monolithic for clarity and performance"
)]
pub fn transpose_csc_f64_i64(a: &Csc<f64, i64>) -> Csc<f64, i64> {
    let nrows_t = a.ncols; // rows of transposed (A^T)
    let ncols_t = a.nrows; // cols of transposed (A^T)
    let nnz = a.data.len();

    // Heuristic for per-thread histograms
    let tiles = rayon::current_num_threads().max(1);
    let elem_size = std::mem::size_of::<i64>() as u128;
    // counts + offsets + pos per thread, sized by target columns (ncols_t)
    let mem_est = (tiles as u128) * (ncols_t as u128) * elem_size * 3u128;
    let mem_cap = 512u128 * 1024 * 1024; // ~512MB

    if mem_est <= mem_cap {
        // Strategy A: per-thread histograms over column tiles
        let tile_cols = (a.ncols + tiles - 1) / tiles;

        // 1) Per-thread counts per target column (original row i)
        let counts: Vec<Vec<i64>> = (0..tiles)
            .into_par_iter()
            .map(|t| {
                let start = t * tile_cols;
                let end = (start + tile_cols).min(a.ncols);
                let mut c = vec![0i64; ncols_t];
                if start < end {
                    for j in start..end {
                        let s = i64_to_usize(a.indptr[j]);
                        let e = i64_to_usize(a.indptr[j + 1]);
                        let mut p = s;
                        let end4 = e - ((e - p) & 3);
                        while p < end4 {
                            let i0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                            let i1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
                            let i2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
                            let i3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });
                            unsafe {
                                *c.as_mut_ptr().add(i0) += 1;
                                *c.as_mut_ptr().add(i1) += 1;
                                *c.as_mut_ptr().add(i2) += 1;
                                *c.as_mut_ptr().add(i3) += 1;
                            }
                            p += 4;
                        }
                        while p < e {
                            let i = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                            unsafe {
                                *c.as_mut_ptr().add(i) += 1;
                            }
                            p += 1;
                        }
                    }
                }
                c
            })
            .collect();

        // 2) Global indptr over target columns
        let mut indptr = vec![0i64; ncols_t + 1];
        for i in 0..ncols_t {
            let mut sum = 0i64;
            for t in 0..tiles {
                sum += counts[t][i];
            }
            indptr[i + 1] = indptr[i] + sum;
        }

        // 3) Per-tile offsets per target column
        let mut offsets: Vec<Vec<i64>> = vec![vec![0i64; ncols_t]; tiles];
        for i in 0..ncols_t {
            let base = indptr[i];
            let mut cur = base;
            for t in 0..tiles {
                offsets[t][i] = cur;
                cur += counts[t][i];
            }
        }

        // 4) Fill in parallel; for each entry (i,j,v) write to column i with row=j
        let mut indices = vec![0i64; nnz]; // B row indices
        let mut data = vec![0.0f64; nnz];
        let pi_addr = indices.as_mut_ptr() as usize;
        let pv_addr = data.as_mut_ptr() as usize;

        (0..tiles).into_par_iter().for_each(|t| {
            let start = t * tile_cols;
            let end = (start + tile_cols).min(a.ncols);
            if start >= end {
                return;
            }
            let mut pos = offsets[t].clone();
            for j in start..end {
                let s = i64_to_usize(a.indptr[j]);
                let e = i64_to_usize(a.indptr[j + 1]);
                let mut p = s;
                let end4 = e - ((e - p) & 3);
                while p < end4 {
                    let i0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                    let i1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
                    let i2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
                    let i3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });

                    let dst0 = i64_to_usize(unsafe { *pos.as_ptr().add(i0) });
                    let dst1 = i64_to_usize(unsafe { *pos.as_ptr().add(i1) });
                    let dst2 = i64_to_usize(unsafe { *pos.as_ptr().add(i2) });
                    let dst3 = i64_to_usize(unsafe { *pos.as_ptr().add(i3) });

                    unsafe {
                        *pos.as_mut_ptr().add(i0) = usize_to_i64(dst0) + 1;
                        *pos.as_mut_ptr().add(i1) = usize_to_i64(dst1) + 1;
                        *pos.as_mut_ptr().add(i2) = usize_to_i64(dst2) + 1;
                        *pos.as_mut_ptr().add(i3) = usize_to_i64(dst3) + 1;

                        let pi = pi_addr as *mut i64;
                        let pv = pv_addr as *mut f64;
                        std::ptr::write(pi.add(dst0), usize_to_i64(j));
                        std::ptr::write(pv.add(dst0), *a.data.get_unchecked(p));
                        std::ptr::write(pi.add(dst1), usize_to_i64(j));
                        std::ptr::write(pv.add(dst1), *a.data.get_unchecked(p + 1));
                        std::ptr::write(pi.add(dst2), usize_to_i64(j));
                        std::ptr::write(pv.add(dst2), *a.data.get_unchecked(p + 2));
                        std::ptr::write(pi.add(dst3), usize_to_i64(j));
                        std::ptr::write(pv.add(dst3), *a.data.get_unchecked(p + 3));
                    }
                    p += 4;
                }
                while p < e {
                    let i = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                    let dst = i64_to_usize(unsafe { *pos.as_ptr().add(i) });
                    unsafe {
                        *pos.as_mut_ptr().add(i) = usize_to_i64(dst) + 1;
                        let pi = pi_addr as *mut i64;
                        let pv = pv_addr as *mut f64;
                        std::ptr::write(pi.add(dst), usize_to_i64(j));
                        std::ptr::write(pv.add(dst), *a.data.get_unchecked(p));
                    }
                    p += 1;
                }
            }
        });

        return Csc::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data);
    }

    // Strategy B: Atomic next pointers + per-column sorting
    let mut indptr = vec![0i64; ncols_t + 1];
    for &i in &a.indices {
        indptr[i64_to_usize(i) + 1] += 1;
    }
    for c in 0..ncols_t {
        indptr[c + 1] += indptr[c];
    }

    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();

    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    (0..a.ncols).into_par_iter().for_each(|j| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let mut p = s;
        let end4 = e - ((e - p) & 3);
        while p < end4 {
            let i0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
            let i1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
            let i2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
            let i3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });

            let dst0 = i64_to_usize(next[i0].fetch_add(1, Ordering::Relaxed));
            let dst1 = i64_to_usize(next[i1].fetch_add(1, Ordering::Relaxed));
            let dst2 = i64_to_usize(next[i2].fetch_add(1, Ordering::Relaxed));
            let dst3 = i64_to_usize(next[i3].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(dst0), usize_to_i64(j));
                std::ptr::write(pv.add(dst0), *a.data.get_unchecked(p));
                std::ptr::write(pi.add(dst1), usize_to_i64(j));
                std::ptr::write(pv.add(dst1), *a.data.get_unchecked(p + 1));
                std::ptr::write(pi.add(dst2), usize_to_i64(j));
                std::ptr::write(pv.add(dst2), *a.data.get_unchecked(p + 2));
                std::ptr::write(pi.add(dst3), usize_to_i64(j));
                std::ptr::write(pv.add(dst3), *a.data.get_unchecked(p + 3));
            }
            p += 4;
        }
        while p < e {
            let i = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
            let dst = i64_to_usize(next[i].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(j));
                std::ptr::write(pv.add(dst), *a.data.get_unchecked(p));
            }
            p += 1;
        }
    });

    // Sort rows within each column
    let pir_addr = indices.as_ptr() as usize;
    let pvr_addr = data.as_ptr() as usize;
    let piw_addr = indices.as_mut_ptr() as usize;
    let pvw_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    (0..ncols_t).into_par_iter().for_each(|c| {
        let s = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(c) });
        let e = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(c + 1) });
        if e <= s + 1 {
            return;
        }
        let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(e - s);
        for t in s..e {
            unsafe {
                let r = *(pir_addr as *const i64).add(t);
                let v = *(pvr_addr as *const f64).add(t);
                pairs.push((r, v));
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (k, (r, v)) in pairs.into_iter().enumerate() {
            unsafe {
                *(piw_addr as *mut i64).add(s + k) = r;
                *(pvw_addr as *mut f64).add(s + k) = v;
            }
        }
    });

    Csc::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data)
}

/// Transpose COO -> COO (swap row/col)
#[must_use]
pub fn transpose_coo_f64_i64(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    // Simple swap: row' = col, col' = row; data cloned; shape swapped
    let row = a.col.clone();
    let col = a.row.clone();
    let data = a.data.clone();
    Coo::from_parts_unchecked(a.ncols, a.nrows, row, col, data)
}

#[must_use]
pub fn permute_axes_coond_f64_i64(a: &CooNd<f64, i64>, perm: &[usize]) -> CooNd<f64, i64> {
    let ndim = a.shape.len();
    assert_eq!(perm.len(), ndim, "perm length must equal ndim");
    let mut seen = vec![false; ndim];
    for &p in perm {
        assert!(p < ndim, "perm index out of bounds");
        assert!(!seen[p], "perm must be a permutation without duplicates");
        seen[p] = true;
    }

    let new_shape: Vec<usize> = (0..ndim).map(|d| a.shape[perm[d]]).collect();
    let nnz = a.data.len();
    if nnz == 0 {
        return CooNd::from_parts_unchecked(new_shape, Vec::new(), Vec::new());
    }
    let mut new_indices = vec![0i64; nnz * ndim];
    let src = &a.indices;
    for k in 0..nnz {
        let base_src = k * ndim;
        let base_dst = base_src;
        for d in 0..ndim {
            let src_d = perm[d];
            new_indices[base_dst + d] = src[base_src + src_d];
        }
    }
    CooNd::from_parts_unchecked(new_shape, new_indices, a.data.clone())
}

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok(), "value must fit in i64");
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Transpose CSR -> CSR (simple histogram-based)
#[allow(
    clippy::similar_names,
    reason = "Pointer aliases (pi/pv/pir/pvr etc.) are intentionally similar in low-level kernels"
)]
#[allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use short names like i/j/k/s/e/p/j"
)]
#[allow(
    clippy::manual_div_ceil,
    reason = "Use of (a + b - 1) / b is intentional for broad compiler compatibility"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "Index-based loops keep CSR math clear and efficient"
)]
#[allow(
    clippy::too_many_lines,
    reason = "Transpose kernel is intentionally monolithic for clarity and performance"
)]
pub fn transpose_f64_i64(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    let nrows_t = a.ncols;
    let ncols_t = a.nrows;
    let nnz = a.data.len();

    // Heuristic: use per-thread histograms when memory is reasonable; else atomic fallback
    let tiles = rayon::current_num_threads().max(1);
    let elem_size = std::mem::size_of::<i64>() as u128;
    // counts + offsets + (approx) pos per thread
    let mem_est = (tiles as u128) * (nrows_t as u128) * elem_size * 3u128;
    let mem_cap = 512u128 * 1024 * 1024; // ~512MB

    if mem_est <= mem_cap {
        // Strategy A: fully parallel without atomics using per-thread histograms
        let tile_rows = (a.nrows + tiles - 1) / tiles;

        // 1) Per-thread counts over assigned row ranges
        let counts: Vec<Vec<i64>> = (0..tiles)
            .into_par_iter()
            .map(|t| {
                let start = t * tile_rows;
                let end = (start + tile_rows).min(a.nrows);
                let mut c = vec![0i64; nrows_t];
                if start < end {
                    for i in start..end {
                        let s = i64_to_usize(a.indptr[i]);
                        let e = i64_to_usize(a.indptr[i + 1]);
                        let mut p = s;
                        let end4 = e - ((e - p) & 3);
                        while p < end4 {
                            let j0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                            let j1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
                            let j2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
                            let j3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });
                            unsafe {
                                *c.as_mut_ptr().add(j0) += 1;
                                *c.as_mut_ptr().add(j1) += 1;
                                *c.as_mut_ptr().add(j2) += 1;
                                *c.as_mut_ptr().add(j3) += 1;
                            }
                            p += 4;
                        }
                        while p < e {
                            let j = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                            unsafe {
                                *c.as_mut_ptr().add(j) += 1;
                            }
                            p += 1;
                        }
                    }
                }
                c
            })
            .collect();

        // 2) Build global indptr by reducing counts across tiles
        let mut indptr = vec![0i64; nrows_t + 1];
        for j in 0..nrows_t {
            let mut sum = 0i64;
            for t in 0..tiles {
                sum += counts[t][j];
            }
            indptr[j + 1] = indptr[j] + sum;
        }

        // 3) Compute per-tile offsets for each transposed row
        let mut offsets: Vec<Vec<i64>> = vec![vec![0i64; nrows_t]; tiles];
        for j in 0..nrows_t {
            let base = indptr[j];
            let mut cur = base;
            for t in 0..tiles {
                offsets[t][j] = cur;
                cur += counts[t][j];
            }
        }

        // 4) Fill output in parallel, each tile writes disjoint segments (naturally sorted)
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        let pi_addr = indices.as_mut_ptr() as usize;
        let pv_addr = data.as_mut_ptr() as usize;

        (0..tiles).into_par_iter().for_each(|t| {
            let start = t * tile_rows;
            let end = (start + tile_rows).min(a.nrows);
            if start >= end {
                return;
            }
            // Local running positions per target row
            let mut pos = offsets[t].clone();
            for i in start..end {
                let s = i64_to_usize(a.indptr[i]);
                let e = i64_to_usize(a.indptr[i + 1]);
                let mut p = s;
                let end4 = e - ((e - p) & 3);
                while p < end4 {
                    let j0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                    let j1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
                    let j2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
                    let j3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });

                    let dst0 = i64_to_usize(unsafe { *pos.as_ptr().add(j0) });
                    let dst1 = i64_to_usize(unsafe { *pos.as_ptr().add(j1) });
                    let dst2 = i64_to_usize(unsafe { *pos.as_ptr().add(j2) });
                    let dst3 = i64_to_usize(unsafe { *pos.as_ptr().add(j3) });

                    unsafe {
                        // bump positions
                        *pos.as_mut_ptr().add(j0) = usize_to_i64(dst0) + 1;
                        *pos.as_mut_ptr().add(j1) = usize_to_i64(dst1) + 1;
                        *pos.as_mut_ptr().add(j2) = usize_to_i64(dst2) + 1;
                        *pos.as_mut_ptr().add(j3) = usize_to_i64(dst3) + 1;

                        let pi = pi_addr as *mut i64;
                        let pv = pv_addr as *mut f64;
                        std::ptr::write(pi.add(dst0), usize_to_i64(i));
                        std::ptr::write(pv.add(dst0), *a.data.get_unchecked(p));
                        std::ptr::write(pi.add(dst1), usize_to_i64(i));
                        std::ptr::write(pv.add(dst1), *a.data.get_unchecked(p + 1));
                        std::ptr::write(pi.add(dst2), usize_to_i64(i));
                        std::ptr::write(pv.add(dst2), *a.data.get_unchecked(p + 2));
                        std::ptr::write(pi.add(dst3), usize_to_i64(i));
                        std::ptr::write(pv.add(dst3), *a.data.get_unchecked(p + 3));
                    }
                    p += 4;
                }
                while p < e {
                    let j = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
                    let dst = i64_to_usize(unsafe { *pos.as_ptr().add(j) });
                    unsafe {
                        *pos.as_mut_ptr().add(j) = usize_to_i64(dst + 1);
                        let pi = pi_addr as *mut i64;
                        let pv = pv_addr as *mut f64;
                        std::ptr::write(pi.add(dst), usize_to_i64(i));
                        std::ptr::write(pv.add(dst), *a.data.get_unchecked(p));
                    }
                    p += 1;
                }
            }
        });

        return Csr::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data);
    }

    // Strategy B: Fallback — atomic next pointers with per-row parallel sort to restore order
    // 1) Count per target row (sequential count, negligible vs. fill contention)
    let mut indptr = vec![0i64; nrows_t + 1];
    for &j in &a.indices {
        indptr[i64_to_usize(j) + 1] += 1;
    }
    for j in 0..nrows_t {
        indptr[j + 1] += indptr[j];
    }

    // 2) Parallel fill using atomic next pointers
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();

    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    (0..a.nrows).into_par_iter().for_each(|i| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let mut p = s;
        let end4 = e - ((e - p) & 3);
        while p < end4 {
            let j0 = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
            let j1 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 1) });
            let j2 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 2) });
            let j3 = i64_to_usize(unsafe { *a.indices.get_unchecked(p + 3) });

            let dst0 = i64_to_usize(next[j0].fetch_add(1, Ordering::Relaxed));
            let dst1 = i64_to_usize(next[j1].fetch_add(1, Ordering::Relaxed));
            let dst2 = i64_to_usize(next[j2].fetch_add(1, Ordering::Relaxed));
            let dst3 = i64_to_usize(next[j3].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(dst0), usize_to_i64(i));
                std::ptr::write(pv.add(dst0), *a.data.get_unchecked(p));
                std::ptr::write(pi.add(dst1), usize_to_i64(i));
                std::ptr::write(pv.add(dst1), *a.data.get_unchecked(p + 1));
                std::ptr::write(pi.add(dst2), usize_to_i64(i));
                std::ptr::write(pv.add(dst2), *a.data.get_unchecked(p + 2));
                std::ptr::write(pi.add(dst3), usize_to_i64(i));
                std::ptr::write(pv.add(dst3), *a.data.get_unchecked(p + 3));
            }
            p += 4;
        }
        while p < e {
            let j = i64_to_usize(unsafe { *a.indices.get_unchecked(p) });
            let dst = i64_to_usize(next[j].fetch_add(1, Ordering::Relaxed));
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(dst), usize_to_i64(i));
                std::ptr::write(pv.add(dst), *a.data.get_unchecked(p));
            }
            p += 1;
        }
    });

    // 3) Per-row parallel sort to enforce strictly increasing column indices
    let pir_addr = indices.as_ptr() as usize;
    let pvr_addr = data.as_ptr() as usize;
    let piw_addr = indices.as_mut_ptr() as usize;
    let pvw_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    (0..nrows_t).into_par_iter().for_each(|j| {
        let s = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j) });
        let e = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j + 1) });
        if e <= s + 1 {
            return;
        }
        let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(e - s);
        for t in s..e {
            unsafe {
                let ii = *(pir_addr as *const i64).add(t);
                let vv = *(pvr_addr as *const f64).add(t);
                pairs.push((ii, vv));
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (k, (ii, vv)) in pairs.into_iter().enumerate() {
            unsafe {
                *(piw_addr as *mut i64).add(s + k) = ii;
                *(pvw_addr as *mut f64).add(s + k) = vv;
            }
        }
    });

    Csr::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data)
}
