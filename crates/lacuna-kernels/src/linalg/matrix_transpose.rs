//! Matrix transpose kernels for CSR, CSC, and COO formats.
//!
//! Implements:
//! - **CSR transpose** (CSR → CSR): Uses adaptive strategy (per-thread histograms or atomic fallback)
//! - **CSC transpose** (CSC → CSC): Uses adaptive strategy (per-thread histograms or atomic fallback)
//! - **COO transpose** (COO → COO): Simple coordinate swap
//!
//! Adaptive strategy:
//! - **Strategy A**: Per-thread histograms when memory budget permits (~512MB)
//! - **Strategy B**: Atomic next pointers with per-row/column sorting fallback
//!
//! Both strategies use 4-way loop unrolling for throughput.

use lacuna_core::{Coo, Csc, Csr};
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

/// Converts i64 to usize with debug assertions for non-negative values.
#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0, "value must be non-negative");
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// Converts usize to i64 with debug assertions for range validity.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok(), "value must fit in i64");
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Transposes a CSC matrix: A (nrows × ncols) → A^T (ncols × nrows) in CSC format.
///
/// Uses adaptive strategy based on memory availability:
/// - **Strategy A (per-thread histograms)** when memory cost ≤ ~512MB:
///   * Partition columns into ranges (one per thread)
///   * Each thread computes histogram of row indices for its columns
///   * Global indptr computed via reduction of histograms
///   * Per-thread offsets allow lock-free fill phase
///   * Result naturally sorted within each column (no post-sort needed)
///
/// - **Strategy B (atomic fallback)** otherwise:
///   * Count row indices globally (sequential)
///   * Build indptr
///   * Parallel fill using atomic next pointers to assign destinations
///   * Per-column sorting to restore strictly increasing row indices
///
/// # Optimization
/// - 4-way loop unrolling in both strategies
/// - Bounds-checking elimination via unsafe
/// - Lock-free writes with per-thread histograms (Strategy A)
///
/// # Arguments
/// * `a` - Input CSC matrix (nrows × ncols)
///
/// # Returns
/// Transposed CSC matrix (ncols × nrows) with column-major storage preserved
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

/// Transposes a COO matrix: A (nrows × ncols) → A^T (ncols × nrows) in COO format.
///
/// Simple coordinate-wise transposition by swapping row and column indices.
/// Data values are unchanged. Shape is swapped.
///
/// # Time Complexity
/// O(nnz) for cloning row, col, and data arrays.
///
/// # Arguments
/// * `a` - Input COO matrix (nrows × ncols)
///
/// # Returns
/// Transposed COO matrix (ncols × nrows)
#[must_use]
pub fn transpose_coo_f64_i64(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    // Simple swap: row' = col, col' = row; data cloned; shape swapped
    let row = a.col.clone();
    let col = a.row.clone();
    let data = a.data.clone();
    Coo::from_parts_unchecked(a.ncols, a.nrows, row, col, data)
}

/// Transposes a CSR matrix: A (nrows × ncols) → A^T (ncols × nrows) in CSR format.
///
/// Uses adaptive strategy based on memory availability:
/// - **Strategy A (per-thread histograms)** when memory cost ≤ ~512MB:
///   * Partition rows into ranges (one per thread)
///   * Each thread computes histogram of column indices for its rows
///   * Global indptr computed via reduction of histograms
///   * Per-thread offsets allow lock-free fill phase
///   * Result naturally sorted within each row (no post-sort needed)
///
/// - **Strategy B (atomic fallback)** otherwise:
///   * Count column indices globally (sequential)
///   * Build indptr
///   * Parallel fill using atomic next pointers to assign destinations
///   * Per-row sorting to restore strictly increasing column indices
///
/// # Optimization
/// - 4-way loop unrolling in both strategies
/// - Bounds-checking elimination via unsafe
/// - Lock-free writes with per-thread histograms (Strategy A)
///
/// # Arguments
/// * `a` - Input CSR matrix (nrows × ncols)
///
/// # Returns
/// Transposed CSR matrix (ncols × nrows) with row-major storage preserved
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
