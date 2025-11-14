use lacuna_core::Csr;
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
    let nrows_t = a.ncols; // rows of transposed
    let ncols_t = a.nrows; // cols of transposed
    let nnz = a.data.len();

    // Heuristic: if per-tile counts matrix would be too large, fallback to simple method
    // Estimate tile_count and memory usage
    let threads = rayon::current_num_threads().max(1);
    let mut tile_count = (threads * 4).min(a.nrows.max(1));
    if tile_count == 0 {
        tile_count = 1;
    }
    let mem_est = (tile_count as u128) * (a.ncols as u128) * (std::mem::size_of::<usize>() as u128);
    // cap ~256MB
    if mem_est > 256u128 * 1024 * 1024 {
        // fallback to atomic+sort implementation
        let mut indptr = vec![0i64; nrows_t + 1];
        for &j in &a.indices {
            indptr[i64_to_usize(j) + 1] += 1;
        }
        for i in 0..nrows_t {
            indptr[i + 1] += indptr[i];
        }
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        let pi_addr = indices.as_mut_ptr() as usize;
        let pv_addr = data.as_mut_ptr() as usize;
        let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();
        (0..a.nrows).into_par_iter().for_each(move |i| {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                let dst = i64_to_usize(next[j].fetch_add(1, Ordering::Relaxed));
                unsafe {
                    let pi = pi_addr as *mut i64;
                    let pv = pv_addr as *mut f64;
                    std::ptr::write(pi.add(dst), usize_to_i64(i));
                    std::ptr::write(pv.add(dst), a.data[p]);
                }
            }
        });
        let piw_addr = indices.as_mut_ptr() as usize;
        let pvw_addr = data.as_mut_ptr() as usize;
        let pir_addr = indices.as_ptr() as usize;
        let pvr_addr = data.as_ptr() as usize;
        let indptr_addr = indptr.as_ptr() as usize;
        (0..nrows_t).into_par_iter().for_each(move |j| {
            let s = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j) });
            let e = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j + 1) });
            if e > s {
                let mut pairs: Vec<(i64, f64)> = {
                    let pir = pir_addr as *const i64;
                    let pvr = pvr_addr as *const f64;
                    (s..e)
                        .map(|t| unsafe { (*pir.add(t), *pvr.add(t)) })
                        .collect()
                };
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                unsafe {
                    let piw = piw_addr as *mut i64;
                    let pvw = pvw_addr as *mut f64;
                    for (k, (ii, vv)) in pairs.into_iter().enumerate() {
                        std::ptr::write(piw.add(s + k), ii);
                        std::ptr::write(pvw.add(s + k), vv);
                    }
                }
            }
        });
        return Csr::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data);
    }

    // Block-partitioned prefix method
    let tiles = tile_count;
    let tile_rows = (a.nrows + tiles - 1) / tiles;

    // 1) Per-tile counts for each target row (i.e., original column)
    let counts: Vec<Vec<usize>> = (0..tiles)
        .into_par_iter()
        .map(|t| {
            let start = t * tile_rows;
            let end = (start + tile_rows).min(a.nrows);
            let mut c = vec![0usize; a.ncols];
            if start < end {
                for i in start..end {
                    let s = i64_to_usize(a.indptr[i]);
                    let e = i64_to_usize(a.indptr[i + 1]);
                    for p in s..e {
                        let j = i64_to_usize(a.indices[p]);
                        c[j] += 1;
                    }
                }
            }
            c
        })
        .collect();

    // 2) Build global indptr (row pointers of transposed) and per-tile offsets
    let mut indptr = vec![0i64; nrows_t + 1];
    for j in 0..a.ncols {
        let mut sum = 0usize;
        for t in 0..tiles {
            sum += counts[t][j];
        }
        indptr[j + 1] = indptr[j] + usize_to_i64(sum);
    }

    // Per-tile starting offsets for each transposed row j
    let mut offsets: Vec<Vec<usize>> = vec![vec![0usize; a.ncols]; tiles];
    for j in 0..a.ncols {
        let base = i64_to_usize(indptr[j]);
        let mut cur = base;
        for t in 0..tiles {
            offsets[t][j] = cur;
            cur += counts[t][j];
        }
    }

    // 3) Fill output using per-tile disjoint segments (no atomics, naturally sorted)
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
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                let dst = pos[j];
                pos[j] = dst + 1;
                unsafe {
                    let pi = pi_addr as *mut i64;
                    let pv = pv_addr as *mut f64;
                    std::ptr::write(pi.add(dst), usize_to_i64(i));
                    std::ptr::write(pv.add(dst), a.data[p]);
                }
            }
        }
    });

    Csr::from_parts_unchecked(nrows_t, ncols_t, indptr, indices, data)
}
