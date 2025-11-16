#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
use crate::convert::csr_to_csc_f64_i64;
use crate::util::i64_to_usize;
use lacuna_core::{Coo, Csc, Csr};
use rayon::prelude::*;
use wide::f64x4;

#[must_use]
pub fn spmm_auto_f64_i64(a: &Csr<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nnz = a.data.len();
    let ncols = a.ncols;
    if ncols == 0 || a.nrows == 0 || k == 0 || nnz == 0 {
        return vec![0.0; a.nrows * k];
    }
    let avg_col_nnz = nnz / ncols.max(1);
    // Heuristic: prefer CSC when k large or column reuse is high
    if k >= 128 || (k >= 64 && avg_col_nnz >= 8) {
        let a_csc: Csc<f64, i64> = csr_to_csc_f64_i64(a);
        spmm_csc_f64_i64(&a_csc, b, k)
    } else {
        spmm_f64_i64(a, b, k)
    }
}
/// Y = A @ B, where B is (ncols, k) row-major; returns Y as (nrows, k) row-major
#[must_use]
pub fn spmm_f64_i64(a: &Csr<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut y = vec![0.0f64; nrows * k];

    // Process per row in parallel; within row, use SIMD across k with tiling.
    y.par_chunks_mut(k).enumerate().for_each(|(i, yi)| {
        let start = i64_to_usize(a.indptr[i]);
        let end = i64_to_usize(a.indptr[i + 1]);
        let _ = ncols;

        let nnz_row = end - start;
        let tile = 128usize;
        for c0 in (0..k).step_by(tile) {
            let c1 = (c0 + tile).min(k);
            let tk = c1 - c0;
            let limit4 = tk & !3;
            if nnz_row <= 8 && k >= 128 {
                for p in start..end {
                    let j = i64_to_usize(a.indices[p]);
                    let aij = a.data[p];
                    let baseb = j * k + c0;
                    let aijv = f64x4::splat(aij);
                    let mut c = 0usize;
                    while c < limit4 {
                        let vb = unsafe {
                            let q = b.as_ptr().add(baseb + c).cast::<[f64; 4]>();
                            f64x4::new(core::ptr::read_unaligned(q))
                        };
                        let vy = unsafe {
                            let q = yi.as_ptr().add(c0 + c).cast::<[f64; 4]>();
                            f64x4::new(core::ptr::read_unaligned(q))
                        };
                        let r = vy + vb * aijv;
                        unsafe {
                            let q = yi.as_mut_ptr().add(c0 + c).cast::<[f64; 4]>();
                            core::ptr::write_unaligned(q, r.to_array());
                        }
                        c += 4;
                    }
                    while c < tk {
                        yi[c0 + c] += aij * unsafe { *b.as_ptr().add(baseb + c) };
                        c += 1;
                    }
                }
            } else {
                let mut c = 0usize;
                while c < limit4 {
                    let mut acc = f64x4::splat(0.0);
                    for p in start..end {
                        let j = i64_to_usize(a.indices[p]);
                        let aijv = f64x4::splat(a.data[p]);
                        let base = j * k + c0 + c;
                        let vb = unsafe {
                            let q = b.as_ptr().add(base).cast::<[f64; 4]>();
                            f64x4::new(core::ptr::read_unaligned(q))
                        };
                        acc += vb * aijv;
                    }
                    unsafe {
                        let q = yi.as_mut_ptr().add(c0 + c).cast::<[f64; 4]>();
                        core::ptr::write_unaligned(q, acc.to_array());
                    }
                    c += 4;
                }
                while c < tk {
                    let mut acc = 0.0f64;
                    for p in start..end {
                        let j = i64_to_usize(a.indices[p]);
                        let aij = a.data[p];
                        let base = j * k + c0 + c;
                        acc += aij * unsafe { *b.as_ptr().add(base) };
                    }
                    yi[c0 + c] = acc;
                    c += 1;
                }
            }
        }
    });
    y
}

/// Y = A @ B for CSC A, B is (ncols, k) row-major
#[must_use]
pub fn spmm_csc_f64_i64(a: &Csc<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nrows = a.nrows;
    let ncols = a.ncols;
    if nrows == 0 || ncols == 0 || k == 0 {
        return vec![0.0; nrows * k];
    }
    let tile = 128usize;
    let mut y = vec![0.0f64; nrows * k];
    let y_addr = y.as_mut_ptr() as usize;
    (0..k)
        .step_by(tile)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|c0| {
            let c1 = (c0 + tile).min(k);
            let tk = c1 - c0;
            let limit4 = tk & !3;
            let y_ptr = y_addr as *mut f64;
            for j in 0..ncols {
                let s = i64_to_usize(a.indptr[j]);
                let e = i64_to_usize(a.indptr[j + 1]);
                let bj_ptr = unsafe { b.as_ptr().add(j * k + c0) };
                for p in s..e {
                    let i = i64_to_usize(a.indices[p]);
                    let aij = a.data[p];
                    let basey = i * k + c0;
                    let mut c = 0usize;
                    while c < limit4 {
                        let vb = unsafe {
                            let q = bj_ptr.add(c).cast::<[f64; 4]>();
                            f64x4::new(core::ptr::read_unaligned(q))
                        };
                        let va = f64x4::splat(aij);
                        let vy = unsafe {
                            let q = y_ptr.add(basey + c).cast::<[f64; 4]>();
                            f64x4::new(core::ptr::read_unaligned(q))
                        };
                        let r = vy + vb * va;
                        unsafe {
                            let q = y_ptr.add(basey + c).cast::<[f64; 4]>();
                            core::ptr::write_unaligned(q, r.to_array());
                        }
                        c += 4;
                    }
                    while c < tk {
                        let bjc = unsafe { *bj_ptr.add(c) };
                        unsafe {
                            *y_ptr.add(basey + c) += aij * bjc;
                        }
                        c += 1;
                    }
                }
            }
        });
    y
}

/// Y = A @ B for COO A, B is (ncols, k) row-major
#[must_use]
pub fn spmm_coo_f64_i64(a: &Coo<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nrows = a.nrows;
    let nnz = a.data.len();
    if nrows == 0 || k == 0 || nnz == 0 {
        return vec![0.0; nrows * k];
    }
    let tile = 128usize;
    let mut y = vec![0.0f64; nrows * k];
    let y_addr = y.as_mut_ptr() as usize;
    (0..k)
        .step_by(tile)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|c0| {
            let c1 = (c0 + tile).min(k);
            let tk = c1 - c0;
            let limit4 = tk & !3;
            let y_ptr = y_addr as *mut f64;
            for p in 0..nnz {
                let i = i64_to_usize(a.row[p]);
                let j = i64_to_usize(a.col[p]);
                let aij = a.data[p];
                let dst_base = i * k + c0;
                let src_base = j * k + c0;
                let mut c = 0usize;
                while c < limit4 {
                    let vb = unsafe {
                        let q = b.as_ptr().add(src_base + c).cast::<[f64; 4]>();
                        f64x4::new(core::ptr::read_unaligned(q))
                    };
                    let va = f64x4::splat(aij);
                    let vy = unsafe {
                        let q = y_ptr.add(dst_base + c).cast::<[f64; 4]>();
                        f64x4::new(core::ptr::read_unaligned(q))
                    };
                    let r = vy + vb * va;
                    unsafe {
                        let q = y_ptr.add(dst_base + c).cast::<[f64; 4]>();
                        core::ptr::write_unaligned(q, r.to_array());
                    }
                    c += 4;
                }
                while c < tk {
                    unsafe { *y_ptr.add(dst_base + c) += aij * *b.as_ptr().add(src_base + c) };
                    c += 1;
                }
            }
        });
    y
}
