#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::comparison_chain
)]

use core::cmp::Ordering;
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;

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

/// Count non-zero entries for power of two sorted rows (or columns).
/// Only positions where both have entries are considered; duplicates in either
/// input are coalesced by summation before computing the power.
#[inline]
unsafe fn pow_row_count(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
) -> usize {
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut cnt = 0usize;
    while pa < alen && pb < blen {
        let ja = unsafe { *ai.add(pa) };
        let jb = unsafe { *bi.add(pb) };
        match ja.cmp(&jb) {
            Ordering::Equal => {
                let j = ja;
                let mut sa = 0.0f64;
                while pa < alen && unsafe { *ai.add(pa) } == j {
                    sa += unsafe { *av.add(pa) };
                    pa += 1;
                }
                let mut sb = 0.0f64;
                while pb < blen && unsafe { *bi.add(pb) } == j {
                    sb += unsafe { *bv.add(pb) };
                    pb += 1;
                }
                let v = sa.powf(sb);
                if v != 0.0 && v.is_finite() {
                    cnt += 1;
                }
            }
            Ordering::Less => {
                let j = ja;
                while pa < alen && unsafe { *ai.add(pa) } == j {
                    pa += 1;
                }
            }
            Ordering::Greater => {
                let j = jb;
                while pb < blen && unsafe { *bi.add(pb) } == j {
                    pb += 1;
                }
            }
        }
    }
    cnt
}

/// Fill power results of two rows (or columns) into output buffers.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn pow_row_fill(
    ai: *const i64,
    av: *const f64,
    alen: usize,
    bi: *const i64,
    bv: *const f64,
    blen: usize,
    out_i: *mut i64,
    out_v: *mut f64,
) -> usize {
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut dst = 0usize;
    while pa < alen && pb < blen {
        let ja = unsafe { *ai.add(pa) };
        let jb = unsafe { *bi.add(pb) };
        match ja.cmp(&jb) {
            Ordering::Equal => {
                let j = ja;
                let mut sa = 0.0f64;
                while pa < alen && unsafe { *ai.add(pa) } == j {
                    sa += unsafe { *av.add(pa) };
                    pa += 1;
                }
                let mut sb = 0.0f64;
                while pb < blen && unsafe { *bi.add(pb) } == j {
                    sb += unsafe { *bv.add(pb) };
                    pb += 1;
                }
                let v = sa.powf(sb);
                if v != 0.0 && v.is_finite() {
                    unsafe {
                        std::ptr::write(out_i.add(dst), j);
                        std::ptr::write(out_v.add(dst), v);
                    }
                    dst += 1;
                }
            }
            Ordering::Less => {
                let j = ja;
                while pa < alen && unsafe { *ai.add(pa) } == j {
                    pa += 1;
                }
            }
            Ordering::Greater => {
                let j = jb;
                while pb < blen && unsafe { *bi.add(pb) } == j {
                    pb += 1;
                }
            }
        }
    }
    dst
}

/// Element-wise power of two CSR matrices: C = pow(A, B) = A ** B (intersection only)
#[must_use]
pub fn pow_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;

    // Count phase (rows in parallel)
    let counts: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let sa = i64_to_usize(a.indptr[i]);
            let ea = i64_to_usize(a.indptr[i + 1]);
            let sb = i64_to_usize(b.indptr[i]);
            let eb = i64_to_usize(b.indptr[i + 1]);
            let alen = ea - sa;
            let blen = eb - sb;
            unsafe {
                pow_row_count(
                    a.indices.as_ptr().add(sa),
                    a.data.as_ptr().add(sa),
                    alen,
                    b.indices.as_ptr().add(sb),
                    b.data.as_ptr().add(sb),
                    blen,
                )
            }
        })
        .collect();

    // Prefix sum to build indptr
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows {
        indptr[i + 1] = indptr[i] + usize_to_i64(counts[i]);
    }
    let nnz = i64_to_usize(indptr[nrows]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    // Fill phase (rows in parallel)
    (0..nrows).into_par_iter().for_each(move |i| {
        let sa = i64_to_usize(a.indptr[i]);
        let ea = i64_to_usize(a.indptr[i + 1]);
        let sb = i64_to_usize(b.indptr[i]);
        let eb = i64_to_usize(b.indptr[i + 1]);
        let alen = ea - sa;
        let blen = eb - sb;
        let row_start = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(i) });
        unsafe {
            let pi = (pi_addr as *mut i64).add(row_start);
            let pv = (pv_addr as *mut f64).add(row_start);
            let written = pow_row_fill(
                a.indices.as_ptr().add(sa),
                a.data.as_ptr().add(sa),
                alen,
                b.indices.as_ptr().add(sb),
                b.data.as_ptr().add(sb),
                blen,
                pi,
                pv,
            );
            let expected = i64_to_usize(*(indptr_addr as *const i64).add(i + 1)) - row_start;
            debug_assert_eq!(written, expected);
        }
    });
    Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data)
}

/// Element-wise power of two CSC matrices: C = pow(A, B) = A ** B (intersection only)
#[must_use]
pub fn pow_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let ncols = a.ncols;

    let counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let sa = i64_to_usize(a.indptr[j]);
            let ea = i64_to_usize(a.indptr[j + 1]);
            let sb = i64_to_usize(b.indptr[j]);
            let eb = i64_to_usize(b.indptr[j + 1]);
            let alen = ea - sa;
            let blen = eb - sb;
            unsafe {
                pow_row_count(
                    a.indices.as_ptr().add(sa),
                    a.data.as_ptr().add(sa),
                    alen,
                    b.indices.as_ptr().add(sb),
                    b.data.as_ptr().add(sb),
                    blen,
                )
            }
        })
        .collect();

    let mut indptr = vec![0i64; ncols + 1];
    for j in 0..ncols {
        indptr[j + 1] = indptr[j] + usize_to_i64(counts[j]);
    }
    let nnz = i64_to_usize(indptr[ncols]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    (0..ncols).into_par_iter().for_each(move |j| {
        let sa = i64_to_usize(a.indptr[j]);
        let ea = i64_to_usize(a.indptr[j + 1]);
        let sb = i64_to_usize(b.indptr[j]);
        let eb = i64_to_usize(b.indptr[j + 1]);
        let alen = ea - sa;
        let blen = eb - sb;
        let col_start = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j) });
        unsafe {
            let pi = (pi_addr as *mut i64).add(col_start);
            let pv = (pv_addr as *mut f64).add(col_start);
            let written = pow_row_fill(
                a.indices.as_ptr().add(sa),
                a.data.as_ptr().add(sa),
                alen,
                b.indices.as_ptr().add(sb),
                b.data.as_ptr().add(sb),
                blen,
                pi,
                pv,
            );
            let expected = i64_to_usize(*(indptr_addr as *const i64).add(j + 1)) - col_start;
            debug_assert_eq!(written, expected);
        }
    });
    Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data)
}

/// Power by scalar for CSR: B = A ** alpha
#[must_use]
pub fn pow_scalar_f64(a: &Csr<f64, i64>, alpha: f64) -> Csr<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = (*v).powf(alpha);
        }
    });
    Csr::from_parts_unchecked(a.nrows, a.ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// Power by scalar for CSC
#[must_use]
pub fn pow_scalar_csc_f64(a: &Csc<f64, i64>, alpha: f64) -> Csc<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = (*v).powf(alpha);
        }
    });
    Csc::from_parts_unchecked(a.nrows, a.ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// Power by scalar for COO
#[must_use]
pub fn pow_scalar_coo_f64(a: &Coo<f64, i64>, alpha: f64) -> Coo<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = (*v).powf(alpha);
        }
    });
    Coo::from_parts_unchecked(a.nrows, a.ncols, a.row.clone(), a.col.clone(), data)
}

/// Power by scalar for COOND
#[must_use]
pub fn pow_scalar_coond_f64(a: &CooNd<f64, i64>, alpha: f64) -> CooNd<f64, i64> {
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        for v in chunk {
            *v = (*v).powf(alpha);
        }
    });
    CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data)
}
