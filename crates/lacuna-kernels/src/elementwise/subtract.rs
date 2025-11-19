//! Elementwise subtraction for all sparse formats (Array API: `subtract`)
//!
//! Implements sparse - sparse subtraction for:
//! - CSR (Compressed Sparse Row)
//! - CSC (Compressed Sparse Column)
//! - COOND (N-dimensional COO)

#![allow(
    clippy::similar_names,
    reason = "Pointer/address aliases are intentional in kernels"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
#![allow(
    clippy::comparison_chain,
    reason = "Explicit comparison chain is clearer for merge logic"
)]

use crate::util::UsizeF64Map;
use lacuna_core::{CooNd, Csc, Csr};
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

/// Helper: count non-zero entries in A - B for sorted sparse rows
#[inline]
unsafe fn sub_row_count(
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
    while pa < alen || pb < blen {
        let ja = if pa < alen {
            unsafe { *ai.add(pa) }
        } else {
            i64::MAX
        };
        let jb = if pb < blen {
            unsafe { *bi.add(pb) }
        } else {
            i64::MAX
        };
        let j = if ja <= jb { ja } else { jb };
        let mut v = 0.0f64;
        while pa < alen && unsafe { *ai.add(pa) } == j {
            v += unsafe { *av.add(pa) };
            pa += 1;
        }
        while pb < blen && unsafe { *bi.add(pb) } == j {
            v -= unsafe { *bv.add(pb) };
            pb += 1;
        }
        if v != 0.0 {
            cnt += 1;
        }
    }
    cnt
}

/// Helper: compute A - B for sorted sparse rows, filtering zeros
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_row_fill(
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
    while pa < alen || pb < blen {
        let ja = if pa < alen {
            unsafe { *ai.add(pa) }
        } else {
            i64::MAX
        };
        let jb = if pb < blen {
            unsafe { *bi.add(pb) }
        } else {
            i64::MAX
        };
        if ja <= jb {
            let j = ja;
            let mut v = 0.0f64;
            while pa < alen && unsafe { *ai.add(pa) } == j {
                v += unsafe { *av.add(pa) };
                pa += 1;
            }
            if jb == j {
                while pb < blen && unsafe { *bi.add(pb) } == j {
                    v -= unsafe { *bv.add(pb) };
                    pb += 1;
                }
            }
            if v != 0.0 {
                unsafe {
                    std::ptr::write(out_i.add(dst), j);
                    std::ptr::write(out_v.add(dst), v);
                }
                dst += 1;
            }
        } else {
            let j = jb;
            let mut v = 0.0f64;
            while pb < blen && unsafe { *bi.add(pb) } == j {
                v -= unsafe { *bv.add(pb) };
                pb += 1;
            }
            if v != 0.0 {
                unsafe {
                    std::ptr::write(out_i.add(dst), j);
                    std::ptr::write(out_v.add(dst), v);
                }
                dst += 1;
            }
        }
    }
    dst
}

/// Subtract two CSR matrices: A - B
///
/// Coalesces duplicates within each row. Result shape matches inputs.
#[must_use]
pub fn sub_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;
    
    // Pass 1: count output nnz per row in parallel
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
                sub_row_count(
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

    // Prefix sum -> indptr
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

    // Pass 2: fill rows in parallel
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
            let written = sub_row_fill(
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

/// Subtract two CSC matrices: A - B
///
/// Coalesces duplicates within each column. Result shape matches inputs.
#[must_use]
pub fn sub_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
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
                sub_row_count(
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
            let written = sub_row_fill(
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

/// Subtract two N-dimensional COO arrays: A - B
///
/// Coalesces duplicates and filters zeros. Shapes must match.
#[must_use]
pub fn sub_coond_f64_i64(a: &CooNd<f64, i64>, b: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    assert_eq!(a.shape.len(), b.shape.len());
    assert_eq!(a.shape, b.shape);
    let ndim = a.shape.len();
    let nnz_a = a.data.len();
    let nnz_b = b.data.len();
    
    // Build row-major strides for linearization
    let mut strides = vec![0usize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(a.shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }
    
    // Accumulate A entries, subtract B entries
    let mut acc = UsizeF64Map::with_capacity(nnz_a + nnz_b);
    for k in 0..nnz_a {
        let mut lin = 0usize;
        let base = k * ndim;
        for d in 0..ndim {
            let idx = i64_to_usize(a.indices[base + d]);
            lin = lin
                .checked_add(idx.checked_mul(strides[d]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc.insert_add(lin, a.data[k]);
    }
    for k in 0..nnz_b {
        let mut lin = 0usize;
        let base = k * ndim;
        for d in 0..ndim {
            let idx = i64_to_usize(b.indices[base + d]);
            lin = lin
                .checked_add(idx.checked_mul(strides[d]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc.insert_add(lin, -b.data[k]); // subtract
    }
    
    // Sort and filter
    let mut pairs = acc.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);
    let mut out_pairs: Vec<(usize, f64)> = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        if v != 0.0 {
            out_pairs.push((k, v));
        }
    }
    let out_nnz = out_pairs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * ndim];
    for (pos, (mut lin, v)) in out_pairs.into_iter().enumerate() {
        let base = pos * ndim;
        for d in 0..ndim {
            let s = strides[d];
            let idx = lin / s;
            lin -= idx * s;
            out_indices[base + d] = usize_to_i64(idx);
        }
        out_data.push(v);
    }
    CooNd::from_parts_unchecked(a.shape.clone(), out_indices, out_data)
}
