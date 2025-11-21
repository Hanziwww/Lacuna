//! Elementwise multiplication (Hadamard product) and scalar multiplication.
//!
//! Implements:
//! - Hadamard product (element-wise multiplication) for CSR, CSC, and N-dimensional COO
//! - Broadcasting support for N-dimensional arrays
//! - Scalar multiplication for all formats with SIMD optimization
//!
//! All Hadamard operations filter out exact zeros from the result.

#![allow(
    clippy::similar_names,
    reason = "Pointer/address aliases are intentional in low-level kernels"
)]
#![allow(
    clippy::suspicious_operation_groupings,
    reason = "Merge loop uses intended precedence for sorted index comparison"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
#![allow(
    clippy::needless_range_loop,
    clippy::comparison_chain,
    reason = "Index-style loops and simple comparisons are intentional for clarity and performance"
)]

use crate::utility::util::UsizeF64Map;
use core::cmp::Ordering;
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::collections::HashMap;
use wide::f64x4;

const SMALL_NNZ_SIMD: usize = 16 * 1024;

/// Converts i64 to usize with debug assertions for non-negative values.
#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// Converts usize to i64 with debug assertions for range validity.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// Builds strides for row-major (C-style) ordering.
/// For shape [d0, d1, ..., dn], computes strides where stride[i] = d[i+1] * d[i+2] * ...
/// This allows converting multi-dimensional indices to linear indices.
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

/// Hadamard product (element-wise multiplication) with broadcasting for N-dimensional COO arrays.
///
/// Multiplies two N-dimensional sparse arrays with broadcasting support:
/// - Shapes are broadcast to a common ndim (left-padded with 1s)
/// - Dimensions with size 1 are broadcast to match the other operand
/// - Result shape is the maximum of broadcast shapes
///
/// # Algorithm
/// 1. Normalize shapes to common ndim with left-padding
/// 2. Validate broadcasting constraints
/// 3. Build masks to identify intersection, free-a, and free-b dimensions
/// 4. Normalize index arrays to common ndim
/// 5. Group both arrays by intersection key (only dims where both >1)
/// 6. For matching intersection keys, compute all pairwise products
/// 7. Accumulate products in per-chunk maps, then merge
/// 8. Reconstruct N-D indices from linearized keys
///
/// # Parallelization
/// Chunks of matching keys are processed in parallel to accumulate products.
///
/// # Panics
/// - If dimensions cannot be broadcast together
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn hadamard_broadcast_coond_f64_i64(
    a: &CooNd<f64, i64>,
    b: &CooNd<f64, i64>,
) -> CooNd<f64, i64> {
    let ad = a.shape.len();
    let bd = b.shape.len();
    let d = ad.max(bd);
    // Pad shapes (left) to same ndim
    let mut ash = vec![1usize; d];
    let mut bsh = vec![1usize; d];
    for i in 0..ad {
        ash[d - ad + i] = a.shape[i];
    }
    for i in 0..bd {
        bsh[d - bd + i] = b.shape[i];
    }
    // Validate broadcasting and compute out shape
    let mut out_shape = vec![0usize; d];
    for i in 0..d {
        let ai = ash[i];
        let bi = bsh[i];
        assert!(
            !(ai != bi && ai != 1 && bi != 1),
            "shape mismatch: cannot broadcast along dim {i}: {ai} vs {bi}"
        );
        out_shape[i] = ai.max(bi);
    }
    if a.data.is_empty() || b.data.is_empty() {
        return CooNd::from_parts_unchecked(out_shape, Vec::new(), Vec::new());
    }

    // Masks
    let mut inter_mask = vec![false; d];
    let mut a_free = vec![false; d];
    let mut b_free = vec![false; d];
    for i in 0..d {
        inter_mask[i] = ash[i] != 1 && bsh[i] != 1;
        a_free[i] = ash[i] == 1 && bsh[i] != 1;
        b_free[i] = bsh[i] == 1 && ash[i] != 1;
    }

    // Strides for output linearization
    let out_strides = build_strides_row_major(&out_shape);

    // Strides for intersection-key linearization (only dims where both >1)
    let mut inter_dims: Vec<usize> = Vec::new();
    for i in 0..d {
        if inter_mask[i] {
            inter_dims.push(i);
        }
    }
    let inter_strides: Vec<usize> = {
        let mut local = vec![0usize; d];
        if !inter_dims.is_empty() {
            let mut s = 1usize;
            for &idx in inter_dims.iter().rev() {
                local[idx] = s;
                s = s
                    .checked_mul(out_shape[idx])
                    .expect("shape product overflow");
            }
        }
        local
    };

    // Normalize indices to d dims for a and b
    let nnz_a = a.data.len();
    let nnz_b = b.data.len();
    let mut a_norm = vec![0i64; nnz_a * d];
    let mut b_norm = vec![0i64; nnz_b * d];
    // A
    if nnz_a > 0 {
        let aoff = d - ad;
        let a_ptr = a_norm.as_mut_ptr() as usize;
        (0..nnz_a).into_par_iter().for_each(|k| {
            let base_a = k * ad;
            let base_n = k * d;
            let p = a_ptr as *mut i64;
            for i in 0..d {
                let val = if i >= aoff {
                    a.indices[base_a + (i - aoff)]
                } else {
                    0
                };
                unsafe {
                    std::ptr::write(p.add(base_n + i), val);
                }
            }
        });
    }
    // B
    if nnz_b > 0 {
        let boff = d - bd;
        let b_ptr = b_norm.as_mut_ptr() as usize;
        (0..nnz_b).into_par_iter().for_each(|k| {
            let base_b = k * bd;
            let base_n = k * d;
            let p = b_ptr as *mut i64;
            for i in 0..d {
                let val = if i >= boff {
                    b.indices[base_b + (i - boff)]
                } else {
                    0
                };
                unsafe {
                    std::ptr::write(p.add(base_n + i), val);
                }
            }
        });
    }

    // Group by intersection key
    let mut map_a: HashMap<usize, Vec<usize>> = HashMap::new();
    for k in 0..nnz_a {
        let base = k * d;
        let mut key: usize = 0;
        for i in 0..d {
            if inter_mask[i] {
                let idx = i64_to_usize(unsafe { *a_norm.get_unchecked(base + i) });
                key = key
                    .checked_add(idx.checked_mul(inter_strides[i]).expect("key overflow"))
                    .expect("key overflow");
            }
        }
        map_a.entry(key).or_default().push(k);
    }
    let mut map_b: HashMap<usize, Vec<usize>> = HashMap::new();
    for k in 0..nnz_b {
        let base = k * d;
        let mut key: usize = 0;
        for i in 0..d {
            if inter_mask[i] {
                let idx = i64_to_usize(unsafe { *b_norm.get_unchecked(base + i) });
                key = key
                    .checked_add(idx.checked_mul(inter_strides[i]).expect("key overflow"))
                    .expect("key overflow");
            }
        }
        map_b.entry(key).or_default().push(k);
    }

    let mut keys: Vec<usize> = Vec::with_capacity(map_a.len().min(map_b.len()));
    for &k in map_a.keys() {
        if map_b.contains_key(&k) {
            keys.push(k);
        }
    }

    // Parallel accumulate products into per-chunk accumulators
    let chunk = 1.max(keys.len() / (rayon::current_num_threads().max(1) * 4));
    let parts: Vec<Vec<(usize, f64)>> = (0..keys.len().div_ceil(chunk))
        .into_par_iter()
        .map(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(keys.len());
            let mut acc = UsizeF64Map::with_capacity(1024);
            for idx in start..end {
                let key = keys[idx];
                let va = map_a.get(&key).unwrap();
                let vb = map_b.get(&key).unwrap();
                for &ka in va {
                    let ba = ka * d;
                    let av = a.data[ka];
                    for &kb in vb {
                        let bb = kb * d;
                        let bv = b.data[kb];
                        // Compose output linear index
                        let mut lin: usize = 0;
                        for i in 0..d {
                            let ia = i64_to_usize(unsafe { *a_norm.get_unchecked(ba + i) });
                            let ib = i64_to_usize(unsafe { *b_norm.get_unchecked(bb + i) });
                            let idx = if ash[i] == 1 { ib } else { ia };
                            lin = lin
                                .checked_add(idx.checked_mul(out_strides[i]).expect("lin overflow"))
                                .expect("lin overflow");
                        }
                        acc.insert_add(lin, av * bv);
                    }
                }
            }
            acc.pairs()
        })
        .collect();

    // Merge parts and coalesce duplicates
    let mut global = UsizeF64Map::with_capacity(parts.iter().map(Vec::len).sum());
    for p in parts {
        for (k, v) in p {
            global.insert_add(k, v);
        }
    }
    let mut pairs = global.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);

    let out_nnz = pairs.len();
    let mut out_indices = vec![0i64; out_nnz * d];
    let mut out_data = Vec::with_capacity(out_nnz);
    for (pos, (mut lin, v)) in pairs.into_iter().enumerate() {
        let base = pos * d;
        for i in 0..d {
            let s = out_strides[i];
            let idx = lin / s;
            lin -= idx * s;
            out_indices[base + i] = usize_to_i64(idx);
        }
        out_data.push(v);
    }
    CooNd::from_parts_unchecked(out_shape, out_indices, out_data)
}

/// Hadamard product (element-wise multiplication) for same-shape N-dimensional COO arrays.
///
/// Multiplies two N-dimensional sparse arrays element-wise. Both arrays must have
/// identical shapes. Only entries with nonzero product are included in the result.
///
/// # Algorithm
/// 1. Linearize all coordinates using row-major strides
/// 2. Accumulate entries from A and B separately (combining duplicates)
/// 3. Sort both accumulated lists by linearized key
/// 4. Merge sorted lists, multiplying values at matching keys
/// 5. Filter out exact zeros
/// 6. Reconstruct N-D coordinates from linearized keys
///
/// # Complexity
/// - Time: O((nnz_A + nnz_B) * ndim + nnz_output * ndim)
/// - Space: O(nnz_A + nnz_B + nnz_output)
///
/// # Panics
/// - If input arrays have different shapes or ndim
#[must_use]
pub fn hadamard_coond_f64_i64(a: &CooNd<f64, i64>, b: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    assert_eq!(a.shape.len(), b.shape.len());
    assert_eq!(a.shape, b.shape);
    let ndim = a.shape.len();
    let nnz_a = a.data.len();
    let nnz_b = b.data.len();
    let mut strides = vec![0usize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(a.shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }
    let mut acc_a = UsizeF64Map::with_capacity(nnz_a);
    for k in 0..nnz_a {
        let mut lin = 0usize;
        let base = k * ndim;
        for d in 0..ndim {
            let idx = i64_to_usize(a.indices[base + d]);
            lin = lin
                .checked_add(idx.checked_mul(strides[d]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc_a.insert_add(lin, a.data[k]);
    }
    let mut acc_b = UsizeF64Map::with_capacity(nnz_b);
    for k in 0..nnz_b {
        let mut lin = 0usize;
        let base = k * ndim;
        for d in 0..ndim {
            let idx = i64_to_usize(b.indices[base + d]);
            lin = lin
                .checked_add(idx.checked_mul(strides[d]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        acc_b.insert_add(lin, b.data[k]);
    }
    let mut pa = acc_a.pairs();
    let mut pb = acc_b.pairs();
    pa.sort_unstable_by_key(|(k, _)| *k);
    pb.sort_unstable_by_key(|(k, _)| *k);
    let mut outs: Vec<(usize, f64)> = Vec::new();
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < pa.len() && ib < pb.len() {
        let (ka, va) = pa[ia];
        let (kb, vb) = pb[ib];
        if ka == kb {
            let v = va * vb;
            if v != 0.0 {
                outs.push((ka, v));
            }
            ia += 1;
            ib += 1;
        } else if ka < kb {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    let out_nnz = outs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * ndim];
    for (pos, (mut lin, v)) in outs.into_iter().enumerate() {
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

/// Counts nonzero entries in the Hadamard product of two sorted sparse rows.
///
/// Uses a two-pointer merge algorithm to find matching column indices in rows A and B,
/// then counts products that are nonzero. Handles duplicate column indices by accumulating
/// values before multiplication.
///
/// # Arguments
/// * `ai`, `av`, `alen` - Column indices and values of matrix A's row segment
/// * `bi`, `bv`, `blen` - Column indices and values of matrix B's row segment
///
/// # Returns
/// Count of nonzero entries in the Hadamard product result
///
/// # Safety
/// Caller must ensure all pointers are valid for their lengths and columns are strictly increasing.
#[inline]
unsafe fn hadamard_row_count(
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
                let v = sa * sb;
                if v != 0.0 {
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

/// Merges two sorted sparse rows via Hadamard product and writes output.
///
/// Performs the same merge and multiplication as `hadamard_row_count` but also
/// writes the resulting column indices and product values to output arrays.
/// Handles duplicate indices by accumulating values before multiplication and filters zeros.
///
/// # Arguments
/// * `ai`, `av`, `alen` - Column indices and values of matrix A's row segment
/// * `bi`, `bv`, `blen` - Column indices and values of matrix B's row segment
/// * `out_i` - Output buffer for product column indices
/// * `out_v` - Output buffer for product values
///
/// # Returns
/// Number of elements written to output (exact zeros are filtered)
///
/// # Safety
/// Caller must ensure all input pointers are valid and output buffers have sufficient capacity.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn hadamard_row_fill(
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
                let v = sa * sb;
                if v != 0.0 {
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

/// Hadamard product (element-wise multiplication) for CSC matrices: A ⊙ B → Result.
///
/// Computes element-wise multiplication by processing columns in parallel.
/// Only positions where both operands have nonzero entries are included in the result.
/// Duplicate column indices are handled via value accumulation before multiplication.
///
/// # Algorithm
/// **Pass 1: Count Phase** (parallel over columns)
/// - For each column j, count nonzeros in A[j] ⊙ B[j] using two-pointer merge
///
/// **Pass 2: Fill Phase** (parallel over columns)
/// - Compute column pointers (indptr) via prefix sum
/// - For each column, merge rows and write Hadamard products
///
/// # Panics
/// - If input matrices have different shapes
#[must_use]
pub fn hadamard_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
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
                hadamard_row_count(
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
            let written = hadamard_row_fill(
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

/// Hadamard product (element-wise multiplication) for CSR matrices: A ⊙ B → Result.
///
/// Computes element-wise multiplication by processing rows in parallel.
/// Only positions where both operands have nonzero entries are included in the result.
/// Duplicate row indices are handled via value accumulation before multiplication.
///
/// # Algorithm
/// **Pass 1: Count Phase** (parallel over rows)
/// - For each row i, count nonzeros in A[i] ⊙ B[i] using two-pointer merge
///
/// **Pass 2: Fill Phase** (parallel over rows)
/// - Compute row pointers (indptr) via prefix sum
/// - For each row, merge columns and write Hadamard products
///
/// # Panics
/// - If input matrices have different shapes
#[must_use]
pub fn hadamard_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;
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
                hadamard_row_count(
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
            let written = hadamard_row_fill(
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

/// Scalar multiplication for N-dimensional COO arrays: alpha * A.
///
/// Multiplies all nonzero values by a scalar using SIMD optimization for speed.
/// Fast paths for alpha=1 (identity) and alpha=0 (zero array).
///
/// # Algorithm
/// - For small nnz: serial SIMD processing (4 elements at a time)
/// - For large nnz: parallel SIMD processing in 4KB chunks
/// - Remaining elements processed serially (non-SIMD)
///
/// # Optimization
/// Uses f64x4 wide vectors to process 4 f64 values simultaneously,
/// with threshold SMALL_NNZ_SIMD to avoid parallelization overhead.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn mul_scalar_coond_f64(a: &CooNd<f64, i64>, alpha: f64) -> CooNd<f64, i64> {
    if alpha == 1.0 {
        return a.clone();
    }
    if alpha == 0.0 {
        let data = vec![0.0f64; a.data.len()];
        return CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data);
    }
    let mut data = a.data.clone();
    let len = data.len();
    if len < SMALL_NNZ_SIMD {
        let aval = f64x4::splat(alpha);
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] *= alpha;
            i += 1;
        }
        return CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data);
    }
    let chunk_size = 4096;
    let aval = f64x4::splat(alpha);
    data.par_chunks_mut(chunk_size).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] *= alpha;
            k += 1;
        }
    });
    CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data)
}

/// Scalar multiplication for CSC matrices: alpha * A.
///
/// Multiplies all nonzero values by a scalar using SIMD optimization for speed.
/// Fast paths for alpha=1 (identity) and alpha=0 (zero matrix).
///
/// # Algorithm
/// - For small nnz: serial SIMD processing (4 elements at a time)
/// - For large nnz: parallel SIMD processing in 4KB chunks
/// - Remaining elements processed serially (non-SIMD)
///
/// # Optimization
/// Uses f64x4 wide vectors to process 4 f64 values simultaneously,
/// with threshold SMALL_NNZ_SIMD to avoid parallelization overhead.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn mul_scalar_csc_f64(a: &Csc<f64, i64>, alpha: f64) -> Csc<f64, i64> {
    if alpha == 1.0 {
        return a.clone();
    }
    let nrows = a.nrows;
    let ncols = a.ncols;
    if alpha == 0.0 {
        let data = vec![0.0f64; a.data.len()];
        return Csc::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }
    let len = a.data.len();
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let aval = f64x4::splat(alpha);
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] *= alpha;
            i += 1;
        }
        return Csc::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }
    let mut data = a.data.clone();
    let chunk_size = 4096;
    let aval = f64x4::splat(alpha);
    data.par_chunks_mut(chunk_size).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] *= alpha;
            k += 1;
        }
    });
    Csc::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// Scalar multiplication for COO matrices: alpha * A.
///
/// Multiplies all nonzero values by a scalar using SIMD optimization for speed.
/// Fast paths for alpha=1 (identity) and alpha=0 (zero matrix).
///
/// # Algorithm
/// - For small nnz: serial SIMD processing (4 elements at a time)
/// - For large nnz: parallel SIMD processing in 4KB chunks
/// - Remaining elements processed serially (non-SIMD)
///
/// # Optimization
/// Uses f64x4 wide vectors to process 4 f64 values simultaneously,
/// with threshold SMALL_NNZ_SIMD to avoid parallelization overhead.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn mul_scalar_coo_f64(a: &Coo<f64, i64>, alpha: f64) -> Coo<f64, i64> {
    if alpha == 1.0 {
        return a.clone();
    }
    let nrows = a.nrows;
    let ncols = a.ncols;
    if alpha == 0.0 {
        let data = vec![0.0f64; a.data.len()];
        return Coo::from_parts_unchecked(nrows, ncols, a.row.clone(), a.col.clone(), data);
    }
    let mut data = a.data.clone();
    let len = data.len();
    if len < SMALL_NNZ_SIMD {
        let aval = f64x4::splat(alpha);
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] *= alpha;
            i += 1;
        }
        return Coo::from_parts_unchecked(nrows, ncols, a.row.clone(), a.col.clone(), data);
    }
    let chunk_size = 4096;
    let aval = f64x4::splat(alpha);
    data.par_chunks_mut(chunk_size).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] *= alpha;
            k += 1;
        }
    });
    Coo::from_parts_unchecked(nrows, ncols, a.row.clone(), a.col.clone(), data)
}

/// Scalar multiplication for CSR matrices: alpha * A.
///
/// Multiplies all nonzero values by a scalar using SIMD optimization for speed.
/// Fast paths for alpha=1 (identity) and alpha=0 (zero matrix).
///
/// # Algorithm
/// - For small nnz: serial SIMD processing (4 elements at a time)
/// - For large nnz: parallel SIMD processing in 4KB chunks
/// - Remaining elements processed serially (non-SIMD)
///
/// # Optimization
/// Uses f64x4 wide vectors to process 4 f64 values simultaneously,
/// with threshold SMALL_NNZ_SIMD to avoid parallelization overhead.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn mul_scalar_f64(a: &Csr<f64, i64>, alpha: f64) -> Csr<f64, i64> {
    // Fast paths
    if alpha == 1.0 {
        return a.clone();
    }
    let nrows = a.nrows;
    let ncols = a.ncols;
    if alpha == 0.0 {
        // Structure unchanged; data all zeros
        let data = vec![0.0f64; a.data.len()];
        return Csr::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }

    let len = a.data.len();
    // avoid parallel overhead for small problems
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let aval = f64x4::splat(alpha);
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] *= alpha;
            i += 1;
        }
        return Csr::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }

    // Large case: parallelize over chunks
    let mut data = a.data.clone();
    let chunk_size = 4096;
    let aval = f64x4::splat(alpha);
    data.par_chunks_mut(chunk_size).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] *= alpha;
            k += 1;
        }
    });
    Csr::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data)
}
