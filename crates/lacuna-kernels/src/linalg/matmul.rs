//! Matrix multiplication kernels: SpMV and SpMM for all sparse formats.
//!
//! This module provides:
//! - **SpMV (Sparse Matrix-Vector product)**: y = A @ x for CSR, CSC, COO, and N-D arrays
//! - **SpMM (Sparse Matrix-Matrix product)**: Y = A @ B for CSR, CSC, COO, and N-D arrays
//!
//! Implementations use:
//! - Adaptive accumulator strategies (sparse hash map vs. dense arrays with striping)
//! - SIMD vectorization with f64x4 for dense operations
//! - Aggressive loop unrolling and prefetching in SpMV row computation
//! - Thread-local accumulators to minimize synchronization
//! - Heuristic format selection for SpMM (CSR vs. CSC based on k and sparsity)

use crate::data_type_functions::astype::csr_to_csc_f64_i64;
use crate::utility::util::{
    DenseStripe, SMALL_DIM_LIMIT, SMALL_NNZ_LIMIT, STRIPE_ROWS, StripeAccs, UsizeF64Map,
    i64_to_usize,
};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use wide::f64x4;

/// Converts usize to i64 with debug assertions for range validity.
#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

// ---------------- SpMV ----------------

/// Sparse Matrix-Vector product for CSC format: y = A @ x.
///
/// Computes a column-oriented SpMV by iterating columns and accumulating contributions
/// to output rows. Uses adaptive accumulation strategies:
/// - **Sparse accumulation**: Hash map for problems with low output density
/// - **Dense small**: Dense array when output is small (<= 2*STRIPE_ROWS)
/// - **Striped dense**: Striped dense arrays for large outputs (reduces memory bandwidth)
///
/// # Algorithm
/// 1. Partition columns into ranges (~128KB nnz per range)
/// 2. For each range in parallel:
///    - Use thread-local accumulator (sparse, dense, or striped)
///    - Iterate columns and accumulate A[i,j]*x[j] contributions
/// 3. Merge thread-local results into output
///
/// # Optimization
/// - Avoids parallelization overhead for small problems
/// - Column-based partitioning for cache locality
/// - Striped dense accumulation reduces NUMA effects on large outputs
///
/// # Panics
/// - If x.len() != ncols
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

/// N-dimensional SpMV (tensor contraction along one axis): out = tensordot(A, x, axes=[axis]).
///
/// Multiplies an N-dimensional sparse array A by a vector x along a specified axis,
/// reducing dimensionality by 1. Sparse coordinates are linearized and accumulated.
///
/// # Algorithm
/// 1. Identify output axes (all except the contracted axis)
/// 2. Compute output shape and row-major strides
/// 3. For each nonzero in A:
///    - Extract coordinate along contracted axis (to index into x)
///    - Linearize remaining coordinates
///    - Accumulate A[...] * x[axis_idx] by linearized key
/// 4. Sort and reconstruct N-D indices
///
/// # Panics
/// - If axis >= ndim
/// - If x.len() != shape[axis]
/// - If contracting over all axes (use sum instead)
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

/// Sparse Matrix-Vector product for COO format: y = A @ x.
///
/// Computes SpMV for COO format by partitioning nonzero elements into chunks
/// and using adaptive accumulation strategies:
/// - **Sparse accumulation**: Hash map for low output density
/// - **Dense small**: Dense array for small outputs (<= 2*STRIPE_ROWS)
/// - **Striped dense**: Striped dense arrays for large outputs
///
/// # Algorithm
/// 1. Partition nnz into chunks (~nnz / (8*nthreads) elements per chunk)
/// 2. For each chunk in parallel:
///    - Use thread-local accumulator
///    - Process COO entries and accumulate y[i] += A[i,j] * x[j]
/// 3. Merge thread-local results into output
///
/// # Panics
/// - If x.len() != ncols
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

/// Computes a single row of SpMV for CSR: y[i] = A[i,:] @ x.
///
/// Performs aggressive loop unrolling and prefetching:
/// - Unrolls in blocks of 16, 8, 4 elements
/// - Uses mul_add for FMA (Fused Multiply-Add) operations
/// - Processes column indices and values without bounds checking
///
/// # Algorithm
/// 1. Unroll main loop by 16: accumulates 4 groups of 4 consecutive elements
/// 2. Unroll remaining by 8: processes 2 groups of 4 elements
/// 3. Unroll remaining by 4: processes 1 group of 4 elements
/// 4. Process remaining 1-3 elements serially
///
/// # Optimization
/// - FMA (multiply-add) for higher throughput
/// - Unsafe bounds checking elimination via preconditions
/// - Aggressive unrolling for ILP (Instruction-Level Parallelism)
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

/// Sparse Matrix-Vector product for CSR format: y = A @ x.
///
/// Computes SpMV by processing rows in parallel. Each row computation uses
/// aggressive loop unrolling with FMA for high throughput.
///
/// # Algorithm
/// 1. Partition rows into ranges (~128K nnz per range)
/// 2. For each range in parallel:
///    - Compute rows using `spmv_row_f64_i64` (unrolled row dot product)
/// 3. Write results directly to output
///
/// # Optimization
/// - Row-based partitioning for load balance
/// - Each row uses aggressive unrolling (16-way, 8-way, 4-way)
/// - Avoids parallelization overhead for small problems
///
/// # Panics
/// - If x.len() != ncols
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

// ---------------- SpMM ----------------

/// Sparse Matrix-Matrix product with automatic format selection: Y = A @ B.
///
/// Heuristically selects between CSR and CSC formats based on:
/// - **k** (number of result columns): Prefer CSC for k >= 128
/// - **Column sparsity**: Prefer CSC if k >= 64 and avg_col_nnz >= 8
/// - Otherwise: Use CSR with tiled SIMD processing
///
/// # Arguments
/// * `a` - CSR matrix (nrows x ncols)
/// * `b` - Dense matrix (ncols x k) in row-major order
/// * `k` - Number of columns in B
///
/// # Returns
/// Dense matrix Y (nrows x k) in row-major order
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

/// Sparse Matrix-Matrix product for CSR format: Y = A @ B.
///
/// Computes SpMM by processing each row of A in parallel. Within a row:
/// - For sparse rows (nnz <= 8) with large k: broadcast A[i,j] across columns of B
/// - Otherwise: tile B into 128-column blocks, accumulate 4 B columns at a time with SIMD
///
/// # Algorithm
/// 1. For each row i of A in parallel:
///    a. Iterate 128-column tiles (c0, c1)
///    b. If row sparse and k large: broadcast A[i,j]*B[j,c] across row
///    c. Otherwise: accumulate 4 columns at a time with SIMD (f64x4)
///    d. Process remainder (1-3 columns) serially
///
/// # Optimization
/// - Row parallelization for load balance
/// - Adaptive processing based on row sparsity
/// - SIMD tiling for cache locality and throughput
/// - FMA for high instruction throughput
///
/// # Panics
/// - If b.len() != ncols * k
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

/// N-dimensional SpMM (tensor mode-axis product): out = mode-axis product with B.
///
/// Multiplies an N-dimensional sparse array A by a dense matrix B along a specified axis,
/// producing an N-D sparse array with (N-1)+1=N dimensions (axis replaced by k).
///
/// # Algorithm
/// 1. Compute output shape: (non-contracted A axes, k)
/// 2. For each nonzero A[...] at position (axis_idx=ax):
///    - Linearize coordinates over (non-contracted A axes)
///    - For each column c in B[ax, 0:k]:
///      * Multiply A[...] * B[ax, c]
///      * Accumulate in output with key=(lin_base + c*stride_k)
/// 3. Filter zeros and reconstruct N-D indices
///
/// # Optimization
/// - Processes B columns 4 at a time when possible
/// - Filters zeros during accumulation
/// - Uses row-major linearization for output shape
///
/// # Panics
/// - If axis >= ndim
/// - If b.len() != shape[axis] * k
#[allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "Doc wording and loop style are intentional in this kernel"
)]
#[must_use]
pub fn spmm_coond_f64_i64(
    a: &CooNd<f64, i64>,
    axis: usize,
    b: &[f64],
    k: usize,
) -> CooNd<f64, i64> {
    let ndim = a.shape.len();
    assert!(axis < ndim, "axis out of bounds");
    assert_eq!(
        b.len(),
        a.shape[axis] * k,
        "B must be shape[axis] x k row-major"
    );

    let nnz = a.data.len();
    // tensordot semantics: output axes are A axes excluding `axis`, followed by B's non-contracted axes (here just k)
    let remain_axes: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();
    let remain_ndim = remain_axes.len();
    let mut out_shape: Vec<usize> = remain_axes.iter().map(|&d| a.shape[d]).collect();
    out_shape.push(k);
    let out_ndim = remain_ndim + 1;
    if nnz == 0 || k == 0 {
        return CooNd::from_parts_unchecked(out_shape, Vec::new(), Vec::new());
    }

    // Row-major strides for output shape (remain_axes..., k)
    let mut strides = vec![0usize; out_ndim];
    strides[out_ndim - 1] = 1;
    for i in (0..out_ndim - 1).rev() {
        let s = strides[i + 1]
            .checked_mul(out_shape[i + 1])
            .expect("shape product overflow");
        strides[i] = s;
    }
    let stride_k = strides[out_ndim - 1];

    let mut acc = UsizeF64Map::with_capacity(nnz * (k.min(8)));
    for p in 0..nnz {
        let base = p * ndim;
        let ax_idx = i64_to_usize(a.indices[base + axis]);
        // linear base over remain axes in the order of remain_axes
        let mut lin_base: usize = 0;
        for (m, &d) in remain_axes.iter().enumerate() {
            let idx = i64_to_usize(a.indices[base + d]);
            lin_base = lin_base
                .checked_add(idx.checked_mul(strides[m]).expect("linear index overflow"))
                .expect("linear index overflow");
        }
        let a_val = a.data[p];
        let b_row_base = ax_idx * k;
        let mut c = 0usize;
        let limit4 = k & !3;
        while c < limit4 {
            let v0 = a_val * unsafe { *b.as_ptr().add(b_row_base + c) };
            let v1 = a_val * unsafe { *b.as_ptr().add(b_row_base + c + 1) };
            let v2 = a_val * unsafe { *b.as_ptr().add(b_row_base + c + 2) };
            let v3 = a_val * unsafe { *b.as_ptr().add(b_row_base + c + 3) };
            if v0 != 0.0 {
                let key = lin_base
                    .checked_add(c.checked_mul(stride_k).expect("linear index overflow"))
                    .expect("linear index overflow");
                acc.insert_add(key, v0);
            }
            if v1 != 0.0 {
                let key = lin_base
                    .checked_add(
                        (c + 1)
                            .checked_mul(stride_k)
                            .expect("linear index overflow"),
                    )
                    .expect("linear index overflow");
                acc.insert_add(key, v1);
            }
            if v2 != 0.0 {
                let key = lin_base
                    .checked_add(
                        (c + 2)
                            .checked_mul(stride_k)
                            .expect("linear index overflow"),
                    )
                    .expect("linear index overflow");
                acc.insert_add(key, v2);
            }
            if v3 != 0.0 {
                let key = lin_base
                    .checked_add(
                        (c + 3)
                            .checked_mul(stride_k)
                            .expect("linear index overflow"),
                    )
                    .expect("linear index overflow");
                acc.insert_add(key, v3);
            }
            c += 4;
        }
        while c < k {
            let v = a_val * unsafe { *b.as_ptr().add(b_row_base + c) };
            if v != 0.0 {
                let key = lin_base
                    .checked_add(c.checked_mul(stride_k).expect("linear index overflow"))
                    .expect("linear index overflow");
                acc.insert_add(key, v);
            }
            c += 1;
        }
    }

    let mut pairs = acc.pairs();
    pairs.sort_unstable_by_key(|(k, _)| *k);
    let mut out_pairs: Vec<(usize, f64)> = Vec::with_capacity(pairs.len());
    for (key, v) in pairs {
        if v != 0.0 {
            out_pairs.push((key, v));
        }
    }
    let out_nnz = out_pairs.len();
    let mut out_data = Vec::with_capacity(out_nnz);
    let mut out_indices = vec![0i64; out_nnz * out_ndim];
    for (pos, (mut lin, v)) in out_pairs.into_iter().enumerate() {
        let base = pos * out_ndim;
        for d in 0..out_ndim {
            let s = strides[d];
            let idx = lin / s;
            lin -= idx * s;
            out_indices[base + d] = usize_to_i64(idx);
        }
        out_data.push(v);
    }
    CooNd::from_parts_unchecked(out_shape, out_indices, out_data)
}

/// Sparse Matrix-Matrix product for CSC format: Y = A @ B (column-major A).
///
/// Computes SpMM by processing tiles of B's columns in parallel. Within a tile:
/// - Iterate A's columns and accumulate B[j, c:c+tile] * A[i,j] contributions
/// - Use SIMD (f64x4) for 4 columns at a time
///
/// # Algorithm
/// 1. Partition B's columns into tiles of 128 columns
/// 2. For each tile in parallel:
///    a. For each A column j:
///       - For each (i, A[i,j]) in column j:
///         * Accumulate B[j, c:c+tile] * A[i,j] into Y[i*k + c:c+tile]
///    b. Use SIMD for groups of 4 columns; process remainder serially
///
/// # Optimization
/// - Column-based B partitioning for cache locality
/// - SIMD processing of 4 result columns per iteration
/// - Thread-level parallelism over B's column tiles
///
/// # Panics
/// - If b.len() != ncols * k
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

/// Sparse Matrix-Matrix product for COO format: Y = A @ B.
///
/// Computes SpMM by processing tiles of B's columns in parallel. For each COO entry:
/// - Multiply A[i,j] by B[j, c:c+tile]
/// - Accumulate into Y[i*k + c:c+tile]
/// - Use SIMD (f64x4) for 4 columns at a time
///
/// # Algorithm
/// 1. Partition B's columns into tiles of 128 columns
/// 2. For each tile in parallel:
///    a. For each nonzero (i, j, A[i,j]):
///       - Accumulate B[j, c:c+tile] * A[i,j] into Y[i*k + c:c+tile]
///    b. Use SIMD for groups of 4 columns; process remainder serially
///
/// # Optimization
/// - Column-based B partitioning
/// - SIMD processing of 4 result columns per iteration
/// - Thread-level parallelism over B's column tiles
///
/// # Panics
/// - If b.len() != ncols * k
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
