#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k for indices"
)]

//! Vector-matrix dot product kernels (y = x^T @ A or y = A @ x).
//!
//! This module provides efficient vector-matrix multiplication operations for sparse matrices
//! in multiple formats (CSR, CSC, COO, CooNd). It supports two fundamental operations:
//!
//! 1. **Left multiplication (y^T = x^T @ A)**: Contract along axis 0, producing a column vector.
//!    Each output element y[j] is the dot product of x with column j of A.
//!
//! 2. **Right multiplication (y = A @ x)**: Contract along axis 1, producing a row vector.
//!    Each output element y[i] is the dot product of row i of A with x.
//!
//! ## Optimization Strategies
//!
//! - **CSR axis 1 (y = A @ x)**: Delegates directly to SpMV kernels for row-oriented efficiency.
//! - **CSR axis 0 (y^T = x^T @ A)**: Uses stripe-accumulator pattern with memory budgeting (~128KB per thread),
//!   adaptively selecting between dense and touched-only aggregation based on sparsity.
//! - **CSC axis 1 (y = A @ x)**: Delegates to SpMV CSC kernels for column-oriented efficiency.
//! - **CSC axis 0 (y^T = x^T @ A)**: Uses 16-way loop unrolling with FMA (fused multiply-add)
//!   to maximize instruction-level parallelism on column-wise dot products.
//! - **COO axis 1 (y = A @ x)**: Delegates to SpMV COO kernels.
//! - **COO axis 0 (y^T = x^T @ A)**: Chunks work across threads with local sparse accumulators,
//!   merging results using hash map aggregation to avoid contention.
//! - **CooNd**: Generalizes to N-dimensional tensors via axis specification,
//!   delegating to SpMV kernels after linearization.

use crate::linalg::matmul::{spmv_coo_f64_i64, spmv_coond_f64_i64, spmv_csc_f64_i64, spmv_f64_i64};
use crate::utility::util::{STRIPE, StripeAccs, UsizeF64Map, i64_to_usize};
use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;

/// Right multiplication: y = A @ x (CSR matrix, axis 1).
///
/// Computes the dot product of each row of matrix A with vector x.
/// **Mathematical operation**: y[i] = `sum_j` A[i, j] * x[j]
///
/// **Algorithm**: Delegates directly to `spmv_f64_i64` which uses row-oriented sparse
/// matrix-vector product optimized for CSR format.
///
/// **Complexity**: O(nnz(A)) time, O(m) space for output.
///
/// # Arguments
/// * `a` - CSR sparse matrix of shape (m, n)
/// * `x` - Dense vector of length n
///
/// # Returns
/// Vector of length m containing the resulting dot products.
///
/// # Panics
/// Panics if `x.len()` != a.ncols
#[must_use]
pub fn vecdot_csr_dense_axis1_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_f64_i64(a, x)
}

/// Left multiplication: y = x^T @ A (CSR matrix, axis 0).
///
/// Computes the weighted sum across rows: each output column j is the sum of rows,
/// weighted by the input vector x. **Mathematical operation**: y[j] = `sum_i` x[i] * A[i, j]
///
/// **Algorithm**: Uses stripe-accumulator pattern with memory-aware parallelization:
/// 1. Chunk rows into contiguous segments with target ~128KB accumulated nonzeros per segment
/// 2. Process each segment in parallel, accumulating weighted column values into per-thread
///    stripe-based accumulators (each stripe covers ~2048 columns)
/// 3. Merge accumulators using selective aggregation: if >50% of stripe is touched,
///    add densely; otherwise, update only non-zero entries
///
/// This strategy avoids false sharing by localizing updates, and adapts between dense
/// and sparse aggregation based on fill density.
///
/// **Complexity**: O(nnz(A)) time, O(m + c) space with c = number of stripes.
///
/// # Arguments
/// * `a` - CSR sparse matrix of shape (m, n)
/// * `x` - Dense weight vector of length m
///
/// # Returns
/// Vector of length n containing weighted column sums.
///
/// # Panics
/// Panics if x.len() != a.nrows or ncols == 0 is handled gracefully.
#[must_use]
pub fn vecdot_csr_dense_axis0_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    if ncols == 0 {
        return Vec::new();
    }

    // Stripe-accumulator pattern across columns, weighting by row vector x.
    // Divides the column space into stripes (typically 2048 columns each) and maintains
    // per-thread stripe-based accumulators to minimize false sharing during parallel updates.
    let stripe = STRIPE;
    let nstripes = ncols.div_ceil(stripe);
    let tls: ThreadLocal<RefCell<StripeAccs>> = ThreadLocal::new();

    // Target ~128KB accumulated nonzeros per work segment to balance memory locality and parallelism.
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

    // Process each row segment in parallel, accumulating weighted column updates into per-thread stripe accumulators.
    ranges.into_par_iter().for_each(|(r0, r1)| {
        let cell = tls.get_or(|| RefCell::new(vec![None; nstripes]));
        let mut accs = cell.borrow_mut();
        let xs = &x[r0..r1];
        for (off_row, &w) in xs.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let i = r0 + off_row;
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            // Iterate through nonzeros in row i, accumulating weighted contributions.
            for p in s..e {
                let j = i64_to_usize(a.indices[p]);
                let sid = j / stripe; // Determine stripe ID
                let base = sid * stripe; // Base column index of stripe
                let off = j - base; // Offset within stripe
                // Initialize stripe accumulator if not yet created.
                if accs[sid].is_none() {
                    let stripe_len = (ncols - base).min(stripe);
                    accs[sid] = Some((vec![0.0f64; stripe_len], vec![0u8; stripe_len], Vec::new()));
                }
                let acc = accs[sid].as_mut().unwrap();
                // Track which offsets have been touched (for sparse finalization).
                if acc.1[off] == 0 {
                    acc.1[off] = 1;
                    acc.2.push(off);
                }
                acc.0[off] += a.data[p] * w;
            }
        }
    });

    // Merge stripe accumulators into output, adapting between dense and sparse aggregation.
    let mut out = vec![0.0f64; ncols];
    for cell in tls {
        let accs = cell.into_inner();
        for (sid, stripe_opt) in accs.into_iter().enumerate() {
            if let Some((vals, mut seen, touched)) = stripe_opt {
                let base = sid * stripe;
                // If more than 50% of stripe elements are non-zero, aggregate densely.
                if touched.len() > vals.len() / 2 {
                    for (off, &v) in vals.iter().enumerate() {
                        out[base + off] += v;
                    }
                } else {
                    // Otherwise, update only touched elements to avoid dense reads.
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

/// Right multiplication: y = A @ x (CSC matrix, axis 1).
///
/// Computes the dot product of each row of matrix A with vector x.
/// **Mathematical operation**: y[i] = `sum_j` A[i, j] * x[j]
///
/// **Algorithm**: Delegates to `spmv_csc_f64_i64` which computes SpMV by iterating columns
/// and scattering their contributions to output rows (column-oriented approach).
///
/// **Complexity**: O(nnz(A)) time, O(m) space for output.
///
/// # Arguments
/// * `a` - CSC sparse matrix of shape (m, n)
/// * `x` - Dense vector of length n
///
/// # Returns
/// Vector of length m containing the resulting dot products.
///
/// # Panics
/// Panics if `x.len()` != a.ncols
#[must_use]
pub fn vecdot_csc_dense_axis1_f64_i64(a: &Csc<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_csc_f64_i64(a, x)
}

#[allow(
    clippy::too_many_lines,
    reason = "Kernel opts and unrolled loops make this long"
)]
/// Left multiplication: y = x^T @ A (CSC matrix, axis 0).
///
/// Computes the weighted sum across rows: each output column j is the sum of rows,
/// weighted by the input vector x. **Mathematical operation**: y[j] = `sum_i` x[i] * A[i, j]
///
/// **Algorithm**: Parallelizes over columns (CSC format is column-native), computing each
/// output element as a dot product of column j with the weight vector x.
/// Uses 16-way loop unrolling with FMA (fused multiply-add) to maximize instruction-level
/// parallelism and hide memory latency:
/// - 16-element blocks: 4 FMA chains of 4 elements each
/// - 8-element blocks: 2 FMA chains of 4 elements each
/// - 4-element blocks: 1 FMA chain of 4 elements
/// - Remainder: scalar loop
///
/// This structure allows the CPU to execute multiple FMA instructions in parallel while
/// waiting for memory loads.
///
/// **Complexity**: O(nnz(A)) time, O(1) auxiliary space per thread.
///
/// # Arguments
/// * `a` - CSC sparse matrix of shape (m, n)
/// * `x` - Dense weight vector of length m
///
/// # Returns
/// Vector of length n containing weighted column dot products.
///
/// # Panics
/// Panics if `x.len()` != a.nrows
#[must_use]
pub fn vecdot_csc_dense_axis0_f64_i64(a: &Csc<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    let mut out = vec![0.0f64; ncols];
    // Process each column in parallel using loop unrolling for instruction-level parallelism.
    out.par_iter_mut().enumerate().for_each(|(j, oj)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let mut acc = 0.0f64;
        let mut p = s;
        // Process 16 elements at a time with FMA chaining to maximize CPU throughput.
        let end16 = e - ((e - p) & 15);
        unsafe {
            // 16-element blocks: unroll into 4 FMA chains of 4 consecutive products each.
            // This allows the CPU to execute multiple FMA instructions in parallel.
            while p < end16 {
                // Prefetch row indices for the 16-element block.
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                let i4 = i64_to_usize(*a.indices.get_unchecked(p + 4));
                let i5 = i64_to_usize(*a.indices.get_unchecked(p + 5));
                let i6 = i64_to_usize(*a.indices.get_unchecked(p + 6));
                let i7 = i64_to_usize(*a.indices.get_unchecked(p + 7));
                let i8 = i64_to_usize(*a.indices.get_unchecked(p + 8));
                let i9 = i64_to_usize(*a.indices.get_unchecked(p + 9));
                let i10 = i64_to_usize(*a.indices.get_unchecked(p + 10));
                let i11 = i64_to_usize(*a.indices.get_unchecked(p + 11));
                let i12 = i64_to_usize(*a.indices.get_unchecked(p + 12));
                let i13 = i64_to_usize(*a.indices.get_unchecked(p + 13));
                let i14 = i64_to_usize(*a.indices.get_unchecked(p + 14));
                let i15 = i64_to_usize(*a.indices.get_unchecked(p + 15));
                // FMA chain 1: elements 0-3 (executes in parallel with chains 2-4).
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                // FMA chain 2: elements 4-7.
                acc = (*a.data.get_unchecked(p + 7)).mul_add(
                    *x.get_unchecked(i7),
                    (*a.data.get_unchecked(p + 6)).mul_add(
                        *x.get_unchecked(i6),
                        (*a.data.get_unchecked(p + 5)).mul_add(
                            *x.get_unchecked(i5),
                            (*a.data.get_unchecked(p + 4)).mul_add(*x.get_unchecked(i4), acc),
                        ),
                    ),
                );
                // FMA chain 3: elements 8-11.
                acc = (*a.data.get_unchecked(p + 11)).mul_add(
                    *x.get_unchecked(i11),
                    (*a.data.get_unchecked(p + 10)).mul_add(
                        *x.get_unchecked(i10),
                        (*a.data.get_unchecked(p + 9)).mul_add(
                            *x.get_unchecked(i9),
                            (*a.data.get_unchecked(p + 8)).mul_add(*x.get_unchecked(i8), acc),
                        ),
                    ),
                );
                // FMA chain 4: elements 12-15.
                acc = (*a.data.get_unchecked(p + 15)).mul_add(
                    *x.get_unchecked(i15),
                    (*a.data.get_unchecked(p + 14)).mul_add(
                        *x.get_unchecked(i14),
                        (*a.data.get_unchecked(p + 13)).mul_add(
                            *x.get_unchecked(i13),
                            (*a.data.get_unchecked(p + 12)).mul_add(*x.get_unchecked(i12), acc),
                        ),
                    ),
                );
                p += 16;
            }
            // 8-element blocks: unroll into 2 FMA chains of 4 elements each.
            let end8 = e - ((e - p) & 7);
            while p < end8 {
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                let i4 = i64_to_usize(*a.indices.get_unchecked(p + 4));
                let i5 = i64_to_usize(*a.indices.get_unchecked(p + 5));
                let i6 = i64_to_usize(*a.indices.get_unchecked(p + 6));
                let i7 = i64_to_usize(*a.indices.get_unchecked(p + 7));
                // FMA chain 1: elements 0-3.
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                // FMA chain 2: elements 4-7.
                acc = (*a.data.get_unchecked(p + 7)).mul_add(
                    *x.get_unchecked(i7),
                    (*a.data.get_unchecked(p + 6)).mul_add(
                        *x.get_unchecked(i6),
                        (*a.data.get_unchecked(p + 5)).mul_add(
                            *x.get_unchecked(i5),
                            (*a.data.get_unchecked(p + 4)).mul_add(*x.get_unchecked(i4), acc),
                        ),
                    ),
                );
                p += 8;
            }
            // 4-element blocks: single FMA chain.
            let end4 = e - ((e - p) & 3);
            while p < end4 {
                let i0 = i64_to_usize(*a.indices.get_unchecked(p));
                let i1 = i64_to_usize(*a.indices.get_unchecked(p + 1));
                let i2 = i64_to_usize(*a.indices.get_unchecked(p + 2));
                let i3 = i64_to_usize(*a.indices.get_unchecked(p + 3));
                // Single FMA chain for 4 elements.
                acc = (*a.data.get_unchecked(p + 3)).mul_add(
                    *x.get_unchecked(i3),
                    (*a.data.get_unchecked(p + 2)).mul_add(
                        *x.get_unchecked(i2),
                        (*a.data.get_unchecked(p + 1)).mul_add(
                            *x.get_unchecked(i1),
                            (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i0), acc),
                        ),
                    ),
                );
                p += 4;
            }
            // Scalar loop for remaining elements (typically 0-3 due to 4-alignment).
            while p < e {
                let i = i64_to_usize(*a.indices.get_unchecked(p));
                acc = (*a.data.get_unchecked(p)).mul_add(*x.get_unchecked(i), acc);
                p += 1;
            }
        }
        *oj = acc;
    });
    out
}

/// Right multiplication: y = A @ x (COO matrix, axis 1).
///
/// Computes the dot product of each row of matrix A with vector x.
/// **Mathematical operation**: y[i] = sum_j A[i, j] * x[j]
///
/// **Algorithm**: Delegates to `spmv_coo_f64_i64` which accumulates nonzeros directly
/// from COO format by iterating through (row, col, value) triplets.
///
/// **Complexity**: O(nnz(A)) time, O(m) space for output.
///
/// # Arguments
/// * `a` - COO sparse matrix of shape (m, n)
/// * `x` - Dense vector of length n
///
/// # Returns
/// Vector of length m containing the resulting dot products.
///
/// # Panics
/// Panics if x.len() != a.ncols
#[must_use]
pub fn vecdot_coo_dense_axis1_f64_i64(a: &Coo<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    spmv_coo_f64_i64(a, x)
}

/// Left multiplication: y = x^T @ A (COO matrix, axis 0).
///
/// Computes the weighted sum across rows: each output column j is the sum of rows,
/// weighted by the input vector x. **Mathematical operation**: y[j] = sum_i x[i] * A[i, j]
///
/// **Algorithm**: Chunks nonzeros across threads (typically 8 chunks per thread), with each
/// thread maintaining a local sparse accumulator (hash map). Threads iterate through their
/// assigned nonzeros, looking up weight x[i] for each triplet (i, j, value) and accumulating
/// value * x[i] into accumulator[j]. Finally, all per-thread accumulators drain into the
/// shared output vector.
///
/// This approach avoids contention by keeping accumulators thread-local, and the sparse
/// accumulator is efficient when output is sparse.
///
/// **Complexity**: O(nnz(A)) time, O(nnz(output)) auxiliary space across all threads.
///
/// # Arguments
/// * `a` - COO sparse matrix of shape (m, n)
/// * `x` - Dense weight vector of length m
///
/// # Returns
/// Vector of length n containing weighted column sums.
///
/// # Panics
/// Panics if `x.len()` != a.nrows; returns zero vector if ncols == 0 or nnz == 0.
#[must_use]
pub fn vecdot_coo_dense_axis0_f64_i64(a: &Coo<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.nrows, "x length must equal nrows");
    let ncols = a.ncols;
    let nnz = a.data.len();
    if ncols == 0 || nnz == 0 {
        return vec![0.0; ncols];
    }
    // Thread-local sparse accumulators (hash maps) for lock-free aggregation.
    let tls: ThreadLocal<RefCell<UsizeF64Map>> = ThreadLocal::new();
    // Chunk nonzeros across threads: typically 8 chunks per available thread.
    let chunk = 1.max(nnz / (rayon::current_num_threads().max(1) * 8));
    (0..nnz.div_ceil(chunk)).into_par_iter().for_each(|t| {
        let start = t * chunk;
        let end = (start + chunk).min(nnz);
        let cell = tls.get_or(|| RefCell::new(UsizeF64Map::with_capacity(1024)));
        let mut acc = cell.borrow_mut();
        // Iterate assigned nonzeros, accumulating weighted contributions.
        for k in start..end {
            let i = i64_to_usize(a.row[k]);
            let j = i64_to_usize(a.col[k]);
            let w = x[i];
            if w != 0.0 {
                acc.insert_add(j, a.data[k] * w);
            }
        }
    });
    // Merge all thread-local accumulators into output.
    let mut out = vec![0.0f64; ncols];
    for cell in tls {
        let mut acc = cell.into_inner();
        acc.drain_to(&mut out);
    }
    out
}

/// N-dimensional tensor-vector contraction along a specified axis (`CooNd` tensor, arbitrary axis).
///
/// Generalizes vector-matrix multiplication to tensors: contracts an N-dimensional sparse tensor
/// along the specified axis with a weight vector, producing an (N-1)-dimensional tensor.
/// **Mathematical operation**: For each tuple of indices (i0, i1, ..., i_{axis-1}, i_{axis+1}, ..., `i_n`),
/// the result is `sum_k` tensor[i0, i1, ..., i_{axis-1}, k, i_{axis+1}, ..., `i_n`] * x[k]
///
/// **Algorithm**: Delegates to `spmv_coond_f64_i64` which linearizes the tensor along the contraction
/// axis and applies SpMV-like logic with axis-specific stride and delinearization.
///
/// **Complexity**: O(nnz(a)) time, O(nnz(output)) space.
///
/// # Arguments
/// * `a` - `CooNd` sparse tensor
/// * `axis` - Axis along which to contract (must be < `a.shape.len()`)
/// * `x` - Dense weight vector of length a.shape[axis]
///
/// # Returns
/// `CooNd` tensor with shape = original shape minus the contracted axis.
///
/// # Panics
/// Panics if axis >= `a.shape.len()` or `x.len()` != a.shape[axis]
#[must_use]
pub fn vecdot_coond_dense_axis_f64_i64(
    a: &CooNd<f64, i64>,
    axis: usize,
    x: &[f64],
) -> CooNd<f64, i64> {
    assert!(axis < a.shape.len(), "axis out of bounds");
    assert_eq!(x.len(), a.shape[axis], "x length must equal shape[axis]");
    spmv_coond_f64_i64(a, axis, x)
}
