//! Prune small entries (utility function, not standard Array API)

use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use wide::f64x4;

const SMALL_NNZ_PRUNE: usize = 16384;

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

#[inline]
fn count_significant(values: &[f64], eps: f64) -> usize {
    let mut retained = 0usize;
    let mut block_offset = 0usize;
    let vectorized_limit = values.len() & !3;
    while block_offset < vectorized_limit {
        let block = unsafe {
            let ptr = values.as_ptr().add(block_offset).cast::<[f64; 4]>();
            f64x4::new(core::ptr::read_unaligned(ptr))
        };
        let block_values = block.to_array();
        if block_values[0].abs() > eps {
            retained += 1;
        }
        if block_values[1].abs() > eps {
            retained += 1;
        }
        if block_values[2].abs() > eps {
            retained += 1;
        }
        if block_values[3].abs() > eps {
            retained += 1;
        }
        block_offset += 4;
    }
    while block_offset < values.len() {
        if values[block_offset].abs() > eps {
            retained += 1;
        }
        block_offset += 1;
    }
    retained
}

#[inline]
fn copy_significant_entries(
    source_indices: &[i64],
    source_values: &[f64],
    eps: f64,
    destination_offset: &mut usize,
    destination_indices: &mut [i64],
    destination_values: &mut [f64],
) {
    for (index_value, value) in source_indices.iter().zip(source_values.iter()) {
        if value.abs() > eps {
            destination_indices[*destination_offset] = *index_value;
            destination_values[*destination_offset] = *value;
            *destination_offset += 1;
        }
    }
}

#[must_use]
pub fn prune_eps_coond(a: &CooNd<f64, i64>, eps: f64) -> CooNd<f64, i64> {
    if eps < 0.0 {
        return a.clone();
    }
    let nnz = a.data.len();
    if nnz == 0 {
        return a.clone();
    }
    let ndim = a.shape.len();
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz * ndim);
    for k in 0..nnz {
        let v = a.data[k];
        if v.abs() > eps {
            data.push(v);
            let base = k * ndim;
            for d in 0..ndim {
                indices.push(a.indices[base + d]);
            }
        }
    }
    CooNd::from_parts_unchecked(a.shape.clone(), indices, data)
}

#[must_use]
pub fn prune_eps_csc(a: &Csc<f64, i64>, eps: f64) -> Csc<f64, i64> {
    if eps < 0.0 {
        return a.clone();
    }
    if eps == 0.0 {
        let has_zero = a.data.par_iter().any(|&v| v == 0.0);
        if !has_zero {
            return a.clone();
        }
    }
    let nnz_total = a.data.len();
    if nnz_total < SMALL_NNZ_PRUNE {
        let ncols = a.ncols;
        let mut indptr = vec![0i64; ncols + 1];
        for (col_idx, window) in a.indptr.windows(2).enumerate() {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let column_values = &a.data[start_idx..end_idx];
            let retained = count_significant(column_values, eps);
            indptr[col_idx + 1] = indptr[col_idx] + usize_to_i64(retained);
        }
        let nnz = i64_to_usize(indptr[ncols]);
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        for (col_idx, window) in a.indptr.windows(2).enumerate() {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let mut destination = i64_to_usize(indptr[col_idx]);
            let column_indices = &a.indices[start_idx..end_idx];
            let column_values = &a.data[start_idx..end_idx];
            copy_significant_entries(
                column_indices,
                column_values,
                eps,
                &mut destination,
                &mut indices,
                &mut data,
            );
        }
        return Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data);
    }
    let ncols = a.ncols;
    let mut counts = vec![0usize; ncols];
    counts
        .par_iter_mut()
        .zip(a.indptr.par_windows(2))
        .for_each(|(count, window)| {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let column_values = &a.data[start_idx..end_idx];
            *count = count_significant(column_values, eps);
        });
    let mut indptr = vec![0i64; ncols + 1];
    for (col_idx, &count) in counts.iter().enumerate() {
        indptr[col_idx + 1] = indptr[col_idx] + usize_to_i64(count);
    }
    let nnz = i64_to_usize(indptr[ncols]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let indices_addr = indices.as_mut_ptr() as usize;
    let values_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    let src_indices_addr = a.indices.as_ptr() as usize;
    let src_values_addr = a.data.as_ptr() as usize;
    a.indptr
        .par_windows(2)
        .enumerate()
        .for_each(|(col_idx, window)| {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let mut destination =
                i64_to_usize(unsafe { *(indptr_addr as *const i64).add(col_idx) });
            unsafe {
                let dst_indices_ptr = indices_addr as *mut i64;
                let dst_values_ptr = values_addr as *mut f64;
                let src_indices_ptr = src_indices_addr as *const i64;
                let src_values_ptr = src_values_addr as *const f64;
                for source_index in start_idx..end_idx {
                    let value = *src_values_ptr.add(source_index);
                    if value.abs() > eps {
                        std::ptr::write(
                            dst_indices_ptr.add(destination),
                            *src_indices_ptr.add(source_index),
                        );
                        std::ptr::write(dst_values_ptr.add(destination), value);
                        destination += 1;
                    }
                }
            }
        });
    Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data)
}

#[must_use]
pub fn prune_eps_coo(a: &Coo<f64, i64>, eps: f64) -> Coo<f64, i64> {
    if eps < 0.0 {
        return a.clone();
    }
    let mut row = Vec::with_capacity(a.row.len());
    let mut col = Vec::with_capacity(a.col.len());
    let mut data = Vec::with_capacity(a.data.len());
    for k in 0..a.data.len() {
        let v = a.data[k];
        if v.abs() > eps {
            row.push(a.row[k]);
            col.push(a.col[k]);
            data.push(v);
        }
    }
    Coo::from_parts_unchecked(a.nrows, a.ncols, row, col, data)
}

#[must_use]
pub fn prune_eps(a: &Csr<f64, i64>, eps: f64) -> Csr<f64, i64> {
    if eps < 0.0 {
        return a.clone();
    }
    if eps == 0.0 {
        let has_zero = a.data.par_iter().any(|&v| v == 0.0);
        if !has_zero {
            return a.clone();
        }
    }
    let nnz_total = a.data.len();
    if nnz_total < SMALL_NNZ_PRUNE {
        let nrows = a.nrows;
        let mut indptr = vec![0i64; nrows + 1];
        for (row_idx, window) in a.indptr.windows(2).enumerate() {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let row_values = &a.data[start_idx..end_idx];
            let retained = count_significant(row_values, eps);
            indptr[row_idx + 1] = indptr[row_idx] + usize_to_i64(retained);
        }
        let nnz = i64_to_usize(indptr[nrows]);
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        for (row_idx, window) in a.indptr.windows(2).enumerate() {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let mut destination = i64_to_usize(indptr[row_idx]);
            let row_indices = &a.indices[start_idx..end_idx];
            let row_values = &a.data[start_idx..end_idx];
            copy_significant_entries(
                row_indices,
                row_values,
                eps,
                &mut destination,
                &mut indices,
                &mut data,
            );
        }
        return Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data);
    }
    let nrows = a.nrows;
    let mut counts = vec![0usize; nrows];
    counts
        .par_iter_mut()
        .zip(a.indptr.par_windows(2))
        .for_each(|(count, window)| {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let row_values = &a.data[start_idx..end_idx];
            *count = count_significant(row_values, eps);
        });
    let mut indptr = vec![0i64; nrows + 1];
    for (row_idx, &count) in counts.iter().enumerate() {
        indptr[row_idx + 1] = indptr[row_idx] + usize_to_i64(count);
    }
    let nnz = i64_to_usize(indptr[nrows]);
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let indices_addr = indices.as_mut_ptr() as usize;
    let values_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    let src_indices_addr = a.indices.as_ptr() as usize;
    let src_values_addr = a.data.as_ptr() as usize;
    a.indptr
        .par_windows(2)
        .enumerate()
        .for_each(|(row_idx, window)| {
            let start_idx = i64_to_usize(window[0]);
            let end_idx = i64_to_usize(window[1]);
            let mut destination =
                i64_to_usize(unsafe { *(indptr_addr as *const i64).add(row_idx) });
            unsafe {
                let dst_indices_ptr = indices_addr as *mut i64;
                let dst_values_ptr = values_addr as *mut f64;
                let src_indices_ptr = src_indices_addr as *const i64;
                let src_values_ptr = src_values_addr as *const f64;
                for source_index in start_idx..end_idx {
                    let value = *src_values_ptr.add(source_index);
                    if value.abs() > eps {
                        std::ptr::write(
                            dst_indices_ptr.add(destination),
                            *src_indices_ptr.add(source_index),
                        );
                        std::ptr::write(dst_values_ptr.add(destination), value);
                        destination += 1;
                    }
                }
            }
        });
    Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data)
}
