//! Elementwise absolute value for sparse formats.
//! Fully parallelized with SIMD where beneficial.

#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::comparison_chain,
    clippy::float_cmp
)]

use lacuna_core::{Coo, CooNd, Csc, Csr};
use rayon::prelude::*;
use wide::f64x4;

const SMALL_NNZ_SIMD: usize = 16 * 1024;

/// |A| for CSR
#[must_use]
pub fn abs_scalar_f64(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let len = a.data.len();
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] = data[i].abs();
            i += 1;
        }
        return Csr::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] = chunk[k].abs();
            k += 1;
        }
    });
    Csr::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// |A| for CSC
#[must_use]
pub fn abs_scalar_csc_f64(a: &Csc<f64, i64>) -> Csc<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let len = a.data.len();
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] = data[i].abs();
            i += 1;
        }
        return Csc::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data);
    }
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] = chunk[k].abs();
            k += 1;
        }
    });
    Csc::from_parts_unchecked(nrows, ncols, a.indptr.clone(), a.indices.clone(), data)
}

/// |A| for COO
#[must_use]
pub fn abs_scalar_coo_f64(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    let nrows = a.nrows;
    let ncols = a.ncols;
    let len = a.data.len();
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] = data[i].abs();
            i += 1;
        }
        return Coo::from_parts_unchecked(nrows, ncols, a.row.clone(), a.col.clone(), data);
    }
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] = chunk[k].abs();
            k += 1;
        }
    });
    Coo::from_parts_unchecked(nrows, ncols, a.row.clone(), a.col.clone(), data)
}

/// |A| for COOND
#[must_use]
pub fn abs_scalar_coond_f64(a: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    let len = a.data.len();
    if len < SMALL_NNZ_SIMD {
        let mut data = a.data.clone();
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            data[i] = data[i].abs();
            i += 1;
        }
        return CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data);
    }
    let mut data = a.data.clone();
    let chunk = 4096usize;
    data.par_chunks_mut(chunk).for_each(|chunk| {
        let mut k = 0usize;
        let limit4 = chunk.len() & !3;
        while k < limit4 {
            unsafe {
                let p = chunk.as_mut_ptr().add(k).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v.abs();
                core::ptr::write_unaligned(p, r.to_array());
            }
            k += 4;
        }
        while k < chunk.len() {
            chunk[k] = chunk[k].abs();
            k += 1;
        }
    });
    CooNd::from_parts_unchecked(a.shape.clone(), a.indices.clone(), data)
}
