#![allow(
    clippy::similar_names,
    reason = "Pointer/address aliases (pi/pv, etc.) are intentionally similar in low-level kernels"
)]
#![allow(
    clippy::suspicious_operation_groupings,
    reason = "Merge loop uses intended precedence for sorted index comparison"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use i/j/k/p for indices"
)]
use core::cmp::Ordering;
use lacuna_core::{Coo, Csc, Csr};
use rayon::prelude::*;
use wide::f64x4;

const SMALL_NNZ_SIMD: usize = 16 * 1024;

#[inline]
fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}
#[must_use]
pub fn add_csc_f64_i64(a: &Csc<f64, i64>, b: &Csc<f64, i64>) -> Csc<f64, i64> {
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
                add_row_count(
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
            let written = add_row_fill(
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

#[inline]
unsafe fn add_row_count(
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
            v += unsafe { *bv.add(pb) };
            pb += 1;
        }
        if v != 0.0 {
            cnt += 1;
        }
    }
    cnt
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn add_row_fill(
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
        let j = if ja <= jb { ja } else { jb };
        let mut v = 0.0f64;
        while pa < alen && unsafe { *ai.add(pa) } == j {
            v += unsafe { *av.add(pa) };
            pa += 1;
        }
        while pb < blen && unsafe { *bi.add(pb) } == j {
            v += unsafe { *bv.add(pb) };
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
    dst
}

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

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

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

#[allow(clippy::float_cmp)]
pub fn scale_inplace_f64(a: &mut Csr<f64, i64>, alpha: f64) {
    if alpha == 1.0 {
        return;
    }
    let len = a.data.len();
    if alpha == 0.0 {
        a.data.fill(0.0);
        return;
    }
    if len < SMALL_NNZ_SIMD {
        let aval = f64x4::splat(alpha);
        let mut i = 0usize;
        let limit4 = len & !3;
        while i < limit4 {
            unsafe {
                let p = a.data.as_mut_ptr().add(i).cast::<[f64; 4]>();
                let v = f64x4::new(core::ptr::read_unaligned(p.cast_const()));
                let r = v * aval;
                core::ptr::write_unaligned(p, r.to_array());
            }
            i += 4;
        }
        while i < len {
            a.data[i] *= alpha;
            i += 1;
        }
        return;
    }
    let chunk_size = 4096;
    let aval = f64x4::splat(alpha);
    a.data.par_chunks_mut(chunk_size).for_each(|chunk| {
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
}

#[must_use]
pub fn add_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
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
                add_row_count(
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
            let written = add_row_fill(
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

/// A - B for CSR matrices; coalesces duplicates within rows.
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

/// Hadamard elementwise product A .* B for CSR matrices; coalesces duplicates per row.
#[must_use]
pub fn hadamard_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;
    // Pass 1: count intersection nnz per row
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
