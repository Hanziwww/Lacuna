#![allow(
    clippy::similar_names,
    reason = "Pointer/address aliases (pi/pv) are intentionally similar in low-level kernels"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "Index-based loops over CSR structure are deliberate for clarity and perf"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Math kernels conventionally use short names like i/j/k/s/e/p/v"
)]
use lacuna_core::{Coo, Csc, Csr, CooNd};
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
pub fn eliminate_zeros_coond(a: &CooNd<f64, i64>) -> CooNd<f64, i64> {
    prune_eps_coond(a, 0.0)
}

#[must_use]
pub fn eliminate_zeros_csc(a: &Csc<f64, i64>) -> Csc<f64, i64> {
    prune_eps_csc(a, 0.0)
}

#[must_use]
#[allow(clippy::too_many_lines)]
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
        for j in 0..ncols {
            let s = i64_to_usize(a.indptr[j]);
            let e = i64_to_usize(a.indptr[j + 1]);
            let col = &a.data[s..e];
            let mut cnt = 0usize;
            let mut k = 0usize;
            let limit4 = col.len() & !3;
            while k < limit4 {
                let v = unsafe {
                    let p = col.as_ptr().add(k).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                let arr = v.to_array();
                if arr[0].abs() > eps {
                    cnt += 1;
                }
                if arr[1].abs() > eps {
                    cnt += 1;
                }
                if arr[2].abs() > eps {
                    cnt += 1;
                }
                if arr[3].abs() > eps {
                    cnt += 1;
                }
                k += 4;
            }
            while k < col.len() {
                if col[k].abs() > eps {
                    cnt += 1;
                }
                k += 1;
            }
            indptr[j + 1] = indptr[j] + usize_to_i64(cnt);
        }
        let nnz = i64_to_usize(indptr[ncols]);
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        for j in 0..ncols {
            let s = i64_to_usize(a.indptr[j]);
            let e = i64_to_usize(a.indptr[j + 1]);
            let mut dst = i64_to_usize(indptr[j]);
            for p in s..e {
                let v = a.data[p];
                if v.abs() > eps {
                    indices[dst] = a.indices[p];
                    data[dst] = v;
                    dst += 1;
                }
            }
        }
        return Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data);
    }
    let ncols = a.ncols;
    let mut counts = vec![0usize; ncols];
    counts.par_iter_mut().enumerate().for_each(|(j, c)| {
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let col = &a.data[s..e];
        let mut cnt = 0usize;
        let mut k = 0usize;
        let limit4 = col.len() & !3;
        while k < limit4 {
            let v = unsafe {
                let p = col.as_ptr().add(k).cast::<[f64; 4]>();
                f64x4::new(core::ptr::read_unaligned(p))
            };
            let arr = v.to_array();
            if arr[0].abs() > eps {
                cnt += 1;
            }
            if arr[1].abs() > eps {
                cnt += 1;
            }
            if arr[2].abs() > eps {
                cnt += 1;
            }
            if arr[3].abs() > eps {
                cnt += 1;
            }
            k += 4;
        }
        while k < col.len() {
            if col[k].abs() > eps {
                cnt += 1;
            }
            k += 1;
        }
        *c = cnt;
    });
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
        let s = i64_to_usize(a.indptr[j]);
        let e = i64_to_usize(a.indptr[j + 1]);
        let mut dst = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(j) });
        unsafe {
            let pi = pi_addr as *mut i64;
            let pv = pv_addr as *mut f64;
            for p in s..e {
                let v = a.data[p];
                if v.abs() > eps {
                    std::ptr::write(pi.add(dst), a.indices[p]);
                    std::ptr::write(pv.add(dst), v);
                    dst += 1;
                }
            }
        }
    });
    Csc::from_parts_unchecked(a.nrows, ncols, indptr, indices, data)
}

#[must_use]
pub fn eliminate_zeros_coo(a: &Coo<f64, i64>) -> Coo<f64, i64> {
    prune_eps_coo(a, 0.0)
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

#[inline]
fn usize_to_i64(x: usize) -> i64 {
    debug_assert!(i64::try_from(x).is_ok());
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    {
        x as i64
    }
}

/// eliminate exact zeros (returns new CSR)
#[must_use]
pub fn eliminate_zeros(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    prune_eps(a, 0.0)
}

/// prune entries with |v| <= eps (returns new CSR)
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn prune_eps(a: &Csr<f64, i64>, eps: f64) -> Csr<f64, i64> {
    // Early exits
    if eps < 0.0 {
        return a.clone();
    }
    if eps == 0.0 {
        // If no zeros, structure unchanged
        let has_zero = a.data.par_iter().any(|&v| v == 0.0);
        if !has_zero {
            return a.clone();
        }
    }
    let nnz_total = a.data.len();
    if nnz_total < SMALL_NNZ_PRUNE {
        let nrows = a.nrows;
        let mut indptr = vec![0i64; nrows + 1];
        for i in 0..nrows {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let row = &a.data[s..e];
            let mut cnt = 0usize;
            let mut k = 0usize;
            let limit4 = row.len() & !3;
            while k < limit4 {
                let v = unsafe {
                    let p = row.as_ptr().add(k).cast::<[f64; 4]>();
                    f64x4::new(core::ptr::read_unaligned(p))
                };
                let arr = v.to_array();
                if arr[0].abs() > eps {
                    cnt += 1;
                }
                if arr[1].abs() > eps {
                    cnt += 1;
                }
                if arr[2].abs() > eps {
                    cnt += 1;
                }
                if arr[3].abs() > eps {
                    cnt += 1;
                }
                k += 4;
            }
            while k < row.len() {
                if row[k].abs() > eps {
                    cnt += 1;
                }
                k += 1;
            }
            indptr[i + 1] = indptr[i] + usize_to_i64(cnt);
        }
        let nnz = i64_to_usize(indptr[nrows]);
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        for i in 0..nrows {
            let s = i64_to_usize(a.indptr[i]);
            let e = i64_to_usize(a.indptr[i + 1]);
            let mut dst = i64_to_usize(indptr[i]);
            for p in s..e {
                let v = a.data[p];
                if v.abs() > eps {
                    indices[dst] = a.indices[p];
                    data[dst] = v;
                    dst += 1;
                }
            }
        }
        return Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data);
    }
    let nrows = a.nrows;
    let mut counts = vec![0usize; nrows];
    counts.par_iter_mut().enumerate().for_each(|(i, c)| {
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let row = &a.data[s..e];
        let mut cnt = 0usize;
        let mut k = 0usize;
        let limit4 = row.len() & !3;
        while k < limit4 {
            let v = unsafe {
                let p = row.as_ptr().add(k).cast::<[f64; 4]>();
                f64x4::new(core::ptr::read_unaligned(p))
            };
            let arr = v.to_array();
            if arr[0].abs() > eps {
                cnt += 1;
            }
            if arr[1].abs() > eps {
                cnt += 1;
            }
            if arr[2].abs() > eps {
                cnt += 1;
            }
            if arr[3].abs() > eps {
                cnt += 1;
            }
            k += 4;
        }
        while k < row.len() {
            if row[k].abs() > eps {
                cnt += 1;
            }
            k += 1;
        }
        *c = cnt;
    });
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
        let s = i64_to_usize(a.indptr[i]);
        let e = i64_to_usize(a.indptr[i + 1]);
        let mut dst = i64_to_usize(unsafe { *(indptr_addr as *const i64).add(i) });
        unsafe {
            let pi = pi_addr as *mut i64;
            let pv = pv_addr as *mut f64;
            for p in s..e {
                let v = a.data[p];
                if v.abs() > eps {
                    std::ptr::write(pi.add(dst), a.indices[p]);
                    std::ptr::write(pv.add(dst), v);
                    dst += 1;
                }
            }
        }
    });
    Csr::from_parts_unchecked(nrows, a.ncols, indptr, indices, data)
}
