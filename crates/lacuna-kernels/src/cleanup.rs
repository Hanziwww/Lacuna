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
use lacuna_core::Csr;
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
                let v = f64x4::from([row[k], row[k + 1], row[k + 2], row[k + 3]]);
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
            let v = f64x4::from([row[k], row[k + 1], row[k + 2], row[k + 3]]);
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
