use lacuna_core::Csr;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

/// Transpose CSR -> CSR (simple histogram-based)
pub fn transpose_f64_i64(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    let nrows = a.ncols;
    let ncols = a.nrows;
    let nnz = a.data.len();
    let mut indptr = vec![0i64; nrows + 1];
    for &j in &a.indices { indptr[j as usize + 1] += 1; }
    for i in 0..nrows { indptr[i + 1] += indptr[i]; }
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let next: Vec<AtomicI64> = indptr.iter().copied().map(AtomicI64::new).collect();
    (0..a.nrows).into_par_iter().for_each(move |i| {
        let s = a.indptr[i] as usize;
        let e = a.indptr[i + 1] as usize;
        for p in s..e {
            let j = a.indices[p] as usize;
            let dst = next[j].fetch_add(1, Ordering::Relaxed) as usize;
            unsafe {
                let pi = pi_addr as *mut i64;
                let pv = pv_addr as *mut f64;
                std::ptr::write(pi.add(dst), i as i64);
                std::ptr::write(pv.add(dst), a.data[p]);
            }
        }
    });
    let piw_addr = indices.as_mut_ptr() as usize;
    let pvw_addr = data.as_mut_ptr() as usize;
    let pir_addr = indices.as_ptr() as usize;
    let pvr_addr = data.as_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    (0..nrows).into_par_iter().for_each(move |j| {
        let s = unsafe { *(indptr_addr as *const i64).add(j) } as usize;
        let e = unsafe { *(indptr_addr as *const i64).add(j + 1) } as usize;
        if e > s {
            let mut pairs: Vec<(i64, f64)> = {
                let pir = pir_addr as *const i64;
                let pvr = pvr_addr as *const f64;
                (s..e)
                    .map(|t| unsafe { (*pir.add(t), *pvr.add(t)) })
                    .collect()
            };
            pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            unsafe {
                let piw = piw_addr as *mut i64;
                let pvw = pvw_addr as *mut f64;
                for (k, (ii, vv)) in pairs.into_iter().enumerate() {
                    std::ptr::write(piw.add(s + k), ii);
                    std::ptr::write(pvw.add(s + k), vv);
                }
            }
        }
    });
    Csr { nrows, ncols, indptr, indices, data }
}
