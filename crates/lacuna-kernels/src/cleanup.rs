use lacuna_core::Csr;
use rayon::prelude::*;

/// eliminate exact zeros (returns new CSR)
pub fn eliminate_zeros(a: &Csr<f64, i64>) -> Csr<f64, i64> {
    prune_eps(a, 0.0)
}

/// prune entries with |v| <= eps (returns new CSR)
pub fn prune_eps(a: &Csr<f64, i64>, eps: f64) -> Csr<f64, i64> {
    let nrows = a.nrows;
    let mut counts = vec![0usize; nrows];
    counts.par_iter_mut().enumerate().for_each(|(i, c)| {
        let s = a.indptr[i] as usize;
        let e = a.indptr[i + 1] as usize;
        let mut cnt = 0usize;
        for p in s..e {
            if a.data[p].abs() > eps { cnt += 1; }
        }
        *c = cnt;
    });
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows { indptr[i + 1] = indptr[i] + counts[i] as i64; }
    let nnz = indptr[nrows] as usize;
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;
    (0..nrows).into_par_iter().for_each(move |i| {
        let s = a.indptr[i] as usize;
        let e = a.indptr[i + 1] as usize;
        let mut dst = unsafe { *(indptr_addr as *const i64).add(i) } as usize;
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
    Csr { nrows, ncols: a.ncols, indptr, indices, data }
}
