use rayon::prelude::*;
use lacuna_core::Csr;
use wide::f64x4;

pub fn mul_scalar_f64(a: &Csr<f64, i64>, alpha: f64) -> Csr<f64, i64> {
    let mut out = a.clone();
    let aval = f64x4::splat(alpha);
    out.data.par_chunks_mut(1024).for_each(|chunk| {
        let mut i = 0usize;
        let limit4 = chunk.len() & !3;
        while i < limit4 {
            // load
            let v = f64x4::from([chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]]);
            let r = v * aval;
            let arr = r.to_array();
            chunk[i] = arr[0];
            chunk[i + 1] = arr[1];
            chunk[i + 2] = arr[2];
            chunk[i + 3] = arr[3];
            i += 4;
        }
        while i < chunk.len() {
            chunk[i] *= alpha;
            i += 1;
        }
    });
    out
}

pub fn add_csr_f64_i64(a: &Csr<f64, i64>, b: &Csr<f64, i64>) -> Csr<f64, i64> {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let nrows = a.nrows;
    // Pass 1: count output nnz per row in parallel
    let counts: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let mut pa = a.indptr[i] as usize;
            let ea = a.indptr[i + 1] as usize;
            let mut pb = b.indptr[i] as usize;
            let eb = b.indptr[i + 1] as usize;
            let mut cnt = 0usize;
            while pa < ea || pb < eb {
                if pb >= eb || (pa < ea && a.indices[pa] <= b.indices[pb]) {
                    let j = a.indices[pa];
                    // accumulate duplicates in A
                    let mut v = 0.0f64;
                    while pa < ea && a.indices[pa] == j { v += a.data[pa]; pa += 1; }
                    // accumulate matching indices in B as well
                    if pb < eb && b.indices[pb] == j {
                        while pb < eb && b.indices[pb] == j { v += b.data[pb]; pb += 1; }
                    }
                    if v != 0.0 { cnt += 1; }
                } else {
                    let j = b.indices[pb];
                    // accumulate duplicates in B
                    let mut v = 0.0f64;
                    while pb < eb && b.indices[pb] == j { v += b.data[pb]; pb += 1; }
                    if v != 0.0 { cnt += 1; }
                }
            }
            cnt
        })
        .collect();

    // Prefix sum -> indptr
    let mut indptr = vec![0i64; nrows + 1];
    for i in 0..nrows { indptr[i + 1] = indptr[i] + counts[i] as i64; }
    let nnz = indptr[nrows] as usize;
    let mut indices = vec![0i64; nnz];
    let mut data = vec![0.0f64; nnz];
    let pi_addr = indices.as_mut_ptr() as usize;
    let pv_addr = data.as_mut_ptr() as usize;
    let indptr_addr = indptr.as_ptr() as usize;

    // Pass 2: fill rows in parallel
    (0..nrows).into_par_iter().for_each(move |i| {
        let mut pa = a.indptr[i] as usize;
        let ea = a.indptr[i + 1] as usize;
        let mut pb = b.indptr[i] as usize;
        let eb = b.indptr[i + 1] as usize;
        let mut dst = unsafe { *(indptr_addr as *const i64).add(i) } as usize;
        unsafe {
            let pi = pi_addr as *mut i64;
            let pv = pv_addr as *mut f64;
            while pa < ea || pb < eb {
                if pb >= eb || (pa < ea && a.indices[pa] <= b.indices[pb]) {
                    let j = a.indices[pa];
                    // accumulate duplicates in A
                    let mut v = 0.0f64;
                    while pa < ea && a.indices[pa] == j { v += a.data[pa]; pa += 1; }
                    // accumulate matching indices in B as well
                    if pb < eb && b.indices[pb] == j {
                        while pb < eb && b.indices[pb] == j { v += b.data[pb]; pb += 1; }
                    }
                    if v != 0.0 {
                        std::ptr::write(pi.add(dst), j);
                        std::ptr::write(pv.add(dst), v);
                        dst += 1;
                    }
                } else {
                    let j = b.indices[pb];
                    // accumulate duplicates in B
                    let mut v = 0.0f64;
                    while pb < eb && b.indices[pb] == j { v += b.data[pb]; pb += 1; }
                    if v != 0.0 {
                        std::ptr::write(pi.add(dst), j);
                        std::ptr::write(pv.add(dst), v);
                        dst += 1;
                    }
                }
            }
        }
    });
    Csr { nrows, ncols: a.ncols, indptr, indices, data }
}
