use rayon::prelude::*;
use lacuna_core::Csr;
use wide::f64x4;

/// y = A @ x
pub fn spmv_f64_i64(a: &Csr<f64, i64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.ncols, "x length must equal ncols");
    let mut y = vec![0.0f64; a.nrows];
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let start = a.indptr[i] as usize;
        let end = a.indptr[i + 1] as usize;
        let mut acc = 0.0f64;
        let mut p = start;
        let limit4 = start + ((end - start) & !3);
        while p < limit4 {
            let j0 = a.indices[p] as usize;
            let j1 = a.indices[p + 1] as usize;
            let j2 = a.indices[p + 2] as usize;
            let j3 = a.indices[p + 3] as usize;
            let vx = f64x4::from([x[j0], x[j1], x[j2], x[j3]]);
            let va = f64x4::from([a.data[p], a.data[p + 1], a.data[p + 2], a.data[p + 3]]);
            let prod = va * vx;
            let arr = prod.to_array();
            acc += arr[0] + arr[1] + arr[2] + arr[3];
            p += 4;
        }
        while p < end {
            let j = a.indices[p] as usize;
            acc += a.data[p] * x[j];
            p += 1;
        }
        *yi = acc;
    });
    y
}
