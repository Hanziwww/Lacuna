use rayon::prelude::*;
use lacuna_core::Csr;
use wide::f64x4;

/// sum of all data
pub fn sum_f64(a: &Csr<f64, i64>) -> f64 {
    // Parallel reduce with SIMD inside chunks
    a.data
        .par_chunks(4096)
        .map(|chunk| {
            let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
            let mut i = 0usize;
            let limit4 = chunk.len() & !3;
            while i < limit4 {
                let v = f64x4::from([chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]]);
                accv = accv + v;
                i += 4;
            }
            let arr = accv.to_array();
            let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
            while i < chunk.len() {
                acc += chunk[i];
                i += 1;
            }
            acc
        })
        .sum()
}

/// row sums
pub fn row_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let mut out = vec![0.0f64; a.nrows];
    out.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let s = a.indptr[i] as usize;
        let e = a.indptr[i + 1] as usize;
        let row = &a.data[s..e];
        let mut accv = f64x4::from([0.0, 0.0, 0.0, 0.0]);
        let mut k = 0usize;
        let limit4 = row.len() & !3;
        while k < limit4 {
            let v = f64x4::from([row[k], row[k + 1], row[k + 2], row[k + 3]]);
            accv = accv + v;
            k += 4;
        }
        let arr = accv.to_array();
        let mut acc = arr[0] + arr[1] + arr[2] + arr[3];
        while k < row.len() { acc += row[k]; k += 1; }
        *oi = acc;
    });
    out
}

/// column sums (parallelized and vectorized)
pub fn col_sums_f64(a: &Csr<f64, i64>) -> Vec<f64> {
    let ncols = a.ncols;
    (0..a.nrows)
        .into_par_iter()
        .fold(|| vec![0.0f64; ncols], |mut acc, i| {
            let s = a.indptr[i] as usize;
            let e = a.indptr[i + 1] as usize;
            for p in s..e {
                let j = a.indices[p] as usize;
                acc[j] += a.data[p];
            }
            acc
        })
        .reduce(|| vec![0.0f64; ncols], |mut a1, a2| {
            for c in 0..ncols {
                a1[c] += a2[c];
            }
            a1
        })
}
