use rayon::prelude::*;
use lacuna_core::Csr;
use wide::f64x4;

/// Y = A @ B, where B is (ncols, k) row-major; returns Y as (nrows, k) row-major
pub fn spmm_f64_i64(a: &Csr<f64, i64>, b: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(b.len(), a.ncols * k, "B must be ncols x k row-major");
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut y = vec![0.0f64; nrows * k];

    // Process per row in parallel; within row, use SIMD across k.
    y.par_chunks_mut(k).enumerate().for_each(|(i, yi)| {
        let start = a.indptr[i] as usize;
        let end = a.indptr[i + 1] as usize;
        let _ = ncols;
        for p in start..end {
            let j = a.indices[p] as usize;
            let aij = a.data[p];
            let row_b = &b[j * k..j * k + k];
            let mut c = 0usize;
            let limit4 = k.saturating_sub(k % 4);
            let aijv = f64x4::splat(aij);
            while c < limit4 {
                let vb = f64x4::from([row_b[c], row_b[c + 1], row_b[c + 2], row_b[c + 3]]);
                let vy = f64x4::from([yi[c], yi[c + 1], yi[c + 2], yi[c + 3]]);
                let vz = vy + vb * aijv;
                let arr = vz.to_array();
                yi[c] = arr[0];
                yi[c + 1] = arr[1];
                yi[c + 2] = arr[2];
                yi[c + 3] = arr[3];
                c += 4;
            }
            while c < k {
                yi[c] += aij * row_b[c];
                c += 1;
            }
        }
    });
    y
}
