//! Permute dimensions for COOND

use lacuna_core::CooNd;

#[must_use]
pub fn permute_axes_coond_f64_i64(a: &CooNd<f64, i64>, perm: &[usize]) -> CooNd<f64, i64> {
    let ndim = a.shape.len();
    assert_eq!(perm.len(), ndim, "perm length must equal ndim");
    let mut seen = vec![false; ndim];
    for &p in perm {
        assert!(p < ndim, "perm index out of bounds");
        assert!(!seen[p], "perm must be a permutation without duplicates");
        seen[p] = true;
    }

    let new_shape: Vec<usize> = (0..ndim).map(|d| a.shape[perm[d]]).collect();
    let nnz = a.data.len();
    if nnz == 0 {
        return CooNd::from_parts_unchecked(new_shape, Vec::new(), Vec::new());
    }
    let mut new_indices = vec![0i64; nnz * ndim];
    let src = &a.indices;
    for k in 0..nnz {
        let base_src = k * ndim;
        let base_dst = base_src;
        for d in 0..ndim {
            let src_d = perm[d];
            new_indices[base_dst + d] = src[base_src + src_d];
        }
    }
    CooNd::from_parts_unchecked(new_shape, new_indices, a.data.clone())
}
