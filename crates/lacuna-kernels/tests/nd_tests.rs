use lacuna_core::CooNd;
use lacuna_kernels::*;

fn sample_coond() -> CooNd<f64, i64> {
    // shape: [2, 3, 2]
    // entries: at (0,0,0)=1, (0,2,1)=2, (1,1,0)=3
    let shape = vec![2usize, 3usize, 2usize];
    let indices = vec![
        0, 0, 0, // k=0
        0, 2, 1, // k=1
        1, 1, 0, // k=2
    ];
    let data = vec![1.0f64, 2.0, 3.0];
    CooNd::from_parts(shape, indices, data, true).unwrap()
}

#[test]
fn test_mean_and_reduce_mean_coond() {
    let a = sample_coond();
    let total_elems: usize = a.shape.iter().product();
    let m = mean_coond_f64(&a);
    let s = sum_coond_f64(&a);
    assert!((m - s / (total_elems as f64)).abs() < 1e-12);

    let rsum = reduce_sum_axes_coond_f64_i64(&a, &[2]);
    let rmean = reduce_mean_axes_coond_f64_i64(&a, &[2]);
    let denom = a.shape[2] as f64;
    assert_eq!(rsum.shape, rmean.shape);
    assert!((sum_coond_f64(&rmean) - sum_coond_f64(&rsum) / denom).abs() < 1e-12);
}

#[test]
fn test_reshape_coond() {
    let a = sample_coond();
    let s = sum_coond_f64(&a);
    let new_shape = vec![3usize, 2usize, 2usize];
    let r = reshape_coond_f64_i64(&a, &new_shape);
    assert_eq!(r.shape, new_shape);
    assert!((sum_coond_f64(&r) - s).abs() < 1e-12);
}

#[test]
fn test_hadamard_broadcast_coond() {
    // a: [2,3,2] same as sample, with values [1,2,3]
    let a = sample_coond();
    // b: [1,3,1] with values per j: [10, 20, 30] => will scale a by factor depending on axis-1
    let b_shape = vec![1usize, 3usize, 1usize];
    let b_indices = vec![0i64, 0, 0, 0, 1, 0, 0, 2, 0];
    let b_data = vec![10.0f64, 20.0, 30.0];
    let b = CooNd::from_parts(b_shape, b_indices, b_data, true).unwrap();
    let h = hadamard_broadcast_coond_f64_i64(&a, &b);
    assert_eq!(h.shape, vec![2usize, 3usize, 2usize]);
    // expected sum: a at (0,0,0)=1*10 + (0,2,1)=2*30 + (1,1,0)=3*20 = 10 + 60 + 60 = 130
    assert!((sum_coond_f64(&h) - 130.0).abs() < 1e-12);
}

#[test]
fn test_sum_coond() {
    let a = sample_coond();
    let s = sum_coond_f64(&a);
    assert!((s - 6.0).abs() < 1e-12);
}

#[test]
fn test_reduce_sum_axes_coond() {
    let a = sample_coond();
    // reduce over axis 2 (last), remain axes [0,1] => shape [2,3]
    let r = reduce_sum_axes_coond_f64_i64(&a, &[2]);
    assert_eq!(r.shape, vec![2usize, 3usize]);
    // total sum should stay the same
    assert!((sum_coond_f64(&r) - sum_coond_f64(&a)).abs() < 1e-12);
}

#[test]
fn test_permute_axes_coond() {
    let a = sample_coond();
    // permute [1,0,2] => shape [3,2,2]
    let p = permute_axes_coond_f64_i64(&a, &[1, 0, 2]);
    assert_eq!(p.shape, vec![3usize, 2usize, 2usize]);
    assert!((sum_coond_f64(&p) - sum_coond_f64(&a)).abs() < 1e-12);
}

#[test]
fn test_spmv_spmm_coond_shapes_and_sums() {
    let a = sample_coond();
    // spmv along axis 1 with x len=3 -> result shape keeps other axes [0,2] => [2,2]
    let x = vec![1.0, 2.0, 3.0];
    let y = spmv_coond_f64_i64(&a, 1, &x);
    assert_eq!(y.shape, vec![2usize, 2usize]);
    // spmm along axis 1 with k=4 -> replace axis 1 by k
    let b = vec![
        1.0, 0.0, 0.0, 0.0, // row0
        0.0, 1.0, 0.0, 0.0, // row1
        0.0, 0.0, 1.0, 0.0, // row2
    ];
    let z = spmm_coond_f64_i64(&a, 1, &b, 4);
    assert_eq!(z.shape, vec![2usize, 4usize, 2usize]);
    assert!((sum_coond_f64(&z) - sum_coond_f64(&a)).abs() < 1e-12);
}

#[test]
fn test_coond_convert_mode_and_axes() {
    let a = sample_coond();
    // mode-0 unfolding => rows=shape[0]=2, cols=shape[1]*shape[2]=6
    let csr0 = coond_mode_to_csr_f64_i64(&a, 0);
    assert_eq!(csr0.nrows, 2);
    assert_eq!(csr0.ncols, 6);
    assert!((sum_f64(&csr0) - sum_coond_f64(&a)).abs() < 1e-12);

    // mode-1 unfolding => rows=3, cols=4
    let csc1 = coond_mode_to_csc_f64_i64(&a, 1);
    assert_eq!(csc1.nrows, 3);
    assert_eq!(csc1.ncols, 4);
    // sum check via CSC->COO and sum over data
    let cs = csc_to_coo_f64_i64(&csc1);
    let s: f64 = cs.data.iter().sum();
    assert!((s - sum_coond_f64(&a)).abs() < 1e-12);

    // axes unfolding: rows=[0,2], cols=[1]
    let csr_axes = coond_axes_to_csr_f64_i64(&a, &[0, 2]);
    assert_eq!(csr_axes.nrows, 2 * 2);
    assert_eq!(csr_axes.ncols, 3);
    assert!((sum_f64(&csr_axes) - sum_coond_f64(&a)).abs() < 1e-12);
}
