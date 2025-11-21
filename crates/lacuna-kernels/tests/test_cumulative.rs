use lacuna_core::Csr;
use lacuna_kernels::{
    csr_cumprod_dense_axis0_f64, csr_cumprod_dense_axis1_f64, csr_cumsum_dense_axis0_f64,
    csr_cumsum_dense_axis1_f64,
};

fn sample_csr() -> Csr<f64, i64> {
    // [[1.0, 0.0, 3.0],
    //  [0.0, -2.0, 0.0]]
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0, 2, 3];
    let indices = vec![0, 2, 1];
    let data = vec![1.0, 3.0, -2.0];
    Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

#[test]
fn test_cumsum_axis1_dense() {
    let a = sample_csr();
    let out = csr_cumsum_dense_axis1_f64(&a);
    // row-major (2x3): [[1,1,4],[0,-2,-2]]
    let expected = vec![1.0, 1.0, 4.0, 0.0, -2.0, -2.0];
    assert_eq!(out, expected);
}

#[test]
fn test_cumsum_axis0_dense() {
    let a = sample_csr();
    let out = csr_cumsum_dense_axis0_f64(&a);
    // [[1,0,3],[1,-2,3]]
    let expected = vec![1.0, 0.0, 3.0, 1.0, -2.0, 3.0];
    assert_eq!(out, expected);
}

#[test]
fn test_cumprod_axis1_dense() {
    let a = sample_csr();
    let out = csr_cumprod_dense_axis1_f64(&a);
    // [[1,0,0],[0,0,0]]
    let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(out, expected);
}

#[test]
fn test_cumprod_axis0_dense() {
    let a = sample_csr();
    let out = csr_cumprod_dense_axis0_f64(&a);
    // [[1,0,3],[0,0,0]]
    let expected = vec![1.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    assert_eq!(out, expected);
}
