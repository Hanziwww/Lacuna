use lacuna_core::{Coo, CooNd, Csc, Csr};
use lacuna_kernels::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn test_pow_csr_pairwise_basic() {
    // A = [[1,0,2],[0,3,0]]
    let a = Csr::from_parts(
        2,
        3,
        vec![0i64, 2, 3],
        vec![0i64, 2, 1],
        vec![1.0, 2.0, 3.0],
        true,
    )
    .unwrap();
    // B (exponents) same sparsity: [2,2,1] aligned
    let b = Csr::from_parts(
        2,
        3,
        vec![0i64, 2, 3],
        vec![0i64, 2, 1],
        vec![2.0, 2.0, 1.0],
        true,
    )
    .unwrap();
    let c = pow_csr_f64_i64(&a, &b);
    assert_eq!(c.indptr, vec![0i64, 2, 3]);
    assert_eq!(c.indices, vec![0i64, 2, 1]);
    // [1^2=1, 2^2=4, 3^1=3]
    assert!(approx_eq(c.data[0], 1.0));
    assert!(approx_eq(c.data[1], 4.0));
    assert!(approx_eq(c.data[2], 3.0));
}

#[test]
fn test_pow_csr_scalar_basic() {
    let a = Csr::from_parts(
        2,
        3,
        vec![0i64, 2, 3],
        vec![0i64, 2, 1],
        vec![1.0, 2.0, 3.0],
        true,
    )
    .unwrap();
    let r = pow_scalar_f64(&a, 2.0);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 4.0));
    assert!(approx_eq(r.data[2], 9.0));
}

#[test]
fn test_pow_csc_pairwise_basic() {
    // A (CSC) for [[1,0,2],[0,3,0]]
    let a = Csc::from_parts(
        2,
        3,
        vec![0i64, 1, 2, 3],
        vec![0i64, 1, 0],
        vec![1.0, 3.0, 2.0],
        true,
    )
    .unwrap();
    // B exponents same sparsity, values [2,1,2]
    let b = Csc::from_parts(
        2,
        3,
        vec![0i64, 1, 2, 3],
        vec![0i64, 1, 0],
        vec![2.0, 1.0, 2.0],
        true,
    )
    .unwrap();
    let c = pow_csc_f64_i64(&a, &b);
    assert_eq!(c.indptr, vec![0i64, 1, 2, 3]);
    assert_eq!(c.indices, vec![0i64, 1, 0]);
    // [1^2=1, 3^1=3, 2^2=4]
    assert!(approx_eq(c.data[0], 1.0));
    assert!(approx_eq(c.data[1], 3.0));
    assert!(approx_eq(c.data[2], 4.0));
}

#[test]
fn test_pow_csc_scalar_basic() {
    let a = Csc::from_parts(
        2,
        3,
        vec![0i64, 1, 2, 3],
        vec![0i64, 1, 0],
        vec![1.0, 3.0, 2.0],
        true,
    )
    .unwrap();
    let r = pow_scalar_csc_f64(&a, 2.0);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 9.0));
    assert!(approx_eq(r.data[2], 4.0));
}

#[test]
fn test_pow_coo_scalar_basic() {
    let a = Coo::from_parts(
        2,
        3,
        vec![0i64, 1, 1],
        vec![0i64, 0, 2],
        vec![1.0, 2.0, 3.0],
        true,
    )
    .unwrap();
    let r = pow_scalar_coo_f64(&a, 2.0);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 4.0));
    assert!(approx_eq(r.data[2], 9.0));
}

#[test]
fn test_pow_coond_scalar_basic() {
    let shape = vec![2usize, 3usize, 2usize];
    let indices = vec![0, 0, 0, 0, 2, 1, 1, 1, 0];
    let data = vec![1.0f64, 2.0, 3.0];
    let a = CooNd::from_parts(shape, indices, data, true).unwrap();
    let r = pow_scalar_coond_f64(&a, 2.0);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 4.0));
    assert!(approx_eq(r.data[2], 9.0));
}
