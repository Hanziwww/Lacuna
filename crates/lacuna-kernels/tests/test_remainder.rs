use lacuna_core::{Coo, CooNd, Csc, Csr};
use lacuna_kernels::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn test_rem_csr_pairwise_basic() {
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
    // B has same sparsity structure; values [2,5,2]
    let b = Csr::from_parts(
        2,
        3,
        vec![0i64, 2, 3],
        vec![0i64, 2, 1],
        vec![2.0, 5.0, 2.0],
        true,
    )
    .unwrap();
    let c = rem_csr_f64_i64(&a, &b);
    assert_eq!(c.nrows, 2);
    assert_eq!(c.ncols, 3);
    assert_eq!(c.indptr, vec![0i64, 2, 3]);
    assert_eq!(c.indices, vec![0i64, 2, 1]);
    // elementwise remainder where both have entries: [1%2=1, 2%5=2, 3%2=1]
    assert!(approx_eq(c.data[0], 1.0));
    assert!(approx_eq(c.data[1], 2.0));
    assert!(approx_eq(c.data[2], 1.0));
}

#[test]
fn test_rem_csr_scalar_basic() {
    let a = Csr::from_parts(
        2,
        3,
        vec![0i64, 2, 3],
        vec![0i64, 2, 1],
        vec![1.0, 2.0, 3.0],
        true,
    )
    .unwrap();
    let r = rem_scalar_f64(&a, 2.0);
    assert_eq!(r.indptr, a.indptr);
    assert_eq!(r.indices, a.indices);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 0.0));
    assert!(approx_eq(r.data[2], 1.0));
}

#[test]
fn test_rem_csc_pairwise_basic() {
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
    let b = Csc::from_parts(
        2,
        3,
        vec![0i64, 1, 2, 3],
        vec![0i64, 1, 0],
        vec![2.0, 2.0, 5.0],
        true,
    )
    .unwrap();
    let c = rem_csc_f64_i64(&a, &b);
    assert_eq!(c.indptr, vec![0i64, 1, 2, 3]);
    assert_eq!(c.indices, vec![0i64, 1, 0]);
    // col0: 1%2=1, col1: 3%2=1, col2: 2%5=2
    assert!(approx_eq(c.data[0], 1.0));
    assert!(approx_eq(c.data[1], 1.0));
    assert!(approx_eq(c.data[2], 2.0));
}

#[test]
fn test_rem_csc_scalar_basic() {
    let a = Csc::from_parts(
        2,
        3,
        vec![0i64, 1, 2, 3],
        vec![0i64, 1, 0],
        vec![1.0, 3.0, 2.0],
        true,
    )
    .unwrap();
    let r = rem_scalar_csc_f64(&a, 2.0);
    assert_eq!(r.indptr, a.indptr);
    assert_eq!(r.indices, a.indices);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 1.0));
    assert!(approx_eq(r.data[2], 0.0));
}

#[test]
fn test_rem_coo_scalar_basic() {
    // COO for [[1,0,0],[2,0,3]]
    let a = Coo::from_parts(
        2,
        3,
        vec![0i64, 1, 1],
        vec![0i64, 0, 2],
        vec![1.0, 2.0, 3.0],
        true,
    )
    .unwrap();
    let r = rem_scalar_coo_f64(&a, 2.0);
    assert_eq!(r.nrows, 2);
    assert_eq!(r.ncols, 3);
    // data: [1%2=1, 2%2=0, 3%2=1]
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 0.0));
    assert!(approx_eq(r.data[2], 1.0));
}

#[test]
fn test_rem_coond_scalar_basic() {
    // shape [2,3,2], indices for 3 entries
    let shape = vec![2usize, 3usize, 2usize];
    let indices = vec![
        0, 0, 0, // 1.0
        0, 2, 1, // 2.0
        1, 1, 0, // 3.0
    ];
    let data = vec![1.0f64, 2.0, 3.0];
    let a = CooNd::from_parts(shape, indices, data, true).unwrap();
    let r = rem_scalar_coond_f64(&a, 2.0);
    assert_eq!(r.shape, vec![2usize, 3usize, 2usize]);
    assert!(approx_eq(r.data[0], 1.0));
    assert!(approx_eq(r.data[1], 0.0));
    assert!(approx_eq(r.data[2], 1.0));
}
