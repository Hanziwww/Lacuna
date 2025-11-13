use lacuna_core::Csr;
use lacuna_kernels::*;

fn simple_csr() -> Csr<f64, i64> {
    // A = [[1,0,2],[0,3,0]]
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2, 3];
    let indices = vec![0i64, 2, 1];
    let data = vec![1.0f64, 2.0, 3.0];
    Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

#[test]
fn test_spmv() {
    let a = simple_csr();
    let x = vec![10.0, 20.0, 30.0];
    let y = spmv_f64_i64(&a, &x);
    assert!(approx_eq(y[0], 1.0*10.0 + 2.0*30.0));
    assert!(approx_eq(y[1], 3.0*20.0));
}

#[test]
fn test_spmm() {
    let a = simple_csr();
    // B row-major (3x2): [[1,2],[3,4],[5,6]]
    let b = vec![1.0,2.0, 3.0,4.0, 5.0,6.0];
    let y = spmm_f64_i64(&a, &b, 2);
    assert_eq!(y.len(), 2*2);
    assert!(approx_eq(y[0], 11.0) && approx_eq(y[1], 14.0));
    assert!(approx_eq(y[2], 9.0) && approx_eq(y[3], 12.0));
}

#[test]
fn test_reductions() {
    let a = simple_csr();
    assert!(approx_eq(sum_f64(&a), 6.0));
    let rs = row_sums_f64(&a);
    assert!(approx_eq(rs[0], 3.0) && approx_eq(rs[1], 3.0));
    let cs = col_sums_f64(&a);
    assert!(approx_eq(cs[0], 1.0) && approx_eq(cs[1], 3.0) && approx_eq(cs[2], 2.0));
}

#[test]
fn test_transpose() {
    let a = simple_csr();
    let t = transpose_f64_i64(&a);
    assert_eq!(t.nrows, 3);
    assert_eq!(t.ncols, 2);
    // Expected CSR for A^T: indptr [0,1,2,3], indices [0,1,0], data [1,3,2]
    assert_eq!(t.indptr, vec![0i64, 1, 2, 3]);
    assert_eq!(t.indices, vec![0i64, 1, 0]);
    assert!(approx_eq(t.data[0], 1.0) && approx_eq(t.data[1], 3.0) && approx_eq(t.data[2], 2.0));
}

#[test]
fn test_prune_and_eliminate() {
    let nrows = 1usize; let ncols = 3usize;
    let indptr = vec![0i64, 3];
    let indices = vec![0i64, 1, 2];
    let data = vec![1.0f64, 0.0, 1e-9];
    let a = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap();
    let az = eliminate_zeros(&a);
    assert_eq!(az.nnz(), 2);
    let ap = prune_eps(&a, 1e-6);
    assert_eq!(ap.nnz(), 1);
}

#[test]
fn test_add_and_scalar_mul() {
    let a = simple_csr();
    let c = add_csr_f64_i64(&a, &a);
    assert_eq!(c.nrows, a.nrows);
    assert_eq!(c.indices, a.indices); // structure same
    for (vd, vs) in c.data.iter().zip(a.data.iter()) {
        assert!(approx_eq(*vd, 2.0 * *vs));
    }
    let m = mul_scalar_f64(&a, 2.0);
    for (vd, vs) in m.data.iter().zip(a.data.iter()) { assert!(approx_eq(*vd, 2.0 * *vs)); }
}

#[test]
fn test_add_coalesces_duplicates() {
    // A has duplicates in a single row (allowed with check=false)
    let nrows = 1usize; let ncols = 3usize;
    let indptr = vec![0i64, 3];
    let indices = vec![0i64, 0, 2]; // duplicate column 0
    let data = vec![1.0f64, 3.0, 2.0];
    let a = Csr::from_parts(nrows, ncols, indptr.clone(), indices, data, false).unwrap();
    // B is empty
    let b = Csr::from_parts(nrows, ncols, vec![0i64, 0], vec![], vec![], true).unwrap();
    let c = add_csr_f64_i64(&a, &b);
    // Expect coalesced: indices [0,2], data [4,2]
    assert_eq!(c.indptr, vec![0i64, 2]);
    assert_eq!(c.indices, vec![0i64, 2]);
    assert!(approx_eq(c.data[0], 4.0) && approx_eq(c.data[1], 2.0));
}
