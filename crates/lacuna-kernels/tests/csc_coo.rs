use lacuna_core::{Coo, Csc};
use lacuna_kernels::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn test_csc_spmm() {
    let a = simple_csc();
    // B row-major (3x2): [[1,2],[3,4],[5,6]]
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = spmm_csc_f64_i64(&a, &b, 2);
    assert_eq!(y.len(), 2 * 2);
    assert!(approx_eq(y[0], 11.0) && approx_eq(y[1], 14.0));
    assert!(approx_eq(y[2], 9.0) && approx_eq(y[3], 12.0));
}

#[test]
fn test_coo_spmm() {
    let a = simple_coo();
    // B row-major (3x2): [[1,2],[3,4],[5,6]]
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = spmm_coo_f64_i64(&a, &b, 2);
    assert_eq!(y.len(), 2 * 2);
    assert!(approx_eq(y[0], 11.0) && approx_eq(y[1], 14.0));
    assert!(approx_eq(y[2], 9.0) && approx_eq(y[3], 12.0));
}

fn simple_csc() -> Csc<f64, i64> {
    // A = [[1,0,2],[0,3,0]] in CSC
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 1, 2, 3];
    let indices = vec![0i64, 1, 0];
    let data = vec![1.0f64, 3.0, 2.0];
    Csc::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

fn simple_coo() -> Coo<f64, i64> {
    // A = [[1,0,2],[0,3,0]] in COO
    let nrows = 2usize;
    let ncols = 3usize;
    let row = vec![0i64, 1, 0];
    let col = vec![0i64, 1, 2];
    let data = vec![1.0f64, 3.0, 2.0];
    Coo::from_parts(nrows, ncols, row, col, data, true).unwrap()
}

#[test]
fn test_csc_spmv_and_reductions() {
    let a = simple_csc();
    let x = vec![10.0, 20.0, 30.0];
    let y = spmv_csc_f64_i64(&a, &x);
    assert!(approx_eq(y[0], 70.0));
    assert!(approx_eq(y[1], 60.0));

    assert!(approx_eq(sum_csc_f64(&a), 6.0));
    let rs = row_sums_csc_f64(&a);
    assert!(approx_eq(rs[0], 3.0) && approx_eq(rs[1], 3.0));
    let cs = col_sums_csc_f64(&a);
    assert!(approx_eq(cs[0], 1.0) && approx_eq(cs[1], 3.0) && approx_eq(cs[2], 2.0));
}

#[test]
fn test_csc_cleanup_and_arith() {
    // construct with a zero and tiny element to prune/eliminate
    let nrows = 2usize;
    let ncols = 2usize;
    let indptr = vec![0i64, 1, 3];
    let indices = vec![0i64, 0, 1];
    let data = vec![1.0f64, 0.0, 1e-9];
    let a = Csc::from_parts(nrows, ncols, indptr, indices, data, true).unwrap();

    let az = eliminate_zeros_csc(&a);
    assert_eq!(az.nnz(), 2);
    let ap = prune_eps_csc(&a, 1e-6);
    assert_eq!(ap.nnz(), 1);

    let s = simple_csc();
    let m = mul_scalar_csc_f64(&s, 2.0);
    assert!(
        m.data
            .iter()
            .zip(s.data.iter())
            .all(|(d, s)| approx_eq(*d, 2.0 * *s))
    );

    let addv = add_csc_f64_i64(&s, &s);
    assert!(
        addv.data
            .iter()
            .zip(s.data.iter())
            .all(|(d, sv)| approx_eq(*d, 2.0 * *sv))
    );

    let subz = sub_csc_f64_i64(&s, &s);
    assert_eq!(subz.nnz(), 0);

    let had = hadamard_csc_f64_i64(&s, &s);
    assert_eq!(had.indptr, s.indptr);
    assert_eq!(had.indices, s.indices);
    for (hv, sv) in had.data.iter().zip(s.data.iter()) {
        assert!(approx_eq(*hv, *sv * *sv));
    }
}

#[test]
fn test_coo_spmv_and_reductions() {
    let a = simple_coo();
    let x = vec![10.0, 20.0, 30.0];
    let y = spmv_coo_f64_i64(&a, &x);
    assert!(approx_eq(y[0], 70.0));
    assert!(approx_eq(y[1], 60.0));

    assert!(approx_eq(sum_coo_f64(&a), 6.0));
    let rs = row_sums_coo_f64(&a);
    assert!(approx_eq(rs[0], 3.0) && approx_eq(rs[1], 3.0));
    let cs = col_sums_coo_f64(&a);
    assert!(approx_eq(cs[0], 1.0) && approx_eq(cs[1], 3.0) && approx_eq(cs[2], 2.0));
}

#[test]
fn test_coo_cleanup_and_scalar() {
    let nrows = 1usize;
    let ncols = 3usize;
    let row = vec![0i64, 0, 0];
    let col = vec![0i64, 1, 2];
    let data = vec![1.0f64, 0.0, 1e-9];
    let a = Coo::from_parts(nrows, ncols, row, col, data, true).unwrap();

    let az = eliminate_zeros_coo(&a);
    assert_eq!(az.nnz(), 2);
    let ap = prune_eps_coo(&a, 1e-6);
    assert_eq!(ap.nnz(), 1);

    let s = simple_coo();
    let m = mul_scalar_coo_f64(&s, 2.0);
    assert!(
        m.data
            .iter()
            .zip(s.data.iter())
            .all(|(d, sv)| approx_eq(*d, 2.0 * *sv))
    );
}
