use lacuna_core::{Coo, Csc, Csr};
use lacuna_kernels::{
    // prod
    prod_f64, row_prods_f64, col_prods_f64, prod_csc_f64, row_prods_csc_f64, col_prods_csc_f64,
    prod_coo_f64, row_prods_coo_f64, col_prods_coo_f64,
    // var/std (CSR)
    var_f64, std_f64, row_vars_f64, row_stds_f64, col_vars_f64, col_stds_f64,
    // var/std (CSC)
    var_csc_f64, std_csc_f64, row_vars_csc_f64, row_stds_csc_f64, col_vars_csc_f64, col_stds_csc_f64,
    // var/std (COO)
    var_coo_f64, std_coo_f64, row_vars_coo_f64, row_stds_coo_f64, col_vars_coo_f64, col_stds_coo_f64,
};

fn make_sample_csr() -> Csr<f64, i64> {
    // Dense:
    // [[1.0, 0.0, 3.0],
    //  [0.0, -2.0, 0.0]]
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0_i64, 2, 3];
    let indices = vec![0_i64, 2, 1];
    let data = vec![1.0_f64, 3.0, -2.0];
    Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

fn make_sample_csc() -> Csc<f64, i64> {
    // transpose of above
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0_i64, 1, 2, 3];
    let indices = vec![0_i64, 1, 0];
    let data = vec![1.0_f64, -2.0, 3.0];
    Csc::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

fn make_sample_coo() -> Coo<f64, i64> {
    let nrows = 2usize;
    let ncols = 3usize;
    let row = vec![0_i64, 0, 1];
    let col = vec![0_i64, 2, 1];
    let data = vec![1.0_f64, 3.0, -2.0];
    Coo::from_parts(nrows, ncols, row, col, data, true).unwrap()
}

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!((a - b).abs() <= eps, "{} !~= {}", a, b);
}

#[test]
fn test_prod_all_formats() {
    let csr = make_sample_csr();
    let csc = make_sample_csc();
    let coo = make_sample_coo();

    // global product should be 0 due to implicit zeros
    assert_eq!(prod_f64(&csr), 0.0);
    assert_eq!(prod_csc_f64(&csc), 0.0);
    assert_eq!(prod_coo_f64(&coo), 0.0);

    // axis=1 (rows)
    assert_eq!(row_prods_f64(&csr), vec![0.0, 0.0]);
    assert_eq!(row_prods_csc_f64(&csc), vec![0.0, 0.0]);
    assert_eq!(row_prods_coo_f64(&coo), vec![0.0, 0.0]);

    // axis=0 (cols)
    assert_eq!(col_prods_f64(&csr), vec![0.0, 0.0, 0.0]);
    assert_eq!(col_prods_csc_f64(&csc), vec![0.0, 0.0, 0.0]);
    assert_eq!(col_prods_coo_f64(&coo), vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_varstd_csr() {
    let csr = make_sample_csr();
    // Global var = 20/9, std = sqrt(20/9)
    approx_eq(var_f64(&csr, 0.0), 20.0 / 9.0, 1e-12);
    approx_eq(std_f64(&csr, 0.0), (20.0_f64 / 9.0_f64).sqrt(), 1e-12);

    // Row-wise var: [14/9, 8/9]
    let rv = row_vars_f64(&csr, 0.0);
    approx_eq(rv[0], 14.0 / 9.0, 1e-12);
    approx_eq(rv[1], 8.0 / 9.0, 1e-12);

    let rs = row_stds_f64(&csr, 0.0);
    approx_eq(rs[0], (14.0_f64 / 9.0_f64).sqrt(), 1e-12);
    approx_eq(rs[1], (8.0_f64 / 9.0_f64).sqrt(), 1e-12);

    // Col-wise var: [0.25, 1.0, 2.25]
    let cv = col_vars_f64(&csr, 0.0);
    approx_eq(cv[0], 0.25, 1e-12);
    approx_eq(cv[1], 1.0, 1e-12);
    approx_eq(cv[2], 2.25, 1e-12);

    let cs = col_stds_f64(&csr, 0.0);
    approx_eq(cs[0], 0.5, 1e-12);
    approx_eq(cs[1], 1.0, 1e-12);
    approx_eq(cs[2], 1.5, 1e-12);
}

#[test]
fn test_varstd_csc() {
    let csc = make_sample_csc();
    approx_eq(var_csc_f64(&csc, 0.0), 20.0 / 9.0, 1e-12);
    approx_eq(std_csc_f64(&csc, 0.0), (20.0_f64 / 9.0_f64).sqrt(), 1e-12);

    let rv = row_vars_csc_f64(&csc, 0.0);
    approx_eq(rv[0], 14.0 / 9.0, 1e-12);
    approx_eq(rv[1], 8.0 / 9.0, 1e-12);

    let rs = row_stds_csc_f64(&csc, 0.0);
    approx_eq(rs[0], (14.0_f64 / 9.0_f64).sqrt(), 1e-12);
    approx_eq(rs[1], (8.0_f64 / 9.0_f64).sqrt(), 1e-12);

    let cv = col_vars_csc_f64(&csc, 0.0);
    approx_eq(cv[0], 0.25, 1e-12);
    approx_eq(cv[1], 1.0, 1e-12);
    approx_eq(cv[2], 2.25, 1e-12);

    let cs = col_stds_csc_f64(&csc, 0.0);
    approx_eq(cs[0], 0.5, 1e-12);
    approx_eq(cs[1], 1.0, 1e-12);
    approx_eq(cs[2], 1.5, 1e-12);
}

#[test]
fn test_varstd_coo() {
    let coo = make_sample_coo();
    approx_eq(var_coo_f64(&coo, 0.0), 20.0 / 9.0, 1e-12);
    approx_eq(std_coo_f64(&coo, 0.0), (20.0_f64 / 9.0_f64).sqrt(), 1e-12);

    let rv = row_vars_coo_f64(&coo, 0.0);
    approx_eq(rv[0], 14.0 / 9.0, 1e-12);
    approx_eq(rv[1], 8.0 / 9.0, 1e-12);

    let rs = row_stds_coo_f64(&coo, 0.0);
    approx_eq(rs[0], (14.0_f64 / 9.0_f64).sqrt(), 1e-12);
    approx_eq(rs[1], (8.0_f64 / 9.0_f64).sqrt(), 1e-12);

    let cv = col_vars_coo_f64(&coo, 0.0);
    approx_eq(cv[0], 0.25, 1e-12);
    approx_eq(cv[1], 1.0, 1e-12);
    approx_eq(cv[2], 2.25, 1e-12);

    let cs = col_stds_coo_f64(&coo, 0.0);
    approx_eq(cs[0], 0.5, 1e-12);
    approx_eq(cs[1], 1.0, 1e-12);
    approx_eq(cs[2], 1.5, 1e-12);
}
