use lacuna_core::{Coo, Csc, Csr};
use lacuna_kernels::{
    col_maxs_coo_f64, col_maxs_csc_f64, col_maxs_f64, col_mins_coo_f64, col_mins_csc_f64,
    col_mins_f64, max_coo_f64, max_csc_f64, max_f64, min_coo_f64, min_csc_f64, min_f64,
    row_maxs_coo_f64, row_maxs_csc_f64, row_maxs_f64, row_mins_coo_f64, row_mins_csc_f64,
    row_mins_f64,
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

fn make_sample_coo() -> Coo<f64, i64> {
    // Same matrix as COO
    let nrows = 2usize;
    let ncols = 3usize;
    let row = vec![0_i64, 0, 1];
    let col = vec![0_i64, 2, 1];
    let data = vec![1.0_f64, 3.0, -2.0];
    Coo::from_parts(nrows, ncols, row, col, data, true).unwrap()
}

fn make_sample_csc() -> Csc<f64, i64> {
    // Same matrix as CSC
    // col0: (0, 1.0), col1: (1, -2.0), col2: (0, 3.0)
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0_i64, 1, 2, 3];
    let indices = vec![0_i64, 1, 0];
    let data = vec![1.0_f64, -2.0, 3.0];
    Csc::from_parts(nrows, ncols, indptr, indices, data, true).unwrap()
}

#[test]
fn test_minmax_global_csr() {
    let a = make_sample_csr();
    let mn = min_f64(&a);
    let mx = max_f64(&a);
    assert_eq!(mn, -2.0);
    assert_eq!(mx, 3.0);
}

#[test]
fn test_minmax_rows_cols_csr() {
    let a = make_sample_csr();
    let rmins = row_mins_f64(&a);
    let rmaxs = row_maxs_f64(&a);
    assert_eq!(rmins, vec![0.0, -2.0]);
    assert_eq!(rmaxs, vec![3.0, 0.0]);

    let cmins = col_mins_f64(&a);
    let cmaxs = col_maxs_f64(&a);
    assert_eq!(cmins, vec![0.0, -2.0, 0.0]);
    assert_eq!(cmaxs, vec![1.0, 0.0, 3.0]);
}

#[test]
fn test_minmax_global_csc() {
    let a = make_sample_csc();
    let mn = min_csc_f64(&a);
    let mx = max_csc_f64(&a);
    assert_eq!(mn, -2.0);
    assert_eq!(mx, 3.0);
}

#[test]
fn test_minmax_rows_cols_csc() {
    let a = make_sample_csc();
    let rmins = row_mins_csc_f64(&a);
    let rmaxs = row_maxs_csc_f64(&a);
    assert_eq!(rmins, vec![0.0, -2.0]);
    assert_eq!(rmaxs, vec![3.0, 0.0]);

    let cmins = col_mins_csc_f64(&a);
    let cmaxs = col_maxs_csc_f64(&a);
    assert_eq!(cmins, vec![0.0, -2.0, 0.0]);
    assert_eq!(cmaxs, vec![1.0, 0.0, 3.0]);
}

#[test]
fn test_minmax_global_coo() {
    let a = make_sample_coo();
    let mn = min_coo_f64(&a);
    let mx = max_coo_f64(&a);
    assert_eq!(mn, -2.0);
    assert_eq!(mx, 3.0);
}

#[test]
fn test_minmax_rows_cols_coo() {
    let a = make_sample_coo();
    let rmins = row_mins_coo_f64(&a);
    let rmaxs = row_maxs_coo_f64(&a);
    assert_eq!(rmins, vec![0.0, -2.0]);
    assert_eq!(rmaxs, vec![3.0, 0.0]);

    let cmins = col_mins_coo_f64(&a);
    let cmaxs = col_maxs_coo_f64(&a);
    assert_eq!(cmins, vec![0.0, -2.0, 0.0]);
    assert_eq!(cmaxs, vec![1.0, 0.0, 3.0]);
}
