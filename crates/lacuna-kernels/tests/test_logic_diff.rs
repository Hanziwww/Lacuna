use lacuna_core::{Csc, Csr};
use lacuna_kernels::{
    all_f64, any_f64, col_alls_f64, col_anys_f64, diff_csc_axis0_f64_i64, diff_csr_axis0_f64_i64,
    diff_csr_axis1_f64_i64, row_alls_f64, row_anys_f64,
};

#[test]
fn test_all_any_csr_basic() {
    // [[1, 0], [0, 2]]
    let a = Csr::from_parts_unchecked(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
    assert!(!all_f64(&a));
    assert!(any_f64(&a));

    let ra = row_anys_f64(&a);
    assert_eq!(ra, vec![true, true]);
    let ca = col_anys_f64(&a);
    assert_eq!(ca, vec![true, true]);

    let rall = row_alls_f64(&a);
    assert_eq!(rall, vec![false, false]);
    let call = col_alls_f64(&a);
    assert_eq!(call, vec![false, false]);
}

#[test]
fn test_diff_axis1_full_row() {
    // 1x4 row: [1, 2, 4, 7]
    let a = Csr::from_parts_unchecked(1, 4, vec![0, 4], vec![0, 1, 2, 3], vec![1.0, 2.0, 4.0, 7.0]);
    let d1 = diff_csr_axis1_f64_i64(&a, 1);
    assert_eq!(d1.ncols, 3);
    assert_eq!(d1.indptr, vec![0, 3]);
    assert_eq!(d1.indices, vec![0, 1, 2]);
    assert_eq!(d1.data, vec![1.0, 2.0, 3.0]);

    let d2 = diff_csr_axis1_f64_i64(&a, 2);
    assert_eq!(d2.ncols, 2);
    assert_eq!(d2.data, vec![1.0, 1.0]);
}

#[test]
fn test_diff_axis0_full_col() {
    // 3x1 col: [1, 2, 4]^T
    let a = Csr::from_parts_unchecked(3, 1, vec![0, 1, 2, 3], vec![0, 0, 0], vec![1.0, 2.0, 4.0]);
    let d1 = diff_csr_axis0_f64_i64(&a, 1);
    assert_eq!(d1.nrows, 2);
    assert_eq!(d1.ncols, 1);
    assert_eq!(d1.indptr, vec![0, 1, 2]);
    assert_eq!(d1.indices, vec![0, 0]);
    assert_eq!(d1.data, vec![1.0, 2.0]);
}

#[test]
fn test_diff_csc_axis0() {
    // 3x1 column in CSC
    let a = Csc::from_parts_unchecked(3, 1, vec![0, 3], vec![0, 1, 2], vec![1.0, 2.0, 4.0]);
    let d1 = diff_csc_axis0_f64_i64(&a, 1);
    assert_eq!(d1.nrows, 2);
    assert_eq!(d1.ncols, 1);
    assert_eq!(d1.indptr, vec![0, 2]);
    assert_eq!(d1.indices, vec![0, 1]);
    assert_eq!(d1.data, vec![1.0, 2.0]);
}
