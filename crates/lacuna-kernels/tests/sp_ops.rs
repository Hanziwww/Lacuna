use lacuna_core::Csr;
use lacuna_kernels::{spmv_f64_i64, spmm_f64_i64, sum_f64, row_sums_f64, transpose_f64_i64, add_csr_f64_i64};

#[test]
fn test_spmv_spmm_basic() {
    // A = [[1, 0, 2],
    //      [0, 3, 0]]
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2, 3];
    let indices = vec![0i64, 2, 1];
    let data = vec![1.0, 2.0, 3.0];
    let a = Csr { nrows, ncols, indptr, indices, data };

    let x = vec![10.0, 20.0, 30.0];
    let y = spmv_f64_i64(&a, &x);
    assert_eq!(y, vec![1.0*10.0 + 2.0*30.0, 3.0*20.0]);

    let b = vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]; // shape (3,2)
    let y2 = spmm_f64_i64(&a, &b, 2);
    // Row 0: [1,0,2] @ B -> [1*1 + 2*5, 1*2 + 2*6] = [11, 14]
    // Row 1: [0,3,0] @ B -> [3*3, 3*4] = [9, 12]
    assert_eq!(y2, vec![11.0, 14.0, 9.0, 12.0]);
}

#[test]
fn test_sum_rows_transpose_add() {
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2, 3];
    let indices = vec![0i64, 2, 1];
    let data = vec![1.0, 2.0, 3.0];
    let a = Csr { nrows, ncols, indptr, indices, data };

    assert_eq!(sum_f64(&a), 6.0);
    assert_eq!(row_sums_f64(&a), vec![3.0, 3.0]);

    let at = transpose_f64_i64(&a);
    assert_eq!(at.nrows, 3);
    assert_eq!(at.ncols, 2);

    let c = add_csr_f64_i64(&a, &a);
    assert_eq!(sum_f64(&c), 12.0);
}
