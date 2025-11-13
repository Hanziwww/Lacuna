use lacuna_core::Csr;

#[test]
fn from_parts_ok() {
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2, 3];
    let indices = vec![0i64, 2, 1];
    let data = vec![1.0f64, 2.0, 3.0];
    let csr = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap();
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csr.shape(), (2, 3));
}

#[test]
fn indptr_first_must_be_zero() {
    let nrows = 1usize;
    let ncols = 3usize;
    let indptr = vec![1i64, 1]; // first element not zero, but still length 2 and last == 1 == nnz
    let indices = vec![0i64];
    let data = vec![1.0f64];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    println!("Error: {:?}", err);
    assert!(err.contains("must be 0"));
}

#[test]
fn nnz_and_lengths_must_match() {
    let nrows = 1usize;
    let ncols = 3usize;
    // indices/data length mismatch
    let indptr = vec![0i64, 2];
    let indices = vec![0i64, 1];
    let data = vec![1.0f64];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    assert!(err.contains("indices and data"));
}

#[test]
fn last_element_must_equal_nnz() {
    let nrows = 1usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 1];
    let indices = vec![0i64, 1];
    let data = vec![1.0f64, 2.0];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    assert!(err.contains("last element"));
}

#[test]
fn indptr_non_decreasing_per_row() {
    let nrows = 2usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2, 1]; // decreasing at the last step; length 3, last element 1 == nnz
    let indices = vec![0i64];
    let data = vec![1.0f64];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    println!("Error: {:?}", err);
    assert!(err.contains("must be non-decreasing"));
}

#[test]
fn strict_increasing_columns_enforced() {
    let nrows = 1usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 2];
    let indices = vec![1i64, 1]; // duplicate within row
    let data = vec![1.0f64, 2.0];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    assert!(err.contains("strictly increasing"));
}

#[test]
fn column_index_out_of_bounds() {
    let nrows = 1usize;
    let ncols = 3usize;
    let indptr = vec![0i64, 1];
    let indices = vec![3i64]; // out of bounds (valid: 0..=2)
    let data = vec![1.0f64];
    let err = Csr::from_parts(nrows, ncols, indptr, indices, data, true).unwrap_err();
    assert!(err.contains("out of bounds"));
}
