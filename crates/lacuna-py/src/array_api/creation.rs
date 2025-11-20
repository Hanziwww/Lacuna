//! Array creation functions (Array API aligned)
//!
//! This module provides functions for creating sparse arrays,
//! following the Array API standard naming conventions.

use numpy::PyArray1;
use pyo3::prelude::*;

/// Create a CSR matrix filled with zeros
///
/// # Arguments
/// * `nrows` - Number of rows
/// * `ncols` - Number of columns
///
/// # Returns
/// Empty CSR matrix (no non-zero elements)
#[pyfunction]
pub(crate) fn zeros_csr<'py>(
    py: Python<'py>,
    nrows: usize,
    _ncols: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let indptr = vec![0_i64; nrows + 1];
    let indices: Vec<i64> = vec![];
    let data: Vec<f64> = vec![];

    Ok((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, data),
    ))
}

/// Create a CSC matrix filled with zeros
#[pyfunction]
pub(crate) fn zeros_csc<'py>(
    py: Python<'py>,
    _nrows: usize,
    ncols: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let indptr = vec![0_i64; ncols + 1];
    let indices: Vec<i64> = vec![];
    let data: Vec<f64> = vec![];

    Ok((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, data),
    ))
}

/// Create a COO matrix filled with zeros
#[pyfunction]
pub(crate) fn zeros_coo<'py>(
    py: Python<'py>,
    _nrows: usize,
    _ncols: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let row: Vec<i64> = vec![];
    let col: Vec<i64> = vec![];
    let data: Vec<f64> = vec![];

    Ok((
        PyArray1::from_vec(py, row),
        PyArray1::from_vec(py, col),
        PyArray1::from_vec(py, data),
    ))
}

/// Create an identity matrix in CSR format
///
/// # Arguments
/// * `n` - Size of the square identity matrix
///
/// # Returns
/// CSR representation of n×n identity matrix
#[pyfunction]
pub(crate) fn eye_csr<'py>(
    py: Python<'py>,
    n: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let indptr: Vec<i64> = (0..=n)
        .map(|i| i64::try_from(i).expect("matrix dimension exceeds i64"))
        .collect();
    let indices: Vec<i64> = (0..n)
        .map(|i| i64::try_from(i).expect("matrix dimension exceeds i64"))
        .collect();
    let data: Vec<f64> = vec![1.0; n];

    Ok((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, data),
    ))
}

/// Create a diagonal matrix in CSR format
///
/// # Arguments
/// * `diag` - Diagonal values
///
/// # Returns
/// CSR representation of diagonal matrix
#[pyfunction]
pub(crate) fn diag_csr<'py>(
    py: Python<'py>,
    diag: Vec<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let n = diag.len();
    let indptr: Vec<i64> = (0..=n)
        .map(|i| i64::try_from(i).expect("matrix dimension exceeds i64"))
        .collect();
    let indices: Vec<i64> = (0..n)
        .map(|i| i64::try_from(i).expect("matrix dimension exceeds i64"))
        .collect();

    Ok((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, diag),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;

    #[test]
    fn test_eye_structure() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let (indptr, indices, data) = eye_csr(py, 5).unwrap();
            let indptr_vec = indptr.to_vec().unwrap();
            let indices_vec = indices.to_vec().unwrap();
            let data_vec = data.to_vec().unwrap();

            assert_eq!(indptr_vec, vec![0, 1, 2, 3, 4, 5]);
            assert_eq!(indices_vec, vec![0, 1, 2, 3, 4]);
            assert_eq!(data_vec, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        });
    }
}
