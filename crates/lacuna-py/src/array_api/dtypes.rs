//! Data type operations and constants (Array API aligned)
//!
//! This module provides dtype-related functions and constants
//! following the Array API standard.

use pyo3::prelude::*;

/// Get the default floating-point data type
///
/// Currently returns "float64" (IEEE 754 double precision)
#[pyfunction]
pub(crate) fn get_default_float_dtype() -> &'static str {
    "float64"
}

/// Get the default integer data type
///
/// Currently returns "int64" (signed 64-bit integer)
#[pyfunction]
pub(crate) fn get_default_int_dtype() -> &'static str {
    "int64"
}

/// Check if a dtype string is a valid floating-point type
#[pyfunction]
pub(crate) fn is_float_dtype(dtype: &str) -> bool {
    matches!(dtype, "float32" | "float64")
}

/// Check if a dtype string is a valid integer type
#[pyfunction]
pub(crate) fn is_int_dtype(dtype: &str) -> bool {
    matches!(dtype, "int32" | "int64" | "uint32" | "uint64")
}

/// Get the size in bytes of a dtype
#[pyfunction]
pub(crate) fn dtype_size(dtype: &str) -> PyResult<usize> {
    match dtype {
        "float32" | "int32" | "uint32" => Ok(4),
        "float64" | "int64" | "uint64" => Ok(8),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown dtype: {dtype}"
        ))),
    }
}

/// Dtype promotion table for binary operations
///
/// Returns the result dtype when combining two dtypes
#[pyfunction]
pub(crate) fn promote_dtypes(dtype1: &str, dtype2: &str) -> PyResult<&'static str> {
    let Some(dtype1_canon) = canonical_dtype(dtype1) else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Cannot promote dtypes: {dtype1} and {dtype2}"
        )));
    };
    let Some(dtype2_canon) = canonical_dtype(dtype2) else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Cannot promote dtypes: {dtype1} and {dtype2}"
        )));
    };

    if dtype1_canon == dtype2_canon {
        return Ok(dtype1_canon);
    }

    if dtype1_canon == "float64"
        || dtype2_canon == "float64"
        || (dtype1_canon == "float32" && dtype2_canon == "int64")
        || (dtype1_canon == "int64" && dtype2_canon == "float32")
    {
        return Ok("float64");
    }

    if (dtype1_canon == "float32" && dtype2_canon == "int32")
        || (dtype1_canon == "int32" && dtype2_canon == "float32")
    {
        return Ok("float32");
    }

    if (dtype1_canon == "int64" && dtype2_canon == "int32")
        || (dtype1_canon == "int32" && dtype2_canon == "int64")
    {
        return Ok("int64");
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Cannot promote dtypes: {dtype1} and {dtype2}"
    )))
}

fn canonical_dtype(dtype: &str) -> Option<&'static str> {
    match dtype {
        "float64" => Some("float64"),
        "float32" => Some("float32"),
        "int64" => Some("int64"),
        "int32" => Some("int32"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size("float64").unwrap(), 8);
        assert_eq!(dtype_size("float32").unwrap(), 4);
        assert_eq!(dtype_size("int64").unwrap(), 8);
        assert!(dtype_size("invalid").is_err());
    }

    #[test]
    fn test_promote_dtypes() {
        assert_eq!(promote_dtypes("float64", "float32").unwrap(), "float64");
        assert_eq!(promote_dtypes("float64", "int64").unwrap(), "float64");
        assert_eq!(promote_dtypes("int64", "int32").unwrap(), "int64");
    }

    #[test]
    fn test_is_float_dtype() {
        assert!(is_float_dtype("float64"));
        assert!(is_float_dtype("float32"));
        assert!(!is_float_dtype("int64"));
    }
}
