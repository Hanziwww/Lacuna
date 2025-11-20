//! Common helper functions to reduce code duplication across Array API bindings

use pyo3::prelude::*;

/// Convert i64 shape array to usize Vec, with validation
///
/// # Errors
/// Returns `PyValueError` if any dimension is negative or overflows usize
pub(crate) fn convert_shape_i64_to_usize(shape: &[i64]) -> PyResult<Vec<usize>> {
    let mut result = Vec::with_capacity(shape.len());
    for &s in shape {
        if s < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "shape dimensions must be non-negative",
            ));
        }
        result.push(usize::try_from(s).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "shape dimension {s} overflows usize"
            ))
        })?);
    }
    Ok(result)
}

/// Convert i64 axes array to usize Vec, with validation
///
/// # Arguments
/// * `axes` - Array of axis indices
/// * `param_name` - Name of parameter for error messages (e.g., "`row_axes`")
///
/// # Errors
/// Returns `PyValueError` if any axis is negative or overflows usize
pub(crate) fn convert_axes_i64_to_usize(axes: &[i64], param_name: &str) -> PyResult<Vec<usize>> {
    let mut result = Vec::with_capacity(axes.len());
    for &ax in axes {
        if ax < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{param_name} must be non-negative, got {ax}"
            )));
        }
        result.push(usize::try_from(ax).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{param_name} value {ax} overflows usize"
            ))
        })?);
    }
    Ok(result)
}

/// Convert usize shape to i64 Vec for Python return
///
/// # Panics
/// Panics if any dimension is too large for i64 (extremely unlikely in practice)
pub(crate) fn convert_shape_usize_to_i64(shape: &[usize]) -> Vec<i64> {
    shape
        .iter()
        .map(|&s| i64::try_from(s).expect("shape dimension too large for i64"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_shape_valid() {
        let shape = vec![10, 20, 30];
        let result = convert_shape_i64_to_usize(&shape).unwrap();
        assert_eq!(result, vec![10_usize, 20, 30]);
    }

    #[test]
    fn test_convert_shape_negative() {
        let shape = vec![10, -1, 30];
        assert!(convert_shape_i64_to_usize(&shape).is_err());
    }

    #[test]
    fn test_convert_axes_valid() {
        let axes = vec![0, 2];
        let result = convert_axes_i64_to_usize(&axes, "axes").unwrap();
        assert_eq!(result, vec![0_usize, 2]);
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![10_usize, 20, 30];
        let i64_shape = convert_shape_usize_to_i64(&original);
        let back = convert_shape_i64_to_usize(&i64_shape).unwrap();
        assert_eq!(original, back);
    }
}
