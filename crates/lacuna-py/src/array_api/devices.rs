//! Device management (Array API aligned)
//!
//! This module provides device-related functions following the Array API standard.
//! Currently only CPU is supported, with GPU support planned for future releases.

use pyo3::prelude::*;

/// Get the default device
///
/// Currently always returns "cpu" as GPU support is not yet implemented.
#[pyfunction]
pub(crate) fn get_default_device() -> &'static str {
    "cpu"
}

/// Check if a device string is valid
///
/// # Arguments
/// * `device` - Device identifier string
///
/// # Returns
/// `true` if the device is recognized, `false` otherwise
#[pyfunction]
pub(crate) fn is_valid_device(device: &str) -> bool {
    matches!(device, "cpu")
    // Future: "cuda" | "cuda:0" | "cuda:1" etc.
}

/// Get the list of available devices
///
/// # Returns
/// Vector of available device identifiers
#[pyfunction]
pub(crate) fn list_devices() -> Vec<&'static str> {
    vec!["cpu"]
    // Future: vec!["cpu", "cuda:0", "cuda:1"]
}

/// Get device capabilities
///
/// # Arguments
/// * `device` - Device identifier
///
/// # Returns
/// Dictionary of device capabilities
#[pyfunction]
pub(crate) fn device_info(py: Python<'_>, device: &str) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    if !is_valid_device(device) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid device: {device}"
        )));
    }

    let info = PyDict::new(py);
    info.set_item("name", device)?;
    info.set_item("type", "cpu")?;
    info.set_item("compute_capability", py.None())?;
    info.set_item("memory_total", py.None())?;

    Ok(info.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_device() {
        assert_eq!(get_default_device(), "cpu");
    }

    #[test]
    fn test_is_valid_device() {
        assert!(is_valid_device("cpu"));
        assert!(!is_valid_device("cuda"));
        assert!(!is_valid_device("invalid"));
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0], "cpu");
    }
}
