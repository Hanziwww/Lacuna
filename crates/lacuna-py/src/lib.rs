use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
