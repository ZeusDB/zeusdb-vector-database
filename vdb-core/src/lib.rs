// lib.rs
mod hnsw_index;
mod pq;
mod persistence;
mod logging;

use pyo3::prelude::*;

/// ZeusDB Vector Database Python Module
/// 
/// Automatically initializes structured logging on import.
/// Logs are controlled by environment variables or optional Python functions.
#[pymodule]
fn zeusdb_vector_database(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    // Auto-initialize logging on module import
    // Respects ZEUSDB_DISABLE_AUTOLOG for power users
    logging::init_logging();

    // Core classes
    m.add_class::<hnsw_index::HNSWIndex>()?;
    m.add_class::<hnsw_index::AddResult>()?;

    // Persistence functions
    m.add_function(wrap_pyfunction!(persistence::load_index, m)?)?;

    // Optional logging control for power users
    m.add_function(wrap_pyfunction!(logging::py_init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(logging::py_init_file_logging, m)?)?;
    m.add_function(wrap_pyfunction!(logging::is_logging_initialized, m)?)?;

    Ok(())
}