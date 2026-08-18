// lib.rs
mod conversion;
mod distance;
mod filter;
mod graph;
mod hnsw_index;
mod logging;
mod persistence;
mod pq;
mod rerank;
// The generator every seeded draw runs on, named in one place so the pin is
// auditable in one place.
mod rng;
// Test data two test modules measure against, so it is defined once.
#[cfg(test)]
mod test_vectors;

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

    // Core classes. Neither carries a `#[new]`, so both are importable for
    // isinstance checks and annotations while direct construction raises
    // TypeError. Indexes come from `_create_hnsw_index` or `_load_index`.
    m.add_class::<hnsw_index::HNSWIndex>()?;
    m.add_class::<hnsw_index::AddResult>()?;

    // Index construction, private because VectorDatabase.create is the
    // documented route and applies the defaults this function does not.
    m.add_function(wrap_pyfunction!(hnsw_index::create_hnsw_index, m)?)?;

    // Persistence functions, private because VectorDatabase.load is the
    // documented route.
    m.add_function(wrap_pyfunction!(persistence::load_index, m)?)?;

    // Optional logging control for power users
    m.add_function(wrap_pyfunction!(logging::py_init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(logging::py_init_file_logging, m)?)?;
    m.add_function(wrap_pyfunction!(logging::is_logging_initialized, m)?)?;

    Ok(())
}
