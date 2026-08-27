// lib.rs
// A column per declared filterable field, which is what a filtered search
// reads instead of walking every record's metadata.
mod checksum;
mod columns;
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
///
/// The module is `_engine`, and is imported as `zeusdb_vector_database._engine`
/// by the Python package. The name is given here and by module-name in
/// pyproject.toml, and is independent of the crate name in Cargo.toml, which
/// stays `zeusdb_vector_database` because it is the root of every log
/// record's target.
#[pymodule(name = "_engine")]
fn engine(py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(logging::py_shutdown_logging, m)?)?;

    // Drain the file appender before the process exits.
    //
    // The `file` target writes from a worker thread fed by a channel, and a
    // worker thread is killed at process exit wherever it happens to be, so
    // without this whatever is still queued is never written. The hook is
    // registered here rather than in the Python package because this is the
    // import that starts the worker, so no other entry point can bypass it.
    //
    // `atexit` runs its callbacks last-in-first-out during interpreter
    // finalization, before any module is torn down. Registering at import time
    // therefore puts this behind every hook the application registers later,
    // which is what a drain wants. It does not run on `os._exit` or on an
    // abort, and nothing can run there.
    let shutdown = m.getattr("shutdown_logging")?;
    py.import("atexit")?.call_method1("register", (shutdown,))?;

    Ok(())
}
