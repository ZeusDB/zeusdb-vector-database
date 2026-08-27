// lib.rs
//
// The binding, which is what touches Python: the classes, the argument
// parsing, the result construction, the logging control, and the conversion
// below of an engine failure into an exception. The engine itself is
// zeusdb-vector-core.
mod conversion;
mod hnsw_index;
mod logging;
mod persistence;

use pyo3::prelude::*;
use zeusdb_vector_core::{Error, Exception};

/// An engine failure on its way to Python.
///
/// Every engine module raises [`Error`], whose `exception` names the class as
/// a value. `Error` and `PyErr` are both foreign to this crate, so the orphan
/// rule refuses `impl From<Error> for PyErr` here, and this newtype is the
/// local type the conversion hangs on. It holds the `PyErr` already built,
/// so a `?` on a PyO3 result lands in it as well, through the second `From`
/// below, and a function this crate exposes to PyO3 returns
/// `Result<T, PyEngineError>` wherever its body raises an engine failure.
/// PyO3 accepts any error type that converts into a `PyErr` from a
/// `#[pymethods]` or `#[pyfunction]` body, which the `From` into `PyErr` is.
/// Nothing else in the crate builds a `PyErr` from an engine failure.
pub struct PyEngineError(PyErr);

impl From<Error> for PyEngineError {
    fn from(error: Error) -> Self {
        let message = error.to_string();
        PyEngineError(match error.exception() {
            Exception::Value => pyo3::exceptions::PyValueError::new_err(message),
            Exception::Runtime => pyo3::exceptions::PyRuntimeError::new_err(message),
            Exception::Key => pyo3::exceptions::PyKeyError::new_err(message),
            Exception::FileNotFound => pyo3::exceptions::PyFileNotFoundError::new_err(message),
        })
    }
}

impl From<PyErr> for PyEngineError {
    fn from(err: PyErr) -> Self {
        PyEngineError(err)
    }
}

/// The one numpy failure the parsers raise through `?`, being a slice taken
/// of an array that is not contiguous.
impl From<numpy::AsSliceError> for PyEngineError {
    fn from(err: numpy::AsSliceError) -> Self {
        PyEngineError(err.into())
    }
}

/// What `#[instrument(err)]` records when a method returns an error. It is
/// the `PyErr`'s own rendering, `<class>: <message>`, so the record is the
/// one the method wrote before this type existed.
impl std::fmt::Display for PyEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<PyEngineError> for PyErr {
    fn from(err: PyEngineError) -> PyErr {
        err.0
    }
}

/// Load an index from a directory. See `persistence::load_index`.
///
/// Registered as `_load_index`. `VectorDatabase.load(path)` is the documented
/// route and is a one line pass through to this.
#[pyfunction]
#[pyo3(name = "_load_index")]
fn load_index(path: &str) -> Result<hnsw_index::HNSWIndex, PyEngineError> {
    Ok(persistence::load_index(path)?)
}

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
    m.add_function(wrap_pyfunction!(load_index, m)?)?;

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
