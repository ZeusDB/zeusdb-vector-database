// lib.rs
//
// The binding, which is what touches Python: the classes, the argument
// parsing, the result construction, the logging control, and the conversion
// below of an engine failure into an exception. The index itself is
// zeusdb-vector-hnsw, over the engine's floor in zeusdb-vector-core, and
// every method here parses, releases the interpreter lock, calls one
// operation on the collection and converts what comes back.
mod conversion;
mod durability;
mod hnsw_index;
mod logging;
mod tokenizer;

use pyo3::prelude::*;
use zeusdb_vector_core::{Error, Exception};
use zeusdb_vector_hnsw::Collection;

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
        // A failure inside a tokenizer the caller wrote is the caller's own
        // exception, carried out through the engine, and it is raised as
        // itself so the caller reads their own class and traceback.
        if let Error::TokenizerFailed(inner) = error {
            return match inner.downcast::<PyErr>() {
                Ok(raised) => PyEngineError(*raised),
                Err(other) => PyEngineError(pyo3::exceptions::PyRuntimeError::new_err(
                    Error::TokenizerFailed(other).to_string(),
                )),
            };
        }
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

/// Load an index from a directory. See `Collection::load_with`.
///
/// Registered as `_load_index`. `VectorDatabase.load(path, tokenizer=None)`
/// is the documented route and is a one line pass through to this.
///
/// `tokenizer` is the tokenizer the directory's text layer was declared
/// with, where it has one: `"simple"` for the built-in tokenizer, which a
/// directory recording `simple` rebuilds on its own and needs none handed,
/// or the callable the space was created with, which a directory records as
/// `external` and cannot reproduce. A directory recording `external` refuses
/// to open without one, one handed whose declaration is not the recorded
/// one is refused, and one handed to a directory that takes no text is
/// refused, since ignoring it would open the index under a tokenizer the
/// caller did not ask for.
///
/// A directory whose manifest names a journal is recovered: every record
/// the journal holds above the checkpoint is replayed, the journal is
/// reopened for append, and the index hands its mutations to it from then
/// on under `durability`, which takes the three names `journal_to` takes
/// and defaults to `"call"`. `interval_ms` belongs to `"interval"`. Naming
/// either for a directory that has no journal is refused rather than
/// ignored, since a caller who named a policy expects one to be in force.
/// `checkpoint_only=True` opens the checkpoint alone, reads no journal,
/// attaches none and takes no policy; it is the way in for a directory
/// copied without its sibling, and for a caller who wants the state as of
/// the last checkpoint.
///
/// The whole load runs with the interpreter lock released, the graph rebuild
/// it may fall back to included. Nothing in the load path touches Python:
/// the directory is read, the collection is built and restored, and the
/// result is wrapped here with the lock back. A tokenizer handed in is
/// stored and never run by the load.
#[pyfunction]
#[pyo3(name = "_load_index", signature = (path, tokenizer = None, durability = None, interval_ms = None, checkpoint_only = false))]
fn load_index(
    py: Python<'_>,
    path: &str,
    tokenizer: Option<&Bound<PyAny>>,
    durability: Option<&str>,
    interval_ms: Option<u64>,
    checkpoint_only: bool,
) -> Result<hnsw_index::HNSWIndex, PyEngineError> {
    let tokenizer = tokenizer
        .map(|argument| tokenizer::tokenizer_from_python(argument, "tokenizer"))
        .transpose()?;
    let named = durability.is_some() || interval_ms.is_some();
    if checkpoint_only {
        if named {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "checkpoint_only=True opens the checkpoint alone and attaches no journal, so \
                 durability and interval_ms do not apply to it",
            )
            .into());
        }
        let inner = py.detach(|| Collection::load_checkpoint_only(path, tokenizer))?;
        return Ok(hnsw_index::HNSWIndex::wrap(inner));
    }
    let durability = durability::parse_durability(durability.unwrap_or("call"), interval_ms)?;
    let (inner, recovery) = py.detach(|| Collection::recover(path, tokenizer, durability))?;
    if named && !recovery.journaled {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "load names a durability and the directory at '{}' has no journal. A durability \
             applies to a journaled directory; index.journal_to(path) opens a journal beside \
             one that has none.",
            path
        ))
        .into());
    }
    Ok(hnsw_index::HNSWIndex::wrap(inner))
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
