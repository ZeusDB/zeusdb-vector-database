// lib.rs
//
// The binding, which is what touches Python: the classes, the argument
// parsing, the result construction, the logging control, and the conversion
// below of an engine failure into an exception. The index itself is
// zeusdb-vector-hnsw, over the engine's floor in zeusdb-vector-core, and
// every method here parses, calls one operation on the collection through the
// rule a panic keeps, and converts what comes back, releasing the interpreter
// lock around any call that may wait for an engine guard.

/// The part of a call that runs with the interpreter released: `$body`
/// inside `Collection::run` on `$inner` as `$operation`, under `$py.detach`,
/// with the result converted for the method that returns it.
///
/// Under the `test-support` feature the released part starts at the pause a
/// test of the interpreter rule arms for its next call; see `test_support`.
/// Without the feature the expansion is the release alone.
///
/// It is a macro defined here, ahead of `hnsw_index`, so that each of its
/// two uses there is one line and adds no line to that module. Every log
/// record carries its file and line, so a line added above a tracing call
/// site changes what the log prints and what the wheel holds.
macro_rules! run_released {
    ($py:expr, $inner:expr, $operation:expr, $body:expr) => {
        Ok($py.detach(|| {
            #[cfg(feature = "test-support")]
            crate::test_support::pause_here($operation);
            $inner.run($operation, || $body)
        })?)
    };
}

mod conversion;
mod durability;
mod hnsw_index;
mod logging;
#[cfg(feature = "test-support")]
mod test_support;
mod tokenizer;

use pyo3::prelude::*;
use zeusdb_vector_core::{Error, Exception};
use zeusdb_vector_hnsw::{run_without_index, Collection, JournalPolicy};

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
/// Registered as `_load_index`. `VectorDatabase.load()` is the documented
/// route and is a one line pass through to this.
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
/// A directory whose manifest names a journal is recovered: every committed
/// record the journal holds above the checkpoint is replayed, the journal is
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
/// A journaled directory saved by a release before 0.14.0 is upgraded by its
/// first load: its journal is replayed and the directory is written again at
/// format 5.0.0, after which those releases do not open it. A load holds the
/// directory's lock, `<path>.zdblock`, while it reads, and a journaled index
/// keeps it for its life, so a second holder is refused with
/// `CollectionLocked`. A journaled directory this process cannot write opens
/// read only, with a `RuntimeWarning` naming the reason: every record is
/// replayed in memory, nothing on the medium changes, every change raises
/// `CollectionReadOnly`, and `save` into another directory copies it. A
/// directory holding a feature this build does not know, which its manifest
/// marks `read_only`, opens read only the same way whatever the medium, and
/// every change, every save, `checkpoint`, `journal_to` and `backup` raises
/// `FeatureReadOnly`.
///
/// `repair=True` loads a journal whose own history is broken: a corrupt
/// record with records after it, damage in a segment that is not the last,
/// a header that will not read, a record that will not decode or apply,
/// and segments that do not form one run of calls. Every whole call before
/// the first call the damage touches is replayed, that call and every byte
/// after it are copied into `<path>.zdbjournal/quarantine/<UTC stamp>-<first
/// sequence set aside>/` with a `repair.json` describing them, the journal
/// is cut there, and a checkpoint is written. A `RuntimeWarning` names the
/// damage, the records set aside and the folder, and `index.recovery`
/// reports the same. A load that finds nothing damaged and no repair
/// unfinished is a plain load. A journal that belongs to another collection
/// or another checkpoint is refused as without it. On a medium this process
/// cannot write, the kept calls are replayed in memory, nothing is set
/// aside or changed, and the index opens read only.
///
/// The whole load runs with the interpreter lock released, the graph rebuild
/// it may fall back to included. Nothing in the load path touches Python:
/// the directory is read, the collection is built and restored, and the
/// result is wrapped here with the lock back. A tokenizer handed in is
/// stored and never run by the load. `interval_ms` is a Python integer, and a
/// negative one raises `ValueError` naming it. A panic inside the engine
/// raises `RuntimeError`; see `run_without_index`.
#[pyfunction]
#[pyo3(name = "_load_index", signature = (path, tokenizer = None, durability = None, interval_ms = None, checkpoint_only = false, repair = false))]
fn load_index(
    py: Python<'_>,
    path: &str,
    tokenizer: Option<&Bound<PyAny>>,
    durability: Option<&str>,
    interval_ms: Option<i64>,
    checkpoint_only: bool,
    repair: bool,
) -> Result<hnsw_index::HNSWIndex, PyEngineError> {
    run_without_index("load()", || {
        load_index_body(
            py,
            path,
            tokenizer,
            durability,
            interval_ms,
            checkpoint_only,
            repair,
        )
    })
}

/// The body of `load_index`, run through the rule a panic keeps by it.
fn load_index_body(
    py: Python<'_>,
    path: &str,
    tokenizer: Option<&Bound<PyAny>>,
    durability: Option<&str>,
    interval_ms: Option<i64>,
    checkpoint_only: bool,
    repair: bool,
) -> Result<hnsw_index::HNSWIndex, PyEngineError> {
    let interval_ms = conversion::count_if_given(interval_ms, "interval_ms")?.map(|ms| ms as u64);
    let tokenizer = tokenizer
        .map(|argument| tokenizer::tokenizer_from_python(argument, "tokenizer"))
        .transpose()?;
    let named = durability.is_some() || interval_ms.is_some();
    let policy = if checkpoint_only {
        if repair {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "repair=True reads the journal to repair it, and checkpoint_only=True reads no \
                 journal, so the two do not go together",
            )
            .into());
        }
        if named {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "checkpoint_only=True opens the checkpoint alone and attaches no journal, so \
                 durability and interval_ms do not apply to it",
            )
            .into());
        }
        JournalPolicy::CheckpointOnly
    } else {
        let durability = durability::parse_durability(durability.unwrap_or("call"), interval_ms)?;
        if repair {
            JournalPolicy::Repair(durability)
        } else {
            JournalPolicy::Replay(durability)
        }
    };
    let (inner, recovery) = py.detach(|| Collection::recover_with(path, tokenizer, policy))?;
    if named && !recovery.journaled {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "load names a durability and the directory at '{}' has no journal. A durability \
             applies to a journaled directory; index.journal_to(path) opens a journal beside \
             one that has none.",
            path
        ))
        .into());
    }
    let warning = recovery
        .repair_warning(path)
        .or_else(|| recovery.read_only_warning(path));
    if let Some(message) = warning {
        let category = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
        let message = std::ffi::CString::new(message)
            .unwrap_or_else(|_| c"The directory was loaded with a warning.".to_owned());
        PyErr::warn(py, category.as_any(), &message, 1)?;
    }
    Ok(hnsw_index::HNSWIndex::wrap_loaded(inner, recovery))
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

    // Close this process's lock files in a child forked from it.
    //
    // A lock on an open file is shared by every copy of the file, and a child
    // forked from this process holds a copy of each lock file the process has
    // open, so a child that lives on would keep the directories locked after
    // this process lets them go or ends. The child closes its copies just
    // after the fork, which leaves every lock with this process. The lock
    // files are held across the fork, so no other thread opens or closes one
    // while the process is copied. `os.register_at_fork` exists on POSIX
    // alone, and the hooks are not attributes of the module.
    let os = py.import("os")?;
    if os.hasattr("register_at_fork")? {
        let hooks = pyo3::types::PyDict::new(py);
        hooks.set_item(
            "before",
            pyo3::types::PyCFunction::new_closure(py, None, None, |_, _| -> PyResult<()> {
                zeusdb_vector_core::hold_lock_files_for_fork();
                Ok(())
            })?,
        )?;
        hooks.set_item(
            "after_in_parent",
            pyo3::types::PyCFunction::new_closure(py, None, None, |_, _| -> PyResult<()> {
                zeusdb_vector_core::release_lock_files_after_fork();
                Ok(())
            })?,
        )?;
        hooks.set_item(
            "after_in_child",
            pyo3::types::PyCFunction::new_closure(py, None, None, |_, _| -> PyResult<()> {
                zeusdb_vector_core::close_lock_files_in_forked_child();
                Ok(())
            })?,
        )?;
        os.call_method("register_at_fork", (), Some(&hooks))?;
    }

    // The pause a test of the interpreter rule arms, registered after every
    // other name and under the `test-support` feature alone; see
    // `test_support`.
    #[cfg(feature = "test-support")]
    m.add_class::<test_support::PyPause>()?;

    Ok(())
}
