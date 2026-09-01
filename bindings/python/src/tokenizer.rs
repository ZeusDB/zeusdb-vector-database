//! A tokenizer written in Python, behind the engine's trait.
//!
//! The engine's text layer splits a text into terms through one method,
//! `Tokenizer::tokenize`, and records a tokenizer it cannot write down as
//! `external`. From Python a tokenizer is a callable: it takes the text as a
//! `str` and returns an iterable of `str`, one per term, in order and with
//! every repeat, since the count is what a term frequency space stores. It
//! is declared at `create()` under `sparse["tokenizer"]` and handed back at
//! `load()` under `tokenizer`, because nothing in a saved directory can
//! reproduce it.
//!
//! # Where the interpreter lock is held
//!
//! `tokenize` attaches to the interpreter for the call. Where the calling
//! thread already holds the lock, which is every call the binding makes,
//! attaching costs nothing and the callable runs at once. Where the engine
//! is running with the lock released, attaching waits for it. That wait is
//! safe only because the engine never runs a tokenizer with one of its
//! guards held: a thread holding the lock and waiting for the dictionary's
//! guard, against a thread holding that guard and waiting for the lock,
//! would wait forever. `Collection::tokenize` is where that rule lives, and
//! the binding tokenizes every text before it releases the lock, in `add`
//! and in `query`, so the callable runs under the lock the caller already
//! holds and the engine's guards are taken afterwards on terms already
//! collected. A load never runs the tokenizer at all; it stores it.
//!
//! # What a failure becomes
//!
//! An exception the callable raises, a return value that is not iterable,
//! and an item that is not a `str` each become `Error::TokenizerFailed`
//! carrying the `PyErr`, and the binding raises that `PyErr` itself at the
//! boundary, so the caller sees their own exception with its own traceback.
//! Inside `add` the failure is a per record error, counted and named, as a
//! malformed vector is.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyString;
use zeusdb_vector_core::Error;
use zeusdb_vector_hnsw::{SimpleTokenizer, Tokenizer, TokenizerConfig};

use crate::PyEngineError;

/// A Python callable as the engine's tokenizer. Declared `external`, which
/// is the trait's default, since the engine cannot write it down.
pub(crate) struct PyTokenizer {
    callable: Py<PyAny>,
}

impl Tokenizer for PyTokenizer {
    fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
        Python::attach(|py| {
            let terms = self.callable.bind(py).call1((text,)).map_err(failed)?;
            for item in terms.try_iter().map_err(failed)? {
                let item = item.map_err(failed)?;
                let Ok(string) = item.cast::<PyString>() else {
                    let found = item
                        .get_type()
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_else(|_| "an object".to_string());
                    return Err(failed(pyo3::exceptions::PyTypeError::new_err(format!(
                        "The tokenizer returned {} where a str was expected. A tokenizer \
                         returns an iterable of str, one per term.",
                        found
                    ))));
                };
                term(string.to_str().map_err(failed)?);
            }
            Ok(())
        })
    }
}

/// A Python failure inside the tokenizer, carried out through the engine.
fn failed(err: PyErr) -> Error {
    Error::TokenizerFailed(Box::new(err))
}

/// The tokenizer a Python argument names.
///
/// `"simple"` is the built-in tokenizer, spelled as `config.json` records
/// it, and a callable is the caller's own. `"external"` is what a directory
/// records a caller's own as, and it names an implementation without
/// supplying one, so it is refused with the remedy. `name` is the argument
/// as the caller wrote it, for the message.
pub(crate) fn tokenizer_from_python(
    argument: &Bound<PyAny>,
    name: &str,
) -> Result<Arc<dyn Tokenizer>, PyEngineError> {
    if let Ok(spelled) = argument.extract::<String>() {
        if spelled == TokenizerConfig::Simple.name() {
            return Ok(Arc::new(SimpleTokenizer));
        }
        if spelled == TokenizerConfig::External.name() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{}='external' names a tokenizer of the caller's own without supplying it. \
                 Pass the callable itself, or 'simple' for the built-in tokenizer.",
                name
            ))
            .into());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}='{}' names a tokenizer this build does not have. Pass 'simple' for the \
             built-in tokenizer, or a callable of your own that takes a str and returns \
             an iterable of str.",
            name, spelled
        ))
        .into());
    }
    if argument.is_callable() {
        return Ok(Arc::new(PyTokenizer {
            callable: argument.clone().unbind(),
        }));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} must be 'simple' or a callable that takes a str and returns an iterable of \
         str, got {}",
        name,
        argument.get_type().name()?
    ))
    .into())
}
