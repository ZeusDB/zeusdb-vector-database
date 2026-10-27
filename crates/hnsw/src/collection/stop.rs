//! The rule a panic inside a call keeps: it stops the index.
//!
//! A panic inside any call on an index may have left a change part made,
//! inside the guard it held or across guards it had let go, and the index
//! cannot tell which. So a panic stops the index. The call that panicked and
//! every call after it are refused with [`Error::Panicked`], naming the first
//! call that panicked and what its panic said, and none of them reads or
//! writes the index. No save, checkpoint or backup writes the state the panic
//! left, so a directory and its journal hold every call whose commit landed,
//! and dropping the index and loading the directory restores them. The
//! journal records nothing for the panic: a call whose commit landed and
//! whose install panicked is replayed by the next load.
//!
//! [`Collection::run`] enforces the rule, and the binding makes every call on
//! an index through it, so no panic reaches the boundary into Python. A call
//! already running when another panics finishes as it would beside any change
//! in flight, and a poisoned lock it then meets panics inside its own `run`,
//! which refuses it in the same words. A load and the creation of an index
//! have no index to stop and go through [`run_without_index`], which refuses
//! the call that panicked and nothing else.
//!
//! The lock wrappers in `crate::locks` hand a poisoned lock on as they find
//! it. After the first panic no call reaches a guard, so they keep no rule of
//! their own.

use super::Collection;
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use tracing::error;
use zeusdb_vector_core::Error;

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::stop";

/// What stopped an index: the first call that panicked inside it, as a
/// caller writes the call, and what the panic said.
#[derive(Debug)]
pub(crate) struct Stop {
    operation: &'static str,
    cause: String,
}

impl Stop {
    /// The refusal every call on the stopped index returns.
    fn refusal(&self) -> Error {
        Error::Panicked {
            operation: self.operation,
            cause: self.cause.clone(),
            stopped: true,
        }
    }
}

/// What a panic said: its message where it carries one as text, and a
/// phrase saying it carried none otherwise.
fn panic_words(payload: &(dyn Any + Send)) -> String {
    if let Some(words) = payload.downcast_ref::<&str>() {
        (*words).to_string()
    } else if let Some(words) = payload.downcast_ref::<String>() {
        words.clone()
    } else {
        "a panic that carried no message".to_string()
    }
}

impl Collection {
    /// Run `call` under the rule a panic keeps, where `operation` names the
    /// call as a caller writes it.
    ///
    /// On a stopped index the call is refused before it runs. Otherwise it
    /// runs, and where it panics the index stops, recording `operation` and
    /// the panic's words unless an earlier panic already stopped it, and the
    /// call returns the refusal. Where nothing panics this is one load of the
    /// stop and one frame around the call.
    pub fn run<T, E: From<Error>>(
        &self,
        operation: &'static str,
        call: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if let Some(stop) = self.stopped.get() {
            return Err(stop.refusal().into());
        }
        match panic::catch_unwind(AssertUnwindSafe(call)) {
            Ok(result) => result,
            Err(payload) => {
                let stop = self.stopped.get_or_init(|| {
                    let cause = panic_words(&*payload);
                    error!(
                        target: LOG_TARGET,
                        operation = operation,
                        cause = %cause,
                        "A call failed inside the engine and stopped the index, which refuses every call from now on"
                    );
                    Stop { operation, cause }
                });
                Err(stop.refusal().into())
            }
        }
    }

    /// Whether a panic has stopped this index.
    pub fn is_stopped(&self) -> bool {
        self.stopped.get().is_some()
    }
}

/// Run `call`, which has no index to stop, and refuse it where it panics with
/// [`Error::Panicked`] naming `operation` and what the panic said. What a load
/// and the creation of an index go through.
pub fn run_without_index<T, E: From<Error>>(
    operation: &'static str,
    call: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    panic::catch_unwind(AssertUnwindSafe(call)).unwrap_or_else(|payload| {
        let cause = panic_words(&*payload);
        error!(
            target: LOG_TARGET,
            operation = operation,
            cause = %cause,
            "A call failed inside the engine"
        );
        Err(Error::Panicked {
            operation,
            cause,
            stopped: false,
        }
        .into())
    })
}
