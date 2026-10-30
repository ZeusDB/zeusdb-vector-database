//! What a test that stops a thread at a pause reaches the engine through,
//! being the checks of the interpreter rule and of a write made while a
//! search is held. Compiled under this crate's `test-support` feature alone,
//! so the wheel and every other build have none of it.
//!
//! `_Pause` is a private class of `_engine`. A test arms a thread on one,
//! either at the start of the released part of the thread's next call or at
//! a named point inside a mutation, and the thread waits there until another
//! thread releases it or the pause's limit passes; see
//! `zeusdb_vector_core::Pause`. Every method of a Python class runs with the
//! interpreter held, so a release that finds the armed thread waiting shows
//! that the call it paused did not hold the interpreter at the pause.

use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeusdb_vector_core::{
    pause_arm, pause_call, pause_disarm, PanicPoint, Pause, PauseAt, PauseState,
};

/// Pause here where this thread is armed for its next call, under the
/// call's name. `run_released!` reaches it at the start of every released
/// part of a call made through `HNSWIndex::released` or the wait of
/// `HNSWIndex::cheap`, and `detach_paused!` at the start of the released
/// part of a search.
pub(crate) fn pause_here(call: &'static str) {
    pause_call(call);
}

/// One pause, as a test arms, waits for and releases it.
#[pyclass(name = "_Pause", frozen)]
pub struct PyPause {
    pause: Arc<Pause>,
}

#[pymethods]
impl PyPause {
    /// A pause no thread has reached, which holds the thread that reaches it
    /// for `limit` seconds at most. `limit` is a number from 0 to 3600, and
    /// any other value raises `ValueError`. A limit of 0 records the call or
    /// the point that reached the pause and holds nothing.
    #[new]
    #[pyo3(signature = (limit = 30.0))]
    fn new(limit: f64) -> PyResult<Self> {
        let longest = Pause::LONGEST.as_secs_f64();
        if !(0.0..=longest).contains(&limit) {
            return Err(PyValueError::new_err(format!(
                "limit is a number of seconds from 0 to {longest}, got {limit}"
            )));
        }
        Ok(PyPause {
            pause: Arc::new(Pause::new(Duration::from_secs_f64(limit))),
        })
    }

    /// Arm the calling thread to pause here at the start of the released
    /// part of its next call, in place of any pause armed on it before.
    fn arm_next_call(&self) {
        pause_arm(PauseAt::NextCall, Arc::clone(&self.pause));
    }

    /// Arm the calling thread to pause here at the named point inside a
    /// mutation, in place of any pause armed on it before. A name that is
    /// no point's raises `ValueError` naming the points.
    fn arm_at(&self, point: &str) -> PyResult<()> {
        let Some(named) = PanicPoint::named(point) else {
            let points: Vec<&str> = PanicPoint::ALL.iter().map(|p| p.name()).collect();
            return Err(PyValueError::new_err(format!(
                "no point is named '{point}'. The points are {}",
                points.join(", ")
            )));
        };
        pause_arm(PauseAt::Point(named), Arc::clone(&self.pause));
        Ok(())
    }

    /// Pause nowhere on the calling thread.
    #[staticmethod]
    fn disarm() {
        pause_disarm();
    }

    /// Wait with the interpreter released until a thread reaches the pause
    /// or its limit passes, and give the name of the call or the point that
    /// reached it, or `None` where no thread has.
    fn wait_reached(&self, py: Python<'_>) -> Option<&'static str> {
        let pause = &self.pause;
        py.detach(|| pause.wait_reached(pause.limit())).reached()
    }

    /// Let the thread waiting at the pause go, and say whether a thread was
    /// waiting. A release that finds no thread waiting changes nothing.
    fn release(&self) -> bool {
        self.pause.release()
    }

    /// The name of the call or the point that reached the pause, or `None`.
    #[getter]
    fn reached(&self) -> Option<&'static str> {
        self.pause.state().reached()
    }

    /// `"armed"` before any thread reaches the pause, `"waiting"` while one
    /// waits at it, `"released"` once a release let it go, and
    /// `"timed_out"` once the limit let it go.
    #[getter]
    fn state(&self) -> &'static str {
        match self.pause.state() {
            PauseState::Armed => "armed",
            PauseState::Waiting(_) => "waiting",
            PauseState::Released(_) => "released",
            PauseState::TimedOut(_) => "timed_out",
        }
    }
}
