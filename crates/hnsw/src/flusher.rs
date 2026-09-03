//! The thread an interval policy syncs from.
//!
//! A journal under [`Durability::PerInterval`](crate::Durability::PerInterval)
//! commits with the bytes in the kernel and nothing on the device, and this
//! thread is what puts them there. It holds a handle that syncs the file and
//! does nothing else, so it never takes the sink's guard, and it wakes only
//! when there is something to sync.
//!
//! # What it does
//!
//! The sink marks the file dirty at each deferred commit. The thread sleeps
//! on a condition variable while the file is clean, so a collection idle
//! for an hour wakes it not once. The first commit after a clean stretch
//! wakes it; it then waits one interval, so the records of the calls that
//! follow gather behind the first, clears the mark and syncs once. A record
//! is therefore on the device within one interval of the call that wrote
//! it returning, and a burst of calls costs one sync an interval rather than
//! one a call.
//!
//! # What stops it
//!
//! Dropping the [`Flusher`], which the sink owns and the collection owns in
//! turn. The drop sets the stop flag, wakes the thread and joins it, and the
//! thread syncs once more before it exits where the file is dirty, so a
//! collection dropped cleanly leaves nothing for the kernel to write. No
//! thread outlives the collection it syncs for.
//!
//! # A sync that fails
//!
//! Is logged at `error`, held, and handed to the sink at its next commit,
//! which fails with it. A failed commit is the one thing the seam refuses
//! records over, so a device that will not flush stops the collection taking
//! mutations until a checkpoint reconciles the journal, exactly as it does
//! under the per-call policy; see `Collection::commit_records`.
//!
//! # The lock
//!
//! The state below is a standard mutex rather than one from the lock rank
//! registry. The registry orders the collection's guards against each other,
//! and this mutex is held by no collection guard's holder for longer than a
//! flag flip: the sink takes it under the collection's sink guard, which is
//! the last leaf of the declared order, and the thread takes it holding
//! nothing. A rank would state that it sits below the last leaf, which is
//! what a leaf already means. The registry cannot wrap it in any case,
//! because a condition variable waits on the standard guard alone.

#![allow(clippy::disallowed_types)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use tracing::{debug, error};
use zeusdb_vector_core::{Error, JournalSyncHandle};

/// The target the log records in this module carry. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::journal";

/// The thread, and the state it shares with the sink.
#[derive(Debug)]
pub(crate) struct Flusher {
    shared: Arc<Shared>,
    path: PathBuf,
    thread: Option<JoinHandle<()>>,
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// Syncs the thread has completed. Read by the tests that hold the
    /// thread to costing nothing on a clean file.
    syncs: AtomicU64,
}

#[derive(Debug, Default)]
struct State {
    /// Bytes have been committed since the last sync.
    dirty: bool,
    /// The sink is being dropped.
    stop: bool,
    /// What the last sync said, where it failed and nothing has taken it.
    failed: Option<String>,
}

impl Flusher {
    /// Start the thread over `handle`, syncing `interval` after the first
    /// commit that dirties the file.
    pub(crate) fn start(
        handle: JournalSyncHandle,
        path: PathBuf,
        interval: Duration,
    ) -> Result<Self, Error> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            syncs: AtomicU64::new(0),
        });
        let worker = Arc::clone(&shared);
        let worker_path = path.clone();
        let thread = std::thread::Builder::new()
            .name("zeusdb-journal-sync".to_string())
            .spawn(move || run(worker, handle, worker_path, interval))
            .map_err(|e| Error::JournalIoFailed {
                path: path.clone(),
                what: "start the sync thread for",
                error: e.to_string(),
            })?;
        debug!(target: LOG_TARGET, operation = "journal_sync_thread_start",
            journal = %path.display(),
            interval_ms = interval.as_millis() as u64,
            "The journal's sync thread is running"
        );
        Ok(Flusher {
            shared,
            path,
            thread: Some(thread),
        })
    }

    /// Bytes were committed and are not on the device. Wakes the thread on
    /// the clean to dirty edge and on no other, so a burst of commits inside
    /// one interval wakes it once.
    pub(crate) fn mark_dirty(&self) {
        let mut state = self.shared.state.lock().unwrap();
        if !state.dirty {
            state.dirty = true;
            self.shared.wake.notify_one();
        }
    }

    /// The sink synced the file itself, so there is nothing for the thread
    /// to do until the next commit, and a sync the thread could not
    /// complete before it no longer matters, since everything it would
    /// have written is on the device now.
    pub(crate) fn mark_clean(&self) {
        let mut state = self.shared.state.lock().unwrap();
        state.dirty = false;
        state.failed = None;
    }

    /// The failure of the last sync, once. `None` where every sync since the
    /// last call succeeded.
    pub(crate) fn take_failure(&self) -> Option<Error> {
        let failed = self.shared.state.lock().unwrap().failed.take()?;
        Some(Error::JournalIoFailed {
            path: self.path.clone(),
            what: "sync",
            error: failed,
        })
    }

    /// Syncs the thread has completed since it started.
    pub(crate) fn syncs(&self) -> u64 {
        self.shared.syncs.load(Ordering::Acquire)
    }

    /// A weak handle on the thread's state, for a test that holds the
    /// thread to stopping with the collection: it upgrades while either the
    /// sink or the thread holds the state and fails once both have let go.
    #[cfg(test)]
    pub(crate) fn watch(&self) -> Watch {
        Watch(Arc::downgrade(&self.shared))
    }
}

impl Drop for Flusher {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.stop = true;
        }
        self.shared.wake.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        debug!(target: LOG_TARGET, operation = "journal_sync_thread_stop",
            journal = %self.path.display(),
            syncs = self.syncs(),
            "The journal's sync thread has stopped"
        );
    }
}

/// The thread's body.
fn run(shared: Arc<Shared>, handle: JournalSyncHandle, path: PathBuf, interval: Duration) {
    let mut state = shared.state.lock().unwrap();
    loop {
        // Clean and not stopping: sleep until a commit or the drop wakes
        // us. No timeout, so an idle collection costs no wakeup at all.
        while !state.dirty && !state.stop {
            state = shared.wake.wait(state).unwrap();
        }
        if state.stop && !state.dirty {
            break;
        }
        // Dirty. Give the interval for more commits to gather behind the
        // one that woke us, unless the collection is going away, in which
        // case what is there is synced now.
        if !state.stop {
            let (guard, _) = shared.wake.wait_timeout(state, interval).unwrap();
            state = guard;
            // A checkpoint may have synced the file itself while we waited,
            // in which case there is nothing to do.
            if !state.dirty {
                continue;
            }
        }
        state.dirty = false;
        drop(state);
        match handle.sync() {
            Ok(()) => {
                shared.syncs.fetch_add(1, Ordering::Release);
            }
            Err(e) => {
                let message = e.to_string();
                error!(target: LOG_TARGET, operation = "journal_sync_failed",
                    journal = %path.display(),
                    error = %message,
                    "The journal's sync thread could not sync the file; the next commit fails \
                     with this"
                );
                shared.state.lock().unwrap().failed = Some(message);
            }
        }
        state = shared.state.lock().unwrap();
    }
}

/// See [`Flusher::watch`].
#[cfg(test)]
#[derive(Debug)]
pub struct Watch(std::sync::Weak<Shared>);

#[cfg(test)]
impl Watch {
    /// Whether the sink or the thread still holds the state.
    pub(crate) fn alive(&self) -> bool {
        self.0.strong_count() > 0
    }

    /// Syncs completed, while the state is alive.
    pub(crate) fn syncs(&self) -> Option<u64> {
        self.0
            .upgrade()
            .map(|shared| shared.syncs.load(Ordering::Acquire))
    }
}
