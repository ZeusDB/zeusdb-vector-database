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
//! one a call. The wait for the interval ends when the interval has passed
//! or the stop is set, and a wakeup for any other reason, a spurious one
//! among them, does not end it.
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
//! mutations until a checkpoint commits, exactly as it does under the
//! per-call policy; see `Collection::commit_records`. A later sync that
//! succeeds clears nothing, the sink's own sync for a backup or for a
//! checkpoint among them, since it does not show that the records the failed
//! sync covered reached the device.
//!
//! # The roll of a checkpoint
//!
//! A checkpoint's roll replaces the segment the thread syncs. The roll waits
//! for a sync the thread has begun to end, and the thread begins none while
//! the roll runs, so every sync of the thread ended before the roll, of a
//! segment the roll replaced, or began after it, of the segment the roll
//! started or a later one. The roll of a checkpoint that commits then
//! discards a failure the thread holds, since the directory that checkpoint
//! wrote holds every record the replaced segments held, and every failure of
//! a sync begun after the roll stands.
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
//! because a condition variable waits on the standard guard alone. A roll
//! waits for the thread's sync holding the sink's guard, and the thread takes
//! no guard of the collection's, so the wait closes no cycle.

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
    /// Wakes the thread: a commit that dirties a clean file, the end of a
    /// roll, and the drop.
    wake: Condvar,
    /// Wakes a roll that waits for the thread's sync to end.
    synced: Condvar,
    /// Syncs the thread has completed, which the log record of its stop
    /// reports.
    syncs: AtomicU64,
}

#[derive(Debug, Default)]
struct State {
    /// Bytes have been committed since the last sync.
    dirty: bool,
    /// The sink is being dropped.
    stop: bool,
    /// The thread is inside a sync, from the hold that clears `dirty` to the
    /// hold that records what the sync said.
    syncing: bool,
    /// A checkpoint's roll is replacing the segment, so the thread begins no
    /// sync.
    rolling: bool,
    /// The failure of the last sync that failed, held until a commit takes
    /// it or the roll of a checkpoint that commits discards it. A sync that
    /// succeeds leaves it.
    failed: Option<Error>,
    /// The holds of the state at the thread's wait for the interval, one as
    /// the wait begins and one at each wakeup, which a test reads to know the
    /// thread has acted on a wakeup it made.
    #[cfg(test)]
    interval_checks: u64,
    /// A checkpoint's roll is inside its wait for a sync of the thread to
    /// end, which a test reads to know the roll waits.
    #[cfg(test)]
    roll_waiting: bool,
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
            synced: Condvar::new(),
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
    /// to do until the next commit. A failed sync the thread holds stays
    /// held, since a sync that succeeds does not show that the records the
    /// failed one covered reached the device. Only the roll of a checkpoint
    /// that commits discards it; see [`Rolling::committed`].
    pub(crate) fn mark_clean(&self) {
        self.shared.state.lock().unwrap().dirty = false;
    }

    /// Begin a checkpoint's roll: wait for a sync the thread has begun to
    /// end, and keep the thread from beginning another until the returned
    /// value is dropped.
    pub(crate) fn roll(&self) -> Rolling<'_> {
        let mut state = self.shared.state.lock().unwrap();
        state.rolling = true;
        while state.syncing {
            #[cfg(test)]
            {
                state.roll_waiting = true;
            }
            state = self.shared.synced.wait(state).unwrap();
            #[cfg(test)]
            {
                state.roll_waiting = false;
            }
        }
        Rolling {
            shared: &self.shared,
        }
    }

    /// The failure the thread holds, once, as the sync said it. `None` where
    /// no sync has failed since a commit last took one or the roll of a
    /// checkpoint that commits discarded one.
    pub(crate) fn take_failure(&self) -> Option<Error> {
        self.shared.state.lock().unwrap().failed.take()
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

/// A checkpoint's roll in progress, from [`Flusher::roll`]. The thread
/// begins no sync until it is dropped.
#[derive(Debug)]
pub(crate) struct Rolling<'a> {
    shared: &'a Shared,
}

impl Rolling<'_> {
    /// The checkpoint committed and the journal rolled onto a segment naming
    /// it. Every sync of the thread ended before the roll, of a segment the
    /// checkpoint holds every record of, so a failure the thread holds is
    /// discarded, and there is nothing to sync until the next commit.
    pub(crate) fn committed(self) {
        let mut state = self.shared.state.lock().unwrap();
        state.failed = None;
        state.dirty = false;
    }
}

impl Drop for Rolling<'_> {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().rolling = false;
        self.shared.wake.notify_one();
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
        // Clean, or a checkpoint's roll replacing the segment, and not
        // stopping: sleep until a commit, the roll's end or the drop wakes
        // us. No timeout, so an idle collection costs no wakeup at all.
        while (!state.dirty || state.rolling) && !state.stop {
            state = shared.wake.wait(state).unwrap();
        }
        if state.stop && !state.dirty {
            break;
        }
        // Dirty. Give the interval for more commits to gather behind the
        // one that woke us, unless the collection is going away, in which
        // case what is there is synced now. The wait ends at the interval or
        // at the stop, whatever else wakes it.
        if !state.stop {
            state = shared
                .wake
                .wait_timeout_while(state, interval, |state| {
                    #[cfg(test)]
                    {
                        state.interval_checks += 1;
                    }
                    !state.stop
                })
                .unwrap()
                .0;
            // A checkpoint may have synced the file itself while we waited,
            // in which case there is nothing to do, or may be rolling it, in
            // which case the roll's end wakes us.
            if !state.dirty || state.rolling {
                continue;
            }
        }
        state.dirty = false;
        state.syncing = true;
        drop(state);
        let synced = handle.sync();
        if let Err(error) = &synced {
            error!(target: LOG_TARGET, operation = "journal_sync_failed",
                journal = %path.display(),
                error = %error,
                "The journal's sync thread could not sync the file; the next commit fails \
                 with this"
            );
        }
        state = shared.state.lock().unwrap();
        state.syncing = false;
        match synced {
            Ok(()) => {
                shared.syncs.fetch_add(1, Ordering::Release);
            }
            Err(error) => state.failed = Some(error),
        }
        shared.synced.notify_all();
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

    /// Whether the thread holds a failed sync that no commit has taken,
    /// while the state is alive.
    pub(crate) fn holds_failure(&self) -> Option<bool> {
        self.0
            .upgrade()
            .map(|shared| shared.state.lock().unwrap().failed.is_some())
    }

    /// Whether the file is marked dirty, so the thread has a sync to make,
    /// while the state is alive.
    pub(crate) fn is_dirty(&self) -> Option<bool> {
        self.0
            .upgrade()
            .map(|shared| shared.state.lock().unwrap().dirty)
    }

    /// Whether a checkpoint's roll is inside its wait for a sync of the
    /// thread to end, while the state is alive. The roll keeps the state
    /// from setting the mark until it waits, and from its wakeup until it
    /// has cleared the mark, so a mark read here is a wait in progress.
    pub(crate) fn roll_waits(&self) -> Option<bool> {
        self.0
            .upgrade()
            .map(|shared| shared.state.lock().unwrap().roll_waiting)
    }

    /// The holds of the state at the thread's wait for the interval, while
    /// the state is alive. The thread keeps the state from each hold until
    /// it waits again or has cleared the mark to sync, so a count read here
    /// is one the thread has acted on.
    pub(crate) fn interval_checks(&self) -> Option<u64> {
        self.0
            .upgrade()
            .map(|shared| shared.state.lock().unwrap().interval_checks)
    }

    /// Wake the thread with nothing changed, as a spurious wakeup does.
    pub(crate) fn wake(&self) {
        if let Some(shared) = self.0.upgrade() {
            shared.wake.notify_all();
        }
    }
}
