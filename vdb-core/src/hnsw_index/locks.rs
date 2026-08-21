//! Every lock on [`super::HNSWIndex`], with its place in the declared order.
//!
//! # What this is for
//!
//! The type declares an acquisition order and one further rule, that the same
//! guard is never taken twice on one thread. Both were enforced by reading.
//! Three separate relays broke one of them, and each was found by a person
//! reading the code after the tests were green.
//!
//! | Relay | Site | Shape |
//! | --- | --- | --- |
//! | 69 | `get_stats` re-read `training_ids` inside its own hold | nested, same lock |
//! | 86 | `search_candidates` called `is_quantized`, which takes `hnsw` again | nested, same lock, through a call |
//! | 97 | `warn_undeclared_filter_field` read `columns` while the caller held it | nested, same lock, through a call |
//!
//! A recursive read is not a deadlock on its own. It becomes one when a writer
//! queues between the two acquisitions, because the standard library queues
//! readers behind a waiting writer. So a single threaded test passes on a build
//! carrying the defect, and the concurrency suite hangs only if it happens to
//! schedule a writer into the window. **That is what this file changes.** The
//! assertion fires on the second acquisition itself, with no writer and no
//! scheduling, so an ordinary `pytest tests` on a debug build finds it.
//!
//! It catches order inversion too, which reading catches least reliably,
//! because an inversion is a property of two call paths rather than of one
//! function.
//!
//! # What it costs
//!
//! In release, nothing. The rank lives in a const generic, the tracked types
//! are `#[repr(transparent)]` over the standard ones, the guards have no `Drop`
//! of their own, and every registry call is behind `#[cfg(debug_assertions)]`.
//! [`tests::a_tracked_lock_is_the_size_of_the_lock_it_wraps`] asserts the sizes,
//! and it is the one test here that is not itself gated on
//! `debug_assertions`, so it runs on a release build too. Measured beyond that:
//! the three assertion messages below appear once each in
//! `target/debug/zeusdb_vector_database.dll` and zero times in
//! `target/release/zeusdb_vector_database.dll`, so the whole body is gone
//! rather than merely unreachable.
//!
//! In debug, one thread local access, a scan of at most fourteen entries and a
//! push per acquisition, against a lock acquisition that already costs an
//! atomic.
//!
//! # Why a field added later cannot bypass it
//!
//! `clippy.toml` disallows `std::sync::RwLock` and `std::sync::Mutex` by name,
//! and the lint gate runs `-D warnings`, so a bare lock anywhere in the crate
//! fails the build. The two modules that legitimately hold one, `pq` and
//! `graph`, carry a module level allow that says why. This file carries one
//! because it is where the wrapping happens.
//!
//! # What it does not catch
//!
//! A guard already dropped. Relay 98's `count` read `rev_map` in a match guard
//! and again in the arm, which is two sequential acquisitions rather than a
//! nested hold, because a match guard is its own temporary scope. The registry
//! sees the first release before the second acquisition and says nothing. That
//! occurrence was a staleness defect rather than a hang, and it is out of this
//! mechanism's reach by construction.
//!
//! Nor does it see across threads. The held set is per thread, so a guard held
//! across a rayon fork is invisible to the workers, which is correct: the rule
//! the order encodes is about one thread's own acquisitions.

#![allow(clippy::disallowed_types)]

use std::sync::{LockResult, Mutex, PoisonError, RwLock};

/// The declared acquisition order, as a rank per lock.
///
/// **Ascending is earlier.** A thread may take a lock only when every lock it
/// already holds ranks strictly below it, so the numbers here are the order
/// [`super::HNSWIndex`] documents in prose, written down once in a form the
/// build can check.
///
/// The prose order is
///
/// ```text
/// id_map < rev_map < hnsw < pq_codes < vector_metadata < columns
///        < training_ids < metadata < id_counter < vector_count
/// ```
///
/// `writers` sits above all of them because the mutating Python entry points
/// take it before any guard and no internal helper takes it at all.
///
/// `rerank_calibration`, `training_completed_at` and `created_at` are the
/// leaves. The prose says they are never held together with any other guard,
/// which is stronger than a rank can express, so they take the bottom ranks:
/// anything may be held while one of them is taken, and none of them may be
/// held while anything else is. That is the weaker half of the claim, and it is
/// the half a rank can state without inventing a rule the code has not agreed
/// to.
pub(crate) mod order {
    /// The mutation lock, taken by a Python entry point before any guard.
    pub(crate) const WRITERS: u8 = 0;
    pub(crate) const ID_MAP: u8 = 1;
    pub(crate) const REV_MAP: u8 = 2;
    pub(crate) const HNSW: u8 = 3;
    pub(crate) const PQ_CODES: u8 = 4;
    pub(crate) const VECTOR_METADATA: u8 = 5;
    pub(crate) const COLUMNS: u8 = 6;
    pub(crate) const TRAINING_IDS: u8 = 7;
    pub(crate) const METADATA: u8 = 8;
    pub(crate) const ID_COUNTER: u8 = 9;
    pub(crate) const VECTOR_COUNT: u8 = 10;
    /// A leaf. See the module documentation.
    pub(crate) const RERANK_CALIBRATION: u8 = 11;
    /// A leaf.
    pub(crate) const TRAINING_COMPLETED_AT: u8 = 12;
    /// A leaf.
    pub(crate) const CREATED_AT: u8 = 13;
}

/// How a rank names itself in an assertion a developer reads.
///
/// A `const` cannot carry a name, so this is the one place the numbers and the
/// field names are paired. A rank with no name here is a rank nobody declared,
/// which is why the fallback says so rather than printing the number alone.
#[cfg(debug_assertions)]
const fn name_of(rank: u8) -> &'static str {
    match rank {
        order::WRITERS => "writers",
        order::ID_MAP => "id_map",
        order::REV_MAP => "rev_map",
        order::HNSW => "hnsw",
        order::PQ_CODES => "pq_codes",
        order::VECTOR_METADATA => "vector_metadata",
        order::COLUMNS => "columns",
        order::TRAINING_IDS => "training_ids",
        order::METADATA => "metadata",
        order::ID_COUNTER => "id_counter",
        order::VECTOR_COUNT => "vector_count",
        order::RERANK_CALIBRATION => "rerank_calibration",
        order::TRAINING_COMPLETED_AT => "training_completed_at",
        order::CREATED_AT => "created_at",
        _ => "a lock with no declared place in the order",
    }
}

// ============================================================================
// THE REGISTRY
// ============================================================================

// What this thread holds, innermost last. A `Vec` rather than a bitset because
// the assertion has to name what is already held, and because fourteen entries
// scanned linearly is cheaper than anything cleverer at this size.
#[cfg(debug_assertions)]
thread_local! {
    static HELD: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that this thread is taking `rank`, asserting it may.
///
/// Called **before** the acquisition rather than after, so a recursive read
/// fails the assertion instead of blocking on the writer that would have turned
/// it into a deadlock. Registering a lock the thread then waits for is exactly
/// right for that purpose: what the set answers is "what would this thread be
/// holding", and a pending acquisition is already committed.
#[cfg(debug_assertions)]
fn enter(rank: u8) {
    let _ = HELD.try_with(|held| {
        let mut held = held.borrow_mut();
        assert!(
            !held.contains(&rank),
            "this thread already holds {}, and taking the same guard twice on one \
             thread is forbidden even for reads. The standard library queues readers \
             behind a waiting writer, so the second acquisition blocks forever the \
             moment a writer lands between them. Held: {:?}",
            name_of(rank),
            held.iter().map(|&r| name_of(r)).collect::<Vec<_>>()
        );
        if let Some(&blocking) = held.iter().find(|&&other| other > rank) {
            panic!(
                "this thread holds {} and is taking {}, which inverts the declared \
                 lock order. Held: {:?}",
                name_of(blocking),
                name_of(rank),
                held.iter().map(|&r| name_of(r)).collect::<Vec<_>>()
            );
        }
        held.push(rank);
    });
}

/// Record that this thread has let `rank` go.
///
/// Removes by value rather than popping, because a struct holding several
/// guards drops its fields in declaration order, which is the order they were
/// acquired in. So the releases are first in first out and a stack would
/// mispair them.
///
/// Asserts nothing. It runs from `Drop`, which may be running inside an unwind
/// from the assertion above, and a panic there aborts the process.
#[cfg(debug_assertions)]
fn leave(rank: u8) {
    let _ = HELD.try_with(|held| {
        let mut held = held.borrow_mut();
        if let Some(at) = held.iter().rposition(|&other| other == rank) {
            held.remove(at);
        }
    });
}

/// What this thread holds, by name, for a test that wants to assert on it.
#[cfg(all(test, debug_assertions))]
fn held_now() -> Vec<&'static str> {
    HELD.try_with(|held| held.borrow().iter().map(|&r| name_of(r)).collect())
        .unwrap_or_default()
}

// ============================================================================
// THE LOCKS
// ============================================================================

/// An `RwLock` that knows its place in the declared order.
///
/// The rank is a const generic, so it sits in the field's own declaration
/// beside the documentation that explains what the field holds, and the two
/// constructors cannot disagree about it.
#[repr(transparent)]
pub(crate) struct RwLockAt<const RANK: u8, T> {
    inner: RwLock<T>,
}

impl<const RANK: u8, T> RwLockAt<RANK, T> {
    pub(crate) const fn new(value: T) -> Self {
        RwLockAt {
            inner: RwLock::new(value),
        }
    }

    pub(crate) fn read(&self) -> LockResult<ReadGuard<'_, RANK, T>> {
        #[cfg(debug_assertions)]
        enter(RANK);
        match self.inner.read() {
            Ok(guard) => Ok(ReadGuard { inner: guard }),
            Err(poison) => Err(PoisonError::new(ReadGuard {
                inner: poison.into_inner(),
            })),
        }
    }

    pub(crate) fn write(&self) -> LockResult<WriteGuard<'_, RANK, T>> {
        #[cfg(debug_assertions)]
        enter(RANK);
        match self.inner.write() {
            Ok(guard) => Ok(WriteGuard { inner: guard }),
            Err(poison) => Err(PoisonError::new(WriteGuard {
                inner: poison.into_inner(),
            })),
        }
    }
}

/// A `Mutex` that knows its place in the declared order.
#[repr(transparent)]
pub(crate) struct MutexAt<const RANK: u8, T> {
    inner: Mutex<T>,
}

impl<const RANK: u8, T> MutexAt<RANK, T> {
    pub(crate) const fn new(value: T) -> Self {
        MutexAt {
            inner: Mutex::new(value),
        }
    }

    pub(crate) fn lock(&self) -> LockResult<Locked<'_, RANK, T>> {
        #[cfg(debug_assertions)]
        enter(RANK);
        match self.inner.lock() {
            Ok(guard) => Ok(Locked { inner: guard }),
            Err(poison) => Err(PoisonError::new(Locked {
                inner: poison.into_inner(),
            })),
        }
    }
}

// ============================================================================
// THE GUARDS
// ============================================================================

/// Declare one guard type, its dereference and its release.
///
/// Three guards differing only in the standard type they wrap. Written out
/// three times it is the same forty lines three times over, and the part that
/// has to be identical between them is exactly the part a macro makes
/// identical.
macro_rules! guard {
    ($name:ident, $inner:ident) => {
        /// A guard that pops its lock's rank off the held set when it is
        /// dropped.
        ///
        /// In release it is a `#[repr(transparent)]` wrapper with no `Drop` of
        /// its own, so it is the standard guard by another name and the
        /// standard guard's own release is all that runs.
        #[repr(transparent)]
        pub(crate) struct $name<'a, const RANK: u8, T> {
            inner: std::sync::$inner<'a, T>,
        }

        impl<const RANK: u8, T> std::ops::Deref for $name<'_, RANK, T> {
            type Target = T;
            #[inline]
            fn deref(&self) -> &T {
                &self.inner
            }
        }

        #[cfg(debug_assertions)]
        impl<const RANK: u8, T> Drop for $name<'_, RANK, T> {
            fn drop(&mut self) {
                leave(RANK);
            }
        }
    };
}

/// The mutable half, for the two guards that hand out `&mut`.
///
/// Separate from [`guard`] rather than a flag inside it, because a macro cannot
/// match on a captured `literal` fragment and the shapes that work around that
/// read worse than two invocations do.
macro_rules! guard_mut {
    ($name:ident) => {
        impl<const RANK: u8, T> std::ops::DerefMut for $name<'_, RANK, T> {
            #[inline]
            fn deref_mut(&mut self) -> &mut T {
                &mut self.inner
            }
        }
    };
}

guard!(ReadGuard, RwLockReadGuard);
guard!(WriteGuard, RwLockWriteGuard);
guard!(Locked, MutexGuard);

guard_mut!(WriteGuard);
guard_mut!(Locked);

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_types)]

    use super::*;
    use std::sync::{RwLockReadGuard, RwLockWriteGuard};

    /// The wrapper is the standard lock's size, so the rank costs no memory.
    ///
    /// This holds in debug as well as in release, because the rank is a const
    /// generic and lives in the type rather than in a field. The guards carry a
    /// `Drop` in debug and none in release, and neither changes their size.
    #[test]
    fn a_tracked_lock_is_the_size_of_the_lock_it_wraps() {
        assert_eq!(
            std::mem::size_of::<RwLockAt<{ order::HNSW }, Vec<u8>>>(),
            std::mem::size_of::<RwLock<Vec<u8>>>()
        );
        assert_eq!(
            std::mem::size_of::<MutexAt<{ order::WRITERS }, ()>>(),
            std::mem::size_of::<Mutex<()>>()
        );
        assert_eq!(
            std::mem::size_of::<ReadGuard<'_, { order::HNSW }, Vec<u8>>>(),
            std::mem::size_of::<RwLockReadGuard<'_, Vec<u8>>>()
        );
        assert_eq!(
            std::mem::size_of::<WriteGuard<'_, { order::HNSW }, Vec<u8>>>(),
            std::mem::size_of::<RwLockWriteGuard<'_, Vec<u8>>>()
        );
    }

    /// A second read on one thread is refused, which is the whole point.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "taking the same guard twice on one thread")]
    fn a_second_read_on_one_thread_is_refused() {
        let lock: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(7);
        let _first = lock.read().unwrap();
        let _second = lock.read().unwrap();
    }

    /// And the same for a write under a read.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "taking the same guard twice on one thread")]
    fn a_write_under_a_read_of_the_same_lock_is_refused() {
        let lock: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(7);
        let _first = lock.read().unwrap();
        let _second = lock.write().unwrap();
    }

    /// Two locks taken out of the declared order are refused.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "inverts the declared lock order")]
    fn taking_a_lower_lock_under_a_higher_one_is_refused() {
        let columns: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(1);
        let id_map: RwLockAt<{ order::ID_MAP }, u32> = RwLockAt::new(2);
        let _held = columns.read().unwrap();
        let _inverted = id_map.read().unwrap();
    }

    /// Two locks in the declared order are allowed, and both are released.
    #[test]
    #[cfg(debug_assertions)]
    fn the_declared_order_is_allowed_and_the_set_empties() {
        let id_map: RwLockAt<{ order::ID_MAP }, u32> = RwLockAt::new(1);
        let columns: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(2);
        {
            let first = id_map.read().unwrap();
            let second = columns.write().unwrap();
            assert_eq!(*first, 1);
            assert_eq!(*second, 2);
            assert_eq!(held_now(), vec!["id_map", "columns"]);
        }
        assert!(held_now().is_empty());
        // And the same pair again, so a release really did happen rather than
        // the first block merely ending.
        let _again = id_map.read().unwrap();
        assert_eq!(held_now(), vec!["id_map"]);
    }

    /// Guards dropped in acquisition order are paired correctly.
    ///
    /// A struct holding several guards drops its fields in declaration order,
    /// which is the order they were taken in, so the releases are first in
    /// first out. A stack would pop the wrong entry and leave the set wrong for
    /// every later acquisition on that thread.
    #[test]
    #[cfg(debug_assertions)]
    fn guards_released_in_acquisition_order_pair_correctly() {
        let id_map: RwLockAt<{ order::ID_MAP }, u32> = RwLockAt::new(1);
        let rev_map: RwLockAt<{ order::REV_MAP }, u32> = RwLockAt::new(2);
        let columns: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(3);

        let first = id_map.write().unwrap();
        let second = rev_map.write().unwrap();
        let third = columns.write().unwrap();
        assert_eq!(held_now(), vec!["id_map", "rev_map", "columns"]);
        drop(first);
        assert_eq!(held_now(), vec!["rev_map", "columns"]);
        drop(second);
        drop(third);
        assert!(held_now().is_empty());
    }

    /// The set is per thread, so one thread's hold does not bind another's.
    #[test]
    #[cfg(debug_assertions)]
    fn the_held_set_is_per_thread() {
        let lock: RwLockAt<{ order::COLUMNS }, u32> = RwLockAt::new(9);
        let _held = lock.read().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert!(held_now().is_empty());
                let _theirs = lock.read().unwrap();
                assert_eq!(held_now(), vec!["columns"]);
            });
        });
        assert_eq!(held_now(), vec!["columns"]);
    }

    /// Every declared rank has a name, so no assertion prints a bare number.
    #[test]
    #[cfg(debug_assertions)]
    fn every_declared_rank_is_named() {
        for rank in 0..=order::CREATED_AT {
            assert_ne!(
                name_of(rank),
                "a lock with no declared place in the order",
                "rank {} has no name",
                rank
            );
        }
        assert_eq!(
            name_of(order::CREATED_AT + 1),
            "a lock with no declared place in the order"
        );
    }
}
