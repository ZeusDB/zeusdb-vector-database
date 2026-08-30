//! Every lock on `Collection` and its spaces, with its place in the declared
//! order.
//!
//! # What this is for
//!
//! The type declares an acquisition order and one further rule, that the same
//! guard is never taken twice on one thread. This module makes both a debug
//! assertion rather than a convention.
//!
//! A recursive read is not a deadlock on its own. It becomes one when a writer
//! queues between the two acquisitions, because the standard library queues
//! readers behind a waiting writer, so a single threaded test passes on a build
//! that carries one. The assertion fires on the second acquisition itself, with
//! no writer and no scheduling, so an ordinary `pytest tests` on a debug build
//! reaches it.
//!
//! It catches order inversion too, which is a property of two call paths rather
//! than of one function.
//!
//! # What it costs
//!
//! In release, nothing. The rank is a field that exists only under
//! `debug_assertions`, the tracked types are `#[repr(transparent)]` over the
//! standard ones there, the guards have no `Drop` of their own, and every
//! registry call is behind `#[cfg(debug_assertions)]`.
//! [`tests::a_tracked_lock_is_the_size_of_the_lock_it_wraps`] asserts the
//! sizes, and it is the one test here that is not itself gated on
//! `debug_assertions`, so it runs on a release build too. The three assertion
//! messages below are absent from a release artefact, so the whole body is gone
//! rather than merely unreachable.
//!
//! In debug, one byte and its padding per lock, one thread local access, a
//! scan of at most a few dozen entries and a push per acquisition, against a
//! lock acquisition that already costs an atomic.
//!
//! # Why the rank is a field and not a const generic
//!
//! It was a const generic, which put the rank in the type beside the field's
//! declaration and cost no memory at all. A collection holding several spaces
//! cannot do that, because every space's index guard sits at the same place
//! in the order relative to the collection's own locks and at a different
//! place relative to the other spaces', so the rank has to carry the space's
//! position, which is decided when the collection is built. The field is
//! compiled out of a release build, so the memory cost is a debug build's
//! alone.
//!
//! # Why a field added later cannot bypass it
//!
//! `clippy.toml` disallows `std::sync::RwLock` and `std::sync::Mutex` by name,
//! and the lint gate runs `-D warnings`, so a bare lock anywhere in the crate
//! fails the build. The two modules that legitimately hold one, `pq` and
//! `graph`, carry a module level allow that says why. This file carries one
//! because it is where the wrapping happens.
//!
//! # What it tracks
//!
//! Guards currently held, on the thread that holds them. Two sequential
//! acquisitions are not a nested hold, so a read taken inside a match guard and
//! again in the arm is two acquisitions rather than one, because a match guard
//! is its own temporary scope.
//!
//! The held set is per thread, so a guard held across a rayon fork is invisible
//! to the workers, which is correct: the rule the order encodes is about one
//! thread's own acquisitions.

#![allow(clippy::disallowed_types)]

use std::sync::{LockResult, Mutex, PoisonError, RwLock};

/// The declared acquisition order, as a rank per lock.
///
/// **Ascending is earlier.** A thread may take a lock only when every lock it
/// already holds ranks strictly below it, so the numbers here are the order
/// `Collection` documents in prose, written down once in a form the
/// build can check.
///
/// The prose order is
///
/// ```text
/// id_map < rev_map < [each space's index guard, then its codes guard,
///                     in the order the spaces were declared]
///        < vector_metadata < columns < training_ids < metadata
///        < id_counter < vector_count
/// ```
///
/// `writers` sits above all of them because the mutating Python entry points
/// take it before any guard and no internal helper takes it at all.
///
/// The spaces sit in one block between the record set and the storage maps,
/// each at a stride of two ranks, so a search over several spaces takes every
/// space's guards after the reverse map and before the metadata, which is the
/// shape a search over one space already has. A space's position in the block
/// is its position in the collection's declaration, and two spaces' guards are
/// therefore ordered against each other, which is what lets the registry see
/// an inversion between them.
///
/// Each space's `rerank_calibration` and `training_completed_at`, and the
/// collection's `created_at`, are the leaves. The prose says they are never
/// held together with any other guard, which is stronger than a rank can
/// express, so they take the bottom ranks: anything may be held while one of
/// them is taken, and none of them may be held while anything else is. That is
/// the weaker half of the claim, and it is the half a rank can state without
/// inventing a rule the code has not agreed to.
pub mod order {
    /// The mutation lock, taken by a Python entry point before any guard.
    pub const WRITERS: u8 = 0;
    pub const ID_MAP: u8 = 1;
    pub const REV_MAP: u8 = 2;

    /// Spaces a collection may hold. Each takes two ranks in the block below
    /// and two among the leaves.
    pub const MAX_SPACES: usize = 4;
    const SPACE_BASE: u8 = 3;
    const SPACE_STRIDE: u8 = 2;

    /// The guard over a space's index, being the graph of a dense space and
    /// the postings of a sparse one, for the space at `space` in the
    /// declaration.
    pub const fn space_index(space: usize) -> u8 {
        SPACE_BASE + (space as u8) * SPACE_STRIDE
    }

    /// The guard over a dense space's code store, taken after its index.
    pub const fn space_codes(space: usize) -> u8 {
        space_index(space) + 1
    }

    const AFTER_SPACES: u8 = SPACE_BASE + (MAX_SPACES as u8) * SPACE_STRIDE;
    pub const VECTOR_METADATA: u8 = AFTER_SPACES;
    pub const COLUMNS: u8 = AFTER_SPACES + 1;
    pub const TRAINING_IDS: u8 = AFTER_SPACES + 2;
    pub const METADATA: u8 = AFTER_SPACES + 3;
    pub const ID_COUNTER: u8 = AFTER_SPACES + 4;
    pub const VECTOR_COUNT: u8 = AFTER_SPACES + 5;

    const LEAF_BASE: u8 = AFTER_SPACES + 6;

    /// A leaf. The rerank calibration of the dense space at `space`.
    pub const fn space_calibration(space: usize) -> u8 {
        LEAF_BASE + (space as u8) * 2
    }

    /// A leaf. When the dense space at `space` fitted its codebook.
    pub const fn space_trained_at(space: usize) -> u8 {
        space_calibration(space) + 1
    }

    const AFTER_LEAVES: u8 = LEAF_BASE + (MAX_SPACES as u8) * 2;
    /// A leaf.
    pub const CREATED_AT: u8 = AFTER_LEAVES;
    /// A leaf, taken by `generate_id` alone and held across nothing.
    pub const GENERATED_IDS: u8 = AFTER_LEAVES + 1;

    /// The first space's four ranks, by the names the collection's
    /// documentation uses for them.
    pub const HNSW: u8 = space_index(0);
    pub const PQ_CODES: u8 = space_codes(0);
    pub const RERANK_CALIBRATION: u8 = space_calibration(0);
    pub const TRAINING_COMPLETED_AT: u8 = space_trained_at(0);

    /// The largest rank declared above.
    ///
    /// `every_declared_rank_is_named` walks up to this and checks that one past
    /// it is unnamed, so adding a rank without naming it fails there rather than
    /// silently reporting the fallback.
    ///
    /// Compiled under `cfg(test)` alone, because the test is its only reader and
    /// a release build otherwise warns that it is never used.
    #[cfg(test)]
    pub const HIGHEST: u8 = GENERATED_IDS;

    /// Which space a rank in the space block or the leaf block belongs to,
    /// and which of its guards it is. `None` for a collection rank.
    #[cfg(debug_assertions)]
    pub(super) fn space_of(rank: u8) -> Option<(usize, &'static str)> {
        if (SPACE_BASE..AFTER_SPACES).contains(&rank) {
            let offset = rank - SPACE_BASE;
            let which = if offset.is_multiple_of(SPACE_STRIDE) {
                "index"
            } else {
                "codes"
            };
            return Some(((offset / SPACE_STRIDE) as usize, which));
        }
        if (LEAF_BASE..AFTER_LEAVES).contains(&rank) {
            let offset = rank - LEAF_BASE;
            let which = if offset.is_multiple_of(2) {
                "rerank_calibration"
            } else {
                "training_completed_at"
            };
            return Some(((offset / 2) as usize, which));
        }
        None
    }
}

/// How a rank names itself in an assertion a developer reads.
///
/// A `const` cannot carry a name, so this is the one place the numbers and the
/// field names are paired. A rank with no name here is a rank nobody declared,
/// which is why the fallback says so rather than printing the number alone.
#[cfg(debug_assertions)]
fn name_of(rank: u8) -> String {
    let fixed = match rank {
        order::WRITERS => Some("writers"),
        order::ID_MAP => Some("id_map"),
        order::REV_MAP => Some("rev_map"),
        order::VECTOR_METADATA => Some("vector_metadata"),
        order::COLUMNS => Some("columns"),
        order::TRAINING_IDS => Some("training_ids"),
        order::METADATA => Some("metadata"),
        order::ID_COUNTER => Some("id_counter"),
        order::VECTOR_COUNT => Some("vector_count"),
        order::CREATED_AT => Some("created_at"),
        order::GENERATED_IDS => Some("generated_ids"),
        _ => None,
    };
    if let Some(name) = fixed {
        return name.to_string();
    }
    match order::space_of(rank) {
        Some((space, which)) => format!("space {}'s {} guard", space, which),
        None => "a lock with no declared place in the order".to_string(),
    }
}

// ============================================================================
// THE REGISTRY
// ============================================================================

// What this thread holds, innermost last. A `Vec` rather than a bitset because
// the assertion has to name what is already held, and because a few dozen
// entries scanned linearly is cheaper than anything cleverer at this size.
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
fn held_now() -> Vec<String> {
    HELD.try_with(|held| held.borrow().iter().map(|&r| name_of(r)).collect())
        .unwrap_or_default()
}

// ============================================================================
// THE LOCKS
// ============================================================================

/// An `RwLock` that knows its place in the declared order.
///
/// The rank is given at construction, beside the field's initialiser, and it
/// exists only in a build that checks it.
#[cfg_attr(not(debug_assertions), repr(transparent))]
pub struct RwLockAt<T> {
    inner: RwLock<T>,
    #[cfg(debug_assertions)]
    rank: u8,
}

impl<T> RwLockAt<T> {
    pub const fn new(rank: u8, value: T) -> Self {
        #[cfg(not(debug_assertions))]
        let _ = rank;
        RwLockAt {
            inner: RwLock::new(value),
            #[cfg(debug_assertions)]
            rank,
        }
    }

    pub fn read(&self) -> LockResult<ReadGuard<'_, T>> {
        #[cfg(debug_assertions)]
        enter(self.rank);
        match self.inner.read() {
            Ok(guard) => Ok(ReadGuard {
                inner: guard,
                #[cfg(debug_assertions)]
                rank: self.rank,
            }),
            Err(poison) => Err(PoisonError::new(ReadGuard {
                inner: poison.into_inner(),
                #[cfg(debug_assertions)]
                rank: self.rank,
            })),
        }
    }

    pub fn write(&self) -> LockResult<WriteGuard<'_, T>> {
        #[cfg(debug_assertions)]
        enter(self.rank);
        match self.inner.write() {
            Ok(guard) => Ok(WriteGuard {
                inner: guard,
                #[cfg(debug_assertions)]
                rank: self.rank,
            }),
            Err(poison) => Err(PoisonError::new(WriteGuard {
                inner: poison.into_inner(),
                #[cfg(debug_assertions)]
                rank: self.rank,
            })),
        }
    }
}

/// A `Mutex` that knows its place in the declared order.
#[cfg_attr(not(debug_assertions), repr(transparent))]
pub struct MutexAt<T> {
    inner: Mutex<T>,
    #[cfg(debug_assertions)]
    rank: u8,
}

impl<T> MutexAt<T> {
    pub const fn new(rank: u8, value: T) -> Self {
        #[cfg(not(debug_assertions))]
        let _ = rank;
        MutexAt {
            inner: Mutex::new(value),
            #[cfg(debug_assertions)]
            rank,
        }
    }

    pub fn lock(&self) -> LockResult<Locked<'_, T>> {
        #[cfg(debug_assertions)]
        enter(self.rank);
        match self.inner.lock() {
            Ok(guard) => Ok(Locked {
                inner: guard,
                #[cfg(debug_assertions)]
                rank: self.rank,
            }),
            Err(poison) => Err(PoisonError::new(Locked {
                inner: poison.into_inner(),
                #[cfg(debug_assertions)]
                rank: self.rank,
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
        #[cfg_attr(not(debug_assertions), repr(transparent))]
        pub struct $name<'a, T> {
            inner: std::sync::$inner<'a, T>,
            #[cfg(debug_assertions)]
            rank: u8,
        }

        impl<T> std::ops::Deref for $name<'_, T> {
            type Target = T;
            #[inline]
            fn deref(&self) -> &T {
                &self.inner
            }
        }

        #[cfg(debug_assertions)]
        impl<T> Drop for $name<'_, T> {
            fn drop(&mut self) {
                leave(self.rank);
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
        impl<T> std::ops::DerefMut for $name<'_, T> {
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

    /// In release the wrapper is the standard lock's size, so the rank costs
    /// no memory. In debug the rank is one byte, and what it costs is that
    /// byte and its padding.
    #[test]
    fn a_tracked_lock_is_the_size_of_the_lock_it_wraps() {
        let lock_extra =
            std::mem::size_of::<RwLockAt<Vec<u8>>>() - std::mem::size_of::<RwLock<Vec<u8>>>();
        let mutex_extra = std::mem::size_of::<MutexAt<()>>() - std::mem::size_of::<Mutex<()>>();
        let read_extra = std::mem::size_of::<ReadGuard<'_, Vec<u8>>>()
            - std::mem::size_of::<RwLockReadGuard<'_, Vec<u8>>>();
        let write_extra = std::mem::size_of::<WriteGuard<'_, Vec<u8>>>()
            - std::mem::size_of::<RwLockWriteGuard<'_, Vec<u8>>>();
        let extras = [lock_extra, mutex_extra, read_extra, write_extra];
        if cfg!(debug_assertions) {
            assert!(
                extras.iter().all(|&extra| (1..=8).contains(&extra)),
                "debug adds one byte and its padding, and added {:?}",
                extras
            );
        } else {
            assert_eq!(extras, [0, 0, 0, 0]);
        }
    }

    /// A second read on one thread is refused, which is the whole point.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "taking the same guard twice on one thread")]
    fn a_second_read_on_one_thread_is_refused() {
        let lock: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 7);
        let _first = lock.read().unwrap();
        let _second = lock.read().unwrap();
    }

    /// And the same for a write under a read.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "taking the same guard twice on one thread")]
    fn a_write_under_a_read_of_the_same_lock_is_refused() {
        let lock: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 7);
        let _first = lock.read().unwrap();
        let _second = lock.write().unwrap();
    }

    /// Two locks taken out of the declared order are refused.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "inverts the declared lock order")]
    fn taking_a_lower_lock_under_a_higher_one_is_refused() {
        let columns: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 1);
        let id_map: RwLockAt<u32> = RwLockAt::new(order::ID_MAP, 2);
        let _held = columns.read().unwrap();
        let _inverted = id_map.read().unwrap();
    }

    /// Two spaces' index guards are ordered by the spaces' declaration, so
    /// taking the second space's before the first's is an inversion the
    /// registry sees.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "inverts the declared lock order")]
    fn taking_the_second_space_before_the_first_is_refused() {
        let first: RwLockAt<u32> = RwLockAt::new(order::space_index(0), 1);
        let second: RwLockAt<u32> = RwLockAt::new(order::space_index(1), 2);
        let _held = second.read().unwrap();
        let _inverted = first.read().unwrap();
    }

    /// Two spaces' guards at the same place in their own blocks are distinct
    /// ranks, so a search over both takes both without tripping the
    /// recursion check, and both sit between the record set and the storage
    /// maps.
    #[test]
    #[cfg(debug_assertions)]
    fn two_spaces_in_declaration_order_are_allowed_between_the_maps() {
        let rev_map: RwLockAt<u32> = RwLockAt::new(order::REV_MAP, 0);
        let first: RwLockAt<u32> = RwLockAt::new(order::space_index(0), 1);
        let first_codes: RwLockAt<u32> = RwLockAt::new(order::space_codes(0), 1);
        let second: RwLockAt<u32> = RwLockAt::new(order::space_index(1), 2);
        let vector_metadata: RwLockAt<u32> = RwLockAt::new(order::VECTOR_METADATA, 3);
        {
            let _a = rev_map.read().unwrap();
            let _b = first.read().unwrap();
            let _c = first_codes.read().unwrap();
            let _d = second.read().unwrap();
            let _e = vector_metadata.read().unwrap();
            assert_eq!(
                held_now(),
                vec![
                    "rev_map",
                    "space 0's index guard",
                    "space 0's codes guard",
                    "space 1's index guard",
                    "vector_metadata"
                ]
            );
        }
        assert!(held_now().is_empty());
    }

    /// Every space's ranks are distinct, in declaration order, and inside
    /// the block the collection's own ranks leave for them.
    #[test]
    fn the_space_ranks_are_distinct_and_sit_in_their_block() {
        let mut seen = Vec::new();
        for space in 0..order::MAX_SPACES {
            for rank in [
                order::space_index(space),
                order::space_codes(space),
                order::space_calibration(space),
                order::space_trained_at(space),
            ] {
                assert!(!seen.contains(&rank), "rank {} declared twice", rank);
                seen.push(rank);
            }
            assert!(order::space_index(space) > order::REV_MAP);
            assert!(order::space_codes(space) < order::VECTOR_METADATA);
            assert!(order::space_calibration(space) > order::VECTOR_COUNT);
            assert!(order::space_trained_at(space) < order::CREATED_AT);
            if space > 0 {
                assert!(order::space_index(space) > order::space_codes(space - 1));
            }
        }
        assert_eq!(order::HNSW, order::space_index(0));
        assert_eq!(order::PQ_CODES, order::space_codes(0));
    }

    /// Two locks in the declared order are allowed, and both are released.
    #[test]
    #[cfg(debug_assertions)]
    fn the_declared_order_is_allowed_and_the_set_empties() {
        let id_map: RwLockAt<u32> = RwLockAt::new(order::ID_MAP, 1);
        let columns: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 2);
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
        let id_map: RwLockAt<u32> = RwLockAt::new(order::ID_MAP, 1);
        let rev_map: RwLockAt<u32> = RwLockAt::new(order::REV_MAP, 2);
        let columns: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 3);

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
        let lock: RwLockAt<u32> = RwLockAt::new(order::COLUMNS, 9);
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
        for rank in 0..=order::HIGHEST {
            assert_ne!(
                name_of(rank),
                "a lock with no declared place in the order",
                "rank {} has no name",
                rank
            );
        }
        assert_eq!(
            name_of(order::HIGHEST + 1),
            "a lock with no declared place in the order"
        );
    }
}
