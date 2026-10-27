//! A point a test panics at.
//!
//! A panic inside a call on an index stops the index, and every later call
//! is refused; see `Collection::run` in the index crate. The tests that hold
//! the index to that rule panic at a named point inside a mutation, where the
//! mutation has written part of a change and not the rest, and this is the
//! hook the panic runs from.
//!
//! # What it costs a build that ships
//!
//! Nothing. The check and the names exist under the `test-support` feature
//! and in this crate's own tests, and every other build compiles
//! [`panic_at`] to an empty function, so the wheel carries neither. The index
//! crate turns the feature on through its dev-dependency on this crate,
//! which cargo applies to that crate's test targets alone.
//!
//! # How a point is armed
//!
//! Per thread, by the point's variant, and for one firing. A test arms the
//! thread that makes the call, so the tests running beside it on other
//! threads reach the same point unharmed, and a point disarms itself as it
//! fires, so the calls after it run as they would.

/// Every point a test can panic at, each named for the step it follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanicPoint {
    /// Inside an insertion's second phase, under its four write guards,
    /// after the record is named in the id store and before the graph takes
    /// it.
    InsertAfterName,
    /// After a record's dense half, its metadata and its columns are
    /// installed, and before its sparse half is, with no guard held.
    InstallBeforeSparse,
    /// After an `add` has committed and removed the records an overwrite
    /// replaces, and before the first record of its segment is installed.
    AddBeforeInstalls,
    /// After a training has installed the codebook, the stamp and the
    /// calibration and cleared the collected ids, and before the graph is
    /// rebuilt over the codes.
    TrainingBeforeRebuild,
}

impl PanicPoint {
    /// What the panic says, being the point's name.
    pub fn name(self) -> &'static str {
        match self {
            PanicPoint::InsertAfterName => "insert:after_name",
            PanicPoint::InstallBeforeSparse => "install:before_sparse",
            PanicPoint::AddBeforeInstalls => "add:before_installs",
            PanicPoint::TrainingBeforeRebuild => "training:before_rebuild",
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
mod live {
    use super::PanicPoint;
    use std::cell::Cell;

    thread_local! {
        /// The point this thread panics at next, where one is armed.
        static ARMED: Cell<Option<PanicPoint>> = const { Cell::new(None) };
    }

    /// Panic the next time this thread reaches `point`, once.
    pub fn arm(point: PanicPoint) {
        ARMED.with(|armed| armed.set(Some(point)));
    }

    /// Stop panicking anywhere on this thread.
    pub fn disarm() {
        ARMED.with(|armed| armed.set(None));
    }

    /// Panic where this thread is armed at `point`, disarming it first, and
    /// do nothing where it is not.
    pub fn at(point: PanicPoint) {
        let fires = ARMED.with(|armed| {
            if armed.get() == Some(point) {
                armed.set(None);
                true
            } else {
                false
            }
        });
        if fires {
            panic!("a panic injected at {}", point.name());
        }
    }
}

#[cfg(not(any(test, feature = "test-support")))]
mod live {
    use super::PanicPoint;

    #[inline(always)]
    pub fn at(_point: PanicPoint) {}
}

pub use live::at;
#[cfg(any(test, feature = "test-support"))]
pub use live::{arm, disarm};

#[cfg(test)]
mod tests {
    use super::*;

    /// A point armed on this thread fires once, at that point alone, with
    /// its name in the panic's words, and a disarmed thread passes every
    /// point.
    #[test]
    fn an_armed_point_fires_once_and_names_itself() {
        disarm();
        at(PanicPoint::InsertAfterName);
        arm(PanicPoint::AddBeforeInstalls);
        at(PanicPoint::InsertAfterName);
        let fired = std::panic::catch_unwind(|| at(PanicPoint::AddBeforeInstalls));
        let words = fired
            .expect_err("the armed point fires")
            .downcast::<String>()
            .expect("the panic carries its words");
        assert_eq!(*words, "a panic injected at add:before_installs");
        at(PanicPoint::AddBeforeInstalls);
        arm(PanicPoint::TrainingBeforeRebuild);
        disarm();
        at(PanicPoint::TrainingBeforeRebuild);
    }
}
