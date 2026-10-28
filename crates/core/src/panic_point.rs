//! A point a test panics or pauses at.
//!
//! A panic inside a call on an index stops the index, and every later call
//! is refused; see `Collection::run` in the index crate. The tests that hold
//! the index to that rule panic at a named point inside a mutation, where the
//! mutation has written part of a change and not the rest, and this is the
//! hook the panic runs from.
//!
//! The checks of the interpreter rule pause a thread instead, at the same
//! points and at `pause_call`, which the binding reaches at the start of the
//! part of a call it runs with the interpreter released. The paused thread
//! waits until another thread releases it, so a release made by a thread that
//! holds the interpreter shows that the paused call did not hold it at the
//! pause. See `Pause`.
//!
//! # What it costs a build that ships
//!
//! Nothing. The checks, the pause and the arming exist under the
//! `pause-points` feature and in this crate's own tests, and every other
//! build compiles [`panic_at`] to an empty function, so the wheel carries none
//! of them. This crate's `test-support` feature turns `pause-points` on, and
//! the index crate turns that on through its dev-dependency on this crate,
//! which cargo applies to that crate's test targets alone. The binding's
//! `test-support` feature turns `pause-points` on and nothing else of this
//! crate.
//!
//! # How a point is armed
//!
//! Per thread, by the point's variant, and for one firing. A test arms the
//! thread that makes the call, so the tests running beside it on other
//! threads reach the same point unharmed, and a point disarms itself as it
//! fires, so the calls after it run as they would. A pause is armed the same
//! way, at one of these points or at the thread's next `pause_call`, and a
//! thread holds one armed panic and one armed pause at most.

/// Every point a test can panic or pause at, each named for the step it
/// follows.
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

#[cfg(any(test, feature = "pause-points"))]
impl PanicPoint {
    /// Every point, in the order the enum declares them.
    pub const ALL: [PanicPoint; 4] = [
        PanicPoint::InsertAfterName,
        PanicPoint::InstallBeforeSparse,
        PanicPoint::AddBeforeInstalls,
        PanicPoint::TrainingBeforeRebuild,
    ];

    /// The point whose name is `name`, where there is one.
    pub fn named(name: &str) -> Option<PanicPoint> {
        PanicPoint::ALL
            .into_iter()
            .find(|point| point.name() == name)
    }
}

#[cfg(any(test, feature = "pause-points"))]
mod live {
    use super::PanicPoint;
    use std::cell::{Cell, RefCell};
    use std::sync::{Arc, Condvar, MutexGuard, PoisonError};
    use std::time::{Duration, Instant};

    thread_local! {
        /// The point this thread panics at next, where one is armed.
        static ARMED: Cell<Option<PanicPoint>> = const { Cell::new(None) };
        /// Where this thread pauses next and the pause it waits on, where
        /// one is armed.
        static PAUSE_ARMED: RefCell<Option<(PauseAt, Arc<Pause>)>> =
            const { RefCell::new(None) };
    }

    /// Panic the next time this thread reaches `point`, once.
    pub fn arm(point: PanicPoint) {
        ARMED.with(|armed| armed.set(Some(point)));
    }

    /// Stop panicking anywhere on this thread.
    pub fn disarm() {
        ARMED.with(|armed| armed.set(None));
    }

    /// Pause where this thread is armed to pause at `point`, then panic
    /// where it is armed to panic at `point`, disarming each as it fires,
    /// and do nothing where it is armed for neither.
    pub fn at(point: PanicPoint) {
        if let Some(pause) = take_pause(PauseAt::Point(point)) {
            pause.hold(point.name());
        }
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

    /// Where an armed pause stops its thread.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PauseAt {
        /// The next `pause_call` the thread reaches, being the start of the
        /// part of a call the binding runs with the interpreter released.
        NextCall,
        /// The named point inside a mutation.
        Point(PanicPoint),
    }

    /// What a pause has seen. Every state after the first carries the name
    /// of the call or the point that reached the pause.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PauseState {
        /// No thread has reached it.
        Armed,
        /// A thread waits at it.
        Waiting(&'static str),
        /// A release let the waiting thread go.
        Released(&'static str),
        /// The limit passed with no release, and the thread went on.
        TimedOut(&'static str),
    }

    impl PauseState {
        /// The name of the call or the point that reached the pause, where
        /// a thread has reached it.
        pub fn reached(self) -> Option<&'static str> {
            match self {
                PauseState::Armed => None,
                PauseState::Waiting(name)
                | PauseState::Released(name)
                | PauseState::TimedOut(name) => Some(name),
            }
        }
    }

    /// A standard mutex rather than one from the index crate's rank
    /// registry, which this crate cannot name. It is a leaf: it is held to
    /// read or change a pause's state and never across another lock, and a
    /// wait on the pause's condition variable gives it up.
    #[allow(clippy::disallowed_types)]
    type StateSlot = std::sync::Mutex<PauseState>;

    /// A place a thread waits until another thread releases it, or until
    /// its limit passes.
    ///
    /// It holds the first thread that reaches it, and every later thread
    /// passes it. The held thread waits on the pause's own mutex and
    /// condition variable and on nothing else, so a release takes no lock
    /// but that mutex. Every wait here ends at a limit of one hour or less.
    #[derive(Debug)]
    pub struct Pause {
        limit: Duration,
        state: StateSlot,
        changed: Condvar,
    }

    impl Pause {
        /// The longest a pause holds a thread, and the longest a wait for
        /// one lasts. A longer limit is cut to it.
        pub const LONGEST: Duration = Duration::from_secs(3600);

        /// A pause no thread has reached, which holds the thread that
        /// reaches it for `limit` at most. A limit of zero records the call
        /// or the point that reached it and holds nothing.
        pub fn new(limit: Duration) -> Self {
            Pause {
                limit: limit.min(Pause::LONGEST),
                state: StateSlot::new(PauseState::Armed),
                changed: Condvar::new(),
            }
        }

        /// The longest it holds the thread that reaches it.
        pub fn limit(&self) -> Duration {
            self.limit
        }

        /// What it has seen so far.
        pub fn state(&self) -> PauseState {
            *self.lock()
        }

        /// Wait until a thread reaches the pause or `limit` passes, and give
        /// what the pause has seen then. A thread that reached it and went
        /// on before this call counts as having reached it.
        pub fn wait_reached(&self, limit: Duration) -> PauseState {
            let deadline = Instant::now() + limit.min(Pause::LONGEST);
            let mut state = self.lock();
            while *state == PauseState::Armed {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                state = self
                    .changed
                    .wait_timeout(state, deadline - now)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            *state
        }

        /// Let the waiting thread go, and say whether a thread was waiting.
        /// A release that finds no thread waiting changes nothing.
        pub fn release(&self) -> bool {
            let mut state = self.lock();
            match *state {
                PauseState::Waiting(name) => {
                    *state = PauseState::Released(name);
                    self.changed.notify_all();
                    true
                }
                _ => false,
            }
        }

        /// Hold this thread here as `name` until a release or the limit,
        /// where no thread has reached the pause before it, and pass where
        /// one has.
        fn hold(&self, name: &'static str) {
            let deadline = Instant::now() + self.limit;
            let mut state = self.lock();
            if *state != PauseState::Armed {
                return;
            }
            *state = PauseState::Waiting(name);
            self.changed.notify_all();
            while *state == PauseState::Waiting(name) {
                let now = Instant::now();
                if now >= deadline {
                    *state = PauseState::TimedOut(name);
                    self.changed.notify_all();
                    break;
                }
                state = self
                    .changed
                    .wait_timeout(state, deadline - now)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }

        /// The state, through a poisoned lock as well, since a test that
        /// panicked while it held the lock left the state whole.
        fn lock(&self) -> MutexGuard<'_, PauseState> {
            self.state.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    /// Pause this thread at `at` the next time it reaches it, once, on
    /// `pause`, in place of any pause armed on it before.
    pub fn pause_arm(at: PauseAt, pause: Arc<Pause>) {
        PAUSE_ARMED.with(|armed| *armed.borrow_mut() = Some((at, pause)));
    }

    /// Pause nowhere on this thread.
    pub fn pause_disarm() {
        PAUSE_ARMED.with(|armed| *armed.borrow_mut() = None);
    }

    /// The start of the part of `call` the binding runs with the interpreter
    /// released, with the call named as it names itself. Pause here where
    /// this thread is armed for its next call, disarming it first, and do
    /// nothing where it is not.
    pub fn pause_call(call: &'static str) {
        if let Some(pause) = take_pause(PauseAt::NextCall) {
            pause.hold(call);
        }
    }

    /// The pause this thread is armed with at `at`, taken off the thread,
    /// where the thread is armed at `at`.
    fn take_pause(at: PauseAt) -> Option<Arc<Pause>> {
        PAUSE_ARMED.with(|armed| {
            let mut armed = armed.borrow_mut();
            if armed.as_ref().is_some_and(|(armed_at, _)| *armed_at == at) {
                armed.take().map(|(_, pause)| pause)
            } else {
                None
            }
        })
    }

    /// Where this thread is armed to pause, where it is.
    #[cfg(test)]
    pub(super) fn pause_armed_at() -> Option<PauseAt> {
        PAUSE_ARMED.with(|armed| armed.borrow().as_ref().map(|(at, _)| *at))
    }
}

#[cfg(not(any(test, feature = "pause-points")))]
mod live {
    use super::PanicPoint;

    #[inline(always)]
    pub fn at(_point: PanicPoint) {}
}

pub use live::at;
#[cfg(any(test, feature = "pause-points"))]
pub use live::{arm, disarm, pause_arm, pause_call, pause_disarm, Pause, PauseAt, PauseState};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

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

    /// A pause holds the thread that reaches it until another thread
    /// releases it, and the release reports that it found the thread
    /// waiting at the call that reached it.
    #[test]
    fn a_pause_holds_its_thread_until_a_release_finds_it_waiting() {
        let limit = Duration::from_secs(30);
        let pause = Arc::new(Pause::new(limit));
        let went_on = Arc::new(AtomicBool::new(false));
        let caller = {
            let pause = Arc::clone(&pause);
            let went_on = Arc::clone(&went_on);
            std::thread::spawn(move || {
                pause_arm(PauseAt::NextCall, pause);
                pause_call("first()");
                went_on.store(true, Ordering::SeqCst);
            })
        };
        assert_eq!(pause.wait_reached(limit), PauseState::Waiting("first()"));
        assert!(
            !went_on.load(Ordering::SeqCst),
            "the thread went on from the pause before any release"
        );
        assert!(pause.release(), "the release finds the thread waiting");
        caller.join().expect("the released thread ends");
        assert!(went_on.load(Ordering::SeqCst));
        assert_eq!(pause.state(), PauseState::Released("first()"));
        assert!(!pause.release(), "a second release finds no thread waiting");
    }

    /// A pause that no thread releases lets its thread go once its limit
    /// has passed, and a release after that finds no thread waiting and
    /// changes nothing.
    #[test]
    fn a_pause_nobody_releases_lets_its_thread_go_at_its_limit() {
        let limit = Duration::from_millis(20);
        let pause = Arc::new(Pause::new(limit));
        pause_arm(PauseAt::NextCall, Arc::clone(&pause));
        let started = Instant::now();
        pause_call("alone()");
        assert!(
            started.elapsed() >= limit,
            "the thread went on before its limit"
        );
        assert_eq!(pause.state(), PauseState::TimedOut("alone()"));
        assert!(!pause.release(), "a release after the limit finds nothing");
        assert_eq!(pause.state(), PauseState::TimedOut("alone()"));
    }

    /// A pause fires once, at the first place of the kind it is armed for,
    /// and disarms its thread as it fires: one armed for the next call
    /// passes every point and fires at the first call, and one armed at a
    /// point passes every call and every other point. A disarmed thread
    /// passes everything.
    #[test]
    fn a_pause_fires_once_at_the_first_place_it_is_armed_for() {
        let next_call = Arc::new(Pause::new(Duration::ZERO));
        pause_arm(PauseAt::NextCall, Arc::clone(&next_call));
        at(PanicPoint::InsertAfterName);
        assert_eq!(next_call.state(), PauseState::Armed);
        pause_call("first()");
        assert_eq!(live::pause_armed_at(), None);
        pause_call("second()");
        assert_eq!(next_call.state(), PauseState::TimedOut("first()"));

        let point = Arc::new(Pause::new(Duration::ZERO));
        let armed_at = PauseAt::Point(PanicPoint::AddBeforeInstalls);
        pause_arm(armed_at, Arc::clone(&point));
        pause_call("call()");
        at(PanicPoint::InsertAfterName);
        assert_eq!(point.state(), PauseState::Armed);
        assert_eq!(live::pause_armed_at(), Some(armed_at));
        at(PanicPoint::AddBeforeInstalls);
        assert_eq!(live::pause_armed_at(), None);
        assert_eq!(point.state(), PauseState::TimedOut("add:before_installs"));

        let disarmed = Arc::new(Pause::new(Duration::ZERO));
        pause_arm(PauseAt::NextCall, Arc::clone(&disarmed));
        pause_disarm();
        pause_call("call()");
        at(PanicPoint::AddBeforeInstalls);
        assert_eq!(disarmed.state(), PauseState::Armed);
    }

    /// Every point is found by its own name, and no other name finds one.
    #[test]
    fn every_point_is_found_by_its_name() {
        for point in PanicPoint::ALL {
            assert_eq!(PanicPoint::named(point.name()), Some(point));
        }
        assert_eq!(PanicPoint::named("insert"), None);
    }
}
