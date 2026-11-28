//! The global allocator of this crate's own tests, which every other test
//! build of the workspace declares as its own, from the `test-support`
//! feature, since a program holds one global allocator.
//!
//! Every request for memory is held to [`CEILING`], and a request made on a
//! thread inside [`bounded`] is held to the bound that call names as well. A
//! request past either is refused before anything is reserved: the allocator
//! writes the request and the bound to the process's standard error and
//! returns no memory, so the process ends with `memory allocation of <n>
//! bytes failed` whatever the machine could have granted. Every other request
//! goes to [`System`] unchanged.
//!
//! A fuzzer reads through [`bounded`] to hold the rule that no allocation is
//! sized from a field the reader has not yet checked against its input, and
//! [`CEILING`] holds every test to requests no larger than it. A reservation
//! that nothing writes to costs only commit charge, so a machine with room
//! for one grants it and either rule is broken with no sign.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;

/// The largest single request any test may make, 1 GiB.
pub const CEILING: usize = 1 << 30;

thread_local! {
    /// The bound on this thread's requests, `usize::MAX` outside [`bounded`].
    static BOUND: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[cfg(test)]
thread_local! {
    /// Whether this thread's requests are counted, which they are inside
    /// [`counted`].
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Bytes this thread has been granted less bytes it has given back since
    /// [`counted`] began, and the largest that figure has been.
    static HELD: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

/// [`System`], with every request held to [`CEILING`] and each thread's
/// requests to that thread's bound.
pub struct Bounded;

#[cfg(test)]
#[global_allocator]
static ALLOCATOR: Bounded = Bounded;

/// Whether a request for `size` bytes is within [`CEILING`] and this
/// thread's bound. A request past them lifts this thread's bound before its
/// words are written, so that writing them allocates as an unbounded request
/// does.
fn admit(size: usize) -> bool {
    let bound = BOUND.try_with(Cell::get).unwrap_or(usize::MAX).min(CEILING);
    if size <= bound {
        return true;
    }
    let _ = BOUND.try_with(|held| held.set(usize::MAX));
    let _ = writeln!(
        std::io::stderr(),
        "a request for {size} bytes in one allocation was refused, and the bound in force is \
         {bound} bytes"
    );
    false
}

/// Count a block of `granted` bytes that took the place of one of `released`
/// bytes on this thread, where the thread is inside [`counted`]. A request
/// the allocator did not meet, which hands back null, moves nothing. The
/// count is this crate's own tests', and every other build compiles none.
#[inline(always)]
#[cfg_attr(not(test), allow(unused_variables))]
fn count(block: *mut u8, granted: usize, released: usize) {
    #[cfg(test)]
    {
        if block.is_null() || !COUNTING.try_with(Cell::get).unwrap_or(false) {
            return;
        }
        let _ = HELD.try_with(|held| {
            let now = held.get() + granted as isize - released as isize;
            held.set(now);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
        });
    }
}

// SAFETY: each method hands its arguments to `System` unchanged, or returns
// null from an allocation, which `GlobalAlloc` allows for a request it cannot
// meet.
unsafe impl GlobalAlloc for Bounded {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            // SAFETY: the caller's contract for `alloc`, passed on unchanged.
            let block = unsafe { System.alloc(layout) };
            count(block, layout.size(), 0);
            block
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            // SAFETY: the caller's contract for `alloc_zeroed`, passed on
            // unchanged.
            let block = unsafe { System.alloc_zeroed(layout) };
            count(block, layout.size(), 0);
            block
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if admit(new_size) {
            // SAFETY: the caller's contract for `realloc`, passed on
            // unchanged. A null return leaves `ptr` allocated, as `realloc`
            // requires.
            let block = unsafe { System.realloc(ptr, layout, new_size) };
            count(block, new_size, layout.size());
            block
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(ptr, 0, layout.size());
        // SAFETY: the caller's contract for `dealloc`, passed on unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// The bound in force before a [`bounded`] call, put back when the call
/// ends, by return or by unwind.
struct Restore(usize);

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = BOUND.try_with(|held| held.set(self.0));
    }
}

/// `read`, run with every allocation this thread asks for held to `bound`
/// bytes as well as to [`CEILING`]. A request past either ends the process,
/// as the module says.
pub fn bounded<R>(bound: usize, read: impl FnOnce() -> R) -> R {
    let _restore = Restore(BOUND.with(|held| held.replace(bound)));
    read()
}

/// What [`counted`] measured of one call, in bytes this thread was granted
/// less bytes it gave back, from the level the thread held when the call
/// began.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Held {
    /// The level when the call ended. Below zero where the call gave back
    /// more than it was granted.
    pub(crate) end: isize,
    /// The highest level during the call, at least zero.
    pub(crate) peak: isize,
}

/// The state of the count before a [`counted`] call, put back when the call
/// ends, by return or by unwind.
#[cfg(test)]
struct Uncounted(bool, isize, isize);

#[cfg(test)]
impl Drop for Uncounted {
    fn drop(&mut self) {
        let _ = COUNTING.try_with(|counting| counting.set(self.0));
        let _ = HELD.try_with(|held| held.set(self.1));
        let _ = PEAK.try_with(|peak| peak.set(self.2));
    }
}

/// `body`, run with every block this thread is granted and gives back
/// counted, and what the count read. Another thread's requests are not
/// counted, so the figures are the call's own whatever else the process
/// runs. A call is not made inside another.
#[cfg(test)]
pub(crate) fn counted<R>(body: impl FnOnce() -> R) -> (R, Held) {
    assert!(
        !COUNTING.with(Cell::get),
        "counted is not called inside counted"
    );
    let _restore = Uncounted(
        COUNTING.with(|counting| counting.replace(true)),
        HELD.with(|held| held.replace(0)),
        PEAK.with(|peak| peak.replace(0)),
    );
    let out = body();
    let held = Held {
        end: HELD.with(Cell::get),
        peak: PEAK.with(Cell::get),
    };
    (out, held)
}

/// The bound a fuzzer reads a mutated input of `bytes` under: sixteen times
/// them and 2 MiB more, the 2 MiB being room for the fixed buffers a reader
/// takes whatever its input, the largest of them the dump readers' 1 MiB. A
/// read that sizes its allocations from what it has checked against the
/// input stays below it at every corpus a fuzzer here reads, and an
/// allocation sized from a count of 2^31 records or more, as the mutators
/// write, is far above it.
pub fn read_bound(bytes: usize) -> usize {
    bytes.saturating_mul(16).saturating_add(2 << 20)
}
