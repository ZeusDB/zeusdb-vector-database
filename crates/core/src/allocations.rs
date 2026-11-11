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

// SAFETY: each method hands its arguments to `System` unchanged, or returns
// null from an allocation, which `GlobalAlloc` allows for a request it cannot
// meet.
unsafe impl GlobalAlloc for Bounded {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            // SAFETY: the caller's contract for `alloc`, passed on unchanged.
            unsafe { System.alloc(layout) }
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if admit(layout.size()) {
            // SAFETY: the caller's contract for `alloc_zeroed`, passed on
            // unchanged.
            unsafe { System.alloc_zeroed(layout) }
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if admit(new_size) {
            // SAFETY: the caller's contract for `realloc`, passed on
            // unchanged. A null return leaves `ptr` allocated, as `realloc`
            // requires.
            unsafe { System.realloc(ptr, layout, new_size) }
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
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
