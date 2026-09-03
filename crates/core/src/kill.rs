//! A point a test aborts the process at.
//!
//! Recovery from a crash is proved by killing a real process in the middle
//! of a real write. An injected error is a different thing: it unwinds, its
//! destructors run, buffered bytes are flushed and files are closed, and a
//! killed process does none of that. So the tests that hold the journal to
//! its contract abort at a named point instead, and this is the hook the
//! abort runs from.
//!
//! # What it costs a build that ships
//!
//! Nothing. Everything below is compiled away when `debug_assertions` are
//! off, which the release profile leaves them, so the wheel carries neither
//! the check nor the names. The dev profile the tests run under and the
//! `checked` profile leave them on, which is where the matrix runs. See the
//! `[profile]` tables in the workspace manifest.
//!
//! # Why the points are an enum
//!
//! Every point is declared once in [`KillPoint`], and a process is armed by
//! the discriminant rather than by a name, so the check on a live path is one
//! relaxed atomic load and an integer comparison and takes no lock at all. A
//! point added to a mutation path has to be added here first, which is what
//! keeps the set of them in one place.
//!
//! # How a point is used
//!
//! A parent process spawns a child armed at one point. The child runs its
//! script, reaches the point, writes the marker file so the parent knows the
//! point was reached rather than skipped, and aborts. The parent then opens
//! the directory the child left and holds it to what the child had
//! acknowledged.

/// Every point a process can be aborted at.
///
/// The six inside an `add` and the eleven inside a checkpoint, each named for
/// the step it follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum KillPoint {
    /// Before the first record of a batch is handed to the sink.
    AddBeforeFirstAppend = 0,
    /// With half a record's bytes written and no more.
    AddMidAppend,
    /// After the second record of a batch is handed over.
    AddAfterSecondAppend,
    /// After every record of a segment is handed over and before the commit.
    AddAfterAppendBeforeCommit,
    /// After the commit and before the first record is installed.
    AddAfterCommitBeforeApply,
    /// After every record of the call is installed.
    AddAfterApply,
    /// After the journal is synced and the sequence recorded, before the save.
    CheckpointBeforeSave,
    /// After the save has written every artefact but the graph dump.
    SaveAfterArtefacts,
    /// After the graph dump.
    SaveAfterDump,
    /// After the manifest, which is the last file the save writes.
    SaveAfterManifest,
    /// Between the two renames, with the whole index beside the target and
    /// nothing at it.
    SaveBetweenRenames,
    /// After the staged directory is in place.
    SaveAfterCommit,
    /// After the save and before the journal is truncated.
    CheckpointAfterSaveBeforeTruncate,
    /// Before the truncation's first step.
    TruncateBefore,
    /// After the body is cut and synced, before the header is restated.
    TruncateAfterSetLen,
    /// After the header is restated and synced.
    TruncateAfterHeader,
    /// After the whole checkpoint.
    CheckpointAfterTruncate,
}

impl KillPoint {
    /// Every point, in the order a run reaches them.
    pub const ALL: &'static [KillPoint] = &[
        KillPoint::AddBeforeFirstAppend,
        KillPoint::AddMidAppend,
        KillPoint::AddAfterSecondAppend,
        KillPoint::AddAfterAppendBeforeCommit,
        KillPoint::AddAfterCommitBeforeApply,
        KillPoint::AddAfterApply,
        KillPoint::CheckpointBeforeSave,
        KillPoint::SaveAfterArtefacts,
        KillPoint::SaveAfterDump,
        KillPoint::SaveAfterManifest,
        KillPoint::SaveBetweenRenames,
        KillPoint::SaveAfterCommit,
        KillPoint::CheckpointAfterSaveBeforeTruncate,
        KillPoint::TruncateBefore,
        KillPoint::TruncateAfterSetLen,
        KillPoint::TruncateAfterHeader,
        KillPoint::CheckpointAfterTruncate,
    ];

    /// What a case names the point, which is what a directory and a marker
    /// are named after.
    pub fn name(self) -> &'static str {
        match self {
            KillPoint::AddBeforeFirstAppend => "add:before_first_append",
            KillPoint::AddMidAppend => "add:mid_append",
            KillPoint::AddAfterSecondAppend => "add:after_second_append",
            KillPoint::AddAfterAppendBeforeCommit => "add:after_append_before_commit",
            KillPoint::AddAfterCommitBeforeApply => "add:after_commit_before_apply",
            KillPoint::AddAfterApply => "add:after_apply",
            KillPoint::CheckpointBeforeSave => "checkpoint:before_save",
            KillPoint::SaveAfterArtefacts => "save:after_artefacts",
            KillPoint::SaveAfterDump => "save:after_dump",
            KillPoint::SaveAfterManifest => "save:after_manifest",
            KillPoint::SaveBetweenRenames => "save:between_renames",
            KillPoint::SaveAfterCommit => "save:after_commit",
            KillPoint::CheckpointAfterSaveBeforeTruncate => "checkpoint:after_save_before_truncate",
            KillPoint::TruncateBefore => "truncate:before",
            KillPoint::TruncateAfterSetLen => "truncate:after_setlen",
            KillPoint::TruncateAfterHeader => "truncate:after_header",
            KillPoint::CheckpointAfterTruncate => "checkpoint:after_truncate",
        }
    }

    /// The point a case names, where it names one.
    pub fn from_name(name: &str) -> Option<KillPoint> {
        KillPoint::ALL
            .iter()
            .copied()
            .find(|point| point.name() == name)
    }
}

#[cfg(debug_assertions)]
mod live {
    use super::KillPoint;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    /// The point this process aborts at, as a discriminant, or [`NONE`].
    static ARMED: AtomicUsize = AtomicUsize::new(NONE);

    /// No point, which is every process but a crash test's child.
    const NONE: usize = usize::MAX;

    /// The file a kill writes before it aborts, so the parent knows the point
    /// was reached. Set once, because a process is spawned for one case.
    static MARKER: OnceLock<PathBuf> = OnceLock::new();

    /// Abort the next time this process reaches `point`, writing `marker`
    /// before it does.
    pub fn arm(point: KillPoint, marker: &Path) {
        let _ = MARKER.set(marker.to_path_buf());
        ARMED.store(point as usize, Ordering::Relaxed);
    }

    /// Stop aborting anywhere.
    pub fn disarm() {
        ARMED.store(NONE, Ordering::Relaxed);
    }

    /// Whether this process is armed at `point`, for a caller that has to do
    /// something before the abort, such as write half a record.
    pub fn armed_at(point: KillPoint) -> bool {
        ARMED.load(Ordering::Relaxed) == point as usize
    }

    /// Abort where this process is armed at `point`, and do nothing where it
    /// is not.
    ///
    /// `std::process::abort` rather than `panic!` or `exit`, so no destructor
    /// runs, no buffer is flushed and no file is closed, which is what a
    /// killed process leaves.
    pub fn at(point: KillPoint) {
        if !armed_at(point) {
            return;
        }
        if let Some(marker) = MARKER.get() {
            if let Ok(mut file) = std::fs::File::create(marker) {
                use std::io::Write;
                let _ = file.write_all(point.name().as_bytes());
                let _ = file.sync_all();
            }
        }
        std::process::abort();
    }
}

#[cfg(not(debug_assertions))]
mod live {
    use super::KillPoint;
    use std::path::Path;

    #[inline(always)]
    pub fn arm(_point: KillPoint, _marker: &Path) {}

    #[inline(always)]
    pub fn disarm() {}

    #[inline(always)]
    pub fn armed_at(_point: KillPoint) -> bool {
        false
    }

    #[inline(always)]
    pub fn at(_point: KillPoint) {}
}

pub use live::{arm, armed_at, at, disarm};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_point_is_named_once_and_reads_back() {
        let mut names: Vec<&str> = KillPoint::ALL.iter().map(|p| p.name()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two points share a name");
        for (index, point) in KillPoint::ALL.iter().enumerate() {
            assert_eq!(*point as usize, index, "the table is in discriminant order");
            assert_eq!(KillPoint::from_name(point.name()), Some(*point));
        }
        assert_eq!(KillPoint::from_name("add:nowhere"), None);
    }

    #[test]
    fn a_process_that_is_not_armed_reaches_every_point() {
        // The whole table, on the thread this test runs on, with nothing
        // armed. Reaching one of them would abort the run rather than fail
        // it, so a pass here is the assertion.
        disarm();
        for point in KillPoint::ALL {
            assert!(!armed_at(*point));
            at(*point);
        }
    }
}
