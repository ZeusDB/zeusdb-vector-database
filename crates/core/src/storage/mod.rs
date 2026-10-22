//! Where a saved collection is kept.
//!
//! The save and load path, the graph dump and the journal reach storage
//! through the traits here and make no filesystem call of their own.
//! [`fs`] holds the one implementation, over the local filesystem, and every
//! filesystem call those paths make is made there.
//!
//! # The parts
//!
//! [`Dir`] is a directory of artefacts, each addressed by its name. A name
//! may carry a prefix, such as `spaces/<name>/`, and the implementation
//! creates whatever the prefix needs before it writes. An artefact is
//! written whole, or streamed through an [`ArtefactWriter`]. A whole one is
//! durable before [`Dir::write`] returns, and a streamed one before
//! [`ArtefactWriter::finish`] does.
//!
//! [`Storage`] is the place one collection is kept. It owns how a save
//! replaces the directory whole, being [`Storage::stage`] and
//! [`Staged::commit`], how a killed save is undone, being
//! [`Storage::recover`], where the journal beside the directory lives, and
//! the lock that keeps a second writer off the collection.
//!
//! [`Staged`] is the directory a save writes before the commit puts it in
//! place. Dropped without a commit, it is removed.
//!
//! [`JournalDir`] is the journal, being a directory of segments with the
//! `quarantine` folder a repair copies what it sets aside into, and
//! [`JournalFile`] is one segment, open to read and write.
//!
//! [`LockGuard`] is a lock on the collection, held until it is dropped.
//!
//! # Why the journal is not an artefact
//!
//! The directory is replaced whole at every checkpoint and the journal stays
//! open across that replacement, so the journal sits beside the directory
//! rather than in it. Its writes are appends at the end of its last segment
//! and a cut back to a length, which no artefact takes, and the last segment
//! is shared with the thread that syncs it on an interval. So it has traits
//! of its own, reached through [`Storage`].
//!
//! # Errors
//!
//! The whole-artefact operations, the staging, the commit and the recovery
//! return the engine's [`Error`], naming the artefact or the path, because
//! every caller reports them the same way. The rest return the I/O error and
//! the caller words it, so a caller can also tell a medium that refuses
//! writes from any other failure; see [`refuses_writes`].

mod fs;

pub use fs::{
    close_lock_files_in_forked_child, hold_lock_files_for_fork, release_lock_files_after_fork,
    FsDir, FsStorage, JOURNAL_SUFFIX, LEGACY_JOURNAL_SUFFIX, LOCK_SUFFIX, SEGMENT_SUFFIX,
};

use crate::error::Error;
use std::io::{self, Read, Write};
use std::path::Path;

/// A directory of artefacts, each addressed by its name.
pub trait Dir {
    /// Write `bytes` as the artefact `name`, replacing any artefact of that
    /// name, and make it durable before returning. Any directory the name
    /// carries is created first.
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), Error>;

    /// The artefact `name`, whole.
    fn read(&self, name: &str) -> Result<Vec<u8>, Error>;

    /// Whether the artefact `name` is present.
    fn exists(&self, name: &str) -> bool;

    /// The artefact's length in bytes.
    fn length(&self, name: &str) -> io::Result<u64>;

    /// Remove the artefact `name`.
    fn remove(&self, name: &str) -> io::Result<()>;

    /// Open the artefact `name` to be written as a stream, replacing any
    /// artefact of that name. Any directory the name carries is created
    /// first.
    fn create(&self, name: &str) -> io::Result<Box<dyn ArtefactWriter>>;

    /// Open the artefact `name` to be read as a stream.
    fn open(&self, name: &str) -> io::Result<Box<dyn Read>>;

    /// Where the artefact `name` is, as a message names it, or the directory
    /// itself where `name` is empty.
    fn locate(&self, name: &str) -> String;
}

/// An artefact written as a stream, durable once it is finished.
pub trait ArtefactWriter: Write {
    /// Write `head` over the artefact's first bytes, which the stream has
    /// already written, and flush it. The one operation that needs the
    /// artefact to be rewritable at its start. A stream that writes its head
    /// first finishes without it.
    fn write_head(&mut self, head: &[u8]) -> io::Result<()>;

    /// Make the artefact durable and close it.
    fn finish(self: Box<Self>) -> io::Result<()>;
}

/// The place one collection is kept: its directory, the directory a save
/// writes before it replaces that one, the journal beside them, and the lock.
pub trait Storage {
    /// Put back a directory a save was killed in the middle of replacing,
    /// and say whether anything moved. A load runs this before it looks for
    /// the directory, and [`Storage::stage`] runs it before it stages.
    fn recover(&self) -> Result<bool, Error>;

    /// Whether a directory is in place.
    fn exists(&self) -> bool;

    /// Whether a directory a save left aside is present, which
    /// [`Storage::recover`] puts back where no directory is in place. A load
    /// asks this and [`Storage::exists`] before it takes the lock, so a load
    /// of nothing leaves no lock file behind.
    fn aside_exists(&self) -> bool;

    /// The directory in place, to read.
    fn dir(&self) -> &dyn Dir;

    /// Clear what an earlier save left behind and open an empty directory
    /// for this save to write into.
    fn stage(&self) -> Result<Box<dyn Staged>, Error>;

    /// What stands at the target that a save must not replace, being a
    /// file, a folder holding entries and no `manifest.json`, or a folder
    /// this process cannot list, as a phrase a refusal quotes. `None` where
    /// nothing stands there, where an empty folder does, and where a folder
    /// holding a manifest does, which is an index directory an earlier save
    /// wrote. A save replaces the directory at its target whole, so it asks
    /// this before anything is written. A target the system will not
    /// describe at all is `None`, and the save's own first write meets the
    /// same refusal.
    fn foreign_target(&self) -> Option<String>;

    /// Whether the folder the directory sits in takes a new entry, found by
    /// creating the staging directory and removing it again. A load runs
    /// this before the first write a checkpoint makes, and reads the I/O
    /// error to tell a medium that refuses writes from any other failure.
    fn probe_writable(&self) -> io::Result<()>;

    /// Where the journal is, being the directory of its segments, which its
    /// errors and its caller name.
    fn journal_path(&self) -> &Path;

    /// Whether `journal` names this collection's journal.
    fn names_journal(&self, journal: &Path) -> bool;

    /// Whether the journal's directory is present.
    fn journal_exists(&self) -> bool;

    /// Create the journal's directory, or empty it of segments where it is
    /// present. Any entry that is not a segment is left where it is.
    fn create_journal(&self) -> io::Result<()>;

    /// The journal's directory of segments, owned, for a writer to keep.
    fn journal_dir(&self) -> Box<dyn JournalDir>;

    /// Where a journal of format version 1 sits, being one file beside the
    /// directory, which a release before 0.14.0 wrote.
    fn legacy_journal_path(&self) -> &Path;

    /// Whether a journal of format version 1 is present.
    fn legacy_journal_exists(&self) -> bool;

    /// The journal of format version 1, whole.
    fn read_legacy_journal(&self) -> io::Result<Vec<u8>>;

    /// Remove the journal of format version 1, which an upgrade does once
    /// its checkpoint holds every record the file held.
    fn remove_legacy_journal(&self) -> io::Result<()>;

    /// Cut the journal of format version 1 to its first `len` bytes and make
    /// the cut durable, which a repair does once its quarantine holds every
    /// byte past `len`.
    fn cut_legacy_journal(&self, len: u64) -> io::Result<()>;

    /// Where the lock file is.
    fn lock_path(&self) -> &Path;

    /// Take the collection's lock in `mode`, without waiting, and record
    /// `holder` in the lock file where the lock is exclusive. An exclusive
    /// lock creates the folder the directory sits in where it is missing, as
    /// the save that takes it would.
    ///
    /// A lock file that cannot be opened for writing, because the medium
    /// refuses writes or a permission is missing, is opened to read and
    /// locked shared, and the answer is [`Locked::ReadOnly`] with the
    /// system's refusal. One that does not exist and cannot be created gives
    /// that answer with no lock at all.
    fn lock(&self, mode: LockMode, holder: &str) -> Result<Locked, LockFailure>;
}

/// The directory a save writes, which the commit puts in place whole.
/// Dropped without a commit, it is removed.
pub trait Staged {
    /// The directory, to write.
    fn dir(&self) -> &dyn Dir;

    /// Put the directory in place of the one there, whole, so a reader sees
    /// the one or the other and never a mixture. `between` is the point a
    /// crash test aborts at between the two renames, being the save's or a
    /// backup's.
    fn commit(self: Box<Self>, between: crate::KillPoint) -> Result<(), Error>;
}

/// The journal, being a directory of segments, each a file named by the
/// sequence of its first record.
///
/// It is owned by the writer that appends to it, so every operation takes
/// `&self` and nothing borrows from the [`Storage`] it came from.
pub trait JournalDir: Send + Sync + std::fmt::Debug {
    /// Where the directory is, which a message names.
    fn path(&self) -> &Path;

    /// Every segment the directory holds, by the number its file is named
    /// for, in increasing order. An entry not named as a segment is left
    /// where it is and not listed.
    fn segments(&self) -> io::Result<Vec<u64>>;

    /// The segment `number`, whole.
    fn read_segment(&self, number: u64) -> io::Result<Vec<u8>>;

    /// Create the segment `number`, which must not exist yet, open to read
    /// and write.
    fn create_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>>;

    /// Open the segment `number` to read and write.
    fn open_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>>;

    /// Remove the segment `number`.
    fn remove_segment(&self, number: u64) -> io::Result<()>;

    /// Make the directory's own entries durable, after a segment is created
    /// or removed, where the system has a call for it. Best effort.
    fn sync(&self);

    /// Where the segment `number` is, which a message names.
    fn segment_path(&self, number: u64) -> std::path::PathBuf;

    /// Create the folder `folder` under the journal's `quarantine` folder,
    /// and the `quarantine` folder where it is missing. The first write a
    /// repair makes, so the error is the system's, for the caller to tell a
    /// medium that refuses writes from any other failure.
    fn create_quarantine(&self, folder: &str) -> io::Result<()>;

    /// The journal's `quarantine` folder as a directory of artefacts, each
    /// named `<folder>/<file>`. A file written there is durable before the
    /// write returns.
    fn quarantine(&self) -> Box<dyn Dir>;

    /// Every folder under the journal's `quarantine` folder, by name, in
    /// name order, and none where it does not exist.
    fn quarantine_folders(&self) -> io::Result<Vec<String>>;

    /// Make the entries of the folder `folder` under `quarantine`, of
    /// `quarantine` and of the journal's directory durable, where the
    /// system has a call for it. Best effort.
    fn sync_quarantine(&self, folder: &str);
}

/// One segment of the journal, open to read and write.
///
/// One file with a position. Records are written at its end, so the writer
/// moves the position itself. The writer and the thread that syncs on an
/// interval share it, so every operation takes `&self`.
pub trait JournalFile: Send + Sync + std::fmt::Debug {
    /// Write `bytes` at the position and leave the position after them.
    fn write_all(&self, bytes: &[u8]) -> io::Result<()>;

    /// Put the position after the last byte.
    fn seek_end(&self) -> io::Result<()>;

    /// Cut the file to `len` bytes.
    fn set_len(&self, len: u64) -> io::Result<()>;

    /// Make the file's data durable.
    fn sync_data(&self) -> io::Result<()>;

    /// Make the file's data and its length durable.
    fn sync_all(&self) -> io::Result<()>;

    /// The file's length.
    fn length(&self) -> io::Result<u64>;
}

/// How a lock is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockMode {
    /// Beside other shared holders, and refused while anyone holds it
    /// exclusive. A load takes it while it reads.
    Shared,
    /// Alone. A journaled collection takes it for its life and a save for
    /// the save.
    Exclusive,
}

impl LockMode {
    /// How the mode names itself in the lock file and in a message.
    pub fn name(self) -> &'static str {
        match self {
            LockMode::Shared => "shared",
            LockMode::Exclusive => "exclusive",
        }
    }
}

/// A lock on a collection, held until it is dropped.
///
/// Dropping it in the process that took the lock unlocks the lock file and
/// then closes it, so the lock goes even while a child process still holds a
/// copy of the file. A child forked from the holder shares the open file, and
/// an unlock there would release the parent's lock, so the child closes its
/// copy without unlocking it, which leaves the lock with the parent; see
/// [`close_lock_files_in_forked_child`].
pub trait LockGuard: Send + Sync + std::fmt::Debug {
    /// The mode the lock is held in.
    fn mode(&self) -> LockMode;
}

/// What taking a lock gave.
#[derive(Debug)]
pub enum Locked {
    /// The lock, in the mode asked for.
    Writable(Box<dyn LockGuard>),
    /// The lock file could not be opened for writing, so whatever holds the
    /// collection must not write to it. `guard` is a shared lock where the
    /// lock file exists, and `reason` is the system's refusal.
    ReadOnly {
        guard: Option<Box<dyn LockGuard>>,
        reason: io::Error,
    },
}

/// Why a lock was not taken.
#[derive(Debug)]
pub enum LockFailure {
    /// Another holder has it, in a mode that excludes this one. `holder` is
    /// what the lock file records of the last exclusive holder, where the
    /// system lets it be read, as a phrase a refusal quotes.
    Held { holder: String },
    /// The filesystem answered the lock with an error of its own, so it
    /// cannot lock.
    Unavailable(io::Error),
    /// The lock file could not be opened.
    Failed(io::Error),
}

/// Whether `error` is the refusal a system gives for a medium that cannot
/// be written or for a permission that is missing, being `EROFS`, `EACCES`
/// and `EPERM` on Linux and macOS and `ERROR_WRITE_PROTECT` and
/// `ERROR_ACCESS_DENIED` on Windows.
pub fn refuses_writes(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ReadOnlyFilesystem | io::ErrorKind::PermissionDenied
    )
}
