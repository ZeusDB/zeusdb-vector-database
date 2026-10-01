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
//! [`Storage::recover`], and where the journal beside the directory lives.
//!
//! [`Staged`] is the directory a save writes before the commit puts it in
//! place. Dropped without a commit, it is removed.
//!
//! [`JournalFile`] is the journal, open to read and write.
//!
//! # Why the journal is not an artefact
//!
//! The directory is replaced whole at every checkpoint and the journal stays
//! open across that replacement, so the journal sits beside the directory
//! rather than in it. Its writes are appends at the end, a cut back to a
//! length and a header rewritten in place, which no artefact takes, and it
//! is shared with the thread that syncs it on an interval. So it is a trait
//! of its own, opened through [`Storage`].
//!
//! # Errors
//!
//! The whole-artefact operations, the staging, the commit and the recovery
//! return the engine's [`Error`], naming the artefact or the path, because
//! every caller reports them the same way. The rest return the I/O error and
//! the caller words it.

mod fs;

pub use fs::{FsDir, FsStorage, JOURNAL_SUFFIX};

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

    /// Every byte the directory holds, under every prefix.
    fn total_bytes(&self) -> io::Result<u64>;

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
/// writes before it replaces that one, and the journal beside them.
pub trait Storage {
    /// Put back a directory a save was killed in the middle of replacing,
    /// and say whether anything moved. A load runs this before it looks for
    /// the directory, and [`Storage::stage`] runs it before it stages.
    fn recover(&self) -> Result<bool, Error>;

    /// Whether a directory is in place.
    fn exists(&self) -> bool;

    /// The directory in place, to read.
    fn dir(&self) -> &dyn Dir;

    /// Clear what an earlier save left behind and open an empty directory
    /// for this save to write into.
    fn stage(&self) -> Result<Box<dyn Staged>, Error>;

    /// Where the journal is, which its errors and its caller name.
    fn journal_path(&self) -> &Path;

    /// Whether `journal` names this collection's journal.
    fn names_journal(&self, journal: &Path) -> bool;

    /// Whether the journal is present.
    fn journal_exists(&self) -> bool;

    /// The journal's bytes, whole.
    fn read_journal(&self) -> io::Result<Vec<u8>>;

    /// Create the journal with no bytes, replacing any journal there, open
    /// to read and write.
    fn create_journal(&self) -> io::Result<Box<dyn JournalFile>>;

    /// Open the journal there to read and write.
    fn open_journal(&self) -> io::Result<Box<dyn JournalFile>>;
}

/// The directory a save writes, which the commit puts in place whole.
/// Dropped without a commit, it is removed.
pub trait Staged {
    /// The directory, to write.
    fn dir(&self) -> &dyn Dir;

    /// Put the directory in place of the one there, whole, so a reader sees
    /// the one or the other and never a mixture.
    fn commit(self: Box<Self>) -> Result<(), Error>;
}

/// The journal, open to read and write.
///
/// One file with a position. Records are written at its end and the header
/// is rewritten at its start, so the writer moves the position itself. The
/// writer and the thread that syncs on an interval share it, so every
/// operation takes `&self`.
pub trait JournalFile: Send + Sync + std::fmt::Debug {
    /// Write `bytes` at the position and leave the position after them.
    fn write_all(&self, bytes: &[u8]) -> io::Result<()>;

    /// Put the position at the first byte.
    fn seek_start(&self) -> io::Result<()>;

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
