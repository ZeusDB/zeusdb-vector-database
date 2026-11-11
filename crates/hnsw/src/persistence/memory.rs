//! A collection's directory held in memory, for the tests that load and save
//! many directories, so a case costs the load's own work and writes nothing
//! to disk.
//!
//! [`MemDir`] is a directory of artefacts by name, a name naming its folders
//! with `/`. [`MemStorage`] keeps one such directory and no journal. A load
//! reads the directory, and a save stages a second directory whose commit
//! puts its artefacts in place whole, as a save to a folder does. Its lock is
//! taken and let go in memory. A journal is never present, so a directory
//! whose manifest names one opens as its checkpoint alone or is refused, as
//! the load's policy says.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use zeusdb_vector_core::{
    ArtefactWriter, Dir, Error, JournalDir, JournalFile, KillPoint, LockFailure, LockGuard,
    LockMode, Locked, Staged, Storage,
};

/// The artefacts of one directory, by name.
pub(crate) type Files = BTreeMap<String, Vec<u8>>;

/// A directory of artefacts in memory. Clones share the artefacts.
#[derive(Clone, Default)]
pub(crate) struct MemDir {
    files: Rc<RefCell<Files>>,
}

impl MemDir {
    /// A directory holding `files`.
    pub(crate) fn holding(files: Files) -> Self {
        MemDir {
            files: Rc::new(RefCell::new(files)),
        }
    }

    /// Every artefact the directory holds, copied.
    pub(crate) fn files(&self) -> Files {
        self.files.borrow().clone()
    }

    fn missing(name: &str) -> io::Error {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("the directory in memory holds no {name}"),
        )
    }
}

impl Dir for MemDir {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        self.files
            .borrow_mut()
            .insert(name.to_string(), bytes.to_vec());
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.files
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::ArtefactReadFailed {
                name: name.to_string(),
                error: Self::missing(name).to_string(),
            })
    }

    fn exists(&self, name: &str) -> bool {
        self.files.borrow().contains_key(name)
    }

    fn length(&self, name: &str) -> io::Result<u64> {
        self.files
            .borrow()
            .get(name)
            .map(|bytes| bytes.len() as u64)
            .ok_or_else(|| Self::missing(name))
    }

    fn remove(&self, name: &str) -> io::Result<()> {
        self.files
            .borrow_mut()
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| Self::missing(name))
    }

    fn create(&self, name: &str) -> io::Result<Box<dyn ArtefactWriter>> {
        Ok(Box::new(MemWriter {
            name: name.to_string(),
            bytes: Vec::new(),
            dir: self.clone(),
        }))
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read>> {
        self.files
            .borrow()
            .get(name)
            .cloned()
            .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read>)
            .ok_or_else(|| Self::missing(name))
    }

    fn locate(&self, name: &str) -> String {
        format!("memory/{name}")
    }
}

/// An artefact written as a stream into a [`MemDir`], which holds it once
/// it is finished.
struct MemWriter {
    name: String,
    bytes: Vec<u8>,
    dir: MemDir,
}

impl Write for MemWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ArtefactWriter for MemWriter {
    fn write_head(&mut self, head: &[u8]) -> io::Result<()> {
        if head.len() > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the head is longer than the artefact written",
            ));
        }
        self.bytes[..head.len()].copy_from_slice(head);
        Ok(())
    }

    fn finish(self: Box<Self>) -> io::Result<()> {
        let MemWriter { name, bytes, dir } = *self;
        dir.files.borrow_mut().insert(name, bytes);
        Ok(())
    }
}

/// The storage of one collection whose directory is held in memory, with no
/// journal beside it.
pub(crate) struct MemStorage {
    dir: MemDir,
    journal: PathBuf,
    legacy: PathBuf,
    lock: PathBuf,
}

impl MemStorage {
    /// A storage whose directory holds `files`, or no directory where
    /// `files` is empty.
    pub(crate) fn holding(files: Files) -> Self {
        MemStorage {
            dir: MemDir::holding(files),
            journal: PathBuf::from("memory.zdbjournal"),
            legacy: PathBuf::from("memory.zdbwal"),
            lock: PathBuf::from("memory.zdblock"),
        }
    }

    /// Every artefact the directory holds, copied.
    pub(crate) fn files(&self) -> Files {
        self.dir.files()
    }

    fn unsupported(what: &str) -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            format!("a directory in memory keeps no {what}"),
        )
    }
}

impl Storage for MemStorage {
    fn recover(&self) -> Result<bool, Error> {
        Ok(false)
    }

    fn exists(&self) -> bool {
        !self.dir.files.borrow().is_empty()
    }

    fn aside_exists(&self) -> bool {
        false
    }

    fn dir(&self) -> &dyn Dir {
        &self.dir
    }

    fn stage(&self) -> Result<Box<dyn Staged>, Error> {
        Ok(Box::new(MemStaged {
            staged: MemDir::default(),
            target: self.dir.clone(),
        }))
    }

    fn foreign_target(&self) -> Option<String> {
        None
    }

    fn probe_writable(&self) -> io::Result<()> {
        Ok(())
    }

    fn journal_path(&self) -> &Path {
        &self.journal
    }

    fn names_journal(&self, journal: &Path) -> bool {
        journal == self.journal
    }

    fn journal_exists(&self) -> bool {
        false
    }

    fn create_journal(&self) -> io::Result<()> {
        Err(Self::unsupported("journal"))
    }

    fn journal_dir(&self) -> Box<dyn JournalDir> {
        Box::new(NoJournal {
            path: self.journal.clone(),
        })
    }

    fn legacy_journal_path(&self) -> &Path {
        &self.legacy
    }

    fn legacy_journal_exists(&self) -> bool {
        false
    }

    fn read_legacy_journal(&self) -> io::Result<Vec<u8>> {
        Err(Self::unsupported("journal"))
    }

    fn remove_legacy_journal(&self) -> io::Result<()> {
        Ok(())
    }

    fn sync_in_place(&self) -> io::Result<()> {
        Ok(())
    }

    fn cut_legacy_journal(&self, _len: u64) -> io::Result<()> {
        Err(Self::unsupported("journal"))
    }

    fn lock_path(&self) -> &Path {
        &self.lock
    }

    fn lock(&self, mode: LockMode, _holder: &str) -> Result<Locked, LockFailure> {
        Ok(Locked::Writable(Box::new(MemLock(mode))))
    }
}

/// The directory a save into a [`MemStorage`] writes, which its commit puts
/// in place whole.
struct MemStaged {
    staged: MemDir,
    target: MemDir,
}

impl Staged for MemStaged {
    fn dir(&self) -> &dyn Dir {
        &self.staged
    }

    fn commit(self: Box<Self>, _between: KillPoint) -> Result<(), Error> {
        let staged = self.staged.files();
        *self.target.files.borrow_mut() = staged;
        Ok(())
    }
}

/// A lock held in memory, which nothing else contends for.
#[derive(Debug)]
struct MemLock(LockMode);

impl LockGuard for MemLock {
    fn mode(&self) -> LockMode {
        self.0
    }
}

/// The journal of a [`MemStorage`], which holds no segment and refuses
/// every write.
#[derive(Debug)]
struct NoJournal {
    path: PathBuf,
}

impl NoJournal {
    fn refused() -> io::Error {
        MemStorage::unsupported("journal")
    }
}

impl JournalDir for NoJournal {
    fn path(&self) -> &Path {
        &self.path
    }

    fn segments(&self) -> io::Result<Vec<u64>> {
        Ok(Vec::new())
    }

    fn read_segment(&self, _number: u64) -> io::Result<Vec<u8>> {
        Err(Self::refused())
    }

    fn create_segment(&self, _number: u64) -> io::Result<Box<dyn JournalFile>> {
        Err(Self::refused())
    }

    fn open_segment(&self, _number: u64) -> io::Result<Box<dyn JournalFile>> {
        Err(Self::refused())
    }

    fn remove_segment(&self, _number: u64) -> io::Result<()> {
        Err(Self::refused())
    }

    fn sync(&self) -> io::Result<()> {
        Ok(())
    }

    fn segment_path(&self, number: u64) -> PathBuf {
        self.path.join(format!("{number:020}"))
    }

    fn create_quarantine(&self, _folder: &str) -> io::Result<()> {
        Err(Self::refused())
    }

    fn quarantine(&self) -> Box<dyn Dir> {
        Box::new(MemDir::default())
    }

    fn quarantine_folders(&self) -> io::Result<Vec<String>> {
        Ok(Vec::new())
    }

    fn sync_quarantine(&self, _folder: &str) -> io::Result<()> {
        Ok(())
    }
}
