//! What the journal's tests share: a temporary folder, a journal read back
//! by hand, faults armed on a journal's directory, and a storage on a medium
//! that refuses every write.
//!
//! A device fault cannot be produced on demand, so a failure is injected
//! through a journal directory that fails a chosen write, sync, cut or
//! creation, and a medium that refuses writes is a storage that answers
//! every write with the system's refusal. Both wrap the filesystem storage
//! and pass every other call through.

#![allow(clippy::disallowed_types)]

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::Value;
use zeusdb_vector_core::{
    Dir, Error, Expected, FsStorage, JournalContents, JournalDir, JournalFile, LockFailure,
    LockGuard, LockMode, Locked, SegmentBase, SegmentBytes, Staged, Storage,
};

use crate::journal::{Durability, JournalSink};

// ============================================================================
// A TEMPORARY FOLDER
// ============================================================================

/// A folder under the system's temporary folder, removed on drop together
/// with everything in it, the journals and lock files beside a directory
/// among them.
pub(super) struct TempDir(pub(super) PathBuf);

impl TempDir {
    pub(super) fn new(prefix: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zeusdb-{prefix}-{}-{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    pub(super) fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ============================================================================
// THE JOURNAL, READ BACK BY HAND
// ============================================================================

/// The manifest of the directory at `directory`.
pub(super) fn manifest(directory: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
        .unwrap()
}

/// Rewrite the manifest of the directory at `directory` through `edit`.
pub(super) fn rewrite_manifest(directory: &Path, edit: impl FnOnce(&mut Value)) {
    let mut m = manifest(directory);
    edit(&mut m);
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
}

fn hex(value: &Value) -> u128 {
    u128::from_str_radix(value.as_str().unwrap(), 16).unwrap()
}

/// The checkpoint the manifest of the directory at `directory` records, as
/// every segment naming it names it.
pub(super) fn expected_of(directory: &Path) -> Expected {
    let m = manifest(directory);
    let identity = &m["identity"];
    Expected {
        collection_id: hex(&identity["collection_id"]),
        base: SegmentBase {
            snapshot: hex(&identity["snapshot"]),
            sequence: identity["sequence"].as_u64().unwrap(),
            generation: identity["generation"].as_u64().unwrap(),
            parent: if identity["parent"].is_null() {
                0
            } else {
                hex(&identity["parent"])
            },
        },
    }
}

/// Every segment of the journal beside a directory, read whole, for the
/// reader to borrow from.
pub(super) struct Segments {
    pub(super) numbers: Vec<u64>,
    pub(super) files: Vec<String>,
    pub(super) bytes: Vec<Vec<u8>>,
}

impl Segments {
    /// The segments of the journal beside the directory at `directory`.
    pub(super) fn of(directory: &Path) -> Self {
        let storage = FsStorage::at(directory).unwrap();
        let dir = storage.journal_dir();
        let numbers = dir.segments().unwrap();
        let files = numbers
            .iter()
            .map(|n| dir.segment_path(*n).display().to_string())
            .collect();
        let bytes = numbers
            .iter()
            .map(|n| dir.read_segment(*n).unwrap())
            .collect();
        Segments {
            numbers,
            files,
            bytes,
        }
    }

    pub(super) fn read(&self, expected: &Expected) -> Result<JournalContents<'_>, Error> {
        let segments: Vec<SegmentBytes<'_>> = self
            .numbers
            .iter()
            .zip(&self.files)
            .zip(&self.bytes)
            .map(|((number, file), bytes)| SegmentBytes {
                number: *number,
                file,
                bytes,
            })
            .collect();
        zeusdb_vector_core::read_segments(&segments, expected, "journal")
    }

    /// Every byte of every segment.
    pub(super) fn total_bytes(&self) -> u64 {
        self.bytes.iter().map(|b| b.len() as u64).sum()
    }
}

/// Every file under each of `roots`, by path, with its bytes, so a test can
/// hold that nothing changed.
pub(super) fn files_under(roots: &[&Path]) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(path: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for entry in entries {
                walk(&entry, out);
            }
        } else if path.is_file() {
            out.push((path.to_path_buf(), std::fs::read(path).unwrap()));
        }
    }
    let mut out = Vec::new();
    for root in roots {
        walk(root, &mut out);
    }
    out
}

/// A copy of every file under `from` at `to`, folders included.
pub(super) fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// A copy of a directory and the journal beside it under another name, so
/// the copy can be recovered while the original's sink is still open.
pub(super) fn copy_journaled(directory: &Path, name: &str) -> PathBuf {
    let target = directory.parent().unwrap().join(name);
    copy_tree(directory, &target);
    let journal = crate::journal::journal_path(directory).unwrap();
    if journal.is_dir() {
        copy_tree(&journal, &crate::journal::journal_path(&target).unwrap());
    }
    target
}

/// A journal sink over `storage` for the collection `collection_id`, under
/// the directory's lock, as `journal_to` makes one, for a test that attaches
/// the sink and writes the checkpoint itself.
pub(super) fn sink_over(
    storage: &dyn Storage,
    directory: &Path,
    collection_id: u128,
    durability: Durability,
) -> JournalSink {
    let directory = directory.to_str().unwrap();
    let lock = crate::journal::lock_for_journal(storage, directory).unwrap();
    JournalSink::create(storage, collection_id, durability, lock).unwrap()
}

// ============================================================================
// THE FIXTURES OLDER RELEASES WROTE
// ============================================================================

/// The journaled directories older releases wrote, under
/// `tests/fixtures/formats`: a raw index at 3.0.0 and one trained with
/// scalar quantization at 3.1.0, both by 0.11.0, and one with a text space
/// at 4.0.0 by 0.13.0. Each folder holds `journaled.zdb`, the journal of
/// format version 1 beside it, ending in a torn tail, and `expected.json`.
pub(super) const FIXTURES: [&str; 3] = [
    "v3.0.0-journaled",
    "v3.1.0-journaled-int8",
    "v4.0.0-journaled-text",
];

/// The folder the fixture `name` is kept in.
pub(super) fn fixture_folder(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("formats")
        .join(name)
}

/// A copy of the fixture `name` under `into`, and the directory in it to
/// load. A load upgrades the directory it opens, so a test loads a copy and
/// the fixture stays as the release wrote it.
pub(super) fn copy_fixture(name: &str, into: &Path) -> PathBuf {
    let target = into.join(name);
    copy_tree(&fixture_folder(name), &target);
    target.join("journaled.zdb")
}

/// What the fixture `name` records.
pub(super) fn fixture_expected(name: &str) -> Value {
    let text = std::fs::read_to_string(fixture_folder(name).join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A vector as the fixtures spell it, the hexadecimal of its float32 bytes.
fn vector_hex(vector: &[f32]) -> String {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Hold `collection` to what the fixture `name` records: every record it
/// names held with its vector, bit for bit, and its metadata, no record it
/// names absent held, nothing else held, and the term count where it records
/// one.
pub(super) fn holds_fixture(collection: &super::Collection, name: &str) -> Result<(), String> {
    let expected = fixture_expected(name);
    let records = expected["records"].as_object().unwrap();
    let names: Vec<String> = records.keys().cloned().collect();
    let views = collection
        .records(names.clone(), true, false)
        .map_err(|e| e.to_string())?;
    if views.len() != names.len() {
        return Err(format!(
            "{name}: {} of the {} records came back",
            views.len(),
            names.len()
        ));
    }
    for view in &views {
        let want = &records[&view.id];
        let vector = view.vector.as_deref().unwrap_or(&[]);
        if vector_hex(vector) != want["vector"].as_str().unwrap() {
            return Err(format!(
                "{name}: record {} came back with another vector",
                view.id
            ));
        }
        let metadata: serde_json::Map<String, Value> = view
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if Value::Object(metadata) != want["metadata"] {
            return Err(format!(
                "{name}: record {} came back with metadata {:?}",
                view.id, view.metadata
            ));
        }
    }
    for absent in expected["absent"].as_array().unwrap() {
        if collection.contains(absent.as_str().unwrap()) {
            return Err(format!("{name}: {absent} is held and should not be"));
        }
    }
    if collection.len() != names.len() {
        return Err(format!(
            "{name}: {} records are held and {} are recorded",
            collection.len(),
            names.len()
        ));
    }
    let terms = expected["term_count"].as_u64().map(|n| n as usize);
    if collection.term_count() != terms {
        return Err(format!(
            "{name}: the term count is {:?} and {:?} is recorded",
            collection.term_count(),
            terms
        ));
    }
    Ok(())
}

// ============================================================================
// FAULTS ARMED ON A JOURNAL'S DIRECTORY
// ============================================================================

/// A failure armed on one kind of call: `pass` calls go through, then the
/// `fail` calls after them fail.
#[derive(Debug, Default)]
pub(super) struct Arm {
    pass: AtomicUsize,
    fail: AtomicUsize,
}

impl Arm {
    pub(super) fn arm(&self, pass: usize, fail: usize) {
        self.pass.store(pass, Ordering::SeqCst);
        self.fail.store(fail, Ordering::SeqCst);
    }

    /// Whether every failure armed has been spent.
    pub(super) fn spent(&self) -> bool {
        self.fail.load(Ordering::SeqCst) == 0
    }

    /// Whether this call fails.
    fn trips(&self) -> bool {
        let down = |n: usize| n.checked_sub(1);
        if self.fail.load(Ordering::SeqCst) == 0 {
            return false;
        }
        if self
            .pass
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, down)
            .is_ok()
        {
            return false;
        }
        self.fail
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, down)
            .is_ok()
    }
}

/// Failures a test arms on a journal's directory. A failing write lands half
/// its bytes first, which is what a device that fills part way leaves.
#[derive(Debug, Default)]
pub(super) struct Faults {
    pub(super) writes: Arm,
    pub(super) syncs: Arm,
    pub(super) cuts: Arm,
    pub(super) creates: Arm,
}

fn injected() -> io::Error {
    io::Error::other("injected: the device refused it")
}

#[derive(Debug)]
struct FaultyDir {
    inner: Box<dyn JournalDir>,
    faults: Arc<Faults>,
}

#[derive(Debug)]
struct FaultyFile {
    inner: Box<dyn JournalFile>,
    faults: Arc<Faults>,
}

impl JournalDir for FaultyDir {
    fn path(&self) -> &Path {
        self.inner.path()
    }
    fn segments(&self) -> io::Result<Vec<u64>> {
        self.inner.segments()
    }
    fn read_segment(&self, number: u64) -> io::Result<Vec<u8>> {
        self.inner.read_segment(number)
    }
    fn create_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>> {
        if self.faults.creates.trips() {
            return Err(injected());
        }
        Ok(Box::new(FaultyFile {
            inner: self.inner.create_segment(number)?,
            faults: self.faults.clone(),
        }))
    }
    fn open_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>> {
        Ok(Box::new(FaultyFile {
            inner: self.inner.open_segment(number)?,
            faults: self.faults.clone(),
        }))
    }
    fn remove_segment(&self, number: u64) -> io::Result<()> {
        self.inner.remove_segment(number)
    }
    fn sync(&self) {
        self.inner.sync()
    }
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
}

impl JournalFile for FaultyFile {
    fn write_all(&self, bytes: &[u8]) -> io::Result<()> {
        if self.faults.writes.trips() {
            self.inner.write_all(&bytes[..bytes.len() / 2])?;
            return Err(injected());
        }
        self.inner.write_all(bytes)
    }
    fn seek_end(&self) -> io::Result<()> {
        self.inner.seek_end()
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        if self.faults.cuts.trips() {
            return Err(injected());
        }
        self.inner.set_len(len)
    }
    fn sync_data(&self) -> io::Result<()> {
        if self.faults.syncs.trips() {
            return Err(injected());
        }
        self.inner.sync_data()
    }
    fn sync_all(&self) -> io::Result<()> {
        self.inner.sync_all()
    }
    fn length(&self) -> io::Result<u64> {
        self.inner.length()
    }
}

/// A collection's storage on the filesystem whose journal is reached through
/// the faults a test arms. Every other call is the filesystem's.
#[derive(Debug)]
pub(super) struct FaultyStorage {
    inner: FsStorage,
    pub(super) faults: Arc<Faults>,
}

impl FaultyStorage {
    pub(super) fn at(directory: &Path) -> Self {
        FaultyStorage {
            inner: FsStorage::at(directory).unwrap(),
            faults: Arc::new(Faults::default()),
        }
    }
}

impl Storage for FaultyStorage {
    fn recover(&self) -> Result<bool, Error> {
        self.inner.recover()
    }
    fn exists(&self) -> bool {
        self.inner.exists()
    }
    fn aside_exists(&self) -> bool {
        self.inner.aside_exists()
    }
    fn dir(&self) -> &dyn Dir {
        self.inner.dir()
    }
    fn stage(&self) -> Result<Box<dyn Staged>, Error> {
        self.inner.stage()
    }
    fn probe_writable(&self) -> io::Result<()> {
        self.inner.probe_writable()
    }
    fn journal_path(&self) -> &Path {
        self.inner.journal_path()
    }
    fn names_journal(&self, journal: &Path) -> bool {
        self.inner.names_journal(journal)
    }
    fn journal_exists(&self) -> bool {
        self.inner.journal_exists()
    }
    fn create_journal(&self) -> io::Result<()> {
        self.inner.create_journal()
    }
    fn journal_dir(&self) -> Box<dyn JournalDir> {
        Box::new(FaultyDir {
            inner: self.inner.journal_dir(),
            faults: self.faults.clone(),
        })
    }
    fn legacy_journal_path(&self) -> &Path {
        self.inner.legacy_journal_path()
    }
    fn legacy_journal_exists(&self) -> bool {
        self.inner.legacy_journal_exists()
    }
    fn read_legacy_journal(&self) -> io::Result<Vec<u8>> {
        self.inner.read_legacy_journal()
    }
    fn remove_legacy_journal(&self) -> io::Result<()> {
        self.inner.remove_legacy_journal()
    }
    fn lock_path(&self) -> &Path {
        self.inner.lock_path()
    }
    fn lock(&self, mode: LockMode, holder: &str) -> Result<Locked, LockFailure> {
        self.inner.lock(mode, holder)
    }
}

// ============================================================================
// A MEDIUM THAT REFUSES EVERY WRITE
// ============================================================================

/// A lock taken shared through a lock file opened to read, which is what a
/// load on a medium that refuses writes holds.
#[derive(Debug)]
struct SharedLock(#[allow(dead_code)] std::fs::File);

impl LockGuard for SharedLock {
    fn mode(&self) -> LockMode {
        LockMode::Shared
    }
}

/// A collection's storage on a medium that refuses every write with the
/// system's refusal of `kind`, being `ReadOnlyFilesystem` or
/// `PermissionDenied`. Reads are the filesystem's. The lock file is opened to
/// read and locked shared where it exists, and not created where it does
/// not, as the filesystem storage does on such a medium.
#[derive(Debug)]
pub(super) struct RefusingStorage {
    inner: FsStorage,
    kind: io::ErrorKind,
}

impl RefusingStorage {
    pub(super) fn at(directory: &Path, kind: io::ErrorKind) -> Self {
        RefusingStorage {
            inner: FsStorage::at(directory).unwrap(),
            kind,
        }
    }

    fn refusal(&self) -> io::Error {
        io::Error::new(self.kind, "the medium refuses writes")
    }
}

#[derive(Debug)]
struct RefusingDir {
    inner: Box<dyn JournalDir>,
    kind: io::ErrorKind,
}

impl JournalDir for RefusingDir {
    fn path(&self) -> &Path {
        self.inner.path()
    }
    fn segments(&self) -> io::Result<Vec<u64>> {
        self.inner.segments()
    }
    fn read_segment(&self, number: u64) -> io::Result<Vec<u8>> {
        self.inner.read_segment(number)
    }
    fn create_segment(&self, _number: u64) -> io::Result<Box<dyn JournalFile>> {
        Err(io::Error::new(self.kind, "the medium refuses writes"))
    }
    fn open_segment(&self, _number: u64) -> io::Result<Box<dyn JournalFile>> {
        Err(io::Error::new(self.kind, "the medium refuses writes"))
    }
    fn remove_segment(&self, _number: u64) -> io::Result<()> {
        Err(io::Error::new(self.kind, "the medium refuses writes"))
    }
    fn sync(&self) {}
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
}

impl Storage for RefusingStorage {
    fn recover(&self) -> Result<bool, Error> {
        if self.inner.aside_exists() && !self.inner.exists() {
            return Err(Error::RecoverRenameFailed {
                target: PathBuf::from(self.inner.dir().locate("")),
                replaced: PathBuf::from(format!("{}.zdbold", self.inner.dir().locate(""))),
                error: self.refusal().to_string(),
            });
        }
        Ok(false)
    }
    fn exists(&self) -> bool {
        self.inner.exists()
    }
    fn aside_exists(&self) -> bool {
        self.inner.aside_exists()
    }
    fn dir(&self) -> &dyn Dir {
        self.inner.dir()
    }
    fn stage(&self) -> Result<Box<dyn Staged>, Error> {
        Err(Error::StagingCreateFailed {
            staging: PathBuf::from(format!("{}.zdbtmp", self.inner.dir().locate(""))),
            error: self.refusal().to_string(),
        })
    }
    fn probe_writable(&self) -> io::Result<()> {
        Err(self.refusal())
    }
    fn journal_path(&self) -> &Path {
        self.inner.journal_path()
    }
    fn names_journal(&self, journal: &Path) -> bool {
        self.inner.names_journal(journal)
    }
    fn journal_exists(&self) -> bool {
        self.inner.journal_exists()
    }
    fn create_journal(&self) -> io::Result<()> {
        Err(self.refusal())
    }
    fn journal_dir(&self) -> Box<dyn JournalDir> {
        Box::new(RefusingDir {
            inner: self.inner.journal_dir(),
            kind: self.kind,
        })
    }
    fn legacy_journal_path(&self) -> &Path {
        self.inner.legacy_journal_path()
    }
    fn legacy_journal_exists(&self) -> bool {
        self.inner.legacy_journal_exists()
    }
    fn read_legacy_journal(&self) -> io::Result<Vec<u8>> {
        self.inner.read_legacy_journal()
    }
    fn remove_legacy_journal(&self) -> io::Result<()> {
        Err(self.refusal())
    }
    fn lock_path(&self) -> &Path {
        self.inner.lock_path()
    }
    fn lock(&self, _mode: LockMode, _holder: &str) -> Result<Locked, LockFailure> {
        match std::fs::File::open(self.inner.lock_path()) {
            Ok(file) => match file.try_lock_shared() {
                Ok(()) => Ok(Locked::ReadOnly {
                    guard: Some(Box::new(SharedLock(file))),
                    reason: self.refusal(),
                }),
                Err(std::fs::TryLockError::WouldBlock) => Err(LockFailure::Held {
                    holder: "another holder".to_string(),
                }),
                Err(std::fs::TryLockError::Error(error)) => Err(LockFailure::Unavailable(error)),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Locked::ReadOnly {
                guard: None,
                reason: self.refusal(),
            }),
            Err(error) => Err(LockFailure::Failed(error)),
        }
    }
}
