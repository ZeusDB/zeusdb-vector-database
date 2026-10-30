//! A load from a medium this process cannot write.
//!
//! A load decides at its first write, before it has changed anything, and a
//! refusal the system gives for a medium that cannot be written or for a
//! missing permission makes a load of a journaled directory read only. The
//! medium is a storage whose every write meets that refusal, under both of
//! the kinds a system gives, and on Linux the real permissions as well.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::json;
use zeusdb_vector_core::{
    Dir, Error, FsStorage, JournalDir, LockFailure, LockMode, Locked, Staged, Storage,
};

use super::journal_support::{
    copy_fixture, files_under, fixture_expected, holds_fixture, manifest, RefusingStorage, TempDir,
    FIXTURES,
};
use super::{Collection, Declaration, ParsedRecord};
use crate::journal::{Durability, JournalPolicy};

const KINDS: [io::ErrorKind; 2] = [
    io::ErrorKind::ReadOnlyFilesystem,
    io::ErrorKind::PermissionDenied,
];

fn load_through(
    storage: &dyn Storage,
    path: &Path,
    policy: JournalPolicy,
) -> Result<(Collection, crate::Recovery), Error> {
    crate::persistence::load_from(storage, path.to_str().unwrap(), None, policy)
}

fn replay() -> JournalPolicy {
    JournalPolicy::Replay(Durability::PerCall)
}

fn record(i: usize, dim: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("new{i}"),
        vector: vec![0.5; dim],
        sparse: None,
        metadata: HashMap::new(),
    }
}

/// Every change a collection read only refuses, each with
/// `CollectionReadOnly`.
fn refuses_every_change(collection: &Collection, own: &Path, elsewhere: &Path, dim: usize) {
    let read_only = |result: Result<(), Error>, what: &str| match result {
        Err(Error::CollectionReadOnly { directory, .. }) => {
            assert_eq!(directory, own.to_str().unwrap(), "{what}")
        }
        other => panic!("{what}: expected the refusal, got {other:?}"),
    };
    let added = collection.add_records(vec![record(0, dim), record(1, dim)], vec![], false);
    assert_eq!(added.total_errors, 2);
    assert!(
        added.errors.iter().all(|e| e.contains("cannot write")),
        "{:?}",
        added.errors
    );
    let overwritten = collection.add_records(vec![record(0, dim)], vec![], true);
    assert_eq!(overwritten.total_errors, 1);
    let held = collection
        .ids()
        .iter()
        .next()
        .map(|(_, id)| id.to_string())
        .unwrap();
    read_only(
        collection
            .remove_points(std::slice::from_ref(&held))
            .map(|_| ()),
        "remove",
    );
    read_only(
        collection
            .update_metadata(&held, HashMap::from([("x".to_string(), json!(1))]))
            .map(|_| ()),
        "update",
    );
    read_only(collection.clear().map(|_| ()), "clear");
    read_only(collection.compact().map(|_| ()), "compact");
    read_only(
        collection
            .add_metadata(HashMap::from([("k".to_string(), "v".to_string())]))
            .map(|_| ()),
        "add_metadata",
    );
    read_only(collection.checkpoint(), "checkpoint");
    read_only(
        collection.journal_to(elsewhere.to_str().unwrap(), Durability::PerCall),
        "journal_to",
    );
    read_only(
        collection.save(own.to_str().unwrap()),
        "save into its own directory",
    );
    assert!(collection.contains(&held));
}

/// R1. Each fixture opens read only through a medium that refuses writes,
/// under both kinds, with every record it recorded. No file of the
/// directory or its journal changes, and no lock file is created. Every
/// change is refused, and the collection reports no journal. A save into
/// another directory writes an unjournaled 5.0.0 copy, which opens
/// writable with the same records.
#[test]
fn a_fixture_on_a_medium_that_refuses_writes_opens_read_only() {
    for kind in KINDS {
        for name in FIXTURES {
            let temp = TempDir::new("read-only-tests");
            let path = copy_fixture(name, &temp.0);
            let folder = path.parent().unwrap().to_path_buf();
            let before = files_under(&[&folder]);
            let storage = RefusingStorage::at(&path, kind);
            let (collection, report) = load_through(&storage, &path, replay()).unwrap();
            let expected = fixture_expected(name);
            let format = expected["format_version"].as_str().unwrap();

            holds_fixture(&collection, name).unwrap();
            assert!(report.read_only.is_some(), "{name}");
            assert_eq!(report.upgraded_from, None, "{name}: nothing was upgraded");
            assert_eq!(report.format_version, format);
            let warning = report.read_only_warning(path.to_str().unwrap()).unwrap();
            assert!(
                warning.contains(&format!("stays at format {format}")),
                "{warning}"
            );
            assert_eq!(
                files_under(&[&folder]),
                before,
                "{name}: nothing on the medium changed"
            );
            assert_eq!(collection.journal_status(), None);
            assert!(!collection.stats().keys().any(|k| k.starts_with("journal_")));

            let dim = expected["dim"].as_u64().unwrap() as usize;
            refuses_every_change(&collection, &path, &temp.at("journal-elsewhere.zdb"), dim);

            let copy = temp.at("copy.zdb");
            collection.save(copy.to_str().unwrap()).unwrap();
            let m = manifest(&copy);
            assert_eq!(m["format_version"], json!("5.0.0"));
            assert!(m.get("journal").is_none());
            assert_eq!(
                m["identity"]["sequence"],
                json!(expected["torn_sequence"].as_u64().unwrap() - 1),
                "the sequence the replayed state holds"
            );
            drop(collection);
            let (writable, report) =
                Collection::recover(copy.to_str().unwrap(), None, Durability::PerCall).unwrap();
            assert_eq!(report.read_only, None);
            holds_fixture(&writable, name).unwrap();
            let added = writable.add_records(vec![record(9, dim)], vec![], false);
            assert_eq!(added.total_errors, 0, "{:?}", added.errors);
        }
    }
}

/// R2. A 5.0.0 journaled directory with a torn tail and a roll a crash
/// interrupted opens read only with every committed record, and the torn
/// tail and the torn segment are read around and left as they are.
#[test]
fn a_journal_of_segments_with_a_torn_tail_and_a_torn_roll_opens_read_only() {
    let temp = TempDir::new("read-only-tests");
    let path = temp.at("segments.zdb");
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap();
    let collection = Collection::build(declaration, None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    let records: Vec<ParsedRecord> = (0..6).map(|i| record(i, 2)).collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let held = collection.len();
    drop(collection);

    let storage = FsStorage::at(&path).unwrap();
    let dir = storage.journal_dir();
    let numbers = dir.segments().unwrap();
    let active = *numbers.last().unwrap();
    let whole = dir.read_segment(active).unwrap();
    // The sequence after six inserts and their commit, after the checkpoint
    // at zero.
    let next = 1 + 6 + 1;
    let journal = crate::journal::journal_path(&path).unwrap();
    let unchanged = |before: &Vec<(PathBuf, Vec<u8>)>| {
        for kind in KINDS {
            let refusing = RefusingStorage::at(&path, kind);
            let (collection, report) = load_through(&refusing, &path, replay()).unwrap();
            assert!(report.read_only.is_some());
            assert_eq!(collection.len(), held);
            drop(collection);
            assert_eq!(&files_under(&[&path, &journal]), before);
        }
    };

    // Half a record after the last commit, which the reader reads around.
    let mut torn = Vec::new();
    zeusdb_vector_core::encode_journal_record(
        next,
        zeusdb_vector_core::OperationKind::Clear,
        &[],
        &mut torn,
    );
    let mut bytes = whole.clone();
    bytes.extend_from_slice(&torn[..torn.len() / 2]);
    std::fs::write(dir.segment_path(active), &bytes).unwrap();
    unchanged(&files_under(&[&path, &journal]));

    // A segment numbered for the next sequence whose creation stopped
    // inside its header, after a segment that ends at a commit.
    std::fs::write(dir.segment_path(active), &whole).unwrap();
    std::fs::write(dir.segment_path(next), [0u8; 40]).unwrap();
    unchanged(&files_under(&[&path, &journal]));
}

/// R3. The checkpoint alone of each fixture, and a directory with no
/// journal, open on such a medium as they always have: nothing ties their
/// collection to the directory, so it is not read only.
#[test]
fn a_checkpoint_alone_and_an_unjournaled_directory_open_as_they_always_have() {
    for name in FIXTURES {
        let temp = TempDir::new("read-only-tests");
        let path = copy_fixture(name, &temp.0);
        let storage = RefusingStorage::at(&path, KINDS[0]);
        let (collection, report) =
            load_through(&storage, &path, JournalPolicy::CheckpointOnly).unwrap();
        assert_eq!(report.read_only, None, "{name}");
        assert_eq!(
            collection.len() as u64,
            fixture_expected(name)["checkpoint_records"]
                .as_u64()
                .unwrap()
        );
    }
    let temp = TempDir::new("read-only-tests");
    let path = temp.at("plain.zdb");
    let collection = Collection::build(
        Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    );
    let records: Vec<ParsedRecord> = (0..3).map(|i| record(i, 2)).collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    collection.save(path.to_str().unwrap()).unwrap();
    drop(collection);
    let storage = RefusingStorage::at(&path, KINDS[1]);
    let (collection, report) = load_through(&storage, &path, replay()).unwrap();
    assert_eq!(report.read_only, None);
    assert_eq!(collection.len(), 3);
    let added = collection.add_records(vec![record(7, 2)], vec![], false);
    assert_eq!(
        added.total_errors, 0,
        "an unjournaled collection takes changes"
    );
}

/// Where a storage of the filesystem fails, with the kind it fails with.
#[derive(Clone, Copy, Debug)]
enum FailAt {
    /// The probe of a checkpoint's first write.
    Probe(io::ErrorKind),
    /// The journal's directory, once the probe has passed.
    CreateJournal(io::ErrorKind),
    /// Opening the journal's last segment for writing.
    OpenSegment(io::ErrorKind),
    /// Deleting the journal of format version 1 once an upgrade commits.
    RemoveLegacy(io::ErrorKind),
}

/// A storage of the filesystem, its lock among it, that fails at one write.
#[derive(Debug)]
struct Fails {
    inner: FsStorage,
    at: FailAt,
}

fn failure(kind: io::ErrorKind) -> io::Error {
    io::Error::new(kind, "the device refused it")
}

/// The journal's directory of a `Fails` that fails at opening a segment.
#[derive(Debug)]
struct OpenFails {
    inner: Box<dyn JournalDir>,
    kind: io::ErrorKind,
}

impl JournalDir for OpenFails {
    fn path(&self) -> &Path {
        self.inner.path()
    }
    fn segments(&self) -> io::Result<Vec<u64>> {
        self.inner.segments()
    }
    fn read_segment(&self, number: u64) -> io::Result<Vec<u8>> {
        self.inner.read_segment(number)
    }
    fn create_segment(&self, number: u64) -> io::Result<Box<dyn zeusdb_vector_core::JournalFile>> {
        self.inner.create_segment(number)
    }
    fn open_segment(&self, _number: u64) -> io::Result<Box<dyn zeusdb_vector_core::JournalFile>> {
        Err(failure(self.kind))
    }
    fn remove_segment(&self, number: u64) -> io::Result<()> {
        self.inner.remove_segment(number)
    }
    fn sync(&self) -> io::Result<()> {
        self.inner.sync()
    }
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
    fn create_quarantine(&self, folder: &str) -> io::Result<()> {
        self.inner.create_quarantine(folder)
    }
    fn quarantine(&self) -> Box<dyn Dir> {
        self.inner.quarantine()
    }
    fn quarantine_folders(&self) -> io::Result<Vec<String>> {
        self.inner.quarantine_folders()
    }
    fn sync_quarantine(&self, folder: &str) -> io::Result<()> {
        self.inner.sync_quarantine(folder)
    }
}

impl Storage for Fails {
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
    fn foreign_target(&self) -> Option<String> {
        self.inner.foreign_target()
    }
    fn probe_writable(&self) -> io::Result<()> {
        match self.at {
            FailAt::Probe(kind) => Err(failure(kind)),
            _ => self.inner.probe_writable(),
        }
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
        match self.at {
            FailAt::CreateJournal(kind) => Err(failure(kind)),
            _ => self.inner.create_journal(),
        }
    }
    fn journal_dir(&self) -> Box<dyn JournalDir> {
        match self.at {
            FailAt::OpenSegment(kind) => Box::new(OpenFails {
                inner: self.inner.journal_dir(),
                kind,
            }),
            _ => self.inner.journal_dir(),
        }
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
        match self.at {
            FailAt::RemoveLegacy(kind) => Err(failure(kind)),
            _ => self.inner.remove_legacy_journal(),
        }
    }
    fn sync_in_place(&self) -> io::Result<()> {
        self.inner.sync_in_place()
    }
    fn cut_legacy_journal(&self, len: u64) -> io::Result<()> {
        self.inner.cut_legacy_journal(len)
    }
    fn lock_path(&self) -> &Path {
        self.inner.lock_path()
    }
    fn lock(&self, mode: LockMode, holder: &str) -> Result<Locked, LockFailure> {
        self.inner.lock(mode, holder)
    }
}

/// R4. A failure of another kind at the load's first write refuses the
/// load rather than opening it read only, at the upgrade's probe and at the
/// journal's last segment. A refusal of writes at a write after the first
/// refuses the load too, since a medium that refuses part way must not
/// leave half an upgrade read as a whole one.
#[test]
fn a_failure_that_is_not_a_refusal_of_writes_refuses_the_load() {
    let refused = |name: &str, at: FailAt| {
        let temp = TempDir::new("read-only-tests");
        let path = copy_fixture(name, &temp.0);
        let storage = Fails {
            inner: FsStorage::at(&path).unwrap(),
            at,
        };
        match load_through(&storage, &path, replay()) {
            Err(error) => {
                assert!(
                    error.to_string().contains("the device refused it"),
                    "{error}"
                );
                assert_eq!(manifest(&path)["format_version"], json!("3.0.0"));
            }
            Ok((_, report)) => panic!("{at:?}: expected a refusal, got {report:?}"),
        }
    };
    refused(FIXTURES[0], FailAt::Probe(io::ErrorKind::Other));
    refused(FIXTURES[0], FailAt::CreateJournal(io::ErrorKind::Other));
    refused(
        FIXTURES[0],
        FailAt::CreateJournal(io::ErrorKind::ReadOnlyFilesystem),
    );
    refused(
        FIXTURES[0],
        FailAt::CreateJournal(io::ErrorKind::PermissionDenied),
    );

    // A 5.0.0 directory whose last segment will not open for another reason.
    let temp = TempDir::new("read-only-tests");
    let path = copy_fixture(FIXTURES[0], &temp.0);
    let (upgraded, _) =
        Collection::recover(path.to_str().unwrap(), None, Durability::PerCall).unwrap();
    drop(upgraded);
    let storage = Fails {
        inner: FsStorage::at(&path).unwrap(),
        at: FailAt::OpenSegment(io::ErrorKind::Other),
    };
    match load_through(&storage, &path, replay()) {
        Err(Error::JournalIoFailed { error, .. }) => {
            assert!(error.contains("the device refused it"), "{error}")
        }
        other => panic!("expected the load refused, got {:?}", other.map(|_| ())),
    }
    for kind in KINDS {
        let storage = Fails {
            inner: FsStorage::at(&path).unwrap(),
            at: FailAt::OpenSegment(kind),
        };
        let (_, report) = load_through(&storage, &path, replay()).unwrap();
        assert!(
            report.read_only.is_some(),
            "{kind:?} at the first write is read only"
        );
    }
}

/// R5. A writer holding the directory exclusive, here a journaled collection
/// in this process, refuses a load through a medium that refuses writes with
/// `CollectionLocked`, since the load still takes the shared lock it can.
#[test]
fn a_held_directory_refuses_a_read_only_load() {
    let temp = TempDir::new("read-only-tests");
    let path = temp.at("held.zdb");
    let collection = Collection::build(
        Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    );
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    let storage = RefusingStorage::at(&path, KINDS[0]);
    assert!(matches!(
        load_through(&storage, &path, replay()),
        Err(Error::CollectionLocked { .. })
    ));
    drop(collection);
    let (opened, report) = load_through(&storage, &path, replay()).unwrap();
    assert!(report.read_only.is_some());
    assert_eq!(opened.len(), 0);
}

/// R6. On Linux, the real permissions: the folder holding a fixture at mode
/// 0555 and every file at 0444. Skipped where the process can write there
/// regardless, which is a process running as root.
#[cfg(unix)]
#[test]
fn a_fixture_under_real_read_only_permissions_opens_read_only() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new("read-only-tests");
    let path = copy_fixture(FIXTURES[2], &temp.0);
    let folder = path.parent().unwrap().to_path_buf();
    let set = |mode_dir: u32, mode_file: u32| {
        for (file, _) in files_under(&[&folder]) {
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(mode_file)).unwrap();
        }
        let mut dirs = vec![folder.clone()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    dirs.push(entry.path());
                }
            }
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode_dir)).unwrap();
        }
    };
    set(0o555, 0o444);
    if std::fs::write(folder.join("probe"), b"x").is_ok() {
        let _ = std::fs::remove_file(folder.join("probe"));
        set(0o755, 0o644);
        eprintln!("skipped: this process writes through read-only permissions");
        return;
    }
    let before = files_under(&[&folder]);
    let result = Collection::recover(path.to_str().unwrap(), None, Durability::PerCall);
    let after = files_under(&[&folder]);
    set(0o755, 0o644);
    let (collection, report) = result.unwrap();
    assert!(report.read_only.is_some());
    holds_fixture(&collection, FIXTURES[2]).unwrap();
    assert_eq!(after, before);
}

/// R7. A journal of format version 1 the system will not delete is left
/// with a warning once its upgrade has committed, and the next load that can
/// delete it does, since the directory holds every record in it. The
/// refusal is injected through the load's storage, under both kinds.
#[test]
fn a_version_1_journal_the_system_will_not_delete_is_left_after_the_upgrade() {
    let name = FIXTURES[0];
    for kind in KINDS {
        let temp = TempDir::new("read-only-tests");
        let path = copy_fixture(name, &temp.0);
        let wal = FsStorage::at(&path)
            .unwrap()
            .legacy_journal_path()
            .to_path_buf();
        let storage = Fails {
            inner: FsStorage::at(&path).unwrap(),
            at: FailAt::RemoveLegacy(kind),
        };
        let (upgraded, report) = load_through(&storage, &path, replay()).unwrap();
        assert!(report.upgraded_from.is_some());
        assert_eq!(report.read_only, None, "the upgrade had already committed");
        holds_fixture(&upgraded, name).unwrap();
        assert_eq!(manifest(&path)["format_version"], json!("5.0.0"));
        assert!(wal.exists(), "the file the system would not delete is left");
        drop(upgraded);

        let (reopened, report) =
            Collection::recover(path.to_str().unwrap(), None, Durability::PerCall).unwrap();
        assert_eq!(report.upgraded_from, None);
        holds_fixture(&reopened, name).unwrap();
        assert!(!wal.exists(), "the next load deleted it");
    }
}

/// R7 on Windows, with the system's own refusal: the file held open by
/// another handle that does not share deletion. The read-only attribute
/// does not block a deletion through this toolchain's standard library,
/// which ignores it.
#[cfg(windows)]
#[test]
fn a_version_1_journal_held_open_on_windows_is_left_after_the_upgrade() {
    use std::os::windows::fs::OpenOptionsExt;
    let name = FIXTURES[0];
    let temp = TempDir::new("read-only-tests");
    let path = copy_fixture(name, &temp.0);
    let wal = FsStorage::at(&path)
        .unwrap()
        .legacy_journal_path()
        .to_path_buf();
    // FILE_SHARE_READ alone, so no other handle may delete the file.
    let held = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(&wal)
        .unwrap();
    let (upgraded, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert!(report.upgraded_from.is_some());
    holds_fixture(&upgraded, name).unwrap();
    drop(upgraded);
    assert!(wal.exists(), "the handle kept the file");
    drop(held);
    let (reopened, _) =
        Collection::recover(path.to_str().unwrap(), None, Durability::PerCall).unwrap();
    holds_fixture(&reopened, name).unwrap();
    assert!(!wal.exists());
}
