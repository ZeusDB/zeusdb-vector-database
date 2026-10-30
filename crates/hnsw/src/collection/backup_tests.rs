//! What a backup writes, what it refuses, and what it leaves alone.
//!
//! A backup is a save of the collection into a directory that is not its
//! own: standalone at 5.0.0, naming no journal, at the next generation under
//! a snapshot of its own with the collection's snapshot as its parent, and
//! moving neither the collection's lineage nor its journal. The kill points
//! inside it are the crash matrix's to hold.

use std::io;
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::time::Duration;

use serde_json::{json, Value};
use zeusdb_vector_core::{
    pause_arm, pause_disarm, Error, FsStorage, KillPoint, LockMode, Locked, Pause, PauseAt,
    RecordState, Storage,
};

use super::journal_support::{
    copy_fixture, copy_journaled, expected_of, files_under, manifest, say_waits, sink_over,
    waited_among, FaultyStorage, RefusingStorage, Said, Segments, TempDir,
};
use super::{Collection, Declaration, IdentityRecord, ParsedRecord};
use crate::journal::{journal_path, Durability, JournalPolicy};
use crate::locks::lock_name;
use crate::SearchParams;

/// How long a check waits for a thread beside it, and the longest the
/// pause holds the backup, which is the tests' own bound on a hang.
const LIMIT: Duration = Duration::from_secs(20);

const DIM: usize = 4;

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 8, 50, 2000, vec![]).unwrap()
}

fn record(i: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: (0..DIM)
            .map(|d| ((i * 7 + d * 3) % 23) as f32 * 0.5 - 5.0)
            .collect(),
        sparse: None,
        metadata: [("i".to_string(), json!(i))].into_iter().collect(),
    }
}

fn add(collection: &Collection, range: std::ops::Range<usize>) {
    let added = collection.add_records(range.map(record).collect(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

/// Every record a collection holds, by id, with its internal id, its vector
/// and its metadata, so two collections can be held equal.
fn contents(collection: &Collection) -> Vec<(String, Option<usize>, Vec<f32>, Value)> {
    let mut ids: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    ids.sort();
    let views = collection.records(ids.clone(), true, false).unwrap();
    ids.iter()
        .zip(views)
        .map(|(id, view)| {
            let metadata: serde_json::Map<String, Value> = view
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            (
                id.clone(),
                collection.internal_id_of(id),
                view.vector.unwrap_or_default(),
                Value::Object(metadata),
            )
        })
        .collect()
}

fn hex(id: u128) -> String {
    format!("{id:032x}")
}

/// Every path under `root` with its bytes, or none for a lock file, which a
/// holder locks against reading, so a test can hold that nothing was
/// written.
fn written_under(root: &Path) -> Vec<(std::path::PathBuf, Option<Vec<u8>>)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            for entry in std::fs::read_dir(&path).unwrap() {
                stack.push(entry.unwrap().path());
            }
            out.push((path, None));
        } else {
            let locked = path.extension().is_some_and(|e| e == "zdblock");
            let bytes = (!locked).then(|| std::fs::read(&path).unwrap());
            out.push((path, bytes));
        }
    }
    out.sort();
    out
}

/// A backup of a journaled collection is a standalone directory at 5.0.0
/// that names no journal and lists no `journal` feature. Its identity is
/// the next generation under a snapshot of its own, with the collection's
/// snapshot as its parent and the sequence the journal reached, and the
/// record the call returns is the one its manifest holds. The collection's
/// lineage and journal do not move, so the next checkpoint names the same
/// parent, and the copy opens holding every record.
#[test]
fn a_backup_is_a_standalone_copy_that_moves_nothing() {
    let tmp = TempDir::new("backup-copy");
    let path = tmp.at("index.zdb");
    let target = tmp.at("copy.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    add(&collection, 0..20);
    let before = collection.identity_record();
    let status = collection.journal_status().unwrap();

    let copy: IdentityRecord = collection.backup(target.to_str().unwrap()).unwrap();
    assert_eq!(copy.identity.collection_id, before.identity.collection_id);
    assert_eq!(copy.identity.generation, before.identity.generation + 1);
    assert_eq!(copy.identity.parent, before.identity.snapshot);
    assert_ne!(copy.identity.snapshot, before.identity.snapshot);
    assert_eq!(copy.sequence, status.sequence_reached);

    let m = manifest(&target);
    assert_eq!(m["format_version"], json!("5.0.0"));
    assert!(m.get("journal").is_none(), "{m}");
    assert!(m["features"].get("journal").is_none(), "{m}");
    assert_eq!(
        m["identity"]["snapshot"],
        json!(hex(copy.identity.snapshot.unwrap()))
    );
    assert_eq!(
        m["identity"]["parent"],
        json!(hex(before.identity.snapshot.unwrap()))
    );
    assert_eq!(m["identity"]["sequence"], json!(copy.sequence));
    assert_eq!(m["saved_at"], json!(copy.saved_at.clone().unwrap()));
    assert!(!journal_path(&target).unwrap().exists());

    assert_eq!(collection.identity_record(), before, "the lineage stays");
    assert_eq!(
        collection.journal_status().unwrap(),
        status,
        "the journal stays"
    );

    let (opened, report) =
        Collection::recover(target.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert!(!report.journaled);
    assert_eq!(contents(&opened), contents(&collection));
    assert_eq!(opened.identity_record(), copy);
    drop(opened);

    collection.checkpoint().unwrap();
    let after = collection.identity_record();
    assert_eq!(after.identity.parent, before.identity.snapshot);
    assert_eq!(after.identity.generation, copy.identity.generation);
}

/// A backup taken between adds, with searches beside it, holds exactly what
/// a replay of the journal to the backup's sequence holds, record for
/// record and internal id for internal id. The backup holds the mutation
/// guard from its sync of the journal to its last write and no guard a
/// reader waits for. Held at `backup:after_sync`, with the journal synced
/// and its sequence read, it lets searches answer from this thread while an
/// add made on another thread reaches its wait at `writers` before it can
/// return, so no add lands between the sequence and the copy.
#[test]
fn a_backup_during_adds_and_searches_equals_a_replay_to_its_sequence() {
    let tmp = TempDir::new("backup-concurrent");
    let path = tmp.at("index.zdb");
    let target = tmp.at("copy.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    for batch in 0..10 {
        add(&collection, batch * 5..batch * 5 + 5);
    }
    let pause = Arc::new(Pause::new(LIMIT));
    let copy = std::thread::scope(|scope| {
        let backer = {
            let (collection, pause, target) = (&collection, Arc::clone(&pause), &target);
            scope.spawn(move || {
                pause_arm(PauseAt::Kill(KillPoint::BackupAfterSync), pause);
                let copy = collection.backup(target.to_str().unwrap());
                pause_disarm();
                copy
            })
        };
        assert_eq!(
            pause.wait_reached(LIMIT).reached(),
            Some("backup:after_sync"),
            "the backup never reached backup:after_sync"
        );
        let params = SearchParams {
            top_k: 5,
            ef: 50,
            return_vector: false,
            rerank: None,
        };
        for i in 0..20 {
            let page = collection
                .search_one(&record(i % 50).vector, None, params)
                .unwrap();
            assert_eq!(page.len(), 5);
        }
        let (said_tx, said) = mpsc::channel::<Said<()>>();
        let adding = {
            let collection = &collection;
            scope.spawn(move || {
                say_waits(&said_tx);
                for batch in 10..50 {
                    add(collection, batch * 5..batch * 5 + 5);
                }
                let _ = said_tx.send(Said::Done(()));
            })
        };
        // The backup holds `writers` alone at its pause, by the rule this
        // test holds, so that is the guard the add waits at.
        waited_among(
            &said,
            LIMIT,
            &[lock_name("writers")],
            "an add waits at the mutation guard the backup holds",
        );
        assert!(
            pause.release(),
            "the backup was not waiting once the searches had answered, so a search waited for it"
        );
        let copy = backer.join().unwrap().unwrap();
        adding.join().unwrap();
        assert_eq!(
            said.try_iter().last(),
            Some(Said::Done(())),
            "the adds returned"
        );
        copy
    });
    assert_eq!(collection.len(), 250);

    // The checkpoint with every record committed up to the backup's
    // sequence applied to it.
    let crashed = copy_journaled(&path, "replayed.zdb");
    let segments = Segments::of(&crashed);
    let read = segments.read(&expected_of(&crashed)).unwrap();
    let (replayed, _) = Collection::recover_with(
        crashed.to_str().unwrap(),
        None,
        JournalPolicy::CheckpointOnly,
    )
    .unwrap();
    for live in read
        .records
        .iter()
        .filter(|live| live.record.sequence <= copy.sequence)
    {
        assert_eq!(live.state, RecordState::Committed);
        let operation = zeusdb_vector_core::Operation::decode(&live.record, DIM, "").unwrap();
        replayed.apply(operation).unwrap();
    }
    let (backup, _) =
        Collection::recover(target.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert_eq!(
        backup.len(),
        50,
        "the copy holds the records the collection held as the backup began"
    );
    assert_eq!(contents(&backup), contents(&replayed));
}

/// Five targets are refused before anything is written: the directory the
/// collection's journal sits beside, the directory a read-only collection
/// was read from, a directory with a journal of either version beside it,
/// a path that is not an index directory, and a directory another holder
/// locks.
#[test]
fn a_backup_refuses_five_targets_before_writing() {
    let tmp = TempDir::new("backup-refused");
    let path = tmp.at("index.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    add(&collection, 0..5);

    // A journaled directory of another collection, and a directory with a
    // journal of format version 1 beside it.
    let other = tmp.at("other.zdb");
    {
        let journaled = Collection::build(declaration(), None);
        journaled
            .journal_to(other.to_str().unwrap(), Durability::PerCall)
            .unwrap();
    }
    let legacy = tmp.at("legacy.zdb");
    Collection::build(declaration(), None)
        .save(legacy.to_str().unwrap())
        .unwrap();
    std::fs::write(
        FsStorage::at(&legacy).unwrap().legacy_journal_path(),
        b"a journal",
    )
    .unwrap();
    // A file, a folder of other files, and a directory another holder locks.
    let file = tmp.at("notes.zdb");
    std::fs::write(&file, b"mine").unwrap();
    let folder = tmp.at("folder");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("notes.txt"), b"mine").unwrap();
    let locked = tmp.at("locked.zdb");
    let holder = FsStorage::at(&locked).unwrap();
    let _held = match holder.lock(LockMode::Exclusive, "another holder") {
        Ok(Locked::Writable(guard)) => guard,
        other => panic!("{other:?}"),
    };

    let before = written_under(&tmp.0);
    let backup = |target: &Path| collection.backup(target.to_str().unwrap());
    assert!(matches!(
        backup(&path),
        Err(Error::BackupIntoOwnDirectory { .. })
    ));
    assert!(matches!(
        backup(&other),
        Err(Error::BackupTargetJournaled { .. })
    ));
    assert!(matches!(
        backup(&legacy),
        Err(Error::BackupTargetJournaled { .. })
    ));
    match backup(&file) {
        Err(Error::TargetNotAnIndex { holds, .. }) => assert_eq!(holds, "a file"),
        other => panic!("{other:?}"),
    }
    match backup(&folder) {
        Err(error @ Error::TargetNotAnIndex { .. }) => {
            let message = error.to_string();
            assert!(
                message.contains("1 entry and no manifest.json"),
                "{message}"
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        backup(&locked),
        Err(Error::CollectionLocked { .. })
    ));
    assert_eq!(written_under(&tmp.0), before, "nothing was written");

    // The directory a read-only collection was read from is its own.
    drop(collection);
    let storage = RefusingStorage::at(&path, io::ErrorKind::PermissionDenied);
    let (read_only, _) = crate::persistence::load_from(
        &storage,
        path.to_str().unwrap(),
        None,
        JournalPolicy::Replay(Durability::PerCall),
    )
    .unwrap();
    let before = written_under(&tmp.0);
    assert!(matches!(
        read_only.backup(path.to_str().unwrap()),
        Err(Error::BackupIntoOwnDirectory { .. })
    ));
    assert_eq!(written_under(&tmp.0), before, "nothing was written");
}

/// A collection with no journal and one read only each back up, at the
/// sequence each holds, and the read-only one keeps refusing every change.
#[test]
fn a_backup_works_unjournaled_and_read_only() {
    let tmp = TempDir::new("backup-kinds");
    let plain = Collection::build(declaration(), None);
    add(&plain, 0..10);
    let copy = plain
        .backup(tmp.at("plain-copy.zdb").to_str().unwrap())
        .unwrap();
    assert_eq!(copy.identity.generation, 1);
    assert_eq!(copy.identity.parent, None);
    assert_eq!(copy.sequence, 0);
    assert_eq!(plain.identity().generation, 0, "the lineage stays");

    let path = tmp.at("index.zdb");
    {
        let journaled = Collection::build(declaration(), None);
        journaled
            .journal_to(path.to_str().unwrap(), Durability::PerCall)
            .unwrap();
        add(&journaled, 0..10);
        add(&journaled, 10..12);
    }
    let storage = RefusingStorage::at(&path, io::ErrorKind::ReadOnlyFilesystem);
    let (read_only, report) = crate::persistence::load_from(
        &storage,
        path.to_str().unwrap(),
        None,
        JournalPolicy::Replay(Durability::PerCall),
    )
    .unwrap();
    assert!(report.read_only.is_some());
    let target = tmp.at("read-only-copy.zdb");
    let copy = read_only.backup(target.to_str().unwrap()).unwrap();
    assert_eq!(copy.sequence, 14, "the sequence the replay reached");
    assert_eq!(copy.sequence, read_only.journal_sequence());
    let (opened, _) =
        Collection::recover(target.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert_eq!(contents(&opened), contents(&read_only));
    assert_eq!(
        read_only
            .add_records(vec![record(50)], vec![], false)
            .total_errors,
        1
    );
}

/// A backup into a path holding a directory of an older format replaces it
/// whole with a directory at 5.0.0.
#[test]
fn a_backup_replaces_a_directory_of_an_older_format() {
    let tmp = TempDir::new("backup-older");
    let fixture = copy_fixture("v4.0.0-journaled-text", &tmp.0);
    let target = tmp.at("older.zdb");
    super::journal_support::copy_tree(&fixture, &target);
    assert_eq!(manifest(&target)["format_version"], json!("4.0.0"));
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    collection.backup(target.to_str().unwrap()).unwrap();
    assert_eq!(manifest(&target)["format_version"], json!("5.0.0"));
    let (opened, _) =
        Collection::recover(target.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert_eq!(contents(&opened), contents(&collection));
}

/// A backup whose sync of the journal fails writes the copy from memory and
/// leaves the fault, since it writes no directory of the collection's own,
/// and the next checkpoint clears it.
#[test]
fn a_failed_sync_before_a_backup_leaves_the_fault() {
    let tmp = TempDir::new("backup-fault");
    let path = tmp.at("index.zdb");
    let collection = Collection::build(declaration(), None);
    let storage = FaultyStorage::at(&path);
    let sink = sink_over(
        &storage,
        &path,
        collection.collection_id(),
        Durability::NoSync,
    );
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..10);
    storage.faults.syncs.arm(0, 1);
    let target = tmp.at("copy.zdb");
    collection.backup(target.to_str().unwrap()).unwrap();
    assert!(storage.faults.syncs.spent());
    let (opened, _) =
        Collection::recover(target.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert_eq!(contents(&opened), contents(&collection));
    let refused = collection.add_records(vec![record(10)], vec![], false);
    assert!(
        refused.errors[0].contains("The journal's commit failed"),
        "{:?}",
        refused.errors
    );
    collection.checkpoint().unwrap();
    add(&collection, 10..11);
}

/// A save, a journal and a backup refuse a target that is a file or a
/// folder holding entries and no `manifest.json`, before anything is
/// written, and take an empty folder and a directory an earlier save wrote.
#[test]
fn a_save_refuses_a_target_that_is_not_an_index_directory() {
    let tmp = TempDir::new("save-foreign");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..5);

    let file = tmp.at("index.zdb");
    std::fs::write(&file, b"mine").unwrap();
    let folder = tmp.at("folder");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("notes.txt"), b"mine").unwrap();
    let before = files_under(&[&tmp.0]);
    for target in [&file, &folder] {
        let target = target.to_str().unwrap();
        assert!(matches!(
            collection.save(target),
            Err(Error::TargetNotAnIndex { .. })
        ));
        assert!(matches!(
            collection.journal_to(target, Durability::PerCall),
            Err(Error::TargetNotAnIndex { .. })
        ));
        assert!(matches!(
            collection.backup(target),
            Err(Error::TargetNotAnIndex { .. })
        ));
    }
    assert_eq!(files_under(&[&tmp.0]), before, "nothing was written");
    assert!(collection.journal_status().is_none());

    let empty = tmp.at("empty.zdb");
    std::fs::create_dir(&empty).unwrap();
    collection.save(empty.to_str().unwrap()).unwrap();
    collection.save(empty.to_str().unwrap()).unwrap();
    assert_eq!(manifest(&empty)["format_version"], json!("5.0.0"));
}
