//! The syncs of a directory's own entries, which a file's sync does not make
//! durable: the journal's directory after a segment is created or deleted,
//! the folder an index directory and its journal sit in, a repair's
//! quarantine, and every directory a save writes.
//!
//! A sync that fails is held as a failed sync is. A live journal takes
//! nothing more until a checkpoint commits, and the call that met the
//! failure raises. A load, a save and the opening of a journal that meet one
//! fail. The faults are armed on the journal's directory through
//! `FaultyStorage`, and on every other directory by its path through the
//! storage's own test hook, so the save, `journal_to` and the load's own
//! syncs meet them where they run.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::Path;

use serde_json::json;
use zeusdb_vector_core::{
    directory_sync_faults, set_segment_roll_bytes, Error, InsertParts, Operation, SparseVector,
    JOURNAL_RECORD_OVERHEAD_BYTES, SEGMENT_HEADER_BYTES,
};
use zeusdb_vector_sparse::SparseConfig;

use super::journal_support::{
    copy_fixture, damaged_journal, holds_fixture, sink_over, FaultyStorage, Segments, TempDir,
    FIXTURES,
};
use super::{Collection, Declaration, ParsedRecord, SparseHalf};
use crate::journal::{journal_path, Durability, JournalPolicy};

fn declaration() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap()
}

fn record(i: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vec![i as f32 * 0.5, 1.0],
        sparse: None,
        metadata: HashMap::from([("i".to_string(), json!(i))]),
    }
}

fn add(collection: &Collection, range: std::ops::Range<usize>) {
    let records: Vec<ParsedRecord> = range.map(record).collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn ids(collection: &Collection) -> Vec<String> {
    let mut out: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    out.sort();
    out
}

fn recover(path: &Path) -> Result<(Collection, crate::Recovery), Error> {
    Collection::recover(path.to_str().unwrap(), None, Durability::PerCall)
}

fn load_through(
    storage: &FaultyStorage,
    path: &Path,
    policy: JournalPolicy,
) -> Result<(Collection, crate::Recovery), Error> {
    crate::persistence::load_from(storage, path.to_str().unwrap(), None, policy)
}

fn replay() -> JournalPolicy {
    JournalPolicy::Replay(Durability::PerCall)
}

fn repair() -> JournalPolicy {
    JournalPolicy::Repair(Durability::PerCall)
}

/// The words of a failure, or the panic of a test that names what it
/// expected where the call succeeded.
fn failure_of<T>(result: Result<T, Error>, what: &str) -> String {
    match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("{what} fails, and it succeeded"),
    }
}

/// The words `add()` names a refused record `r<i>` with, for the refusal
/// `error`.
fn refused_as(i: usize, error: &str) -> String {
    format!("Vector r{i}: RuntimeError: {error}")
}

/// The words the writer refuses every append with once it takes nothing
/// more, for the journal beside `path` and the reason `reason`.
fn append_refusal(path: &Path, reason: &str) -> String {
    format!(
        "Failed to append to the journal {}: {reason}",
        journal_path(path).unwrap().display()
    )
}

/// The refusal of a call of one record, or the panic of a test that names
/// what it expected where the record went in.
fn refused_add(collection: &Collection, i: usize, what: &str) -> String {
    let refused = collection.add_records(vec![record(i)], vec![], false);
    assert_eq!(
        refused.total_errors, 1,
        "{what} is refused: {:?}",
        refused.errors
    );
    refused.errors[0].clone()
}

/// A collection journaled to `path` through `storage`, saved, holding no
/// record.
fn journaled_through(storage: &FaultyStorage, path: &Path) -> Collection {
    let collection = Collection::build(declaration(), None);
    let sink = sink_over(
        storage,
        path,
        collection.collection_id(),
        Durability::PerCall,
    );
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    collection
}

/// A collection journaled to `path` the way `journal_to` journals one.
fn journaled(path: &Path) -> Collection {
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    collection
}

/// The bytes a call of one record adds to a segment, being its insert and
/// its commit.
fn call_bytes() -> u64 {
    let probe = record(1);
    let mut payload = Vec::new();
    Operation::encode_insert(
        &InsertParts {
            id: &probe.id,
            vector: &probe.vector,
            metadata: &probe.metadata,
            sparse: None,
        },
        &mut payload,
    );
    (payload.len() + 8 + 2 * JOURNAL_RECORD_OVERHEAD_BYTES) as u64
}

/// A directory journaled at `path` holding five records, checkpointed, and
/// two more, with the segment the checkpoint deleted put back, as a crash
/// inside the checkpoint's deletion leaves it. The load that opens it
/// deletes that stale segment. Returns the ids the directory holds.
fn with_a_stale_segment(path: &Path) -> Vec<String> {
    let collection = journaled(path);
    add(&collection, 0..5);
    let old = Segments::of(path);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 5..7);
    let before = ids(&collection);
    drop(collection);
    std::fs::write(&old.files[0], &old.bytes[0]).unwrap();
    assert_eq!(Segments::of(path).numbers.len(), 2);
    before
}

/// The syncs of `directory` that a run of `run` asks, armed to let every one
/// through, so a later run can fail the last of them.
fn syncs_asked(directory: &Path, run: impl FnOnce()) -> usize {
    directory_sync_faults::arm(directory, usize::MAX, 0);
    run();
    let asked = directory_sync_faults::asked_of(directory);
    directory_sync_faults::disarm(directory);
    asked
}

/// Run `run` with the sync of `directory` that comes after `pass` others
/// failing, and say whether that sync was asked.
fn failing_sync<T>(directory: &Path, pass: usize, run: impl FnOnce() -> T) -> (T, bool) {
    directory_sync_faults::arm(directory, pass, 1);
    let ran = run();
    let spent = directory_sync_faults::spent(directory);
    directory_sync_faults::disarm(directory);
    (ran, spent)
}

// ============================================================================
// A LIVE JOURNAL
// ============================================================================

/// A sync of the journal's directory that fails once a roll by size has
/// created the next segment stops the index until a checkpoint commits,
/// since a record in that segment may lose its name on a power loss. The
/// call whose commit rolled returns with its records committed, every call
/// after it is refused in words that name the directory's sync and
/// `checkpoint()`, and the checkpoint lets the index take records again.
#[test]
fn a_failed_directory_sync_at_a_roll_by_size_stops_the_index_until_a_checkpoint() {
    for durability in [Durability::PerCall, Durability::NoSync] {
        let temp = TempDir::new("directory-sync-tests");
        let path = temp.at("rolled.zdb");
        let storage = FaultyStorage::at(&path);
        let faults = storage.faults.clone();
        let collection = Collection::build(declaration(), None);
        set_segment_roll_bytes(SEGMENT_HEADER_BYTES as u64 + 2 * call_bytes() + 1);
        let sink = sink_over(&storage, &path, collection.collection_id(), durability);
        set_segment_roll_bytes(0);
        collection.attach_sink(Box::new(sink));
        collection.save(path.to_str().unwrap()).unwrap();
        add(&collection, 1..3);
        faults.dir_syncs.arm(0, 1);
        add(&collection, 3..4);
        assert!(
            faults.dir_syncs.spent(),
            "{durability:?}: the directory's sync failed"
        );
        let numbers = Segments::of(&path).numbers;
        assert_eq!(numbers.len(), 2, "{durability:?}: the next segment exists");
        let reason = format!(
            "the journal's directory could not be synced once the segment numbered {} was \
             created (injected: the device refused it), so the journal takes nothing more until \
             checkpoint() succeeds.",
            numbers[1]
        );
        assert_eq!(
            refused_add(
                &collection,
                4,
                &format!("{durability:?}: the call after a failed directory sync")
            ),
            refused_as(4, &append_refusal(&path, &reason)),
            "{durability:?}"
        );
        collection.checkpoint().unwrap();
        add(&collection, 5..6);
        let before = ids(&collection);
        assert_eq!(before, ["r1", "r2", "r3", "r5"]);
        drop(collection);
        let (recovered, _) = recover(&path).unwrap();
        assert_eq!(ids(&recovered), before);
    }
}

/// A sync of the journal's directory that fails in a checkpoint's roll,
/// once the segment naming the checkpoint is created or once the segments
/// it holds are deleted, makes the checkpoint raise and stops the index
/// until a checkpoint commits.
#[test]
fn a_checkpoint_whose_roll_fails_its_directory_sync_raises_and_stops_the_index() {
    for (passed, what) in [(0, "its segment's creation"), (1, "its deletions")] {
        let temp = TempDir::new("directory-sync-tests");
        let path = temp.at("checkpoint.zdb");
        let storage = FaultyStorage::at(&path);
        let faults = storage.faults.clone();
        let collection = journaled_through(&storage, &path);
        add(&collection, 0..2);
        let reason = if passed == 0 {
            "the journal's directory could not be synced once the segment naming the checkpoint \
             at sequence 3 was created (injected: the device refused it), so the journal takes \
             nothing more until checkpoint() succeeds."
        } else {
            "the journal's directory could not be synced once the checkpoint at sequence 3 \
             deleted the segments it holds (injected: the device refused it), so the journal \
             takes nothing more until checkpoint() succeeds."
        };
        faults.dir_syncs.arm(passed, 1);
        let checkpointed = failure_of(collection.checkpoint(), what);
        assert!(
            faults.dir_syncs.spent(),
            "{what}: the directory's sync failed"
        );
        assert_eq!(
            checkpointed,
            format!(
                "Failed to roll the journal {}: {reason}",
                journal_path(&path).unwrap().display()
            ),
            "{what}"
        );
        assert_eq!(
            refused_add(&collection, 2, &format!("{what}: the call after it")),
            refused_as(2, &append_refusal(&path, reason)),
            "{what}"
        );
        collection.checkpoint().unwrap();
        add(&collection, 3..4);
        let before = ids(&collection);
        drop(collection);
        let (recovered, _) = recover(&path).unwrap();
        assert_eq!(ids(&recovered), before, "{what}");
    }
}

/// `journal_to` whose journal's directory, or the folder it sits in, does
/// not sync raises before it saves anything, and the index is not
/// journaled.
#[test]
fn journal_to_whose_journal_does_not_sync_raises_and_journals_nothing() {
    for which in ["the journal's directory", "the folder it sits in"] {
        let temp = TempDir::new("directory-sync-tests");
        let path = temp.at("opened.zdb");
        let journal = journal_path(&path).unwrap();
        let synced = if which == "the folder it sits in" {
            temp.0.clone()
        } else {
            journal.clone()
        };
        let collection = Collection::build(declaration(), None);
        add(&collection, 0..2);
        let (opened, spent) = failing_sync(&synced, 0, || {
            collection.journal_to(path.to_str().unwrap(), Durability::PerCall)
        });
        assert!(spent, "{which}: the sync was asked");
        assert_eq!(
            failure_of(opened, which),
            format!(
                "Failed to create the journal {}: injected: the device refused it",
                journal.display()
            ),
            "{which}"
        );
        assert!(
            collection.journal_status().is_none(),
            "{which}: the index is not journaled"
        );
        assert!(!path.exists(), "{which}: nothing is saved");
    }
}

/// `journal_to` whose first segment's name does not sync raises once its
/// checkpoint has committed, and the index takes nothing until a checkpoint
/// commits.
#[test]
fn journal_to_whose_first_segment_does_not_sync_raises_and_stops_the_index() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("first.zdb");
    let journal = journal_path(&path).unwrap();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..2);
    // The journal's creation syncs its directory once, and the first
    // segment's creation syncs it next.
    let (opened, spent) = failing_sync(&journal, 1, || {
        collection.journal_to(path.to_str().unwrap(), Durability::PerCall)
    });
    assert!(spent, "the first segment's sync was asked");
    let reason = "the journal's directory could not be synced once the segment naming the \
                  checkpoint at sequence 0 was created (injected: the device refused it), so the \
                  journal takes nothing more until checkpoint() succeeds.";
    assert_eq!(
        failure_of(opened, "journal_to whose first segment does not sync"),
        format!("Failed to roll the journal {}: {reason}", journal.display())
    );
    assert_eq!(
        refused_add(&collection, 2, "the call after it"),
        refused_as(2, &append_refusal(&path, reason))
    );
    collection.checkpoint().unwrap();
    add(&collection, 3..4);
    let before = ids(&collection);
    drop(collection);
    let (recovered, _) = recover(&path).unwrap();
    assert_eq!(ids(&recovered), before);
}

/// A checkpoint makes each of its directory syncs through the system's own
/// call, being the journal's directory once its segment is created and once
/// the segments it holds are deleted, and the folder its directory sits in.
#[test]
fn a_checkpoint_syncs_the_journal_and_the_folder_through_the_system() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("synced.zdb");
    let journal = journal_path(&path).unwrap();
    let collection = journaled(&path);
    add(&collection, 0..2);
    directory_sync_faults::arm(&journal, usize::MAX, 0);
    directory_sync_faults::arm(&temp.0, usize::MAX, 0);
    collection.checkpoint().unwrap();
    let journal_syncs = (
        directory_sync_faults::asked_of(&journal),
        directory_sync_faults::completed_of(&journal),
    );
    let folder_syncs = (
        directory_sync_faults::asked_of(&temp.0),
        directory_sync_faults::completed_of(&temp.0),
    );
    directory_sync_faults::disarm(&journal);
    directory_sync_faults::disarm(&temp.0);
    assert_eq!(
        journal_syncs,
        (2, 2),
        "the journal's directory, asked and completed"
    );
    assert_eq!(folder_syncs, (1, 1), "the folder, asked and completed");
}

// ============================================================================
// A LOAD
// ============================================================================

/// A load syncs the journal's directory before the index takes a record,
/// since the active segment it opens may have been created by a process
/// whose sync of it failed or never ran, and a sync that fails fails the
/// load.
#[test]
fn a_load_whose_journal_does_not_sync_fails() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("resumed.zdb");
    let collection = journaled(&path);
    add(&collection, 0..3);
    drop(collection);
    let storage = FaultyStorage::at(&path);
    storage.faults.dir_syncs.arm(0, 1);
    let loaded = load_through(&storage, &path, replay());
    assert!(storage.faults.dir_syncs.spent(), "the sync was asked");
    assert_eq!(
        failure_of(loaded, "a load whose journal does not sync"),
        format!(
            "Failed to sync the entries of the journal {}: injected: the device refused it",
            journal_path(&path).unwrap().display()
        )
    );
}

/// A load that deletes a stale segment fails where the journal's directory
/// does not sync after the deletion. The sync failed is the load's last of
/// the journal's directory, found by a first load of a copy.
#[test]
fn a_load_whose_stale_segments_removal_does_not_sync_fails() {
    let temp = TempDir::new("directory-sync-tests");
    let counted = temp.at("counted.zdb");
    with_a_stale_segment(&counted);
    let asked = syncs_asked(&journal_path(&counted).unwrap(), || {
        drop(recover(&counted).unwrap());
    });
    assert_eq!(asked, 2, "the load syncs the journal's directory twice");

    let path = temp.at("stale.zdb");
    with_a_stale_segment(&path);
    let journal = journal_path(&path).unwrap();
    let (loaded, spent) = failing_sync(&journal, asked - 1, || recover(&path));
    assert!(spent, "the sync after the deletion was asked");
    assert_eq!(
        failure_of(loaded, "a load whose stale segment's removal does not sync"),
        format!(
            "Failed to sync the entries of the journal {}: injected: the device refused it",
            journal.display()
        )
    );
}

/// A load makes the index directory's name durable before it changes the
/// journal, since a save in an earlier process may have moved the directory
/// into place without that sync landing, and the stale segments it deletes
/// are held by that directory alone. A sync of the folder it sits in that
/// fails fails the load, and the stale segment stays.
#[test]
fn a_load_whose_directory_does_not_sync_fails_and_changes_nothing() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("placed.zdb");
    with_a_stale_segment(&path);
    let before = Segments::of(&path).numbers;
    let (loaded, spent) = failing_sync(&temp.0, 0, || recover(&path));
    assert!(spent, "the folder's sync was asked");
    assert_eq!(
        failure_of(loaded, "a load whose directory's name does not sync"),
        format!(
            "Failed to sync the index directory {} and the folder it sits in: {}: injected: the \
             device refused it",
            path.display(),
            temp.0.display()
        )
    );
    assert_eq!(
        Segments::of(&path).numbers,
        before,
        "the stale segment stays"
    );
}

/// A load whose sync of the directory in place is refused as a write is
/// refused, by a medium or a permission, opens read only, as a load whose
/// first write meets that refusal does, with every record replayed and the
/// journal as it was.
#[test]
fn a_load_whose_directory_sync_is_refused_as_a_write_opens_read_only() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("refused.zdb");
    let before = with_a_stale_segment(&path);
    let segments = Segments::of(&path);
    let storage = FaultyStorage::at(&path);
    storage.faults.in_place_denied.arm(0, 1);
    let (collection, report) = load_through(&storage, &path, replay()).unwrap();
    assert!(storage.faults.in_place_denied.spent(), "the sync was asked");
    assert_eq!(
        report.read_only.as_deref(),
        Some("injected: the folder may not be written"),
        "{report:?}"
    );
    assert_eq!(ids(&collection), before);
    let after = Segments::of(&path);
    assert_eq!(after.numbers, segments.numbers);
    assert_eq!(after.bytes, segments.bytes, "the journal is as it was");
}

/// An upgrade whose removal of the journal of format version 1 does not
/// sync fails the load, and the next load opens the directory with every
/// record it holds. The sync failed is the load's last of the folder, found
/// by a first load of a copy.
#[test]
fn an_upgrade_whose_old_journals_removal_does_not_sync_fails() {
    let name = FIXTURES[0];
    let temp = TempDir::new("directory-sync-tests");
    let counted = copy_fixture(name, &temp.at("counted"));
    let asked = syncs_asked(counted.parent().unwrap(), || {
        drop(recover(&counted).unwrap());
    });
    assert_eq!(asked, 3, "the upgrade syncs the folder three times");

    let path = copy_fixture(name, &temp.at("upgraded"));
    let folder = path.parent().unwrap().to_path_buf();
    let (loaded, spent) = failing_sync(&folder, asked - 1, || recover(&path));
    assert!(spent, "the sync after the removal was asked");
    assert_eq!(
        failure_of(
            loaded,
            "an upgrade whose old journal's removal does not sync"
        ),
        format!(
            "Failed to sync the index directory {} and the folder it sits in: {}: injected: the \
             device refused it",
            path.display(),
            folder.display()
        )
    );
    let (reopened, _) = recover(&path).unwrap();
    holds_fixture(&reopened, name).unwrap();
}

// ============================================================================
// A REPAIR
// ============================================================================

/// A repair whose quarantine folder does not sync fails the load before the
/// journal changes, since the cut would leave the bytes it sets aside with
/// no durable name.
#[test]
fn a_repair_whose_quarantine_does_not_sync_fails_and_leaves_the_journal() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("quarantine.zdb");
    damaged_journal(&path, "corrupt", declaration(), record);
    let segments = Segments::of(&path);
    let storage = FaultyStorage::at(&path);
    storage.faults.quarantine_syncs.arm(0, 1);
    let loaded = load_through(&storage, &path, repair());
    assert!(
        storage.faults.quarantine_syncs.spent(),
        "the quarantine's sync failed"
    );
    assert_eq!(
        failure_of(loaded, "a repair whose quarantine does not sync"),
        format!(
            "Failed to sync the quarantine of the journal {}: injected: the device refused it",
            journal_path(&path).unwrap().display()
        )
    );
    let after = Segments::of(&path);
    assert_eq!(after.numbers, segments.numbers);
    assert_eq!(after.bytes, segments.bytes, "the journal is as it was");
}

/// A repair whose cut, or whose removal of the stale segments it set aside,
/// is not followed by a sync of the journal's directory that lands fails
/// the load.
#[test]
fn a_repair_whose_cut_or_removal_does_not_sync_fails() {
    for (class, what) in [("corrupt", "cut"), ("stale", "set aside")] {
        let temp = TempDir::new("directory-sync-tests");
        let path = temp.at("cut.zdb");
        damaged_journal(&path, class, declaration(), record);
        let storage = FaultyStorage::at(&path);
        storage.faults.dir_syncs.arm(0, 1);
        let loaded = load_through(&storage, &path, repair());
        assert!(
            storage.faults.dir_syncs.spent(),
            "{class}: the directory's sync failed"
        );
        assert_eq!(
            failure_of(loaded, class),
            format!(
                "Failed to {what} the journal {}: injected: the device refused it",
                journal_path(&path).unwrap().display()
            ),
            "{class}"
        );
    }
}

/// A repair whose marker's removal does not sync once its checkpoint has
/// landed fails the load, and the next load opens the directory the
/// repair's checkpoint wrote.
#[test]
fn a_repair_whose_seal_does_not_sync_fails() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("seal.zdb");
    let (calls, _) = damaged_journal(&path, "corrupt", declaration(), record);
    let storage = FaultyStorage::at(&path);
    storage.faults.quarantine_syncs.arm(1, 1);
    let loaded = load_through(&storage, &path, repair());
    assert!(
        storage.faults.quarantine_syncs.spent(),
        "the seal's sync failed"
    );
    assert_eq!(
        failure_of(loaded, "a repair whose seal does not sync"),
        format!(
            "Failed to sync the quarantine of the journal {}: injected: the device refused it",
            journal_path(&path).unwrap().display()
        )
    );
    let (reopened, _) = recover(&path).unwrap();
    assert_eq!(
        reopened.vector_count(),
        calls * super::journal_support::DAMAGED_CALL_RECORDS
    );
}

// ============================================================================
// A SAVE
// ============================================================================

/// A save whose staging directory does not sync raises before anything
/// moves, and the directory in place is the one the last save wrote.
#[test]
fn a_save_whose_staging_directory_does_not_sync_raises_and_keeps_the_directory() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("staged.zdb");
    let staging = temp.at("staged.zdb.zdbtmp");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..2);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 2..3);
    let (saved, spent) = failing_sync(&staging, 0, || collection.save(path.to_str().unwrap()));
    assert!(spent, "the staging directory's sync was asked");
    assert_eq!(
        failure_of(saved, "a save whose staging directory does not sync"),
        format!(
            "Failed to sync {} before the saved index moved into place: injected: the device \
             refused it. Nothing was moved, and the directory at {} is as it was.",
            staging.display(),
            path.display()
        )
    );
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(ids(&loaded), ["r0", "r1"]);
}

/// A save whose folder does not sync once the directory has moved into
/// place raises, since the move may not survive a power loss.
#[test]
fn a_save_whose_folder_does_not_sync_raises() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("moved.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..2);
    let (saved, spent) = failing_sync(&temp.0, 0, || collection.save(path.to_str().unwrap()));
    assert!(spent, "the folder's sync was asked");
    assert_eq!(
        failure_of(saved, "a save whose folder does not sync"),
        format!(
            "The saved index is in place at {}, and the folder it sits in could not be synced: \
             injected: the device refused it. The move may not survive a power loss.",
            path.display()
        )
    );
}

/// A checkpoint whose folder does not sync once its directory has moved
/// into place raises and stops the index until a checkpoint commits, since
/// a record the journal took under the checkpoint before would not replay
/// over the directory that moved in.
#[test]
fn a_checkpoint_whose_folder_does_not_sync_raises_and_stops_the_index() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("moved.zdb");
    let collection = journaled(&path);
    add(&collection, 0..2);
    let (checkpointed, spent) = failing_sync(&temp.0, 0, || collection.checkpoint());
    assert!(spent, "the folder's sync was asked");
    assert_eq!(
        failure_of(checkpointed, "a checkpoint whose folder does not sync"),
        format!(
            "The saved index is in place at {}, and the folder it sits in could not be synced: \
             injected: the device refused it. The move may not survive a power loss, so the \
             index it replaced is kept at {} until the next save.",
            path.display(),
            temp.at("moved.zdb.zdbold").display()
        )
    );
    assert_eq!(
        refused_add(&collection, 2, "the call after it"),
        refused_as(
            2,
            &format!(
                "The checkpoint moved its directory into place at {} and the folder it sits in \
                 could not be synced (injected: the device refused it), so nothing more is \
                 recorded until checkpoint() succeeds.",
                path.display()
            )
        )
    );
    collection.checkpoint().unwrap();
    add(&collection, 3..4);
    let before = ids(&collection);
    drop(collection);
    let (recovered, _) = recover(&path).unwrap();
    assert_eq!(ids(&recovered), before);
}

/// A save of an index with a sparse space syncs the space's directory,
/// which holds its postings, and one that does not sync makes the save
/// raise before anything moves.
#[test]
fn a_save_whose_space_directory_does_not_sync_raises() {
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("spaced.zdb");
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![])
        .unwrap()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let collection = Collection::build(declaration, None);
    let mut first = record(0);
    first.sparse = Some(SparseHalf::Vector(SparseVector {
        dims: vec![1, 5],
        values: vec![0.5, 1.0],
    }));
    let added = collection.add_records(vec![first], vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    let space = temp.at("spaced.zdb.zdbtmp").join("spaces").join("terms");
    let (saved, spent) = failing_sync(&space, 0, || collection.save(path.to_str().unwrap()));
    assert!(spent, "the space's directory's sync was asked");
    assert_eq!(
        failure_of(saved, "a save whose space's directory does not sync"),
        format!(
            "Failed to sync {} before the saved index moved into place: injected: the device \
             refused it. Nothing was moved, and the directory at {} is as it was.",
            space.display(),
            path.display()
        )
    );
    assert!(!path.exists(), "nothing moved");
    collection.save(path.to_str().unwrap()).unwrap();
}

/// A save into a folder it creates syncs the folder above it, so the path
/// to the directory survives a power loss. A sync that fails refuses the
/// save, removes the folder created, and leaves the next save to create
/// and sync it again.
#[test]
fn a_save_into_a_new_folder_syncs_the_folder_above_it() {
    let temp = TempDir::new("directory-sync-tests");
    let folder = temp.at("new");
    let path = folder.join("made.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..2);
    let (saved, spent) = failing_sync(&temp.0, 0, || collection.save(path.to_str().unwrap()));
    assert!(spent, "the sync of the folder above was asked");
    let saved = failure_of(saved, "a save whose new folder's entry does not sync");
    assert!(
        saved.starts_with(&format!(
            "The lock file {} cannot be locked: the folder {} was created and the folder above \
             it could not be synced (injected: the device refused it), so the folders created \
             were removed.",
            folder.join("made.zdb.zdblock").display(),
            folder.display()
        )),
        "{saved}"
    );
    assert!(!folder.exists(), "the folder created is removed");
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(
        ids(&Collection::load(path.to_str().unwrap()).unwrap()),
        ["r0", "r1"]
    );
}

/// On a system that opens a directory to read, a journal directory this
/// process may write and may not read refuses the open its sync makes,
/// which is a refusal the filesystem itself gives, so a checkpoint raises
/// and the index takes nothing until a checkpoint commits. Root reads every
/// directory, so the test runs as any other user.
#[cfg(unix)]
#[test]
fn a_checkpoint_under_a_journal_this_process_cannot_read_raises() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new("directory-sync-tests");
    let path = temp.at("unread.zdb");
    let journal = journal_path(&path).unwrap();
    let collection = journaled(&path);
    add(&collection, 0..2);
    std::fs::set_permissions(&journal, std::fs::Permissions::from_mode(0o300)).unwrap();
    if std::fs::File::open(&journal).is_ok() {
        std::fs::set_permissions(&journal, std::fs::Permissions::from_mode(0o700)).unwrap();
        eprintln!("skipped: this user reads a folder whose mode refuses reads");
        return;
    }
    let checkpointed = collection.checkpoint();
    let refused = collection.add_records(vec![record(2)], vec![], false);
    std::fs::set_permissions(&journal, std::fs::Permissions::from_mode(0o700)).unwrap();
    let checkpointed = failure_of(checkpointed, "a checkpoint under a journal it cannot read");
    assert!(
        checkpointed.starts_with(&format!(
            "Failed to roll the journal {}: the journal's directory could not be synced once the \
             segment naming the checkpoint at sequence 3 was created (Permission denied",
            journal.display()
        )),
        "{checkpointed}"
    );
    assert_eq!(refused.total_errors, 1, "{:?}", refused.errors);
    collection.checkpoint().unwrap();
    add(&collection, 3..4);
}
