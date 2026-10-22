//! What a load asked to repair a damaged journal keeps, sets aside and
//! writes.
//!
//! Each class of damage is built on a journal of six calls in two segments,
//! see `journal_support::damaged_journal`, and held to the repair's rules:
//! every call before the damaged one kept, that call and every byte after
//! it copied whole into one quarantine folder, the journal cut there, a
//! checkpoint taken and the folder's marker removed. A kill at each of the
//! repair's points, followed by a plain load and a second repair, is the
//! crash matrix's to hold.

use std::io;
use std::path::Path;

use serde_json::{json, Value};
use zeusdb_vector_core::{
    encode_journal_record, read_legacy_journal, Error, FsStorage, Operation, Storage,
    JOURNAL_RECORD_HEADER_BYTES,
};

use super::journal_support::{
    copy_fixture, damaged_journal, expected_of, files_under, manifest, quarantine_of,
    FaultyStorage, RefusingStorage, Segments, TempDir, DAMAGED_CALL_RECORDS, REPAIR_CLASSES,
};
use super::{Collection, Declaration, ParsedRecord};
use crate::journal::{journal_path, CheckpointTaken, Durability, JournalPolicy};
use crate::Recovery;

const DIM: usize = 4;

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 8, 50, 200, vec![]).unwrap()
}

fn record(i: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: (0..DIM).map(|d| (i * DIM + d) as f32 * 0.25).collect(),
        sparse: None,
        metadata: [("i".to_string(), json!(i))].into_iter().collect(),
    }
}

fn repair(path: &Path) -> Result<(Collection, Recovery), Error> {
    Collection::recover_with(
        path.to_str().unwrap(),
        None,
        JournalPolicy::Repair(Durability::PerCall),
    )
}

fn plain(path: &Path) -> Result<(Collection, Recovery), Error> {
    Collection::recover(path.to_str().unwrap(), None, Durability::PerCall)
}

fn ids(collection: &Collection) -> Vec<String> {
    let mut ids: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    ids.sort();
    ids
}

/// The ids the first `calls` calls of a damaged journal add, sorted.
fn kept_ids(calls: usize) -> Vec<String> {
    let mut ids: Vec<String> = (0..DAMAGED_CALL_RECORDS * calls)
        .map(|i| format!("r{i}"))
        .collect();
    ids.sort();
    ids
}

/// The kind of damage a repair names each class by.
fn kind_of(class: &str) -> &'static str {
    match class {
        "corrupt" | "sealed" => "corrupt",
        "header" => "header",
        _ => "decode",
    }
}

/// The whole operation records the bytes a repair of `class` sets aside
/// hold, as the reader's scans count them. A record whose checksum fails is
/// not whole, and one whose checksum holds is, whether it decodes or not.
fn records_set_aside(class: &str) -> u64 {
    match class {
        // The fifth and sixth calls, one record of them damaged.
        "corrupt" => 5,
        // The second to the sixth calls, one record of them damaged.
        "sealed" => 14,
        // The fifth and sixth calls, behind a header that does not read.
        "header" => 6,
        // The third to the sixth calls.
        _ => 12,
    }
}

/// A journaled directory of `calls` calls of three records each.
fn journaled(path: &Path, calls: usize) {
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    for call in 0..calls {
        let first = DAMAGED_CALL_RECORDS * call;
        let batch = (first..first + DAMAGED_CALL_RECORDS).map(record).collect();
        let added = collection.add_records(batch, vec![], false);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    }
}

/// Each class of damage is cut at the first record of the call it touches:
/// every call before it is kept, that call and every byte after it are set
/// aside whole in one folder whose name ends at the first sequence set
/// aside, holding no marker and a report of the cut, the report and the
/// warning say what was cut and why, and the directory opens plainly from
/// then on. A load that does not repair refuses each class first, and the
/// refusal of a corrupt middle names `repair=True` with the cut.
#[test]
fn a_repair_keeps_every_call_before_the_damage_for_each_class() {
    for class in REPAIR_CLASSES.into_iter().filter(|class| *class != "stale") {
        let tmp = TempDir::new("repair-class");
        let path = tmp.at("index.zdb");
        let (calls, aside) = damaged_journal(&path, class, declaration(), record);
        let first_discarded = 1 + 4 * calls as u64;

        let refused = plain(&path)
            .err()
            .unwrap_or_else(|| panic!("{class}: a plain load opened a damaged journal"));
        match (class, &refused) {
            ("corrupt" | "sealed", Error::JournalCorrupt { cut_sequence, .. }) => {
                assert_eq!(*cut_sequence, first_discarded, "{class}");
                let message = refused.to_string();
                assert!(message.contains("repair=True"), "{message}");
                assert!(
                    message.contains(&format!("before record {first_discarded}")),
                    "{message}"
                );
            }
            ("header", Error::JournalHeaderInvalid { .. }) => {}
            ("decode", Error::JournalRecordInvalid { .. }) => {}
            (class, other) => panic!("{class}: a plain load refused with {other:?}"),
        }

        let (repaired, report) = repair(&path).unwrap_or_else(|e| panic!("{class}: {e}"));
        assert_eq!(ids(&repaired), kept_ids(calls), "{class}");
        let cut = report.repair.clone().unwrap();
        assert_eq!(cut.discarded_from, Some(first_discarded), "{class}");
        assert_eq!(cut.kept_through, first_discarded - 1, "{class}");
        assert_eq!(cut.bytes_discarded, aside.len() as u64, "{class}");
        assert_eq!(cut.records_discarded, records_set_aside(class), "{class}");
        let damage = report.damage.clone().unwrap();
        assert_eq!(damage.kind, kind_of(class), "{class}");
        assert!(damage.detail.is_some(), "{class}");
        assert_eq!(
            report.checkpoint_taken,
            Some(CheckpointTaken::Repair),
            "{class}"
        );
        assert_eq!(report.read_only, None, "{class}");

        let (files, folders, unsealed) = quarantine_of(&path);
        assert_eq!(folders.len(), 1, "{class}");
        assert!(!unsealed, "{class}: the repair's marker is gone");
        let folder = &folders[0];
        assert!(
            folder.ends_with(&format!("-{first_discarded}")) && !folder.contains(':'),
            "{class}: {folder}"
        );
        let held: Vec<u8> = files.iter().flat_map(|(_, bytes)| bytes.clone()).collect();
        assert!(
            held == aside,
            "{class}: the quarantine holds {} bytes where the cut took {}",
            held.len(),
            aside.len()
        );
        let quarantine = journal_path(&path).unwrap().join("quarantine").join(folder);
        assert_eq!(cut.quarantine.as_deref(), Some(quarantine.as_path()));
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(quarantine.join("repair.json")).unwrap())
                .unwrap();
        assert_eq!(
            written["cut"]["sequence"],
            json!(first_discarded),
            "{class}"
        );
        assert_eq!(written["kept_through"], json!(first_discarded - 1));
        assert_eq!(
            written["records_discarded"],
            json!(records_set_aside(class))
        );
        assert_eq!(written["bytes_discarded"], json!(aside.len()));
        assert_eq!(written["damage"]["kind"], json!(kind_of(class)));
        assert_eq!(
            written["files"].as_array().unwrap().len(),
            files.len(),
            "{class}: the report names every file"
        );
        let warning = report.repair_warning("index.zdb").unwrap();
        assert!(
            warning.contains(&format!("record {first_discarded}"))
                && warning.contains(&quarantine.display().to_string()),
            "{warning}"
        );
        assert_eq!(report.read_only_warning("index.zdb"), None);

        // The checkpoint sealed the history, so the journal holds the one
        // segment it rolled onto, and every later load is a plain one.
        assert_eq!(
            Segments::of(&path).numbers,
            vec![first_discarded],
            "{class}"
        );
        drop(repaired);
        let (again, again_report) = plain(&path).unwrap_or_else(|e| panic!("{class}: {e}"));
        assert_eq!(ids(&again), kept_ids(calls), "{class}");
        assert_eq!(again_report.damage, None, "{class}");
        drop(again);
        let (third, third_report) = repair(&path).unwrap();
        assert_eq!(third_report.repair, None, "{class}: nothing more to cut");
        assert_eq!(third_report.checkpoint_taken, None, "{class}");
        assert_eq!(ids(&third), kept_ids(calls), "{class}");
    }
}

/// A repair of a journal with no damage and no repair unfinished is a plain
/// load: the same report but for the repair's part, and no byte of the
/// directory or the journal changes.
#[test]
fn a_repair_of_an_undamaged_journal_changes_nothing() {
    let tmp = TempDir::new("repair-clean");
    let path = tmp.at("index.zdb");
    journaled(&path, 4);
    let journal = journal_path(&path).unwrap();
    let before = files_under(&[&path, &journal]);
    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(ids(&repaired), kept_ids(4));
    drop(repaired);
    assert_eq!(files_under(&[&path, &journal]), before);
    assert_eq!(report.repair, None);
    assert_eq!(report.damage, None);
    assert_eq!(report.checkpoint_taken, None);
    assert_eq!(report.repair_warning("index.zdb"), None);
    let (_, plain_report) = plain(&path).unwrap();
    assert_eq!(report, plain_report);
    assert_eq!(files_under(&[&path, &journal]), before);
}

/// A directory with no journal opens under a repair as it opens without
/// one.
#[test]
fn a_repair_of_a_directory_with_no_journal_is_a_plain_load() {
    let tmp = TempDir::new("repair-none");
    let path = tmp.at("index.zdb");
    let collection = Collection::build(declaration(), None);
    collection.add_records((0..5).map(record).collect(), vec![], false);
    collection.save(path.to_str().unwrap()).unwrap();
    drop(collection);
    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(repaired.len(), 5);
    assert!(!report.journaled);
    assert_eq!(report.repair, None);
    assert!(!journal_path(&path).unwrap().exists());
}

/// A journal that is not the directory's is refused under a repair as
/// without one, and nothing is set aside.
#[test]
fn a_repair_refuses_a_journal_that_is_not_the_directorys() {
    let tmp = TempDir::new("repair-other");
    let path = tmp.at("index.zdb");
    let other = tmp.at("other.zdb");
    journaled(&path, 2);
    journaled(&other, 2);
    let journal = journal_path(&path).unwrap();
    for (number, bytes) in Segments::of(&other)
        .numbers
        .iter()
        .zip(Segments::of(&other).bytes)
    {
        std::fs::write(journal.join(format!("{number:020}.zdbseg")), bytes).unwrap();
    }
    let before = files_under(&[&path, &journal]);
    assert!(matches!(
        repair(&path),
        Err(Error::JournalNotThisCollection { .. })
    ));
    assert_eq!(files_under(&[&path, &journal]), before);
    assert!(!journal.join("quarantine").exists());
}

/// A record that decodes and will not apply is found during the replay,
/// after the records of its call before it were applied. The repair reads
/// the checkpoint again and replays the calls before that call, so no part
/// of it is applied, and cuts at its first record.
#[test]
fn a_record_that_will_not_apply_is_cut_with_its_call() {
    let tmp = TempDir::new("repair-apply");
    let path = tmp.at("index.zdb");
    journaled(&path, 6);
    // The fourth call's third record names an internal id other than the
    // one the collection would issue, under a checksum that holds.
    let sequence = 1 + 4 * 3 + 2;
    let segments = Segments::of(&path);
    let contents = segments.read(&expected_of(&path)).unwrap();
    let live = contents
        .records
        .iter()
        .find(|live| live.record.sequence == sequence)
        .unwrap();
    let (internal_id, level) = match Operation::decode(&live.record, DIM, "").unwrap() {
        Operation::Insert {
            internal_id, level, ..
        } => (internal_id, level),
        other => panic!("{other:?}"),
    };
    let mut payload = live.record.payload.to_vec();
    Operation::stamp_insert(&mut payload, internal_id + 7, level);
    let mut out = Vec::new();
    encode_journal_record(sequence, live.record.kind, &payload, &mut out);
    let at = live.record.offset as usize;
    let mut bytes = segments.bytes[0].clone();
    bytes[at..at + out.len()].copy_from_slice(&out);
    std::fs::write(
        journal_path(&path)
            .unwrap()
            .join(format!("{:020}.zdbseg", segments.numbers[0])),
        &bytes,
    )
    .unwrap();

    assert!(matches!(
        plain(&path),
        Err(Error::JournalReplayFailed { .. })
    ));
    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(
        ids(&repaired),
        kept_ids(3),
        "none of the fourth call is held"
    );
    assert!(!repaired.contains("r9") && !repaired.contains("r10"));
    let damage = report.damage.clone().unwrap();
    assert_eq!(damage.kind, "apply");
    assert_eq!(damage.sequence, sequence);
    let cut = report.repair.clone().unwrap();
    assert_eq!(cut.discarded_from, Some(1 + 4 * 3));
    assert_eq!(cut.records_discarded, 9);
    assert_eq!(report.replayed, 9, "the calls before the cut, applied once");
    drop(repaired);
    let (again, _) = plain(&path).unwrap();
    assert_eq!(ids(&again), kept_ids(3));
}

/// Where the medium refuses the repair's first write, the kept calls are
/// replayed in memory, nothing on the medium changes, the collection opens
/// read only and refuses every change, and the report and the warning
/// describe the cut the repair would make with no quarantine. A medium that
/// refuses the lock file's write and one that refuses the quarantine's
/// folder are each such a medium.
#[test]
fn a_repair_on_a_medium_that_cannot_be_written_opens_read_only() {
    for denied_at_quarantine in [false, true] {
        let tmp = TempDir::new("repair-read-only");
        let path = tmp.at("index.zdb");
        let (calls, aside) = damaged_journal(&path, "corrupt", declaration(), record);
        let journal = journal_path(&path).unwrap();
        let before = files_under(&[&path, &journal]);
        let loaded = if denied_at_quarantine {
            let storage = FaultyStorage::at(&path);
            storage.faults.quarantines_denied.arm(0, 1);
            crate::persistence::load_from(
                &storage,
                path.to_str().unwrap(),
                None,
                JournalPolicy::Repair(Durability::PerCall),
            )
        } else {
            let storage = RefusingStorage::at(&path, io::ErrorKind::PermissionDenied);
            crate::persistence::load_from(
                &storage,
                path.to_str().unwrap(),
                None,
                JournalPolicy::Repair(Durability::PerCall),
            )
        };
        let (collection, report) = loaded.unwrap();
        assert_eq!(ids(&collection), kept_ids(calls));
        assert!(report.read_only.is_some(), "{report:?}");
        let cut = report.repair.clone().unwrap();
        assert_eq!(cut.quarantine, None);
        assert_eq!(cut.bytes_discarded, aside.len() as u64);
        assert_eq!(report.checkpoint_taken, None);
        let warning = report.repair_warning("index.zdb").unwrap();
        assert!(
            warning.contains("read only") && warning.contains(report.read_only.as_ref().unwrap()),
            "{warning}"
        );
        assert_eq!(report.read_only_warning("index.zdb"), None);
        let added = collection.add_records(vec![record(99)], vec![], false);
        assert_eq!(added.total_errors, 1);
        drop(collection);
        assert_eq!(files_under(&[&path, &journal]), before);
        assert!(!journal.join("quarantine").exists());
    }
}

/// A failure at the repair's first write that is not a medium refusing
/// writes refuses the load, and nothing changes.
#[test]
fn a_failure_at_the_quarantine_refuses_the_load() {
    let tmp = TempDir::new("repair-fails");
    let path = tmp.at("index.zdb");
    damaged_journal(&path, "corrupt", declaration(), record);
    let journal = journal_path(&path).unwrap();
    let before = files_under(&[&path, &journal]);
    let storage = FaultyStorage::at(&path);
    storage.faults.quarantines.arm(0, 1);
    let refused = crate::persistence::load_from(
        &storage,
        path.to_str().unwrap(),
        None,
        JournalPolicy::Repair(Durability::PerCall),
    );
    match refused {
        Err(Error::JournalIoFailed { what, .. }) => assert_eq!(what, "set aside"),
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("the load went on past a quarantine it could not write"),
    }
    assert_eq!(files_under(&[&path, &journal]), before);
}

/// A repair that finds a journal holding a withdrawn call and a marker
/// left by a repair whose checkpoint never landed under the directory's own
/// snapshot takes one checkpoint for both, reported as the repair's, and
/// removes the marker.
#[test]
fn a_repair_that_meets_an_abort_takes_one_checkpoint() {
    let tmp = TempDir::new("repair-abort");
    let path = tmp.at("index.zdb");
    {
        let collection = Collection::build(declaration(), None);
        let storage = FaultyStorage::at(&path);
        let sink = super::journal_support::sink_over(
            &storage,
            &path,
            collection.collection_id(),
            Durability::PerCall,
        );
        collection.attach_sink(Box::new(sink));
        collection.save(path.to_str().unwrap()).unwrap();
        collection.add_records((0..3).map(record).collect(), vec![], false);
        storage.faults.syncs.arm(0, 1);
        let refused = collection.add_records((3..6).map(record).collect(), vec![], false);
        assert_eq!(refused.total_errors, 3);
    }
    let identity = &manifest(&path)["identity"];
    let folder = journal_path(&path)
        .unwrap()
        .join("quarantine")
        .join("20261021T000000000Z-9");
    std::fs::create_dir_all(&folder).unwrap();
    let report = json!({
        "damage": {"kind": "corrupt", "segment": "x", "at": 0, "sequence": 9, "detail": null},
        "base": {"snapshot": identity["snapshot"], "sequence": identity["sequence"]},
        "cut": {"segment": "x", "at": 0, "sequence": 9},
        "kept_through": 8,
        "records_discarded": 0,
        "bytes_discarded": 0,
        "files": [],
        "repaired_at": "2026-10-21T00:00:00+00:00"
    });
    std::fs::write(folder.join("repair.json"), report.to_string()).unwrap();
    std::fs::write(folder.join("UNSEALED"), b"").unwrap();

    let generation = manifest(&path)["identity"]["generation"].as_u64().unwrap();
    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(ids(&repaired), kept_ids(1));
    assert_eq!(report.checkpoint_taken, Some(CheckpointTaken::Repair));
    assert_eq!(report.repair, None, "nothing was damaged");
    drop(repaired);
    assert!(!folder.join("UNSEALED").exists());
    assert_eq!(
        manifest(&path)["identity"]["generation"].as_u64().unwrap(),
        generation + 1,
        "one checkpoint"
    );
}

/// A journal of format version 1 with a corrupt middle is repaired the same
/// way: every record before the damaged one is kept, since version 1 frames
/// no call, the bytes from it on are set aside in the quarantine of the new
/// journal's directory, the file is cut, and the upgrade's checkpoint seals
/// it. A load that does not repair refuses it, naming `repair=True`.
#[test]
fn a_version_1_journal_is_repaired_into_the_new_journals_quarantine() {
    let tmp = TempDir::new("repair-legacy");
    let path = copy_fixture("v3.0.0-journaled", &tmp.0);
    let storage = FsStorage::at(&path).unwrap();
    let wal = storage.legacy_journal_path().to_path_buf();
    let mut bytes = std::fs::read(&wal).unwrap();
    let checkpoint = manifest(&path)["journal"]["sequence"].as_u64().unwrap();
    let (damaged, at) = {
        let contents = read_legacy_journal(&bytes, "w").unwrap();
        let above: Vec<_> = contents
            .records
            .iter()
            .filter(|record| record.sequence > checkpoint)
            .collect();
        assert!(
            above.len() >= 3,
            "the fixture's journal holds records to cut"
        );
        let damaged = above[above.len() / 2];
        (damaged.sequence, damaged.offset as usize)
    };
    bytes[at + JOURNAL_RECORD_HEADER_BYTES + 1] ^= 0x40;
    std::fs::write(&wal, &bytes).unwrap();

    // What the records before the damaged one build, applied by hand onto
    // the checkpoint.
    let expected = {
        let copy = tmp.at("expected.zdb");
        super::journal_support::copy_tree(&path, &copy);
        let (built, _) =
            Collection::recover_with(copy.to_str().unwrap(), None, JournalPolicy::CheckpointOnly)
                .unwrap();
        let contents = read_legacy_journal(&bytes, "w").unwrap();
        for record in contents
            .records
            .iter()
            .filter(|record| record.sequence > checkpoint && record.sequence < damaged)
        {
            let operation = Operation::decode_legacy(record, built.dim(), "w").unwrap();
            built.apply_legacy(operation).unwrap();
        }
        ids(&built)
    };

    let refused = plain(&path).err().unwrap();
    assert!(refused.to_string().contains("repair=True"), "{refused}");
    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(ids(&repaired), expected);
    assert_eq!(report.upgraded_from.as_deref(), Some("3.0.0"));
    assert_eq!(report.checkpoint_taken, Some(CheckpointTaken::Repair));
    let cut = report.repair.clone().unwrap();
    assert_eq!(cut.discarded_from, Some(damaged));
    assert_eq!(cut.bytes_discarded, (bytes.len() - at) as u64);
    assert_eq!(report.damage.as_ref().unwrap().kind, "corrupt");
    drop(repaired);
    assert_eq!(manifest(&path)["format_version"], json!("5.0.0"));
    assert!(!wal.exists(), "the file goes once the upgrade commits");
    let (files, folders, unsealed) = quarantine_of(&path);
    assert_eq!(folders.len(), 1);
    assert!(!unsealed);
    assert_eq!(files.len(), 1);
    let wal_name = wal.file_name().unwrap().to_string_lossy().into_owned();
    assert_eq!(files[0].0, format!("{wal_name}.{at}"));
    assert!(files[0].1 == bytes[at..]);
    let (again, _) = plain(&path).unwrap();
    assert_eq!(ids(&again), expected);
}

/// A segment named at or below the checkpoint's sequence is stale by its
/// name, whatever its header holds. Where its header does not read, a plain
/// load refuses the journal and names `repair=True`, and a repair copies the
/// segment whole into the quarantine and removes it, cuts nothing, and keeps
/// every record of the live segment beside it.
#[test]
fn a_stale_segment_whose_header_does_not_read_is_set_aside_alone() {
    let tmp = TempDir::new("repair-stale");
    let path = tmp.at("index.zdb");
    let (calls, aside) = damaged_journal(&path, "stale", declaration(), record);
    let refused = plain(&path)
        .err()
        .expect("a plain load refuses the stale segment");

    let (repaired, report) = repair(&path).unwrap();
    assert_eq!(ids(&repaired), kept_ids(calls), "every live call is kept");
    let set_aside = report.repair.clone().unwrap();
    assert_eq!(
        set_aside.discarded_from, None,
        "nothing past the checkpoint"
    );
    assert_eq!(set_aside.records_discarded, 0);
    assert_eq!(set_aside.bytes_discarded, aside.len() as u64);
    assert_eq!(set_aside.kept_through, 4 * calls as u64);
    let damage = report.damage.clone().unwrap();
    assert_eq!((damage.kind, damage.sequence, damage.at), ("header", 1, 0));
    assert_eq!(report.checkpoint_taken, Some(CheckpointTaken::Repair));
    let (files, folders, unsealed) = quarantine_of(&path);
    assert_eq!(folders.len(), 1);
    assert!(!unsealed, "the repair's marker is gone");
    assert!(folders[0].ends_with("-1"), "{}", folders[0]);
    assert_eq!(files.len(), 1, "the stale segment alone is set aside");
    assert_eq!(files[0].0, format!("{:020}.zdbseg", 1));
    assert!(
        files[0].1 == aside,
        "the quarantine holds the segment whole"
    );
    drop(repaired);

    let (again, again_report) = plain(&path).unwrap();
    assert_eq!(ids(&again), kept_ids(calls));
    assert_eq!(again_report.damage, None);
    let message = refused.to_string();
    assert!(
        matches!(refused, Error::JournalHeaderInvalid { .. }) && message.contains("repair=True"),
        "{message}"
    );
}

/// A collection a repair load opened holds its directory exclusive for its
/// life, as one a plain load replayed does, whether its journal was damaged
/// or not. A second repair, a plain load and the checkpoint alone are each
/// refused while it is open, and the directory opens again once it is
/// dropped.
#[test]
fn a_repaired_collection_holds_its_directory_exclusive() {
    let exclusive = if cfg!(windows) {
        "does not let a second handle read"
    } else {
        "mode exclusive"
    };
    for damaged in [true, false] {
        let tmp = TempDir::new("repair-lock");
        let path = tmp.at("index.zdb");
        let calls = if damaged {
            damaged_journal(&path, "corrupt", declaration(), record).0
        } else {
            journaled(&path, 2);
            2
        };
        let (repaired, _) = repair(&path).unwrap();
        let attempts = [
            ("a second repair", repair(&path).map(|_| ())),
            ("a plain load", plain(&path).map(|_| ())),
            (
                "the checkpoint alone",
                Collection::load_checkpoint_only(path.to_str().unwrap(), None).map(|_| ()),
            ),
        ];
        for (what, attempt) in attempts {
            match attempt {
                Err(Error::CollectionLocked { holder, .. }) => {
                    assert!(holder.contains(exclusive), "{what}: {holder}");
                }
                Err(other) => panic!("damaged {damaged}, {what}: refused with {other}"),
                Ok(()) => panic!("damaged {damaged}, {what}: opened beside the repair"),
            }
        }
        drop(repaired);
        let (again, _) = repair(&path).unwrap();
        assert_eq!(ids(&again), kept_ids(calls), "damaged {damaged}");
    }
}
