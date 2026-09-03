//! A directory and the journal beside it.
//!
//! What a checkpoint writes, what pairs the two, what is refused and what a
//! recovery gives back. The crashes are next door in `crash_tests`; nothing
//! here kills a process, and everything here is a rule a caller can reach by
//! copying, renaming or editing a directory.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;

use serde_json::{json, Value};
use zeusdb_vector_core::{Error, SparseVector, DUMP_FILENAME};
use zeusdb_vector_sparse::SparseConfig;

use super::{Collection, Declaration, ParsedRecord};
use crate::journal::{journal_path, Durability};

// ============================================================================
// FIXTURES
// ============================================================================

/// A directory under the system's temporary directory, removed on drop
/// together with the journals beside anything in it.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zeusdb-journal-tests-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn at(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn declaration() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()]).unwrap()
}

fn record(i: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vec![i as f32 * 0.25, (i % 5) as f32],
        sparse: None,
        metadata: HashMap::from([
            (
                "cat".to_string(),
                json!(if i.is_multiple_of(2) { "a" } else { "b" }),
            ),
            ("i".to_string(), json!(i)),
        ]),
    }
}

fn add(collection: &Collection, range: std::ops::Range<usize>) {
    let records: Vec<ParsedRecord> = range.map(record).collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn manifest(dir: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn rewrite_manifest(dir: &std::path::Path, edit: impl FnOnce(&mut Value)) {
    let mut m = manifest(dir);
    edit(&mut m);
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
}

/// Every id the collection holds, in order, with its internal id.
fn ids(collection: &Collection) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = collection
        .id_map()
        .iter()
        .map(|(id, internal)| (id.clone(), *internal))
        .collect();
    out.sort();
    out
}

/// A journaled collection at `path`, with the empty checkpoint written.
fn journaled(dir: &std::path::Path) -> Collection {
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(dir.to_str().unwrap(), Durability::default())
        .unwrap();
    collection
}

// ============================================================================
// WHAT A CHECKPOINT WRITES
// ============================================================================

/// Turning the journal on writes the checkpoint it replays onto, and that
/// checkpoint names the journal, the collection and the sequence.
#[test]
fn opening_a_journal_writes_the_empty_checkpoint_it_replays_onto() {
    let temp = TempDir::new();
    let path = temp.at("fresh.zdb");
    let collection = journaled(&path);

    assert!(path.is_dir(), "the checkpoint's directory exists");
    assert!(path.join("manifest.json").is_file());
    assert!(path.join("config.json").is_file());
    // An empty index dumps no graph, which is what a save of one has always
    // done.
    assert!(!path.join(DUMP_FILENAME).exists());
    let wal = journal_path(&path).unwrap();
    assert!(wal.is_file(), "the journal is beside the directory");
    assert_eq!(
        std::fs::metadata(&wal).unwrap().len(),
        zeusdb_vector_core::JOURNAL_HEADER_BYTES as u64,
        "a journal with no record is its header alone"
    );

    let m = manifest(&path);
    assert_eq!(m["format_version"], json!("3.0.0"));
    assert_eq!(m["journal"]["file"], json!("fresh.zdb.zdbwal"));
    assert_eq!(m["journal"]["sequence"], json!(0));
    assert_eq!(
        m["journal"]["collection_id"],
        json!(format!("{:032x}", collection.collection_id()))
    );
    assert_eq!(collection.journal_sequence(), 0);
}

/// A save of a collection holding no journal is what it was: the first
/// major, and no `journal` field in the manifest at all.
#[test]
fn a_directory_saved_without_a_journal_carries_no_journal_field() {
    let temp = TempDir::new();
    let path = temp.at("plain.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..20);
    collection.save(path.to_str().unwrap()).unwrap();

    let m = manifest(&path);
    assert_eq!(m["format_version"], json!("1.1.0"));
    assert!(
        m.as_object().unwrap().get("journal").is_none(),
        "the field is absent rather than null"
    );
    assert!(!journal_path(&path).unwrap().exists());
    assert!(Collection::load(path.to_str().unwrap()).is_ok());
}

/// A checkpoint syncs the journal, names the sequence it had reached, and
/// leaves the journal holding nothing.
#[test]
fn a_checkpoint_names_the_sequence_it_synced_and_empties_the_journal() {
    let temp = TempDir::new();
    let path = temp.at("checkpoint.zdb");
    let collection = journaled(&path);
    let wal = journal_path(&path).unwrap();

    add(&collection, 0..10);
    assert!(
        std::fs::metadata(&wal).unwrap().len() > zeusdb_vector_core::JOURNAL_HEADER_BYTES as u64,
        "ten records are in the journal"
    );

    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(
        collection.journal_sequence(),
        10,
        "ten records, ten sequences"
    );
    assert_eq!(manifest(&path)["journal"]["sequence"], json!(10));
    assert_eq!(
        std::fs::metadata(&wal).unwrap().len(),
        zeusdb_vector_core::JOURNAL_HEADER_BYTES as u64,
        "the truncation left the header alone"
    );

    // And it keeps going from there.
    add(&collection, 10..15);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 15);
    assert_eq!(manifest(&path)["journal"]["sequence"], json!(15));
}

// ============================================================================
// WHAT A RECOVERY GIVES BACK
// ============================================================================

/// A collection recovered from a checkpoint and the records past it is the
/// collection the script built, id for id and page for page.
#[test]
fn a_recovered_collection_is_the_collection_the_script_built() {
    let temp = TempDir::new();
    let path = temp.at("replay.zdb");
    let collection = journaled(&path);
    add(&collection, 0..30);
    collection.save(path.to_str().unwrap()).unwrap();
    // Past the checkpoint: every kind the script can reach without a space.
    add(&collection, 30..45);
    collection
        .remove_points(&["r3".to_string(), "r4".to_string()])
        .unwrap();
    collection
        .update_metadata("r5", HashMap::from([("cat".to_string(), json!("z"))]))
        .unwrap();
    collection.compact().unwrap();
    add(&collection, 45..50);
    collection
        .add_metadata(HashMap::from([("owner".to_string(), "test".to_string())]))
        .unwrap();

    let before = ids(&collection);
    let before_pages = page(&collection);
    let before_counter = collection.id_counter();
    drop(collection);

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.checkpoint_sequence, 30);
    assert_eq!(
        report.replayed, 24,
        "15 inserts, a removal, a replacement, a compaction, 5 inserts and the index metadata"
    );
    assert_eq!(report.skipped, 0, "the checkpoint truncated what it held");
    assert_eq!(report.damage, None);
    assert!(!report.graph_rebuilt);
    assert_eq!(ids(&recovered), before);
    assert_eq!(page(&recovered), before_pages);
    assert_eq!(recovered.id_counter(), before_counter);
    assert_eq!(recovered.metadata("owner").as_deref(), Some("test"));
}

/// A collection that never checkpoints again still opens, because opening
/// the journal wrote the checkpoint the whole journal replays onto.
#[test]
fn a_collection_that_never_saves_again_replays_in_full() {
    let temp = TempDir::new();
    let path = temp.at("never.zdb");
    let collection = journaled(&path);
    add(&collection, 0..25);
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.checkpoint_sequence, 0);
    assert_eq!(report.replayed, 25);
    assert_eq!(report.skipped, 0);
    assert_eq!(ids(&recovered), before);
}

/// The records the checkpoint already holds are skipped by sequence, which
/// is what makes a crash between a checkpoint and its truncation harmless.
///
/// The journal is put back to what it held before the save, which is the
/// file a process killed after the save and before the truncation leaves.
#[test]
fn records_at_or_below_the_checkpoints_sequence_are_skipped() {
    let temp = TempDir::new();
    let path = temp.at("skip.zdb");
    let collection = journaled(&path);
    let wal = journal_path(&path).unwrap();
    add(&collection, 0..8);
    let before_the_save = std::fs::read(&wal).unwrap();
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 8);
    let before = ids(&collection);
    drop(collection);

    std::fs::write(&wal, &before_the_save).unwrap();
    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.checkpoint_sequence, 8);
    assert_eq!(report.records_in_journal, 8);
    assert_eq!(report.skipped, 8, "the checkpoint already holds every one");
    assert_eq!(report.replayed, 0);
    assert_eq!(ids(&recovered), before);
    // The reopen appends after the last whole record rather than dropping
    // the eight, since cutting a journal is the checkpoint's to do and a
    // record below the checkpoint is skipped again at the next open. The
    // next checkpoint takes them.
    add(&recovered, 8..11);
    assert_eq!(recovered.journal_sequence(), 8);
    recovered.save(path.to_str().unwrap()).unwrap();
    assert_eq!(recovered.journal_sequence(), 11);
    drop(recovered);
    assert_eq!(
        std::fs::metadata(&wal).unwrap().len(),
        zeusdb_vector_core::JOURNAL_HEADER_BYTES as u64
    );
    let reread = std::fs::read(&wal).unwrap();
    let contents = zeusdb_vector_core::read_journal(&reread, "wal").unwrap();
    assert_eq!(contents.header.first_sequence, 12);
}

/// A crash between the truncation's two steps is recoverable, and the
/// records appended after the recovery survive without a further checkpoint.
///
/// The first step cuts the body and syncs it. The second rewrites the header
/// to name the next sequence as its first and syncs that. A process killed
/// between the two leaves an empty body under a header naming the old first
/// sequence, and the reopen has to restate the header before it appends.
///
/// What this avoids is checking the recovery by checkpointing again: a
/// checkpoint puts every record into the directory and truncates the journal,
/// so a header that was never restated would be replaced before anything read
/// it back and the run would look correct. Here the records appended after
/// the recovery are read back off the journal with no checkpoint between.
#[test]
fn a_crash_between_the_truncations_two_steps_is_recoverable_end_to_end() {
    let temp = TempDir::new();
    let path = temp.at("halfdone.zdb");
    let collection = journaled(&path);
    let wal = journal_path(&path).unwrap();
    add(&collection, 0..8);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 8);
    let collection_id = collection.collection_id();
    drop(collection);

    // The file the first step alone leaves: the body cut back to the header,
    // and the header still naming sequence 1 as its first.
    let after_the_first_step = zeusdb_vector_core::encode_journal_header(collection_id, 1).to_vec();
    std::fs::write(&wal, &after_the_first_step).unwrap();
    let read_back = zeusdb_vector_core::read_journal(&after_the_first_step, "wal").unwrap();
    assert!(read_back.records.is_empty());
    assert_eq!(read_back.header.first_sequence, 1);
    assert_eq!(read_back.damage, None);

    // The reopen completes the second step, and the records appended after it
    // carry the sequences the checkpoint left off at.
    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.checkpoint_sequence, 8);
    assert_eq!(report.records_in_journal, 0);
    assert_eq!(report.replayed, 0);
    add(&recovered, 8..14);
    let before = ids(&recovered);
    assert_eq!(
        recovered.journal_sequence(),
        8,
        "no checkpoint has run since"
    );
    drop(recovered);

    let restated = std::fs::read(&wal).unwrap();
    let contents = zeusdb_vector_core::read_journal(&restated, "wal").unwrap();
    assert_eq!(
        contents.header.first_sequence, 9,
        "the reopen restated the header the crash had left at 1"
    );
    assert_eq!(contents.records.len(), 6);
    assert_eq!(contents.damage, None);

    // And the six records are still there at the next open, which is what a
    // header left at 1 would have lost.
    let (reopened, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.checkpoint_sequence, 8);
    assert_eq!(report.replayed, 6);
    assert_eq!(ids(&reopened), before);
}

// ============================================================================
// WHAT PAIRS A DIRECTORY WITH ITS JOURNAL
// ============================================================================

/// A journal from another index is refused by content, whatever it is named.
#[test]
fn a_journal_from_another_index_is_refused() {
    let temp = TempDir::new();
    let one = temp.at("one.zdb");
    let two = temp.at("two.zdb");
    let a = journaled(&one);
    add(&a, 0..6);
    let b = journaled(&two);
    add(&b, 0..6);
    let a_id = a.collection_id();
    let b_id = b.collection_id();
    assert_ne!(a_id, b_id, "two collections draw two ids");
    drop(a);
    drop(b);

    // The other index's journal, under this one's name.
    std::fs::copy(journal_path(&two).unwrap(), journal_path(&one).unwrap()).unwrap();
    match Collection::recover(one.to_str().unwrap(), None, Durability::default()) {
        Err(Error::JournalNotThisCollection {
            journal_id,
            directory_id,
            ..
        }) => {
            assert_eq!(journal_id, format!("{b_id:032x}"));
            assert_eq!(directory_id, format!("{a_id:032x}"));
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A directory copied without its sibling is refused by name, and the
/// checkpoint alone opens when the caller asks for it.
#[test]
fn a_directory_whose_manifest_names_an_absent_journal_is_refused() {
    let temp = TempDir::new();
    let path = temp.at("copied.zdb");
    let collection = journaled(&path);
    add(&collection, 0..10);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 10..14);
    drop(collection);
    std::fs::remove_file(journal_path(&path).unwrap()).unwrap();

    match Collection::load(path.to_str().unwrap()) {
        Err(Error::JournalMissing {
            recorded, sequence, ..
        }) => {
            assert_eq!(recorded, "copied.zdb.zdbwal");
            assert_eq!(sequence, 10);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    let message = Collection::load(path.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("copied.zdb.zdbwal"), "{message}");

    // The checkpoint alone, by name, holds what the checkpoint held.
    let checkpoint = Collection::load_checkpoint_only(path.to_str().unwrap(), None).unwrap();
    assert_eq!(checkpoint.len(), 10);
    assert_eq!(checkpoint.journal_sequence(), 10);
}

/// A journal whose first record is above the one after the checkpoint's
/// sequence is refused, because the records between the two are in neither.
#[test]
fn a_journal_that_starts_above_the_checkpoint_is_refused() {
    let temp = TempDir::new();
    let path = temp.at("ahead.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 6..10);
    drop(collection);

    // The checkpoint names sequence 6. Put it back to 2, so the journal's
    // first record at 7 is four above what it may be.
    rewrite_manifest(&path, |m| m["journal"]["sequence"] = json!(2));
    match Collection::recover(path.to_str().unwrap(), None, Durability::default()) {
        Err(Error::JournalStartsAfterCheckpoint {
            first, checkpoint, ..
        }) => {
            assert_eq!(first, 7);
            assert_eq!(checkpoint, 2);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A record that changed after it was written, with records after it, is
/// refused rather than skipped.
#[test]
fn a_corrupt_middle_refuses_the_open() {
    let temp = TempDir::new();
    let path = temp.at("corrupt.zdb");
    let collection = journaled(&path);
    add(&collection, 0..5);
    add(&collection, 5..10);
    drop(collection);

    let wal = journal_path(&path).unwrap();
    let mut bytes = std::fs::read(&wal).unwrap();
    let contents = zeusdb_vector_core::read_journal(&bytes, "wal").unwrap();
    assert_eq!(contents.records.len(), 10);
    // One byte inside the third record's payload, which has seven records
    // after it to prove it landed.
    let victim = contents.records[2];
    let at = victim.offset as usize + zeusdb_vector_core::JOURNAL_RECORD_HEADER_BYTES + 4;
    let sequence = victim.sequence;
    bytes[at] ^= 0x5a;
    std::fs::write(&wal, &bytes).unwrap();

    match Collection::recover(path.to_str().unwrap(), None, Durability::default()) {
        Err(Error::JournalCorrupt {
            sequence: named, ..
        }) => assert_eq!(named, sequence),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A record that will not apply refuses the open, naming its sequence, and
/// nothing from it on is applied.
#[test]
fn a_record_that_does_not_belong_refuses_the_open_naming_its_sequence() {
    let temp = TempDir::new();
    let path = temp.at("mismatch.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);
    drop(collection);

    // A checkpoint claiming to hold the first two records, which it does
    // not, so the third record's internal id is two ahead of the counter.
    rewrite_manifest(&path, |m| m["journal"]["sequence"] = json!(2));
    match Collection::recover(path.to_str().unwrap(), None, Durability::default()) {
        Err(Error::JournalReplayFailed {
            sequence, detail, ..
        }) => {
            assert_eq!(sequence, 3);
            assert!(detail.contains("internal id"), "{detail}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

// ============================================================================
// THE FORMAT MAJOR
// ============================================================================

/// A manifest below the third major that names a journal is a directory
/// nothing wrote, and is refused rather than read.
#[test]
fn a_manifest_below_the_third_major_that_names_a_journal_is_refused() {
    let temp = TempDir::new();
    let path = temp.at("downgraded.zdb");
    let collection = journaled(&path);
    add(&collection, 0..5);
    collection.save(path.to_str().unwrap()).unwrap();
    drop(collection);

    rewrite_manifest(&path, |m| m["format_version"] = json!("1.1.0"));
    match Collection::load(path.to_str().unwrap()) {
        Err(Error::FormatVersionJournal { format_version }) => {
            assert_eq!(format_version, "1.1.0")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    rewrite_manifest(&path, |m| m["format_version"] = json!("2.0.0"));
    assert!(matches!(
        Collection::load(path.to_str().unwrap()),
        Err(Error::FormatVersionJournal { .. })
    ));
}

/// The third major reads what the first two did, and a fourth is refused
/// with the majors this build reads.
#[test]
fn the_third_major_is_read_and_a_fourth_is_not() {
    let temp = TempDir::new();
    let path = temp.at("majors.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..8);
    collection.save(path.to_str().unwrap()).unwrap();

    // A 3.x label on a directory with no journal opens, since a 3.x reader
    // reads everything the earlier majors wrote.
    rewrite_manifest(&path, |m| m["format_version"] = json!("3.0.0"));
    assert_eq!(Collection::load(path.to_str().unwrap()).unwrap().len(), 8);
    rewrite_manifest(&path, |m| m["format_version"] = json!("3.7.2"));
    assert!(Collection::load(path.to_str().unwrap()).is_ok());

    rewrite_manifest(&path, |m| m["format_version"] = json!("4.0.0"));
    let message = Collection::load(path.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("format version 4.0.0 cannot be opened"),
        "{message}"
    );
    assert!(message.contains("1.x, 2.x and 3.x"), "{message}");
}

/// A journaled collection holding a sparse space still declares the third
/// major, since the journal's is the later of the two.
#[test]
fn a_journaled_directory_with_a_space_takes_the_third_major() {
    let temp = TempDir::new();
    let path = temp.at("spaced.zdb");
    let declaration = declaration()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let collection = Collection::build(declaration, None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    let records: Vec<ParsedRecord> = (0..12usize)
        .map(|i| {
            let mut r = record(i);
            r.sparse = Some(super::SparseHalf::Vector(SparseVector {
                dims: vec![(i % 7) as u32, ((i * 3) % 11) as u32 + 11],
                values: vec![1.0, 2.0],
            }));
            r
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    collection.save(path.to_str().unwrap()).unwrap();
    let before = ids(&collection);
    drop(collection);

    assert_eq!(manifest(&path)["format_version"], json!("3.0.0"));
    let (recovered, _) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(ids(&recovered), before);
}

// ============================================================================
// WHAT A RECOVERY SAYS
// ============================================================================

/// A checkpoint whose dump could not be read replays onto a graph built from
/// the records, which is not the graph the crashed process held, and the
/// recovery reports it.
#[test]
fn a_checkpoint_whose_dump_was_refused_reports_the_rebuild() {
    let temp = TempDir::new();
    let path = temp.at("nodump.zdb");
    let collection = journaled(&path);
    add(&collection, 0..20);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 20..25);
    let before = ids(&collection);
    drop(collection);

    std::fs::remove_file(path.join(DUMP_FILENAME)).unwrap();
    rewrite_manifest(&path, |m| {
        m["files_included"] = json!(m["files_included"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f.as_str() != Some(DUMP_FILENAME))
            .cloned()
            .collect::<Vec<Value>>());
        m["file_digests"]
            .as_object_mut()
            .unwrap()
            .remove(DUMP_FILENAME);
    });

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert!(report.graph_rebuilt, "the dump was not read");
    assert_eq!(report.replayed, 5);
    // The rebuild keeps the ids the mappings hold, so the records and the
    // ids are the checkpoint's; the graph's topology is not.
    assert_eq!(ids(&recovered), before);
}

/// The policy is the caller's and the default is the one a caller who
/// names nothing gets.
#[test]
fn the_sink_takes_the_mode_it_is_given() {
    let temp = TempDir::new();
    let path = temp.at("mode.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::NoSync)
        .unwrap();
    add(&collection, 0..4);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 4..8);
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::NoSync).unwrap();
    assert_eq!(report.replayed, 4);
    assert_eq!(ids(&recovered), before);
    assert_eq!(
        Durability::default().commit_mode(),
        zeusdb_vector_core::CommitMode::Sync
    );
}

/// A collection is journaled once.
#[test]
fn a_collection_that_already_has_a_sink_is_not_journaled_again() {
    let temp = TempDir::new();
    let path = temp.at("once.zdb");
    let second = temp.at("twice.zdb");
    let collection = journaled(&path);
    add(&collection, 0..4);
    let message = collection
        .journal_to(second.to_str().unwrap(), Durability::default())
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("journaled once"), "{message}");
    assert!(!second.exists(), "nothing was written");
    assert!(!journal_path(&second).unwrap().exists());
    // And the first journal still takes records.
    add(&collection, 4..8);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 8);
}

/// A checkpoint saves to the directory its journal sits beside and to no
/// other, since the manifest it writes names that journal as a sibling.
#[test]
fn a_checkpoint_refuses_a_directory_its_journal_does_not_sit_beside() {
    let temp = TempDir::new();
    let path = temp.at("home.zdb");
    let elsewhere = temp.at("elsewhere.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);

    match collection.save(elsewhere.to_str().unwrap()) {
        Err(Error::JournalDirectoryMismatch { journal, target }) => {
            assert!(journal.ends_with("home.zdb.zdbwal"), "{journal}");
            assert_eq!(target, elsewhere.to_str().unwrap());
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(!elsewhere.exists(), "nothing was written");
    // Its own directory, named the long way round, is still its own.
    let same = path.parent().unwrap().join(".").join("home.zdb");
    collection.save(same.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 6);

    // And a collection with no journal saves wherever it is asked to.
    let plain = Collection::build(declaration(), None);
    add(&plain, 0..6);
    plain.save(elsewhere.to_str().unwrap()).unwrap();
    assert!(elsewhere.is_dir());
}

/// One page, so a recovered collection is compared by what a caller sees as
/// well as by its ids.
fn page(collection: &Collection) -> Vec<(String, f32)> {
    let params = collection.search_params(10, None, false, None).unwrap();
    collection
        .search_one(&[1.5, 3.0], None, params)
        .unwrap()
        .into_iter()
        .map(|hit| (hit.0, hit.1))
        .collect()
}
