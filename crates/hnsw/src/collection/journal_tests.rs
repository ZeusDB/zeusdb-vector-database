//! A directory and the journal beside it.
//!
//! What a checkpoint writes, what pairs the two, what is refused and what a
//! recovery gives back. The crashes are next door in `crash_tests`; nothing
//! here kills a process, and everything here is a rule a caller can reach by
//! copying, renaming or editing a directory or its journal.
//!
//! Every call's records are closed by a commit record, which takes a
//! sequence like any other, so a call of ten records reaches eleven
//! sequences.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};
use zeusdb_vector_core::{
    CommitMode, Error, FsStorage, JournalWriter, Operation, SparseVector, Storage, DUMP_FILENAME,
    MAX_INTERNAL_ID, SEGMENT_HEADER_BYTES,
};
use zeusdb_vector_sparse::SparseConfig;

use super::journal_support::{
    copy_journaled, copy_tree, expected_of, manifest, rewrite_manifest, Segments, TempDir,
};
use super::{Collection, Declaration, ParsedRecord};
use crate::journal::{journal_path, Durability};

// ============================================================================
// FIXTURES
// ============================================================================

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

/// Every id the collection holds, in order, with its internal id.
fn ids(collection: &Collection) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = collection
        .internal_ids()
        .into_iter()
        .map(|(internal, id)| (id, internal))
        .collect();
    out.sort();
    out
}

/// A journaled collection at `path`, with the empty checkpoint written.
fn journaled(dir: &Path) -> Collection {
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(dir.to_str().unwrap(), Durability::default())
        .unwrap();
    collection
}

fn recover(path: &Path) -> Result<(Collection, crate::Recovery), Error> {
    Collection::recover(path.to_str().unwrap(), None, Durability::default())
}

/// Empty the journal beside `path` of segments and put back `segments`,
/// each by its number.
fn restore_segments(path: &Path, segments: &Segments) {
    let journal = journal_path(path).unwrap();
    for entry in std::fs::read_dir(&journal).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    for (file, bytes) in segments.files.iter().zip(&segments.bytes) {
        std::fs::write(file, bytes).unwrap();
    }
}

/// Append, by hand, records closed by a commit to the journal beside `path`,
/// as a process writing a journal of its own would.
fn append_by_hand(path: &Path, operations: &[Operation]) {
    let segments = Segments::of(path);
    let expected = expected_of(path);
    let contents = segments.read(&expected).unwrap();
    let storage = FsStorage::at(path).unwrap();
    let mut writer = JournalWriter::resume(
        storage.journal_dir(),
        expected.collection_id,
        expected.base,
        &contents,
    )
    .unwrap();
    for operation in operations {
        let mut payload = Vec::new();
        operation.encode(&mut payload);
        writer.append(operation.kind(), &payload).unwrap();
    }
    writer.commit(CommitMode::Sync).unwrap();
}

// ============================================================================
// WHAT A CHECKPOINT WRITES
// ============================================================================

/// Turning the journal on writes the checkpoint it replays onto, and that
/// checkpoint names the journal, the collection and the sequence. The
/// journal is a directory holding one segment, a header naming the
/// checkpoint, and the lock file sits beside the directory.
#[test]
fn opening_a_journal_writes_the_empty_checkpoint_it_replays_onto() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("fresh.zdb");
    let collection = journaled(&path);

    assert!(path.is_dir(), "the checkpoint's directory exists");
    assert!(path.join("manifest.json").is_file());
    assert!(path.join("config.json").is_file());
    // An empty index dumps no graph, which is what a save of one has always
    // done.
    assert!(!path.join(DUMP_FILENAME).exists());
    let journal = journal_path(&path).unwrap();
    assert!(journal.is_dir(), "the journal is a directory beside it");
    assert_eq!(journal, temp.at("fresh.zdb.zdbjournal"));
    assert!(
        temp.at("fresh.zdb.zdblock").is_file(),
        "the lock file beside it"
    );
    let segments = Segments::of(&path);
    assert_eq!(segments.numbers, vec![1]);
    assert!(segments.files[0].ends_with("00000000000000000001.zdbseg"));
    assert_eq!(
        segments.total_bytes(),
        SEGMENT_HEADER_BYTES as u64,
        "a journal with no record is one segment's header alone"
    );
    let header = zeusdb_vector_core::read_segment_header(&segments.bytes[0], "s").unwrap();
    assert_eq!(
        header.base,
        expected_of(&path).base,
        "it names the checkpoint"
    );

    let m = manifest(&path);
    assert_eq!(m["format_version"], json!("5.0.0"));
    assert_eq!(m["features"], json!({"journal": "incompatible"}));
    assert_eq!(m["journal"]["file"], json!("fresh.zdb.zdbjournal"));
    assert_eq!(m["journal"]["sequence"], json!(0));
    assert_eq!(m["identity"]["sequence"], json!(0));
    assert_eq!(
        m["journal"]["collection_id"],
        json!(format!("{:032x}", collection.collection_id()))
    );
    assert_eq!(
        m["journal"]["collection_id"],
        m["identity"]["collection_id"]
    );
    assert_eq!(collection.journal_sequence(), 0);
}

/// A save of a collection holding no journal lists no feature, carries no
/// `journal` field at all, and records sequence zero in its identity.
#[test]
fn a_directory_saved_without_a_journal_carries_no_journal_field() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("plain.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..20);
    collection.save(path.to_str().unwrap()).unwrap();

    let m = manifest(&path);
    assert_eq!(m["features"], json!({}));
    assert_eq!(m["identity"]["sequence"], json!(0));
    assert!(
        m.as_object().unwrap().get("journal").is_none(),
        "the field is absent rather than null"
    );
    assert!(!journal_path(&path).unwrap().exists());
    assert!(Collection::load(path.to_str().unwrap()).is_ok());
}

/// A checkpoint syncs the journal, names the sequence it had reached, and
/// rolls the journal onto one segment naming it, which holds nothing.
#[test]
fn a_checkpoint_names_the_sequence_it_synced_and_rolls_the_journal() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("checkpoint.zdb");
    let collection = journaled(&path);

    add(&collection, 0..10);
    assert!(
        Segments::of(&path).total_bytes() > SEGMENT_HEADER_BYTES as u64,
        "ten records are in the journal"
    );

    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(
        collection.journal_sequence(),
        11,
        "ten records and their commit, eleven sequences"
    );
    let m = manifest(&path);
    assert_eq!(m["journal"]["sequence"], json!(11));
    assert_eq!(m["identity"]["sequence"], json!(11));
    let segments = Segments::of(&path);
    assert_eq!(
        segments.numbers,
        vec![12],
        "the roll deleted the segment it held"
    );
    assert_eq!(segments.total_bytes(), SEGMENT_HEADER_BYTES as u64);
    let header = zeusdb_vector_core::read_segment_header(&segments.bytes[0], "s").unwrap();
    assert_eq!(header.base, expected_of(&path).base);
    assert_eq!(header.base.sequence, 11);

    // And it keeps going from there.
    add(&collection, 10..15);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 17);
    assert_eq!(manifest(&path)["journal"]["sequence"], json!(17));
    assert_eq!(Segments::of(&path).numbers, vec![18]);
}

// ============================================================================
// WHAT A RECOVERY GIVES BACK
// ============================================================================

/// A collection recovered from a checkpoint and the records past it is the
/// collection the script built, id for id and page for page.
#[test]
fn a_recovered_collection_is_the_collection_the_script_built() {
    let temp = TempDir::new("journal-tests");
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

    let (recovered, report) = recover(&path).unwrap();
    assert_eq!(report.checkpoint_sequence, 31);
    assert_eq!(
        report.replayed, 24,
        "15 inserts, a removal, a replacement, a compaction, 5 inserts and the index metadata"
    );
    assert_eq!(report.withdrawn, 0);
    assert_eq!(report.skipped, 0, "the checkpoint rolled past what it held");
    assert_eq!(report.damage, None);
    assert_eq!(report.segments, 1);
    assert!(!report.graph_rebuilt);
    assert_eq!(report.checkpoint_taken, None);
    assert_eq!(report.format_version, "5.0.0");
    assert_eq!(ids(&recovered), before);
    assert_eq!(page(&recovered), before_pages);
    assert_eq!(recovered.id_counter(), before_counter);
    assert_eq!(recovered.metadata("owner").as_deref(), Some("test"));
}

/// A collection that never checkpoints again still opens, because opening
/// the journal wrote the checkpoint the whole journal replays onto.
#[test]
fn a_collection_that_never_saves_again_replays_in_full() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("never.zdb");
    let collection = journaled(&path);
    add(&collection, 0..25);
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) = recover(&path).unwrap();
    assert_eq!(report.checkpoint_sequence, 0);
    assert_eq!(report.replayed, 25);
    assert_eq!(report.skipped, 0);
    assert_eq!(report.first_sequence, 1);
    assert_eq!(ids(&recovered), before);
}

/// A process killed between a checkpoint's commit and its roll leaves the
/// segments the checkpoint holds and none naming it. They are stale: every
/// record in them is skipped, the open creates the segment the roll would
/// have, deletes them, and the records appended after it survive without a
/// further checkpoint.
#[test]
fn segments_the_checkpoint_holds_are_skipped_and_go_at_the_next_open() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("skip.zdb");
    let collection = journaled(&path);
    add(&collection, 0..8);
    let before_the_save = Segments::of(&path);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 9);
    let before = ids(&collection);
    drop(collection);

    restore_segments(&path, &before_the_save);
    assert_eq!(Segments::of(&path).numbers, vec![1]);
    let (recovered, report) = recover(&path).unwrap();
    assert_eq!(report.checkpoint_sequence, 9);
    assert_eq!(report.skipped, 8, "the checkpoint already holds every one");
    assert_eq!(report.replayed, 0);
    assert_eq!(report.records_in_journal, 0);
    assert_eq!(ids(&recovered), before);
    assert_eq!(
        Segments::of(&path).numbers,
        vec![10],
        "the stale segment went and the one naming the checkpoint was created"
    );

    add(&recovered, 8..11);
    assert_eq!(
        recovered.journal_sequence(),
        9,
        "no checkpoint has run since"
    );
    let after = ids(&recovered);
    drop(recovered);
    let (reopened, report) = recover(&path).unwrap();
    assert_eq!(report.replayed, 3);
    assert_eq!(ids(&reopened), after);
}

/// A process killed inside a checkpoint's deletion leaves the segment naming
/// it beside some it holds, and the open deletes those.
#[test]
fn a_stale_segment_beside_the_new_one_goes_at_the_next_open() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("beside.zdb");
    let collection = journaled(&path);
    add(&collection, 0..5);
    let old = Segments::of(&path);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 5..7);
    let before = ids(&collection);
    drop(collection);

    std::fs::write(&old.files[0], &old.bytes[0]).unwrap();
    assert_eq!(Segments::of(&path).numbers, vec![1, 7]);
    let (recovered, report) = recover(&path).unwrap();
    assert_eq!(report.skipped, 5);
    assert_eq!(report.replayed, 2);
    assert_eq!(ids(&recovered), before);
    assert_eq!(Segments::of(&path).numbers, vec![7]);
}

// ============================================================================
// WHAT PAIRS A DIRECTORY WITH ITS JOURNAL
// ============================================================================

/// A journal from another index is refused by content, whatever it is named.
#[test]
fn a_journal_from_another_index_is_refused() {
    let temp = TempDir::new("journal-tests");
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
    std::fs::remove_dir_all(journal_path(&one).unwrap()).unwrap();
    copy_tree(&journal_path(&two).unwrap(), &journal_path(&one).unwrap());
    match recover(&one) {
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

/// A journal of another history of the same collection is refused naming
/// both checkpoints, which is a journal written after another checkpoint of
/// the collection and copied beside this one.
///
/// A takes five records into its journal alone. Its checkpoint is opened
/// alone, which keeps the collection's id, and journaled to B, whose journal
/// takes the index metadata alone, so B's id counter stays at A's. A's
/// journal copied over B's names A's checkpoint and holds records above B's
/// sequence.
#[test]
fn a_journal_of_another_history_of_the_collection_is_refused_naming_both_checkpoints() {
    let temp = TempDir::new("journal-tests");
    let a_path = temp.at("a.zdb");
    let b_path = temp.at("b.zdb");
    let a = journaled(&a_path);
    add(&a, 0..5);
    drop(a);
    let b = Collection::load_checkpoint_only(a_path.to_str().unwrap(), None).unwrap();
    b.journal_to(b_path.to_str().unwrap(), Durability::default())
        .unwrap();
    b.add_metadata(HashMap::from([("owner".to_string(), "b".to_string())]))
        .unwrap();
    assert_eq!(b.collection_id(), expected_of(&a_path).collection_id);
    drop(b);
    let a_snapshot = expected_of(&a_path).base.snapshot;
    let b_snapshot = expected_of(&b_path).base.snapshot;
    assert_ne!(a_snapshot, b_snapshot);

    std::fs::remove_dir_all(journal_path(&b_path).unwrap()).unwrap();
    copy_tree(
        &journal_path(&a_path).unwrap(),
        &journal_path(&b_path).unwrap(),
    );
    match recover(&b_path) {
        Err(Error::JournalNotThisCheckpoint { detail, .. }) => {
            assert!(detail.contains(&format!("{a_snapshot:032x}")), "{detail}");
            assert!(detail.contains(&format!("{b_snapshot:032x}")), "{detail}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A journal copied beside an older directory of its own collection, and
/// one of another branch copied beside a newer directory, are each refused,
/// since each holds records above the directory's sequence after another
/// checkpoint than the directory's.
#[test]
fn a_journal_copied_beside_an_older_or_a_newer_directory_is_refused() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("lineage.zdb");
    let collection = journaled(&path);
    add(&collection, 0..4);
    collection.save(path.to_str().unwrap()).unwrap();
    // The directory and journal as the first checkpoint left them.
    let first = copy_journaled(&path, "first.zdb");
    add(&collection, 4..8);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 8..10);
    drop(collection);

    // The current journal, after the second checkpoint, beside the first.
    let older = temp.at("older.zdb");
    copy_tree(&first, &older);
    copy_tree(
        &journal_path(&path).unwrap(),
        &journal_path(&older).unwrap(),
    );
    assert!(matches!(
        recover(&older),
        Err(Error::JournalNotThisCheckpoint { .. })
    ));

    // A branch from the first checkpoint, which took records of its own,
    // beside the second.
    let (branch, _) = recover(&first).unwrap();
    add(&branch, 20..30);
    drop(branch);
    let newer = temp.at("newer.zdb");
    copy_tree(&path, &newer);
    copy_tree(
        &journal_path(&first).unwrap(),
        &journal_path(&newer).unwrap(),
    );
    assert!(matches!(
        recover(&newer),
        Err(Error::JournalNotThisCheckpoint { .. })
    ));
}

/// A directory copied without its journal is refused by name, and the
/// checkpoint alone opens when the caller asks for it.
#[test]
fn a_directory_whose_manifest_names_an_absent_journal_is_refused() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("copied.zdb");
    let collection = journaled(&path);
    add(&collection, 0..10);
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 10..14);
    drop(collection);
    std::fs::remove_dir_all(journal_path(&path).unwrap()).unwrap();

    match Collection::load(path.to_str().unwrap()) {
        Err(Error::JournalMissing {
            recorded, sequence, ..
        }) => {
            assert_eq!(recorded, "copied.zdb.zdbjournal");
            assert_eq!(sequence, 11);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    let message = Collection::load(path.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("copied.zdb.zdbjournal"), "{message}");

    // The checkpoint alone, by name, holds what the checkpoint held.
    let checkpoint = Collection::load_checkpoint_only(path.to_str().unwrap(), None).unwrap();
    assert_eq!(checkpoint.len(), 10);
    assert_eq!(checkpoint.journal_sequence(), 11);
}

/// A journal whose segments after the checkpoint do not begin at the
/// sequence after its own is refused, and so is one with a segment missing
/// between two, because the records between are in neither.
#[test]
fn a_journal_missing_a_segment_is_refused() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("gap.zdb");
    zeusdb_vector_core::set_segment_roll_bytes(256);
    let collection = journaled(&path);
    zeusdb_vector_core::set_segment_roll_bytes(0);
    for i in 0..4 {
        add(&collection, i..i + 1);
    }
    drop(collection);
    let segments = Segments::of(&path);
    assert!(segments.numbers.len() >= 3, "{:?}", segments.numbers);

    // The second segment gone, between the first and the third.
    std::fs::remove_file(&segments.files[1]).unwrap();
    assert!(matches!(
        recover(&path),
        Err(Error::JournalSegmentsInvalid { .. })
    ));
    // The first gone, so the journal begins above the checkpoint.
    std::fs::remove_file(&segments.files[0]).unwrap();
    match recover(&path) {
        Err(Error::JournalStartsAfterCheckpoint {
            first, checkpoint, ..
        }) => {
            assert_eq!(first, segments.numbers[2]);
            assert_eq!(checkpoint, 0);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A record that changed after it was written, with records after it, is
/// refused rather than skipped.
#[test]
fn a_corrupt_middle_refuses_the_open() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("corrupt.zdb");
    let collection = journaled(&path);
    add(&collection, 0..5);
    add(&collection, 5..10);
    drop(collection);

    let segments = Segments::of(&path);
    let (at, sequence) = {
        let contents = segments.read(&expected_of(&path)).unwrap();
        assert_eq!(contents.records.len(), 10);
        // One byte inside the third record's payload, which has seven
        // records after it to prove it landed.
        let victim = contents.records[2].record;
        (
            victim.offset as usize + zeusdb_vector_core::JOURNAL_RECORD_HEADER_BYTES + 4,
            victim.sequence,
        )
    };
    let mut bytes = segments.bytes[0].clone();
    bytes[at] ^= 0x5a;
    std::fs::write(&segments.files[0], &bytes).unwrap();

    match recover(&path) {
        Err(Error::JournalCorrupt {
            sequence: named, ..
        }) => assert_eq!(named, sequence),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// An insert naming an internal id two ahead of the counter, appended by
/// hand.
fn insert_out_of_turn(internal_id: u64) -> Operation {
    Operation::Insert {
        id: "intruder".to_string(),
        internal_id,
        level: 0,
        vector: vec![0.5, 0.5],
        metadata: serde_json::Map::new(),
        sparse: None,
    }
}

/// A record that will not apply refuses the open, naming its sequence, and
/// nothing from it on is applied.
#[test]
fn a_record_that_does_not_belong_refuses_the_open_naming_its_sequence() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("mismatch.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);
    drop(collection);

    append_by_hand(&path, &[insert_out_of_turn(9)]);
    match recover(&path) {
        Err(Error::JournalReplayFailed {
            sequence, detail, ..
        }) => {
            assert_eq!(sequence, 8, "six inserts and a commit before it");
            assert!(detail.contains("internal id"), "{detail}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A refusal to replay says each sentence once: what refused the record, as
/// whole sentences each ending in a full stop, and then what was applied. A
/// record that does not belong gives its first sentence alone, and words that
/// end without a full stop are given one.
#[test]
fn a_refusal_to_replay_says_each_sentence_once() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("sentences.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);
    drop(collection);
    append_by_hand(&path, &[insert_out_of_turn(9)]);
    let mismatch = match recover(&path) {
        Err(error @ Error::JournalReplayFailed { .. }) => error.to_string(),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    let unpunctuated = Error::JournalReplayFailed {
        file: "00000000000000000001.zdbseg".to_string(),
        sequence: 4,
        detail: Error::Engine("PQ not available".to_string()).replay_detail(),
    }
    .to_string();
    let last = " Nothing from that record on was applied and the directory was not opened.";
    for message in [&mismatch, &unpunctuated] {
        assert_eq!(message.matches("Nothing from").count(), 1, "{message}");
        assert!(message.ends_with(last), "{message}");
        assert!(
            message[..message.len() - last.len()].ends_with('.'),
            "{message}"
        );
        assert!(!message.contains(".."), "{message}");
    }
    assert!(
        mismatch.contains("do not belong together: the record names internal id 9"),
        "{mismatch}"
    );
    assert!(
        mismatch.contains("00000000000000000001.zdbseg"),
        "the refusal names the segment: {mismatch}"
    );
    assert!(
        unpunctuated.contains("could not be applied: PQ not available. Nothing from that"),
        "{unpunctuated}"
    );
}

/// A commit or an abort record reaching `apply` is refused, since the reader
/// settles a call by them and hands none of them on.
#[test]
fn a_framing_record_does_not_apply() {
    let collection = Collection::build(declaration(), None);
    for operation in [
        Operation::Commit { first: 1 },
        Operation::Abort {
            first: 1,
            reason: zeusdb_vector_core::AbortReason::SyncFailed,
        },
    ] {
        for applied in [
            collection.apply(operation.clone()),
            collection.apply_withdrawn(operation),
        ] {
            assert!(
                matches!(applied, Err(Error::JournalReplayMismatch { .. })),
                "{applied:?}"
            );
        }
    }
}

// ============================================================================
// THE LAST INTERNAL ID
// ============================================================================

/// A journaled collection that has issued its last internal id refuses the
/// next record before the journal holds it, so its directory opens with
/// what it held and refuses the same record again. `clear` starts the ids
/// again from 1, and the directory opens after that too.
#[test]
fn a_journaled_collection_at_its_last_internal_id_refuses_an_add_and_still_opens() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("last.zdb");
    let mut collection = Collection::build(declaration(), None);
    add(&collection, 0..3);
    collection.set_counters(MAX_INTERNAL_ID, 3);
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    let journal = Segments::of(&path).bytes;

    let refused = |collection: &Collection, record: ParsedRecord, overwrite: bool| {
        let added = collection.add_records(vec![record], vec![], overwrite);
        assert_eq!(added.total_errors, 1);
        assert!(
            added.errors[0].contains("has issued its last internal id, 4294967294"),
            "{}",
            added.errors[0]
        );
    };
    refused(&collection, record(3), false);
    refused(&collection, record(0), true);
    assert_eq!(
        Segments::of(&path).bytes,
        journal,
        "nothing reached the journal"
    );
    assert_eq!(collection.id_counter(), MAX_INTERNAL_ID);
    assert!(collection.contains("r0"));

    // A removal still records and replays.
    collection.remove_points(&["r1".to_string()]).unwrap();
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) = recover(&path).unwrap();
    assert_eq!(report.replayed, 1, "the removal alone");
    assert_eq!(ids(&recovered), before);
    assert_eq!(recovered.id_counter(), MAX_INTERNAL_ID);
    refused(&recovered, record(3), false);

    // clear() starts the ids again, and what follows it replays.
    recovered.clear().unwrap();
    add(&recovered, 10..12);
    let after = vec![("r10".to_string(), 1), ("r11".to_string(), 2)];
    assert_eq!(ids(&recovered), after);
    drop(recovered);
    let (again, report) = recover(&path).unwrap();
    assert_eq!(report.replayed, 4, "the removal, the clear and two inserts");
    assert_eq!(ids(&again), after);
    assert_eq!(again.id_counter(), 2);
}

// ============================================================================
// THE FORMAT MAJOR
// ============================================================================

/// A manifest below the third major that names a journal is a directory
/// nothing wrote, and is refused rather than read.
#[test]
fn a_manifest_below_the_third_major_that_names_a_journal_is_refused() {
    let temp = TempDir::new("journal-tests");
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

/// Every major this build reads opens a directory it wrote, and a sixth is
/// refused with the majors this build reads. A 4.x label lists the identity
/// among its features, as a 4.x save does.
#[test]
fn every_major_up_to_the_fifth_is_read_and_a_sixth_is_not() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("majors.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..8);
    collection.save(path.to_str().unwrap()).unwrap();

    // A 3.x label on a directory with no journal opens, since its four
    // binary artefacts are read in the frame wherever one is found.
    rewrite_manifest(&path, |m| m["format_version"] = json!("3.0.0"));
    assert_eq!(Collection::load(path.to_str().unwrap()).unwrap().len(), 8);
    rewrite_manifest(&path, |m| m["format_version"] = json!("3.7.2"));
    assert!(Collection::load(path.to_str().unwrap()).is_ok());
    rewrite_manifest(&path, |m| {
        m["format_version"] = json!("4.9.0");
        m["features"]["identity"] = json!("compatible");
    });
    assert!(Collection::load(path.to_str().unwrap()).is_ok());
    rewrite_manifest(&path, |m| {
        m["format_version"] = json!("5.3.1");
        m["features"].as_object_mut().unwrap().remove("identity");
    });
    assert!(Collection::load(path.to_str().unwrap()).is_ok());

    rewrite_manifest(&path, |m| m["format_version"] = json!("6.0.0"));
    let message = Collection::load(path.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("format version 6.0.0 cannot be opened"),
        "{message}"
    );
    assert!(message.contains("1.x, 2.x, 3.x, 4.x and 5.x"), "{message}");
}

/// A 5.x manifest lists its features, records its identity, and records the
/// journal sequence its state holds in the identity and again in its journal
/// record, and one missing any of them, or whose two sequences disagree, is
/// refused.
#[test]
fn a_5x_manifest_missing_what_every_save_records_is_refused() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("required.zdb");
    let collection = journaled(&path);
    add(&collection, 0..4);
    collection.save(path.to_str().unwrap()).unwrap();
    drop(collection);
    let original = std::fs::read(path.join("manifest.json")).unwrap();
    let restore = || std::fs::write(path.join("manifest.json"), &original).unwrap();

    rewrite_manifest(&path, |m| {
        m.as_object_mut().unwrap().remove("features");
    });
    match Collection::load(path.to_str().unwrap()) {
        Err(Error::FeaturesInvalid { detail, .. }) => {
            assert!(detail.contains("carries no features record"), "{detail}")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    restore();
    rewrite_manifest(&path, |m| {
        m.as_object_mut().unwrap().remove("identity");
    });
    assert!(matches!(
        Collection::load(path.to_str().unwrap()),
        Err(Error::IdentityInvalid { .. })
    ));
    restore();
    rewrite_manifest(&path, |m| {
        m["identity"].as_object_mut().unwrap().remove("sequence");
    });
    match Collection::load(path.to_str().unwrap()) {
        Err(Error::IdentityInvalid { detail }) => {
            assert!(detail.contains("records no sequence"), "{detail}")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    restore();
    rewrite_manifest(&path, |m| m["journal"]["sequence"] = json!(2));
    match Collection::load(path.to_str().unwrap()) {
        Err(Error::JournalManifestInvalid { detail }) => {
            assert!(detail.contains("its sequence is 2"), "{detail}")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    restore();
    rewrite_manifest(&path, |m| m["features"]["identity"] = json!("compatible"));
    assert!(matches!(
        Collection::load(path.to_str().unwrap()),
        Err(Error::FeaturesInvalid { .. })
    ));
    restore();
    assert_eq!(Collection::load(path.to_str().unwrap()).unwrap().len(), 4);
}

/// A journaled collection holding a sparse space lists both among its
/// features, and recovers to the records it held.
#[test]
fn a_journaled_directory_with_a_space_lists_both_features() {
    let temp = TempDir::new("journal-tests");
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

    assert_eq!(
        manifest(&path)["features"],
        json!({"journal": "incompatible", "sparse": "incompatible"})
    );
    let (recovered, _) = recover(&path).unwrap();
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
    let temp = TempDir::new("journal-tests");
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

    let (recovered, report) = recover(&path).unwrap();
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
    let temp = TempDir::new("journal-tests");
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

/// A collection is journaled once, and the second call writes nothing, its
/// lock file among it.
#[test]
fn a_collection_that_already_has_a_sink_is_not_journaled_again() {
    let temp = TempDir::new("journal-tests");
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
    assert!(!temp.at("twice.zdb.zdblock").exists());
    // And the first journal still takes records.
    add(&collection, 4..8);
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 10);
}

/// A checkpoint saves to the directory its journal sits beside and to no
/// other, since the manifest it writes names that journal as a sibling.
#[test]
fn a_checkpoint_refuses_a_directory_its_journal_does_not_sit_beside() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("home.zdb");
    let elsewhere = temp.at("elsewhere.zdb");
    let collection = journaled(&path);
    add(&collection, 0..6);

    match collection.save(elsewhere.to_str().unwrap()) {
        Err(Error::JournalDirectoryMismatch { journal, target }) => {
            assert!(journal.ends_with("home.zdb.zdbjournal"), "{journal}");
            assert_eq!(target, elsewhere.to_str().unwrap());
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(!elsewhere.exists(), "nothing was written");
    // Its own directory, named the long way round, is still its own.
    let same = path.parent().unwrap().join(".").join("home.zdb");
    collection.save(same.to_str().unwrap()).unwrap();
    assert_eq!(collection.journal_sequence(), 7);

    // And a collection with no journal saves wherever it is asked to.
    let plain = Collection::build(declaration(), None);
    add(&plain, 0..6);
    plain.save(elsewhere.to_str().unwrap()).unwrap();
    assert!(elsewhere.is_dir());
}

/// The journal's directory takes no part in a directory's own listing: a
/// save writes the directory and the lock file beside it, and nothing else.
#[test]
fn an_unjournaled_save_writes_the_directory_and_its_lock_file_alone() {
    let temp = TempDir::new("journal-tests");
    let path = temp.at("alone.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..3);
    collection.save(path.to_str().unwrap()).unwrap();
    let mut beside: Vec<String> = std::fs::read_dir(&temp.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    beside.sort();
    assert_eq!(beside, ["alone.zdb", "alone.zdb.zdblock"]);
    let storage = FsStorage::at(&path).unwrap();
    assert!(!storage.journal_exists());
    assert!(!storage.legacy_journal_exists());
}

/// One page, so a recovered collection is compared by what a caller sees as
/// well as by its ids.
fn page(collection: &Collection) -> Vec<(String, f32)> {
    let params = collection.search_params(10, None, false, None).unwrap();
    collection
        .search_one(&[1.5, 3.0], None, params)
        .unwrap()
        .iter()
        .map(|hit| (hit.id().to_string(), hit.score()))
        .collect()
}
