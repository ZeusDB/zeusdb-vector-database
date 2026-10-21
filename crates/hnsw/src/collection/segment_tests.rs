//! The sink over a journal of segments.
//!
//! A segment rolls at a commit once it passes the roll length, which the
//! tests set low through the override debug builds carry, so a journal of
//! several segments is reached with a few records. A write that fails part
//! way is cut back, and one whose cut fails too stops the sink until a
//! checkpoint. A sink in a process other than the one that opened it
//! refuses everything.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use zeusdb_vector_core::{set_segment_roll_bytes, Error, FsStorage, SEGMENT_HEADER_BYTES};

use super::journal_support::{
    copy_journaled, expected_of, sink_over, FaultyStorage, Segments, TempDir,
};
use super::{Collection, Declaration, OperationSink, ParsedRecord};
use crate::journal::Durability;

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

fn recover(path: &Path) -> (Collection, crate::Recovery) {
    Collection::recover(path.to_str().unwrap(), None, Durability::PerCall).unwrap()
}

/// A journaled collection whose segments roll at `bytes`.
fn rolling(path: &Path, bytes: u64, durability: Durability) -> Collection {
    set_segment_roll_bytes(bytes);
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), durability)
        .unwrap();
    set_segment_roll_bytes(0);
    collection
}

/// Records spread over segments read back as one journal, the stats count
/// every segment and every byte, a checkpoint rolls past all of them onto
/// one, and a load opens a journal of several segments and appends to the
/// last.
#[test]
fn a_journal_of_several_segments_reads_back_as_one() {
    let temp = TempDir::new("segment-tests");
    let path = temp.at("rolled.zdb");
    let collection = rolling(&path, 300, Durability::PerCall);
    for i in 0..12 {
        add(&collection, i..i + 1);
    }
    let segments = Segments::of(&path);
    assert!(segments.numbers.len() >= 4, "{:?}", segments.numbers);
    let status = collection.journal_status().unwrap();
    assert_eq!(status.segments, segments.numbers.len());
    assert_eq!(status.bytes, Some(segments.total_bytes()));
    assert_eq!(status.records_since_checkpoint(), 12);
    let stats = collection.stats();
    assert_eq!(
        stats.get("journal_segments"),
        Some(&segments.numbers.len().to_string())
    );
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) = recover(&path);
    assert_eq!(report.replayed, 12);
    assert_eq!(report.segments, segments.numbers.len());
    assert_eq!(ids(&recovered), before);
    add(&recovered, 12..13);
    assert_eq!(
        Segments::of(&path).numbers.len(),
        segments.numbers.len(),
        "a reopened writer appends to the last segment"
    );
    recovered.checkpoint().unwrap();
    let after = Segments::of(&path);
    assert_eq!(
        after.numbers.len(),
        1,
        "the checkpoint rolled past every segment"
    );
    assert_eq!(after.total_bytes(), SEGMENT_HEADER_BYTES as u64);
    assert_eq!(
        zeusdb_vector_core::read_segment_header(&after.bytes[0], "s")
            .unwrap()
            .base,
        expected_of(&path).base
    );
}

/// The interval policy's thread follows the active segment through every
/// roll, so what it syncs is the segment records land in, and a recovery
/// holds every record the calls returned from.
#[test]
fn the_interval_thread_follows_the_active_segment() {
    let temp = TempDir::new("segment-tests");
    let path = temp.at("interval.zdb");
    let collection = rolling(
        &path,
        300,
        Durability::PerInterval(Duration::from_millis(2)),
    );
    for i in 0..10 {
        add(&collection, i..i + 1);
        std::thread::sleep(Duration::from_millis(5));
    }
    let watch = collection.flusher_watch().unwrap();
    assert!(watch.syncs().unwrap() > 0);
    assert!(Segments::of(&path).numbers.len() > 2);
    let before = ids(&collection);
    drop(collection);
    let (recovered, report) = recover(&path);
    assert_eq!(report.replayed, 10);
    assert_eq!(ids(&recovered), before);
}

/// A write that fails part way is cut back, so the record after it in the
/// same call follows the last whole one and is acknowledged with the call,
/// and a crash afterwards recovers it. One more call before the crash is
/// recovered too, where a write that was not cut back would have turned the
/// journal into one no load opens.
#[test]
fn a_write_that_fails_part_way_is_cut_back_and_what_follows_survives() {
    let temp = TempDir::new("segment-tests");
    let path = temp.at("cut.zdb");
    let collection = Collection::build(declaration(), None);
    let storage = FaultyStorage::at(&path);
    let faults = storage.faults.clone();
    collection.attach_sink(Box::new(sink_over(
        &storage,
        &path,
        collection.collection_id(),
        Durability::PerCall,
    )));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..3);

    // r3's write lands half its bytes and fails; r4 follows in the call.
    faults.writes.arm(0, 1);
    let added = collection.add_records(vec![record(3), record(4)], vec![], false);
    assert_eq!(added.total_errors, 1, "{:?}", added.errors);
    assert!(
        added.errors[0].starts_with("Vector r3:"),
        "{}",
        added.errors[0]
    );
    assert!(
        added.errors[0].contains("were cut back"),
        "{}",
        added.errors[0]
    );
    assert!(collection.contains("r4"));
    let after_r4 = copy_journaled(&path, "after-r4.zdb");
    add(&collection, 5..6);
    let after_r5 = copy_journaled(&path, "after-r5.zdb");

    let (recovered, report) = recover(&after_r4);
    assert_eq!(report.damage, None, "nothing of r3 is left");
    assert_eq!(ids(&recovered), ["r0", "r1", "r2", "r4"]);
    let (recovered, _) = recover(&after_r5);
    assert_eq!(ids(&recovered), ["r0", "r1", "r2", "r4", "r5"]);
    assert_eq!(ids(&recovered), ids(&collection));
}

/// A write that fails part way and whose cut back fails too leaves the
/// sink taking nothing more, so every record after it is refused, until a
/// checkpoint rolls the journal onto a new segment.
#[test]
fn a_failed_cut_stops_the_sink_until_a_checkpoint() {
    let temp = TempDir::new("segment-tests");
    let path = temp.at("stuck.zdb");
    let collection = Collection::build(declaration(), None);
    let storage = FaultyStorage::at(&path);
    let faults = storage.faults.clone();
    collection.attach_sink(Box::new(sink_over(
        &storage,
        &path,
        collection.collection_id(),
        Durability::PerCall,
    )));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..2);

    faults.writes.arm(0, 1);
    faults.cuts.arm(0, 1);
    let added = collection.add_records(vec![record(2), record(3)], vec![], false);
    assert_eq!(added.total_errors, 2, "{:?}", added.errors);
    assert!(
        added.errors[0].contains("could not be cut back"),
        "{}",
        added.errors[0]
    );
    assert!(
        added.errors[1].contains("takes nothing more until a checkpoint"),
        "{}",
        added.errors[1]
    );
    match collection.remove_points(&["r0".to_string()]) {
        Err(Error::JournalIoFailed { error, .. }) => {
            assert!(error.contains("takes nothing more"), "{error}")
        }
        other => panic!("expected the refusal, got {other:?}"),
    }
    collection.checkpoint().unwrap();
    add(&collection, 4..6);
    let before = ids(&collection);
    drop(collection);
    let (recovered, report) = recover(&path);
    assert_eq!(report.replayed, 2);
    assert_eq!(ids(&recovered), before);
}

/// A sink in a process other than the one that opened it, which is what a
/// child forked from it is, refuses every append, commit, sync and roll
/// with `JournalForked` and writes nothing. In its own process the same
/// sink takes its checkpoint and its records.
#[test]
fn a_forked_sink_refuses_everything() {
    let temp = TempDir::new("segment-tests");
    let path = temp.at("forked.zdb");
    let collection = Collection::build(declaration(), None);
    let storage = FsStorage::at(&path).unwrap();
    let mut sink = sink_over(
        &storage,
        &path,
        collection.collection_id(),
        Durability::PerCall,
    );
    sink.pretend_forked();
    let forked = |result: Result<(), Error>| {
        assert!(
            matches!(result, Err(Error::JournalForked { .. })),
            "{result:?}"
        )
    };
    forked(sink.append(zeusdb_vector_core::OperationKind::Clear, &[]));
    forked(sink.commit());
    forked(sink.sync());
    forked(sink.roll(zeusdb_vector_core::SegmentBase::default()));
    assert!(
        Segments::of(&path).numbers.is_empty(),
        "nothing was written"
    );

    sink.pretend_own_process();
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..2);
    drop(collection);
    let (recovered, report) = recover(&path);
    assert_eq!(report.replayed, 2);
    assert_eq!(recovered.len(), 2);
}
