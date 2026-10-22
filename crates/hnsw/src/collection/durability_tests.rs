//! The three durability policies, the thread the interval one syncs from,
//! and what a commit that fails leaves.
//!
//! Nothing here kills a process; that is `crash_tests`. What is here is the
//! rest of the contract a caller acts on: that the interval policy's thread
//! lives exactly as long as the collection and does nothing while the
//! journal is clean, that a failed commit is withdrawn from the journal and
//! refuses everything after it until a checkpoint, and that a recovery takes
//! the tokenizer a text layer was declared with.
//!
//! A device fault cannot be produced on demand, so a failure is injected
//! through a journal directory that fails a chosen write, sync or cut; see
//! `journal_support`.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use zeusdb_vector_core::{Error, IdfScope, SEGMENT_HEADER_BYTES};
use zeusdb_vector_sparse::SparseConfig;
use zeusdb_vector_text::Tokenizer;

use super::journal_support::{
    copy_journaled, expected_of, manifest, sink_over, Faults, FaultyStorage, Segments, TempDir,
};
use super::{Added, Collection, Declaration, ParsedRecord, SparseHalf};
use crate::journal::{journal_path, CheckpointTaken, Durability};

// ============================================================================
// FIXTURES
// ============================================================================

fn declaration() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap()
}

fn record(i: usize) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vec![i as f32 * 0.25, (i % 5) as f32],
        sparse: None,
        metadata: HashMap::from([("i".to_string(), json!(i))]),
    }
}

fn try_add(collection: &Collection, range: std::ops::Range<usize>) -> Added {
    collection.add_records(range.map(record).collect(), vec![], false)
}

fn add(collection: &Collection, range: std::ops::Range<usize>) {
    let added = try_add(collection, range);
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

fn recover(path: &Path, durability: Durability) -> (Collection, crate::Recovery) {
    Collection::recover(path.to_str().unwrap(), None, durability).unwrap()
}

/// Wait for `condition`, up to `limit`.
fn wait_for(limit: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    condition()
}

/// A collection journaled at `path` through a journal directory whose
/// faults the test arms, with the empty checkpoint written, and the faults.
fn faulty(path: &Path, durability: Durability) -> (Collection, Arc<Faults>) {
    let collection = Collection::build(declaration(), None);
    let storage = FaultyStorage::at(path);
    let faults = storage.faults.clone();
    let sink = sink_over(&storage, path, collection.collection_id(), durability);
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    (collection, faults)
}

/// What the journal beside `path` holds, read by hand.
fn journal_holds(path: &Path) -> (usize, usize, usize) {
    let segments = Segments::of(path);
    let contents = segments.read(&expected_of(path)).unwrap();
    (contents.committed(), contents.withdrawn(), contents.aborts)
}

// ============================================================================
// THE INTERVAL THREAD
// ============================================================================

/// The thread lives exactly as long as the collection. The watch upgrades
/// while either the sink or the thread holds the shared state, and the
/// drop joins the thread, so the watch failing after the drop proves both
/// have let go and that nothing the thread would still write is in flight.
#[test]
fn the_interval_threads_lifetime_is_the_collections() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("lifetime.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(
            path.to_str().unwrap(),
            Durability::PerInterval(Duration::from_millis(5)),
        )
        .unwrap();
    let watch = collection
        .flusher_watch()
        .expect("the interval policy runs a thread");
    assert!(watch.alive());
    add(&collection, 0..3);
    drop(collection);
    assert!(
        !watch.alive(),
        "the sink and the thread both let go of the state when the collection was dropped"
    );

    // What the thread synced as it stopped is on the disk, and the
    // recovery starts a thread of its own that stops with the recovered
    // collection the same way.
    let (recovered, report) = recover(&path, Durability::PerInterval(Duration::from_millis(5)));
    assert_eq!(report.replayed, 3);
    assert_eq!(recovered.len(), 3);
    let watch = recovered.flusher_watch().unwrap();
    assert!(watch.alive());
    drop(recovered);
    assert!(!watch.alive());

    // No thread under the other two policies.
    let plain = Collection::build(declaration(), None);
    plain
        .journal_to(temp.at("call.zdb").to_str().unwrap(), Durability::PerCall)
        .unwrap();
    assert!(plain.flusher_watch().is_none());
    let (none, _) = recover(&path, Durability::NoSync);
    assert!(none.flusher_watch().is_none());
}

/// A clean journal costs no sync. The thread sleeps until a commit dirties
/// it, syncs once an interval while it is dirty, and goes back to sleep; a
/// checkpoint that syncs the journal itself leaves it nothing to do.
#[test]
fn the_interval_thread_syncs_nothing_while_the_file_is_clean() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("clean.zdb");
    let collection = Collection::build(declaration(), None);
    let interval = Duration::from_millis(200);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerInterval(interval))
        .unwrap();
    let watch = collection.flusher_watch().unwrap();

    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(watch.syncs(), Some(0), "an idle collection syncs nothing");

    add(&collection, 0..1);
    assert!(
        wait_for(Duration::from_secs(5), || watch.syncs() == Some(1)),
        "one commit is synced within the interval"
    );
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(watch.syncs(), Some(1), "and nothing is synced after it");

    // A burst gathers behind the interval rather than behind the commits.
    // The count is held against the intervals the burst spanned rather than
    // against the number two, because whether five commits fit inside one
    // interval is a statement about the machine while one sync an interval
    // is the policy. A sink that synced per commit reads five here.
    let before = watch.syncs().expect("the interval policy counts its syncs");
    let started = Instant::now();
    for i in 1..6 {
        add(&collection, i..i + 1);
    }
    assert!(
        wait_for(Duration::from_secs(5), || watch.syncs() > Some(before)),
        "the burst was synced"
    );
    std::thread::sleep(Duration::from_millis(300));
    let spanned = started.elapsed().as_millis() / interval.as_millis() + 1;
    let synced = u128::from(watch.syncs().unwrap() - before);
    assert!(
        synced <= spanned,
        "five commits cost {synced} syncs over {spanned} intervals"
    );

    // A checkpoint syncs the journal itself, so the thread finds it clean.
    // Read the count as the checkpoint returns rather than before the
    // commit, since whether the thread fired between the two is the
    // machine's business and what the checkpoint promises is that nothing
    // is left to sync after it.
    add(&collection, 6..7);
    collection.checkpoint().unwrap();
    let after_checkpoint = watch.syncs();
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(
        watch.syncs(),
        after_checkpoint,
        "the checkpoint left the thread nothing to sync"
    );
    assert_eq!(
        collection.journal_sequence(),
        14,
        "seven calls of one record, each with its commit"
    );
}

// ============================================================================
// A COMMIT THAT FAILS
// ============================================================================

/// Withdraw and hold, under the per-call policy. A commit whose sync fails
/// refuses its call, and the journal holds an abort after the call's
/// records. Every later mutation is refused naming `checkpoint()` and the
/// withdrawal. A copy recovers the four records the collection holds and
/// not the four it refused, at the live id counter, and its load takes a
/// checkpoint past the abort. `checkpoint()` then clears the fault, and the
/// segment holding the abort goes with it.
#[test]
fn a_commit_whose_sync_fails_is_withdrawn_and_refuses_everything_after_until_a_checkpoint() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("fault.zdb");
    let (collection, faults) = faulty(&path, Durability::PerCall);
    add(&collection, 0..4);
    assert_eq!(collection.id_counter(), 4);

    // The commit that fails. Every record of the batch is refused with the
    // sink's own error, saying the call was withdrawn, none is installed,
    // and the ids stay issued.
    faults.syncs.arm(0, 1);
    let failed = try_add(&collection, 4..8);
    assert_eq!(failed.inserted.len(), 0);
    assert_eq!(failed.total_errors, 4);
    for (i, error) in failed.errors.iter().enumerate() {
        assert!(
            error.starts_with(&format!("Vector r{}: RuntimeError: ", i + 4)),
            "{error}"
        );
        assert!(error.contains("injected: the device refused it"), "{error}");
        assert!(error.contains("withdrawn by an abort record"), "{error}");
    }
    assert_eq!(collection.len(), 4);
    assert!(!collection.contains("r4"));
    assert_eq!(collection.id_counter(), 8, "the ids stay issued");
    assert_eq!(
        journal_holds(&path),
        (4, 4, 1),
        "four committed, four withdrawn"
    );

    // A retry, and every other mutation, is refused naming the sequence,
    // the withdrawal and the checkpoint, and issues nothing. Four inserts
    // and a commit, then four inserts, a commit and the abort, reach 11.
    let retry = try_add(&collection, 4..5);
    assert_eq!(retry.total_errors, 1);
    for words in [
        "failed at sequence 11",
        "withdrawn by an abort record",
        "checkpoint()",
    ] {
        assert!(retry.errors[0].contains(words), "{}", retry.errors[0]);
    }
    assert_eq!(
        collection.id_counter(),
        8,
        "a refused record takes its id back"
    );
    match collection.remove_points(&["r0".to_string()]) {
        Err(Error::JournalCommitFailed { sequence: 11, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    assert!(collection.contains("r0"));
    match collection.update_metadata("r0", HashMap::from([("i".to_string(), json!(99))])) {
        Err(Error::JournalCommitFailed { sequence: 11, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    match collection.clear() {
        Err(Error::JournalCommitFailed { sequence: 11, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    assert_eq!(collection.len(), 4);

    // What a crash here recovers: the four the collection holds and none
    // of the four it refused, and the ids the refused records took stay
    // taken. The load meets the abort and checkpoints past it.
    let copy = copy_journaled(&path, "fault-crashed.zdb");
    let (crashed, report) = recover(&copy, Durability::NoSync);
    assert_eq!(report.replayed, 4);
    assert_eq!(report.withdrawn, 4);
    assert_eq!(
        report.checkpoint_taken,
        Some(CheckpointTaken::Abort),
        "the load checkpointed past the abort"
    );
    assert_eq!(crashed.len(), 4);
    assert!(!crashed.contains("r4") && !crashed.contains("r7"));
    assert_eq!(crashed.id_counter(), collection.id_counter());
    drop(crashed);
    let segments = Segments::of(&copy);
    assert_eq!(segments.numbers.len(), 1);
    let contents = segments.read(&expected_of(&copy)).unwrap();
    assert_eq!((contents.records.len(), contents.aborts), (0, 0));
    assert_eq!(manifest(&copy)["journal"]["sequence"], json!(11));

    // The checkpoint clears the fault. Its directory holds what the
    // collection holds, and the segment holding the abort goes.
    collection.checkpoint().unwrap();
    assert_eq!(collection.journal_sequence(), 11);
    assert_eq!(Segments::of(&path).numbers, vec![12]);
    assert_eq!(collection.len(), 4);
    add(&collection, 8..10);
    assert_eq!(collection.id_counter(), 10);
    collection.remove_points(&["r0".to_string()]).unwrap();
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) = recover(&path, Durability::PerCall);
    assert_eq!(report.checkpoint_sequence, 11);
    assert_eq!(report.replayed, 3, "two inserts and a removal");
    assert_eq!(ids(&recovered), before);
    assert!(
        !recovered.contains("r4"),
        "the refused records are gone for good"
    );
    assert_eq!(recovered.id_counter(), 10);
}

/// Withdraw and hold, where the abort itself cannot be written. The segment
/// is cut back to the boundary, so a copy recovers without the refused
/// records and the segment holds none of their bytes. The fault stands.
#[test]
fn a_commit_whose_abort_cannot_be_written_is_cut_back() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("cut.zdb");
    let (collection, faults) = faulty(&path, Durability::PerCall);
    add(&collection, 0..4);
    let boundary = Segments::of(&path).total_bytes();

    // The four inserts and the commit record go through, the commit's sync
    // fails, and the abort's write, the call's sixth, fails.
    faults.writes.arm(5, 1);
    faults.syncs.arm(0, 1);
    let failed = try_add(&collection, 4..8);
    assert_eq!(failed.total_errors, 4);
    assert!(
        failed.errors[0].contains("cut from the journal"),
        "{}",
        failed.errors[0]
    );
    assert_eq!(
        Segments::of(&path).total_bytes(),
        boundary,
        "no byte of the call is left"
    );
    assert_eq!(journal_holds(&path), (4, 0, 0));
    assert!(try_add(&collection, 8..9).errors[0].contains("checkpoint()"));

    let copy = copy_journaled(&path, "cut-crashed.zdb");
    let (crashed, report) = recover(&copy, Durability::NoSync);
    assert_eq!(report.replayed, 4);
    assert_eq!(report.withdrawn, 0);
    assert_eq!(
        report.checkpoint_taken, None,
        "nothing in the journal says a commit failed"
    );
    assert_eq!(ids(&crashed), ids(&collection));

    collection.checkpoint().unwrap();
    add(&collection, 8..9);
}

/// Withdraw and hold, where the sync, the abort and the cut back all fail.
/// The message says the refused records may come back after a crash until a
/// checkpoint, and they do: a copy taken then holds them. A copy taken after
/// `checkpoint()` does not, since the checkpoint writes the collection from
/// memory and deletes the segment that held them.
#[test]
fn a_commit_whose_abort_and_cut_both_fail_says_its_records_may_come_back() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("neither.zdb");
    let (collection, faults) = faulty(&path, Durability::PerCall);
    add(&collection, 0..4);

    // The commit's sync fails, the abort's write fails, and both the cut of
    // the half abort and the cut back to the boundary fail.
    faults.syncs.arm(0, 1);
    faults.writes.arm(5, 1);
    faults.cuts.arm(0, 2);
    let failed = try_add(&collection, 4..8);
    assert_eq!(failed.total_errors, 4);
    assert!(
        failed.errors[0].contains("may come back after a crash until a checkpoint"),
        "{}",
        failed.errors[0]
    );
    let early = copy_journaled(&path, "neither-early.zdb");
    let (crashed, report) = recover(&early, Durability::NoSync);
    assert_eq!(
        report.replayed, 8,
        "the refused records came back, as the message said"
    );
    assert_eq!(crashed.len(), 8);
    drop(crashed);

    collection.checkpoint().unwrap();
    let late = copy_journaled(&path, "neither-late.zdb");
    let (recovered, report) = recover(&late, Durability::NoSync);
    assert_eq!(report.replayed, 0);
    assert_eq!(ids(&recovered), ids(&collection));
    assert_eq!(recovered.len(), 4);
    add(&collection, 8..9);
}

/// A checkpoint whose sync of the old segment fails still commits, writes
/// the collection from memory and clears the fault, whether the fault was
/// there before or the failed sync is the first failure.
#[test]
fn a_checkpoint_runs_through_a_failed_sync_and_clears_the_fault() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("through.zdb");
    let (collection, faults) = faulty(&path, Durability::PerCall);
    add(&collection, 0..4);
    faults.syncs.arm(0, 1);
    assert_eq!(try_add(&collection, 4..6).total_errors, 2);

    // The fault stands and the checkpoint's own sync fails as well.
    faults.syncs.arm(0, 1);
    collection.checkpoint().unwrap();
    add(&collection, 6..8);
    let before = ids(&collection);
    drop(collection);
    let (recovered, report) = recover(&path, Durability::PerCall);
    assert_eq!(ids(&recovered), before);
    assert_eq!(report.replayed, 2);
    drop(recovered);

    // Under the policy that never syncs, the checkpoint's sync is the first
    // to fail.
    let path = temp.at("through-none.zdb");
    let (collection, faults) = faulty(&path, Durability::NoSync);
    add(&collection, 0..5);
    faults.syncs.arm(0, 1);
    collection.checkpoint().unwrap();
    add(&collection, 5..6);
    let before = ids(&collection);
    drop(collection);
    let (recovered, report) = recover(&path, Durability::PerCall);
    assert_eq!(ids(&recovered), before);
    assert_eq!(report.checkpoint_sequence, 6);
    assert_eq!(report.replayed, 1);
}

/// Under the interval policy, a sync the thread could not complete refuses
/// and withdraws the next call, by an abort with no commit before it, and
/// the fault stands until `checkpoint()`. The call before it was committed
/// and acknowledged, so a process that stops keeps it.
#[test]
fn a_failed_sync_of_the_interval_thread_withdraws_the_next_call() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("interval-fault.zdb");
    let (collection, faults) = faulty(&path, Durability::PerInterval(Duration::from_millis(5)));
    add(&collection, 0..2);
    faults.syncs.arm(0, 1);
    add(&collection, 2..3);
    assert!(
        wait_for(Duration::from_secs(10), || faults.syncs.spent()),
        "the thread reached its failing sync"
    );
    let failed = try_add(&collection, 3..4);
    assert_eq!(failed.total_errors, 1);
    assert!(
        failed.errors[0].contains("withdrawn by an abort record"),
        "{}",
        failed.errors[0]
    );
    assert!(try_add(&collection, 4..5).errors[0].contains("checkpoint()"));
    assert_eq!(journal_holds(&path), (3, 1, 1));

    let copy = copy_journaled(&path, "interval-crashed.zdb");
    let (crashed, report) = recover(&copy, Durability::NoSync);
    assert_eq!(report.replayed, 3);
    assert_eq!(report.withdrawn, 1);
    assert_eq!(crashed.len(), 3);
    drop(crashed);

    collection.checkpoint().unwrap();
    add(&collection, 4..6);
    assert_eq!(collection.len(), 5);
}

// ============================================================================
// THE CHECKPOINT AND THE STATUS
// ============================================================================

/// `checkpoint` is a save into the journal's own directory, and needs one.
/// The status reports the journal's directory, the operation records a
/// replay would apply, the bytes of every segment and the segments.
#[test]
fn checkpoint_writes_to_the_journals_own_directory_and_needs_one() {
    let plain = Collection::build(declaration(), None);
    match plain.checkpoint() {
        Err(Error::NotJournaled) => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(plain.journal_status(), None);

    let temp = TempDir::new("durability-tests");
    let path = temp.at("checkpoint.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    let journal = journal_path(&path).unwrap();
    let header = SEGMENT_HEADER_BYTES as u64;

    let status = collection.journal_status().unwrap();
    assert_eq!(status.path, journal);
    assert_eq!(status.durability, Durability::PerCall);
    assert_eq!(status.checkpoint_sequence, 0);
    assert_eq!(status.sequence_reached, 0);
    assert_eq!(status.records_since_checkpoint(), 0);
    assert_eq!(status.bytes, Some(header));
    assert_eq!(status.segments, 1);

    add(&collection, 0..6);
    let status = collection.journal_status().unwrap();
    assert_eq!(status.records_since_checkpoint(), 6);
    assert_eq!(status.sequence_reached, 7, "six records and a commit");
    assert!(status.bytes.unwrap() > header);
    assert_eq!(status.bytes, Some(Segments::of(&path).total_bytes()));

    collection.checkpoint().unwrap();
    assert_eq!(collection.journal_sequence(), 7);
    let status = collection.journal_status().unwrap();
    assert_eq!(status.checkpoint_sequence, 7);
    assert_eq!(status.sequence_reached, 7);
    assert_eq!(status.records_since_checkpoint(), 0);
    assert_eq!(
        status.bytes,
        Some(header),
        "the checkpoint rolled onto an empty segment"
    );
    assert_eq!(status.segments, 1);
    let stats = collection.stats();
    assert_eq!(stats.get("journal_segments").map(String::as_str), Some("1"));
    assert_eq!(
        stats.get("journal_bytes").map(String::as_str),
        Some(header.to_string().as_str())
    );
    assert_eq!(manifest(&path)["journal"]["sequence"], json!(7));

    let interval = Collection::build(declaration(), None);
    interval
        .journal_to(
            temp.at("interval.zdb").to_str().unwrap(),
            Durability::PerInterval(Duration::from_millis(25)),
        )
        .unwrap();
    assert_eq!(
        interval.journal_status().unwrap().durability,
        Durability::PerInterval(Duration::from_millis(25))
    );
    drop(collection);
    let (recovered, report) = recover(&path, Durability::PerCall);
    assert_eq!(report.replayed, 0);
    assert_eq!(recovered.len(), 6);
}

// ============================================================================
// AN EXTERNAL TOKENIZER
// ============================================================================

/// A tokenizer of the caller's own: bars separate terms, case is kept.
#[derive(Debug)]
struct Bars;

impl Tokenizer for Bars {
    fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
        text.split('|').filter(|t| !t.is_empty()).for_each(term);
        Ok(())
    }
}

fn hits(collection: &Collection, text: &str) -> Vec<String> {
    collection
        .search_text(text, None, 5, IdfScope::Corpus)
        .unwrap()
        .into_iter()
        .map(|hit| hit.0)
        .collect()
}

/// A text layer declared with an external tokenizer recovers only when the
/// tokenizer is handed to `recover`, as it is to `load_with`, and the
/// records that live in the journal alone are found through it.
#[test]
fn a_recovery_takes_the_tokenizer_the_text_layer_was_declared_with() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("bars.zdb");
    let declaration = declaration()
        .with_text("text", SparseConfig::default(), Arc::new(Bars))
        .unwrap();
    let collection = Collection::build(declaration, None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();

    // Past the checkpoint, so the records and their terms live in the
    // journal alone.
    let text_record = |id: &str, dense: [f32; 2], text: &str| ParsedRecord {
        id: id.to_string(),
        vector: dense.to_vec(),
        sparse: Some(SparseHalf::Terms(collection.tokenize(text).unwrap())),
        metadata: HashMap::new(),
    };
    let added = collection.add_records(
        vec![
            text_record("a", [1.0, 0.0], "Alpha|beta"),
            text_record("b", [0.0, 1.0], "beta|GAMMA"),
        ],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(hits(&collection, "GAMMA"), ["b"]);
    drop(collection);

    match Collection::recover(path.to_str().unwrap(), None, Durability::PerCall) {
        Err(Error::TokenizerRequired { space }) => assert_eq!(space, "text"),
        other => panic!(
            "expected the tokenizer to be required, got {:?}",
            other.map(|_| "a collection")
        ),
    }
    let (recovered, report) = Collection::recover(
        path.to_str().unwrap(),
        Some(Arc::new(Bars)),
        Durability::PerCall,
    )
    .unwrap();
    assert_eq!(report.replayed, 5, "two inserts and three interned terms");
    assert_eq!(hits(&recovered, "GAMMA"), ["b"]);
    assert_eq!(hits(&recovered, "Alpha"), ["a"]);
    assert_eq!(
        hits(&recovered, "alpha"),
        Vec::<String>::new(),
        "the queries run through the caller's tokenizer, which keeps case"
    );
    assert_eq!(hits(&recovered, "gamma delta"), Vec::<String>::new());

    // And the recovered collection keeps taking text through it.
    let added = recovered.add_records(
        vec![ParsedRecord {
            id: "c".to_string(),
            vector: vec![1.0, 1.0],
            sparse: Some(SparseHalf::Terms(
                recovered.tokenize("GAMMA|delta").unwrap(),
            )),
            metadata: HashMap::new(),
        }],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    let mut gamma = hits(&recovered, "GAMMA");
    gamma.sort();
    assert_eq!(gamma, ["b", "c"]);
}

/// A term whose record the journal refused is not interned, so the next
/// term takes its id and a replay meets the ids the live dictionary issued.
#[test]
fn a_term_whose_record_was_refused_is_not_interned() {
    let temp = TempDir::new("durability-tests");
    let path = temp.at("terms.zdb");
    let declaration = declaration()
        .with_text("text", SparseConfig::default(), Arc::new(Bars))
        .unwrap();
    let collection = Collection::build(declaration, None);
    let storage = FaultyStorage::at(&path);
    let faults = storage.faults.clone();
    collection.attach_sink(Box::new(sink_over(
        &storage,
        &path,
        collection.collection_id(),
        Durability::PerCall,
    )));
    collection.save(path.to_str().unwrap()).unwrap();
    let text_record = |id: &str, text: &str| ParsedRecord {
        id: id.to_string(),
        vector: vec![1.0, 0.0],
        sparse: Some(SparseHalf::Terms(collection.tokenize(text).unwrap())),
        metadata: HashMap::new(),
    };

    // The write of the first term's record fails, and is cut back.
    faults.writes.arm(0, 1);
    let refused = collection.add_records(vec![text_record("a", "lost|kept")], vec![], false);
    assert_eq!(refused.total_errors, 1);
    assert_eq!(
        collection.term_count(),
        Some(0),
        "no term of the refused record is held"
    );
    let added = collection.add_records(vec![text_record("b", "kept|other")], vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(collection.term_count(), Some(2));
    drop(collection);

    let (recovered, report) = Collection::recover(
        path.to_str().unwrap(),
        Some(Arc::new(Bars)),
        Durability::PerCall,
    )
    .unwrap();
    assert_eq!(report.replayed, 3, "one insert and two interned terms");
    assert_eq!(hits(&recovered, "kept"), ["b"]);
    assert_eq!(recovered.term_count(), Some(2));
}

// ============================================================================
// AN OVERWRITE AND A TRAINING WHOSE COMMIT FAILS
// ============================================================================

/// A journal sink that fails the sync of the first commit to follow a record
/// of `kind` once it is armed, by arming the storage's fault just before
/// that commit, so the writer withdraws the call as it withdraws any whose
/// sync fails.
#[derive(Debug)]
struct FailsAfter {
    inner: crate::JournalSink,
    faults: Arc<Faults>,
    kind: zeusdb_vector_core::OperationKind,
    armed: Arc<std::sync::atomic::AtomicBool>,
    pending: bool,
}

impl crate::OperationSink for FailsAfter {
    fn append(
        &mut self,
        kind: zeusdb_vector_core::OperationKind,
        payload: &[u8],
    ) -> Result<(), Error> {
        self.pending |= kind == self.kind;
        self.inner.append(kind, payload)
    }
    fn commit(&mut self) -> Result<(), Error> {
        let pending = std::mem::take(&mut self.pending);
        if pending && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.faults.syncs.arm(0, 1);
        }
        self.inner.commit()
    }
    fn sync(&mut self) -> Result<(), Error> {
        self.inner.sync()
    }
    fn sequence_reached(&self) -> u64 {
        self.inner.sequence_reached()
    }
    fn roll(&mut self, base: zeusdb_vector_core::SegmentBase) -> Result<(), Error> {
        self.inner.roll(base)
    }
    fn journal_file(&self) -> Option<&str> {
        self.inner.journal_file()
    }
    fn journal_collection_id(&self) -> Option<u128> {
        self.inner.journal_collection_id()
    }
    fn journal_path(&self) -> Option<&Path> {
        crate::OperationSink::journal_path(&self.inner)
    }
    fn durability(&self) -> Option<Durability> {
        crate::OperationSink::durability(&self.inner)
    }
}

/// A collection declared by `declaration`, journaled at `path` through a
/// sink that fails the first commit after a record of `kind` once the flag
/// it returns is set, with the empty checkpoint written.
fn failing_after(
    path: &Path,
    declaration: Declaration,
    quantization: Option<super::QuantizationConfig>,
    kind: zeusdb_vector_core::OperationKind,
) -> (Collection, Arc<std::sync::atomic::AtomicBool>) {
    let collection = Collection::build(declaration, quantization);
    let storage = FaultyStorage::at(path);
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sink = FailsAfter {
        inner: sink_over(
            &storage,
            path,
            collection.collection_id(),
            Durability::PerCall,
        ),
        faults: storage.faults.clone(),
        kind,
        armed: armed.clone(),
        pending: false,
    };
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    (collection, armed)
}

/// The record `i` as an overwrite replaces it, with another vector.
fn replacement(i: usize) -> ParsedRecord {
    ParsedRecord {
        vector: vec![-1.0 - i as f32, 7.0],
        ..record(i)
    }
}

fn vector_of(collection: &Collection, id: &str) -> Vec<f32> {
    collection
        .records(vec![id.to_string()], true, false)
        .unwrap()[0]
        .vector
        .clone()
        .unwrap()
}

/// An overwrite whose commit fails keeps every record it would
/// replace, live and after a crash, since its removal and its inserts are
/// one call and the removal runs only once that call commits.
#[test]
fn an_overwrite_whose_commit_fails_keeps_what_it_replaces() {
    let tmp = TempDir::new("overwrite-fault");
    let path = tmp.at("index.zdb");
    let (collection, armed) = failing_after(
        &path,
        declaration(),
        None,
        zeusdb_vector_core::OperationKind::Insert,
    );
    add(&collection, 0..4);
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let replaced = collection.add_records(vec![replacement(0), replacement(1)], vec![], true);
    assert_eq!(replaced.total_errors, 2, "{:?}", replaced.errors);
    assert_eq!(ids(&collection), ["r0", "r1", "r2", "r3"]);
    assert_eq!(vector_of(&collection, "r0"), record(0).vector);
    assert_eq!(vector_of(&collection, "r1"), record(1).vector);

    let crashed = copy_journaled(&path, "overwrite-fault-crashed.zdb");
    let (recovered, _) = recover(&crashed, Durability::PerCall);
    assert_eq!(ids(&recovered), ["r0", "r1", "r2", "r3"]);
    assert_eq!(vector_of(&recovered, "r0"), record(0).vector);
    assert_eq!(vector_of(&recovered, "r1"), record(1).vector);
}

/// A journal sink that copies the directory and the journal aside at the
/// first insert handed over once it is armed, which is the state a process
/// killed at that moment leaves.
#[derive(Debug)]
struct CopiesAtTheInsert {
    inner: crate::JournalSink,
    directory: std::path::PathBuf,
    armed: Arc<std::sync::atomic::AtomicBool>,
}

impl crate::OperationSink for CopiesAtTheInsert {
    fn append(
        &mut self,
        kind: zeusdb_vector_core::OperationKind,
        payload: &[u8],
    ) -> Result<(), Error> {
        if kind == zeusdb_vector_core::OperationKind::Insert
            && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            copy_journaled(&self.directory, "overwrite-crash-copy.zdb");
        }
        self.inner.append(kind, payload)
    }
    fn commit(&mut self) -> Result<(), Error> {
        self.inner.commit()
    }
    fn sync(&mut self) -> Result<(), Error> {
        self.inner.sync()
    }
    fn sequence_reached(&self) -> u64 {
        self.inner.sequence_reached()
    }
    fn roll(&mut self, base: zeusdb_vector_core::SegmentBase) -> Result<(), Error> {
        self.inner.roll(base)
    }
    fn journal_file(&self) -> Option<&str> {
        self.inner.journal_file()
    }
    fn journal_collection_id(&self) -> Option<u128> {
        self.inner.journal_collection_id()
    }
    fn journal_path(&self) -> Option<&Path> {
        crate::OperationSink::journal_path(&self.inner)
    }
    fn durability(&self) -> Option<Durability> {
        crate::OperationSink::durability(&self.inner)
    }
}

/// A crash inside an overwrite, once its removal has reached the
/// journal and before its inserts have, recovers every record it would
/// replace, since nothing of the call is committed until its inserts are.
#[test]
fn a_crash_inside_an_overwrite_keeps_what_it_replaces() {
    let tmp = TempDir::new("overwrite-crash");
    let path = tmp.at("index.zdb");
    let collection = Collection::build(declaration(), None);
    let storage = FaultyStorage::at(&path);
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sink = CopiesAtTheInsert {
        inner: sink_over(
            &storage,
            &path,
            collection.collection_id(),
            Durability::PerCall,
        ),
        directory: path.clone(),
        armed: armed.clone(),
    };
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..4);
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let replaced = collection.add_records(vec![replacement(0), replacement(1)], vec![], true);
    assert_eq!(replaced.total_errors, 0, "{:?}", replaced.errors);
    drop(collection);

    let copy = tmp.at("overwrite-crash-copy.zdb");
    let (recovered, _) = recover(&copy, Durability::PerCall);
    assert_eq!(ids(&recovered), ["r0", "r1", "r2", "r3"]);
    assert_eq!(vector_of(&recovered, "r0"), record(0).vector);
    assert_eq!(vector_of(&recovered, "r1"), record(1).vector);
}

/// A declaration with product quantization that trains at 1,000 records,
/// the least a declaration takes.
fn trained_declaration() -> (Declaration, super::QuantizationConfig) {
    let declaration = Declaration::validate(2, "l2", 4, 50, 2000, vec![]).unwrap();
    let quantization = declaration
        .quantization(1, 2, 1000, None, super::StorageMode::QuantizedWithRaw)
        .unwrap();
    (declaration, quantization)
}

/// The codebook a trained collection holds, as the bits of every value.
fn codebook_of(collection: &Collection) -> Vec<u32> {
    collection
        .dense()
        .pq
        .as_ref()
        .unwrap()
        .with_centroids(|centroids| {
            centroids
                .iter()
                .flatten()
                .flatten()
                .map(|value| value.to_bits())
                .collect()
        })
}

/// A training whose record fails to commit trains nothing, live
/// or after a crash: what was fitted is dropped and the live collection
/// stays untrained with its set full, and a replay of the records trains
/// nothing at the insert that filled the set and passes over the withdrawn
/// record. The first call after the checkpoint trains, and a replay of its
/// record installs the codebook it fitted.
#[test]
fn a_training_whose_record_fails_trains_nothing_live_or_replayed() {
    let tmp = TempDir::new("training-fault");
    let path = tmp.at("index.zdb");
    let (declaration, quantization) = trained_declaration();
    let (collection, armed) = failing_after(
        &path,
        declaration,
        Some(quantization),
        zeusdb_vector_core::OperationKind::Train,
    );
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let added = try_add(&collection, 0..1000);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(
        added.errors.iter().any(|e| e.contains("Training failed")),
        "{:?}",
        added.errors
    );
    assert_eq!(collection.len(), 1000);
    assert!(!collection.can_use_quantization(), "nothing trained live");
    assert!(collection.is_training_ready());

    let crashed = copy_journaled(&path, "training-fault-crashed.zdb");
    let (recovered, _) = recover(&crashed, Durability::PerCall);
    assert_eq!(recovered.len(), 1000);
    assert!(
        !recovered.can_use_quantization(),
        "nothing trained on the replay either"
    );
    assert!(recovered.is_training_ready());
    drop(recovered);

    collection.checkpoint().unwrap();
    add(&collection, 1000..1001);
    assert!(collection.can_use_quantization(), "the next call trains");
    let crashed = copy_journaled(&path, "training-crashed.zdb");
    let (recovered, _) = recover(&crashed, Durability::PerCall);
    assert!(recovered.can_use_quantization());
    assert_eq!(codebook_of(&recovered), codebook_of(&collection));
    assert_eq!(
        recovered.training_completed_at(),
        collection.training_completed_at()
    );
    assert_eq!(ids(&recovered), ids(&collection));
}
