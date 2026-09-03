//! The three durability policies, the thread the interval one syncs from,
//! and what a commit that fails leaves.
//!
//! Nothing here kills a process; that is `crash_tests`. What is here is the
//! rest of the contract a caller acts on: that the interval policy's thread
//! lives exactly as long as the collection and does nothing while the file
//! is clean, that a failed commit refuses its records and everything after
//! them until a checkpoint, and that a recovery takes the tokenizer a text
//! layer was declared with.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use zeusdb_vector_core::{Error, IdfScope, OperationKind};
use zeusdb_vector_sparse::SparseConfig;
use zeusdb_vector_text::Tokenizer;

use super::{Added, Collection, Declaration, OperationSink, ParsedRecord, SparseHalf};
use crate::journal::{journal_path, Durability, JournalSink};

// ============================================================================
// FIXTURES
// ============================================================================

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-durability-tests-{}-{}",
            std::process::id(),
            n
        ));
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
    let mut out: Vec<String> = collection.id_map().keys().cloned().collect();
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

// ============================================================================
// THE INTERVAL THREAD
// ============================================================================

/// The thread lives exactly as long as the collection. The watch upgrades
/// while either the sink or the thread holds the shared state, and the
/// drop joins the thread, so the watch failing after the drop proves both
/// have let go and that nothing the thread would still write is in flight.
#[test]
fn the_interval_threads_lifetime_is_the_collections() {
    let temp = TempDir::new();
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

/// A clean file costs no sync. The thread sleeps until a commit dirties
/// the file, syncs once an interval while it is dirty, and goes back to
/// sleep; a checkpoint that syncs the file itself leaves it nothing to do.
#[test]
fn the_interval_thread_syncs_nothing_while_the_file_is_clean() {
    let temp = TempDir::new();
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

    // A burst inside one interval gathers behind one sync.
    for i in 1..6 {
        add(&collection, i..i + 1);
    }
    assert!(wait_for(Duration::from_secs(5), || watch.syncs() == Some(2)));
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(
        watch.syncs(),
        Some(2),
        "five commits inside one interval, one sync"
    );

    // A checkpoint syncs the file itself, so the thread finds it clean.
    add(&collection, 6..7);
    collection.checkpoint().unwrap();
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(
        watch.syncs(),
        Some(2),
        "the checkpoint left the thread nothing to sync"
    );
    assert_eq!(collection.journal_sequence(), 7);
}

// ============================================================================
// A COMMIT THAT FAILS
// ============================================================================

/// A journal sink whose commit fails once, at a named call.
#[derive(Debug)]
struct FailingCommit {
    inner: JournalSink,
    fail_at: usize,
    commits: usize,
}

impl OperationSink for FailingCommit {
    fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<(), Error> {
        self.inner.append(kind, payload)
    }

    fn commit(&mut self) -> Result<(), Error> {
        self.commits += 1;
        if self.commits == self.fail_at {
            return Err(Error::JournalIoFailed {
                path: self.inner.path().to_path_buf(),
                what: "sync",
                error: "the device refused the flush".to_string(),
            });
        }
        self.inner.commit()
    }

    fn sync(&mut self) -> Result<(), Error> {
        self.inner.sync()
    }

    fn sequence_reached(&self) -> u64 {
        self.inner.sequence_reached()
    }

    fn truncate(&mut self) -> Result<(), Error> {
        self.inner.truncate()
    }

    fn journal_file(&self) -> Option<&str> {
        self.inner.journal_file()
    }

    fn journal_collection_id(&self) -> Option<u128> {
        self.inner.journal_collection_id()
    }

    fn journal_path(&self) -> Option<&Path> {
        self.inner.journal_path()
    }

    fn durability(&self) -> Option<Durability> {
        Some(self.inner.durability())
    }
}

/// Copy a directory and the journal beside it under another name, so the
/// copy can be recovered while the original's sink is still open.
fn copy_journaled(dir: &Path, name: &str) -> std::path::PathBuf {
    let target = dir.parent().unwrap().join(name);
    std::fs::create_dir_all(&target).unwrap();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
    }
    std::fs::copy(journal_path(dir).unwrap(), journal_path(&target).unwrap()).unwrap();
    target
}

/// What a caller sees when a commit fails, what the index then holds, what
/// a retry does, what a crash before the checkpoint recovers, and what the
/// checkpoint reconciles.
#[test]
fn a_commit_that_fails_refuses_its_records_and_everything_after_until_a_checkpoint() {
    let temp = TempDir::new();
    let path = temp.at("fault.zdb");
    let collection = Collection::build(declaration(), None);
    let wal = journal_path(&path).unwrap();
    let sink = FailingCommit {
        inner: JournalSink::create(&wal, collection.collection_id(), Durability::PerCall).unwrap(),
        fail_at: 2,
        commits: 0,
    };
    collection.attach_sink(Box::new(sink));
    collection.save(path.to_str().unwrap()).unwrap();
    add(&collection, 0..4);
    assert_eq!(collection.id_counter(), 4);

    // The commit that fails. Every record of the batch is refused with the
    // sink's own error, none is installed, and the ids stay issued.
    let failed = try_add(&collection, 4..8);
    assert_eq!(failed.inserted.len(), 0);
    assert_eq!(failed.total_errors, 4);
    for (i, error) in failed.errors.iter().enumerate() {
        assert!(
            error.starts_with(&format!("Vector r{}: RuntimeError: ", i + 4)),
            "{error}"
        );
        assert!(error.contains("the device refused the flush"), "{error}");
    }
    assert_eq!(collection.len(), 4);
    assert!(!collection.contains("r4"));
    assert_eq!(
        collection.id_counter(),
        8,
        "the ids stay issued, because the records may be on the device"
    );

    // A retry, and every other mutation, is refused naming the sequence,
    // and issues nothing.
    let retry = try_add(&collection, 4..5);
    assert_eq!(retry.total_errors, 1);
    assert!(
        retry.errors[0].contains("failed at sequence 8"),
        "{}",
        retry.errors[0]
    );
    assert!(
        retry.errors[0].contains("checkpoint()"),
        "{}",
        retry.errors[0]
    );
    assert_eq!(
        collection.id_counter(),
        8,
        "a refused record takes its id back"
    );
    let fresh = try_add(&collection, 8..9);
    assert_eq!(fresh.total_errors, 1);
    assert_eq!(collection.id_counter(), 8);
    match collection.remove_points(&["r0".to_string()]) {
        Err(Error::JournalCommitFailed { sequence: 8, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    assert!(collection.contains("r0"));
    match collection.update_metadata("r0", HashMap::from([("i".to_string(), json!(99))])) {
        Err(Error::JournalCommitFailed { sequence: 8, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    match collection.clear() {
        Err(Error::JournalCommitFailed { sequence: 8, .. }) => {}
        other => panic!("expected the fault, got {other:?}"),
    }
    assert_eq!(collection.len(), 4);

    // What a crash here recovers. The refused records reached the journal,
    // so a replay installs them: that is the "may have reached the device"
    // half of the contract, and it is why a retry is refused.
    let copy = copy_journaled(&path, "fault-crashed.zdb");
    let (crashed, report) = recover(&copy, Durability::NoSync);
    assert_eq!(report.replayed, 8);
    assert_eq!(crashed.len(), 8);
    assert!(crashed.contains("r4") && crashed.contains("r7"));
    drop(crashed);

    // The checkpoint reconciles. Its sync succeeds, the sequence it records
    // claims the refused records so a replay skips them, the directory
    // holds what the collection holds, and the journal is emptied.
    collection.checkpoint().unwrap();
    assert_eq!(collection.journal_sequence(), 8);
    assert_eq!(collection.len(), 4);
    add(&collection, 8..10);
    assert_eq!(collection.id_counter(), 10);
    collection.remove_points(&["r0".to_string()]).unwrap();
    let before = ids(&collection);
    drop(collection);

    let (recovered, report) = recover(&path, Durability::PerCall);
    assert_eq!(report.checkpoint_sequence, 8);
    assert_eq!(report.replayed, 3, "two inserts and a removal");
    assert_eq!(ids(&recovered), before);
    assert!(
        !recovered.contains("r4"),
        "the refused records are gone for good"
    );
    assert_eq!(recovered.id_counter(), 10);
}

// ============================================================================
// THE CHECKPOINT AND THE STATUS
// ============================================================================

/// `checkpoint` is a save into the journal's own directory, and needs one.
#[test]
fn checkpoint_writes_to_the_journals_own_directory_and_needs_one() {
    let plain = Collection::build(declaration(), None);
    match plain.checkpoint() {
        Err(Error::NotJournaled) => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(plain.journal_status(), None);

    let temp = TempDir::new();
    let path = temp.at("checkpoint.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    let wal = journal_path(&path).unwrap();
    let header = zeusdb_vector_core::JOURNAL_HEADER_BYTES as u64;

    let status = collection.journal_status().unwrap();
    assert_eq!(status.path, wal);
    assert_eq!(status.durability, Durability::PerCall);
    assert_eq!(status.checkpoint_sequence, 0);
    assert_eq!(status.sequence_reached, 0);
    assert_eq!(status.records_since_checkpoint(), 0);
    assert_eq!(status.bytes, Some(header));

    add(&collection, 0..6);
    let status = collection.journal_status().unwrap();
    assert_eq!(status.records_since_checkpoint(), 6);
    assert!(status.bytes.unwrap() > header);

    collection.checkpoint().unwrap();
    assert_eq!(collection.journal_sequence(), 6);
    let status = collection.journal_status().unwrap();
    assert_eq!(status.checkpoint_sequence, 6);
    assert_eq!(status.sequence_reached, 6);
    assert_eq!(status.records_since_checkpoint(), 0);
    assert_eq!(
        status.bytes,
        Some(header),
        "the checkpoint emptied the journal"
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["journal"]["sequence"], json!(6));

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
    let temp = TempDir::new();
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
