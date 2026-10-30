//! What the journal's tests share: a temporary folder, a journal read back
//! by hand, a journal damaged for a repair, a graph dump whose topology is
//! damaged, faults armed on a journal's directory, a record of the interval
//! thread's syncs through that directory, a storage on a medium that
//! refuses every write, and what a thread says while a test holds a write.
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
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use serde_json::Value;
use zeusdb_vector_core::{
    encode_journal_record, Dir, Error, Expected, FsStorage, InsertParts, JournalContents,
    JournalDir, JournalFile, LockFailure, LockGuard, LockMode, Locked, Operation, SegmentBase,
    SegmentBytes, Staged, Storage, JOURNAL_RECORD_HEADER_BYTES, JOURNAL_RECORD_OVERHEAD_BYTES,
    SEGMENT_HEADER_BYTES, SEGMENT_ROLL_BYTES,
};

use super::{Collection, Declaration, ParsedRecord};
use crate::journal::{Durability, JournalSink};
use crate::locks::{on_wait, Wait, WAIT_HOOK};

// ============================================================================
// A GRAPH DUMP DAMAGED
// ============================================================================

/// The directory size a 5.x manifest records, written from the lengths it
/// records, after an edit to one of them.
pub(super) fn resized(manifest: &mut Value) {
    let total: u64 = manifest["file_digests"]
        .as_object()
        .unwrap()
        .values()
        .map(|digest| digest["bytes"].as_u64().unwrap())
        .sum();
    manifest["total_size_mb"] = serde_json::json!(total as f64 / (1024.0 * 1024.0));
}

/// Damage the topology region of the version 2 graph dump in `directory`,
/// leaving its length and every other region as they were, so a load
/// refuses the topology alone and rebuilds the graph from the rows the dump
/// holds.
pub(super) fn break_topology(directory: &Path) {
    let path = directory.join(zeusdb_vector_core::DUMP_FILENAME);
    let mut blob = std::fs::read(&path).unwrap();
    let length = |at: usize| u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
    let topology = 128 + length(48) + length(56);
    blob[topology] ^= 0x01;
    std::fs::write(&path, &blob).unwrap();
}

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
// A JOURNAL DAMAGED FOR A REPAIR
// ============================================================================

/// The classes of damage a repair is held to: a corrupt middle in the last
/// segment, damage in the segment that is not the last, a header that will
/// not read, a record whose checksum holds and which will not decode, and a
/// stale segment whose header will not read beside a live one. A repair cuts
/// at the first four, and sets the stale segment aside whole.
pub(super) const REPAIR_CLASSES: [&str; 5] = ["corrupt", "sealed", "header", "decode", "stale"];

/// The calls a damaged journal holds, and the records each call adds.
pub(super) const DAMAGED_CALLS: usize = 6;
pub(super) const DAMAGED_CALL_RECORDS: usize = 3;

/// A journaled directory at `dir`, declared by `declaration`, holding
/// [`DAMAGED_CALLS`] calls of [`DAMAGED_CALL_RECORDS`] records each, every
/// record built by `record` from its index, the first four calls in one
/// segment and the last two in the next, damaged as `class` names.
///
/// Returns the calls a repair keeps, being every call before the one the
/// damage touches, and the bytes the repair sets aside, being that call's
/// first byte and every byte after it, segment by segment. Each call is
/// four records, three inserts and the commit, from sequence one.
///
/// For `stale` a checkpoint follows the first four calls, so the second
/// segment holds the last two calls and the first is deleted, and a copy of
/// the first is put back with its header damaged, as a crash inside that
/// deletion and damage to the copy leave it. A repair keeps every call and
/// sets that copy aside whole, which is the bytes returned.
pub(super) fn damaged_journal(
    dir: &Path,
    class: &str,
    declaration: Declaration,
    record: impl Fn(usize) -> ParsedRecord,
) -> (usize, Vec<u8>) {
    let probe = record(10);
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
    let insert = (payload.len() + JOURNAL_RECORD_OVERHEAD_BYTES) as u64;
    let commit = (8 + JOURNAL_RECORD_OVERHEAD_BYTES) as u64;
    let first_segment = crate::journal::journal_path(dir)
        .unwrap()
        .join(format!("{:020}.zdbseg", 1));
    let mut stale_copy = None;
    {
        // Ten inserts and four commits past the header: the fourth call's
        // commit passes it and the sixth's does not. A writer takes the
        // length when it is made.
        zeusdb_vector_core::set_segment_roll_bytes(
            SEGMENT_HEADER_BYTES as u64 + 10 * insert + 4 * commit,
        );
        let collection = Collection::build(declaration, None);
        collection
            .journal_to(dir.to_str().unwrap(), Durability::PerCall)
            .unwrap();
        for call in 0..DAMAGED_CALLS {
            let first = DAMAGED_CALL_RECORDS * call;
            let batch = (first..first + DAMAGED_CALL_RECORDS).map(&record).collect();
            let added = collection.add_records(batch, vec![], false);
            assert_eq!(added.total_errors, 0, "{:?}", added.errors);
            if class == "stale" && call == 3 {
                stale_copy = Some(std::fs::read(&first_segment).unwrap());
                collection.checkpoint().unwrap();
            }
        }
        zeusdb_vector_core::set_segment_roll_bytes(SEGMENT_ROLL_BYTES);
    }
    if let Some(copy) = &stale_copy {
        assert!(
            !first_segment.exists(),
            "the checkpoint deleted the first segment"
        );
        std::fs::write(&first_segment, copy).unwrap();
    }
    let segments = Segments::of(dir);
    assert_eq!(segments.numbers, vec![1, 17], "the calls fill two segments");
    let contents = segments.read(&expected_of(dir)).unwrap();
    let place_of = |sequence: u64| -> (usize, usize) {
        let live = contents
            .records
            .iter()
            .find(|live| live.record.sequence == sequence)
            .unwrap();
        (usize::from(sequence >= 17), live.record.offset as usize)
    };
    let first_of_call = |call: u64| 1 + 4 * call;
    let mut bytes = segments.bytes.clone();
    let (kept, cut_segment, cut_at) = match class {
        // The fifth call's second record in the last segment, with the
        // sixth call whole after it.
        "corrupt" => {
            let (segment, at) = place_of(first_of_call(4) + 1);
            bytes[segment][at + JOURNAL_RECORD_HEADER_BYTES + 1] ^= 0x40;
            (4, 1, SEGMENT_HEADER_BYTES)
        }
        // The second call's second record in the first segment, which is
        // not the last.
        "sealed" => {
            let (segment, at) = place_of(first_of_call(1) + 1);
            bytes[segment][at + JOURNAL_RECORD_HEADER_BYTES + 1] ^= 0x40;
            (1, 0, place_of(first_of_call(1)).1)
        }
        // The last segment's header, which then does not read.
        "header" => {
            bytes[1][40] ^= 0x40;
            (4, 1, 0)
        }
        // The third call's second record, its payload replaced by bytes
        // that do not decode under a checksum that holds.
        "decode" => {
            let sequence = first_of_call(2) + 1;
            let (segment, at) = place_of(sequence);
            let damaged = contents
                .records
                .iter()
                .find(|live| live.record.sequence == sequence)
                .unwrap()
                .record;
            let mut out = Vec::new();
            encode_journal_record(
                sequence,
                damaged.kind,
                &vec![0xFF; damaged.payload.len()],
                &mut out,
            );
            bytes[segment][at..at + out.len()].copy_from_slice(&out);
            (2, 0, place_of(first_of_call(2)).1)
        }
        // The stale first segment's header, which then does not read.
        "stale" => {
            bytes[0][40] ^= 0x40;
            (DAMAGED_CALLS, 0, 0)
        }
        other => panic!("no repair class {other}"),
    };
    let journal = crate::journal::journal_path(dir).unwrap();
    for (number, segment) in segments.numbers.iter().zip(&bytes) {
        std::fs::write(journal.join(format!("{number:020}.zdbseg")), segment).unwrap();
    }
    if class == "stale" {
        return (kept, bytes[0].clone());
    }
    let mut aside = bytes[cut_segment][cut_at..].to_vec();
    for later in &bytes[cut_segment + 1..] {
        aside.extend_from_slice(later);
    }
    (kept, aside)
}

/// What a quarantine holds: every file but the reports, by name in name
/// order with its bytes, the folders in name order, and whether any folder
/// holds the marker of an unfinished repair.
pub(super) type Quarantine = (Vec<(String, Vec<u8>)>, Vec<String>, bool);

/// What the quarantine of the journal beside `directory` holds.
pub(super) fn quarantine_of(directory: &Path) -> Quarantine {
    let root = crate::journal::journal_path(directory)
        .unwrap()
        .join("quarantine");
    let mut folders: Vec<String> = std::fs::read_dir(&root)
        .map(|entries| {
            entries
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    folders.sort();
    let mut files = Vec::new();
    let mut unsealed = false;
    for (path, bytes) in files_under(&[&root]) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        match name.as_str() {
            "repair.json" => {}
            "UNSEALED" => unsealed = true,
            _ => files.push((name, bytes)),
        }
    }
    files.sort();
    (files, folders, unsealed)
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
/// `quarantines` fails the creation of a repair's quarantine folder, and
/// `quarantines_denied` fails it with the system's refusal of a missing
/// permission. `dir_syncs` fails a sync of the journal directory's own
/// entries, `quarantine_syncs` a sync of a quarantine folder's, and
/// `in_place_denied` the sync of the directory in place with the system's
/// refusal of a missing permission.
#[derive(Debug, Default)]
pub(super) struct Faults {
    pub(super) writes: Arm,
    pub(super) syncs: Arm,
    pub(super) cuts: Arm,
    pub(super) creates: Arm,
    pub(super) quarantines: Arm,
    pub(super) quarantines_denied: Arm,
    pub(super) dir_syncs: Arm,
    pub(super) quarantine_syncs: Arm,
    pub(super) in_place_denied: Arm,
    /// Every sync the interval thread made through this directory.
    pub(super) thread_syncs: ThreadSyncs,
}

fn injected() -> io::Error {
    io::Error::other("injected: the device refused it")
}

/// The name `Flusher::start` gives the interval policy's thread, by which a
/// faulty directory tells that thread's syncs from every other.
const INTERVAL_THREAD: &str = "zeusdb-journal-sync";

/// The longest a closed gate holds the interval thread, and the longest a
/// test waits for one of its syncs.
pub(super) const SYNC_LIMIT: Duration = Duration::from_secs(20);

/// Every sync the interval thread made through a faulty directory, in the
/// order they began, each with the segment it synced and whether it has
/// ended, and a gate that holds the thread's syncs while a test keeps it
/// closed.
///
/// A sync is recorded as it begins, before the gate and before any armed
/// fault is read, and again as it ends, failed or not. A test that reads
/// the record once the collection is dropped, which joins the thread, reads
/// every sync the thread made.
#[derive(Debug, Default)]
pub(super) struct ThreadSyncs {
    state: Mutex<SyncRecord>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct SyncRecord {
    /// The segment of each sync, in the order they began, and whether it
    /// has ended.
    syncs: Vec<(u64, bool)>,
    /// The gate is closed.
    closed: bool,
    /// A sync waited at the gate for the whole limit and went on.
    timed_out: bool,
}

impl ThreadSyncs {
    /// Record a sync of `segment` beginning, and hold it while the gate is
    /// closed, for [`SYNC_LIMIT`] at most. Gives the sync's place.
    fn begin(&self, segment: u64) -> usize {
        let mut record = self.lock();
        let at = record.syncs.len();
        record.syncs.push((segment, false));
        self.changed.notify_all();
        let deadline = Instant::now() + SYNC_LIMIT;
        while record.closed {
            let now = Instant::now();
            if now >= deadline {
                record.timed_out = true;
                break;
            }
            record = self
                .changed
                .wait_timeout(record, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        at
    }

    /// Record the sync at place `at` ending.
    fn end(&self, at: usize) {
        self.lock().syncs[at].1 = true;
        self.changed.notify_all();
    }

    /// The syncs begun so far.
    pub(super) fn begun(&self) -> usize {
        self.lock().syncs.len()
    }

    /// The syncs ended so far.
    pub(super) fn ended(&self) -> usize {
        self.lock().syncs.iter().filter(|(_, ended)| *ended).count()
    }

    /// Hold the thread's syncs from here on.
    pub(super) fn close(&self) {
        self.lock().closed = true;
    }

    /// Let the thread's syncs go.
    pub(super) fn open(&self) {
        self.lock().closed = false;
        self.changed.notify_all();
    }

    /// Whether a sync waited at the gate for the whole limit.
    pub(super) fn timed_out(&self) -> bool {
        self.lock().timed_out
    }

    /// The segment of the sync at place `at`, once it has begun, or `None`
    /// where none begins there within [`SYNC_LIMIT`].
    pub(super) fn wait_begun(&self, at: usize) -> Option<u64> {
        self.wait(|record| record.syncs.get(at).map(|(segment, _)| *segment))
    }

    /// The segment of the first sync at place `from` or after it to end, or
    /// `None` where none ends within [`SYNC_LIMIT`].
    pub(super) fn wait_ended(&self, from: usize) -> Option<u64> {
        self.wait(|record| {
            record
                .syncs
                .get(from..)
                .and_then(|later| later.iter().find(|(_, ended)| *ended))
                .map(|(segment, _)| *segment)
        })
    }

    fn wait<T>(&self, found: impl Fn(&SyncRecord) -> Option<T>) -> Option<T> {
        let deadline = Instant::now() + SYNC_LIMIT;
        let mut record = self.lock();
        loop {
            if let Some(found) = found(&record) {
                return Some(found);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            record = self
                .changed
                .wait_timeout(record, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn lock(&self) -> MutexGuard<'_, SyncRecord> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
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
    /// The segment this file is.
    number: u64,
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
            number,
        }))
    }
    fn open_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>> {
        Ok(Box::new(FaultyFile {
            inner: self.inner.open_segment(number)?,
            faults: self.faults.clone(),
            number,
        }))
    }
    fn remove_segment(&self, number: u64) -> io::Result<()> {
        self.inner.remove_segment(number)
    }
    fn sync(&self) -> io::Result<()> {
        if self.faults.dir_syncs.trips() {
            return Err(injected());
        }
        self.inner.sync()
    }
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
    fn create_quarantine(&self, folder: &str) -> io::Result<()> {
        if self.faults.quarantines_denied.trips() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected: the folder may not be written",
            ));
        }
        if self.faults.quarantines.trips() {
            return Err(injected());
        }
        self.inner.create_quarantine(folder)
    }
    fn quarantine(&self) -> Box<dyn Dir> {
        self.inner.quarantine()
    }
    fn quarantine_folders(&self) -> io::Result<Vec<String>> {
        self.inner.quarantine_folders()
    }
    fn sync_quarantine(&self, folder: &str) -> io::Result<()> {
        if self.faults.quarantine_syncs.trips() {
            return Err(injected());
        }
        self.inner.sync_quarantine(folder)
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
        let thread = (std::thread::current().name() == Some(INTERVAL_THREAD))
            .then(|| self.faults.thread_syncs.begin(self.number));
        let synced = if self.faults.syncs.trips() {
            Err(injected())
        } else {
            self.inner.sync_data()
        };
        if let Some(at) = thread {
            self.faults.thread_syncs.end(at);
        }
        synced
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
    fn foreign_target(&self) -> Option<String> {
        self.inner.foreign_target()
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
    fn sync_in_place(&self) -> io::Result<()> {
        if self.faults.in_place_denied.trips() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected: the folder may not be written",
            ));
        }
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
    /// A medium that refuses every write has nothing of its own to sync.
    fn sync(&self) -> io::Result<()> {
        Ok(())
    }
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
    fn create_quarantine(&self, _folder: &str) -> io::Result<()> {
        Err(io::Error::new(self.kind, "the medium refuses writes"))
    }
    fn quarantine(&self) -> Box<dyn Dir> {
        self.inner.quarantine()
    }
    fn quarantine_folders(&self) -> io::Result<Vec<String>> {
        self.inner.quarantine_folders()
    }
    fn sync_quarantine(&self, _folder: &str) -> io::Result<()> {
        Ok(())
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
    fn foreign_target(&self) -> Option<String> {
        self.inner.foreign_target()
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
    /// A medium that refuses every write has nothing of its own to sync.
    fn sync_in_place(&self) -> io::Result<()> {
        Ok(())
    }
    fn cut_legacy_journal(&self, _len: u64) -> io::Result<()> {
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

// ============================================================================
// WHAT A THREAD SAYS WHILE A WRITE IS HELD
// ============================================================================

/// What a thread beside a held write says, in the order it says it: each
/// guard it begins to wait at, as the registry's wait hook hears it, and
/// what its call gave once it returned.
#[derive(Debug, PartialEq)]
pub(super) enum Said<T> {
    Waits(String),
    Done(T),
}

/// Say each guard this thread begins to wait at through `said`, from now on.
pub(super) fn say_waits<T: Send + 'static>(said: &mpsc::Sender<Said<T>>) {
    let waits = said.clone();
    on_wait(move |name| {
        let _ = waits.send(Said::Waits(name.to_string()));
    });
}

/// Whether `name`, the guard a wait was heard at, is one of `guards`, the
/// guards a held phase names. A build that keeps no ranks hears every wait
/// with an empty name and names no guard held, so there every wait is at
/// one of them.
pub(super) fn among(guards: &[String], name: &str) -> bool {
    !cfg!(debug_assertions) || guards.iter().any(|guard| guard == name)
}

/// The guard a thread waits at among `guards`, the guards a held phase
/// holds, read from what it says past any wait at another guard. A thread
/// that answers first went on without waiting for the phase, and one that
/// says nothing more within `limit` hung, and either fails the check with
/// `expected`, the words of what the check needs.
pub(super) fn waited_among<T>(
    said: &mpsc::Receiver<Said<T>>,
    limit: Duration,
    guards: &[String],
    expected: &str,
) -> String {
    wait_among(said, limit, guards, expected).unwrap_or_else(|words| panic!("{words}"))
}

/// [`waited_among`] with the failure given as its words rather than
/// raised, for a check read inside a hook, where a panic would unwind
/// under the guards the hooked call holds.
pub(super) fn wait_among<T>(
    said: &mpsc::Receiver<Said<T>>,
    limit: Duration,
    guards: &[String],
    expected: &str,
) -> Result<String, String> {
    let deadline = Instant::now() + limit;
    loop {
        match said.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Said::Waits(name)) if among(guards, &name) => return Ok(name),
            Ok(Said::Waits(_)) => {}
            Ok(Said::Done(_)) => {
                return Err(format!(
                    "{expected}, and it returned without waiting at any of {guards:?}"
                ))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "{expected}, and it said nothing within {limit:?}, so it hung"
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(format!("{expected}, and its thread ended without a word"))
            }
        }
    }
}

/// What a thread beside a held write gives once the hold is let go. The
/// write lets its guards go one at a time, so the thread can wait again
/// first, at a guard the write has yet to let go or one another thread
/// holds, and every wait it reports is read past until its answer. A thread
/// that gives no answer within `limit` hung, and one whose channel closes
/// without one ended first, and either fails the check with `expected`.
pub(super) fn answer_after<T>(
    said: &mpsc::Receiver<Said<T>>,
    limit: Duration,
    expected: &str,
) -> T {
    let deadline = Instant::now() + limit;
    loop {
        match said.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Said::Waits(_)) => {}
            Ok(Said::Done(answer)) => return answer,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("{expected}, and it gave no answer within {limit:?}, so it hung")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("{expected}, and its thread ended without an answer")
            }
        }
    }
}

/// The guards the threads that listen wait at now, one entry for each wait
/// in progress, for a phase held until one of them waits at a guard the
/// phase holds.
#[derive(Debug, Default)]
pub(super) struct Heard {
    waiting: Mutex<Vec<String>>,
}

impl Heard {
    /// Hear this thread's waits from now on, each from its beginning to its
    /// end, through the registry's wait hook.
    pub(super) fn listen(self: &Arc<Self>) {
        let heard = Arc::clone(self);
        WAIT_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |name, wait| {
                let mut waiting = heard.lock();
                match wait {
                    Wait::Begins => waiting.push(name.to_string()),
                    Wait::Ends => {
                        if let Some(at) = waiting.iter().position(|guard| guard == name) {
                            waiting.swap_remove(at);
                        }
                    }
                }
            }));
        });
    }

    /// Whether a thread that listens waits now at one of `guards`.
    pub(super) fn at_any(&self, guards: &[String]) -> bool {
        self.lock().iter().any(|name| among(guards, name))
    }

    fn lock(&self) -> MutexGuard<'_, Vec<String>> {
        self.waiting.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
