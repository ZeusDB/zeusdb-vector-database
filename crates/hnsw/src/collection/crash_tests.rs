//! What a killed process leaves, and what opening it again recovers.
//!
//! The journal exists for one case, being a process that stops without
//! warning, so it is held to that case by stopping processes without
//! warning. Every abort here is a real `std::process::abort` inside a real
//! write, taken at a named point in the engine's own code, with no
//! destructor run, no buffer flushed and no file closed. An injected error
//! would prove something else, because it unwinds.
//!
//! # The shape
//!
//! A child process runs a scripted sequence of mutations against a journaled
//! collection, acknowledging each step by writing the ids it took to a file
//! it fsyncs, and aborts at one named point. The parent then opens the
//! directory the child left, and holds it to four rules.
//!
//! Every record an acknowledged step added is present, with the vector and
//! the metadata it was given. Nothing the script never added is present. No
//! record of a call that returned refused is present. A record the step
//! the child died inside replaces is present in its old form or its new
//! one, and never absent. An unacknowledged record may be present or
//! absent, since a batch the child died inside leaves whatever prefix of
//! itself had reached the journal, and either is correct.
//!
//! The backup the script writes is held to its own rule: its directory is
//! absent where no backup was acknowledged, holds the last acknowledged
//! backup as it was, or holds the backup the child died inside complete.
//!
//! The parent then resumes the script in a second child from the recovered
//! state, runs it to the end, and holds the finished collection to the same
//! rules, and the backup to the state its last step backed up. That is what
//! proves the journal is usable after a recovery and not merely readable.
//!
//! # The matrix
//!
//! Thirty-six points: eleven inside an `add`, three of them reached only
//! where its commit fails and one only where it replaces a held record, two
//! in a roll by size, ten inside a checkpoint, five inside a backup, four
//! inside the upgrade of a journal of format version 1, three inside a
//! repair, and one with the lock held. The script's segments roll at a
//! length set so its last step passes it and no earlier step does. The
//! ordinary gate runs a reduced matrix, being every point once: the add,
//! overwrite, roll, checkpoint and backup points under the per-call arm, the
//! backup points at the second backup, the three failing points under the
//! failing arm at each step it fails, the lock, the upgrade over one fixture
//! and the repair of one class of damage. `ZEUSDB_KILL_MATRIX=full` runs
//! every point at every step under every arm, the upgrade over every fixture
//! and the repair of every class.
//!
//! # The quantized collections
//!
//! The full matrix runs the script over two more collections as well, at
//! every checkpoint and backup point under the three durability arms: a
//! trained scalar one and a trained product quantized one under
//! `quantized_only`. Each takes 1,000 records before the script, which train
//! it, so every save point runs over a quantized directory at format 5.0.0,
//! whose rows the graph dump alone holds. A record's vector there is the
//! decoding of its codes, which the parent takes through the recovered
//! collection's own codec or codebook.
//!
//! # The four arms
//!
//! Three are the durability policies, each by name. `sync` is
//! [`Durability::PerCall`], so every record a call returns from is on the
//! device. `none` is [`Durability::NoSync`], so the records are in the
//! kernel and nothing syncs them; a process abort loses nothing, which is
//! what these tests kill with. `interval` is [`Durability::PerInterval`] at
//! ten milliseconds, so the engine's own thread syncs the journal, and an
//! abort lands wherever that thread happens to be. The fourth, `fail`, runs
//! the per-call policy through a journal directory whose sync fails at the
//! commit of two steps, an `add` and the first overwrite, so each returns
//! its records refused and the child then takes the checkpoint the fault
//! requires before it goes on.
//!
//! # The repair cases
//!
//! Each builds a journal of six calls in two segments and damages it one
//! way: a corrupt middle in the last segment, damage in the segment that is
//! not the last, a header that will not read, or a record whose checksum
//! holds and which will not decode. A child loads it with a repair armed at
//! one of the repair's points. A plain load then refuses the damage where
//! the child died before its cut and holds the kept calls otherwise, and a
//! second repair finishes, holding the kept calls, one quarantine folder
//! with no marker, and the bytes set aside equal to the bytes the cut took.
//!
//! # How the child is run
//!
//! The parent spawns this test binary again with `--exact` naming the child
//! test, which is `#[ignore]`d so the ordinary run does not reach it, and
//! hands it the case in the environment. It is the only way to abort a
//! process from a test and keep the suite.

#![allow(clippy::disallowed_types)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};
use zeusdb_vector_core::{
    InsertParts, KillPoint, LockFailure, LockMode, Locked, Operation, Storage,
    JOURNAL_RECORD_OVERHEAD_BYTES, SEGMENT_HEADER_BYTES,
};

use crate::journal::{Durability, JournalPolicy};

use super::journal_support::{
    copy_fixture, damaged_journal, holds_fixture, manifest, quarantine_of, sink_over,
    FaultyStorage, DAMAGED_CALL_RECORDS, REPAIR_CLASSES,
};
use super::{Collection, Declaration, Int8Scale, ParsedRecord, QuantizationConfig, StorageMode};

// ============================================================================
// THE SCRIPT
// ============================================================================

/// The width of the vectors the script inserts.
const DIM: usize = 8;

/// One step of the script.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Step {
    /// Insert these records, by index into the corpus.
    Add(&'static [usize]),
    /// Insert these records in their overwritten form with `overwrite`, the
    /// first of them replacing records the collection holds and the last
    /// new.
    Overwrite(&'static [usize]),
    /// Remove these records.
    Remove(&'static [usize]),
    /// Replace this record's metadata.
    Update(usize),
    /// Save, which on a journaled collection is a checkpoint.
    Checkpoint,
    /// Write a backup into the script's one backup directory.
    Backup,
}

/// Every step, in order. Every record index is added exactly once, and
/// replaced at most once by an overwrite, so a record present that no
/// acknowledged step added is a defect rather than a duplicate.
const SCRIPT: &[Step] = &[
    Step::Add(&[0, 1, 2]),
    Step::Add(&[3, 4, 5, 6]),
    Step::Checkpoint,
    Step::Add(&[7]),
    Step::Remove(&[1]),
    Step::Add(&[8, 9, 10, 11, 12]),
    Step::Update(2),
    Step::Checkpoint,
    Step::Backup,
    Step::Add(&[13, 14]),
    Step::Overwrite(&[3, 4, 26]),
    Step::Add(&[15, 16, 17]),
    Step::Overwrite(&[5, 6, 27]),
    Step::Backup,
    Step::Add(&[18, 19, 20, 21, 22, 23, 24, 25]),
];

/// The highest record index the script ever adds.
const HIGHEST_RECORD: usize = 27;

/// The records a quantized collection takes before the script, which train
/// it, by index from `HIGHEST_RECORD + 1`.
const BASE_RECORDS: usize = 1000;

/// The collection the script runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// The raw collection the script was written for.
    Raw,
    /// A scalar quantized one, trained by the records it takes before the
    /// script.
    Int8,
    /// A product quantized one under `quantized_only`, trained the same way.
    Pq,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Raw => "raw",
            Shape::Int8 => "int8",
            Shape::Pq => "pq",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "int8" => Shape::Int8,
            "pq" => Shape::Pq,
            _ => Shape::Raw,
        }
    }

    /// The records it takes before the script.
    fn base(self) -> usize {
        match self {
            Shape::Raw => 0,
            Shape::Int8 | Shape::Pq => BASE_RECORDS,
        }
    }

    /// The highest record index it ever holds.
    fn highest(self) -> usize {
        HIGHEST_RECORD + self.base()
    }

    /// The quantization it declares.
    fn quantization(self, declaration: &Declaration) -> Option<QuantizationConfig> {
        match self {
            Shape::Raw => None,
            Shape::Int8 => Some(
                declaration
                    .scalar_quantization(
                        Int8Scale::PER_DIMENSION,
                        BASE_RECORDS,
                        None,
                        StorageMode::QuantizedOnly,
                    )
                    .unwrap(),
            ),
            Shape::Pq => Some(
                declaration
                    .quantization(4, 4, BASE_RECORDS, None, StorageMode::QuantizedOnly)
                    .unwrap(),
            ),
        }
    }

    /// The vector `collection` gives back for a record given `vector`: the
    /// vector on a raw collection, and the decoding of its codes through the
    /// collection's own codec or codebook on a quantized one.
    fn stored(self, collection: &Collection, vector: Vec<f32>) -> Vec<f32> {
        match self {
            Shape::Raw => vector,
            Shape::Int8 => {
                let codec = collection
                    .int8_codec()
                    .expect("a trained scalar collection holds its codec");
                codec
                    .reconstruct(&codec.quantize(&vector).unwrap())
                    .unwrap()
            }
            Shape::Pq => {
                let pq = collection
                    .pq()
                    .expect("a trained product quantized collection holds its codebook")
                    .clone();
                pq.reconstruct(&pq.quantize(&vector).unwrap()).unwrap()
            }
        }
    }
}

/// The steps whose commit the failing arm fails, being an `add` and the
/// first overwrite.
const FAILING_STEPS: [usize; 2] = [5, 10];

/// The step that passes the segment's roll length.
const ROLL_STEP: usize = 14;

/// The points inside an `add` whose commit holds, in the order the call
/// reaches them.
const ADD_POINTS: &[KillPoint] = &[
    KillPoint::AddBeforeFirstAppend,
    KillPoint::AddMidAppend,
    KillPoint::AddAfterSecondAppend,
    KillPoint::AddAfterAppendBeforeCommit,
    KillPoint::AddAfterCommitRecord,
    KillPoint::AddAfterCommitBeforeApply,
    KillPoint::AddAfterApply,
];

/// The points inside an `add` that replaces held records, in the order the
/// call reaches them.
const OVERWRITE_POINTS: &[KillPoint] = &[
    KillPoint::AddBeforeFirstAppend,
    KillPoint::AddMidAppend,
    KillPoint::OverwriteAfterRemovalAppend,
    KillPoint::AddAfterSecondAppend,
    KillPoint::AddAfterAppendBeforeCommit,
    KillPoint::AddAfterCommitRecord,
    KillPoint::AddAfterCommitBeforeApply,
    KillPoint::AddAfterApply,
];

/// The points inside an `add` whose commit's sync fails.
const FAILING_POINTS: &[KillPoint] = &[
    KillPoint::AddAfterFailedCommit,
    KillPoint::AddMidAbort,
    KillPoint::AddAfterAbortBeforeSync,
];

/// The points of a roll by size.
const ROLL_POINTS: &[KillPoint] = &[KillPoint::RollAfterSeal, KillPoint::RollAfterCreate];

/// The points inside a checkpoint, in the order the call reaches them.
const CHECKPOINT_POINTS: &[KillPoint] = &[
    KillPoint::CheckpointBeforeSave,
    KillPoint::SaveAfterArtefacts,
    KillPoint::SaveAfterDump,
    KillPoint::SaveAfterManifest,
    KillPoint::SaveBetweenRenames,
    KillPoint::SaveAfterCommit,
    KillPoint::CheckpointAfterSaveBeforeRoll,
    KillPoint::CheckpointAfterRoll,
    KillPoint::CheckpointMidDelete,
    KillPoint::CheckpointAfterDelete,
];

/// The points inside a backup, in the order the call reaches them.
const BACKUP_POINTS: &[KillPoint] = &[
    KillPoint::BackupAfterSync,
    KillPoint::BackupAfterArtefacts,
    KillPoint::BackupAfterManifest,
    KillPoint::BackupBetweenRenames,
    KillPoint::BackupAfterCommit,
];

/// The points inside the upgrade of a journal of format version 1.
const UPGRADE_POINTS: &[KillPoint] = &[
    KillPoint::UpgradeAfterReplay,
    KillPoint::UpgradeAfterCommit,
    KillPoint::UpgradeAfterSegment,
    KillPoint::UpgradeAfterOldRemoved,
];

/// The points inside a repair, in the order the load reaches them.
const REPAIR_POINTS: &[KillPoint] = &[
    KillPoint::RepairAfterQuarantine,
    KillPoint::RepairAfterCut,
    KillPoint::RepairAfterCheckpoint,
];

/// The corpus, drawn the same way in every process, so the parent knows what
/// vector each record was given.
fn corpus() -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(HIGHEST_RECORD + 1 + BASE_RECORDS);
    // A generator of its own rather than the engine's, because the parent
    // and the child have to agree on the bytes and nothing else depends on
    // them. One multiplicative step per value.
    let mut state: u64 = 0x5eed_0148;
    for _ in 0..=HIGHEST_RECORD + BASE_RECORDS {
        let mut vector = Vec::with_capacity(DIM);
        for _ in 0..DIM {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            vector.push(((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        vectors.push(vector);
    }
    vectors
}

/// The form a record takes: as the script first adds it, with its metadata
/// replaced, or as an overwrite inserts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Original,
    Updated,
    Overwritten,
}

fn metadata_for(i: usize) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("i".to_string(), json!(i)),
        ("updated".to_string(), json!(false)),
    ])
}

fn updated_metadata(i: usize) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("i".to_string(), json!(i)),
        ("updated".to_string(), json!(true)),
    ])
}

fn overwritten_metadata(i: usize) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("i".to_string(), json!(i)),
        ("overwritten".to_string(), json!(true)),
    ])
}

/// The vector an overwrite gives record `i`, which is not the one the
/// script first gave it.
fn overwritten_vector(i: usize, vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors[i].iter().map(|value| 0.25 - value).collect()
}

/// The vector and the metadata record `i` holds in `form`.
fn form_of(i: usize, form: Form, vectors: &[Vec<f32>]) -> (Vec<f32>, BTreeMap<String, Value>) {
    match form {
        Form::Original => (vectors[i].clone(), metadata_for(i)),
        Form::Updated => (vectors[i].clone(), updated_metadata(i)),
        Form::Overwritten => (overwritten_vector(i, vectors), overwritten_metadata(i)),
    }
}

fn parsed(i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vectors[i].clone(),
        sparse: None,
        metadata: metadata_for(i).into_iter().collect(),
    }
}

fn overwritten(i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: overwritten_vector(i, vectors),
        sparse: None,
        metadata: overwritten_metadata(i).into_iter().collect(),
    }
}

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 8, 50, 200, vec![]).unwrap()
}

/// The bytes one insert record of the script and one commit record take.
fn record_lengths(vectors: &[Vec<f32>]) -> (u64, u64) {
    let metadata: HashMap<String, Value> = metadata_for(10).into_iter().collect();
    let mut payload = Vec::new();
    Operation::encode_insert(
        &InsertParts {
            id: "r10",
            vector: &vectors[10],
            metadata: &metadata,
            sparse: None,
        },
        &mut payload,
    );
    let insert = (payload.len() + JOURNAL_RECORD_OVERHEAD_BYTES) as u64;
    let commit = (8 + JOURNAL_RECORD_OVERHEAD_BYTES) as u64;
    (insert, commit)
}

/// The length the child's segments roll at: thirteen of the script's
/// inserts and four commits past a segment's header. The segment after the
/// second checkpoint takes eleven inserts, two removals and four commits by
/// the second backup, short of it, and passes it at the last step, as the
/// failing arm's segment after its last checkpoint does with eight inserts,
/// one removal and two commits before the last step. Every earlier segment
/// holds less, so the last step's commit rolls and no earlier commit does.
fn roll_length(vectors: &[Vec<f32>]) -> u64 {
    let (insert, commit) = record_lengths(vectors);
    SEGMENT_HEADER_BYTES as u64 + 13 * insert + 4 * commit
}

/// The one directory every backup of the script writes into, beside the
/// collection's.
fn backup_target(dir: &Path) -> PathBuf {
    dir.with_extension("backup.zdb")
}

// ============================================================================
// THE CASE, PASSED THROUGH THE ENVIRONMENT
// ============================================================================

const ENV_DIR: &str = "ZEUSDB_KILL_DIR";
const ENV_MODE: &str = "ZEUSDB_KILL_MODE";
const ENV_TARGET: &str = "ZEUSDB_KILL_TARGET";
const ENV_MARKER: &str = "ZEUSDB_KILL_MARKER";
const ENV_RESUME: &str = "ZEUSDB_KILL_RESUME";
const ENV_MATRIX: &str = "ZEUSDB_KILL_MATRIX";
const ENV_UPGRADE: &str = "ZEUSDB_KILL_UPGRADE";
const ENV_REPAIR: &str = "ZEUSDB_KILL_REPAIR";
const ENV_SHAPE: &str = "ZEUSDB_KILL_SHAPE";

/// The name of the child test, as `--exact` spells it.
const CHILD_TEST: &str = "collection::crash_tests::the_kill_matrixs_child";

/// The policy each arm runs under.
fn durability(arm: &str) -> Durability {
    match arm {
        "sync" | "fail" => Durability::PerCall,
        "interval" => Durability::PerInterval(Duration::from_millis(10)),
        _ => Durability::NoSync,
    }
}

// ============================================================================
// THE ACKNOWLEDGEMENTS
// ============================================================================

/// Append a line naming what a step did and fsync it, so a process killed
/// afterwards leaves the acknowledgement behind.
fn acknowledge(path: &Path, line: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("the acknowledgement file opens");
    writeln!(file, "{line}").expect("the acknowledgement is written");
    file.sync_all().expect("the acknowledgement is synced");
}

/// What the acknowledgements say the collection and its backup must hold.
#[derive(Debug, Default)]
struct Acknowledged {
    /// Every record an acknowledged step added and no acknowledged step
    /// removed, against the form it was acknowledged in.
    present: BTreeMap<String, Form>,
    /// Every record of a call that returned refused that no acknowledged
    /// step holds.
    refused: BTreeSet<String>,
    /// Every acknowledged backup: the snapshot the copy records, and what
    /// the collection held when it was taken.
    backups: Vec<(String, BTreeMap<String, Form>)>,
    steps: usize,
    finished: bool,
}

fn acknowledged(path: &Path) -> Acknowledged {
    let mut out = Acknowledged::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        if line == "done" {
            out.finished = true;
            continue;
        }
        if let Some(ids) = line.strip_prefix("base add ") {
            for id in ids.split(',') {
                out.present.insert(id.to_string(), Form::Original);
            }
            continue;
        }
        let parts: Vec<&str> = line.split(' ').collect();
        if parts.len() < 3 || parts[0] != "step" {
            continue;
        }
        out.steps += 1;
        match parts[2] {
            "add" => {
                for id in parts[3].split(',') {
                    out.present.insert(id.to_string(), Form::Original);
                }
            }
            "overwrite" => {
                for id in parts[3].split(',') {
                    out.present.insert(id.to_string(), Form::Overwritten);
                }
            }
            "refused" => {
                for id in parts[3].split(',') {
                    if !out.present.contains_key(id) {
                        out.refused.insert(id.to_string());
                    }
                }
            }
            "remove" => {
                for id in parts[3].split(',') {
                    out.present.remove(id);
                }
            }
            "update" => {
                out.present.insert(parts[3].to_string(), Form::Updated);
            }
            "backup" => {
                out.backups
                    .push((parts[3].to_string(), out.present.clone()));
            }
            _ => {}
        }
    }
    out
}

/// Hold a recovered collection to the acknowledgements, with `in_flight`
/// the step the child died inside, where it died inside one.
///
/// Every acknowledged record present with the vector and the metadata of
/// the form it was acknowledged in, or of its overwritten form where the
/// step in flight replaces it, nothing present that the script never added,
/// and no record of a call that returned refused present. What is not held
/// is the presence of an unacknowledged record, which is correct either
/// way. Returns the records present that no acknowledgement names.
fn holds(
    collection: &Collection,
    expected: &Acknowledged,
    vectors: &[Vec<f32>],
    in_flight: Option<&Step>,
    shape: Shape,
) -> Result<usize, String> {
    let replacing: BTreeSet<String> = match in_flight {
        Some(Step::Overwrite(records)) => records.iter().map(|i| format!("r{i}")).collect(),
        _ => BTreeSet::new(),
    };
    holds_present(collection, &expected.present, vectors, &replacing, shape)?;
    let live: BTreeSet<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    for id in &expected.refused {
        if live.contains(id) {
            return Err(format!(
                "record {id} of a call that returned refused is present"
            ));
        }
    }
    for id in &live {
        let i: usize = id[1..]
            .parse()
            .map_err(|_| format!("record {id} was never in the script"))?;
        if i > shape.highest() {
            return Err(format!("record {id} was never in the script"));
        }
    }
    Ok(live.len() - expected.present.len())
}

/// Every record of `present` held in its form, or in its overwritten form
/// where `replacing` names it, and never absent, its vector the one `shape`
/// gives back.
fn holds_present(
    collection: &Collection,
    present: &BTreeMap<String, Form>,
    vectors: &[Vec<f32>],
    replacing: &BTreeSet<String>,
    shape: Shape,
) -> Result<(), String> {
    for (id, form) in present {
        if !collection.contains(id) {
            return Err(format!("acknowledged record {id} is missing"));
        }
        let view = collection
            .records(vec![id.clone()], true, true)
            .map_err(|e| e.to_string())?;
        let view = view
            .first()
            .ok_or_else(|| format!("{id} came back empty"))?;
        let i: usize = id[1..].parse().unwrap();
        let matches = |form: Form| {
            let (vector, metadata) = form_of(i, form, vectors);
            let vector = shape.stored(collection, vector);
            view.vector.as_deref() == Some(&vector[..])
                && metadata.len() == view.metadata.len()
                && metadata
                    .iter()
                    .all(|(key, value)| view.metadata.get(key) == Some(value))
        };
        if !(matches(*form) || replacing.contains(id) && matches(Form::Overwritten)) {
            return Err(format!(
                "record {id} came back as {:?} {:?}, which is neither its acknowledged form \
                 {form:?} nor a form the step in flight gives it",
                view.vector, view.metadata
            ));
        }
    }
    Ok(())
}

/// Hold `collection` to holding exactly `present`, each in its form.
fn holds_exactly(
    collection: &Collection,
    present: &BTreeMap<String, Form>,
    vectors: &[Vec<f32>],
    shape: Shape,
) -> Result<(), String> {
    holds_present(collection, present, vectors, &BTreeSet::new(), shape)?;
    if collection.len() != present.len() {
        return Err(format!(
            "{} records are held and {} were expected",
            collection.len(),
            present.len()
        ));
    }
    Ok(())
}

/// Hold the backup directory beside `dir` to the acknowledgements, with
/// `in_flight` the step the child died inside: absent where no backup was
/// acknowledged, the last acknowledged backup as it was, or, where the
/// child died inside a backup, that backup complete.
fn backup_holds(
    dir: &Path,
    expected: &Acknowledged,
    vectors: &[Vec<f32>],
    in_flight: Option<&Step>,
    shape: Shape,
) -> Result<&'static str, String> {
    let target = backup_target(dir);
    let aside = target.with_extension("zdb.zdbold");
    if !target.exists() && !aside.exists() {
        return match expected.backups.last() {
            None => Ok("absent"),
            Some((snapshot, _)) => Err(format!("the acknowledged backup {snapshot} is gone")),
        };
    }
    let (copy, report) = Collection::recover(&target.to_string_lossy(), None, Durability::NoSync)
        .map_err(|e| format!("the backup does not open: {e}"))?;
    if report.journaled {
        return Err("the backup names a journal".to_string());
    }
    let snapshot = copy
        .identity()
        .snapshot
        .map(|snapshot| format!("{snapshot:032x}"))
        .unwrap_or_default();
    if let Some((acknowledged, present)) = expected.backups.last() {
        if *acknowledged == snapshot {
            holds_exactly(&copy, present, vectors, shape)?;
            return Ok("as it was");
        }
    }
    if in_flight != Some(&Step::Backup) {
        return Err(format!(
            "the backup records snapshot {snapshot}, which no acknowledged backup names, and no \
             backup was in flight"
        ));
    }
    holds_exactly(&copy, &expected.present, vectors, shape)?;
    Ok("complete")
}

// ============================================================================
// THE CHILD
// ============================================================================

/// The child of the kill matrix, run in a process of its own.
///
/// Ignored, so the ordinary suite does not reach it, and a run that reaches
/// it with no case in the environment does nothing.
#[test]
#[ignore = "the parent runs this in a process of its own, armed to abort"]
fn the_kill_matrixs_child() {
    let Ok(dir) = std::env::var(ENV_DIR) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let arm = std::env::var(ENV_MODE).unwrap_or_else(|_| "sync".into());
    let target = std::env::var(ENV_TARGET).unwrap_or_default();
    let resume = std::env::var(ENV_RESUME).is_ok();
    let marker = PathBuf::from(std::env::var(ENV_MARKER).unwrap_or_default());
    let acknowledgements = dir.with_extension("ack");
    let (target_step, target_point) = match target.split_once(':') {
        Some((step, point)) => (step.parse::<usize>().ok(), point.to_string()),
        None => (None, String::new()),
    };
    let point = KillPoint::from_name(&target_point);
    let shape = Shape::from_name(&std::env::var(ENV_SHAPE).unwrap_or_default());
    let vectors = corpus();
    let durability = durability(&arm);
    zeusdb_vector_core::set_segment_roll_bytes(roll_length(&vectors));

    // An upgrade case loads a copy of a fixture, and a repair case a
    // damaged journal with a repair, each armed inside the load.
    let policy = if std::env::var(ENV_UPGRADE).is_ok() {
        Some(JournalPolicy::Replay(Durability::PerCall))
    } else if std::env::var(ENV_REPAIR).is_ok() {
        Some(JournalPolicy::Repair(Durability::PerCall))
    } else {
        None
    };
    if let Some(policy) = policy {
        if let Some(point) = point {
            zeusdb_vector_core::kill_arm(point, &marker);
        }
        let opened = Collection::recover_with(&dir.to_string_lossy(), None, policy);
        zeusdb_vector_core::kill_disarm();
        opened.expect("the directory opens");
        return;
    }

    let mut faults = None;
    let (collection, start_at) = if resume {
        let (collection, report) = Collection::recover(&dir.to_string_lossy(), None, durability)
            .expect("the directory the parent left opens");
        eprintln!("resumed: {report:?}");
        let done = std::fs::read_to_string(&acknowledgements)
            .map(|text| {
                text.lines()
                    .filter(|line| line.starts_with("step "))
                    .count()
            })
            .unwrap_or(0);
        (collection, done)
    } else {
        let collection = Collection::build(declaration(), shape.quantization(&declaration()));
        let _ = std::fs::remove_file(&acknowledgements);
        if point == Some(KillPoint::LockHeld) {
            zeusdb_vector_core::kill_arm(KillPoint::LockHeld, &marker);
        }
        if arm == "fail" {
            let storage = FaultyStorage::at(&dir);
            faults = Some(storage.faults.clone());
            let sink = sink_over(&storage, &dir, collection.collection_id(), durability);
            collection.attach_sink(Box::new(sink));
            collection
                .save(&dir.to_string_lossy())
                .expect("the journal's first checkpoint is written");
        } else {
            collection
                .journal_to(&dir.to_string_lossy(), durability)
                .expect("the journal opens beside the directory");
        }
        // The records a quantized collection takes before the script, which
        // train it.
        if shape.base() > 0 {
            let base: Vec<usize> = (HIGHEST_RECORD + 1..=shape.highest()).collect();
            let added = collection.add_records(
                base.iter().map(|&i| parsed(i, &vectors)).collect(),
                vec![],
                false,
            );
            assert_eq!(added.total_errors, 0, "{:?}", added.errors);
            assert!(collection.is_quantized(), "the base trains the collection");
            let ids: Vec<String> = base.iter().map(|i| format!("r{i}")).collect();
            acknowledge(&acknowledgements, &format!("base add {}", ids.join(",")));
        }
        (collection, 0)
    };

    for (index, step) in SCRIPT.iter().enumerate().skip(start_at) {
        match (target_step == Some(index)).then_some(point).flatten() {
            Some(point) => zeusdb_vector_core::kill_arm(point, &marker),
            None => zeusdb_vector_core::kill_disarm(),
        }
        match step {
            Step::Add(records) | Step::Overwrite(records) => {
                let replaces = matches!(step, Step::Overwrite(_));
                let ids: Vec<String> = records.iter().map(|i| format!("r{i}")).collect();
                let build = |i: usize| {
                    if replaces {
                        overwritten(i, &vectors)
                    } else {
                        parsed(i, &vectors)
                    }
                };
                // The failing arm's steps: the commit's sync fails, the call
                // returns every record refused, and the fault it leaves is
                // cleared by the checkpoint the caller then takes. A refused
                // overwrite leaves every record it replaces as it was.
                if let (Some(faults), true) = (&faults, FAILING_STEPS.contains(&index)) {
                    faults.syncs.arm(0, 1);
                    let batch: Vec<ParsedRecord> = records.iter().map(|&i| build(i)).collect();
                    let added = collection.add_records(batch, vec![], replaces);
                    assert_eq!(added.total_errors, records.len(), "{:?}", added.errors);
                    acknowledge(
                        &acknowledgements,
                        &format!("step {index} refused {}", ids.join(",")),
                    );
                    collection
                        .checkpoint()
                        .expect("the checkpoint the fault requires runs");
                    acknowledge(
                        &acknowledgements,
                        &format!("fault {index} checkpoint {}", collection.journal_sequence()),
                    );
                    continue;
                }
                // A step re-run after a recovery may find some of its
                // records already durable and replayed. An add refuses them
                // as duplicates exactly as it would any retry, so a caller
                // resuming skips what is present, and an overwrite replaces
                // them again with the same records.
                let batch: Vec<ParsedRecord> = records
                    .iter()
                    .filter(|&&i| replaces || !collection.contains(&format!("r{i}")))
                    .map(|&i| build(i))
                    .collect();
                if !batch.is_empty() {
                    let added = collection.add_records(batch, vec![], replaces);
                    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
                }
                let verb = if replaces { "overwrite" } else { "add" };
                acknowledge(
                    &acknowledgements,
                    &format!("step {index} {verb} {}", ids.join(",")),
                );
            }
            Step::Remove(records) => {
                let ids: Vec<String> = records.iter().map(|i| format!("r{i}")).collect();
                collection.remove_points(&ids).expect("the removal runs");
                acknowledge(
                    &acknowledgements,
                    &format!("step {index} remove {}", ids.join(",")),
                );
            }
            Step::Update(i) => {
                collection
                    .update_metadata(&format!("r{i}"), updated_metadata(*i).into_iter().collect())
                    .expect("the metadata replacement runs");
                acknowledge(&acknowledgements, &format!("step {index} update r{i}"));
            }
            Step::Checkpoint => {
                collection
                    .save(&dir.to_string_lossy())
                    .expect("the checkpoint runs");
                acknowledge(
                    &acknowledgements,
                    &format!("step {index} checkpoint {}", collection.journal_sequence()),
                );
            }
            Step::Backup => {
                let copy = collection
                    .backup(&backup_target(&dir).to_string_lossy())
                    .expect("the backup runs");
                let snapshot = copy.identity.snapshot.expect("a backup draws a snapshot");
                acknowledge(
                    &acknowledgements,
                    &format!("step {index} backup {snapshot:032x}"),
                );
            }
        }
    }
    zeusdb_vector_core::kill_disarm();
    acknowledge(&acknowledgements, "done");
}

// ============================================================================
// THE PARENT
// ============================================================================

/// One case of the matrix, being an arm, a step of the script and a point
/// inside it, a fixture and a point inside its upgrade, or a class of
/// damage and a point inside its repair.
#[derive(Clone, Debug)]
struct Case {
    arm: &'static str,
    step: usize,
    point: KillPoint,
    fixture: Option<&'static str>,
    repair: Option<&'static str>,
    shape: Shape,
}

/// What a case is called in the matrix's lines and its folders.
fn case_name(case: &Case) -> String {
    let point = case.point.name().replace(':', "_");
    match (case.fixture, case.repair) {
        (Some(fixture), _) => format!("upgrade-{fixture}-{point}"),
        (None, Some(class)) => format!("repair-{class}-{point}"),
        (None, None) if case.shape == Shape::Raw => format!("{}-{}-{point}", case.arm, case.step),
        (None, None) => format!("{}-{}-{}-{point}", case.shape.name(), case.arm, case.step),
    }
}

/// The points a step of the script reaches, with the roll's at the step
/// that rolls.
fn points_of(step: usize) -> Vec<KillPoint> {
    let mut points: Vec<KillPoint> = match &SCRIPT[step] {
        Step::Add(_) => ADD_POINTS.to_vec(),
        Step::Overwrite(_) => OVERWRITE_POINTS.to_vec(),
        Step::Checkpoint => CHECKPOINT_POINTS.to_vec(),
        Step::Backup => BACKUP_POINTS.to_vec(),
        Step::Remove(_) | Step::Update(_) => Vec::new(),
    };
    if step == ROLL_STEP {
        points.extend_from_slice(ROLL_POINTS);
    }
    points
}

/// Every case the script can reach, or the reduced set the ordinary gate
/// runs, being every point once.
fn cases(full: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    let case = |arm: &'static str, step: usize, point: KillPoint| Case {
        arm,
        step,
        point,
        fixture: None,
        repair: None,
        shape: Shape::Raw,
    };
    let arms: &[&'static str] = if full {
        &["sync", "none", "interval"]
    } else {
        &["sync"]
    };
    // The first backup writes into a path with nothing at it, so its commit
    // is one rename and the point between the two is reached at the second
    // alone, which is where the reduced matrix takes every backup point.
    let last_backup = SCRIPT
        .iter()
        .rposition(|step| *step == Step::Backup)
        .expect("the script backs up");
    for &arm in arms {
        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for (step, kind) in SCRIPT.iter().enumerate() {
            if !full && *kind == Step::Backup && step != last_backup {
                continue;
            }
            for point in points_of(step) {
                if full || seen.insert(point.name()) {
                    cases.push(case(arm, step, point));
                }
            }
        }
    }
    // The failing arm: the three failing points at each step it fails, and
    // in the full matrix every point of every step after the first.
    for step in FAILING_STEPS {
        for point in FAILING_POINTS {
            cases.push(case("fail", step, *point));
        }
    }
    if full {
        for step in FAILING_STEPS[0] + 1..SCRIPT.len() {
            for point in points_of(step) {
                cases.push(case("fail", step, point));
            }
        }
    }
    cases.push(case("sync", 0, KillPoint::LockHeld));
    let fixtures: &[&'static str] = if full {
        &super::journal_support::FIXTURES
    } else {
        &super::journal_support::FIXTURES[..1]
    };
    for fixture in fixtures {
        for point in UPGRADE_POINTS {
            cases.push(Case {
                fixture: Some(fixture),
                ..case("upgrade", 0, *point)
            });
        }
    }
    let classes: &[&'static str] = if full {
        &REPAIR_CLASSES
    } else {
        &REPAIR_CLASSES[..1]
    };
    for class in classes {
        for point in REPAIR_POINTS {
            cases.push(Case {
                repair: Some(class),
                ..case("repair", 0, *point)
            });
        }
    }
    // The quantized collections: every checkpoint and backup point at each
    // step that saves, under the three durability arms.
    if full {
        for shape in [Shape::Int8, Shape::Pq] {
            for &arm in arms {
                for (step, kind) in SCRIPT.iter().enumerate() {
                    if !matches!(kind, Step::Checkpoint | Step::Backup) {
                        continue;
                    }
                    for point in points_of(step) {
                        cases.push(Case {
                            shape,
                            ..case(arm, step, point)
                        });
                    }
                }
            }
        }
    }
    cases
}

fn child_command(dir: &Path, arm: &str, shape: Shape) -> std::process::Command {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
        .env(ENV_DIR, dir)
        .env(ENV_MODE, arm)
        .env(ENV_SHAPE, shape.name())
        .env_remove(ENV_MATRIX)
        .env_remove(ENV_UPGRADE)
        .env_remove(ENV_REPAIR);
    command
}

/// A directory the whole matrix runs under, removed when it is done.
struct Work(PathBuf);

impl Work {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("zeusdb-kill-matrix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Work(path)
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The damage a report found, as a line says it.
fn damage_of(report: &crate::Recovery) -> String {
    match &report.damage {
        Some(damage) => format!("{} at {}", damage.kind.replace('_', " "), damage.sequence),
        None => "none".to_string(),
    }
}

/// An upgrade case: the child loads a copy of the fixture armed at the
/// point, and the directory it leaves opens holding every record the
/// fixture records, at 5.0.0, and again on a second open.
fn upgrade_case(work: &Path, case: &Case, name: &str) -> Option<String> {
    let fixture = case.fixture.unwrap();
    let dir = copy_fixture(fixture, &work.join(name));
    let marker = work.join(format!("{name}.marker"));
    let status = child_command(&dir, "upgrade", Shape::Raw)
        .env(ENV_UPGRADE, "1")
        .env(ENV_TARGET, format!("0:{}", case.point.name()))
        .env(ENV_MARKER, &marker)
        .output()
        .expect("the child runs");
    assert!(
        marker.exists(),
        "{name}: the upgrade never reached its point"
    );
    assert!(!status.status.success(), "{name}: the child did not abort");
    let (upgraded, report) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
        .unwrap_or_else(|e| panic!("recovering {name} refused: {e}"));
    holds_fixture(&upgraded, fixture).unwrap_or_else(|e| panic!("after {name}: {e}"));
    assert_eq!(manifest(&dir)["format_version"], json!("5.0.0"), "{name}");
    drop(upgraded);
    let (again, _) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
        .unwrap_or_else(|e| panic!("reopening {name} refused: {e}"));
    holds_fixture(&again, fixture).unwrap_or_else(|e| panic!("reopening {name}: {e}"));
    Some(format!(
        "{name}: upgraded from {:?} replayed {} damage {}",
        report.upgraded_from,
        report.replayed,
        damage_of(&report)
    ))
}

/// The records the first `calls` calls of a repair case's journal add.
fn kept_records(calls: usize) -> BTreeMap<String, Form> {
    (0..DAMAGED_CALL_RECORDS * calls)
        .map(|i| (format!("r{i}"), Form::Original))
        .collect()
}

/// A repair case: a damaged journal, a child loading it with a repair armed
/// at the point, then a plain load, a second repair and a plain load again,
/// each held to the calls the repair keeps.
fn repair_case(work: &Path, case: &Case, name: &str, vectors: &[Vec<f32>]) -> Option<String> {
    let class = case.repair.unwrap();
    let dir = work.join(format!("{name}.zdb"));
    let (calls, aside) = damaged_journal(&dir, class, declaration(), |i| parsed(i, vectors));
    let kept = kept_records(calls);
    let marker = work.join(format!("{name}.marker"));
    let status = child_command(&dir, "repair", Shape::Raw)
        .env(ENV_REPAIR, "1")
        .env(ENV_TARGET, format!("0:{}", case.point.name()))
        .env(ENV_MARKER, &marker)
        .output()
        .expect("the child runs");
    assert!(
        marker.exists(),
        "{name}: the repair never reached its point"
    );
    assert!(!status.status.success(), "{name}: the child did not abort");

    // A plain load meets the damage where the child died before its cut,
    // and the journal the cut left otherwise.
    match Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync) {
        Ok((plain, _)) => {
            assert_ne!(
                case.point,
                KillPoint::RepairAfterQuarantine,
                "{name}: a plain load opened a journal still damaged"
            );
            holds_exactly(&plain, &kept, vectors, Shape::Raw)
                .unwrap_or_else(|e| panic!("{name}, the plain load: {e}"));
        }
        Err(error) => assert_eq!(
            case.point,
            KillPoint::RepairAfterQuarantine,
            "{name}: a plain load refused the journal the cut left: {error}"
        ),
    }

    // A second repair finishes what the first began.
    let (repaired, report) = Collection::recover_with(
        &dir.to_string_lossy(),
        None,
        JournalPolicy::Repair(Durability::NoSync),
    )
    .unwrap_or_else(|e| panic!("{name}: the second repair refused: {e}"));
    holds_exactly(&repaired, &kept, vectors, Shape::Raw)
        .unwrap_or_else(|e| panic!("{name}, the second repair: {e}"));
    drop(repaired);
    let (files, folders, unsealed) = quarantine_of(&dir);
    assert_eq!(folders.len(), 1, "{name}: one quarantine folder");
    assert!(!unsealed, "{name}: the repair's marker is gone");
    let held: Vec<u8> = files.into_iter().flat_map(|(_, bytes)| bytes).collect();
    assert!(
        held == aside,
        "{name}: the quarantine holds {} bytes where the cut took {}",
        held.len(),
        aside.len()
    );

    // And the directory opens plainly from then on.
    let (again, _) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
        .unwrap_or_else(|e| panic!("{name}: reopening after the repair refused: {e}"));
    holds_exactly(&again, &kept, vectors, Shape::Raw)
        .unwrap_or_else(|e| panic!("{name}, reopened: {e}"));
    Some(format!(
        "{name}: kept {} calls, second repair cut {} checkpoint {:?}",
        calls,
        report.repair.is_some(),
        report.checkpoint_taken
    ))
}

/// The lock case: the child aborts holding the lock, before anything is
/// written, so the directory does not exist, the lock is free again, and
/// the script run from the start in a new child reaches its end.
fn lock_case(work: &Path, name: &str, vectors: &[Vec<f32>]) -> Option<String> {
    let dir = work.join(format!("{name}.zdb"));
    let marker = work.join(format!("{name}.marker"));
    let status = child_command(&dir, "sync", Shape::Raw)
        .env(ENV_TARGET, format!("0:{}", KillPoint::LockHeld.name()))
        .env(ENV_MARKER, &marker)
        .env_remove(ENV_RESUME)
        .output()
        .expect("the child runs");
    assert!(marker.exists(), "{name}: the child never reached the lock");
    assert!(!status.status.success(), "{name}: the child did not abort");
    assert!(!dir.exists(), "{name}: nothing was written under the lock");
    let storage = zeusdb_vector_core::FsStorage::at(&dir).unwrap();
    // Windows may release the lock a few milliseconds after the process has
    // exited, so the lock is tried again for a bounded time.
    let start = std::time::Instant::now();
    loop {
        match storage.lock(LockMode::Exclusive, "") {
            Ok(Locked::Writable(guard)) => {
                drop(guard);
                break;
            }
            Err(LockFailure::Held { .. }) if start.elapsed() < Duration::from_secs(10) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            other => panic!("{name}: the killed holder's lock was not released: {other:?}"),
        }
    }
    let status = child_command(&dir, "sync", Shape::Raw)
        .env(ENV_TARGET, "")
        .env_remove(ENV_RESUME)
        .env_remove(ENV_MARKER)
        .output()
        .expect("the child runs");
    assert!(
        status.status.success(),
        "{name}: the script after the lock case failed:\n{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let expected = acknowledged(&dir.with_extension("ack"));
    assert!(expected.finished, "{name}");
    let (recovered, _) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
        .unwrap_or_else(|e| panic!("reopening {name} refused: {e}"));
    holds(&recovered, &expected, vectors, None, Shape::Raw)
        .unwrap_or_else(|e| panic!("after {name}: {e}"));
    Some(format!("{name}: the lock was released with the process"))
}

/// The matrix.
///
/// The count this asserts is the count of points the script actually
/// reaches. Some points are unreachable by construction, being the second
/// append of an `add` of one record, the point between a backup's two
/// renames where its directory does not exist yet, the half record of a
/// step the failing arm fails, whose sync is the sync that arm fails, and
/// the points after the commit of such a step, and the parent says so
/// rather than failing on them.
#[test]
fn the_kill_matrix_holds_at_every_point() {
    let full = std::env::var(ENV_MATRIX).as_deref() == Ok("full");
    let work = Work::new();
    let vectors = corpus();
    let cases = cases(full);
    let mut reached = 0usize;
    let mut unreachable: Vec<String> = Vec::new();
    let mut lines: Vec<String> = Vec::new();

    for case in &cases {
        let name = case_name(case);
        if case.fixture.is_some() {
            lines.extend(upgrade_case(&work.0, case, &name));
            reached += 1;
            continue;
        }
        if case.repair.is_some() {
            lines.extend(repair_case(&work.0, case, &name, &vectors));
            reached += 1;
            continue;
        }
        if case.point == KillPoint::LockHeld {
            lines.extend(lock_case(&work.0, &name, &vectors));
            reached += 1;
            continue;
        }
        let dir = work.0.join(format!("{name}.zdb"));
        let marker = work.0.join(format!("{name}.marker"));
        let in_flight = Some(&SCRIPT[case.step]);

        // The child, armed at this point.
        let status = child_command(&dir, case.arm, case.shape)
            .env(ENV_TARGET, format!("{}:{}", case.step, case.point.name()))
            .env(ENV_MARKER, &marker)
            .env_remove(ENV_RESUME)
            .output()
            .expect("the child runs");
        if !marker.exists() {
            // The point is not on the path this step takes.
            assert!(
                status.status.success(),
                "the child failed at {name} without reaching the point:\n{}",
                String::from_utf8_lossy(&status.stderr)
            );
            unreachable.push(name);
            continue;
        }
        reached += 1;
        assert!(
            !status.status.success(),
            "the child reached {name} and did not abort"
        );

        // What the child had acknowledged, and what opening the directory
        // again gives back.
        let expected = acknowledged(&dir.with_extension("ack"));
        let (recovered, report) =
            Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
                .unwrap_or_else(|e| panic!("recovering {name} refused: {e}"));
        let unacknowledged = holds(&recovered, &expected, &vectors, in_flight, case.shape)
            .unwrap_or_else(|e| panic!("after {name}: {e}"));
        assert!(
            report
                .damage
                .as_ref()
                .is_none_or(|damage| damage.kind == "torn_tail"),
            "recovering {name} found damage no crash produces: {:?}",
            report.damage
        );
        // A save killed between its two renames left the whole index
        // beside the target and nothing at it, so opening it is the rename
        // back and then the load. Nothing else reaches that.
        assert_eq!(
            report.restored_from_aside,
            case.point == KillPoint::SaveBetweenRenames,
            "{name} put the index back from beside the target"
        );
        let backup = backup_holds(&dir, &expected, &vectors, in_flight, case.shape)
            .unwrap_or_else(|e| panic!("the backup after {name}: {e}"));
        lines.push(format!(
            "{name}: {} acknowledged, {} refused, {unacknowledged} unacknowledged present, \
             replayed {} withdrawn {} skipped {} segments {} aside {} checkpoint {:?} damage {} \
             backup {backup}",
            expected.present.len(),
            expected.refused.len(),
            report.replayed,
            report.withdrawn,
            report.skipped,
            report.segments,
            report.restored_from_aside,
            report.checkpoint_taken,
            damage_of(&report)
        ));
        // The journal is open for append on this collection, and the
        // resuming child opens it too, so let it go first.
        drop(recovered);

        // The script resumed from the recovered state, to the end.
        let status = child_command(&dir, case.arm, case.shape)
            .env(ENV_TARGET, "")
            .env(ENV_RESUME, "1")
            .env_remove(ENV_MARKER)
            .output()
            .expect("the child resumes");
        assert!(
            status.status.success(),
            "resuming {name} failed:\n{}",
            String::from_utf8_lossy(&status.stderr)
        );
        let expected = acknowledged(&dir.with_extension("ack"));
        assert!(expected.finished, "resuming {name} did not reach the end");
        assert_eq!(
            expected.present.len(),
            HIGHEST_RECORD - expected.refused.len() + case.shape.base(),
            "the finished script holds every record but the one it removes and those refused"
        );
        let (recovered, _) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
            .unwrap_or_else(|e| panic!("reopening {name} after the resume refused: {e}"));
        holds(&recovered, &expected, &vectors, None, case.shape)
            .unwrap_or_else(|e| panic!("after resuming {name}: {e}"));
        drop(recovered);
        let finished = backup_holds(&dir, &expected, &vectors, None, case.shape)
            .unwrap_or_else(|e| panic!("the backup after resuming {name}: {e}"));
        assert_eq!(finished, "as it was", "{name}: the last backup stands");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(backup_target(&dir));
        let _ = std::fs::remove_file(dir.with_extension("ack"));
        let _ = std::fs::remove_dir_all(crate::journal::journal_path(&dir).unwrap());
    }

    for line in &lines {
        println!("{line}");
    }
    println!(
        "{} cases reached, {} unreachable {:?}",
        reached,
        unreachable.len(),
        unreachable
    );
    assert!(reached > 0, "no point was reached");
    assert_eq!(
        reached + unreachable.len(),
        cases.len(),
        "every case was either reached or named unreachable"
    );
    let named: BTreeSet<&str> = cases.iter().map(|case| case.point.name()).collect();
    assert_eq!(
        named.len(),
        KillPoint::ALL.len(),
        "every point the engine declares has a case"
    );
    let reached_points: BTreeSet<&str> = cases
        .iter()
        .filter(|case| {
            case.fixture.is_some()
                || case.repair.is_some()
                || !unreachable.contains(&case_name(case))
        })
        .map(|case| case.point.name())
        .collect();
    assert_eq!(
        reached_points.len(),
        KillPoint::ALL.len(),
        "every point the engine declares is reached at least once"
    );
}
