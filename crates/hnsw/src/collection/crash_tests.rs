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
//! directory the child left, and holds it to three rules.
//!
//! Every record an acknowledged step added is present, with the vector and
//! the metadata it was given. Nothing the script never added is present. No
//! record of a call that returned refused is present. An unacknowledged
//! record may be present or absent, since a batch the child died inside
//! leaves whatever prefix of itself had reached the journal, and either is
//! correct.
//!
//! The parent then resumes the script in a second child from the recovered
//! state, runs it to the end, and holds the finished collection to the same
//! rules. That is what proves the journal is usable after a recovery and not
//! merely readable.
//!
//! # The matrix
//!
//! Twenty-seven points: ten inside an `add`, three of them reached only
//! where its commit fails, two in a roll by size, ten inside a checkpoint,
//! four inside the upgrade of a journal of format version 1, and one with
//! the lock held. The script's segments roll at a length set so its last
//! step passes it and no earlier step does. The ordinary gate runs a
//! reduced matrix, being every point once: the add, roll and checkpoint
//! points under the per-call arm, the three failing points under the
//! failing arm, the lock, and the upgrade over one fixture.
//! `ZEUSDB_KILL_MATRIX=full` runs every point at every step under every arm
//! and the upgrade over every fixture.
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
//! the per-call policy through a journal directory whose sync fails at one
//! step's commit, so that step returns its records refused, and the child
//! then takes the checkpoint the fault requires before it goes on.
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
    InsertParts, JournalDamage, KillPoint, LockFailure, LockMode, Locked, Operation, Storage,
    JOURNAL_RECORD_OVERHEAD_BYTES, SEGMENT_HEADER_BYTES,
};

use crate::journal::Durability;

use super::journal_support::{copy_fixture, holds_fixture, manifest, sink_over, FaultyStorage};
use super::{Collection, Declaration, ParsedRecord};

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
    /// Remove these records.
    Remove(&'static [usize]),
    /// Replace this record's metadata.
    Update(usize),
    /// Save, which on a journaled collection is a checkpoint.
    Checkpoint,
}

/// Every step, in order. Every record index is added exactly once, so a
/// record present that no acknowledged step added is a defect rather than a
/// duplicate.
const SCRIPT: &[Step] = &[
    Step::Add(&[0, 1, 2]),
    Step::Add(&[3, 4, 5, 6]),
    Step::Checkpoint,
    Step::Add(&[7]),
    Step::Remove(&[1]),
    Step::Add(&[8, 9, 10, 11, 12]),
    Step::Update(2),
    Step::Checkpoint,
    Step::Add(&[13, 14]),
    Step::Add(&[15, 16, 17]),
    Step::Add(&[18, 19, 20, 21, 22, 23, 24, 25]),
];

/// The highest record index the script ever adds.
const HIGHEST_RECORD: usize = 25;

/// The step whose commit the failing arm fails.
const FAILING_STEP: usize = 5;

/// The step that passes the segment's roll length.
const ROLL_STEP: usize = 10;

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

/// The points inside the upgrade of a journal of format version 1.
const UPGRADE_POINTS: &[KillPoint] = &[
    KillPoint::UpgradeAfterReplay,
    KillPoint::UpgradeAfterCommit,
    KillPoint::UpgradeAfterSegment,
    KillPoint::UpgradeAfterOldRemoved,
];

/// The corpus, drawn the same way in every process, so the parent knows what
/// vector each record was given.
fn corpus() -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(HIGHEST_RECORD + 1);
    // A generator of its own rather than the engine's, because the parent
    // and the child have to agree on the bytes and nothing else depends on
    // them. One multiplicative step per value.
    let mut state: u64 = 0x5eed_0148;
    for _ in 0..=HIGHEST_RECORD {
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

fn parsed(i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vectors[i].clone(),
        sparse: None,
        metadata: metadata_for(i).into_iter().collect(),
    }
}

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 8, 50, 200, vec![]).unwrap()
}

/// The length the child's segments roll at: ten of the script's inserts and
/// two commits past a segment's header. The first segment takes seven
/// inserts and the second six with a removal and a replacement, both short
/// of it, and the third takes thirteen by the end of the last step, so the
/// last step's commit rolls and no earlier commit does.
fn roll_length(vectors: &[Vec<f32>]) -> u64 {
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
    SEGMENT_HEADER_BYTES as u64 + 10 * insert + 2 * commit
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

/// What the acknowledgements say the collection must hold.
#[derive(Debug, Default)]
struct Acknowledged {
    /// Every record an acknowledged step added and no acknowledged step
    /// removed, against whether its metadata was replaced.
    present: BTreeMap<String, bool>,
    /// Every record of a call that returned refused.
    refused: BTreeSet<String>,
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
        let parts: Vec<&str> = line.split(' ').collect();
        if parts.len() < 3 || parts[0] != "step" {
            continue;
        }
        out.steps += 1;
        match parts[2] {
            "add" => {
                for id in parts[3].split(',') {
                    out.present.insert(id.to_string(), false);
                }
            }
            "refused" => {
                for id in parts[3].split(',') {
                    out.refused.insert(id.to_string());
                }
            }
            "remove" => {
                for id in parts[3].split(',') {
                    out.present.remove(id);
                }
            }
            "update" => {
                out.present.insert(parts[3].to_string(), true);
            }
            _ => {}
        }
    }
    out
}

/// Hold a recovered collection to the acknowledgements.
///
/// Every acknowledged record present with the vector and the metadata it was
/// given, nothing present that the script never added, and no record of a
/// call that returned refused present. What is not held is the presence of
/// an unacknowledged record, which is correct either way.
fn holds(
    collection: &Collection,
    expected: &Acknowledged,
    vectors: &[Vec<f32>],
) -> Result<usize, String> {
    let live: BTreeSet<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    for (id, updated) in &expected.present {
        if !live.contains(id) {
            return Err(format!("acknowledged record {id} is missing"));
        }
        let view = collection
            .records(vec![id.clone()], true, true)
            .map_err(|e| e.to_string())?;
        let view = view
            .first()
            .ok_or_else(|| format!("{id} came back empty"))?;
        let i: usize = id[1..].parse().unwrap();
        if view.vector.as_deref() != Some(&vectors[i][..]) {
            return Err(format!("record {id} came back with a different vector"));
        }
        let wanted = if *updated {
            updated_metadata(i)
        } else {
            metadata_for(i)
        };
        for (key, value) in &wanted {
            if view.metadata.get(key) != Some(value) {
                return Err(format!(
                    "record {id} has {key} of {:?} where {value:?} was acknowledged",
                    view.metadata.get(key)
                ));
            }
        }
    }
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
        if i > HIGHEST_RECORD {
            return Err(format!("record {id} was never in the script"));
        }
    }
    Ok(live.len() - expected.present.len())
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
    let vectors = corpus();
    let durability = durability(&arm);
    zeusdb_vector_core::set_segment_roll_bytes(roll_length(&vectors));

    // An upgrade case loads a copy of a fixture, armed inside its upgrade.
    if std::env::var(ENV_UPGRADE).is_ok() {
        if let Some(point) = point {
            zeusdb_vector_core::kill_arm(point, &marker);
        }
        let opened = Collection::recover(&dir.to_string_lossy(), None, Durability::PerCall);
        zeusdb_vector_core::kill_disarm();
        opened.expect("the fixture opens");
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
        let collection = Collection::build(declaration(), None);
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
        (collection, 0)
    };

    for (index, step) in SCRIPT.iter().enumerate().skip(start_at) {
        match (target_step == Some(index)).then_some(point).flatten() {
            Some(point) => zeusdb_vector_core::kill_arm(point, &marker),
            None => zeusdb_vector_core::kill_disarm(),
        }
        match step {
            Step::Add(records) => {
                let ids: Vec<String> = records.iter().map(|i| format!("r{i}")).collect();
                // The failing arm's step: the commit's sync fails, the call
                // returns every record refused, and the fault it leaves is
                // cleared by the checkpoint the caller then takes.
                if let (Some(faults), FAILING_STEP) = (&faults, index) {
                    faults.syncs.arm(0, 1);
                    let batch: Vec<ParsedRecord> =
                        records.iter().map(|&i| parsed(i, &vectors)).collect();
                    let added = collection.add_records(batch, vec![], false);
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
                // records already durable and replayed, which the engine
                // refuses as duplicates exactly as it would any retry, so a
                // caller resuming skips what is present.
                let batch: Vec<ParsedRecord> = records
                    .iter()
                    .filter(|&&i| !collection.contains(&format!("r{i}")))
                    .map(|&i| parsed(i, &vectors))
                    .collect();
                if !batch.is_empty() {
                    let added = collection.add_records(batch, vec![], false);
                    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
                }
                acknowledge(
                    &acknowledgements,
                    &format!("step {index} add {}", ids.join(",")),
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
        }
    }
    zeusdb_vector_core::kill_disarm();
    acknowledge(&acknowledgements, "done");
}

// ============================================================================
// THE PARENT
// ============================================================================

/// One case of the matrix, being an arm, a step of the script and a point
/// inside it, or a fixture and a point inside its upgrade.
#[derive(Clone, Debug)]
struct Case {
    arm: &'static str,
    step: usize,
    point: KillPoint,
    fixture: Option<&'static str>,
}

/// Every case the script can reach, or the reduced set the ordinary gate
/// runs, being every point once.
fn cases(full: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    let arms: &[&'static str] = if full {
        &["sync", "none", "interval"]
    } else {
        &["sync"]
    };
    for &arm in arms {
        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for (step, kind) in SCRIPT.iter().enumerate() {
            let mut points: Vec<KillPoint> = match kind {
                Step::Add(_) => ADD_POINTS.to_vec(),
                Step::Checkpoint => CHECKPOINT_POINTS.to_vec(),
                _ => continue,
            };
            if step == ROLL_STEP {
                points.extend_from_slice(ROLL_POINTS);
            }
            for point in points {
                if full || seen.insert(point.name()) {
                    cases.push(Case {
                        arm,
                        step,
                        point,
                        fixture: None,
                    });
                }
            }
        }
    }
    // The failing arm: the three failing points at the failing step, and
    // in the full matrix every point of every step after it.
    for point in FAILING_POINTS {
        cases.push(Case {
            arm: "fail",
            step: FAILING_STEP,
            point: *point,
            fixture: None,
        });
    }
    if full {
        for (step, kind) in SCRIPT.iter().enumerate().skip(FAILING_STEP + 1) {
            let mut points: Vec<KillPoint> = match kind {
                Step::Add(_) => ADD_POINTS.to_vec(),
                Step::Checkpoint => CHECKPOINT_POINTS.to_vec(),
                _ => continue,
            };
            if step == ROLL_STEP {
                points.extend_from_slice(ROLL_POINTS);
            }
            for point in points {
                cases.push(Case {
                    arm: "fail",
                    step,
                    point,
                    fixture: None,
                });
            }
        }
    }
    cases.push(Case {
        arm: "sync",
        step: 0,
        point: KillPoint::LockHeld,
        fixture: None,
    });
    let fixtures: &[&'static str] = if full {
        &super::journal_support::FIXTURES
    } else {
        &super::journal_support::FIXTURES[..1]
    };
    for fixture in fixtures {
        for point in UPGRADE_POINTS {
            cases.push(Case {
                arm: "upgrade",
                step: 0,
                point: *point,
                fixture: Some(fixture),
            });
        }
    }
    cases
}

fn child_command(dir: &Path, arm: &str) -> std::process::Command {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
        .env(ENV_DIR, dir)
        .env(ENV_MODE, arm)
        .env_remove(ENV_MATRIX)
        .env_remove(ENV_UPGRADE);
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
        Some(JournalDamage::TornTail { sequence, .. }) => format!("torn tail at {sequence}"),
        Some(JournalDamage::Corrupt { sequence, .. }) => format!("corrupt at {sequence}"),
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
    let status = child_command(&dir, "upgrade")
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

/// The lock case: the child aborts holding the lock, before anything is
/// written, so the directory does not exist, the lock is free again, and
/// the script run from the start in a new child reaches its end.
fn lock_case(work: &Path, name: &str, vectors: &[Vec<f32>]) -> Option<String> {
    let dir = work.join(format!("{name}.zdb"));
    let marker = work.join(format!("{name}.marker"));
    let status = child_command(&dir, "sync")
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
    let status = child_command(&dir, "sync")
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
    holds(&recovered, &expected, vectors).unwrap_or_else(|e| panic!("after {name}: {e}"));
    Some(format!("{name}: the lock was released with the process"))
}

/// The matrix.
///
/// The count this asserts is the count of points the script actually
/// reaches. One point is unreachable by construction, being the second
/// append of an `add` of one record, and the parent says so rather than
/// failing on it.
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
        let name = match case.fixture {
            Some(fixture) => format!("upgrade-{fixture}-{}", case.point.name().replace(':', "_")),
            None => format!(
                "{}-{}-{}",
                case.arm,
                case.step,
                case.point.name().replace(':', "_")
            ),
        };
        if case.fixture.is_some() {
            lines.extend(upgrade_case(&work.0, case, &name));
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

        // The child, armed at this point.
        let status = child_command(&dir, case.arm)
            .env(ENV_TARGET, format!("{}:{}", case.step, case.point.name()))
            .env(ENV_MARKER, &marker)
            .env_remove(ENV_RESUME)
            .output()
            .expect("the child runs");
        if !marker.exists() {
            // The point is not on the path this step takes, which the
            // one-record `add` makes true of the second append.
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
        let unacknowledged =
            holds(&recovered, &expected, &vectors).unwrap_or_else(|e| panic!("after {name}: {e}"));
        assert!(
            !matches!(report.damage, Some(JournalDamage::Corrupt { .. })),
            "recovering {name} found a corrupt middle, which no crash produces: {:?}",
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
        lines.push(format!(
            "{name}: {} acknowledged, {} refused, {unacknowledged} unacknowledged present, \
             replayed {} withdrawn {} skipped {} segments {} aside {} checkpointed {} damage {}",
            expected.present.len(),
            expected.refused.len(),
            report.replayed,
            report.withdrawn,
            report.skipped,
            report.segments,
            report.restored_from_aside,
            report.checkpointed,
            damage_of(&report)
        ));
        // The journal is open for append on this collection, and the
        // resuming child opens it too, so let it go first.
        drop(recovered);

        // The script resumed from the recovered state, to the end.
        let status = child_command(&dir, case.arm)
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
            HIGHEST_RECORD - expected.refused.len(),
            "the finished script holds every record but the one it removes and those refused"
        );
        let (recovered, _) = Collection::recover(&dir.to_string_lossy(), None, Durability::NoSync)
            .unwrap_or_else(|e| panic!("reopening {name} after the resume refused: {e}"));
        holds(&recovered, &expected, &vectors)
            .unwrap_or_else(|e| panic!("after resuming {name}: {e}"));
        drop(recovered);
        let _ = std::fs::remove_dir_all(&dir);
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
}
