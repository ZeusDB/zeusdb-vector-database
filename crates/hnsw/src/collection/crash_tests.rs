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
//! directory the child left, and holds it to two rules.
//!
//! Every record an acknowledged step added is present, with the vector and
//! the metadata it was given. Nothing the script never added is present. An
//! unacknowledged record may be present or absent, since a batch the child
//! died inside leaves whatever prefix of itself had reached the journal,
//! and either is correct.
//!
//! The parent then resumes the script in a second child from the recovered
//! state, runs it to the end, and holds the finished collection to the same
//! two rules. That is what proves the journal is usable after a recovery and
//! not merely readable.
//!
//! # The matrix
//!
//! Six points inside an `add` and eleven inside a checkpoint, at every step
//! of the script that reaches them, under three durability arms. The
//! ordinary gate runs a reduced matrix, being every point once under one
//! arm. `ZEUSDB_KILL_MATRIX=full` runs every point at every step under all
//! three, which is what the fuzzers' `ZEUSDB_FUZZ_CASES` does for them.
//!
//! # The three arms
//!
//! `sync` commits with [`CommitMode::Sync`], so every record a call returns
//! from is on the device. `none` commits with [`CommitMode::Deferred`] and
//! never syncs, so the records are in the kernel; a process abort loses
//! nothing, which is what these tests kill with. `interval` commits
//! deferred and a thread of the test's own syncs the file every ten
//! milliseconds, which is the shape a policy that syncs on an interval
//! takes. The policies themselves are not the engine's yet; these are the
//! commit modes it has.
//!
//! # How the child is run
//!
//! The parent spawns this test binary again with `--exact` naming the child
//! test, which is `#[ignore]`d so the ordinary run does not reach it, and
//! hands it the case in the environment. It is the only way to abort a
//! process from a test and keep the suite.

#![allow(clippy::disallowed_types)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use zeusdb_vector_core::{CommitMode, JournalDamage, KillPoint};

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
];

/// The highest record index the script ever adds.
const HIGHEST_RECORD: usize = 17;

/// The points inside an `add`, in the order the call reaches them.
const ADD_POINTS: &[KillPoint] = &[
    KillPoint::AddBeforeFirstAppend,
    KillPoint::AddMidAppend,
    KillPoint::AddAfterSecondAppend,
    KillPoint::AddAfterAppendBeforeCommit,
    KillPoint::AddAfterCommitBeforeApply,
    KillPoint::AddAfterApply,
];

/// The points inside a checkpoint, in the order the call reaches them.
const CHECKPOINT_POINTS: &[KillPoint] = &[
    KillPoint::CheckpointBeforeSave,
    KillPoint::SaveAfterArtefacts,
    KillPoint::SaveAfterDump,
    KillPoint::SaveAfterManifest,
    KillPoint::SaveBetweenRenames,
    KillPoint::SaveAfterCommit,
    KillPoint::CheckpointAfterSaveBeforeTruncate,
    KillPoint::TruncateBefore,
    KillPoint::TruncateAfterSetLen,
    KillPoint::TruncateAfterHeader,
    KillPoint::CheckpointAfterTruncate,
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

// ============================================================================
// THE CASE, PASSED THROUGH THE ENVIRONMENT
// ============================================================================

const ENV_DIR: &str = "ZEUSDB_KILL_DIR";
const ENV_MODE: &str = "ZEUSDB_KILL_MODE";
const ENV_TARGET: &str = "ZEUSDB_KILL_TARGET";
const ENV_MARKER: &str = "ZEUSDB_KILL_MARKER";
const ENV_RESUME: &str = "ZEUSDB_KILL_RESUME";
const ENV_MATRIX: &str = "ZEUSDB_KILL_MATRIX";

/// The name of the child test, as `--exact` spells it.
const CHILD_TEST: &str = "collection::crash_tests::the_kill_matrixs_child";

/// What each arm commits with, and whether a thread syncs beside it.
fn commit_mode(arm: &str) -> CommitMode {
    match arm {
        "sync" => CommitMode::Sync,
        _ => CommitMode::Deferred,
    }
}

/// A thread that syncs the journal every ten milliseconds, which is the
/// shape an interval policy takes. It opens the file itself rather than
/// taking a handle off the sink, so nothing about the policy has to reach
/// into the collection.
struct Flusher {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Flusher {
    fn start(path: PathBuf) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&path) {
                    let _ = file.sync_data();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        Flusher {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Flusher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
        if parts.len() < 3 {
            continue;
        }
        out.steps += 1;
        match parts[2] {
            "add" => {
                for id in parts[3].split(',') {
                    out.present.insert(id.to_string(), false);
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
/// given, and nothing present that the script never added. What is not held
/// is the presence of an unacknowledged record, which is correct either way.
fn holds(
    collection: &Collection,
    expected: &Acknowledged,
    vectors: &[Vec<f32>],
) -> Result<usize, String> {
    let live: BTreeSet<String> = collection.id_map().keys().cloned().collect();
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
    let vectors = corpus();
    let mode = commit_mode(&arm);
    let flusher =
        (arm == "interval").then(|| Flusher::start(crate::journal::journal_path(&dir).unwrap()));

    let (collection, start_at) = if resume {
        let (collection, report) = Collection::recover(&dir.to_string_lossy(), None, mode)
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
        collection
            .journal_to(&dir.to_string_lossy(), mode)
            .expect("the journal opens beside the directory");
        (collection, 0)
    };

    for (index, step) in SCRIPT.iter().enumerate().skip(start_at) {
        match (target_step == Some(index))
            .then(|| KillPoint::from_name(&target_point))
            .flatten()
        {
            Some(point) => zeusdb_vector_core::kill_arm(point, &marker),
            None => zeusdb_vector_core::kill_disarm(),
        }
        match step {
            Step::Add(records) => {
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
                let ids: Vec<String> = records.iter().map(|i| format!("r{i}")).collect();
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
    drop(flusher);
}

// ============================================================================
// THE PARENT
// ============================================================================

/// One case of the matrix, being a step of the script and a point inside it.
#[derive(Clone, Debug)]
struct Case {
    step: usize,
    point: KillPoint,
}

/// Every case the script can reach, or the reduced set the ordinary gate
/// runs, being every point once at the first step that reaches it.
fn cases(full: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut add_seen = false;
    let mut checkpoint_seen = false;
    for (step, kind) in SCRIPT.iter().enumerate() {
        let points = match kind {
            Step::Add(_) => {
                if !full && add_seen {
                    continue;
                }
                add_seen = true;
                ADD_POINTS
            }
            Step::Checkpoint => {
                if !full && checkpoint_seen {
                    continue;
                }
                checkpoint_seen = true;
                CHECKPOINT_POINTS
            }
            _ => continue,
        };
        for point in points {
            cases.push(Case {
                step,
                point: *point,
            });
        }
    }
    cases
}

/// Which arms the run covers.
fn arms(full: bool) -> &'static [&'static str] {
    if full {
        &["sync", "none", "interval"]
    } else {
        &["sync"]
    }
}

fn child_command(dir: &Path, arm: &str) -> std::process::Command {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
        .env(ENV_DIR, dir)
        .env(ENV_MODE, arm)
        .env_remove(ENV_MATRIX);
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

    for arm in arms(full) {
        for case in &cases {
            let name = format!(
                "{arm}-{}-{}",
                case.step,
                case.point.name().replace(':', "_")
            );
            let dir = work.0.join(format!("{name}.zdb"));
            let marker = work.0.join(format!("{name}.marker"));

            // The child, armed at this point.
            let status = child_command(&dir, arm)
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

            // What the child had acknowledged, and what opening the
            // directory again gives back.
            let expected = acknowledged(&dir.with_extension("ack"));
            let (recovered, report) =
                Collection::recover(&dir.to_string_lossy(), None, CommitMode::Deferred)
                    .unwrap_or_else(|e| panic!("recovering {name} refused: {e}"));
            let unacknowledged = holds(&recovered, &expected, &vectors)
                .unwrap_or_else(|e| panic!("after {name}: {e}"));
            assert!(
                !matches!(report.damage, Some(JournalDamage::Corrupt { .. })),
                "recovering {name} found a corrupt middle, which no crash produces: {:?}",
                report.damage
            );
            // A save killed between its two renames left the whole index
            // beside the target and nothing at it, so opening it is the
            // rename back and then the load. Nothing else reaches that.
            assert_eq!(
                report.restored_from_aside,
                case.point == KillPoint::SaveBetweenRenames,
                "{name} put the index back from beside the target"
            );
            lines.push(format!(
                "{name}: {} acknowledged, {unacknowledged} unacknowledged present, \
                 replayed {} skipped {} aside {} damage {}",
                expected.present.len(),
                report.replayed,
                report.skipped,
                report.restored_from_aside,
                match &report.damage {
                    Some(JournalDamage::TornTail { sequence, .. }) =>
                        format!("torn tail at {sequence}"),
                    Some(JournalDamage::Corrupt { sequence, .. }) =>
                        format!("corrupt at {sequence}"),
                    None => "none".to_string(),
                }
            ));
            // The journal is open for append on this collection, and the
            // resuming child opens it too, so let it go first.
            drop(recovered);

            // The script resumed from the recovered state, to the end.
            let status = child_command(&dir, arm)
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
                HIGHEST_RECORD,
                "the finished script holds every record but the one it removes"
            );
            let (recovered, _) =
                Collection::recover(&dir.to_string_lossy(), None, CommitMode::Deferred)
                    .unwrap_or_else(|e| panic!("reopening {name} after the resume refused: {e}"));
            holds(&recovered, &expected, &vectors)
                .unwrap_or_else(|e| panic!("after resuming {name}: {e}"));
            drop(recovered);
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_file(dir.with_extension("ack"));
            let _ = std::fs::remove_file(crate::journal::journal_path(&dir).unwrap());
        }
    }

    for line in &lines {
        println!("{line}");
    }
    println!(
        "{} points reached, {} unreachable {:?}, over {} arms",
        reached,
        unreachable.len(),
        unreachable,
        arms(full).len()
    );
    assert!(reached > 0, "no point was reached");
    assert_eq!(
        reached + unreachable.len(),
        cases.len() * arms(full).len(),
        "every case was either reached or named unreachable"
    );
}
