//! The lock that keeps a second writer off a collection.
//!
//! `<name>.zdblock` beside the directory, taken without waiting. A journaled
//! collection holds it exclusive for its life, a save for the save, and a
//! load holds it shared while it reads. These tests hold it to that in one
//! process, where a second handle on the file is a second holder, and across
//! processes, through a child this test binary runs, which holds the lock or
//! asks for it.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::json;
use zeusdb_vector_core::{Error, FsStorage, LockMode, Locked, Storage};

use super::journal_support::TempDir;
use super::{Collection, Declaration, ParsedRecord};
use crate::journal::Durability;

fn declaration() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap()
}

fn record(id: &str) -> ParsedRecord {
    ParsedRecord {
        id: id.to_string(),
        vector: vec![0.5, 1.5],
        sparse: None,
        metadata: HashMap::from([("from".to_string(), json!(id))]),
    }
}

fn add(collection: &Collection, id: &str) {
    let added = collection.add_records(vec![record(id)], vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn locked<T>(result: Result<T, Error>, what: &str) -> String {
    match result {
        Err(Error::CollectionLocked { path, holder }) => {
            assert!(!path.is_empty(), "{what}");
            holder
        }
        Err(other) => panic!("{what}: expected the lock's refusal, got {other}"),
        Ok(_) => panic!("{what}: expected the lock's refusal, and it opened"),
    }
}

// ============================================================================
// IN ONE PROCESS
// ============================================================================

/// A journaled collection holds its directory for its life. A second
/// collection in the same process is refused whatever it asks: a load, the
/// checkpoint alone, a save into the directory and a journal opened there.
/// Dropping the first lets the directory go.
#[test]
fn a_journaled_collection_holds_its_directory_against_a_second_handle() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("held.zdb");
    let first = Collection::build(declaration(), None);
    first
        .journal_to(path.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    add(&first, "a");
    let dir = path.to_str().unwrap();

    locked(Collection::load(dir), "a load");
    locked(
        Collection::load_checkpoint_only(dir, None),
        "the checkpoint alone",
    );
    let second = Collection::build(declaration(), None);
    add(&second, "b");
    locked(second.save(dir), "a save into the directory");
    locked(
        second.journal_to(dir, Durability::PerCall),
        "a journal there",
    );
    let holder = locked(Collection::load(dir), "a load");
    if cfg!(windows) {
        assert!(
            holder.contains("does not let a second handle read"),
            "{holder}"
        );
    } else {
        assert!(
            holder.contains(&format!("process {}", std::process::id())),
            "{holder}"
        );
        assert!(holder.contains("mode exclusive"), "{holder}");
    }
    // The first collection still takes records and checkpoints.
    add(&first, "c");
    first.checkpoint().unwrap();
    drop(first);

    let (reopened, _) = Collection::recover(dir, None, Durability::PerCall).unwrap();
    assert!(reopened.contains("a") && reopened.contains("c"));
    assert!(!reopened.contains("b"));
}

/// Shared holders sit together and refuse an exclusive one: a load beside a
/// shared holder opens, and a save beside it is refused until it lets go.
#[test]
fn shared_holders_sit_together_and_refuse_a_save() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("shared.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(path.to_str().unwrap()).unwrap();

    let storage = FsStorage::at(&path).unwrap();
    let shared = match storage.lock(LockMode::Shared, "") {
        Ok(Locked::Writable(guard)) => guard,
        other => panic!("expected a shared lock, got {other:?}"),
    };
    assert_eq!(shared.mode(), LockMode::Shared);
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 1, "a load beside a shared holder opens");
    locked(
        collection.save(path.to_str().unwrap()),
        "a save beside a shared holder",
    );
    drop(shared);
    collection.save(path.to_str().unwrap()).unwrap();
}

/// A load holds the lock while it reads and lets it go when it returns, so
/// two loads one after the other and a save after them all go through.
#[test]
fn a_load_lets_the_lock_go_when_it_returns() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("released.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(path.to_str().unwrap()).unwrap();
    let one = Collection::load(path.to_str().unwrap()).unwrap();
    let two = Collection::load(path.to_str().unwrap()).unwrap();
    two.save(path.to_str().unwrap()).unwrap();
    assert_eq!(one.len(), 1);
    assert!(temp.at("released.zdb.zdblock").is_file(), "the file stays");
}

// ============================================================================
// ACROSS PROCESSES
// ============================================================================

const ENV_DIR: &str = "ZEUSDB_LOCK_DIR";
const ENV_ROLE: &str = "ZEUSDB_LOCK_ROLE";
const CHILD_TEST: &str = "collection::lock_tests::the_lock_child";

/// The child, run in a process of its own. Under `hold` it journals the
/// directory, adds a record, says it is ready and waits until it is told to
/// go or is killed. Under `load` it loads the directory and writes what the
/// load said.
#[test]
#[ignore = "the parent runs this in a process of its own"]
fn the_lock_child() {
    let Ok(dir) = std::env::var(ENV_DIR) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let role = std::env::var(ENV_ROLE).unwrap_or_default();
    let said = dir.with_extension("said");
    match role.as_str() {
        "hold" => {
            let collection = Collection::build(declaration(), None);
            collection
                .journal_to(dir.to_str().unwrap(), Durability::PerCall)
                .unwrap();
            add(&collection, "child");
            std::fs::write(&said, "ready").unwrap();
            let go = dir.with_extension("go");
            let start = Instant::now();
            while !go.exists() && start.elapsed() < Duration::from_secs(120) {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        "load" => {
            let outcome = match Collection::load(dir.to_str().unwrap()) {
                Ok(collection) => format!("opened {}", collection.len()),
                Err(Error::CollectionLocked { .. }) => "locked".to_string(),
                Err(other) => format!("failed {other}"),
            };
            std::fs::write(&said, outcome).unwrap();
        }
        _ => {}
    }
}

fn child(dir: &Path, role: &str) -> std::process::Child {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
        .env(ENV_DIR, dir)
        .env(ENV_ROLE, role)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the child runs")
}

fn wait_for_word(file: &Path, word: &str) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(60) {
        if std::fs::read_to_string(file).is_ok_and(|text| text.contains(word)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the child never wrote {word} to {}", file.display());
}

/// Another process holding the journal refuses this one's load at once,
/// and the load opens once it lets go, holding what it recorded.
#[test]
fn another_process_holding_the_journal_refuses_a_load() {
    let temp = TempDir::new("lock-tests");
    let dir = temp.at("theirs.zdb");
    let mut holder = child(&dir, "hold");
    wait_for_word(&dir.with_extension("said"), "ready");

    locked(
        Collection::load(dir.to_str().unwrap()),
        "a load in another process",
    );
    let other = Collection::build(declaration(), None);
    locked(
        other.save(dir.to_str().unwrap()),
        "a save in another process",
    );

    std::fs::write(dir.with_extension("go"), "").unwrap();
    assert!(holder.wait().unwrap().success());
    let (opened, _) =
        Collection::recover(dir.to_str().unwrap(), None, Durability::PerCall).unwrap();
    assert!(opened.contains("child"));
}

/// A holder killed without warning releases its lock with its process, and
/// the next load acquires it and recovers what the holder acknowledged.
#[test]
fn a_killed_holder_releases_the_lock() {
    let temp = TempDir::new("lock-tests");
    let dir = temp.at("killed.zdb");
    let mut holder = child(&dir, "hold");
    wait_for_word(&dir.with_extension("said"), "ready");
    holder.kill().unwrap();
    let _ = holder.wait();
    // Windows may release the lock a few milliseconds after the process has
    // exited, so the load is tried again for a bounded time.
    let start = Instant::now();
    let opened = loop {
        match Collection::recover(dir.to_str().unwrap(), None, Durability::PerCall) {
            Ok((opened, _)) => break opened,
            Err(Error::CollectionLocked { .. }) if start.elapsed() < Duration::from_secs(10) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Err(other) => panic!("the load after the kill refused: {other}"),
        }
    };
    assert!(
        opened.contains("child"),
        "the acknowledged record survived the kill"
    );
}

/// Two writers on one directory: this process holds it journaled, and
/// another process's load is refused rather than both appending at one
/// offset of one segment. Once this one lets go, the other opens it holding
/// both of this one's records.
#[test]
fn a_second_writer_in_another_process_is_refused_at_its_load() {
    let temp = TempDir::new("lock-tests");
    let dir = temp.at("ours.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(dir.to_str().unwrap(), Durability::PerCall)
        .unwrap();
    add(&collection, "base");
    let said = dir.with_extension("said");

    let mut second = child(&dir, "load");
    assert!(second.wait().unwrap().success());
    assert_eq!(std::fs::read_to_string(&said).unwrap(), "locked");

    add(&collection, "ours");
    drop(collection);
    let mut third = child(&dir, "load");
    assert!(third.wait().unwrap().success());
    assert_eq!(std::fs::read_to_string(&said).unwrap(), "opened 2");
}
