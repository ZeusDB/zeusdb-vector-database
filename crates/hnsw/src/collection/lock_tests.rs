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
use zeusdb_vector_core::{Error, FsStorage, LockGuard, LockMode, Locked, Storage};

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

/// The words a user reads in the lock's refusal, held to being that
/// refusal.
fn refusal_words<T>(result: Result<T, Error>, what: &str) -> String {
    match result {
        Err(refusal @ Error::CollectionLocked { .. }) => refusal.to_string(),
        Err(other) => panic!("{what}: expected the lock's refusal, got {other}"),
        Ok(_) => panic!("{what}: expected the lock's refusal, and it went through"),
    }
}

/// The lock a load takes, shared and handed the record a load hands it,
/// held until the guard drops. A shared holder writes nothing into the lock
/// file.
fn held_as_a_load(storage: &FsStorage) -> Box<dyn LockGuard> {
    let record = crate::journal::holder_record(LockMode::Shared);
    match storage.lock(LockMode::Shared, &record) {
        Ok(Locked::Writable(guard)) => guard,
        other => panic!("expected a shared lock, got {other:?}"),
    }
}

/// What a refusal says, after the words that name the holder only as
/// another holder, where it quotes the lock file's last record.
const LAST_RECORD: &str =
    "is held by another holder, and its lock file holds the last record an exclusive holder wrote, ";

/// What a refusal says where the lock file is empty.
const NO_HOLDER_RECORDED: &str =
    "is held by another holder, and its lock file records no holder, so it cannot be";

/// What a refusal says where the lock file's bytes are not UTF-8.
const NO_READABLE_RECORD: &str =
    "is held by another holder, and its lock file holds no readable record, so it cannot be";

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

/// A save refused while a load holds the lock, where the lock file's last
/// record is this process's own from its earlier save, quotes that record as
/// the last record an exclusive holder wrote and names the holder, which is
/// the load, only as another holder.
#[test]
fn a_refusal_beside_a_load_quotes_the_refused_process_as_the_last_record() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("own.zdb");
    let dir = path.to_str().unwrap();
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(dir).unwrap();
    let storage = FsStorage::at(&path).unwrap();
    let _load = held_as_a_load(&storage);
    let words = refusal_words(collection.save(dir), "a second save beside a load");
    assert!(
        words.contains(&format!(
            "{LAST_RECORD}process {}, host ",
            std::process::id()
        )),
        "{words}"
    );
}

/// A directory with no lock file gets an empty one from the first holder
/// that takes the lock shared, as a load does, and a shared holder writes
/// nothing into it, though it is handed a record. A save refused beside that
/// holder names the lock file as recording no holder, and a read of the file
/// beside the holder gives its contents, which are empty.
#[test]
fn a_refusal_beside_an_empty_lock_file_names_it_as_recording_no_holder() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("empty.zdb");
    let dir = path.to_str().unwrap();
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(dir).unwrap();
    let storage = FsStorage::at(&path).unwrap();
    std::fs::remove_file(storage.lock_path()).unwrap();
    let _load = held_as_a_load(&storage);
    assert_eq!(
        std::fs::read(storage.lock_path()).unwrap(),
        b"",
        "a second handle reads the lock file beside the shared holder"
    );
    let words = refusal_words(collection.save(dir), "a save beside a load");
    assert!(words.contains(NO_HOLDER_RECORDED), "{words}");
}

/// A lock file whose bytes are not UTF-8 is read, and its read gives them
/// as no text. A save refused beside a holder that holds the lock shared, as
/// a load holds it, names the lock file as holding no readable record, and
/// not as one the system does not let a second handle read.
#[test]
fn a_refusal_beside_a_lock_file_that_is_not_utf8_names_no_readable_record() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("bytes.zdb");
    let dir = path.to_str().unwrap();
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(dir).unwrap();
    let storage = FsStorage::at(&path).unwrap();
    let bytes = b"process 1\nhost \xff\xfe\nmode exclusive\n";
    std::fs::write(storage.lock_path(), bytes).unwrap();
    let _load = held_as_a_load(&storage);
    assert_eq!(
        std::fs::read(storage.lock_path()).unwrap(),
        bytes,
        "a second handle reads the lock file's bytes beside the shared holder"
    );
    let words = refusal_words(collection.save(dir), "a save beside a load");
    assert!(words.contains(NO_READABLE_RECORD), "{words}");
}

/// A load of a directory that has no lock file creates the lock file empty
/// and writes nothing into it. A save refused while the next load holds the
/// lock names the lock file as recording no holder.
#[test]
fn a_load_without_a_lock_file_leaves_one_that_records_no_holder() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("unlocked.zdb");
    let dir = path.to_str().unwrap();
    let collection = Collection::build(declaration(), None);
    add(&collection, "a");
    collection.save(dir).unwrap();
    let storage = FsStorage::at(&path).unwrap();
    std::fs::remove_file(storage.lock_path()).unwrap();
    let loaded = Collection::load(dir).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        std::fs::read(storage.lock_path()).unwrap(),
        b"",
        "the load created the lock file empty"
    );
    let _next_load = held_as_a_load(&storage);
    let words = refusal_words(collection.save(dir), "a save beside the next load");
    assert!(words.contains(NO_HOLDER_RECORDED), "{words}");
}

/// A child process holds a copy of every file its parent has open from its
/// fork to its exec, the lock file among them, and a lock goes only when its
/// file is unlocked or the last copy of the file closes. The index the
/// parent lets go releases its lock first, so the directory opens again at
/// once while the child still holds its copy. The child here waits before
/// its exec until the reopen is done, so the case does not rest on timing.
#[cfg(unix)]
#[test]
fn a_child_holding_a_copy_of_the_lock_file_keeps_no_lock() {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::os::unix::process::CommandExt;

    let temp = TempDir::new("lock-tests");
    let path = temp.at("copied.zdb");
    let dir = path.to_str().unwrap();
    let first = Collection::build(declaration(), None);
    first.journal_to(dir, Durability::PerCall).unwrap();
    add(&first, "a");

    // The child says it has forked, then waits before its exec until the
    // parent says go, holding a copy of each of the parent's files.
    let (forked, forked_in_child) = UnixStream::pair().unwrap();
    let (go, go_in_child) = UnixStream::pair().unwrap();
    let spawner = std::thread::spawn(move || {
        let mut command = std::process::Command::new("true");
        // SAFETY: the closure runs in the child between its fork and its
        // exec, and calls only `write` and `read` on two sockets it owns.
        unsafe {
            command.pre_exec(move || {
                (&forked_in_child).write_all(b"f")?;
                let mut byte = [0u8; 1];
                (&go_in_child).read_exact(&mut byte)
            });
        }
        command.status()
    });
    let mut byte = [0u8; 1];
    (&forked).read_exact(&mut byte).unwrap();

    drop(first);
    let reopened = Collection::recover(dir, None, Durability::PerCall);
    (&go).write_all(b"g").unwrap();
    let status = spawner.join().unwrap().unwrap();
    assert!(status.success(), "{status:?}");
    let (reopened, _) = reopened.expect("the directory opens while the child holds its copy");
    assert!(reopened.contains("a"));
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
/// load said. Under `save` it saves a collection of one record into the
/// directory, writes its process id and ends.
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
        "save" => {
            let collection = Collection::build(declaration(), None);
            add(&collection, "child");
            collection.save(dir.to_str().unwrap()).unwrap();
            std::fs::write(&said, std::process::id().to_string()).unwrap();
        }
        _ => {}
    }
}

/// `child`'s exit, waiting at most a minute, after which the child is ended
/// and the test fails.
fn ended(mut child: std::process::Child) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the child did not end within a minute");
        }
        std::thread::sleep(Duration::from_millis(10));
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

/// A process that saved the directory and ended left its record in the lock
/// file. A save refused while a load holds the lock quotes that record as
/// the last record an exclusive holder wrote and names the holder, which is
/// the load, only as another holder. The child has ended before the lock is
/// held shared.
#[test]
fn a_refusal_beside_a_load_quotes_an_ended_savers_record_as_the_last_record() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("ended.zdb");
    assert!(ended(child(&path, "save")).success());
    let saver = std::fs::read_to_string(path.with_extension("said")).unwrap();
    let storage = FsStorage::at(&path).unwrap();
    let _load = held_as_a_load(&storage);
    let collection = Collection::build(declaration(), None);
    add(&collection, "ours");
    let words = refusal_words(
        collection.save(path.to_str().unwrap()),
        "a save beside a load",
    );
    assert!(
        words.contains(&format!("{LAST_RECORD}process {saver}, host ")),
        "{words}"
    );
}

/// The refusal a lock taken for a save gave, where it gave one.
fn lock_refused<T>(lock: &Result<T, Error>) -> Option<String> {
    lock.as_ref().err().map(ToString::to_string)
}

/// Two saves into a directory whose lock file this process cannot open to
/// write never hold the save's lock at once, so they never write the
/// directory at once: each is refused, naming the lock file.
#[test]
fn two_saves_never_hold_the_lock_at_once_where_its_file_refuses_writes() {
    let temp = TempDir::new("lock-tests");
    let path = temp.at("unwritable.zdb");
    let storage = FsStorage::at(&path).unwrap();
    std::fs::write(storage.lock_path(), b"").unwrap();
    let set_readonly = |readonly: bool| {
        let mut permissions = std::fs::metadata(storage.lock_path())
            .unwrap()
            .permissions();
        permissions.set_readonly(readonly);
        std::fs::set_permissions(storage.lock_path(), permissions).unwrap();
    };
    set_readonly(true);
    if std::fs::OpenOptions::new()
        .write(true)
        .open(storage.lock_path())
        .is_ok()
    {
        set_readonly(false);
        eprintln!("this process writes through a read-only permission, so nothing is held here");
        return;
    }
    let directory = path.to_str().unwrap();
    let first = crate::journal::lock_for_save(&storage, directory);
    let second = crate::journal::lock_for_save(&storage, directory);
    set_readonly(false);
    assert!(
        !(first.is_ok() && second.is_ok()),
        "both saves hold the lock: {first:?} and {second:?}"
    );
    let opening = format!(
        "The lock file {} cannot be opened to write: ",
        storage.lock_path().display()
    );
    for refused in [lock_refused(&first), lock_refused(&second)] {
        let words = refused.expect("a save whose lock file refuses writes is refused");
        assert!(words.starts_with(&opening), "{words}");
    }
}
