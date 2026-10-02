//! What a saved directory records about what it is, and what a load takes
//! back.
//!
//! The collection, the generation and the snapshot every save records, what
//! a copy of a directory looks like beside the directory saved again, what a
//! directory saved before the record existed does, what a journaled
//! directory's three ids are, and what is refused.

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use zeusdb_vector_core::{Error, FsStorage};

use super::{Collection, Declaration, Identity, ParsedRecord};
use crate::journal::{journal_path, Durability, JournalSink};

// ============================================================================
// FIXTURES
// ============================================================================

/// A directory under the system's temporary directory, removed on drop
/// together with the journals beside anything in it.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-identity-tests-{}-{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

fn save(collection: &Collection, path: &Path) {
    collection.save(path.to_str().unwrap()).unwrap();
}

fn load(path: &Path) -> Collection {
    Collection::load(path.to_str().unwrap()).unwrap()
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn rewrite_manifest(dir: &Path, edit: impl FnOnce(&mut Value)) {
    let mut m = manifest(dir);
    edit(&mut m);
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
}

fn hex(id: u128) -> String {
    format!("{id:032x}")
}

/// The identity a directory's manifest records, read back as the collection
/// reports one.
fn recorded(dir: &Path) -> Identity {
    let m = manifest(dir);
    let identity = &m["identity"];
    let id = |value: &Value| u128::from_str_radix(value.as_str().unwrap(), 16).unwrap();
    Identity {
        collection_id: id(&identity["collection_id"]),
        generation: identity["generation"].as_u64().unwrap(),
        snapshot: Some(id(&identity["snapshot"])),
        parent: identity.get("parent").map(id),
    }
}

/// The collection id the journal beside `dir` names in its header.
fn header_id(dir: &Path) -> u128 {
    let bytes = std::fs::read(journal_path(dir).unwrap()).unwrap();
    u128::from_le_bytes(bytes[24..40].try_into().unwrap())
}

/// A whole directory, copied byte for byte, which is what a backup is.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

// ============================================================================
// WHAT A SAVE RECORDS
// ============================================================================

/// A new collection is at generation 0 with no snapshot. Each save records
/// the collection, one generation more, a snapshot of its own and the
/// snapshot before it as the parent, and the collection reports what it
/// last wrote.
#[test]
fn every_save_records_the_collection_one_generation_on_and_the_snapshot_before() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    let id = collection.collection_id();
    assert_eq!(
        collection.identity(),
        Identity {
            collection_id: id,
            generation: 0,
            snapshot: None,
            parent: None,
        }
    );

    add(&collection, 0..10);
    let first = temp.at("first.zdb");
    save(&collection, &first);
    let one = recorded(&first);
    assert_eq!(one.collection_id, id);
    assert_eq!(one.generation, 1);
    assert_eq!(one.parent, None);
    assert!(
        manifest(&first)["identity"].get("parent").is_none(),
        "absent rather than null"
    );
    assert_eq!(
        collection.identity(),
        one,
        "the collection reports what it wrote"
    );

    // The same records saved again are a second snapshot.
    save(&collection, &first);
    let two = recorded(&first);
    assert_eq!(two.collection_id, id);
    assert_eq!(two.generation, 2);
    assert_eq!(two.parent, one.snapshot);
    assert_ne!(two.snapshot, one.snapshot);

    // Elsewhere is the same collection's next save.
    let second = temp.at("second.zdb");
    save(&collection, &second);
    let three = recorded(&second);
    assert_eq!(three.collection_id, id);
    assert_eq!(three.generation, 3);
    assert_eq!(three.parent, two.snapshot);
    assert_eq!(collection.identity(), three);

    // Every id is spelled as 32 hexadecimal digits.
    let m = manifest(&second);
    for field in ["collection_id", "snapshot", "parent"] {
        let value = m["identity"][field].as_str().unwrap();
        assert_eq!(value.len(), 32, "{field}");
        assert!(value.bytes().all(|b| b.is_ascii_hexdigit()), "{field}");
    }
    assert_eq!(m["identity"]["collection_id"], json!(hex(id)));
}

/// A save that fails leaves the collection at the generation it had, so the
/// next save records the generation after the last one committed.
#[test]
fn a_save_that_fails_moves_nothing() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..5);
    let path = temp.at("kept.zdb");
    save(&collection, &path);
    let one = collection.identity();

    std::fs::write(temp.at("afile"), b"x").unwrap();
    let under_a_file = temp.at("afile").join("sub.zdb");
    assert!(collection.save(under_a_file.to_str().unwrap()).is_err());
    assert_eq!(collection.identity(), one, "nothing was committed");

    save(&collection, &path);
    assert_eq!(recorded(&path).generation, 2);
    assert_eq!(recorded(&path).parent, one.snapshot);
}

// ============================================================================
// WHAT IDENTITY IS FOR
// ============================================================================

/// A directory copied and then saved over tells the copy apart: the copy is
/// one generation behind, and the directory saved over names the copy's
/// snapshot as its parent. A copy saved again in its own place is a second
/// child of that snapshot, at the same generation, under a snapshot of its
/// own.
#[test]
fn a_copy_left_behind_by_a_later_save_is_the_snapshot_the_later_one_names() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    let current = temp.at("current.zdb");
    save(&collection, &current);
    let copy = temp.at("copy.zdb");
    copy_dir(&current, &copy);
    assert_eq!(
        recorded(&copy),
        recorded(&current),
        "a copy is the same snapshot"
    );

    add(&collection, 10..15);
    save(&collection, &current);
    let (old, new) = (recorded(&copy), recorded(&current));
    assert_eq!(old.collection_id, new.collection_id);
    assert_eq!(old.generation + 1, new.generation);
    assert_eq!(
        new.parent, old.snapshot,
        "the current directory was saved from the copy's state"
    );

    // The copy opened and saved where it is: a fork, which the snapshots show.
    let forked = load(&copy);
    save(&forked, &copy);
    let fork = recorded(&copy);
    assert_eq!(fork.collection_id, new.collection_id);
    assert_eq!(fork.generation, new.generation);
    assert_eq!(fork.parent, new.parent);
    assert_ne!(fork.snapshot, new.snapshot);
}

/// A directory loaded and saved keeps its collection and moves one
/// generation on, with what it was loaded from as the parent.
#[test]
fn a_directory_loaded_and_saved_keeps_its_collection() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    let path = temp.at("kept.zdb");
    save(&collection, &path);
    let written = recorded(&path);
    drop(collection);

    let loaded = load(&path);
    assert_eq!(loaded.identity(), written);
    save(&loaded, &path);
    let again = recorded(&path);
    assert_eq!(again.collection_id, written.collection_id);
    assert_eq!(again.generation, 2);
    assert_eq!(again.parent, written.snapshot);
}

/// An unjournaled directory loaded twice is one collection, and the one its
/// manifest names.
#[test]
fn an_unjournaled_directory_loaded_twice_is_one_collection() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    let path = temp.at("plain.zdb");
    save(&collection, &path);
    let id = collection.collection_id();
    drop(collection);

    let first = load(&path);
    let second = load(&path);
    assert_eq!(first.collection_id(), id);
    assert_eq!(second.collection_id(), id);
    assert_eq!(first.identity(), second.identity());
}

/// The manifest's identity, its journal record and the journal's header
/// name one collection, through `journal_to`, a checkpoint and a recovery.
/// `journal_to`'s checkpoint is a save, so it records a generation, and a
/// checkpoint after it records the next.
#[test]
fn a_journaled_directory_names_one_collection_three_times() {
    let temp = TempDir::new();
    let path = temp.at("journaled.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    let id = collection.collection_id();
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();

    let agree = |dir: &Path| {
        let m = manifest(dir);
        assert_eq!(m["identity"]["collection_id"], json!(hex(id)));
        assert_eq!(m["journal"]["collection_id"], json!(hex(id)));
        assert_eq!(header_id(dir), id);
    };
    agree(&path);
    let one = recorded(&path);
    assert_eq!(one.generation, 1);

    add(&collection, 10..20);
    collection.checkpoint().unwrap();
    agree(&path);
    let two = recorded(&path);
    assert_eq!(two.generation, 2);
    assert_eq!(two.parent, one.snapshot);
    add(&collection, 20..25);
    drop(collection);

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.replayed, 5);
    assert_eq!(recovered.identity(), two, "a replay is not a save");
    recovered.checkpoint().unwrap();
    agree(&path);
    assert_eq!(recorded(&path).generation, 3);
}

/// `clear()` keeps the collection, its journal's header and its place among
/// its saves, so the next checkpoint is the next generation of the same
/// collection and replays onto the same journal.
#[test]
fn clear_keeps_the_identity_whole() {
    let temp = TempDir::new();
    let path = temp.at("cleared.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    add(&collection, 0..10);
    collection.checkpoint().unwrap();
    let before = collection.identity();
    assert_eq!(before.generation, 2);

    assert_eq!(collection.clear().unwrap(), 10);
    assert_eq!(collection.identity(), before, "a clear is not a save");
    add(&collection, 10..12);
    collection.checkpoint().unwrap();
    let after = recorded(&path);
    assert_eq!(after.collection_id, before.collection_id);
    assert_eq!(after.generation, 3);
    assert_eq!(after.parent, before.snapshot);
    assert_eq!(header_id(&path), before.collection_id);
    drop(collection);

    let reopened = load(&path);
    assert_eq!(reopened.identity(), after);
    assert_eq!(reopened.len(), 2);
}

// ============================================================================
// A DIRECTORY SAVED BEFORE THE RECORD EXISTED
// ============================================================================

/// A directory with no identity opens as it always did: unjournaled, it
/// takes the id drawn at assembly, which differs from load to load, at
/// generation 0. Its next save records that id at generation 1 with no
/// parent, and from then on it is one collection.
#[test]
fn a_directory_recording_no_identity_is_named_at_its_next_save() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..10);
    let path = temp.at("older.zdb");
    save(&collection, &path);
    drop(collection);
    rewrite_manifest(&path, |m| {
        m.as_object_mut().unwrap().remove("identity");
    });

    let first = load(&path);
    let second = load(&path);
    assert_ne!(first.collection_id(), second.collection_id());
    assert_eq!(first.identity().generation, 0);
    assert_eq!(first.identity().snapshot, None);
    assert_eq!(first.identity().parent, None);
    drop(second);

    save(&first, &path);
    let named = recorded(&path);
    assert_eq!(named.collection_id, first.collection_id());
    assert_eq!(named.generation, 1);
    assert_eq!(named.parent, None);
    assert_eq!(load(&path).collection_id(), named.collection_id);
    assert_eq!(load(&path).collection_id(), named.collection_id);
}

/// A journaled directory with no identity takes its journal record's id, as
/// it always did, and its next checkpoint records that id.
#[test]
fn a_journaled_directory_recording_no_identity_takes_its_journal_records_id() {
    let temp = TempDir::new();
    let path = temp.at("older-journaled.zdb");
    let collection = Collection::build(declaration(), None);
    let id = collection.collection_id();
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    add(&collection, 0..6);
    drop(collection);
    rewrite_manifest(&path, |m| {
        m.as_object_mut().unwrap().remove("identity");
    });

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.replayed, 6);
    assert_eq!(recovered.collection_id(), id);
    assert_eq!(recovered.identity().generation, 0);
    recovered.checkpoint().unwrap();
    let named = recorded(&path);
    assert_eq!(named.collection_id, id);
    assert_eq!(named.generation, 1);
    assert_eq!(named.parent, None);
    assert_eq!(header_id(&path), id);
}

// ============================================================================
// WHAT IS REFUSED
// ============================================================================

/// An identity a save could not have written is refused before any
/// artefact is read, naming the field. A field of the wrong type is refused
/// by the parser, as any manifest field is.
#[test]
fn an_identity_no_save_writes_is_refused() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..5);
    let good = temp.at("good.zdb");
    save(&collection, &good);

    let refused = |name: &str, edit: &dyn Fn(&mut Value)| -> Error {
        let path = temp.at(name);
        copy_dir(&good, &path);
        rewrite_manifest(&path, edit);
        match Collection::load(path.to_str().unwrap()) {
            Err(error) => error,
            Ok(_) => panic!("{name} opened"),
        }
    };

    for (name, field, value) in [
        ("short.zdb", "collection_id", json!("abc")),
        (
            "signed.zdb",
            "collection_id",
            json!(format!("+{}", "0".repeat(31))),
        ),
        ("snapshot.zdb", "snapshot", json!("z".repeat(32))),
        ("parent.zdb", "parent", json!("0".repeat(31))),
    ] {
        match refused(name, &|m: &mut Value| m["identity"][field] = value.clone()) {
            Error::IdentityInvalid { detail } => {
                assert!(detail.starts_with(&format!("its {field} is '")), "{detail}");
                assert!(
                    detail.ends_with("which is not 32 hexadecimal digits"),
                    "{detail}"
                );
            }
            other => panic!("{name}: expected IdentityInvalid, got {other:?}"),
        }
    }

    let zero = refused("zero.zdb", &|m: &mut Value| {
        m["identity"]["generation"] = json!(0)
    });
    assert!(matches!(zero, Error::IdentityInvalid { .. }), "{zero:?}");
    assert_eq!(
        zero.to_string(),
        "manifest.json records an identity this build cannot read: its generation is 0, and \
         a collection's first save records 1. A saved directory records the id of its \
         collection, its generation counting from 1, the id of its snapshot and, where it has \
         one, the id of the snapshot it was saved from, every id as 32 hexadecimal digits."
    );
    let negative = refused("negative.zdb", &|m: &mut Value| {
        m["identity"]["generation"] = json!(-1)
    });
    assert!(
        matches!(
            negative,
            Error::ArtefactParseFailed {
                name: "manifest.json",
                ..
            }
        ),
        "{negative:?}"
    );
    let missing = refused("missing.zdb", &|m: &mut Value| {
        m["identity"].as_object_mut().unwrap().remove("snapshot");
    });
    assert!(
        matches!(
            missing,
            Error::ArtefactParseFailed {
                name: "manifest.json",
                ..
            }
        ),
        "{missing:?}"
    );
}

/// A journal record naming another collection than the identity is refused
/// where the journal is read, and the checkpoint alone still opens, as the
/// identity's collection. A journal header naming another collection than
/// the identity is refused as a journal from another index.
#[test]
fn a_journal_record_and_an_identity_naming_two_collections_are_refused() {
    let temp = TempDir::new();
    let path = temp.at("two.zdb");
    let collection = Collection::build(declaration(), None);
    let id = collection.collection_id();
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    add(&collection, 0..6);
    drop(collection);
    let other = id ^ 1;

    let record_apart = temp.at("record-apart.zdb");
    copy_dir(&path, &record_apart);
    std::fs::copy(
        journal_path(&path).unwrap(),
        journal_path(&record_apart).unwrap(),
    )
    .unwrap();
    rewrite_manifest(&record_apart, |m| {
        m["journal"]["collection_id"] = json!(hex(other))
    });
    match Collection::recover(record_apart.to_str().unwrap(), None, Durability::default()) {
        Err(error @ Error::JournalManifestInvalid { .. }) => assert_eq!(
            error.to_string(),
            format!(
                "manifest.json names a journal this build cannot read: its collection_id is \
                 '{}' and the directory's identity names collection '{}'. A directory saved \
                 with a journal records its file name, the collection id both it and the \
                 directory carry, and the sequence the checkpoint holds.",
                hex(other),
                hex(id)
            )
        ),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    let alone = Collection::load_checkpoint_only(record_apart.to_str().unwrap(), None).unwrap();
    assert_eq!(alone.collection_id(), id);

    let identity_apart = temp.at("identity-apart.zdb");
    copy_dir(&path, &identity_apart);
    std::fs::copy(
        journal_path(&path).unwrap(),
        journal_path(&identity_apart).unwrap(),
    )
    .unwrap();
    rewrite_manifest(&identity_apart, |m| {
        m["identity"]["collection_id"] = json!(hex(other))
    });
    match Collection::recover(
        identity_apart.to_str().unwrap(),
        None,
        Durability::default(),
    ) {
        Err(Error::JournalNotThisCollection {
            journal_id,
            directory_id,
            ..
        }) => {
            assert_eq!(journal_id, hex(id));
            assert_eq!(directory_id, hex(other));
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// A journal record's id of fewer than 32 digits is refused on every
/// policy, since the id would come back spelled otherwise.
#[test]
fn a_journal_records_id_of_fewer_digits_is_refused() {
    let temp = TempDir::new();
    let path = temp.at("short.zdb");
    let collection = Collection::build(declaration(), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    drop(collection);
    rewrite_manifest(&path, |m| m["journal"]["collection_id"] = json!("abc"));

    for opened in [
        Collection::load(path.to_str().unwrap()).map(|_| ()),
        Collection::load_checkpoint_only(path.to_str().unwrap(), None).map(|_| ()),
    ] {
        match opened {
            Err(Error::JournalManifestInvalid { detail }) => assert_eq!(
                detail,
                "its collection_id is 'abc', which is not 32 hexadecimal digits"
            ),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}

/// A sink naming another collection is refused before anything is
/// written, so no manifest names two collections.
#[test]
fn a_save_refuses_a_sink_naming_another_collection() {
    let temp = TempDir::new();
    let path = temp.at("foreign.zdb");
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..4);
    let other = collection.collection_id() ^ 1;
    let sink =
        JournalSink::create(&FsStorage::at(&path).unwrap(), other, Durability::PerCall).unwrap();
    collection.attach_sink(Box::new(sink));

    match collection.save(path.to_str().unwrap()) {
        Err(Error::JournalNotThisCollection {
            journal_id,
            directory_id,
            ..
        }) => {
            assert_eq!(journal_id, hex(other));
            assert_eq!(directory_id, hex(collection.collection_id()));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(!path.exists(), "nothing was written");
    assert_eq!(collection.identity().generation, 0);
}

/// The largest generation a manifest can hold opens, and a save of it is
/// refused before anything is written.
#[test]
fn the_last_generation_opens_and_cannot_be_saved_again() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..4);
    let path = temp.at("last.zdb");
    save(&collection, &path);
    drop(collection);
    rewrite_manifest(&path, |m| m["identity"]["generation"] = json!(u64::MAX));
    let before = std::fs::read(path.join("manifest.json")).unwrap();

    let loaded = load(&path);
    assert_eq!(loaded.identity().generation, u64::MAX);
    match loaded.save(path.to_str().unwrap()) {
        Err(Error::Engine(message)) => {
            assert!(message.contains("18446744073709551615"), "{message}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(std::fs::read(path.join("manifest.json")).unwrap(), before);
    assert_eq!(loaded.identity().generation, u64::MAX);
}

// ============================================================================
// WHAT TWO SAVES OF THE SAME RECORDS DIFFER IN
// ============================================================================

/// The manifest fields two saves differ in, every other artefact being the
/// same bytes. Two saves from one snapshot differ in the stamp and the
/// snapshot. A save and the one it was saved from differ in the generation
/// and the parent too.
#[test]
fn two_saves_of_the_same_records_differ_in_the_manifest_alone() {
    let temp = TempDir::new();
    let collection = Collection::build(declaration(), None);
    add(&collection, 0..30);
    let original = temp.at("original.zdb");
    save(&collection, &original);
    drop(collection);

    let one = temp.at("one.zdb");
    let two = temp.at("two.zdb");
    save(&load(&original), &one);
    save(&load(&original), &two);

    let bytes = |dir: &Path, name: &str| std::fs::read(dir.join(name)).unwrap();
    for entry in std::fs::read_dir(&original).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name == "manifest.json" {
            continue;
        }
        assert_eq!(bytes(&one, &name), bytes(&two, &name), "{name}");
        assert_eq!(bytes(&one, &name), bytes(&original, &name), "{name}");
    }

    let differing = |a: &Path, b: &Path| {
        let (a, b) = (manifest(a), manifest(b));
        let mut out = Vec::new();
        for (key, value) in a.as_object().unwrap() {
            if key == "identity" {
                for field in ["collection_id", "generation", "snapshot", "parent"] {
                    if value.get(field) != b[key].get(field) {
                        out.push(format!("identity.{field}"));
                    }
                }
            } else if Some(value) != b.get(key) {
                out.push(key.clone());
            }
        }
        out
    };
    assert_eq!(differing(&one, &two), ["identity.snapshot", "saved_at"]);
    assert_eq!(
        differing(&original, &one),
        [
            "identity.generation",
            "identity.snapshot",
            "identity.parent",
            "saved_at"
        ]
    );
}
