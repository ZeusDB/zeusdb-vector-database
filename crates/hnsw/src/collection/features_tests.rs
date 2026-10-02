//! The features a saved directory lists, and what a load does with them.
//!
//! What every kind of save lists, a feature this build does not know refused
//! by its name unless the directory marks it compatible, one marked
//! compatible opened and not kept, a record that does not describe its
//! directory refused, what a directory below 4.0.0 or one saved before the
//! record existed does, and the words of both refusals.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use zeusdb_vector_core::{test_support::clustered, Error, SparseVector};
use zeusdb_vector_sparse::SparseConfig;
use zeusdb_vector_text::SimpleTokenizer;

use super::{Collection, Declaration, Int8Scale, ParsedRecord, SparseHalf, StorageMode};
use crate::Durability;

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
            "zeusdb-features-tests-{}-{}",
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

const DIM: usize = 8;

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 6, 60, 2000, vec!["cat".to_string()]).unwrap()
}

/// `n` records of clustered vectors, each with the sparse half `sparse`
/// gives it.
fn records(n: usize, sparse: impl Fn(usize) -> Option<SparseHalf>) -> Vec<ParsedRecord> {
    clustered(n, DIM, 0x0018_4001)
        .into_iter()
        .enumerate()
        .map(|(i, vector)| ParsedRecord {
            id: format!("r{i}"),
            vector,
            sparse: sparse(i),
            metadata: HashMap::from([("cat".to_string(), json!(["a", "b", "c"][i % 3]))]),
        })
        .collect()
}

fn filled(
    collection: Collection,
    n: usize,
    sparse: impl Fn(usize) -> Option<SparseHalf>,
) -> Collection {
    let added = collection.add_records(records(n, sparse), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    collection
}

fn dense(n: usize) -> Collection {
    filled(Collection::build(declaration(), None), n, |_| None)
}

fn pq(mode: StorageMode, n: usize) -> Collection {
    let d = declaration();
    let q = d.quantization(4, 4, 1000, None, mode).unwrap();
    filled(Collection::build(d, Some(q)), n, |_| None)
}

fn int8(n: usize) -> Collection {
    let d = declaration();
    let q = d
        .scalar_quantization(
            Int8Scale::PER_DIMENSION,
            1000,
            None,
            StorageMode::QuantizedOnly,
        )
        .unwrap();
    filled(Collection::build(d, Some(q)), n, |_| None)
}

fn sparse(n: usize) -> Collection {
    let d = declaration()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    filled(Collection::build(d, None), n, |i| {
        Some(SparseHalf::Vector(SparseVector {
            dims: vec![(i % 7) as u32, 7 + (i % 5) as u32],
            values: vec![1.0, 2.0],
        }))
    })
}

fn text(n: usize) -> Collection {
    let d = declaration()
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    filled(Collection::build(d, None), n, |i| {
        Some(SparseHalf::Terms(vec![
            format!("word{}", i % 11),
            "common".to_string(),
        ]))
    })
}

fn save(collection: &Collection, path: &Path) {
    collection.save(path.to_str().unwrap()).unwrap();
}

fn load(path: &Path) -> Result<Collection, Error> {
    Collection::load(path.to_str().unwrap())
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

/// A copy of `source` at `name` whose manifest `edit` has changed.
fn forged(temp: &TempDir, source: &Path, name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let path = temp.at(name);
    copy_dir(source, &path);
    rewrite_manifest(&path, edit);
    path
}

fn unsupported(result: Result<Collection, Error>) -> Vec<String> {
    match result {
        Err(Error::FeatureUnsupported { features, .. }) => features,
        Err(other) => panic!("expected an unknown feature refused, got {other}"),
        Ok(_) => panic!("expected an unknown feature refused, and the directory opened"),
    }
}

fn invalid(result: Result<Collection, Error>) -> String {
    match result {
        Err(Error::FeaturesInvalid { detail, .. }) => detail,
        Err(other) => panic!("expected the features refused, got {other}"),
        Ok(_) => panic!("expected the features refused, and the directory opened"),
    }
}

/// Every record's id, sorted, which is what an opened directory is held to.
fn ids(collection: &Collection) -> Vec<String> {
    let mut ids: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    ids.sort();
    ids
}

// ============================================================================
// WHAT A SAVE LISTS
// ============================================================================

/// Every kind of save lists exactly the features the directory holds, each
/// under its mark, written second in the manifest after the version, and the
/// directory opens.
#[test]
fn every_save_lists_exactly_the_features_it_holds() {
    let temp = TempDir::new();
    let cases: Vec<(&str, Collection, Value)> = vec![
        ("dense", dense(40), json!({"identity": "compatible"})),
        (
            "pq-raw",
            pq(StorageMode::QuantizedWithRaw, 1100),
            json!({"identity": "compatible", "pq": "incompatible"}),
        ),
        (
            "pq-only",
            pq(StorageMode::QuantizedOnly, 1100),
            json!({"identity": "compatible", "pq": "incompatible"}),
        ),
        (
            "pq-collecting",
            pq(StorageMode::QuantizedOnly, 40),
            json!({"identity": "compatible", "pq": "incompatible"}),
        ),
        (
            "int8",
            int8(1100),
            json!({"identity": "compatible", "int8": "incompatible"}),
        ),
        (
            "int8-collecting",
            int8(40),
            json!({"identity": "compatible", "int8": "incompatible"}),
        ),
        (
            "sparse",
            sparse(40),
            json!({"identity": "compatible", "sparse": "incompatible"}),
        ),
        (
            "text",
            text(40),
            json!({"identity": "compatible", "sparse": "incompatible", "text": "incompatible"}),
        ),
    ];
    for (name, collection, expected) in cases {
        let path = temp.at(&format!("{name}.zdb"));
        save(&collection, &path);
        assert_eq!(manifest(&path)["features"], expected, "{name}");
        let text = std::fs::read_to_string(path.join("manifest.json")).unwrap();
        let version = text.find("\"format_version\"").unwrap();
        let features = text.find("\"features\"").unwrap();
        let identity = text.find("\"identity\": {").unwrap();
        assert!(version < features && features < identity, "{name}: {text}");
        assert_eq!(ids(&load(&path).unwrap()), ids(&collection), "{name}");
    }

    // A journal, and a journal beside a scalar space and a sparse one.
    let journaled = temp.at("journaled.zdb");
    let collection = dense(30);
    collection
        .journal_to(journaled.to_str().unwrap(), Durability::default())
        .unwrap();
    assert_eq!(
        manifest(&journaled)["features"],
        json!({"identity": "compatible", "journal": "incompatible"})
    );
    drop(collection);
    assert_eq!(load(&journaled).unwrap().len(), 30);

    let both = temp.at("journaled-text.zdb");
    let collection = text(30);
    collection
        .journal_to(both.to_str().unwrap(), Durability::default())
        .unwrap();
    assert_eq!(
        manifest(&both)["features"],
        json!({
            "identity": "compatible",
            "journal": "incompatible",
            "sparse": "incompatible",
            "text": "incompatible"
        })
    );
}

/// What a save lists follows what the collection holds at the save, so a
/// collection that clears keeps its space's features, and two saves of the
/// same records list them in the same bytes.
#[test]
fn the_features_follow_the_collection_and_are_byte_stable() {
    let temp = TempDir::new();
    let collection = text(50);
    let first = temp.at("first.zdb");
    let second = temp.at("second.zdb");
    save(&collection, &first);
    save(&collection, &second);
    let block = |dir: &Path| {
        let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        let start = text.find("\"features\"").unwrap();
        let end = start + text[start..].find('}').unwrap();
        text[start..=end].to_string()
    };
    assert_eq!(block(&first), block(&second));

    collection.clear().unwrap();
    let cleared = temp.at("cleared.zdb");
    save(&collection, &cleared);
    assert_eq!(
        manifest(&cleared)["features"],
        json!({"identity": "compatible", "sparse": "incompatible", "text": "incompatible"})
    );
    assert_eq!(load(&cleared).unwrap().len(), 0);
}

// ============================================================================
// WHAT A LOAD DOES WITH A FEATURE IT DOES NOT KNOW
// ============================================================================

/// A feature this build does not know is refused by its name unless the
/// directory marks it compatible, whatever else the mark says, and every
/// such feature is named, in name order.
#[test]
fn a_feature_this_build_does_not_know_is_refused_by_name() {
    let temp = TempDir::new();
    let source = temp.at("source.zdb");
    save(&dense(20), &source);
    for (i, mark) in ["incompatible", "read_only", ""].iter().enumerate() {
        let path = forged(&temp, &source, &format!("later-{i}.zdb"), |m| {
            m["features"]["a_later_feature"] = json!(mark);
        });
        assert_eq!(
            unsupported(load(&path)),
            vec!["a_later_feature".to_string()],
            "{mark:?}"
        );
    }
    let path = forged(&temp, &source, "two.zdb", |m| {
        m["features"]["zz_later"] = json!("incompatible");
        m["features"]["a_later_feature"] = json!("incompatible");
        m["features"]["a_later_note"] = json!("compatible");
    });
    assert_eq!(
        unsupported(load(&path)),
        vec!["a_later_feature".to_string(), "zz_later".to_string()]
    );

    // Refused before any artefact is read, so a directory missing every
    // artefact is refused the same way.
    std::fs::remove_file(path.join("config.json")).unwrap();
    std::fs::remove_file(path.join("mappings.bin")).unwrap();
    assert_eq!(unsupported(load(&path)).len(), 2);

    // Under every policy a load takes.
    let journaled = temp.at("journaled.zdb");
    let collection = dense(10);
    collection
        .journal_to(journaled.to_str().unwrap(), Durability::default())
        .unwrap();
    drop(collection);
    rewrite_manifest(&journaled, |m| {
        m["features"]["a_later_feature"] = json!("incompatible")
    });
    assert_eq!(unsupported(load(&journaled)).len(), 1);
    match Collection::load_checkpoint_only(journaled.to_str().unwrap(), None) {
        Err(Error::FeatureUnsupported { features, .. }) => {
            assert_eq!(features, vec!["a_later_feature".to_string()])
        }
        Err(other) => panic!("expected an unknown feature refused, got {other}"),
        Ok(_) => panic!("expected an unknown feature refused, and the checkpoint opened"),
    }
}

/// A feature this build does not know that the directory marks compatible
/// is opened without, every record held, and a save from this build lists
/// only what it holds.
#[test]
fn a_compatible_feature_this_build_does_not_know_opens_and_is_not_kept() {
    let temp = TempDir::new();
    let source = temp.at("source.zdb");
    let collection = pq(StorageMode::QuantizedOnly, 1100);
    save(&collection, &source);
    let path = forged(&temp, &source, "noted.zdb", |m| {
        m["features"]["a_later_note"] = json!("compatible");
    });
    let opened = load(&path).unwrap();
    assert_eq!(ids(&opened), ids(&collection));
    assert!(opened.is_quantized());
    let again = temp.at("again.zdb");
    save(&opened, &again);
    assert_eq!(
        manifest(&again)["features"],
        json!({"identity": "compatible", "pq": "incompatible"})
    );
}

// ============================================================================
// WHAT A LOAD DOES WITH A RECORD THAT DOES NOT DESCRIBE ITS DIRECTORY
// ============================================================================

/// A feature this build knows listed under another mark than its own is
/// refused, naming the mark, quoted where it is neither of the two.
#[test]
fn a_known_feature_under_another_mark_is_refused() {
    let temp = TempDir::new();
    let source = temp.at("source.zdb");
    save(&dense(20), &source);
    let path = forged(&temp, &source, "identity.zdb", |m| {
        m["features"]["identity"] = json!("incompatible")
    });
    assert_eq!(invalid(load(&path)), "it marks identity incompatible");
    let path = forged(&temp, &source, "read-only.zdb", |m| {
        m["features"]["identity"] = json!("read_only")
    });
    assert_eq!(invalid(load(&path)), "it marks identity 'read_only'");
}

/// Of the features this build knows, a record lists exactly the ones the
/// directory holds, each way round and for every feature, once every part
/// they name is read.
#[test]
fn the_listed_features_are_the_held_ones() {
    let temp = TempDir::new();
    let drop_feature = |source: &Path, name: &str, feature: &'static str| {
        forged(&temp, source, name, |m| {
            m["features"].as_object_mut().unwrap().remove(feature);
        })
    };
    let add_feature = |source: &Path, name: &str, feature: &'static str| {
        forged(&temp, source, name, |m| {
            m["features"][feature] = json!("incompatible");
        })
    };

    let plain = temp.at("plain.zdb");
    save(&dense(20), &plain);
    assert_eq!(
        invalid(load(&drop_feature(&plain, "no-identity.zdb", "identity"))),
        "it lists no feature, and the directory holds identity"
    );
    for (feature, expected) in [
        (
            "int8",
            "it lists identity and int8, and the directory holds identity",
        ),
        (
            "journal",
            "it lists identity and journal, and the directory holds identity",
        ),
        (
            "pq",
            "it lists identity and pq, and the directory holds identity",
        ),
        (
            "sparse",
            "it lists identity and sparse, and the directory holds identity",
        ),
        (
            "text",
            "it lists identity and text, and the directory holds identity",
        ),
    ] {
        let path = add_feature(&plain, &format!("with-{feature}.zdb"), feature);
        assert_eq!(invalid(load(&path)), expected, "{feature}");
    }

    let quantized = temp.at("pq.zdb");
    save(&pq(StorageMode::QuantizedOnly, 1100), &quantized);
    assert_eq!(
        invalid(load(&drop_feature(&quantized, "no-pq.zdb", "pq"))),
        "it lists identity, and the directory holds identity and pq"
    );
    let scalar = temp.at("int8.zdb");
    save(&int8(1100), &scalar);
    assert_eq!(
        invalid(load(&drop_feature(&scalar, "no-int8.zdb", "int8"))),
        "it lists identity, and the directory holds identity and int8"
    );
    let words = temp.at("text.zdb");
    save(&text(30), &words);
    assert_eq!(
        invalid(load(&drop_feature(&words, "no-text.zdb", "text"))),
        "it lists identity and sparse, and the directory holds identity, sparse and text"
    );
    assert_eq!(
        invalid(load(&drop_feature(&words, "no-sparse.zdb", "sparse"))),
        "it lists identity and text, and the directory holds identity, sparse and text"
    );

    let journaled = temp.at("journaled.zdb");
    let collection = dense(10);
    collection
        .journal_to(journaled.to_str().unwrap(), Durability::default())
        .unwrap();
    drop(collection);
    let unlisted = drop_feature(&journaled, "no-journal.zdb", "journal");
    assert_eq!(
        invalid(load(&unlisted)),
        "it lists identity, and the directory holds identity and journal"
    );
    // The journal record is the manifest's, so the checkpoint alone is
    // refused too.
    match Collection::load_checkpoint_only(unlisted.to_str().unwrap(), None) {
        Err(Error::FeaturesInvalid { detail, .. }) => assert_eq!(
            detail,
            "it lists identity, and the directory holds identity and journal"
        ),
        Err(other) => panic!("expected the features refused, got {other}"),
        Ok(_) => panic!("expected the features refused, and the checkpoint opened"),
    }
}

// ============================================================================
// DIRECTORIES THAT LIST NO FEATURES, AND DIRECTORIES BELOW 4.0.0
// ============================================================================

/// A 4.0.0 directory saved before the record existed lists no features and
/// opens as it did, and its next save lists them.
#[test]
fn a_directory_listing_no_features_opens_and_lists_them_at_its_next_save() {
    let temp = TempDir::new();
    for (name, collection) in [("plain", dense(20)), ("text", text(20))] {
        let path = temp.at(&format!("{name}.zdb"));
        save(&collection, &path);
        let expected = manifest(&path)["features"].clone();
        rewrite_manifest(&path, |m| {
            m.as_object_mut().unwrap().remove("features");
        });
        let opened = load(&path).unwrap();
        assert_eq!(ids(&opened), ids(&collection), "{name}");
        let again = temp.at(&format!("{name}-again.zdb"));
        save(&opened, &again);
        assert_eq!(manifest(&again)["features"], expected, "{name}");
    }
}

/// Below 4.0.0 no release listed features, so a record there is read for a
/// feature this build does not know and is not held to the directory, whose
/// version decides what it holds, as it did.
#[test]
fn below_the_fourth_major_the_version_decides() {
    let temp = TempDir::new();
    let source = temp.at("source.zdb");
    save(&dense(20), &source);
    let path = forged(&temp, &source, "third.zdb", |m| {
        m["format_version"] = json!("3.0.0");
        m["features"]["journal"] = json!("incompatible");
        m["features"].as_object_mut().unwrap().remove("identity");
    });
    assert_eq!(load(&path).unwrap().len(), 20);
    let path = forged(&temp, &source, "third-later.zdb", |m| {
        m["format_version"] = json!("3.0.0");
        m["features"]["a_later_feature"] = json!("incompatible");
    });
    assert_eq!(
        unsupported(load(&path)),
        vec!["a_later_feature".to_string()]
    );
}

// ============================================================================
// THE WORDS
// ============================================================================

/// Both refusals name what a caller needs: the feature, what this build
/// knows and what to do, or what the record says against what the directory
/// holds and the marks this build gives.
#[test]
fn the_words_of_both_refusals() {
    let temp = TempDir::new();
    let source = temp.at("source.zdb");
    save(&dense(20), &source);
    let message = |path: &Path| load(path).err().unwrap().to_string();

    let one = forged(&temp, &source, "one.zdb", |m| {
        m["features"]["a_later_feature"] = json!("incompatible")
    });
    assert_eq!(
        message(&one),
        "The directory holds the feature 'a_later_feature', which this build does not \
         know. Its manifest does not mark it compatible, so a build without it cannot open \
         the directory. This build knows identity, int8, journal, pq, sparse and text. The \
         directory was written by a newer release of zeusdb-vector-database, so upgrade the \
         package to open it."
    );
    let three = forged(&temp, &source, "three.zdb", |m| {
        m["features"]["a"] = json!("incompatible");
        m["features"]["b"] = json!("incompatible");
        m["features"]["c"] = json!("incompatible");
    });
    assert_eq!(
        message(&three),
        "The directory holds the features 'a', 'b' and 'c', which this build does not \
         know. Its manifest does not mark them compatible, so a build without them cannot \
         open the directory. This build knows identity, int8, journal, pq, sparse and text. \
         The directory was written by a newer release of zeusdb-vector-database, so upgrade \
         the package to open it."
    );
    let listed = forged(&temp, &source, "listed.zdb", |m| {
        m["features"]["journal"] = json!("incompatible")
    });
    assert_eq!(
        message(&listed),
        "manifest.json lists features that do not describe the directory: it lists \
         identity and journal, and the directory holds identity. A saved directory lists \
         every feature it holds and no other, and this build marks identity compatible and \
         int8, journal, pq, sparse and text incompatible."
    );
}
