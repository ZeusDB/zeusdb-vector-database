//! A collection with a sparse space on disk.
//!
//! Every shape a collection can take, saved, reopened and searched to the
//! same page: a dense space alone, a dense and a sparse space, and both with
//! a text layer. What the directory holds, what the manifest declares, what
//! an older reader would meet, and what a tokenizer the engine cannot write
//! down does to a load. And what runs beside a save held between two of its
//! phases.

use std::collections::HashMap;
use std::sync::{mpsc, Arc};
use std::time::Duration;

use serde_json::{json, Value};

use zeusdb_vector_core::{
    compile_filter, pause_arm, pause_disarm, Error, FsDir, IdfScope, KillPoint, Operation, Pause,
    PauseAt, PauseState, Position, SparseVector, VectorGraph, VectorIndex, DUMP_FILENAME,
    NB_LAYER_MAX,
};
use zeusdb_vector_sparse::{SparseConfig, Weighting};
use zeusdb_vector_text::{SimpleTokenizer, Tokenizer, TokenizerConfig};

use super::journal_support::{answer_after, say_waits, waited_among, Said};
use super::{Collection, Declaration, ParsedRecord, SpaceConfig, SparseHalf, StorageMode};
use crate::journal::Durability;
use crate::locks::lock_name;

/// A directory under the system's temporary directory, removed on drop.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zeusdb-persist-tests-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn record(id: &str, dense: &[f32], sparse: Option<(&[u32], &[f32])>, cat: &str) -> ParsedRecord {
    let mut metadata: HashMap<String, Value> = HashMap::new();
    metadata.insert("cat".to_string(), json!(cat));
    ParsedRecord {
        id: id.to_string(),
        vector: dense.to_vec(),
        sparse: sparse.map(|(dims, values)| {
            SparseHalf::Vector(SparseVector {
                dims: dims.to_vec(),
                values: values.to_vec(),
            })
        }),
        metadata,
    }
}

/// A record carrying `text` as the terms the collection's tokenizer splits
/// it into, which the collection counts into ids as it inserts the record.
fn text_record(collection: &Collection, id: &str, dense: &[f32], text: &str) -> ParsedRecord {
    ParsedRecord {
        id: id.to_string(),
        vector: dense.to_vec(),
        sparse: Some(SparseHalf::Terms(collection.tokenize(text).unwrap())),
        metadata: HashMap::new(),
    }
}

/// The id the text layer's dictionary holds for `term`, where it holds one.
fn term_id(collection: &Collection, term: &str) -> Option<u32> {
    let text = collection.sparse().unwrap().text.as_ref().unwrap();
    let dictionary = text.dictionary.read().unwrap();
    dictionary.id_of(term)
}

/// Every live record's external id with its internal id, sorted.
fn ids_of(collection: &Collection) -> Vec<(String, usize)> {
    let mut ids: Vec<(String, usize)> = collection
        .internal_ids()
        .into_iter()
        .map(|(internal, id)| (id, internal))
        .collect();
    ids.sort();
    ids
}

fn base() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()]).unwrap()
}

/// Sixty records, a third of them without a sparse vector, over the two
/// spaces.
fn fill(collection: &Collection) -> Vec<ParsedRecord> {
    let records: Vec<ParsedRecord> = (0..60u32)
        .map(|i| {
            let dims: Vec<u32> = (0..4).map(|j| (i * 7 + j * 13) % 50).collect::<Vec<u32>>();
            let mut dims: Vec<u32> = dims;
            dims.sort_unstable();
            dims.dedup();
            let values: Vec<f32> = dims.iter().map(|d| 1.0 + (*d % 5) as f32).collect();
            let sparse = (i % 3 != 2).then_some((dims.as_slice(), values.as_slice()));
            record(
                &format!("r{i}"),
                &[i as f32 * 0.1, (i % 7) as f32],
                sparse,
                if i % 2 == 0 { "a" } else { "b" },
            )
        })
        .collect();
    let added = collection.add_records(records.clone(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    records
}

fn manifest(dir: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn config(dir: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn rewrite_manifest(dir: &std::path::Path, edit: impl FnOnce(&mut Value)) {
    let mut m = manifest(dir);
    edit(&mut m);
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
}

/// One page, as an external id and a score per hit.
type Page = Vec<(String, f32)>;

/// The three pages every shape is held to across a save and a load.
fn pages(collection: &Collection) -> (Page, Page, Page) {
    let params = collection.search_params(5, None, false, None).unwrap();
    let dense = collection.search_one(&[1.5, 3.0], None, params).unwrap();
    let dense: Page = dense
        .iter()
        .map(|h| (h.id().to_string(), h.score()))
        .collect();
    let query = SparseVector {
        dims: vec![1, 14, 27],
        values: vec![1.0, 2.0, 1.0],
    };
    let sparse = collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    let filter = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
    let filtered = collection
        .search_sparse(query.as_ref(), Some(&filter), 10, IdfScope::Corpus)
        .unwrap();
    (dense, sparse, filtered)
}

/// A dense-only directory keeps the flat names, lists no space among its
/// features, and its config.json carries no `spaces` field.
#[test]
fn a_dense_only_directory_keeps_the_flat_names() {
    let collection = Collection::build(base(), None);
    let records: Vec<ParsedRecord> = (0..10u32)
        .map(|i| record(&format!("r{i}"), &[i as f32, 0.0], None, "a"))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let dir = TempDir::new();
    let path = dir.path().join("dense.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let m = manifest(&path);
    assert_eq!(m["features"], json!({}));
    assert!(!m["files_included"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n.as_str().unwrap().starts_with("spaces/")));
    assert!(config(&path).get("spaces").is_none());
    assert!(!path.join("spaces").exists());
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 10);
    assert_eq!(loaded.space_configs().len(), 1);
}

/// A manifest that is not UTF-8 is refused as a read that failed, in the
/// words `std::fs::read_to_string` gives such a file.
#[test]
fn a_manifest_that_is_not_utf8_is_refused_as_a_text_read_refuses_it() {
    let collection = Collection::build(base(), None);
    let records: Vec<ParsedRecord> = (0..3u32)
        .map(|i| record(&format!("r{i}"), &[i as f32, 0.0], None, "a"))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let dir = TempDir::new();
    let path = dir.path().join("utf8.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let manifest = path.join("manifest.json");
    std::fs::write(&manifest, b"\xff\xfe{}").unwrap();
    let words = std::fs::read_to_string(&manifest).unwrap_err().to_string();
    match Collection::load(path.to_str().unwrap()) {
        Err(Error::ArtefactReadFailed { name, error }) => {
            assert_eq!(name, "manifest.json");
            assert_eq!(error, words);
        }
        Err(other) => panic!("the manifest was refused as {other}"),
        Ok(_) => panic!("a manifest that is not UTF-8 opened"),
    }
}

/// A collection with a sparse space writes `spaces/<name>/postings.zdbsparse`,
/// names it in the manifest by length alone, declares the space in
/// config.json by value, lists `sparse` among its features, and reopens to
/// the same three pages.
#[test]
fn a_sparse_space_round_trips_through_the_directory() {
    let declaration = base()
        .with_sparse(
            "terms",
            SparseConfig {
                weighting: Weighting::Bm25 { k1: 1.5, b: 0.6 },
                ..SparseConfig::default()
            },
        )
        .unwrap();
    let collection = Collection::build(declaration, None);
    fill(&collection);
    // A removal before the save, so the artefact carries only live records
    // and the mappings and the space agree on what is held.
    assert!(collection.remove_point("r4".to_string()).unwrap());
    let before = pages(&collection);
    assert!(!before.1.is_empty() && !before.2.is_empty());

    let dir = TempDir::new();
    let path = dir.path().join("both.zdb");
    collection.save(path.to_str().unwrap()).unwrap();

    let m = manifest(&path);
    assert_eq!(m["features"], json!({"sparse": "incompatible"}));
    let names: Vec<&str> = m["files_included"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    assert!(names.contains(&"spaces/terms/postings.zdbsparse"));
    assert!(!names.contains(&"spaces/terms/terms.zdbdict"));
    let digest = &m["file_digests"]["spaces/terms/postings.zdbsparse"];
    assert_eq!(
        digest["bytes"].as_u64().unwrap(),
        std::fs::metadata(path.join("spaces/terms/postings.zdbsparse"))
            .unwrap()
            .len()
    );
    assert!(
        digest.get("checksum").is_none(),
        "a framed artefact is recorded by length alone"
    );
    let c = config(&path);
    assert_eq!(
        c["spaces"],
        json!([{
            "name": "terms",
            "kind": "sparse",
            "index": {
                "unlink": "lazy",
                "lazy_threshold_percent": 10,
                "weighting": {"type": "bm25", "k1": 1.5, "b": 0.6}
            }
        }])
    );
    assert!(m["total_size_mb"].as_f64().unwrap() > 0.0);

    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 59);
    let configs = loaded.space_configs();
    assert_eq!(configs.len(), 2);
    assert_eq!(configs[1].0.as_str(), "terms");
    match &configs[1].1 {
        SpaceConfig::Sparse(config) => {
            assert_eq!(config.weighting, Weighting::Bm25 { k1: 1.5, b: 0.6 })
        }
        other => panic!("expected a sparse space, got {other:?}"),
    }
    assert_eq!(pages(&loaded), before);
    let sparse = loaded.sparse().unwrap().index.read().unwrap();
    assert_eq!(
        sparse.len(),
        39,
        "the removed record and the twenty without a vector"
    );
    assert_eq!(sparse.stranded(), 0);
    assert!(
        sparse.unit_costs() == zeusdb_vector_sparse::UnitCosts::FLOOR
            || sparse.unit_costs().measured
    );
    drop(sparse);

    // A save of the loaded collection is the same directory again.
    let again = dir.path().join("again.zdb");
    loaded.save(again.to_str().unwrap()).unwrap();
    assert_eq!(
        std::fs::read(path.join("spaces/terms/postings.zdbsparse")).unwrap(),
        std::fs::read(again.join("spaces/terms/postings.zdbsparse")).unwrap()
    );
    assert_eq!(config(&again)["spaces"], c["spaces"]);
}

/// A text layer writes its dictionary beside the postings, records its
/// tokenizer as `simple`, and reopens to the same text search with no
/// tokenizer handed.
#[test]
fn a_text_layer_round_trips_with_its_dictionary() {
    let declaration = base()
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    let collection = Collection::build(declaration, None);
    let texts = [
        "the quick brown fox",
        "a lazy dog sleeps",
        "the fox and the dog",
        "quick quick slow",
        "nothing in common here",
    ];
    let records: Vec<ParsedRecord> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| text_record(&collection, &format!("t{i}"), &[i as f32, 1.0], text))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let terms = collection.term_count().unwrap();
    let before = collection
        .search_text("quick fox", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(before[0].0, "t0");

    let dir = TempDir::new();
    let path = dir.path().join("text.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let m = manifest(&path);
    assert_eq!(
        m["features"],
        json!({"sparse": "incompatible", "text": "incompatible"})
    );
    assert!(path.join("spaces/text/terms.zdbdict").exists());
    let digest = &m["file_digests"]["spaces/text/terms.zdbdict"];
    assert!(digest.get("checksum").is_none());
    assert_eq!(config(&path)["spaces"][0]["tokenizer"], "simple");

    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.term_count(), Some(terms));
    assert_eq!(
        loaded
            .search_text("quick fox", None, 5, IdfScope::Corpus)
            .unwrap(),
        before
    );
    // A new term after the load takes the next id, so the dictionary came
    // back whole.
    let added = loaded.add_records(
        vec![text_record(&loaded, "t9", &[9.0, 1.0], "zebra")],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(term_id(&loaded, "zebra"), Some(terms as u32));
    // The built-in tokenizer may be handed as well, and its declaration
    // matches.
    assert!(Collection::load_with(path.to_str().unwrap(), Some(Arc::new(SimpleTokenizer))).is_ok());
}

/// A caller's own tokenizer is recorded as `external`, and the directory
/// refuses to open without one, opens with one, and refuses one whose
/// declaration is not the recorded one.
#[test]
fn an_external_tokenizer_must_be_handed_back() {
    struct Whitespace;
    impl Tokenizer for Whitespace {
        fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
            text.split_whitespace().for_each(term);
            Ok(())
        }
    }
    let declaration = base()
        .with_text("text", SparseConfig::default(), Arc::new(Whitespace))
        .unwrap();
    let collection = Collection::build(declaration, None);
    let records: Vec<ParsedRecord> = ["Alpha beta", "beta GAMMA"]
        .iter()
        .enumerate()
        .map(|(i, text)| text_record(&collection, &format!("t{i}"), &[i as f32, 0.0], text))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let before = collection
        .search_text("GAMMA", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(before.len(), 1);

    let dir = TempDir::new();
    let path = dir.path().join("external.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(config(&path)["spaces"][0]["tokenizer"], "external");

    let refused = Collection::load(path.to_str().unwrap());
    match refused {
        Err(Error::TokenizerRequired { space }) => assert_eq!(space, "text"),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    assert!(Collection::load(path.to_str().unwrap())
        .err()
        .unwrap()
        .to_string()
        .contains("records as external"),);
    match Collection::load_with(path.to_str().unwrap(), Some(Arc::new(SimpleTokenizer))) {
        Err(Error::TokenizerMismatch {
            recorded, handed, ..
        }) => {
            assert_eq!(recorded, "external");
            assert_eq!(handed, "simple");
        }
        other => panic!("expected a mismatch, got {:?}", other.map(|_| ())),
    }
    let loaded = Collection::load_with(path.to_str().unwrap(), Some(Arc::new(Whitespace))).unwrap();
    assert_eq!(
        loaded
            .search_text("GAMMA", None, 5, IdfScope::Corpus)
            .unwrap(),
        before
    );
    let configs = loaded.space_configs();
    match &configs[1].1 {
        SpaceConfig::Text(text) => assert_eq!(text.tokenizer, TokenizerConfig::External),
        other => panic!("expected a text space, got {other:?}"),
    }

    // A tokenizer handed to a directory that takes no text is refused.
    let plain = Collection::build(base(), None);
    let plain_path = dir.path().join("plain.zdb");
    plain.save(plain_path.to_str().unwrap()).unwrap();
    assert!(matches!(
        Collection::load_with(plain_path.to_str().unwrap(), Some(Arc::new(Whitespace))),
        Err(Error::TokenizerUnexpected)
    ));
}

/// The version rule in both directions, as this build sees it. A 1.x
/// manifest declaring a space is refused, a 6.x manifest is refused naming
/// the majors this build reads, and a 2.x dense-only manifest opens.
#[test]
fn the_version_rule_holds_in_both_directions() {
    let declaration = base()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let collection = Collection::build(declaration, None);
    fill(&collection);
    let dir = TempDir::new();
    let path = dir.path().join("both.zdb");
    collection.save(path.to_str().unwrap()).unwrap();

    // A 1.x label over a config that declares a space.
    let downgraded = dir.path().join("downgraded.zdb");
    copy_dir(&path, &downgraded);
    rewrite_manifest(&downgraded, |m| m["format_version"] = json!("1.1.0"));
    match Collection::load(downgraded.to_str().unwrap()) {
        Err(Error::FormatVersionSpaces { format_version }) => {
            assert_eq!(format_version, "1.1.0")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }

    // A later major, refused with the majors this build reads. Five is the
    // major this build writes and reads, so the refusal starts at six.
    let future = dir.path().join("future.zdb");
    copy_dir(&path, &future);
    rewrite_manifest(&future, |m| m["format_version"] = json!("6.0.0"));
    let message = Collection::load(future.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("format version 6.0.0 cannot be opened"),
        "{message}"
    );
    assert!(message.contains("1.x, 2.x, 3.x, 4.x and 5.x"), "{message}");
    assert!(message.contains("newer"), "{message}");

    // A 2.x label on a dense-only directory written as one before format
    // 5.0.0 opens, since the minor and the major are both read.
    let plain = Collection::build(base(), None);
    let plain_path = dir.path().join("plain.zdb");
    plain.save(plain_path.to_str().unwrap()).unwrap();
    crate::persistence::older::rewrite_before_five(&plain_path, "2.3.0");
    assert!(Collection::load(plain_path.to_str().unwrap()).is_ok());

    // A space declared under a name the collection refuses.
    let renamed = dir.path().join("renamed.zdb");
    copy_dir(&path, &renamed);
    let mut c = config(&renamed);
    c["spaces"][0]["name"] = json!("default");
    let text = serde_json::to_string_pretty(&c).unwrap();
    std::fs::write(renamed.join("config.json"), &text).unwrap();
    rewrite_manifest(&renamed, |m| {
        m["file_digests"]["config.json"] = json!({
            "bytes": text.len(),
            "checksum": format!("{:016x}", zeusdb_vector_core::checksum_of(text.as_bytes())),
        });
        super::journal_support::resized(m);
    });
    assert!(matches!(
        Collection::load(renamed.to_str().unwrap()),
        Err(Error::SpaceRecordInvalid { .. })
    ));
}

/// The two artefacts of a save agree by construction, and a directory whose
/// postings name a record the mappings do not, or whose dictionary is
/// shorter than the term ids the postings carry, is refused rather than
/// opened.
#[test]
fn a_space_out_of_step_with_the_mappings_is_refused() {
    let declaration = base()
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    let collection = Collection::build(declaration, None);
    let records: Vec<ParsedRecord> = ["one two", "two three", "three four"]
        .iter()
        .enumerate()
        .map(|(i, text)| text_record(&collection, &format!("t{i}"), &[i as f32, 0.0], text))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let dir = TempDir::new();
    let path = dir.path().join("text.zdb");
    collection.save(path.to_str().unwrap()).unwrap();

    // The postings from a save before a record was removed, over mappings
    // from after it. The slot count is held to the counter, which the
    // removal does not lower, so the check that fires is the one on the
    // record itself.
    let stale = dir.path().join("stale.zdb");
    copy_dir(&path, &stale);
    assert!(collection.remove_point("t1".to_string()).unwrap());
    let shrunk = dir.path().join("shrunk.zdb");
    collection.save(shrunk.to_str().unwrap()).unwrap();
    std::fs::copy(
        path.join("spaces/text/postings.zdbsparse"),
        shrunk.join("spaces/text/postings.zdbsparse"),
    )
    .unwrap();
    let stale_manifest = manifest(&stale);
    rewrite_manifest(&shrunk, |m| {
        m["file_digests"]["spaces/text/postings.zdbsparse"] =
            stale_manifest["file_digests"]["spaces/text/postings.zdbsparse"].clone();
        super::journal_support::resized(m);
    });
    match Collection::load(shrunk.to_str().unwrap()) {
        Err(Error::SparseRecordUnmapped { space, id }) => {
            assert_eq!(space, "text");
            assert_eq!(id, 2);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }

    // A dictionary from a collection that saw fewer terms.
    let short = dir.path().join("short.zdb");
    copy_dir(&path, &short);
    let fewer = Collection::build(
        base()
            .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
            .unwrap(),
        None,
    );
    let added = fewer.add_records(
        vec![text_record(&fewer, "f0", &[0.0, 0.0], "one two")],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    let fewer_path = dir.path().join("fewer.zdb");
    fewer.save(fewer_path.to_str().unwrap()).unwrap();
    std::fs::copy(
        fewer_path.join("spaces/text/terms.zdbdict"),
        short.join("spaces/text/terms.zdbdict"),
    )
    .unwrap();
    let fewer_manifest = manifest(&fewer_path);
    rewrite_manifest(&short, |m| {
        m["file_digests"]["spaces/text/terms.zdbdict"] =
            fewer_manifest["file_digests"]["spaces/text/terms.zdbdict"].clone();
        super::journal_support::resized(m);
    });
    match Collection::load(short.to_str().unwrap()) {
        Err(Error::TermIdBeyondDictionary { term, terms, .. }) => {
            assert_eq!(terms, 2);
            assert!(term >= 2);
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }

    // A postings artefact the manifest names and the directory lacks.
    let missing = dir.path().join("missing.zdb");
    copy_dir(&path, &missing);
    std::fs::remove_file(missing.join("spaces/text/postings.zdbsparse")).unwrap();
    match Collection::load(missing.to_str().unwrap()) {
        Err(Error::ArtefactsMissing { missing, contents }) => {
            assert_eq!(missing, vec!["spaces/text/postings.zdbsparse".to_string()]);
            assert!(contents.contains("postings of a sparse space"));
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// The graph's rebuild from the dump's rows keeps every internal id the
/// mappings name, so the sparse space, restored under those ids before the
/// graph, needs nothing carried through it: the loaded collection answers the
/// same sparse page and holds the ids and the counter a load from the dump
/// holds.
#[test]
fn the_graph_rebuild_fallback_carries_the_sparse_space() {
    let declaration = base()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let collection = Collection::build(declaration, None);
    fill(&collection);
    let before = pages(&collection);
    let dir = TempDir::new();
    let path = dir.path().join("both.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    // With its dump's topology damaged the loader rebuilds the graph from
    // the rows the dump holds.
    super::journal_support::break_topology(&path);
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 60);
    let after = pages(&loaded);
    assert_eq!(after.1, before.1);
    assert_eq!(after.2, before.2);
    assert_eq!(loaded.sparse().unwrap().index.read().unwrap().len(), 40);
    assert!(loaded.positions_agree());
    assert_eq!(ids_of(&loaded), ids_of(&collection));
    assert_eq!(loaded.id_counter(), collection.id_counter());
}

/// The raw rebuild from the dump's rows keeps every internal id the mappings
/// name and the counter `config.json` records, so a directory loaded through
/// it is the directory its own artefacts describe, and a stranded node's row
/// is dropped.
#[test]
fn the_raw_rebuild_fallback_keeps_every_internal_id_and_the_counter() {
    let collection = Collection::build(base(), None);
    let records: Vec<ParsedRecord> = (0..300u32)
        .map(|i| {
            record(
                &format!("r{i}"),
                &[(i % 17) as f32 * 0.3, (i % 11) as f32 * 0.7],
                None,
                if i % 2 == 0 { "a" } else { "b" },
            )
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    // A removal and an overwrite, so the ids are not simply one to the count.
    assert!(collection.remove_point("r10".to_string()).unwrap());
    assert!(collection.remove_point("r11".to_string()).unwrap());
    let overwritten = record("r5", &[9.0, 9.0], None, "b");
    assert_eq!(
        collection
            .add_records(vec![overwritten], vec![], true)
            .total_errors,
        0
    );
    let ids = ids_of(&collection);
    assert_eq!(ids.len(), 298);
    assert_eq!(collection.id_counter(), 301);
    assert_eq!(ids.iter().find(|(id, _)| id == "r5").unwrap().1, 301);

    let dir = TempDir::new();
    let path = dir.path().join("raw.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(config(&path)["id_counter"], 301);
    let fallback = dir.path().join("fallback.zdb");
    copy_dir(&path, &fallback);
    super::journal_support::break_topology(&fallback);

    let from_dump = Collection::load(path.to_str().unwrap()).unwrap();
    let rebuilt = Collection::load(fallback.to_str().unwrap()).unwrap();
    assert_eq!(ids_of(&from_dump), ids);
    assert_eq!(ids_of(&rebuilt), ids);
    assert_eq!(rebuilt.id_counter(), 301);
    assert_eq!(rebuilt.len(), 298);
    assert!(rebuilt.positions_agree());

    // The rebuilt graph holds one node per live record, under those ids, and
    // none for the id the overwrite retired. The records hold the first 298
    // positions.
    {
        let index = rebuilt.dense().index.read().unwrap();
        assert_eq!(index.graph().nb_points(), 298);
        let held: Vec<usize> = (0..298)
            .map(|node| index.graph().origin_id_at(node))
            .collect();
        for &(_, internal_id) in &ids {
            assert!(held.contains(&internal_id));
        }
        assert!(!held.contains(&6));
        assert!((0..298).all(|at| index.holds(Position::from_slot(at))));
        assert!(!index.holds(Position::from_slot(298)));
    }

    // The next record takes the id after the saved counter, and the rebuilt
    // collection filters and searches over the ids it kept.
    assert_eq!(
        rebuilt
            .add_records(vec![record("r900", &[1.0, 1.0], None, "a")], vec![], false)
            .total_errors,
        0
    );
    assert_eq!(rebuilt.internal_id_of("r900"), Some(302));
    assert_eq!(rebuilt.ids().slot_of("r900"), Some(298));
    let filter = compile_filter(&HashMap::from([("cat".to_string(), json!("b"))])).unwrap();
    assert_eq!(rebuilt.count(Some(&filter)), from_dump.count(Some(&filter)));
    let params = rebuilt.search_params(5, None, false, None).unwrap();
    let page = rebuilt.search_one(&[1.5, 3.0], None, params).unwrap();
    assert_eq!(page.len(), 5);
    for hit in page.iter() {
        assert!(ids.iter().any(|(id, _)| *id == hit.id()) || hit.id() == "r900");
    }
}

/// The collection's insert path, which draws the level before it writes
/// anything for the record and plans at it under the index's read guard,
/// dumps the graph an outright `VectorGraph::insert` over the same vectors
/// in the same order dumps, byte for byte.
#[test]
fn the_collections_insert_path_builds_the_graph_the_outright_insert_builds() {
    let collection = Collection::build(base(), None);
    let vectors: Vec<Vec<f32>> = (0..200)
        .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
        .collect();
    for (batch, chunk) in vectors.chunks(50).enumerate() {
        let records: Vec<ParsedRecord> = chunk
            .iter()
            .enumerate()
            .map(|(j, vector)| record(&format!("r{}", batch * 50 + j), vector, None, "a"))
            .collect();
        assert_eq!(
            collection.add_records(records, vec![], false).total_errors,
            0
        );
    }
    let dir = TempDir::new();
    let path = dir.path().join("built.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let saved = std::fs::read(path.join(DUMP_FILENAME)).unwrap();

    let mut graph = VectorGraph::new_raw("l2", 2, 4, 100, NB_LAYER_MAX as usize, 50);
    for (i, vector) in vectors.iter().enumerate() {
        graph.insert(vector, i + 1, Position::from_slot(i + 1));
    }
    let outright = dir.path().join("outright");
    std::fs::create_dir_all(&outright).unwrap();
    graph.dump(&FsDir::new(&outright), DUMP_FILENAME).unwrap();
    assert_eq!(saved, std::fs::read(outright.join(DUMP_FILENAME)).unwrap());
}

/// `clear`, `compact` and a removal each leave the saved shape correct.
#[test]
fn every_mutating_path_keeps_the_saved_shape_correct() {
    let declaration = base()
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    let collection = Collection::build(declaration, None);
    let texts: Vec<String> = (0..40)
        .map(|i| format!("word{} word{} common", i, i % 5))
        .collect();
    let records: Vec<ParsedRecord> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| text_record(&collection, &format!("t{i}"), &[i as f32, 0.0], text))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let dir = TempDir::new();

    // A removal then a compaction: the saved postings hold the live records
    // and the dictionary keeps every term.
    for i in 0..10 {
        assert!(collection.remove_point(format!("t{i}")).unwrap());
    }
    let terms = collection.term_count().unwrap();
    collection.compact().unwrap();
    assert_eq!(collection.term_count(), Some(terms));
    let compacted = dir.path().join("compacted.zdb");
    collection.save(compacted.to_str().unwrap()).unwrap();
    let loaded = Collection::load(compacted.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 30);
    assert_eq!(loaded.term_count(), Some(terms));
    assert_eq!(
        loaded
            .search_text("word3 common", None, 5, IdfScope::Corpus)
            .unwrap(),
        collection
            .search_text("word3 common", None, 5, IdfScope::Corpus)
            .unwrap()
    );

    // A clear: the saved space is empty and the dictionary starts again,
    // and the directory still declares the space.
    collection.clear().unwrap();
    let cleared = dir.path().join("cleared.zdb");
    collection.save(cleared.to_str().unwrap()).unwrap();
    let m = manifest(&cleared);
    assert_eq!(
        m["features"],
        json!({"sparse": "incompatible", "text": "incompatible"})
    );
    assert!(cleared.join("spaces/text/postings.zdbsparse").exists());
    let loaded = Collection::load(cleared.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.term_count(), Some(0));
    let added = loaded.add_records(
        vec![text_record(&loaded, "t0", &[0.0, 0.0], "fresh start")],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(term_id(&loaded, "fresh"), Some(0));
    assert_eq!(term_id(&loaded, "start"), Some(1));
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
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

/// The dense graph's request and slack, its store's, and the raw vectors'.
fn graph_reservation(collection: &Collection) -> (usize, usize, usize, usize, usize) {
    let index = collection.dense().index.read().unwrap();
    let graph = index.graph();
    (
        graph.links_memory_bytes(),
        graph.links_reserved_bytes(),
        graph.store_memory_bytes(),
        graph.store_reserved_bytes(),
        graph.raw_vectors_memory_bytes(),
    )
}

/// A directory holding no record keeps the graph the loader built its
/// collection around, and the load gives it the reservation a created
/// collection's graph holds. The metadata and column stores come back at the
/// declared reservation as well, whether or not a field is declared.
///
/// A save of a collection holding nothing writes no graph dump, so the load
/// reads none and takes the rebuild over no record.
#[test]
fn a_loaded_empty_collection_reserves_what_a_created_one_does() {
    for fields in [vec!["cat".to_string()], vec![]] {
        let declaration = Declaration::validate(16, "l2", 8, 64, 5_000, fields.clone()).unwrap();
        let created = Collection::build(declaration, None);
        let dir = TempDir::new();
        let path = dir.path().join("empty.zdb");
        created.save(path.to_str().unwrap()).unwrap();
        assert!(!path.join(DUMP_FILENAME).exists());
        let loaded = Collection::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.vector_count(), 0);

        assert!(graph_reservation(&created).1 > 0);
        assert_eq!(
            graph_reservation(&loaded),
            graph_reservation(&created),
            "fields {:?}",
            fields
        );
        let loaded_metadata = loaded.vector_metadata.read().unwrap().heap_bytes();
        let created_metadata = created.vector_metadata.read().unwrap().heap_bytes();
        assert_eq!(loaded_metadata, created_metadata, "fields {:?}", fields);
        let loaded_columns = loaded.columns.read().unwrap().heap_bytes();
        let created_columns = created.columns.read().unwrap().heap_bytes();
        assert_eq!(loaded_columns, created_columns, "fields {:?}", fields);
        let (loaded_stats, created_stats) = (loaded.stats(), created.stats());
        for key in [
            "graph_memory_mb",
            "raw_vectors_memory_mb",
            "reserved_memory_mb",
        ] {
            assert_eq!(
                loaded_stats[key], created_stats[key],
                "{} fields {:?}",
                key, fields
            );
        }
    }
}

/// The collection the loader builds before it reads a directory reserves
/// nothing for the declared size, in its graph or in its metadata, id or
/// column store, where a created collection reserves for it in all four. Its
/// graph, given the declared reservation, is the one a created collection
/// holds.
#[test]
fn the_collection_a_load_starts_from_reserves_nothing() {
    let fields = vec!["cat".to_string()];
    let empty = Collection::new_empty(16, "l2".to_string(), 8, 64, 100_000, fields.clone(), None);
    let created = Collection::build(
        Declaration::validate(16, "l2", 8, 64, 100_000, fields).unwrap(),
        None,
    );
    // Each guard in a statement of its own, in the declared order.
    let stores = |collection: &Collection| {
        let ids = collection.ids.read().unwrap().heap_bytes();
        let metadata = collection.vector_metadata.read().unwrap().heap_bytes();
        let columns = collection.columns.read().unwrap().heap_bytes();
        (metadata, ids, columns)
    };

    let (_, links_slack, _, store_slack, _) = graph_reservation(&empty);
    assert_eq!(links_slack + store_slack, 0);
    let (metadata, ids, columns) = stores(&empty);
    assert!(
        metadata <= 16 && ids <= 16 && columns < 1024,
        "{:?}",
        stores(&empty)
    );

    let (_, links_slack, _, store_slack, _) = graph_reservation(&created);
    assert!(links_slack + store_slack > 1 << 20);
    let (metadata, ids, columns) = stores(&created);
    assert!(
        metadata > 1 << 20 && ids > 1 << 20 && columns > 100_000,
        "{:?}",
        stores(&created)
    );

    empty.reserve_dense_graph();
    assert_eq!(graph_reservation(&empty), graph_reservation(&created));
}

/// Save `collection` as `name`, recover the directory, require the graph to
/// be the one its dump holds rather than a rebuild, and save the recovered
/// collection again to the same dump bytes. Hands back the recovered one.
fn reads_its_dump_and_writes_it_back(collection: &Collection, name: &str) -> Collection {
    let dir = TempDir::new();
    let path = dir.path().join(name);
    collection.save(path.to_str().unwrap()).unwrap();
    let (loaded, recovery) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert!(
        !recovery.graph_rebuilt,
        "{} was rebuilt rather than read from its dump",
        name
    );
    let again = dir.path().join(format!("again-{}", name));
    loaded.save(again.to_str().unwrap()).unwrap();
    assert_eq!(
        std::fs::read(path.join(DUMP_FILENAME)).unwrap(),
        std::fs::read(again.join(DUMP_FILENAME)).unwrap(),
        "{}",
        name
    );
    loaded
}

/// The page a dense query gives, as an external id and a score per hit.
fn dense_page(collection: &Collection, query: &[f32]) -> Page {
    let params = collection.search_params(5, None, false, None).unwrap();
    collection
        .search_one(query, None, params)
        .unwrap()
        .iter()
        .map(|h| (h.id().to_string(), h.score()))
        .collect()
}

/// A directory of one record reads its dump, one node under the only id the
/// counter issued, and a save after the load writes the same dump.
#[test]
fn a_directory_of_one_record_reads_its_dump_and_writes_it_back() {
    let collection = Collection::build(base(), None);
    assert_eq!(
        collection
            .add_records(vec![record("r0", &[0.5, 1.5], None, "a")], vec![], false)
            .total_errors,
        0
    );
    let loaded = reads_its_dump_and_writes_it_back(&collection, "one.zdb");
    assert_eq!(loaded.id_counter(), 1);
    assert_eq!(loaded.dense().index.read().unwrap().graph().nb_points(), 1);
    assert_eq!(
        dense_page(&loaded, &[0.5, 1.5]),
        dense_page(&collection, &[0.5, 1.5])
    );
}

/// A directory whose graph holds a node for every id its counter issued,
/// three of them stranded by two removals and an overwrite, reads its dump
/// with every node in place, and a save after the load writes the same dump.
#[test]
fn a_directory_holding_removed_and_overwritten_records_reads_its_dump_and_writes_it_back() {
    let collection = Collection::build(base(), None);
    let records: Vec<ParsedRecord> = (0..300u32)
        .map(|i| {
            record(
                &format!("r{i}"),
                &[(i % 17) as f32 * 0.3, (i % 11) as f32 * 0.7],
                None,
                if i % 2 == 0 { "a" } else { "b" },
            )
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    assert!(collection.remove_point("r10".to_string()).unwrap());
    assert!(collection.remove_point("r11".to_string()).unwrap());
    let overwritten = record("r5", &[9.0, 9.0], None, "b");
    assert_eq!(
        collection
            .add_records(vec![overwritten], vec![], true)
            .total_errors,
        0
    );
    assert_eq!(collection.id_counter(), 301);

    let loaded = reads_its_dump_and_writes_it_back(&collection, "removed.zdb");
    assert_eq!(
        loaded.dense().index.read().unwrap().graph().nb_points(),
        301
    );
    assert_eq!(loaded.len(), 298);
    assert_eq!(loaded.stats()["stranded_graph_nodes"], "3");
    assert_eq!(ids_of(&loaded), ids_of(&collection));
    assert_eq!(
        dense_page(&loaded, &[1.5, 3.0]),
        dense_page(&collection, &[1.5, 3.0])
    );
}

/// A directory at the largest `m` and the smallest dimension reads its dump,
/// and a save after the load writes the same dump.
#[test]
fn a_directory_at_the_largest_m_and_the_smallest_dimension_reads_its_dump_and_writes_it_back() {
    let declaration = Declaration::validate(1, "l2", 256, 64, 300, vec![]).unwrap();
    let collection = Collection::build(declaration, None);
    let records: Vec<ParsedRecord> = (0..300u32)
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: vec![((i * 37) % 101) as f32 * 0.01],
            sparse: None,
            metadata: HashMap::new(),
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let loaded = reads_its_dump_and_writes_it_back(&collection, "wide.zdb");
    assert_eq!((loaded.m(), loaded.dim()), (256, 1));
    assert_eq!(
        dense_page(&loaded, &[0.42]),
        dense_page(&collection, &[0.42])
    );
}

/// Lists at or below their owners' levels that a version 2 graph dump
/// leaves empty, the trailing ones the writer trims included, read off the
/// file.
fn empty_lists_at_or_below_levels(dump: &[u8]) -> usize {
    let length = |at: usize| u64::from_le_bytes(dump[at..at + 8].try_into().unwrap()) as usize;
    let count = |at: usize| u32::from_le_bytes(dump[at..at + 4].try_into().unwrap()) as usize;
    let layers = NB_LAYER_MAX as usize;
    let mut at = 128 + length(48) + length(56);
    let mut empty = 0;
    for level in 0..layers {
        for _ in 0..count(128 + 4 * level) {
            let lists = dump[at] as usize;
            at += 1;
            for layer in 0..lists {
                let entries = u16::from_le_bytes([dump[at], dump[at + 1]]) as usize;
                at += 2 + 4 * entries;
                if (1..=level).contains(&layer) && entries == 0 {
                    empty += 1;
                }
            }
            empty += (lists.max(1)..=level).count();
        }
    }
    empty
}

/// The largest origin id a version 2 graph dump names, read off the file.
fn largest_origin_id(dump: &[u8]) -> u64 {
    let nodes = u32::from_le_bytes(dump[32..36].try_into().unwrap()) as usize;
    let first = 128 + 4 * NB_LAYER_MAX as usize;
    (0..nodes)
        .map(|node| {
            let at = first + 4 * node;
            u32::from_le_bytes(dump[at..at + 4].try_into().unwrap()) as u64
        })
        .max()
        .unwrap_or(0)
}

/// Four values set by one record's number, the same every run.
fn spread(i: u32) -> Vec<f32> {
    let x = i as f32;
    vec![
        (x * 0.37).sin(),
        (x * 0.11).cos(),
        (x * 0.23).sin(),
        (x * 0.07).cos(),
    ]
}

/// A collection's dense pages over `queries`, and the graph dump a save of it
/// writes.
fn answers(collection: &Collection, queries: &[Vec<f32>]) -> (Vec<Page>, Vec<u8>) {
    let pages = queries
        .iter()
        .map(|query| dense_page(collection, query))
        .collect();
    let dir = TempDir::new();
    let path = dir.path().join("answers.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    (pages, std::fs::read(path.join(DUMP_FILENAME)).unwrap())
}

/// A collection whose saved graph holds empty lists at or below its levels,
/// read back from its dump and then changed, answers and writes what the
/// collection that saved it answers and writes after the same changes.
///
/// At `m` 2 the levels drawn over three hundred records put a node more than
/// one layer above every earlier one, so its lists at the layers between stay
/// empty, and the load holds each as a word. Both collections then take the
/// same inserts at recorded levels through `apply`, reaching every layer the
/// graph holds and two above it, a removal, an overwrite and a compaction.
/// After each step the two answer the same pages and a save of each writes
/// the same dump, and so do the two after a save and a load of each.
#[test]
fn a_collection_changed_after_its_load_matches_the_collection_that_saved_it() {
    let declaration = Declaration::validate(4, "l2", 2, 64, 400, vec![]).unwrap();
    let built = Collection::build(declaration, None);
    let records: Vec<ParsedRecord> = (0..300u32)
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: spread(i),
            sparse: None,
            metadata: HashMap::new(),
        })
        .collect();
    assert_eq!(built.add_records(records, vec![], false).total_errors, 0);
    let dir = TempDir::new();
    let path = dir.path().join("deep.zdb");
    built.save(path.to_str().unwrap()).unwrap();
    assert!(
        empty_lists_at_or_below_levels(&std::fs::read(path.join(DUMP_FILENAME)).unwrap()) > 0,
        "the saved graph holds no empty list at or below a level"
    );
    let (loaded, recovery) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert!(
        !recovery.graph_rebuilt,
        "the graph was rebuilt rather than read from its dump"
    );

    let queries: Vec<Vec<f32>> = (1_000..1_012u32).map(spread).collect();
    let same = |label: &str| {
        let (built_pages, built_dump) = answers(&built, &queries);
        let (loaded_pages, loaded_dump) = answers(&loaded, &queries);
        assert_eq!(loaded_pages, built_pages, "{}", label);
        assert!(loaded_dump == built_dump, "{}: the two dumps differ", label);
    };
    let apply = |operation: Operation| {
        built.apply(operation.clone()).unwrap();
        loaded.apply(operation).unwrap();
    };
    same("after the load");

    let mut issued = built.id_counter();
    assert_eq!(loaded.id_counter(), issued);
    for k in 0..40u32 {
        issued += 1;
        apply(Operation::Insert {
            id: format!("n{k}"),
            internal_id: issued as u64,
            level: (k * 5 % 12) as u8,
            vector: spread(2_000 + k),
            metadata: serde_json::Map::new(),
            sparse: None,
        });
    }
    same("after inserts at recorded levels");

    let removed: Vec<String> = (0..300u32)
        .step_by(9)
        .map(|i| format!("r{i}"))
        .chain(["n3".to_string(), "n17".to_string()])
        .collect();
    apply(Operation::Remove { ids: removed });
    apply(Operation::Remove {
        ids: vec!["r1".to_string()],
    });
    issued += 1;
    apply(Operation::Insert {
        id: "r1".to_string(),
        internal_id: issued as u64,
        level: 3,
        vector: spread(3_000),
        metadata: serde_json::Map::new(),
        sparse: None,
    });
    same("after a removal and an overwrite");

    apply(Operation::Compact);
    same("after a compaction");

    let built_again = reads_its_dump_and_writes_it_back(&built, "built.zdb");
    let loaded_again = reads_its_dump_and_writes_it_back(&loaded, "loaded.zdb");
    let (built_pages, built_dump) = answers(&built_again, &queries);
    let (loaded_pages, loaded_dump) = answers(&loaded_again, &queries);
    assert_eq!(loaded_pages, built_pages);
    assert!(
        loaded_dump == built_dump,
        "after a save and a load: the two dumps differ"
    );
    assert_eq!(ids_of(&loaded_again), ids_of(&built_again));
}

/// A directory whose counter is above every id its files name reads its
/// dump, holds what was saved, issues the counter's next id, and writes the
/// same dump again.
///
/// Removing the newest records and compacting drops their stranded nodes and
/// keeps the counter, so `config.json` counts ids that neither the id map nor
/// the graph dump names. Nothing a directory holds bounds the counter
/// tighter, and the load reads the dump as it reads one at its counter.
#[test]
fn a_directory_whose_counter_is_above_its_largest_id_reads_its_dump_and_writes_it_back() {
    let collection = Collection::build(base(), None);
    let records: Vec<ParsedRecord> = (0..300u32)
        .map(|i| {
            record(
                &format!("r{i}"),
                &[(i % 17) as f32 * 0.3, (i % 11) as f32 * 0.7],
                None,
                if i % 2 == 0 { "a" } else { "b" },
            )
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    for i in 250..300 {
        assert!(collection.remove_point(format!("r{i}")).unwrap());
    }
    collection.compact().unwrap();
    assert_eq!(collection.id_counter(), 300);

    let dir = TempDir::new();
    let path = dir.path().join("counted.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    assert_eq!(config(&path)["id_counter"], json!(300));
    assert_eq!(
        largest_origin_id(&std::fs::read(path.join(DUMP_FILENAME)).unwrap()),
        250
    );
    assert_eq!(
        ids_of(&collection)
            .iter()
            .map(|(_, internal)| *internal)
            .max(),
        Some(250)
    );

    let loaded = reads_its_dump_and_writes_it_back(&collection, "counted.zdb");
    assert_eq!(loaded.id_counter(), 300);
    assert_eq!(
        loaded.dense().index.read().unwrap().graph().nb_points(),
        250
    );
    assert_eq!(ids_of(&loaded), ids_of(&collection));
    assert_eq!(
        dense_page(&loaded, &[1.5, 3.0]),
        dense_page(&collection, &[1.5, 3.0])
    );
    assert_eq!(
        loaded
            .add_records(vec![record("late", &[0.4, 0.8], None, "a")], vec![], false)
            .total_errors,
        0
    );
    assert!(ids_of(&loaded).contains(&("late".to_string(), 301)));
}

// ============================================================================
// WHAT RUNS BESIDE A SAVE
// ============================================================================

/// How long a check waits for a thread beside it, and the longest the
/// pause holds the save, which is the tests' own bound on a hang.
const LIMIT: Duration = Duration::from_secs(20);

/// A vector of width eight for record `i`, the same every time.
fn wide(i: usize) -> Vec<f32> {
    (0..8)
        .map(|d| ((i * 7 + d * 13) % 29) as f32 * 0.25 - 3.5)
        .collect()
}

/// Run `write` on a thread of `scope`, and give what that thread says in
/// order: each guard it begins to wait at, then its return.
fn spawn_writer<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    write: impl FnOnce() + Send + 'scope,
) -> mpsc::Receiver<Said<()>> {
    let (said_tx, said) = mpsc::channel::<Said<()>>();
    scope.spawn(move || {
        say_waits(&said_tx);
        write();
        let _ = said_tx.send(Said::Done(()));
    });
    said
}

/// A save holds the mutation guard from its first phase to its last and no
/// guard a reader waits for. Held between its artefacts and its graph dump,
/// at `save:after_artefacts`, it lets a search, a count and a page of `list`
/// answer from this thread while the pause still holds it. A removal and an
/// add made on two more threads each reach their wait at `writers` before
/// they can return. Once the save is let go, the directory it wrote loads
/// with exactly the records the collection held as the save began, and the
/// removal and the add land after it. On a raw declaration and on a product
/// quantized one.
#[test]
fn a_save_held_between_its_phases_answers_searches_and_holds_writers_back() {
    for quantized in [false, true] {
        let declaration =
            Declaration::validate(8, "l2", 8, 50, 4000, vec!["cat".to_string()]).unwrap();
        let quantization = quantized.then(|| {
            declaration
                .quantization(4, 4, 1000, None, StorageMode::QuantizedWithRaw)
                .unwrap()
        });
        let collection = Collection::build(declaration, quantization);
        let records: Vec<ParsedRecord> = (0..1200)
            .map(|i| {
                let cat = if i % 2 == 0 { "a" } else { "b" };
                record(&format!("r{i}"), &wide(i), None, cat)
            })
            .collect();
        let added = collection.add_records(records, vec![], false);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
        assert_eq!(collection.is_quantized(), quantized);
        let cat_a = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
        let temp = TempDir::new();
        let dir = temp.path().join("held.zdb");
        let pause = Arc::new(Pause::new(LIMIT));

        std::thread::scope(|scope| {
            let saver = {
                let (collection, pause, dir) = (&collection, Arc::clone(&pause), &dir);
                scope.spawn(move || {
                    pause_arm(PauseAt::Kill(KillPoint::SaveAfterArtefacts), pause);
                    let saved = collection.save(dir.to_str().unwrap());
                    pause_disarm();
                    saved
                })
            };
            assert_eq!(
                pause.wait_reached(LIMIT).reached(),
                Some("save:after_artefacts"),
                "the save never reached save:after_artefacts"
            );

            // The readers, on this thread, while the save waits.
            let params = collection.search_params(5, None, false, None).unwrap();
            let page = collection.search_one(&wide(7), None, params).unwrap();
            let count = collection.count(Some(&cat_a));
            let listed = collection.list(10, 0, None).unwrap();
            assert_eq!(
                pause.state(),
                PauseState::Waiting("save:after_artefacts"),
                "a reader waited for the save"
            );

            // The writers, each on a thread of its own, reach their wait.
            let removal = spawn_writer(scope, || {
                assert!(collection.remove_point("r0".to_string()).unwrap());
            });
            let addition = spawn_writer(scope, || {
                let late = record("late", &wide(5_000), None, "a");
                let added = collection.add_records(vec![late], vec![], false);
                assert_eq!(added.total_errors, 0, "{:?}", added.errors);
            });
            // The save holds `writers` alone at its pause, by the rule this
            // test holds, so that is the guard each writer waits at.
            let save_held = [lock_name("writers")];
            for (name, said) in [("removal", &removal), ("add", &addition)] {
                waited_among(
                    said,
                    LIMIT,
                    &save_held,
                    &format!("the {name} waits at the mutation guard the save holds"),
                );
            }
            assert!(pause.release(), "the save was not waiting at its pause");
            saver.join().unwrap().unwrap();
            for (name, said) in [("removal", &removal), ("add", &addition)] {
                answer_after(
                    said,
                    LIMIT,
                    &format!("the {name} returns once the save is done"),
                );
            }
            assert_eq!(page.len(), 5);
            assert_eq!(count, 600);
            assert_eq!(listed.len(), 10);
        });

        let loaded = Collection::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(
            loaded.len(),
            1200,
            "the directory holds the records the collection held as the save began"
        );
        assert!(loaded.contains("r0") && !loaded.contains("late"));
        assert!(!collection.contains("r0") && collection.contains("late"));
        assert_eq!(loaded.is_quantized(), quantized);
    }
}
