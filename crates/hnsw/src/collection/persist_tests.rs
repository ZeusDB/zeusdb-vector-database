//! A collection with a sparse space on disk.
//!
//! Every shape a collection can take, saved, reopened and searched to the
//! same page: a dense space alone, a dense and a sparse space, and both with
//! a text layer. What the directory holds, what the manifest declares, what
//! an older reader would meet, and what a tokenizer the engine cannot write
//! down does to a load.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use zeusdb_vector_core::{compile_filter, Error, IdfScope, SparseVector, VectorIndex};
use zeusdb_vector_sparse::{SparseConfig, Weighting};
use zeusdb_vector_text::{SimpleTokenizer, Tokenizer, TokenizerConfig};

use super::{Collection, Declaration, ParsedRecord, SpaceConfig};

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
        sparse: sparse.map(|(dims, values)| SparseVector {
            dims: dims.to_vec(),
            values: values.to_vec(),
        }),
        metadata,
    }
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
    let dense: Page = dense.into_iter().map(|h| (h.0, h.1)).collect();
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

/// A dense-only directory keeps the flat names and the 1.1.0 version, and
/// its config.json carries no `spaces` field, so it is what 0.9.0 wrote.
#[test]
fn a_dense_only_directory_stays_at_the_first_major() {
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
    assert_eq!(m["format_version"], "1.1.0");
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

/// A collection with a sparse space writes `spaces/<name>/postings.zdbsparse`,
/// names it in the manifest by length alone, declares the space in
/// config.json by value, is a 2.0.0 directory, and reopens to the same
/// three pages.
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
    assert_eq!(m["format_version"], "2.0.0");
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
    let vectors = collection.vectorize_texts(&texts).unwrap();
    let records: Vec<ParsedRecord> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, sparse)| ParsedRecord {
            id: format!("t{i}"),
            vector: vec![i as f32, 1.0],
            sparse: Some(sparse),
            metadata: HashMap::new(),
        })
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
    assert_eq!(m["format_version"], "2.0.0");
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
    let more = loaded.vectorize_texts(&["zebra"]).unwrap();
    assert_eq!(more[0].dims, vec![terms as u32]);
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
    let vectors = collection
        .vectorize_texts(&["Alpha beta", "beta GAMMA"])
        .unwrap();
    let records: Vec<ParsedRecord> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, sparse)| ParsedRecord {
            id: format!("t{i}"),
            vector: vec![i as f32, 0.0],
            sparse: Some(sparse),
            metadata: HashMap::new(),
        })
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
/// manifest declaring a space is refused, a 3.x manifest is refused naming
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

    // A later major, refused with the majors this build reads.
    let future = dir.path().join("future.zdb");
    copy_dir(&path, &future);
    rewrite_manifest(&future, |m| m["format_version"] = json!("3.0.0"));
    let message = Collection::load(future.to_str().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("format version 3.0.0 cannot be opened"),
        "{message}"
    );
    assert!(message.contains("1.x and 2.x"), "{message}");
    assert!(message.contains("newer"), "{message}");

    // A 2.x label on a dense-only directory opens, since the minor and the
    // major are both read.
    let plain = Collection::build(base(), None);
    let plain_path = dir.path().join("plain.zdb");
    plain.save(plain_path.to_str().unwrap()).unwrap();
    rewrite_manifest(&plain_path, |m| m["format_version"] = json!("2.3.0"));
    assert!(Collection::load(plain_path.to_str().unwrap()).is_ok());

    // A space declared under a name the collection refuses.
    let renamed = dir.path().join("renamed.zdb");
    copy_dir(&path, &renamed);
    let mut c = config(&renamed);
    c["spaces"][0]["name"] = json!("default");
    std::fs::write(
        renamed.join("config.json"),
        serde_json::to_string_pretty(&c).unwrap(),
    )
    .unwrap();
    let mut m = manifest(&renamed);
    m["file_digests"]
        .as_object_mut()
        .unwrap()
        .remove("config.json");
    std::fs::write(
        renamed.join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
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
    let vectors = collection
        .vectorize_texts(&["one two", "two three", "three four"])
        .unwrap();
    let records: Vec<ParsedRecord> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, sparse)| ParsedRecord {
            id: format!("t{i}"),
            vector: vec![i as f32, 0.0],
            sparse: Some(sparse),
            metadata: HashMap::new(),
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let dir = TempDir::new();
    let path = dir.path().join("text.zdb");
    collection.save(path.to_str().unwrap()).unwrap();

    // The postings from a save before a record was removed, over mappings
    // from after it. The middle record, so the largest live id still
    // admits the artefact's slot count and the check that fires is the one
    // on the record itself.
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
    fewer.vectorize_texts(&["one two"]).unwrap();
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

/// The graph's rebuild fallback replays every record through `add`, which
/// reissues internal ids. The sparse space rides through it: the loaded
/// collection answers the same sparse page under the new ids.
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
    // Without its dump the loader rebuilds the graph from the records.
    std::fs::remove_file(path.join("hnsw_index.zdbgraph")).unwrap();
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 60);
    let after = pages(&loaded);
    assert_eq!(after.1, before.1);
    assert_eq!(after.2, before.2);
    assert_eq!(loaded.sparse().unwrap().index.read().unwrap().len(), 40);
    debug_assert!(loaded.live_sets_agree());
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
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let vectors = collection.vectorize_texts(&refs).unwrap();
    let records: Vec<ParsedRecord> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, sparse)| ParsedRecord {
            id: format!("t{i}"),
            vector: vec![i as f32, 0.0],
            sparse: Some(sparse),
            metadata: HashMap::new(),
        })
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
    assert_eq!(m["format_version"], "2.0.0");
    assert!(cleared.join("spaces/text/postings.zdbsparse").exists());
    let loaded = Collection::load(cleared.to_str().unwrap()).unwrap();
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.term_count(), Some(0));
    let vectors = loaded.vectorize_texts(&["fresh start"]).unwrap();
    assert_eq!(vectors[0].dims, vec![0, 1]);
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
