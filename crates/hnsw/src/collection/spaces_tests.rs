//! A collection holding a dense space and a sparse space at once.
//!
//! What the binding cannot reach yet, exercised here: a record that fills
//! both spaces, a sparse search under the collection's own admit sets, a
//! removal that leaves both, a compaction that reclaims the sparse space's
//! dead postings, and the refusals at the doors.

use std::collections::HashMap;

use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use zeusdb_vector_core::{compile_filter, Error, IdfScope, SpaceName, SparseRef, SparseVector};
use zeusdb_vector_sparse::{SparseConfig, Unlink, Weighting};
use zeusdb_vector_text::{SimpleTokenizer, Tokenizer, TokenizerConfig};

use super::{
    Arm, Collection, Declaration, ParsedRecord, Query, SpaceConfig, TextConfig, DEFAULT_SPACE,
};

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

fn collection() -> Collection {
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()])
        .unwrap()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    Collection::build(declaration, None)
}

/// Every record's sparse dot product against the query, by brute force over
/// the vectors the test inserted, best first and ties by external id.
fn brute(
    records: &[ParsedRecord],
    query: SparseRef<'_>,
    admit: impl Fn(&ParsedRecord) -> bool,
) -> Vec<(String, f32)> {
    let mut page: Vec<(String, f32)> = records
        .iter()
        .filter(|r| admit(r))
        .filter_map(|r| {
            let score = r.sparse.as_ref()?.as_ref().dot(query);
            (score != 0.0).then(|| (r.id.clone(), score))
        })
        .collect();
    page.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    page
}

#[test]
fn a_collection_holds_both_arms_and_answers_each() {
    let collection = collection();
    let names: Vec<(SpaceName, SpaceConfig)> = collection.space_configs();
    assert_eq!(names.len(), 2);
    assert_eq!(names[0].0.as_str(), DEFAULT_SPACE);
    assert!(matches!(names[0].1, SpaceConfig::Dense(_)));
    assert_eq!(names[1].0.as_str(), "terms");
    assert!(matches!(names[1].1, SpaceConfig::Sparse(_)));

    let records = vec![
        record("r1", &[0.0, 0.0], Some((&[1, 5], &[1.0, 2.0])), "a"),
        record("r2", &[1.0, 0.0], Some((&[1, 7], &[3.0, 1.0])), "b"),
        record("r3", &[0.0, 1.0], Some((&[5, 7], &[1.0, 1.0])), "a"),
        record("r4", &[2.0, 2.0], None, "b"),
        record("r5", &[3.0, 0.0], Some((&[1], &[0.5])), "a"),
    ];
    let added = collection.add_records(records.clone(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(added.inserted.len(), 5);
    assert_eq!(collection.len(), 5);

    // The dense arm is untouched: the nearest to the origin is r1.
    let params = collection.search_params(2, None, false, None).unwrap();
    let dense = collection.search_one(&[0.0, 0.0], None, params).unwrap();
    assert_eq!(dense[0].0, "r1");

    // The sparse arm under the live set.
    let query = SparseVector {
        dims: vec![1, 7],
        values: vec![1.0, 1.0],
    };
    let page = collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    assert_eq!(page, brute(&records, query.as_ref(), |_| true));
    assert_eq!(page[0].0, "r2");

    // Under a filter over a declared field, which is a bitmap the index
    // scans, and under one over an undeclared field, which is a walk.
    let declared = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
    let page = collection
        .search_sparse(query.as_ref(), Some(&declared), 10, IdfScope::Corpus)
        .unwrap();
    assert_eq!(
        page,
        brute(&records, query.as_ref(), |r| r.metadata["cat"] == "a")
    );
    let undeclared =
        compile_filter(&HashMap::from([("missing".to_string(), json!(null))])).unwrap();
    let page = collection
        .search_sparse(query.as_ref(), Some(&undeclared), 10, IdfScope::Corpus)
        .unwrap();
    assert!(page.is_empty());

    // A short page, since one record shares the query's only term.
    let rare = SparseVector {
        dims: vec![7],
        values: vec![1.0],
    };
    let page = collection
        .search_sparse(rare.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    assert_eq!(page.len(), 2);

    // A removal leaves both arms, and the sparse space counts what it left
    // behind until a compaction takes it back.
    assert!(collection.remove_point("r2".to_string()).unwrap());
    assert!(!collection.contains("r2"));
    let page = collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    assert!(page.iter().all(|(id, _)| id != "r2"));
    let live: Vec<ParsedRecord> = records.iter().filter(|r| r.id != "r2").cloned().collect();
    assert_eq!(page, brute(&live, query.as_ref(), |_| true));
    let dense = collection.search_one(&[1.0, 0.0], None, params).unwrap();
    assert!(dense.iter().all(|hit| hit.0 != "r2"));

    let reclaimed = collection.compact().unwrap();
    assert_eq!(reclaimed, 1, "the stranded graph node");
    assert_eq!(
        collection
            .sparse()
            .unwrap()
            .index
            .read()
            .unwrap()
            .dead_records(),
        0
    );
    let page = collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    assert_eq!(page, brute(&live, query.as_ref(), |_| true));

    // Clearing empties both.
    assert_eq!(collection.clear().unwrap(), 4);
    assert!(collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap()
        .is_empty());
    assert_eq!(collection.len(), 0);
}

#[test]
fn a_record_refused_for_its_sparse_half_leaves_nothing_behind() {
    let collection = collection();
    let bad = record("r1", &[0.0, 0.0], Some((&[5, 1], &[1.0, 1.0])), "a");
    let added = collection.add_records(vec![bad], vec![], false);
    assert_eq!(added.total_errors, 1);
    assert_eq!(added.inserted.len(), 0);
    assert!(
        added.errors[0]
            .starts_with("Vector r1: ValueError: Sparse vector dims must be strictly increasing"),
        "{}",
        added.errors[0]
    );
    assert!(!collection.contains("r1"));
    assert_eq!(collection.len(), 0);

    // The weighting's own rule is applied at the same door, so a value at
    // or below zero under term frequency weighting leaves nothing behind
    // either, and the graph holds no node for it.
    let weighted = Collection::build(
        Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()])
            .unwrap()
            .with_sparse(
                "terms",
                SparseConfig {
                    weighting: Weighting::BM25,
                    ..SparseConfig::default()
                },
            )
            .unwrap(),
        None,
    );
    let zero = record("r1", &[0.0, 0.0], Some((&[1, 5], &[1.0, 0.0])), "a");
    let added = weighted.add_records(vec![zero], vec![], false);
    assert_eq!(added.total_errors, 1);
    assert_eq!(
        added.errors[0],
        "Vector r1: ValueError: Sparse vector value at index 1 is 0, and a space weighted by term frequency takes values above zero alone"
    );
    assert!(!weighted.contains("r1"));
    assert_eq!(weighted.len(), 0);
    assert_eq!(weighted.stats()["graph_nodes"], "0");
    assert_eq!(weighted.stats()["sparse_records"], "0");
}

#[test]
fn a_sparse_vector_without_a_sparse_space_is_refused() {
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap();
    let collection = Collection::build(declaration, None);
    assert!(collection.sparse().is_none());
    let added = collection.add_records(
        vec![record("r1", &[0.0, 0.0], Some((&[1], &[1.0])), "a")],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 1);
    assert_eq!(
        added.errors[0],
        "Vector r1: ValueError: This collection declares no sparse space"
    );
    let query = SparseVector {
        dims: vec![1],
        values: vec![1.0],
    };
    assert!(matches!(
        collection.search_sparse(query.as_ref(), None, 10, IdfScope::Corpus),
        Err(Error::NoSparseSpace)
    ));
}

#[test]
fn the_declaration_refuses_the_default_name_and_a_second_sparse_space() {
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap();
    assert!(matches!(
        declaration.clone().with_sparse("", SparseConfig::default()),
        Err(Error::SpaceNameEmpty)
    ));
    assert!(matches!(
        declaration
            .clone()
            .with_sparse(DEFAULT_SPACE, SparseConfig::default()),
        Err(Error::SpaceDeclaredTwice { .. })
    ));
    let once = declaration
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    assert!(matches!(
        once.with_sparse("more", SparseConfig::default()),
        Err(Error::SpacesTooMany { max: 2 })
    ));
}

/// The dense index's live set follows the collection's through an
/// overwrite, which is a removal and an insertion under a fresh id.
#[test]
fn an_overwrite_keeps_the_live_sets_in_step() {
    let collection = collection();
    let first = vec![record("r1", &[0.0, 0.0], Some((&[1], &[1.0])), "a")];
    collection.add_records(first, vec![], false);
    let second = vec![record("r1", &[5.0, 5.0], Some((&[9], &[2.0])), "b")];
    let added = collection.add_records(second, vec![], true);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(collection.len(), 1);
    let stats = collection.stats();
    assert_eq!(stats["graph_nodes"], "2");
    assert_eq!(stats["stranded_graph_nodes"], "1");
    let old = SparseVector {
        dims: vec![1],
        values: vec![1.0],
    };
    let new = SparseVector {
        dims: vec![9],
        values: vec![1.0],
    };
    assert!(collection
        .search_sparse(old.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap()
        .is_empty());
    assert_eq!(
        collection
            .search_sparse(new.as_ref(), None, 10, IdfScope::Corpus)
            .unwrap()[0]
            .0,
        "r1"
    );
    assert_eq!(collection.compact().unwrap(), 1);
    assert_eq!(collection.stats()["stranded_graph_nodes"], "0");
}

/// A text space counts each record's text into the sparse space, reports
/// its tokenizer by value, searches by text under the collection's admit
/// sets with the term frequency weighting, refuses text where it has no
/// layer, and starts its dictionary again on `clear`.
#[test]
fn a_text_space_indexes_and_searches_text() {
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()])
        .unwrap()
        .with_text(
            "text",
            SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            Arc::new(SimpleTokenizer),
        )
        .unwrap();
    let collection = Collection::build(declaration, None);
    let configs = collection.space_configs();
    assert_eq!(
        configs[1].1,
        SpaceConfig::Text(TextConfig {
            index: SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            tokenizer: TokenizerConfig::Simple,
        })
    );

    let texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fox is a small wild dog",
        "The dog sleeps",
        "Nothing here about animals",
    ];
    let vectors = collection.vectorize_texts(&texts).unwrap();
    assert_eq!(collection.term_count(), Some(17));
    let records: Vec<ParsedRecord> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, sparse)| {
            let mut r = record(&format!("r{}", i + 1), &[i as f32, 0.0], None, "a");
            r.sparse = Some(sparse);
            r
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);

    // "fox" is in r1 and r2, "dog" in r1, r2 and r3. r2 is the shortest of
    // those carrying both, so it leads, then r1, then r3 on "dog" alone.
    let page = collection
        .search_text("Fox, DOG!", None, 10, IdfScope::Corpus)
        .unwrap();
    let ids: Vec<&str> = page.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["r2", "r1", "r3"]);
    assert!(page[0].1 > page[1].1 && page[1].1 > page[2].1);

    // A term no record carries is dropped, and a text of none is an empty
    // page.
    assert_eq!(
        collection
            .search_text("zebra", None, 10, IdfScope::Corpus)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(collection.term_count(), Some(17), "a query issues no id");

    // The same page by term ids through the pre-tokenized path, which is
    // the same search.
    let query = {
        let space = collection.sparse().unwrap();
        let text = space.text.as_ref().unwrap();
        let dictionary = text.dictionary.read().unwrap();
        SparseVector {
            dims: vec![dictionary.id_of("dog").unwrap()],
            values: vec![1.0],
        }
    };
    let by_ids = collection
        .search_sparse(query.as_ref(), None, 10, IdfScope::Corpus)
        .unwrap();
    let by_text = collection
        .search_text("dog", None, 10, IdfScope::Corpus)
        .unwrap();
    assert_eq!(by_ids, by_text);

    // Under a filter over a declared field the rarity is counted over the
    // admitted records by default and over every record on request.
    let declared = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
    let filtered = collection
        .search_text("dog", Some(&declared), 10, IdfScope::Corpus)
        .unwrap();
    let global = collection
        .search_text("dog", Some(&declared), 10, IdfScope::Global)
        .unwrap();
    assert_eq!(filtered.len(), 3);
    assert_eq!(
        filtered, global,
        "every record is admitted, so the two agree"
    );

    assert_eq!(collection.clear().unwrap(), 4);
    assert_eq!(collection.term_count(), Some(0));
}

#[test]
fn text_is_refused_where_the_sparse_space_takes_term_ids_alone() {
    let collection = collection();
    assert!(matches!(
        collection.vectorize_texts(&["a"]),
        Err(Error::NoTextLayer)
    ));
    assert!(matches!(
        collection.search_text("a", None, 10, IdfScope::Corpus),
        Err(Error::NoTextLayer)
    ));
    assert_eq!(collection.term_count(), None);
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap();
    assert!(matches!(
        declaration.with_sparse(
            "terms",
            SparseConfig {
                weighting: Weighting::Bm25 { k1: 1.2, b: 1.5 },
                ..SparseConfig::default()
            }
        ),
        Err(Error::SparseWeightingInvalid { parameter: "b", .. })
    ));
}

/// A text arm and a terms arm over the same text answer the same page, the
/// collection's `tokenize` hands over what the layer's tokenizer does, and
/// `vectorize_terms` over those terms is `vectorize_texts` over the text.
#[test]
fn a_text_arm_and_a_terms_arm_answer_the_same_page() {
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![])
        .unwrap()
        .with_text(
            "text",
            SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            Arc::new(SimpleTokenizer),
        )
        .unwrap();
    let collection = Collection::build(declaration, None);
    let texts = [
        "The quick brown fox",
        "a lazy DOG",
        "the fox and the dog",
        "quick quick",
    ];
    let by_text = collection.vectorize_texts(&texts).unwrap();
    let terms: Vec<Vec<String>> = texts
        .iter()
        .map(|text| collection.tokenize(text).unwrap())
        .collect();
    assert_eq!(terms[1], ["a", "lazy", "dog"]);
    // The same terms counted again issue no id and give the same vectors.
    let by_terms = collection.vectorize_terms(&terms).unwrap();
    assert_eq!(by_terms, by_text);
    assert_eq!(collection.term_count(), Some(8));

    let records: Vec<ParsedRecord> = by_text
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

    let query_terms = collection.tokenize("Quick fox!").unwrap();
    assert_eq!(query_terms, ["quick", "fox"]);
    let text_arm = [Arm::Text {
        text: "Quick fox!",
        idf: IdfScope::Corpus,
    }];
    let terms_arm = [Arm::Terms {
        terms: &query_terms,
        idf: IdfScope::Corpus,
    }];
    assert_eq!(terms_arm[0].kind(), zeusdb_vector_core::SpaceKind::Sparse);
    let by_text = collection.query(&Query::new(&text_arm, 5)).unwrap();
    let by_terms = collection.query(&Query::new(&terms_arm, 5)).unwrap();
    assert_eq!(by_text, by_terms);
    assert_eq!(by_text.hits[0].id, "t0");
    assert_eq!(by_text.hits.len(), 3);
    // A query issues no id, and an unknown term is dropped.
    assert_eq!(collection.term_count(), Some(8));
    let unknown = [Arm::Terms {
        terms: &["zebra".to_string()],
        idf: IdfScope::Corpus,
    }];
    assert!(collection
        .query(&Query::new(&unknown, 5))
        .unwrap()
        .hits
        .is_empty());
    let dense_only = Collection::build(
        Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    );
    assert!(matches!(
        dense_only.query(&Query::new(&terms_arm, 5)),
        Err(Error::NoSparseSpace)
    ));
    assert!(matches!(
        dense_only.tokenize("x"),
        Err(Error::NoSparseSpace)
    ));
    assert!(matches!(
        dense_only.vectorize_terms(&[vec!["x".to_string()]]),
        Err(Error::NoSparseSpace)
    ));
}

/// The tokenizer runs under no guard. A tokenizer that reads the collection
/// while it runs takes the dictionary's guard itself, which would deadlock
/// or trip the registry if the collection held that guard around it.
#[test]
fn the_tokenizer_runs_under_no_guard() {
    struct Probing {
        collection: OnceLock<Weak<Collection>>,
        calls: AtomicUsize,
    }
    impl Tokenizer for Probing {
        fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
            if let Some(collection) = self.collection.get().and_then(Weak::upgrade) {
                // The dictionary's read guard, taken inside the tokenizer.
                let _ = collection.term_count();
                // And the whole of `stats`, which takes every guard in turn.
                let _ = collection.stats();
            }
            self.calls.fetch_add(1, Ordering::Relaxed);
            text.split_whitespace().for_each(term);
            Ok(())
        }
    }
    let probe = Arc::new(Probing {
        collection: OnceLock::new(),
        calls: AtomicUsize::new(0),
    });
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![])
        .unwrap()
        .with_text("text", SparseConfig::default(), probe.clone())
        .unwrap();
    let collection = Arc::new(Collection::build(declaration, None));
    probe
        .collection
        .set(Arc::downgrade(&collection))
        .ok()
        .unwrap();

    let vectors = collection
        .vectorize_texts(&["alpha beta", "beta gamma"])
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
    let page = collection
        .search_text("gamma", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(probe.calls.load(Ordering::Relaxed), 3);
}

/// A tokenizer's own failure comes back from every door it can be reached
/// through, as the tokenizer returned it.
#[test]
fn a_failing_tokenizer_reaches_the_caller() {
    struct Broken;
    impl Tokenizer for Broken {
        fn tokenize(&self, text: &str, _term: &mut dyn FnMut(&str)) -> Result<(), Error> {
            Err(Error::TokenizerFailed(
                format!("cannot split {text:?}").into(),
            ))
        }
    }
    let declaration = Declaration::validate(2, "l2", 4, 50, 100, vec![])
        .unwrap()
        .with_text("text", SparseConfig::default(), Arc::new(Broken))
        .unwrap();
    let collection = Collection::build(declaration, None);
    for result in [
        collection.tokenize("a b").map(|_| ()),
        collection.vectorize_texts(&["a b"]).map(|_| ()),
        collection
            .search_text("a b", None, 5, IdfScope::Corpus)
            .map(|_| ()),
    ] {
        match result {
            Err(Error::TokenizerFailed(inner)) => {
                assert_eq!(inner.to_string(), "cannot split \"a b\"")
            }
            other => panic!("expected the tokenizer's failure, got {other:?}"),
        }
    }
    assert_eq!(
        collection.tokenize("a b").unwrap_err().to_string(),
        "The tokenizer raised cannot split \"a b\""
    );
    assert_eq!(
        collection.tokenize("x").unwrap_err().exception(),
        zeusdb_vector_core::Exception::Runtime
    );
    // Terms already split need no tokenizer, so they still count.
    assert!(collection.vectorize_terms(&[vec!["a".to_string()]]).is_ok());
}

/// A declaration read back by name takes the defaults for what it leaves
/// out, the names it takes are the ones `config.json` writes, and the
/// weighting's name agrees with its tag.
#[test]
fn a_declaration_by_name_takes_its_defaults() {
    let named: SparseConfig =
        serde_json::from_value(json!({"weighting": {"type": "bm25"}})).unwrap();
    assert_eq!(
        named,
        SparseConfig {
            weighting: Weighting::BM25,
            ..SparseConfig::default()
        }
    );
    let empty: SparseConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, SparseConfig::default());
    let full: SparseConfig = serde_json::from_value(json!({
        "unlink": "eager",
        "lazy_threshold_percent": 25,
        "weighting": {"type": "bm25", "k1": 1.5, "b": 0.6}
    }))
    .unwrap();
    assert_eq!(full.unlink, Unlink::Eager);
    assert_eq!(full.lazy_threshold_percent, 25);
    assert_eq!(full.weighting, Weighting::Bm25 { k1: 1.5, b: 0.6 });
    // What the file writes is what the name reads.
    let written = serde_json::to_value(&full).unwrap();
    assert_eq!(
        serde_json::from_value::<SparseConfig>(written).unwrap(),
        full
    );
    for weighting in [Weighting::Dot, Weighting::BM25] {
        assert_eq!(
            serde_json::to_value(weighting).unwrap()["type"],
            json!(weighting.name())
        );
    }
    assert!(
        serde_json::from_value::<SparseConfig>(json!({"weighting": {"type": "bm26"}})).is_err()
    );
    assert!(serde_json::from_value::<SparseConfig>(json!({"unlink": "sometimes"})).is_err());
    assert_eq!(
        serde_json::from_value::<TokenizerConfig>(json!("simple")).unwrap(),
        TokenizerConfig::Simple
    );
}
