//! A collection holding a dense space and a sparse space at once.
//!
//! What the binding cannot reach yet, exercised here: a record that fills
//! both spaces, a sparse search under the collection's own admit sets, a
//! removal that leaves both, a compaction that reclaims the sparse space's
//! dead postings, and the refusals at the doors.

use std::collections::HashMap;

use serde_json::{json, Value};
use zeusdb_vector_core::{compile_filter, Error, SpaceName, SparseRef, SparseVector};
use zeusdb_vector_sparse::SparseConfig;

use super::{Collection, Declaration, ParsedRecord, SpaceConfig, DEFAULT_SPACE};

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
    let page = collection.search_sparse(query.as_ref(), None, 10).unwrap();
    assert_eq!(page, brute(&records, query.as_ref(), |_| true));
    assert_eq!(page[0].0, "r2");

    // Under a filter over a declared field, which is a bitmap the index
    // scans, and under one over an undeclared field, which is a walk.
    let declared = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
    let page = collection
        .search_sparse(query.as_ref(), Some(&declared), 10)
        .unwrap();
    assert_eq!(
        page,
        brute(&records, query.as_ref(), |r| r.metadata["cat"] == "a")
    );
    let undeclared =
        compile_filter(&HashMap::from([("missing".to_string(), json!(null))])).unwrap();
    let page = collection
        .search_sparse(query.as_ref(), Some(&undeclared), 10)
        .unwrap();
    assert!(page.is_empty());

    // A short page, since one record shares the query's only term.
    let rare = SparseVector {
        dims: vec![7],
        values: vec![1.0],
    };
    let page = collection.search_sparse(rare.as_ref(), None, 10).unwrap();
    assert_eq!(page.len(), 2);

    // A removal leaves both arms, and the sparse space counts what it left
    // behind until a compaction takes it back.
    assert!(collection.remove_point("r2".to_string()).unwrap());
    assert!(!collection.contains("r2"));
    let page = collection.search_sparse(query.as_ref(), None, 10).unwrap();
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
    let page = collection.search_sparse(query.as_ref(), None, 10).unwrap();
    assert_eq!(page, brute(&live, query.as_ref(), |_| true));

    // Clearing empties both.
    assert_eq!(collection.clear().unwrap(), 4);
    assert!(collection
        .search_sparse(query.as_ref(), None, 10)
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
        collection.search_sparse(query.as_ref(), None, 10),
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
        .search_sparse(old.as_ref(), None, 10)
        .unwrap()
        .is_empty());
    assert_eq!(
        collection.search_sparse(new.as_ref(), None, 10).unwrap()[0].0,
        "r1"
    );
    assert_eq!(collection.compact().unwrap(), 1);
    assert_eq!(collection.stats()["stranded_graph_nodes"], "0");
}
