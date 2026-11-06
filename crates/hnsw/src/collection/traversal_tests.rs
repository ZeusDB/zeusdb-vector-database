//! A search through a graph whose nodes are mostly stranded, through every
//! path that reaches the traversal.
//!
//! What is held. Once a removal leaves most of a graph's nodes stranded, an
//! unfiltered search still returns the live records nearest the query, as
//! many as it asks for where the collection holds as many, at every width.
//! At a width above the twenty records held the index scores the records it
//! holds, and at a width below them it traverses. The page is the page the
//! exact scan gives for the same query under a filter admitting exactly the
//! live records, ids and score bits both, where the index scores with the
//! scan's kernel or the page is reranked against the raw vectors, and is as
//! long where it does not. The single search, the batch run in turn, the
//! batch fanned across workers and a query's dense arm each take the same
//! path, on a raw, a scalar quantized and two product quantized collections,
//! and the plan a query reports, `explain`'s among them, names that path as
//! exact or not.

use std::collections::HashMap;

use serde_json::{json, Value};

use zeusdb_vector_core::{compile_filter, test_support::clustered, Filter};

use super::{Arm, Collection, Declaration, ParsedRecord, Query, QueryHits, StorageMode};

/// Records held, and the one record in fifty-one each search is asked
/// about. Record `i` sits in cluster `i % 50` of the fixture, so the records
/// kept sit in twenty clusters, one in each.
const RECORDS: usize = 1_000;
const DIM: usize = 8;

fn kept(i: usize) -> bool {
    i % 51 == 2
}

fn declaration() -> Declaration {
    Declaration::validate(DIM, "l2", 8, 60, 4_000, vec!["keep".to_string()]).unwrap()
}

/// A collection of `kind` holding the fixture, with every record not kept
/// removed, so its graph keeps the removed records' nodes stranded.
fn stranded(kind: &str, vectors: &[Vec<f32>]) -> Collection {
    let declared = declaration();
    let quantization = match kind {
        "raw" => None,
        "int8" => Some(
            declared
                .scalar_quantization("per_dimension", 1_000, None, StorageMode::QuantizedOnly)
                .unwrap(),
        ),
        "pq with raw" => Some(
            declared
                .quantization(4, 4, 1_000, None, StorageMode::QuantizedWithRaw)
                .unwrap(),
        ),
        _ => Some(
            declared
                .quantization(4, 4, 1_000, None, StorageMode::QuantizedOnly)
                .unwrap(),
        ),
    };
    let collection = Collection::build(declared, quantization);
    let records: Vec<ParsedRecord> = vectors
        .iter()
        .enumerate()
        .map(|(i, vector)| {
            let mut metadata: HashMap<String, Value> = HashMap::new();
            metadata.insert("keep".to_string(), json!(if kept(i) { "y" } else { "n" }));
            ParsedRecord {
                id: format!("r{i:04}"),
                vector: vector.clone(),
                sparse: None,
                metadata,
            }
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(collection.is_quantized(), kind != "raw", "{kind}");
    let removed: Vec<String> = (0..RECORDS)
        .filter(|&i| !kept(i))
        .map(|i| format!("r{i:04}"))
        .collect();
    assert!(
        collection.remove_points(&removed).unwrap().is_empty(),
        "every id removed was held"
    );
    collection
}

/// One page as ids and score bits.
fn bits(page: &QueryHits) -> Vec<(String, u32)> {
    page.iter()
        .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
        .collect()
}

/// Twenty live records among a thousand nodes, each search asked for ten
/// at a live record's own vector: every path returns ten live records at
/// every width, and at the default width and above the page is the exact
/// scan's where the kinds score alike. The single search, the two batch
/// paths and the dense arm return one page, and the dense arm's plan is
/// exact at every width above the twenty records held.
#[test]
fn a_search_through_stranded_nodes_returns_the_live_records_nearest_the_query() {
    let vectors = clustered(RECORDS, DIM, 0x0223);
    let live: Vec<usize> = (0..RECORDS).filter(|&i| kept(i)).collect();
    assert_eq!(live.len(), 20);
    let keep: Filter = compile_filter(&HashMap::from([("keep".to_string(), json!("y"))])).unwrap();
    for kind in ["raw", "int8", "pq with raw", "pq only"] {
        let collection = stranded(kind, &vectors);
        assert_eq!(collection.len(), live.len(), "{kind}");
        // The traversal scores with the scan's kernel on a raw and a scalar
        // graph, and a product quantized graph keeping raw vectors reranks
        // its page against them, so those three pages are the scan's.
        let scored_as_scanned = kind != "pq only";
        let queries: Vec<Vec<f32>> = live
            .iter()
            .map(|&i| collection.process_vector_for_space(vectors[i].clone()))
            .collect();
        for ef in [None, Some(10), Some(400)] {
            let params = collection.search_params(10, ef, false, None).unwrap();
            let exact: Vec<Vec<(String, u32)>> = queries
                .iter()
                .map(|query| {
                    let page = collection.search_one(query, Some(&keep), params).unwrap();
                    bits(&page)
                })
                .collect();
            let single: Vec<Vec<(String, u32)>> = queries
                .iter()
                .map(|query| bits(&collection.search_one(query, None, params).unwrap()))
                .collect();
            let in_turn: Vec<Vec<(String, u32)>> = collection
                .search_batch(&queries[..5], None, params)
                .unwrap()
                .iter()
                .map(bits)
                .collect();
            let fanned: Vec<Vec<(String, u32)>> = collection
                .search_batch(&queries, None, params)
                .unwrap()
                .iter()
                .map(bits)
                .collect();
            let arm: Vec<Vec<(String, u32)>> = queries
                .iter()
                .map(|query| {
                    let arms = [Arm::Dense {
                        vector: query,
                        ef,
                        rerank: None,
                    }];
                    let asked = Query::new(&arms, 10);
                    let page = collection.query(&asked).unwrap();
                    assert_eq!(
                        collection.explain(&asked).unwrap(),
                        page.plan,
                        "{kind} width {ef:?}"
                    );
                    assert_eq!(
                        page.plan.arms[0].cost.exact,
                        ef != Some(10),
                        "{kind} width {ef:?}: the plan names the path the dense arm took"
                    );
                    page.hits
                        .iter()
                        .map(|hit| (hit.id.clone(), hit.score.to_bits()))
                        .collect()
                })
                .collect();
            for (q, &i) in live.iter().enumerate() {
                let label = format!("{kind} query r{i:04} width {ef:?}");
                assert_eq!(exact[q].len(), 10, "{label}");
                for (path, page) in [
                    ("single", &single[q]),
                    ("fanned", &fanned[q]),
                    ("arm", &arm[q]),
                ] {
                    assert_eq!(page.len(), 10, "{label} {path}");
                    assert!(
                        page.iter().all(|(id, _)| kept(id[1..].parse().unwrap())),
                        "{label} {path}"
                    );
                    if scored_as_scanned && ef != Some(10) {
                        assert_eq!(page, &exact[q], "{label} {path}");
                    }
                }
                if q < 5 {
                    assert_eq!(in_turn[q], fanned[q], "{label} in turn");
                }
                assert_eq!(single[q], fanned[q], "{label}");
                assert_eq!(single[q], arm[q], "{label}");
            }
        }
    }
}
