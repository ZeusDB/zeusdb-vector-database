//! The numbering between positions and nodes, and the sparse space's keys,
//! held through every path that changes a collection.
//!
//! A record's position names it in every per-record structure and a node
//! names it in the graph. The dense index keeps each position's node and the
//! graph keeps each node's position, and the sparse space keys each posting
//! by its record's position. These tests run a collection through every path
//! that writes either numbering, being additions, removals, overwrites, the
//! training transition, a journal's replay, compaction, a load from the dump,
//! a load through a rebuild and a clear, and check the numbering after each.
//! They run once with the counter as churn leaves it, so the records' internal
//! ids sit far past their positions, and once from a counter at zero, and the
//! two answer every query alike.

use std::collections::HashMap;

use serde_json::{json, Value};
use zeusdb_vector_core::{compile_filter, IdfScope, SparseVector};
use zeusdb_vector_sparse::SparseConfig;

use super::{
    Arm, Collection, Declaration, Int8Scale, ParsedRecord, Query, SparseHalf, StorageMode,
};
use crate::journal::Durability;

/// A directory under the system's temporary directory, removed on drop
/// together with the journals beside anything in it.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-numbering-tests-{}-{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn at(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fields(i: u32) -> HashMap<String, Value> {
    HashMap::from([
        ("cat".to_string(), json!(["a", "b", "c"][(i % 3) as usize])),
        ("i".to_string(), json!(i)),
    ])
}

fn wide_records(vectors: &[Vec<f32>], ids: impl IntoIterator<Item = usize>) -> Vec<ParsedRecord> {
    ids.into_iter()
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: vectors[i].clone(),
            sparse: None,
            metadata: fields(i as u32),
        })
        .collect()
}

fn remove(collection: &Collection, records: impl Iterator<Item = u32>, gone: impl Fn(u32) -> bool) {
    let ids: Vec<String> = records
        .filter(|&i| gone(i))
        .map(|i| format!("r{i}"))
        .collect();
    let missing = collection.remove_points(&ids).unwrap();
    assert!(missing.is_empty(), "{missing:?}");
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn reload(collection: &Collection, temp: &TempDir, name: &str) -> Collection {
    let path = temp.at(name);
    collection.save(path.to_str().unwrap()).unwrap();
    let (loaded, recovery) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert!(
        !recovery.graph_rebuilt,
        "{name}: the graph was rebuilt rather than read from its dump"
    );
    loaded
}

/// The dense index's positions, its node for each, and the graph's position
/// for each node agree with each other and with the id store, through every
/// path that changes the graph or the records, on a raw, a product quantized
/// and a scalar quantized collection.
#[test]
fn the_live_nodes_follow_every_path_that_changes_the_graph_or_the_live_set() {
    let temp = TempDir::new();
    let vectors = zeusdb_vector_core::test_support::clustered(1_700, 16, 196);
    let put = |collection: &Collection, ids: Vec<usize>, overwrite: bool| {
        let added = collection.add_records(wide_records(&vectors, ids), vec![], overwrite);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    };
    for kind in ["raw", "pq", "int8"] {
        let declaration =
            Declaration::validate(16, "l2", 8, 40, 100, vec!["cat".to_string()]).unwrap();
        let quantization = match kind {
            "pq" => Some(
                declaration
                    .quantization(4, 4, 1_000, None, StorageMode::QuantizedWithRaw)
                    .unwrap(),
            ),
            "int8" => Some(
                declaration
                    .scalar_quantization(
                        Int8Scale::PER_DIMENSION,
                        1_000,
                        None,
                        StorageMode::QuantizedOnly,
                    )
                    .unwrap(),
            ),
            _ => None,
        };
        let mut collection = Collection::build(declaration, quantization);
        // The counter as churn leaves it, so every record's internal id sits
        // far past its position.
        collection.set_counters(200_000, 0);
        let path = temp.at(&format!("numbering-{kind}.zdb"));
        collection
            .journal_to(path.to_str().unwrap(), Durability::default())
            .unwrap();
        let agree = |collection: &Collection, step: &str| {
            assert!(collection.positions_agree(), "{kind}: {step}");
        };

        put(&collection, (0..600).collect(), false);
        agree(&collection, "built");
        collection.checkpoint().unwrap();
        remove(&collection, 0..600, |i| i % 3 != 0);
        agree(&collection, "removed");
        put(&collection, (0..90).step_by(3).collect(), true);
        agree(&collection, "overwritten");
        put(&collection, (600..1_500).collect(), false);
        agree(&collection, "added past the training set");
        assert_eq!(collection.is_quantized(), kind != "raw", "{kind}");
        drop(collection);

        let (recovered, report) =
            Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
        assert!(report.replayed > 0 && !report.graph_rebuilt, "{kind}");
        agree(&recovered, "replayed");
        recovered.compact().unwrap();
        agree(&recovered, "compacted");
        recovered.checkpoint().unwrap();
        drop(recovered);

        // The checkpoint copied without its journal, which is the whole of
        // the collection once the journal is cut back to it.
        let copy = temp.at(&format!("numbering-{kind}-copy.zdb"));
        copy_tree(&path, &copy);
        let loaded = Collection::load_checkpoint_only(copy.to_str().unwrap(), None).unwrap();
        agree(&loaded, "loaded");
        remove(&loaded, 600..1_500, |i| i % 5 == 0);
        put(&loaded, (1_500..1_600).collect(), false);
        put(&loaded, (601..700).step_by(5).collect(), true);
        agree(&loaded, "loaded then changed");
        loaded.compact().unwrap();
        agree(&loaded, "loaded, changed and compacted");
        let loaded = reload(&loaded, &temp, &format!("numbering-{kind}-again.zdb"));
        agree(&loaded, "loaded again");

        let rebuilt_path = temp.at(&format!("numbering-{kind}-rebuilt.zdb"));
        loaded.save(rebuilt_path.to_str().unwrap()).unwrap();
        super::journal_support::break_topology(&rebuilt_path);
        let rebuilt = Collection::load(rebuilt_path.to_str().unwrap()).unwrap();
        agree(&rebuilt, "loaded through a rebuild");
        remove(&rebuilt, 1_500..1_600, |i| i % 2 == 0);
        agree(&rebuilt, "rebuilt then removed");

        rebuilt.clear().unwrap();
        agree(&rebuilt, "cleared");
        put(&rebuilt, (0..50).collect(), false);
        agree(&rebuilt, "cleared then added");
    }
}

/// One query's page, as external ids and score bits.
type Page = Vec<(String, u32)>;

/// The sparse arm's pages and the fused query's, with and without a filter.
fn sparse_pages(collection: &Collection, dense: &[Vec<f32>]) -> Vec<Page> {
    let filter = compile_filter(&HashMap::from([("cat".to_string(), json!("b"))])).unwrap();
    let mut out: Vec<Page> = Vec::new();
    for q in 0..3u32 {
        let query = SparseVector {
            dims: vec![q, 9 + q, 30 + q],
            values: vec![1.0, 2.0, 1.0],
        };
        for filter in [None, Some(&filter)] {
            let hits = collection
                .search_sparse(query.as_ref(), filter, 10, IdfScope::Corpus)
                .unwrap();
            out.push(
                hits.iter()
                    .map(|(id, score)| (id.clone(), score.to_bits()))
                    .collect(),
            );
            let arms = [
                Arm::Dense {
                    vector: &dense[q as usize],
                    ef: None,
                    rerank: None,
                },
                Arm::Sparse {
                    vector: query.as_ref(),
                    idf: IdfScope::Corpus,
                },
            ];
            let mut fused = Query::new(&arms, 10);
            fused.filter = filter;
            let page = collection.query(&fused).unwrap();
            out.push(
                page.hits
                    .iter()
                    .map(|hit| (hit.id.clone(), hit.score.to_bits()))
                    .collect(),
            );
        }
    }
    out
}

/// Every posting is keyed by a position the sparse space holds a record at,
/// in increasing order on every list, through every path that changes the
/// sparse space, and a collection whose internal ids are scattered answers
/// every sparse and fused query what one whose ids are consecutive
/// answers.
#[test]
fn the_sparse_keys_follow_every_path_that_changes_the_sparse_space() {
    let temp = TempDir::new();
    let vectors = zeusdb_vector_core::test_support::clustered(1_700, 16, 197);
    let sparse_of = |i: usize| -> SparseHalf {
        let mut dims: Vec<u32> = (0..5).map(|j| ((i * 7 + j * 11) % 64) as u32).collect();
        dims.sort_unstable();
        dims.dedup();
        let values = dims.iter().map(|&d| 1.0 + (d % 4) as f32).collect();
        SparseHalf::Vector(SparseVector { dims, values })
    };
    let put = |collection: &Collection, ids: Vec<usize>, overwrite: bool| {
        let records: Vec<ParsedRecord> = ids
            .into_iter()
            .map(|i| ParsedRecord {
                id: format!("r{i}"),
                vector: vectors[i].clone(),
                sparse: Some(sparse_of(i)),
                metadata: fields(i as u32),
            })
            .collect();
        let added = collection.add_records(records, vec![], overwrite);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    };
    for kind in ["raw", "pq"] {
        let mut runs: Vec<Vec<(String, Vec<Page>)>> = Vec::new();
        for spread in [true, false] {
            let declaration = Declaration::validate(16, "l2", 8, 40, 100, vec!["cat".to_string()])
                .unwrap()
                .with_sparse(
                    "terms",
                    SparseConfig {
                        weighting: zeusdb_vector_sparse::Weighting::BM25,
                        ..SparseConfig::default()
                    },
                )
                .unwrap();
            let quantization = (kind == "pq").then(|| {
                declaration
                    .quantization(4, 4, 1_000, None, StorageMode::QuantizedWithRaw)
                    .unwrap()
            });
            let mut collection = Collection::build(declaration, quantization);
            if spread {
                collection.set_counters(200_000, 0);
            }
            let path = temp.at(&format!("sparse-keys-{kind}-{spread}.zdb"));
            collection
                .journal_to(path.to_str().unwrap(), Durability::default())
                .unwrap();
            let mut steps: Vec<(String, Vec<Page>)> = Vec::new();
            let mut step = |collection: &Collection, name: &str| {
                {
                    let index = collection.sparse().unwrap().index.read().unwrap();
                    assert!(index.keys_agree(), "{kind} {spread}: {name}");
                }
                assert!(collection.positions_agree(), "{kind} {spread}: {name}");
                steps.push((name.to_string(), sparse_pages(collection, &vectors)));
            };

            put(&collection, (0..600).collect(), false);
            step(&collection, "built");
            collection.checkpoint().unwrap();
            remove(&collection, 0..599, |i| i % 3 != 0);
            step(&collection, "removed");
            put(&collection, (0..90).step_by(3).collect(), true);
            step(&collection, "overwritten");
            put(&collection, (600..1_500).collect(), false);
            step(&collection, "added past the training set");
            assert_eq!(collection.is_quantized(), kind != "raw", "{kind}");
            drop(collection);

            let (recovered, report) =
                Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
            assert!(report.replayed > 0 && !report.graph_rebuilt, "{kind}");
            step(&recovered, "replayed");
            recovered.compact().unwrap();
            step(&recovered, "compacted");
            recovered.checkpoint().unwrap();
            drop(recovered);

            let copy = temp.at(&format!("sparse-keys-{kind}-{spread}-copy.zdb"));
            copy_tree(&path, &copy);
            let loaded = Collection::load_checkpoint_only(copy.to_str().unwrap(), None).unwrap();
            step(&loaded, "loaded");
            remove(&loaded, 600..1_499, |i| i % 5 == 0);
            put(&loaded, (1_500..1_600).collect(), false);
            put(&loaded, (601..700).step_by(5).collect(), true);
            step(&loaded, "loaded then changed");
            loaded.compact().unwrap();
            step(&loaded, "loaded, changed and compacted");
            let loaded = reload(
                &loaded,
                &temp,
                &format!("sparse-keys-{kind}-{spread}-again.zdb"),
            );
            step(&loaded, "loaded again");

            let rebuilt_path = temp.at(&format!("sparse-keys-{kind}-{spread}-rebuilt.zdb"));
            loaded.save(rebuilt_path.to_str().unwrap()).unwrap();
            super::journal_support::break_topology(&rebuilt_path);
            let rebuilt = Collection::load(rebuilt_path.to_str().unwrap()).unwrap();
            step(&rebuilt, "loaded through a rebuild");
            remove(&rebuilt, 1_500..1_599, |i| i % 2 == 0);
            step(&rebuilt, "rebuilt then removed");

            rebuilt.clear().unwrap();
            step(&rebuilt, "cleared");
            put(&rebuilt, (0..50).collect(), false);
            step(&rebuilt, "cleared then added");
            runs.push(steps);
        }
        for ((name, spread), (_, dense)) in runs[0].iter().zip(&runs[1]) {
            assert_eq!(spread, dense, "{kind}: {name}");
        }
    }
}
