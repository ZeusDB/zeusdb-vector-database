//! A collection whose internal ids spread over a range wider than its records.
//!
//! Internal ids are never reused, so removals, overwrites and a counter that
//! churn has raised leave a collection's records under ids spread over a
//! range wider than the records themselves. Every per-record structure is
//! indexed by position, and positions run from zero in internal id order, so
//! a structure reaches the records held and the holes since the last
//! close-up, and after a compaction or a load the records alone. These tests
//! hold a collection of each shape its ids reach to what its records cost,
//! and hold two collections that differ in their declared size alone to
//! answering and writing the same, before and after a load and as both are
//! changed.

use std::collections::HashMap;

use serde_json::{json, Value};
use zeusdb_vector_core::{compile_filter, IdfScope, Operation, SparseVector, DUMP_FILENAME};
use zeusdb_vector_sparse::SparseConfig;

use super::{Collection, Declaration, ParsedRecord, SparseHalf};
use crate::journal::Durability;

/// A directory under the system's temporary directory, removed on drop
/// together with the journals beside anything in it.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-id-range-tests-{}-{}",
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

/// Four values a vector at `m` 4, the field `cat` declared, `expected`
/// records declared, and a sparse space where `sparse` is set.
fn declaration(expected: usize, sparse: bool) -> Declaration {
    let declaration =
        Declaration::validate(4, "l2", 4, 40, expected, vec!["cat".to_string()]).unwrap();
    if sparse {
        declaration
            .with_sparse("terms", SparseConfig::default())
            .unwrap()
    } else {
        declaration
    }
}

fn vector(i: u32) -> Vec<f32> {
    let x = i as f32;
    vec![
        (x * 0.37).sin(),
        (x * 0.11).cos(),
        (x * 0.23).sin(),
        (x * 0.07).cos(),
    ]
}

fn fields(i: u32) -> HashMap<String, Value> {
    HashMap::from([
        ("cat".to_string(), json!(["a", "b", "c"][(i % 3) as usize])),
        ("i".to_string(), json!(i)),
    ])
}

/// A record whose vector, fields and sparse half follow from `i`.
fn record(i: u32, sparse: bool) -> ParsedRecord {
    let sparse = sparse.then(|| {
        let mut dims: Vec<u32> = (0..3).map(|j| (i * 7 + j * 13) % 40).collect();
        dims.sort_unstable();
        dims.dedup();
        let values = dims.iter().map(|d| 1.0 + (d % 3) as f32).collect();
        SparseHalf::Vector(SparseVector { dims, values })
    });
    ParsedRecord {
        id: format!("r{i}"),
        vector: vector(i),
        sparse,
        metadata: fields(i),
    }
}

fn add(collection: &Collection, records: impl Iterator<Item = u32>, sparse: bool) {
    let records: Vec<ParsedRecord> = records.map(|i| record(i, sparse)).collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

/// Remove every record of `records` that `gone` picks.
fn remove(collection: &Collection, records: impl Iterator<Item = u32>, gone: impl Fn(u32) -> bool) {
    let ids: Vec<String> = records
        .filter(|&i| gone(i))
        .map(|i| format!("r{i}"))
        .collect();
    let missing = collection.remove_points(&ids).unwrap();
    assert!(missing.is_empty(), "{missing:?}");
}

/// Every record the collection holds, by its external and internal id.
fn ids_of(collection: &Collection) -> Vec<(String, usize)> {
    let mut ids: Vec<(String, usize)> = collection
        .internal_ids()
        .into_iter()
        .map(|(internal, id)| (id, internal))
        .collect();
    ids.sort();
    ids
}

/// How far each per-record structure reaches: the id store, the dense
/// index's `node_at`, the metadata, the columns, and the sparse space where
/// there is one. Each guard is taken alone.
#[derive(Debug, PartialEq)]
struct Extents {
    ids: usize,
    node_at: usize,
    metadata: usize,
    columns: usize,
    sparse: Option<usize>,
}

fn extents(collection: &Collection) -> Extents {
    let ids = collection.ids().extent();
    let node_at = collection.dense().index.read().unwrap().node_at().len();
    let sparse = collection
        .sparse()
        .map(|space| space.index.read().unwrap().extent());
    let metadata = collection.vector_metadata.read().unwrap().extent();
    let columns = collection.columns.read().unwrap().extent();
    Extents {
        ids,
        node_at,
        metadata,
        columns,
        sparse,
    }
}

/// Every structure reaching `records` positions, the sparse space's too
/// where there is one.
fn reach(records: usize, sparse: bool) -> Extents {
    Extents {
        ids: records,
        node_at: records,
        metadata: records,
        columns: records,
        sparse: sparse.then_some(records),
    }
}

/// Bytes the structures that hold a record's entries hold: the id store,
/// the dense index's `node_at`, the graph's links, the sparse space's
/// tables and sets, the metadata store and the columns. Each guard is taken
/// alone.
fn keyed_bytes(collection: &Collection) -> usize {
    let ids = collection.ids().heap_bytes();
    let (numbering, links) = {
        let index = collection.dense().index.read().unwrap();
        (
            index.node_at_heap_bytes(),
            index.graph().links_memory_bytes(),
        )
    };
    let sparse = collection.sparse().map_or(0, |space| {
        let heap = space.index.read().unwrap().heap_bytes();
        heap.records + heap.lengths + heap.dead
    });
    let metadata = collection.vector_metadata.read().unwrap().heap_bytes();
    let columns = collection.columns.read().unwrap().heap_bytes();
    ids + numbering + links + sparse + metadata + columns
}

/// One page, as an external id and score bits per hit.
type Page = Vec<(String, u32)>;

/// The pages a collection answers: six queries unfiltered and filtered on
/// the declared field, and two on the sparse space where there is one.
fn pages(collection: &Collection) -> Vec<Page> {
    let filter = compile_filter(&HashMap::from([("cat".to_string(), json!("b"))])).unwrap();
    let mut out: Vec<Page> = Vec::new();
    for q in 0..6u32 {
        let query = vector(90_000 + q);
        for filter in [None, Some(&filter)] {
            let params = collection.search_params(10, None, false, None).unwrap();
            let hits = collection.search_one(&query, filter, params).unwrap();
            out.push(
                hits.iter()
                    .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                    .collect(),
            );
        }
    }
    if collection.sparse().is_some() {
        let query = SparseVector {
            dims: vec![1, 14, 27],
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
        }
    }
    out
}

/// The artefacts a save writes that name records by internal id, being the
/// graph dump, the id map, the vectors, the metadata and the sparse postings
/// where there are any, as their bytes.
fn saved(collection: &Collection, temp: &TempDir, name: &str) -> Vec<(String, Vec<u8>)> {
    let path = temp.at(name);
    collection.save(path.to_str().unwrap()).unwrap();
    [
        DUMP_FILENAME,
        "mappings.bin",
        "vectors.bin",
        "metadata.json",
        "spaces/terms/postings.zdbsparse",
    ]
    .iter()
    .filter(|file| path.join(file).exists())
    .map(|file| (file.to_string(), std::fs::read(path.join(file)).unwrap()))
    .collect()
}

/// A save of `collection` read back, its graph from its dump.
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

/// A collection whose ids are dense holds every per-record structure at its
/// records as built, at the positions its records reached after removals
/// that leave half of them, and at its records again after a compaction and
/// after a save and a load, past its declaration and within it, with a
/// sparse space and without, and within a declaration far past its records.
/// The removals keep the newest record, which a sparse space's artefact needs
/// to load.
#[test]
fn a_collection_over_dense_ids_holds_every_structure_at_its_records() {
    let temp = TempDir::new();
    for (expected, records, sparse) in [
        (100, 5_000, false),
        (6_000, 5_000, true),
        (100_000, 600, false),
    ] {
        let collection = Collection::build(declaration(expected, sparse), None);
        add(&collection, 0..records, sparse);
        assert_eq!(
            extents(&collection),
            reach(records as usize, sparse),
            "as built"
        );
        remove(&collection, 0..records, |i| i % 2 == 0);
        assert_eq!(
            extents(&collection),
            reach(records as usize, sparse),
            "half removed, the holes kept"
        );
        collection.compact().unwrap();
        let held = records as usize / 2;
        assert_eq!(extents(&collection), reach(held, sparse), "compacted");
        let loaded = reload(&collection, &temp, &format!("dense-{expected}.zdb"));
        assert_eq!(extents(&loaded), reach(held, sparse), "loaded");
        assert_eq!(pages(&loaded), pages(&collection));
        assert_eq!(ids_of(&loaded), ids_of(&collection));
    }
}

/// A collection that took 12,000 records a hundred at a time and kept every
/// twentieth, the newest among them, compacted, holds every per-record
/// structure at its 600 records and costs what they cost: within 32 KiB of a
/// collection holding the same records under dense ids, where a slot per id
/// up to the largest would hold twenty slots a record. It answers as many
/// pages as before the compaction, reads its dump back, holds its structures
/// at its records through a load, and writes the same artefacts again.
#[test]
fn a_sparse_range_after_removals_and_a_compaction_costs_what_its_records_cost() {
    let temp = TempDir::new();
    for sparse in [false, true] {
        let thinned = Collection::build(declaration(100, sparse), None);
        for batch in 0..120u32 {
            let records = batch * 100..(batch + 1) * 100;
            add(&thinned, records.clone(), sparse);
            remove(&thinned, records, |i| i % 20 != 19);
        }
        let before = pages(&thinned);
        thinned.compact().unwrap();
        let held = Collection::build(declaration(100, sparse), None);
        add(&held, (0..12_000).filter(|i| i % 20 == 19), sparse);

        assert_eq!(extents(&thinned), reach(600, sparse));
        assert_eq!(extents(&held), reach(600, sparse));
        let (bytes, dense) = (keyed_bytes(&thinned), keyed_bytes(&held));
        assert!(
            bytes < dense + 32 * 1024,
            "{bytes} bytes against {dense} for the same records under dense ids"
        );
        assert_eq!(pages(&thinned)[0].len(), 10);
        assert_eq!(pages(&thinned).len(), before.len());

        let first = saved(&thinned, &temp, &format!("thinned-{sparse}.zdb"));
        let loaded = reload(&thinned, &temp, &format!("thinned-load-{sparse}.zdb"));
        assert_eq!(extents(&loaded), reach(600, sparse), "loaded");
        assert!(keyed_bytes(&loaded) < dense + 32 * 1024);
        assert_eq!(pages(&loaded), pages(&thinned));
        assert_eq!(ids_of(&loaded), ids_of(&thinned));
        assert_eq!(
            saved(&loaded, &temp, &format!("thinned-again-{sparse}.zdb")),
            first
        );
    }
}

/// One record left under a large id by churn through the public calls,
/// records added a hundred at a time and every one removed but the newest,
/// the graph compacted as it goes, costs what one record costs: every
/// per-record structure reaches one position, at a few kilobytes where a
/// slot per id would cost over 250 kilobytes. A search finds it, a load
/// reads it back to the same extents and answers, and its removal and the
/// records after it follow from the counter.
#[test]
fn a_single_record_under_a_large_id_costs_what_one_record_costs() {
    let temp = TempDir::new();
    let last = 8_000u32;
    let id = last as usize;
    for sparse in [false, true] {
        let collection = Collection::build(declaration(100, sparse), None);
        for batch in 0..last / 100 {
            let records = batch * 100..(batch + 1) * 100;
            add(&collection, records.clone(), sparse);
            remove(&collection, records, |i| i != last - 1);
            if batch % 20 == 19 {
                collection.compact().unwrap();
            }
        }
        assert_eq!(ids_of(&collection), vec![(format!("r{}", last - 1), id)]);
        assert_eq!(extents(&collection), reach(1, sparse));
        assert!(
            keyed_bytes(&collection) < 64 * 1024,
            "{}",
            keyed_bytes(&collection)
        );
        assert_eq!(pages(&collection)[0].len(), 1, "the one record answers");

        let loaded = reload(&collection, &temp, &format!("one-{sparse}.zdb"));
        assert_eq!(extents(&loaded), reach(1, sparse), "loaded");
        assert!(keyed_bytes(&loaded) < 64 * 1024);
        assert_eq!(pages(&loaded), pages(&collection));

        assert!(loaded.remove_point(format!("r{}", last - 1)).unwrap());
        add(&loaded, last..last + 2, sparse);
        assert_eq!(
            ids_of(&loaded),
            vec![
                (format!("r{last}"), id + 1),
                (format!("r{}", last + 1), id + 2)
            ]
        );
        let again = reload(&loaded, &temp, &format!("one-again-{sparse}.zdb"));
        assert_eq!(extents(&again), reach(2, sparse));
        assert_eq!(pages(&again), pages(&loaded));
        assert_eq!(again.id_counter(), id + 2);
    }
}

/// A collection that removes every record keeps its counter, and once
/// compacted holds every per-record structure at nothing. The records it
/// takes next carry on from the counter, take the first positions and cost
/// what they cost, through a save and a load.
#[test]
fn every_record_removed_with_the_counter_kept_then_new_records() {
    let temp = TempDir::new();
    for sparse in [false, true] {
        let collection = Collection::build(declaration(100, sparse), None);
        add(&collection, 0..5_000, sparse);
        remove(&collection, 0..5_000, |_| true);
        collection.compact().unwrap();
        assert_eq!(collection.id_counter(), 5_000);
        assert!(ids_of(&collection).is_empty());
        assert_eq!(extents(&collection), reach(0, sparse));
        let empty = keyed_bytes(&collection);

        add(&collection, 10_000..10_050, sparse);
        let ids = ids_of(&collection);
        assert_eq!(ids.len(), 50);
        assert!(ids
            .iter()
            .all(|(_, internal)| (5_001..=5_050).contains(internal)));
        assert_eq!(extents(&collection), reach(50, sparse));
        assert!(keyed_bytes(&collection) < empty + 64 * 1024);

        let loaded = reload(&collection, &temp, &format!("emptied-{sparse}.zdb"));
        assert_eq!(ids_of(&loaded), ids);
        assert_eq!(pages(&loaded), pages(&collection));
        assert_eq!(loaded.id_counter(), 5_050);
        assert_eq!(extents(&loaded), reach(50, sparse));
    }
}

/// Two collections that differ in their declared size alone answer and
/// write the same, operation for operation, as built, after a save and a
/// load of each, and as both are then changed by inserts at recorded levels,
/// a removal, an overwrite, a compaction and a further save and load. The
/// declaration sets what each structure reserves and nothing it holds, so
/// one reserves far less than its records reach and the other far more.
#[test]
fn a_collection_answers_and_writes_alike_whatever_its_declared_size() {
    let temp = TempDir::new();
    let small = Collection::build(declaration(100, false), None);
    let large = Collection::build(declaration(13_000, false), None);
    for collection in [&small, &large] {
        add(collection, 0..12_000, false);
        remove(collection, 0..12_000, |i| i % 10 != 0);
        collection.compact().unwrap();
        assert_eq!(extents(collection), reach(1_200, false));
    }

    let mut step = 0usize;
    let mut same = |small: &Collection, large: &Collection, label: &str| {
        step += 1;
        assert_eq!(pages(small), pages(large), "{label}");
        assert_eq!(ids_of(small), ids_of(large), "{label}");
        assert_eq!(extents(small), extents(large), "{label}");
        let a = saved(small, &temp, &format!("small-{step}.zdb"));
        let b = saved(large, &temp, &format!("large-{step}.zdb"));
        assert_eq!(a.len(), b.len(), "{label}");
        for ((name, a), (_, b)) in a.iter().zip(&b) {
            assert!(a == b, "{label}: {name} differs");
        }
    };
    same(&small, &large, "as built");

    let small = reload(&small, &temp, "small.zdb");
    let large = reload(&large, &temp, "large.zdb");
    same(&small, &large, "after a load");

    let apply = |operation: Operation| {
        small.apply(operation.clone()).unwrap();
        large.apply(operation).unwrap();
    };
    let mut issued = small.id_counter();
    assert_eq!(large.id_counter(), issued);
    let insert = |issued: usize, id: String, k: u32| Operation::Insert {
        id,
        internal_id: issued as u64,
        level: (k * 3 % 5) as u8,
        vector: vector(20_000 + k),
        metadata: fields(k).into_iter().collect(),
        sparse: None,
    };
    for k in 0..60u32 {
        issued += 1;
        apply(insert(issued, format!("n{k}"), k));
    }
    same(&small, &large, "after inserts at recorded levels");

    apply(Operation::Remove {
        ids: (0..12_000u32)
            .filter(|i| i % 30 == 0)
            .map(|i| format!("r{i}"))
            .chain(["n3".to_string(), "n17".to_string()])
            .collect(),
    });
    apply(Operation::Remove {
        ids: vec!["r10".to_string()],
    });
    issued += 1;
    apply(insert(issued, "r10".to_string(), 77));
    same(&small, &large, "after a removal and an overwrite");

    apply(Operation::Compact);
    same(&small, &large, "after a compaction");

    let small = reload(&small, &temp, "small-again.zdb");
    let large = reload(&large, &temp, "large-again.zdb");
    same(&small, &large, "after a further save and load");
}

/// A journal replays into a collection whose ids are sparse: the removals
/// that thin it, the compaction, new records under ids past the gap, an
/// overwrite and a metadata update all replay onto its checkpoint, and the
/// recovered collection holds the same records, answers the same pages,
/// counts from the same counter and holds its structures at the same
/// positions. A second checkpoint of the sparse collection and a second
/// journal replay the same way.
#[test]
fn a_journal_replays_into_a_collection_whose_ids_are_sparse() {
    let temp = TempDir::new();
    let path = temp.at("journaled.zdb");
    let collection = Collection::build(declaration(100, false), None);
    collection
        .journal_to(path.to_str().unwrap(), Durability::default())
        .unwrap();
    add(&collection, 0..5_000, false);
    collection.checkpoint().unwrap();
    remove(&collection, 0..5_000, |i| i % 25 != 0);
    collection.compact().unwrap();
    add(&collection, 6_000..6_040, false);
    collection
        .update_metadata("r25", HashMap::from([("cat".to_string(), json!("z"))]))
        .unwrap();
    remove(&collection, [50u32].into_iter(), |_| true);
    add(&collection, 50..51, false);
    assert_eq!(
        extents(&collection),
        reach(241, false),
        "one hole before the last"
    );
    let before = (
        ids_of(&collection),
        pages(&collection),
        collection.id_counter(),
        extents(&collection),
    );
    drop(collection);

    let (recovered, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(
        report.replayed, 45,
        "a removal, a compaction, forty inserts, a metadata update, a removal and an insert"
    );
    assert!(!report.graph_rebuilt);
    assert_eq!(
        (
            ids_of(&recovered),
            pages(&recovered),
            recovered.id_counter(),
            extents(&recovered)
        ),
        before
    );

    recovered.checkpoint().unwrap();
    add(&recovered, 7_000..7_020, false);
    remove(&recovered, [7_005u32].into_iter(), |_| true);
    let before = (
        ids_of(&recovered),
        pages(&recovered),
        recovered.id_counter(),
    );
    drop(recovered);
    let (again, report) =
        Collection::recover(path.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.replayed, 21);
    assert!(!report.graph_rebuilt);
    assert_eq!((ids_of(&again), pages(&again), again.id_counter()), before);
    // The checkpoint loaded with its records at the first positions, and the
    // replay appended after them with the one hole its removal left.
    assert_eq!(extents(&again), reach(240 + 20, false));
}
