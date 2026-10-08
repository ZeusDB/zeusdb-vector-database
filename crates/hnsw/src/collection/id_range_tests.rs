//! A collection whose internal ids spread over a range wider than its records.
//!
//! Internal ids are never reused, so removals, overwrites and a counter that
//! churn has raised leave a collection's records under ids spread over a
//! range wider than the records themselves. Every structure keyed by internal
//! id holds its entries flat while the ids are dense and in pages once they
//! are sparse. These tests hold a collection of each shape its ids reach to
//! what its records cost, and hold a collection whose maps are paged to
//! answering and writing what the same collection with flat maps answers and
//! writes, before and after a load and as both are changed.

use std::collections::HashMap;

use serde_json::{json, Value};
use zeusdb_vector_core::{compile_filter, IdfScope, Operation, SparseVector, DUMP_FILENAME};
use zeusdb_vector_sparse::SparseConfig;

use super::{Collection, Declaration, Int8Scale, ParsedRecord, SparseHalf, StorageMode};
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
    let store = collection.ids();
    let mut ids: Vec<(String, usize)> = store
        .iter()
        .map(|(internal, id)| (id.to_string(), internal))
        .collect();
    ids.sort();
    ids
}

/// Whether each structure keyed by internal id is flat: the id store's
/// entries and its live set, the metadata store, the columns with their live
/// set, the dense index's live set, the graph's id-to-node map, and the sparse
/// space where there is one. Each guard is taken alone.
#[derive(Debug, PartialEq)]
struct Forms {
    ids: bool,
    ids_live: bool,
    metadata: bool,
    columns: bool,
    dense_live: bool,
    id_map: bool,
    sparse: Option<bool>,
}

fn forms(collection: &Collection) -> Forms {
    let (ids, ids_live) = {
        let store = collection.ids();
        (store.is_flat(), store.live().is_flat())
    };
    let (dense_live, id_map) = {
        let index = collection.dense().index.read().unwrap();
        (index.live_set().is_flat(), index.graph().id_map_is_flat())
    };
    let sparse = collection
        .sparse()
        .map(|space| space.index.read().unwrap().is_flat());
    let metadata = collection.vector_metadata.read().unwrap().is_flat();
    let columns = collection.columns.read().unwrap().is_flat();
    Forms {
        ids,
        ids_live,
        metadata,
        columns,
        dense_live,
        id_map,
        sparse,
    }
}

/// Every structure flat, the sparse space's too where there is one.
fn all_flat(sparse: bool) -> Forms {
    Forms {
        ids: true,
        ids_live: true,
        metadata: true,
        columns: true,
        dense_live: true,
        id_map: true,
        sparse: sparse.then_some(true),
    }
}

/// Bytes the structures that hold a record's entries hold: the id store, the
/// dense index's live set, the graph's links, the sparse space's tables and
/// sets, the metadata store and the columns. Each guard is taken alone.
fn keyed_bytes(collection: &Collection) -> usize {
    let ids = collection.ids().heap_bytes();
    let (live, links) = {
        let index = collection.dense().index.read().unwrap();
        (index.live_heap_bytes(), index.graph().links_memory_bytes())
    };
    let sparse = collection.sparse().map_or(0, |space| {
        let heap = space.index.read().unwrap().heap_bytes();
        heap.records + heap.lengths + heap.dead
    });
    let metadata = collection.vector_metadata.read().unwrap().heap_bytes();
    let columns = collection.columns.read().unwrap().heap_bytes();
    ids + live + links + sparse + metadata + columns
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

/// A collection whose ids are dense keeps every structure flat, which is the
/// vector indexed by id each structure always held, as built, after removals
/// that leave half its records and a compaction, and after a save and a load,
/// past its declaration and within it, with a sparse space and without, and
/// within a declaration past one page that its records fill a small part of.
/// The removals keep the newest record, which a sparse space's artefact needs
/// to load.
#[test]
fn a_collection_over_dense_ids_keeps_every_structure_flat() {
    let temp = TempDir::new();
    for (expected, records, sparse) in [
        (100, 5_000, false),
        (6_000, 5_000, true),
        (100_000, 600, false),
    ] {
        let collection = Collection::build(declaration(expected, sparse), None);
        add(&collection, 0..records, sparse);
        assert_eq!(forms(&collection), all_flat(sparse), "as built");
        remove(&collection, 0..records, |i| i % 2 == 0);
        assert_eq!(forms(&collection), all_flat(sparse), "half removed");
        collection.compact().unwrap();
        assert_eq!(forms(&collection), all_flat(sparse), "compacted");
        let loaded = reload(&collection, &temp, &format!("dense-{expected}.zdb"));
        assert_eq!(forms(&loaded), all_flat(sparse), "loaded");
        assert_eq!(pages(&loaded), pages(&collection));
        assert_eq!(ids_of(&loaded), ids_of(&collection));
    }
}

/// A collection that took 12,000 records a hundred at a time and kept every
/// twentieth, the newest among them, compacted, holds its maps paged and
/// costs what its 600 records cost: within 32 KiB of a collection holding
/// the same records under dense ids, being the record sets, which stay flat
/// below one page, and two bytes an entry of a sparse page's offsets, where
/// a slot per id up to the largest would hold twenty slots a record.
/// It answers as many pages as before the compaction, reads its dump back,
/// keeps its form through a load, and writes the same artefacts again.
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

        let paged = Forms {
            ids: false,
            ids_live: true,
            metadata: false,
            columns: false,
            dense_live: true,
            id_map: false,
            sparse: sparse.then_some(false),
        };
        assert_eq!(forms(&thinned), paged);
        assert_eq!(forms(&held), all_flat(sparse));
        let (bytes, dense) = (keyed_bytes(&thinned), keyed_bytes(&held));
        assert!(
            bytes < dense + 32 * 1024,
            "{bytes} bytes against {dense} for the same records under dense ids"
        );
        assert_eq!(pages(&thinned)[0].len(), 10);
        assert_eq!(pages(&thinned).len(), before.len());

        let first = saved(&thinned, &temp, &format!("thinned-{sparse}.zdb"));
        let loaded = reload(&thinned, &temp, &format!("thinned-load-{sparse}.zdb"));
        assert_eq!(forms(&loaded), paged, "loaded");
        assert!(keyed_bytes(&loaded) < dense + 32 * 1024);
        assert_eq!(pages(&loaded), pages(&thinned));
        assert_eq!(ids_of(&loaded), ids_of(&thinned));
        assert_eq!(
            saved(&loaded, &temp, &format!("thinned-again-{sparse}.zdb")),
            first
        );
    }
}

/// One record left under an id past the maps' floor by churn through the
/// public calls, records added a hundred at a time and every one removed but
/// the newest, the graph compacted as it goes, costs what one record costs:
/// every map holds it paged and the record sets stay flat below one page, at a
/// few kilobytes where a slot per id would cost over 250 kilobytes. A search
/// finds it, a load reads it back to the same form and answers, and its
/// removal and the records after it follow from the counter.
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
        let paged = Forms {
            ids: false,
            ids_live: true,
            metadata: false,
            columns: false,
            dense_live: true,
            id_map: false,
            sparse: sparse.then_some(false),
        };
        assert_eq!(forms(&collection), paged);
        assert!(
            keyed_bytes(&collection) < 64 * 1024,
            "{}",
            keyed_bytes(&collection)
        );
        assert_eq!(pages(&collection)[0].len(), 1, "the one record answers");

        let loaded = reload(&collection, &temp, &format!("one-{sparse}.zdb"));
        assert_eq!(forms(&loaded), paged, "loaded");
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
        assert_eq!(forms(&again), paged);
        assert_eq!(pages(&again), pages(&loaded));
        assert_eq!(again.id_counter(), id + 2);
    }
}

/// A collection that removes every record keeps its counter, and once
/// compacted holds its maps paged with nothing in them. The records it takes
/// next carry on from the counter and cost what they cost, through a save and
/// a load.
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
        let emptied = forms(&collection);
        assert!(!emptied.ids && !emptied.metadata && !emptied.columns);
        let empty = keyed_bytes(&collection);

        add(&collection, 10_000..10_050, sparse);
        let ids = ids_of(&collection);
        assert_eq!(ids.len(), 50);
        assert!(ids
            .iter()
            .all(|(_, internal)| (5_001..=5_050).contains(internal)));
        assert!(
            !forms(&collection).id_map,
            "the new graph pages its map out"
        );
        assert!(keyed_bytes(&collection) < empty + 64 * 1024);

        let loaded = reload(&collection, &temp, &format!("emptied-{sparse}.zdb"));
        assert_eq!(ids_of(&loaded), ids);
        assert_eq!(pages(&loaded), pages(&collection));
        assert_eq!(loaded.id_counter(), 5_050);
        assert!(!forms(&loaded).ids && !forms(&loaded).metadata);
    }
}

/// A collection whose maps are paged answers and writes what the same
/// collection with flat maps answers and writes, operation for operation, as
/// built, after a save and a load of each, and as both are then changed by
/// inserts at recorded levels, a removal, an overwrite, a compaction and a
/// further save and load.
///
/// The two take the same operations at the same internal ids and differ in
/// their declared size alone, which keeps every map of the larger one flat
/// within its reservation while the smaller one's metadata and columns page
/// out. The smaller one's id store entries stay flat as built, since a
/// removal pages them out only below one record in twice their ratio, and a
/// load's plan pages them. Every record set stays flat in both, its extent
/// being under one page, and the graph's id-to-node map stays flat in both,
/// its records filling one id in ten, within its ratio.
#[test]
fn a_paged_collection_answers_and_writes_what_a_flat_one_does() {
    let temp = TempDir::new();
    let paged = Collection::build(declaration(100, false), None);
    let flat = Collection::build(declaration(13_000, false), None);
    for collection in [&paged, &flat] {
        add(collection, 0..12_000, false);
        remove(collection, 0..12_000, |i| i % 10 != 0);
        collection.compact().unwrap();
    }
    let maps_paged = Forms {
        ids: true,
        ids_live: true,
        metadata: false,
        columns: false,
        dense_live: true,
        id_map: true,
        sparse: None,
    };
    assert_eq!(forms(&paged), maps_paged);
    assert_eq!(forms(&flat), all_flat(false));

    let mut step = 0usize;
    let mut same = |paged: &Collection, flat: &Collection, label: &str| {
        step += 1;
        assert_eq!(pages(paged), pages(flat), "{label}");
        assert_eq!(ids_of(paged), ids_of(flat), "{label}");
        let a = saved(paged, &temp, &format!("paged-{step}.zdb"));
        let b = saved(flat, &temp, &format!("flat-{step}.zdb"));
        assert_eq!(a.len(), b.len(), "{label}");
        for ((name, a), (_, b)) in a.iter().zip(&b) {
            assert!(a == b, "{label}: {name} differs");
        }
    };
    same(&paged, &flat, "as built");

    let paged = reload(&paged, &temp, "paged.zdb");
    let flat = reload(&flat, &temp, "flat.zdb");
    assert_eq!(
        forms(&paged),
        Forms {
            ids: false,
            ..maps_paged
        },
        "loaded"
    );
    let loaded_flat = forms(&flat);
    assert!(loaded_flat.ids && loaded_flat.metadata && loaded_flat.columns);
    same(&paged, &flat, "after a load");

    let apply = |operation: Operation| {
        paged.apply(operation.clone()).unwrap();
        flat.apply(operation).unwrap();
    };
    let mut issued = paged.id_counter();
    assert_eq!(flat.id_counter(), issued);
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
    same(&paged, &flat, "after inserts at recorded levels");

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
    same(&paged, &flat, "after a removal and an overwrite");

    apply(Operation::Compact);
    same(&paged, &flat, "after a compaction");

    let paged = reload(&paged, &temp, "paged-again.zdb");
    let flat = reload(&flat, &temp, "flat-again.zdb");
    same(&paged, &flat, "after a further save and load");
}

/// A journal replays into a collection whose ids are sparse: the removals
/// that thin it, the compaction, new records under ids past the gap, an
/// overwrite and a metadata update all replay onto its checkpoint, and the
/// recovered collection holds the same records, answers the same pages,
/// counts from the same counter and holds its maps paged. A second checkpoint
/// of the sparse collection and a second journal replay the same way.
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
    assert!(!forms(&collection).ids && !forms(&collection).id_map);
    let before = (
        ids_of(&collection),
        pages(&collection),
        collection.id_counter(),
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
            recovered.id_counter()
        ),
        before
    );
    let replayed = forms(&recovered);
    assert!(!replayed.ids && !replayed.metadata && !replayed.columns && !replayed.id_map);

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
    assert!(!forms(&again).ids && !forms(&again).id_map);
}

/// A collection keeps its graph's id-to-node map flat while it holds one
/// record in sixteen ids or more and pages it below that, as built and
/// compacted and after a load: one record in eleven keeps it flat and one in
/// twenty pages it. Each answers the pages a collection holding the same
/// records with every map flat answers, its filtered searches scoring their
/// admitted records exactly, in increasing id order.
#[test]
fn a_collection_keeps_its_id_to_node_map_flat_to_its_ratio() {
    let temp = TempDir::new();
    for (every, flat_map) in [(11u32, true), (20, false)] {
        let thinned = Collection::build(declaration(100, false), None);
        let held = Collection::build(declaration(13_000, false), None);
        for collection in [&thinned, &held] {
            for batch in 0..120u32 {
                let records = batch * 100..(batch + 1) * 100;
                add(collection, records.clone(), false);
                remove(collection, records, |i| i % every != every - 1);
            }
            collection.compact().unwrap();
        }
        assert_eq!(forms(&thinned).id_map, flat_map, "one in {every}");
        assert_eq!(forms(&held), all_flat(false), "one in {every}, declared");
        assert_eq!(pages(&thinned), pages(&held), "one in {every}");
        let loaded = reload(&thinned, &temp, &format!("ratio-{every}.zdb"));
        assert_eq!(forms(&loaded).id_map, flat_map, "loaded, one in {every}");
        assert_eq!(pages(&loaded), pages(&held), "loaded, one in {every}");
    }
}

/// Records `ids` of `vectors`, each with the fields `fields` gives.
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

/// Copy the directory `from` to `to`, with every file and subdirectory.
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

/// The dense index's live set by node follows every path that changes the
/// graph or the live set, over ids past the record sets' first page, so the
/// live set by id is paged: insertions, removals that strand their nodes, an
/// overwrite, a journal replayed onto its checkpoint, a compaction, a load of
/// a checkpoint from its dump, a loaded collection then changed and loaded
/// again, a load that rebuilds the graph, and `clear`. A raw graph, a product
/// quantized graph and a scalar
/// quantized graph take the same steps, the two quantized ones training part
/// way through.
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
        // The counter as churn leaves it, so every record's id is past the
        // record sets' first page.
        collection.set_counters(200_000, 0);
        let path = temp.at(&format!("live-nodes-{kind}.zdb"));
        collection
            .journal_to(path.to_str().unwrap(), Durability::default())
            .unwrap();
        let agree = |collection: &Collection, step: &str| {
            assert!(collection.live_sets_agree(), "{kind}: {step}");
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
        assert!(
            !forms(&collection).dense_live,
            "{kind}: the live set by id pages"
        );
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
        let copy = temp.at(&format!("live-nodes-{kind}-copy.zdb"));
        copy_tree(&path, &copy);
        let loaded = Collection::load_checkpoint_only(copy.to_str().unwrap(), None).unwrap();
        agree(&loaded, "loaded");
        remove(&loaded, 600..1_500, |i| i % 5 == 0);
        put(&loaded, (1_500..1_600).collect(), false);
        put(&loaded, (601..700).step_by(5).collect(), true);
        agree(&loaded, "loaded then changed");
        loaded.compact().unwrap();
        agree(&loaded, "loaded, changed and compacted");
        let loaded = reload(&loaded, &temp, &format!("live-nodes-{kind}-again.zdb"));
        agree(&loaded, "loaded again");

        let rebuilt_path = temp.at(&format!("live-nodes-{kind}-rebuilt.zdb"));
        loaded.save(rebuilt_path.to_str().unwrap()).unwrap();
        std::fs::remove_file(rebuilt_path.join(DUMP_FILENAME)).unwrap();
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
