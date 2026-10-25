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
use zeusdb_vector_core::{
    compile_filter, IdfScope, Operation, SparseVector, DUMP_FILENAME, MAX_INTERNAL_ID,
};
use zeusdb_vector_sparse::SparseConfig;

use super::{Collection, Declaration, ParsedRecord, SparseHalf, StorageMode};
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

/// Bytes the structures by position hold, being the id store, the metadata,
/// the columns and the sparse space's tables and sets: `keyed_bytes` less
/// the graph's links, which follow its nodes, and less the dense index's
/// `node_at`, whose capacity a compaction takes from the replacement graph's
/// reservation for its nodes and a close-up on removal from the records and
/// the declared size. Each guard is taken alone.
fn by_position_bytes(collection: &Collection) -> usize {
    let (numbering, links) = {
        let index = collection.dense().index.read().unwrap();
        (
            index.node_at_heap_bytes(),
            index.graph().links_memory_bytes(),
        )
    };
    keyed_bytes(collection) - numbering - links
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
/// The removals take the newest record, so a sparse space's artefact records
/// a slot count past every id the mappings hold, which the load reads.
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
        remove(&collection, 0..records, |i| i % 2 == 1);
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

/// A collection that removes most of its records without compacting holds
/// every per-record structure within twice its records plus 4,096
/// positions. 12,000 records are removed in batches of one to 2,000 records,
/// by id and by batch, down to 600, and never compacted. After every batch
/// every structure reaches at most twice the records held or the records
/// held and 4,096, whichever is more, where a slot per id would keep 12,000,
/// and the positions have closed up on removal along the way. The structures
/// by position then cost the same collection's compacted bytes and at most
/// 64 bytes for each position past the records, the id store's table, sized
/// for the most records it held, being the same in both. `node_at` reaches
/// the positions, and its capacity past them is what `shrink_to_fit` gives
/// back.
#[test]
fn a_collection_that_removes_most_of_its_records_without_compacting_costs_what_its_records_cost() {
    let (records, kept) = (12_000u32, 600usize);
    // A permutation of the records, so each batch removes records spread
    // over the positions.
    let order: Vec<u32> = (0..records).map(|k| (k * 7_919) % records).collect();
    let sizes = [
        1usize, 2_000, 37, 1_500, 600, 3, 900, 250, 1_200, 1, 700, 64,
    ];
    for sparse in [false, true] {
        let collection = Collection::build(declaration(100, sparse), None);
        add(&collection, 0..records, sparse);
        let (mut at, mut batch, mut closed) = (0usize, 0usize, 0usize);
        while at < records as usize - kept {
            let size = sizes[batch % sizes.len()].min(records as usize - kept - at);
            batch += 1;
            let gone = &order[at..at + size];
            if size == 1 {
                assert!(collection.remove_point(format!("r{}", gone[0])).unwrap());
            } else {
                remove(&collection, gone.iter().copied(), |_| true);
            }
            at += size;
            let held = records as usize - at;
            let bound = (2 * held).max(held + 4_096);
            let reach = extents(&collection);
            for (name, extent) in [
                ("the id store", reach.ids),
                ("node_at", reach.node_at),
                ("the metadata", reach.metadata),
                ("the columns", reach.columns),
                ("the sparse space", reach.sparse.unwrap_or(0)),
            ] {
                assert!(
                    extent <= bound,
                    "{name} reaches {extent} positions for {held} records"
                );
            }
            if reach.ids == held {
                closed += 1;
            }
        }
        assert!(closed > 0, "the positions closed up on removal");
        let positions = extents(&collection).ids;
        assert!(positions <= (2 * kept).max(kept + 4_096));
        let bytes = by_position_bytes(&collection);
        let pages_before = pages(&collection);

        {
            let mut index = collection.dense().index.write().unwrap();
            assert_eq!(index.node_at().len(), positions);
            index.shrink_to_fit();
            assert_eq!(
                index.node_at_heap_bytes(),
                positions * std::mem::size_of::<u32>(),
                "node_at gives back its capacity past the positions"
            );
        }
        assert_eq!(pages(&collection), pages_before);

        collection.compact().unwrap();
        assert_eq!(extents(&collection), reach(kept, sparse), "compacted");
        let compacted = by_position_bytes(&collection);
        assert!(
            bytes <= compacted + 64 * (positions - kept),
            "{bytes} bytes against {compacted} compacted, for {positions} positions and {kept} \
             records"
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

/// A record the mappings lack is refused by the load's rebuild fallback,
/// from records and from codes alike, in the words the readers of the two
/// stores refuse one in, the least such id named, before the id store or the
/// counter is touched, so no record is filed under an internal id the
/// directory never gave it.
#[test]
fn the_rebuild_refuses_a_record_the_mappings_lack() {
    let positions_of = |collection: &Collection| -> Vec<u32> {
        let mut positions: Vec<u32> = ids_of(collection)
            .into_iter()
            .map(|(_, internal)| internal as u32)
            .collect();
        positions.sort_unstable();
        positions
    };
    let unmapped = |file: &str, id: &str| {
        format!(
            "Failed to deserialize {}: names record '{}', which mappings.bin does not hold",
            file, id
        )
    };

    // From records, two of them unmapped, the lesser named.
    let collection = Collection::build(declaration(100, false), None);
    add(&collection, 0..3, false);
    let positions = positions_of(&collection);
    let counter = collection.id_counter();
    let records = [0u32, 1, 2, 7, 3]
        .into_iter()
        .map(|i| (format!("r{i}"), vector(i)))
        .collect();
    let refused = collection
        .rebuild_from_records(records, &positions)
        .unwrap_err();
    assert_eq!(refused.to_string(), unmapped("vectors.bin", "r3"));
    assert_eq!(ids_of(&collection).len(), 3, "the id store was not touched");
    assert_eq!(collection.id_counter(), counter, "no id was issued");

    // From codes, on a trained product quantized collection, a code and a
    // raw vector each naming a record the mappings lack.
    let vectors = zeusdb_vector_core::test_support::clustered(1_010, 16, 0x0020_8b02);
    let declaration = Declaration::validate(16, "l2", 8, 40, 100, vec![]).unwrap();
    let quantization = declaration
        .quantization(4, 4, 1_000, None, StorageMode::QuantizedWithRaw)
        .unwrap();
    let mut collection = Collection::build(declaration, Some(quantization));
    let records: Vec<ParsedRecord> = (0..vectors.len())
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: vectors[i].clone(),
            sparse: None,
            metadata: HashMap::new(),
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(collection.is_quantized());
    let held = collection.dense().pq_codes.read().unwrap().clone();
    let positions = positions_of(&collection);
    let counter = collection.id_counter();
    for (code, raw, named) in [
        (Some("lost"), None, unmapped("pq_codes.bin", "lost")),
        (None, Some("astray"), unmapped("vectors.bin", "astray")),
        (
            Some("lost"),
            Some("astray"),
            unmapped("vectors.bin", "astray"),
        ),
    ] {
        let mut codes = held.clone();
        if let Some(id) = code {
            codes.insert(id.to_string(), held["r0"].clone());
        }
        let raws: HashMap<String, Vec<f32>> = raw
            .map(|id| (id.to_string(), vectors[0].clone()))
            .into_iter()
            .collect();
        let refused = collection
            .rebuild_graph_from_codes(&codes, &raws, &positions)
            .unwrap_err();
        assert_eq!(refused.to_string(), named);
        assert_eq!(
            ids_of(&collection).len(),
            held.len(),
            "the id store was not touched"
        );
        assert_eq!(collection.id_counter(), counter, "no id was issued");
    }
}

/// A version 1 journal's insert past the last id, which a release before
/// 0.12.0 issued and its id store then refused, is passed over by the
/// replay where the counter stands at the last id, between two held
/// records and with a clear after it, so the replay holds the records and
/// the counter that release held. The same insert is refused from a version
/// 2 journal, whose writer refuses such an id before issuing it, and from a
/// version 1 journal whose counter stands below the last id, since no
/// release issued one there.
#[test]
fn a_version_1_insert_past_the_last_id_is_passed_over_at_the_last_id() {
    let insert = |id: &str, internal_id: u64, i: u32| Operation::Insert {
        id: id.to_string(),
        internal_id,
        level: 0,
        vector: vector(i),
        metadata: serde_json::Map::new(),
        sparse: None,
    };
    let last = MAX_INTERNAL_ID as u64;
    let mut collection = Collection::build(declaration(100, false), None);
    collection.set_counters(MAX_INTERNAL_ID - 1, 0);
    collection.apply_legacy(insert("held", last, 1)).unwrap();
    for past in [last + 1, last + 2] {
        collection
            .apply_legacy(insert(&format!("refused{past}"), past, 2))
            .unwrap();
    }
    assert_eq!(
        ids_of(&collection),
        vec![("held".to_string(), MAX_INTERNAL_ID)]
    );
    assert_eq!(collection.id_counter(), MAX_INTERNAL_ID);
    assert_eq!(collection.vector_count(), 1);

    collection.apply_legacy(Operation::Clear).unwrap();
    collection.apply_legacy(insert("after", 1, 3)).unwrap();
    assert_eq!(ids_of(&collection), vec![("after".to_string(), 1)]);
    assert_eq!(collection.id_counter(), 1);

    let refused = |result: Result<(), zeusdb_vector_core::Error>| {
        assert!(
            matches!(
                result,
                Err(zeusdb_vector_core::Error::JournalReplayMismatch { ref detail })
                    if detail.ends_with("above the id ceiling")
            ),
            "{result:?}"
        );
    };
    let mut second = Collection::build(declaration(100, false), None);
    second.set_counters(MAX_INTERNAL_ID, 0);
    refused(second.apply(insert("refused", last + 1, 2)));
    let mut early = Collection::build(declaration(100, false), None);
    early.set_counters(7, 0);
    refused(early.apply_legacy(insert("refused", last + 1, 2)));
    assert_eq!(early.id_counter(), 7);
    assert!(ids_of(&early).is_empty());
}

/// A collection with a sparse space whose newest record has gone saves a
/// directory whose sparse slot count lies past every id its mappings hold,
/// with a compaction and without, and the load reads it. The loaded
/// collection holds the records, the pages, the counter and the record
/// counts the saved one held, and a save of it reads back the same.
#[test]
fn a_sparse_space_whose_newest_record_went_saves_a_directory_its_load_reads() {
    let temp = TempDir::new();
    let counts = |collection: &Collection| {
        let stats = collection.stats();
        (
            stats["total_vectors"].clone(),
            stats["sparse_records"].clone(),
        )
    };
    for compacted in [false, true] {
        let collection = Collection::build(declaration(100, true), None);
        add(&collection, 0..300, true);
        remove(&collection, 0..300, |i| i >= 290 || i % 7 == 3);
        if compacted {
            collection.compact().unwrap();
        }
        let loaded = reload(&collection, &temp, &format!("newest-gone-{compacted}.zdb"));
        assert_eq!(ids_of(&loaded), ids_of(&collection));
        assert_eq!(pages(&loaded), pages(&collection));
        assert_eq!(loaded.id_counter(), collection.id_counter());
        assert_eq!(counts(&loaded), counts(&collection));
        let again = reload(
            &loaded,
            &temp,
            &format!("newest-gone-again-{compacted}.zdb"),
        );
        assert_eq!(ids_of(&again), ids_of(&collection));
        assert_eq!(pages(&again), pages(&collection));
        assert_eq!(counts(&again), counts(&collection));
    }
}

/// What `reserved_memory_mb` reports is what `shrink_to_fit` gives back, on
/// a raw, a product quantized and a scalar quantized collection, with a
/// declared field and without, declared far past its records and within
/// them, after removals and again after a clear. The graph's slack, the
/// vector stores', the numbering's, and the id store's, the metadata
/// store's and the columns' are counted in both. Each of the stores' parts
/// is above a hundredth of a mebibyte where the declaration is past the
/// records, so a part counted in one figure and not the other moves the
/// figure as reported. Once shrunk the collection reports none, a second
/// shrink gives back nothing, `total_memory_mb` fell by what was given back
/// and every page is the page before.
#[test]
fn the_reserved_figure_is_what_a_shrink_gives_back_for_every_store() {
    let mib = |bytes: usize| format!("{:.2}", bytes as f64 / (1024.0 * 1024.0));
    let figure =
        |collection: &Collection, key: &str| -> f64 { collection.stats()[key].parse().unwrap() };
    for (kind, expected, fields) in [
        ("raw", 200_000usize, true),
        ("raw", 200_000, false),
        ("raw", 100, true),
        ("pq", 200_000, true),
        ("pq only", 200_000, true),
        ("int8", 200_000, true),
    ] {
        let label = format!("{kind} declared {expected} fields {fields}");
        let names = if fields {
            vec!["cat".to_string()]
        } else {
            vec![]
        };
        let declaration = Declaration::validate(4, "l2", 4, 40, expected, names).unwrap();
        let quantization = match kind {
            "pq" => Some(
                declaration
                    .quantization(4, 4, 1_000, None, StorageMode::QuantizedWithRaw)
                    .unwrap(),
            ),
            "pq only" => Some(
                declaration
                    .quantization(4, 4, 1_000, None, StorageMode::QuantizedOnly)
                    .unwrap(),
            ),
            "int8" => Some(
                declaration
                    .scalar_quantization(
                        super::Int8Scale::PER_DIMENSION,
                        1_000,
                        None,
                        StorageMode::QuantizedOnly,
                    )
                    .unwrap(),
            ),
            _ => None,
        };
        let collection = Collection::build(declaration, quantization);
        add(&collection, 0..1_200, false);
        assert_eq!(collection.is_quantized(), kind != "raw", "{label}");
        remove(&collection, 0..1_200, |i| i % 7 == 3);
        for phase in ["built", "cleared"] {
            if phase == "cleared" {
                collection.clear().unwrap();
                add(&collection, 0..300, false);
            }
            let before = collection.stats();
            let pages_before = pages(&collection);
            let freed = collection.shrink_to_fit();
            assert_eq!(before["reserved_memory_mb"], mib(freed), "{label} {phase}");
            let after = collection.stats();
            assert_eq!(after["reserved_memory_mb"], "0.00", "{label} {phase}");
            assert_eq!(collection.shrink_to_fit(), 0, "{label} {phase}");
            let fell = before["total_memory_mb"].parse::<f64>().unwrap()
                - figure(&collection, "total_memory_mb");
            assert!(
                (fell - freed as f64 / (1024.0 * 1024.0)).abs() <= 0.011,
                "{label} {phase}: the total fell {fell} and {freed} bytes were given back"
            );
            assert_eq!(pages(&collection), pages_before, "{label} {phase}");
            if expected > 1_200 {
                assert!(freed > 1 << 20, "{label} {phase}: {freed}");
            }
        }
    }
}
