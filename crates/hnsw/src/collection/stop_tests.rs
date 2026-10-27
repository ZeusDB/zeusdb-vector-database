//! A panic inside a call on a collection, and what every later call sees.
//!
//! Each test panics inside a call made through `Collection::run`, as the
//! binding makes every call, at a point armed on the test's thread or in a
//! closure of its own, and holds every later call to the rule the `stop`
//! module states: the call that panicked and every call after it are refused
//! with the refusal naming the first call, none of them runs, the stopped
//! collection drops and lets its directory go, and a journaled directory
//! reopens with every call its journal committed.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Barrier;

use serde_json::{json, Value};
use zeusdb_vector_core::{compile_filter, panic_arm, Error, IdfScope, PanicPoint, SparseVector};
use zeusdb_vector_sparse::SparseConfig;

use super::journal_support::TempDir;
use super::{
    run_without_index, Arm, Collection, Declaration, ParsedRecord, Query, SparseHalf, StorageMode,
};
use crate::journal::Durability;

fn record(id: &str, dense: &[f32], dims: &[u32]) -> ParsedRecord {
    let mut metadata: HashMap<String, Value> = HashMap::new();
    metadata.insert("cat".to_string(), json!("a"));
    ParsedRecord {
        id: id.to_string(),
        vector: dense.to_vec(),
        sparse: Some(SparseHalf::Vector(SparseVector {
            dims: dims.to_vec(),
            values: vec![1.0; dims.len()],
        })),
        metadata,
    }
}

fn declaration() -> Declaration {
    Declaration::validate(2, "l2", 4, 50, 100, vec!["cat".to_string()])
        .unwrap()
        .with_sparse("terms", SparseConfig::default())
        .unwrap()
}

fn three() -> Vec<ParsedRecord> {
    vec![
        record("r1", &[0.0, 0.0], &[1]),
        record("r2", &[1.0, 0.0], &[2]),
        record("r3", &[0.0, 1.0], &[3]),
    ]
}

/// The ids a dense search returns nearest `query`.
fn dense_ids(collection: &Collection, query: &[f32], k: usize) -> Vec<String> {
    let params = collection.search_params(k, None, false, None).unwrap();
    let hits = collection.search_one(query, None, params).unwrap();
    hits.iter().map(|hit| hit.id().to_string()).collect()
}

/// `add` of `records` through the rule, as the binding makes it.
fn add_through_run(
    collection: &Collection,
    records: Vec<ParsedRecord>,
    overwrite: bool,
) -> Result<super::Added, Error> {
    collection.run("add()", || {
        Ok::<_, Error>(collection.add_records(records, vec![], overwrite))
    })
}

/// The refusal a call on an index that `first` stopped with `cause`
/// returns, in the words a caller reads.
fn stopped_words(first: &str, cause: &str) -> String {
    format!(
        "This index stopped when {first} failed inside the engine: {cause}. It refuses every \
         call from then on, since the failed call may have left a change part made. Drop it, \
         and load() opens the directory it was saved or journaled to."
    )
}

/// Assert `result` is the refusal of a stopped index naming `first` and
/// `cause`.
fn assert_stopped<T: std::fmt::Debug>(
    result: Result<T, Error>,
    first: &str,
    cause: &str,
    call: &str,
) {
    match result {
        Err(Error::Panicked {
            operation,
            cause: said,
            stopped: true,
        }) => {
            assert_eq!(operation, first, "{call}: the call that stopped the index");
            assert_eq!(said, cause, "{call}: what its panic said");
            let refusal = Error::Panicked {
                operation,
                cause: said,
                stopped: true,
            };
            assert_eq!(refusal.to_string(), stopped_words(first, cause), "{call}");
            assert_eq!(refusal.exception().name(), "RuntimeError", "{call}");
        }
        other => panic!("{call}: expected the refusal of a stopped index, got {other:?}"),
    }
}

/// Every operation the binding makes on an index, each through `run` under
/// the name a caller writes it by, is refused naming `first` and `cause`,
/// and none of them runs.
fn assert_every_call_refused(collection: &Collection, first: &str, cause: &str, dir: &TempDir) {
    let target = dir.at("elsewhere.zdb");
    let target = target.to_str().unwrap().to_string();
    let filter = compile_filter(&HashMap::from([("cat".to_string(), json!("a"))])).unwrap();
    let query_vector = [0.0f32, 0.0];
    let sparse_query = SparseVector {
        dims: vec![1],
        values: vec![1.0],
    };
    let arms = [Arm::Dense {
        vector: &query_vector,
        ef: None,
        rerank: None,
    }];
    type Call<'a> = Box<dyn Fn(&Collection) -> Result<(), Error> + 'a>;
    let calls: Vec<(&'static str, Call<'_>)> = vec![
        (
            "add()",
            Box::new(|c| {
                c.add_records(vec![record("r9", &[9.0, 9.0], &[9])], vec![], true);
                Ok(())
            }),
        ),
        (
            "search()",
            Box::new(|c| {
                let params = c.search_params(2, None, false, None)?;
                c.search_one(&query_vector, Some(&filter), params).map(drop)
            }),
        ),
        (
            "search() of a batch",
            Box::new(|c| {
                let params = c.search_params(2, None, false, None)?;
                c.search_batch(&[query_vector.to_vec()], None, params)
                    .map(drop)
            }),
        ),
        (
            "search() of a sparse vector",
            Box::new(|c| {
                c.search_sparse(sparse_query.as_ref(), None, 2, IdfScope::Corpus)
                    .map(drop)
            }),
        ),
        (
            "query()",
            Box::new(|c| c.query(&Query::new(&arms, 2)).map(drop)),
        ),
        (
            "explain()",
            Box::new(|c| c.explain(&Query::new(&arms, 2)).map(drop)),
        ),
        (
            "count()",
            Box::new(|c| {
                c.count(Some(&filter));
                Ok(())
            }),
        ),
        (
            "get_records()",
            Box::new(|c| c.records(vec!["r1".to_string()], true, false).map(drop)),
        ),
        ("list()", Box::new(|c| c.list(10, 0, None).map(drop))),
        (
            "len()",
            Box::new(|c| {
                c.len();
                Ok(())
            }),
        ),
        (
            "contains()",
            Box::new(|c| {
                c.contains("r1");
                Ok(())
            }),
        ),
        (
            "get_vector_count()",
            Box::new(|c| {
                c.vector_count();
                Ok(())
            }),
        ),
        (
            "get_stats()",
            Box::new(|c| {
                c.stats();
                Ok(())
            }),
        ),
        (
            "info()",
            Box::new(|c| {
                c.info();
                Ok(())
            }),
        ),
        (
            "get_quantization_info()",
            Box::new(|c| {
                c.quantization_report();
                Ok(())
            }),
        ),
        (
            "get_storage_mode()",
            Box::new(|c| {
                c.storage_mode();
                Ok(())
            }),
        ),
        (
            "get_training_progress()",
            Box::new(|c| {
                c.training_progress();
                Ok(())
            }),
        ),
        (
            "indexed_fields",
            Box::new(|c| {
                c.indexed_fields();
                Ok(())
            }),
        ),
        (
            "add_metadata()",
            Box::new(|c| c.add_metadata(HashMap::from([("k".to_string(), "v".to_string())]))),
        ),
        (
            "get_all_metadata()",
            Box::new(|c| {
                c.all_metadata();
                Ok(())
            }),
        ),
        ("save()", Box::new(|c| c.save(&target))),
        ("checkpoint()", Box::new(|c| c.checkpoint())),
        ("backup()", Box::new(|c| c.backup(&target).map(drop))),
        (
            "journal_to()",
            Box::new(|c| c.journal_to(&target, Durability::default())),
        ),
        (
            "journal_path",
            Box::new(|c| {
                c.journal_status();
                Ok(())
            }),
        ),
        (
            "identity",
            Box::new(|c| {
                c.identity_record();
                Ok(())
            }),
        ),
        (
            "remove_point()",
            Box::new(|c| c.remove_point("r1".to_string()).map(drop)),
        ),
        (
            "remove_points()",
            Box::new(|c| c.remove_points(&["r1".to_string()]).map(drop)),
        ),
        (
            "remove_where()",
            Box::new(|c| c.remove_where(&filter).map(drop)),
        ),
        (
            "update_metadata()",
            Box::new(|c| c.update_metadata("r1", HashMap::new()).map(drop)),
        ),
        ("clear()", Box::new(|c| c.clear().map(drop))),
        ("compact()", Box::new(|c| c.compact().map(drop))),
        (
            "rebuild()",
            Box::new(|c| {
                let plan = c.plan_rebuild(Some(8), None, None)?;
                c.rebuild(plan).map(drop)
            }),
        ),
        (
            "rebuild_with_quantization()",
            Box::new(|c| c.rebuild_with_quantization().map(drop)),
        ),
        (
            "shrink_to_fit()",
            Box::new(|c| {
                c.shrink_to_fit();
                Ok(())
            }),
        ),
        (
            "benchmark_concurrent_reads()",
            Box::new(|c| c.benchmark_reads(2, Some(1)).map(drop)),
        ),
    ];
    for (name, call) in &calls {
        let ran = Cell::new(false);
        let result = collection.run(name, || {
            ran.set(true);
            call(collection)
        });
        assert!(!ran.get(), "{name} ran on a stopped index");
        assert_stopped(result, first, cause, name);
    }
    assert!(collection.is_stopped());
}

/// A panic after a record's dense half is installed and before its sparse
/// half is, with no storage guard held, stops the index: the call is
/// refused, every later call is refused naming it, and nothing reads the
/// record that went into one space and not the other.
#[test]
fn a_panic_between_the_two_halves_of_a_record_stops_the_index() {
    let dir = TempDir::new("stop-halves");
    let collection = Collection::build(declaration(), None);
    let added = add_through_run(&collection, three(), false).unwrap();
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(!collection.is_stopped());

    panic_arm(PanicPoint::InstallBeforeSparse);
    let refused = add_through_run(&collection, vec![record("r4", &[5.0, 5.0], &[9])], false);
    let cause = "a panic injected at install:before_sparse";
    assert_stopped(refused, "add()", cause, "the add that panicked");
    assert_every_call_refused(&collection, "add()", cause, &dir);
}

/// A panic under the four write guards of an insertion's second phase,
/// which poisons all four, stops the index, so no later reader meets a
/// poisoned guard: each is refused before it takes one.
#[test]
fn a_panic_under_the_install_guards_stops_the_index() {
    let dir = TempDir::new("stop-guards");
    let collection = Collection::build(declaration(), None);
    let added = add_through_run(&collection, three(), false).unwrap();
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);

    panic_arm(PanicPoint::InsertAfterName);
    let refused = add_through_run(&collection, vec![record("r4", &[5.0, 5.0], &[9])], false);
    let cause = "a panic injected at insert:after_name";
    assert_stopped(refused, "add()", cause, "the add that panicked");
    assert_every_call_refused(&collection, "add()", cause, &dir);
}

/// A panic in an overwrite after its commit and its removal of the record
/// it replaces, and before the replacement is installed, stops a journaled
/// index, so neither a checkpoint nor a save writes the index without
/// either record. Dropped, the index lets its directory go, and the
/// directory reopens with the whole call its journal committed.
#[test]
fn a_panic_between_an_overwrite_removal_and_its_install_reopens_with_the_call() {
    let dir = TempDir::new("stop-overwrite");
    let path = dir.at("journaled.zdb");
    let path = path.to_str().unwrap();
    let collection = Collection::build(declaration(), None);
    collection.journal_to(path, Durability::default()).unwrap();
    let added = add_through_run(&collection, three(), false).unwrap();
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);

    panic_arm(PanicPoint::AddBeforeInstalls);
    let refused = add_through_run(&collection, vec![record("r2", &[7.0, 7.0], &[7])], true);
    let cause = "a panic injected at add:before_installs";
    assert_stopped(refused, "add()", cause, "the overwrite that panicked");
    assert_every_call_refused(&collection, "add()", cause, &dir);
    drop(collection);

    let (reopened, _) = Collection::recover(path, None, Durability::default())
        .expect("the directory reopens once the stopped index lets it go");
    assert_eq!(reopened.len(), 3);
    assert!(reopened.contains("r2"));
    assert_eq!(dense_ids(&reopened, &[7.0, 7.0], 1), ["r2"]);
    assert!(!reopened.is_stopped());
}

/// A panic inside a training's install, after the codebook is fitted and
/// before the graph is rebuilt over it, stops the index, so nothing reports
/// a trained index that searches raw. The journal committed the training,
/// so the directory reopens trained and quantized.
#[test]
fn a_panic_inside_a_training_stops_the_index_and_reopens_trained() {
    let dir = TempDir::new("stop-training");
    let path = dir.at("trained.zdb");
    let path = path.to_str().unwrap();
    let declaration = Declaration::validate(2, "l2", 4, 50, 2000, vec![]).unwrap();
    let quantization = declaration
        .quantization(1, 2, 1000, None, StorageMode::QuantizedWithRaw)
        .unwrap();
    let collection = Collection::build(declaration, Some(quantization));
    collection.journal_to(path, Durability::default()).unwrap();
    let records = |range: std::ops::Range<usize>| -> Vec<ParsedRecord> {
        range
            .map(|i| {
                let x = (i % 97) as f32 / 97.0;
                let y = (i % 89) as f32 / 89.0;
                ParsedRecord::from((format!("t{i}"), vec![x, y], HashMap::new()))
            })
            .collect()
    };
    let added = add_through_run(&collection, records(0..999), false).unwrap();
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(!collection.can_use_quantization());

    panic_arm(PanicPoint::TrainingBeforeRebuild);
    let refused = add_through_run(&collection, records(999..1000), false);
    let cause = "a panic injected at training:before_rebuild";
    assert_stopped(refused, "add()", cause, "the add that trained");
    for (name, ran) in [
        (
            "is_quantized()",
            collection.run("is_quantized()", || {
                Ok::<_, Error>(collection.is_quantized())
            }),
        ),
        (
            "can_use_quantization()",
            collection.run("can_use_quantization()", || {
                Ok::<_, Error>(collection.can_use_quantization())
            }),
        ),
    ] {
        assert_stopped(ran, "add()", cause, name);
    }
    assert_every_call_refused(&collection, "add()", cause, &dir);
    drop(collection);

    let (reopened, _) = Collection::recover(path, None, Durability::default()).unwrap();
    assert_eq!(reopened.len(), 1000);
    assert!(reopened.can_use_quantization());
    assert!(reopened.is_quantized());
}

/// A reader that panics while it holds a mutex stops the index as a writer
/// does, so the writer that would next take the poisoned mutex is refused
/// naming the reader rather than meeting the poison.
#[test]
fn a_panic_in_a_reader_stops_the_index() {
    let dir = TempDir::new("stop-reader");
    let collection = Collection::build(declaration(), None);
    let added = add_through_run(&collection, three(), false).unwrap();
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);

    let refused: Result<usize, Error> = collection.run("get_stats()", || {
        let _count = collection.vector_count.lock().unwrap();
        panic!("a reader failed under the count's mutex");
    });
    let cause = "a reader failed under the count's mutex";
    assert_stopped(refused, "get_stats()", cause, "the reader that panicked");
    assert_every_call_refused(&collection, "get_stats()", cause, &dir);
}

/// The first call to panic names every refusal. A call already running
/// when it panics, which then panics itself, is refused in the same words,
/// and the second panic replaces nothing.
#[test]
fn the_first_panic_names_every_refusal() {
    let collection = Collection::build(declaration(), None);
    let both_inside = Barrier::new(2);
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            collection.run("add()", || -> Result<(), Error> {
                both_inside.wait();
                panic!("the first panic");
            })
        });
        let second = scope.spawn(|| {
            collection.run("search()", || -> Result<(), Error> {
                both_inside.wait();
                while !collection.is_stopped() {
                    std::thread::yield_now();
                }
                panic!("the second panic");
            })
        });
        (first.join().unwrap(), second.join().unwrap())
    });
    assert_stopped(first, "add()", "the first panic", "the first call");
    assert_stopped(
        second,
        "add()",
        "the first panic",
        "the call already running",
    );
    let later = collection.run("len()", || Ok::<_, Error>(collection.len()));
    assert_stopped(later, "add()", "the first panic", "a later call");
}

/// A call that does not panic returns what it returns, a refusal of its
/// own included, and stops nothing.
#[test]
fn a_call_that_does_not_panic_passes_through() {
    let collection = Collection::build(declaration(), None);
    let added = add_through_run(&collection, three(), false).unwrap();
    assert_eq!(added.inserted.len(), 3);
    let refused: Result<(), Error> = collection.run("search()", || Err(Error::SearchVectorEmpty));
    assert!(matches!(refused, Err(Error::SearchVectorEmpty)));
    assert!(!collection.is_stopped());
    assert_eq!(
        collection
            .run("len()", || Ok::<_, Error>(collection.len()))
            .unwrap(),
        3
    );
}

/// A call with no index to stop is refused where it panics, in the words
/// for a call alone, and returns what it returns where it does not.
#[test]
fn a_call_without_an_index_is_refused_where_it_panics() {
    let refused: Result<(), Error> = run_without_index("load()", || panic!("a load failed."));
    match refused {
        Err(error @ Error::Panicked { stopped: false, .. }) => {
            assert_eq!(
                error.to_string(),
                "load() failed inside the engine: a load failed."
            );
            assert_eq!(error.exception().name(), "RuntimeError");
        }
        other => panic!("expected the refusal of a call alone, got {other:?}"),
    }
    let passed: Result<usize, Error> = run_without_index("create()", || Ok(7));
    assert_eq!(passed.unwrap(), 7);
}
