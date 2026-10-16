//! What runs under the mutation guard, and what a record carries across
//! its boundary.
//!
//! A record's terms are counted into term ids as the record is inserted,
//! under the guard, so a `clear` cannot land between a record's ids being
//! issued and its postings being written, and a query's terms are counted
//! under the guards its search holds, so a `clear` cannot land between a
//! query's ids being counted and its postings being searched either.
//! `add_metadata` takes the guard as every other durable mutation does. And
//! the stamp training records is a parameter, read from the clock at the
//! trigger and handed down.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use zeusdb_vector_core::{IdfScope, PQ};
use zeusdb_vector_sparse::{SparseConfig, Weighting};
use zeusdb_vector_text::SimpleTokenizer;

#[cfg(debug_assertions)]
use crate::locks::held_now;

use super::insert::{ClearPoint, CLEAR_HOOK};
use super::query::AFTER_RESOLVE;
use super::{Collection, Declaration, ParsedRecord, SparseHalf, StorageMode};

/// What this thread holds by the lock registry's count, which a build with
/// debug assertions keeps. A build without them counts nothing, so this reads
/// an empty list.
#[cfg(not(debug_assertions))]
fn held_now() -> Vec<String> {
    Vec::new()
}

fn text_collection() -> Collection {
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
    Collection::build(declaration, None)
}

/// A record carrying `text` as the terms the tokenizer split it into, which
/// hold no id until the record is inserted.
fn text_record(collection: &Collection, id: &str, dense: [f32; 2], text: &str) -> ParsedRecord {
    ParsedRecord {
        id: id.to_string(),
        vector: dense.to_vec(),
        sparse: Some(SparseHalf::Terms(collection.tokenize(text).unwrap())),
        metadata: HashMap::new(),
    }
}

fn ids_of(page: &[(String, f32)]) -> Vec<&str> {
    page.iter().map(|(id, _)| id.as_str()).collect()
}

/// Every term the text layer's dictionary holds, in id order.
fn terms_by_id(collection: &Collection) -> Vec<String> {
    let text = collection.sparse().unwrap().text.as_ref().unwrap();
    let dictionary = text.dictionary.read().unwrap();
    dictionary.terms().into_iter().map(str::to_string).collect()
}

/// A `clear` between a record's tokenizing and its insert cannot put the
/// record's postings over term ids the dictionary reissued, because the
/// record carries terms across that gap and not ids: the ids are issued
/// under the mutation guard as the record is inserted, after the clear.
///
/// The sequence this holds is the one that used to go wrong. A caller
/// counted "zebra yak" into ids 0 and 1 outside the mutation guard, `clear`
/// emptied the dictionary, the caller inserted its postings over 0 and 1,
/// the next text "apple" took id 0, and a search for "apple" returned the
/// record that never held it while a search for "zebra" returned nothing.
#[test]
fn clear_between_a_records_tokenizing_and_its_insert_cannot_reissue_its_terms() {
    let collection = text_collection();
    let a = text_record(&collection, "A", [1.0, 0.0], "zebra yak");
    assert_eq!(collection.term_count(), Some(0), "tokenizing issues no id");

    // The clear lands in the gap and finds nothing of A's to empty.
    assert_eq!(collection.clear().unwrap(), 0);

    assert_eq!(
        collection.add_records(vec![a], vec![], false).total_errors,
        0
    );
    let b = text_record(&collection, "B", [0.0, 1.0], "apple");
    assert_eq!(
        collection.add_records(vec![b], vec![], false).total_errors,
        0
    );

    // The ids were issued as each record was inserted, after the clear.
    assert_eq!(terms_by_id(&collection), ["zebra", "yak", "apple"]);
    let apple = collection
        .search_text("apple", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(ids_of(&apple), ["B"]);
    let zebra = collection
        .search_text("zebra", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(ids_of(&zebra), ["A"]);

    // And a clear after both leaves a dictionary the next record starts.
    assert_eq!(collection.clear().unwrap(), 2);
    let c = text_record(&collection, "C", [1.0, 1.0], "apple zebra");
    assert_eq!(
        collection.add_records(vec![c], vec![], false).total_errors,
        0
    );
    assert_eq!(terms_by_id(&collection), ["apple", "zebra"]);
    let apple = collection
        .search_text("apple", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(ids_of(&apple), ["C"]);
}

/// Under clears running on another thread, a search for a term never
/// returns a record that did not carry it. Every record carries one term of
/// its own, and after each insert its own term is searched for. The clearing
/// thread signals each clear it finishes on a channel of one place, which
/// the inserting thread empties after its hundredth insert and waits on
/// after its two hundredth, so a clear finishes while the inserts run.
#[test]
fn a_search_never_returns_a_record_that_never_held_the_term_under_concurrent_clears() {
    let collection = Arc::new(text_collection());
    let stop = Arc::new(AtomicBool::new(false));
    let (cleared_tx, cleared) = std::sync::mpsc::sync_channel::<()>(1);
    let clearer = {
        let collection = collection.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let mut clears = 0usize;
            while !stop.load(Ordering::Relaxed) {
                collection.clear().unwrap();
                clears += 1;
                let _ = cleared_tx.try_send(());
                thread::yield_now();
            }
            clears
        })
    };

    let mut wrong = Vec::new();
    for i in 0..400 {
        if i == 100 {
            while cleared.try_recv().is_ok() {}
        }
        if i == 200 {
            cleared
                .recv_timeout(std::time::Duration::from_secs(20))
                .expect("a clear finishes while the inserts run");
        }
        let own = format!("only{i}");
        let record = text_record(
            &collection,
            &format!("r{i}"),
            [i as f32, 1.0],
            &format!("shared {own}"),
        );
        assert_eq!(
            collection
                .add_records(vec![record], vec![], false)
                .total_errors,
            0
        );
        let page = collection
            .search_text(&own, None, 5, IdfScope::Corpus)
            .unwrap();
        for (id, _) in &page {
            if *id != format!("r{i}") {
                wrong.push((own.clone(), id.clone()));
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    let clears = clearer.join().unwrap();
    assert!(
        wrong.is_empty(),
        "records returned for a term they never held: {wrong:?}"
    );
    assert!(clears > 0, "the clearing thread ran");
}

/// A `clear` and an insert cannot land between a text arm's terms being
/// counted into ids and its postings being searched, because the count
/// runs under the dictionary's guard taken after the sparse index's and
/// held through the search, so the clear waits at the record set's guard
/// until the page is made.
///
/// The sequence this holds is the read half of the one above. A query
/// counted "zebra" into id 0 and released the dictionary, `clear` emptied
/// it, the next text "apple" took id 0, and the query then searched the
/// postings over id 0 and returned the record that never held "zebra".
/// The interleaving is arranged rather than raced for: the mutation is
/// started from the hook the query runs once its ids are counted, and the
/// hook waits for it, up to a bound. Where the mutation can land inside
/// the window it does so within milliseconds, and where the query's guards
/// hold it out the wait ends with it still pending.
#[test]
fn clear_between_a_querys_counting_and_its_search_cannot_reissue_its_terms() {
    /// What the hook saw: whether the mutation it started had landed by the
    /// time the hook returned, and the thread to join.
    struct Held {
        landed_in_window: bool,
        mutation: thread::JoinHandle<()>,
    }

    let collection = Arc::new(text_collection());
    let a = text_record(&collection, "A", [1.0, 0.0], "zebra");
    assert_eq!(
        collection.add_records(vec![a], vec![], false).total_errors,
        0
    );
    assert_eq!(terms_by_id(&collection), ["zebra"]);

    let outcome: Rc<RefCell<Option<Held>>> = Rc::new(RefCell::new(None));
    AFTER_RESOLVE.with(|hook| {
        let collection = collection.clone();
        let outcome = outcome.clone();
        let run: Box<dyn FnOnce()> = Box::new(move || {
            let mutation = {
                let collection = collection.clone();
                thread::spawn(move || {
                    assert_eq!(collection.clear().unwrap(), 1);
                    let b = text_record(&collection, "B", [0.0, 1.0], "apple");
                    assert_eq!(
                        collection.add_records(vec![b], vec![], false).total_errors,
                        0
                    );
                })
            };
            let deadline = Instant::now() + Duration::from_millis(500);
            while !mutation.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            *outcome.borrow_mut() = Some(Held {
                landed_in_window: mutation.is_finished(),
                mutation,
            });
        });
        *hook.borrow_mut() = Some(run);
    });

    let page = collection
        .search_text("zebra", None, 5, IdfScope::Corpus)
        .unwrap();
    let Held {
        landed_in_window,
        mutation,
    } = outcome.borrow_mut().take().expect("the hook ran");
    assert_eq!(
        ids_of(&page),
        ["A"],
        "the page holds the record that carried the term when it was counted"
    );
    assert!(
        !landed_in_window,
        "the clear and the insert landed between the count and the search"
    );
    mutation.join().unwrap();

    // The clear and the insert landed once the page was made.
    assert_eq!(terms_by_id(&collection), ["apple"]);
    let apple = collection
        .search_text("apple", None, 5, IdfScope::Corpus)
        .unwrap();
    assert_eq!(ids_of(&apple), ["B"]);
    let zebra = collection
        .search_text("zebra", None, 5, IdfScope::Corpus)
        .unwrap();
    assert!(zebra.is_empty());
}

/// Under clears and inserts running on another thread, a search for a
/// term never returns a record that did not carry it. The other thread
/// clears and inserts one record carrying a term of its own each time, so
/// every clear reissues id 0 to a new term, which is the reissue the query
/// path is held against, under a race rather than by arrangement.
///
/// `wrong` is the invariant and holds however the race falls. That the
/// pairing works at all is checked where it is decided rather than raced
/// for: the mutator reads `stop` at the top of its loop, so it always
/// finishes its insert and its generation store before returning, and the
/// collection it leaves holds exactly the record of the last round.
/// Counting the searches that landed inside a record's window instead is a
/// count of how often the searching thread got in between an insert and the
/// clear after it, which is a property of the machine.
#[test]
fn a_search_never_returns_a_record_that_never_held_the_term_under_concurrent_reissues() {
    let collection = Arc::new(text_collection());
    let stop = Arc::new(AtomicBool::new(false));
    let generation = Arc::new(AtomicUsize::new(0));
    let mutator = {
        let collection = collection.clone();
        let stop = stop.clone();
        let generation = generation.clone();
        thread::spawn(move || {
            let mut clears = 0usize;
            while !stop.load(Ordering::Relaxed) {
                collection.clear().unwrap();
                clears += 1;
                let record = text_record(
                    &collection,
                    &format!("r{clears}"),
                    [1.0, 0.0],
                    &format!("only{clears}"),
                );
                assert_eq!(
                    collection
                        .add_records(vec![record], vec![], false)
                        .total_errors,
                    0
                );
                generation.store(clears, Ordering::Release);
                thread::yield_now();
            }
            clears
        })
    };

    let mut wrong = Vec::new();
    let mut queries = 0usize;
    while queries < 3000 {
        let current = generation.load(Ordering::Acquire);
        if current == 0 {
            thread::yield_now();
            continue;
        }
        queries += 1;
        let page = collection
            .search_text(&format!("only{current}"), None, 5, IdfScope::Corpus)
            .unwrap();
        for (id, _) in &page {
            if *id != format!("r{current}") {
                wrong.push((current, id.clone()));
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    let clears = mutator.join().unwrap();
    assert!(
        wrong.is_empty(),
        "records returned for a term they never held: {wrong:?}"
    );
    assert!(clears > 0, "the mutating thread ran");

    // Settled. One record, and a search for its term returns it.
    assert_eq!(collection.len(), 1);
    let settled = collection
        .search_text(&format!("only{clears}"), None, 5, IdfScope::Corpus)
        .unwrap();
    let last = format!("r{clears}");
    assert_eq!(ids_of(&settled), [last.as_str()]);
}

/// A clear trades every structure it empties under one acquisition of their
/// guards, the sparse space's postings and its dictionary among them. Held
/// open once the trade is done and every guard let go, with `writers` still
/// held, a reader on another thread finds the cleared state in every
/// structure at once: no record, no posting, no term, and statistics that
/// count none of them, and never the old postings or the old terms beside
/// the cleared record set.
#[test]
fn a_reader_beside_a_clear_finds_every_structure_cleared_at_once() {
    let collection = Arc::new(text_collection());
    let records: Vec<ParsedRecord> = (0..30)
        .map(|i| {
            text_record(
                &collection,
                &format!("r{i}"),
                [i as f32, 1.0],
                &format!("shared only{i}"),
            )
        })
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let read = |collection: &Collection| {
        let stats = collection.stats();
        let text = collection
            .search_text("shared", None, 5, IdfScope::Corpus)
            .unwrap();
        let params = collection.search_params(5, None, false, None).unwrap();
        let dense = collection.search_one(&[1.0, 1.0], None, params).unwrap();
        (
            stats["total_vectors"].clone(),
            stats["sparse_records"].clone(),
            stats["term_count"].clone(),
            text.len(),
            dense.len(),
        )
    };
    assert_eq!(
        read(&collection),
        ("30".to_string(), "30".to_string(), "31".to_string(), 5, 5)
    );

    let (seen_tx, seen) = std::sync::mpsc::channel();
    {
        let collection = Arc::clone(&collection);
        CLEAR_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |point| {
                if point == ClearPoint::Traded {
                    let collection = Arc::clone(&collection);
                    let reader = thread::spawn(move || read(&collection));
                    let _ = seen_tx.send(reader.join().expect("the reader answers"));
                }
            }));
        });
    }
    assert_eq!(collection.clear().unwrap(), 30);
    CLEAR_HOOK.with(|hook| *hook.borrow_mut() = None);
    let seen = seen.try_recv().expect("the clear reached its trade");
    let cleared = ("0".to_string(), "0".to_string(), "0".to_string(), 0, 0);
    assert_eq!(seen, cleared, "the reader beside the clear");
    assert_eq!(read(&collection), cleared, "after the clear");
}

/// What a clear traded out is dropped once `writers` is let go, with no
/// guard held. While its drop is held open, the dropping thread holds no
/// guard by the lock registry's own count, a second writer adds a record
/// and returns, and a search answers it. The clear returns only once the
/// drop is done.
#[test]
fn a_second_writer_goes_while_a_clears_old_structures_drop() {
    let collection = text_collection();
    let records: Vec<ParsedRecord> = (0..30)
        .map(|i| text_record(&collection, &format!("r{i}"), [i as f32, 1.0], "shared"))
        .collect();
    assert_eq!(
        collection.add_records(records, vec![], false).total_errors,
        0
    );
    let wait = Duration::from_secs(20);
    thread::scope(|scope| {
        let (dropping_tx, dropping) = std::sync::mpsc::channel::<Vec<String>>();
        let (dropped_tx, dropped) = std::sync::mpsc::channel::<Vec<String>>();
        let (go, go_rx) = std::sync::mpsc::channel::<()>();
        let (returned_tx, returned) = std::sync::mpsc::channel::<Instant>();
        let clearer = {
            let collection = &collection;
            scope.spawn(move || {
                CLEAR_HOOK.with(|hook| {
                    *hook.borrow_mut() = Some(Box::new(move |point| match point {
                        ClearPoint::Dropping => {
                            let _ = dropping_tx.send(held_now());
                            let _ = go_rx.recv();
                        }
                        ClearPoint::Dropped => {
                            let _ = dropped_tx.send(held_now());
                        }
                        ClearPoint::Traded => {}
                    }));
                });
                let removed = collection.clear().unwrap();
                returned_tx.send(Instant::now()).unwrap();
                CLEAR_HOOK.with(|hook| *hook.borrow_mut() = None);
                removed
            })
        };

        let held = dropping
            .recv_timeout(wait)
            .expect("the clear drops what it traded out");
        assert_eq!(
            held,
            Vec::<String>::new(),
            "no guard is held as the drop begins"
        );
        assert_eq!(collection.len(), 0, "the trade ran before the drop");

        // A second writer goes and returns, and a search answers its record.
        let (wrote_tx, wrote) = std::sync::mpsc::channel::<usize>();
        {
            let collection = &collection;
            scope.spawn(move || {
                let record = text_record(collection, "later", [1.0, 1.0], "after");
                let added = collection.add_records(vec![record], vec![], false);
                wrote_tx.send(added.total_errors).unwrap();
            });
        }
        assert_eq!(
            wrote
                .recv_timeout(wait)
                .expect("a second writer completes a call during the drop"),
            0
        );
        let page = collection
            .search_text("after", None, 5, IdfScope::Corpus)
            .unwrap();
        assert_eq!(ids_of(&page), ["later"]);
        assert!(
            returned.recv_timeout(Duration::from_millis(50)).is_err(),
            "the clear waits for its drop"
        );

        let released = Instant::now();
        go.send(()).unwrap();
        assert_eq!(
            dropped.recv_timeout(wait).expect("the drop finishes"),
            Vec::<String>::new(),
            "no guard is held as the drop ends"
        );
        assert_eq!(clearer.join().unwrap(), 30);
        assert!(
            returned.recv().unwrap() > released,
            "the clear returned after the drop"
        );
    });
    assert_eq!(collection.len(), 1);
    assert!(collection.positions_agree());
}

/// `add_metadata` takes the mutation guard, as every durable mutation does,
/// so a caller holding the guard sees it wait, and it merges once the guard
/// is free.
#[test]
fn add_metadata_waits_for_the_mutation_guard() {
    let collection = Arc::new(Collection::build(
        Declaration::validate(2, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    ));
    let held = collection.writers.lock().unwrap();
    let writer = {
        let collection = collection.clone();
        thread::spawn(move || {
            collection.add_metadata(HashMap::from([("owner".to_string(), "held".to_string())]))
        })
    };
    thread::sleep(Duration::from_millis(200));
    assert!(
        !writer.is_finished(),
        "add_metadata wrote while the mutation guard was held"
    );
    assert_eq!(collection.metadata("owner"), None);
    drop(held);
    writer.join().unwrap().unwrap();
    assert_eq!(collection.metadata("owner").as_deref(), Some("held"));

    collection
        .add_metadata(HashMap::from([("dataset".to_string(), "docs".to_string())]))
        .unwrap();
    let all = collection.all_metadata();
    assert_eq!(all.len(), 2);
    assert_eq!(all.get("owner").map(String::as_str), Some("held"));
}

/// A thousand records of width eight, deterministic, for a training run.
fn training_records() -> Vec<ParsedRecord> {
    let mut state = 0x5EED_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..1000)
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: (0..8).map(|_| next()).collect(),
            sparse: None,
            metadata: HashMap::new(),
        })
        .collect()
}

/// The stamp training records is a parameter. Through the trigger it is the
/// clock, read as training starts; handed a stamp, training records that
/// stamp and nothing else about the training changes.
#[test]
fn training_is_stamped_with_the_stamp_it_is_handed() {
    let declare = || Declaration::validate(8, "l2", 8, 50, 2000, vec![]).unwrap();
    let config = declare()
        .quantization(4, 4, 1000, None, StorageMode::QuantizedOnly)
        .unwrap();

    // Through the trigger, on the insert that fills the training set.
    let triggered = Collection::build(declare(), Some(config.clone()));
    let before = Utc::now();
    let added = triggered.add_records(training_records(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(triggered.is_quantized());
    let stamp = triggered.training_completed_at().unwrap();
    let stamped = DateTime::parse_from_rfc3339(&stamp).unwrap();
    assert!(stamped >= before - chrono::Duration::seconds(1));
    assert!(stamped <= Utc::now() + chrono::Duration::seconds(1));

    // Handed a stamp, on a collection holding the same records raw, as the
    // loader assembles one.
    let mut handed = Collection::build(declare(), None);
    let added = handed.add_records(training_records(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(!handed.has_quantization());
    handed.set_quantization_config(Some(config));
    handed.set_pq(Some(Arc::new(PQ::new(8, 4, 4, 1000, None))));
    handed.set_training_ids((0..1000).map(|i| format!("r{i}")).collect());
    let given = "2001-02-03T04:05:06+00:00".to_string();
    {
        let writers = handed.hold_writers();
        handed
            .train_quantization_from_ids(&writers, given.clone())
            .unwrap();
    }
    assert!(handed.is_quantized());
    assert_eq!(handed.training_completed_at(), Some(given));
    assert_eq!(handed.storage_mode(), "quantized_active");
    assert_eq!(handed.len(), 1000);
}
