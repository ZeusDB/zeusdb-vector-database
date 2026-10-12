//! The close-up built off to the side, held to the close-up made in place,
//! and what searches, counts and writes see while it runs.
//!
//! A close-up builds new copies of every per-record structure under read
//! guards and swaps them in under every write guard. The first test holds the
//! structures it leaves to the ones the close-up made in place leaves, part
//! for part and capacity for capacity, from the same states under every
//! declaration. The rest run beside a close-up on other threads: searches and
//! counts answer for the state before the change they overlap or the state
//! after it and for nothing else, a search runs while the build is held open
//! and waits while the swap is, a second writer waits for the whole of it,
//! and a count under a filter the columns only bound reads its bound in the
//! metadata under the same hold.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use zeusdb_vector_core::{
    compile_filter, matches_filter, Filter, IdfScope, Selection, SparseVector,
};
use zeusdb_vector_sparse::{SparseConfig, Unlink, Weighting};

use super::insert::{CloseUpPoint, CLOSE_UP_HOOK};
use super::{
    Collection, Declaration, DenseIndex, Int8Scale, ParsedRecord, SparseHalf, StorageMode,
    COUNT_HOOK,
};

/// Dimension of every vector.
const DIM: usize = 8;
/// The holes a collection keeps whatever its size.
const ALLOWANCE: usize = 4_096;
/// The training size every quantized declaration names.
const TRAINING: usize = 1_000;
/// The page size every search asks for, and the traversal width.
const K: usize = 10;
const EF: usize = 40;

/// What a collection declares besides its fields and its sparse space.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Raw,
    PqWithRaw,
    PqOnly,
    Int8,
}

/// A collection at dimension 8 and a small degree, so the thousands of
/// records a close-up on removal needs build quickly, declaring `fields` and
/// a sparse space `terms` under `unlink` and `weighting`.
fn collection(
    kind: Kind,
    expected_size: usize,
    fields: &[&str],
    unlink: Unlink,
    weighting: Weighting,
) -> Collection {
    let declaration = Declaration::validate(
        DIM,
        "l2",
        4,
        16,
        expected_size,
        fields.iter().map(|field| field.to_string()).collect(),
    )
    .unwrap()
    .with_sparse(
        "terms",
        SparseConfig {
            unlink,
            weighting,
            ..SparseConfig::default()
        },
    )
    .unwrap();
    let quantization = match kind {
        Kind::Raw => None,
        Kind::PqWithRaw => Some(
            declaration
                .quantization(4, 4, TRAINING, None, StorageMode::QuantizedWithRaw)
                .unwrap(),
        ),
        Kind::PqOnly => Some(
            declaration
                .quantization(4, 4, TRAINING, None, StorageMode::QuantizedOnly)
                .unwrap(),
        ),
        Kind::Int8 => Some(
            declaration
                .scalar_quantization(
                    Int8Scale::PER_DIMENSION,
                    TRAINING,
                    None,
                    StorageMode::QuantizedOnly,
                )
                .unwrap(),
        ),
    };
    Collection::build(declaration, quantization)
}

/// splitmix64, for the values a record is given.
fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Record `i`, named `r{i}`, the same every time it is asked for: a vector
/// on a grid of a quarter, so some distances tie, `cat` from ten values of
/// which `a` is one in ten, `tag`, `i` and a unique `u`, and a sparse half
/// three times in four.
fn record(i: usize) -> ParsedRecord {
    let mut state = i as u64 ^ 0x00c1_05e0_0206;
    let mut next = || {
        state = mix(state);
        state
    };
    let vector = (0..DIM)
        .map(|_| ((next() % 41) as f32 - 20.0) * 0.25)
        .collect();
    let tag = ["x", "y"][(next() % 2) as usize];
    let sparse = (!i.is_multiple_of(4)).then(|| {
        let first = (next() % 13) as u32;
        SparseHalf::Vector(SparseVector {
            dims: vec![first, 20 + (i % 7) as u32],
            values: vec![1.0, 1.0 + (i % 3) as f32],
        })
    });
    ParsedRecord {
        id: format!("r{i}"),
        vector,
        sparse,
        metadata: HashMap::from([
            (
                "cat".to_string(),
                json!(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"][i % 10]),
            ),
            ("tag".to_string(), json!(tag)),
            ("i".to_string(), json!(i)),
            ("u".to_string(), json!(format!("u{i}"))),
        ]),
    }
}

fn add(collection: &Collection, records: impl Iterator<Item = usize>) {
    let added = collection.add_records(records.map(record).collect(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn remove(collection: &Collection, records: impl Iterator<Item = usize>) {
    let ids: Vec<String> = records.map(|i| format!("r{i}")).collect();
    let missing = collection.remove_points(&ids).unwrap();
    assert!(missing.is_empty(), "{missing:?}");
}

fn filter(value: Value) -> Filter {
    let map: HashMap<String, Value> = serde_json::from_value(value).unwrap();
    compile_filter(&map).unwrap()
}

/// The filter the columns bound and the metadata decides: `cat` is declared
/// and its value is one record in ten, `tag` is not declared.
fn mixed() -> Filter {
    filter(json!({"cat": "a", "tag": "x"}))
}

/// Whether the columns only bound `filter` on `collection`, being the
/// arm of `count` the close-up could reach between two guards.
fn narrowed(collection: &Collection, filter: &Filter) -> bool {
    let columns = collection.columns.read().unwrap();
    matches!(columns.select(filter), Selection::Narrowed(..))
}

/// The positions below the extent that hold no record.
fn holes(collection: &Collection) -> usize {
    let ids = collection.ids.read().unwrap();
    ids.extent() - ids.len()
}

// ============================================================================
// THE TWO CLOSE-UPS, FROM THE SAME STATES
// ============================================================================

/// Run both close-ups from the collection's state, the one built off to the
/// side and the one made in place, and hold every structure each leaves to
/// the other, part for part, value for value and capacity for capacity. The
/// collection is left closed up.
///
/// One part differs by design. Where the graph keeps its nodes, `node_at`
/// keeps what it holds, or the records or the reservation the declared size
/// takes, whichever is more, where that is less; the pass in place kept it at
/// the graph's reservation for its nodes. Its values are held to the pass in
/// place's and its capacity to that rule.
fn both_ways(collection: &Collection, replace: bool, label: &str) {
    let (capacity, held, declared, by_node) = {
        let ids = collection.ids.read().unwrap();
        let index = collection.dense().index.read().unwrap();
        (
            index.node_at_heap_bytes() / std::mem::size_of::<u32>(),
            ids.len(),
            index.graph().declared_records(collection.expected_size()),
            index.graph().record_capacity(),
        )
    };
    let graph = replace.then(|| {
        collection
            .replacement_graph(
                collection.m(),
                collection.expected_size(),
                collection.ef_construction(),
            )
            .unwrap()
    });
    let timed = graph.as_ref().and_then(DenseIndex::time_graph);
    let ((aside, aside_text), (in_place, in_place_text)) =
        collection.close_up_both_ways(graph, timed);
    assert_eq!(aside_text, in_place_text, "{label}: the id text let go of");
    assert_eq!(
        aside.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        in_place.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        "{label}"
    );
    for ((name, parts), (_, other_parts)) in aside.iter().zip(&in_place) {
        assert_eq!(parts.len(), other_parts.len(), "{label}: {name}");
        for (part, other) in parts.iter().zip(other_parts) {
            assert_eq!(part.0, other.0, "{label}: {name}");
            if *name == "index" && part.0 == "node_at" && !replace {
                assert_eq!(part.2, other.2, "{label}: node_at's values");
                assert_eq!(
                    part.1,
                    capacity.min(held.max(declared)),
                    "{label}: node_at keeps the records or the declared reservation"
                );
                assert_eq!(
                    other.1,
                    capacity.min(held.max(by_node)),
                    "{label}: the pass in place kept the graph's reservation"
                );
                continue;
            }
            assert!(
                part == other,
                "{label}: {name}'s {} differs: capacity {} against {}",
                part.0,
                part.1,
                other.1
            );
        }
    }
    assert!(collection.positions_agree(), "{label}");
}

/// From the same states, under raw, product quantized with and without raw
/// vectors, and scalar quantized declarations, each with a sparse space
/// under every removal policy and both weightings, the close-up built off to
/// the side leaves every structure the close-up made in place leaves: the id
/// store's entries, buckets, tags, live set and text, the metadata's
/// entries, every column's codes or values with the column store's live set
/// and extent, the dense index's numbering and the graph's positions by
/// node, and the sparse space's lists, record table, lengths, live and dead
/// sets and counts. Each at the same capacity, `node_at` aside, whose
/// capacity follows its own rule where the graph keeps its nodes. The states
/// are holes left by removals with the graph kept, holes with a replacement
/// graph built as a compaction builds it, a replacement with no hole, the
/// same before and after a quantized declaration trains, a declared size
/// below the records and above them, and, on a raw declaration, a column
/// traded for its plain form, every record removed, and holes whose removed
/// ids hold no text.
#[test]
fn the_close_up_built_aside_leaves_what_the_close_up_in_place_leaves() {
    for (kind, unlink, weighting, expected) in [
        (Kind::Raw, Unlink::Lazy, Weighting::Dot, 100usize),
        (Kind::Raw, Unlink::Strand, Weighting::BM25, 50_000),
        (Kind::PqWithRaw, Unlink::Eager, Weighting::Dot, 100),
        (Kind::PqOnly, Unlink::Strand, Weighting::BM25, 2_000),
        (Kind::Int8, Unlink::Lazy, Weighting::BM25, 100),
    ] {
        let label = format!("{kind:?} {unlink:?} {weighting:?} {expected}");
        let collection = collection(kind, expected, &["cat"], unlink, weighting);
        // Before training, where the declaration quantizes.
        add(&collection, 0..800);
        remove(&collection, (0..800).filter(|i| i % 3 == 1));
        both_ways(&collection, false, &format!("{label}: holes, untrained"));
        remove(&collection, (0..800).filter(|i| i % 5 == 2 && i % 3 != 1));
        both_ways(&collection, true, &format!("{label}: replaced, untrained"));
        // Past the training size, so a quantized declaration trains.
        add(&collection, 800..1_700);
        if kind != Kind::Raw {
            assert!(collection.is_quantized(), "{label}");
        }
        remove(&collection, (800..1_700).filter(|i| i % 4 == 3));
        both_ways(&collection, false, &format!("{label}: holes"));
        remove(
            &collection,
            (800..1_700).filter(|i| i % 7 == 1 && i % 4 != 3),
        );
        both_ways(&collection, true, &format!("{label}: replaced"));
        both_ways(
            &collection,
            true,
            &format!("{label}: replaced with no hole"),
        );
        remove(
            &collection,
            (0..1_700).filter(|i| collection.contains(&format!("r{i}"))),
        );
        assert_eq!(collection.len(), 0, "{label}");
        both_ways(
            &collection,
            false,
            &format!("{label}: every record removed"),
        );
    }

    // A column traded for its plain form: `u` holds a value of its own on
    // every record, which trades the dictionary once it holds 4,096.
    let plain = collection(Kind::Raw, 100, &["cat", "u"], Unlink::Lazy, Weighting::Dot);
    add(&plain, 0..4_400);
    remove(&plain, (0..4_400).filter(|i| i % 6 == 5));
    both_ways(&plain, false, "plain column: holes");
    remove(&plain, (0..4_400).filter(|i| i % 9 == 4 && i % 6 != 5));
    both_ways(&plain, true, "plain column: replaced");

    // Holes whose removed ids hold no text, so the arena is copied whole.
    let empty = collection(Kind::Raw, 100, &["cat"], Unlink::Lazy, Weighting::Dot);
    let mut named = record(0);
    named.id = String::new();
    let added = empty.add_records(vec![named], vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    add(&empty, 1..300);
    assert!(empty.remove_point(String::new()).unwrap());
    assert_eq!(empty.ids().dead_text(), 0);
    both_ways(&empty, false, "removed ids with no text");
}

// ============================================================================
// WHAT RUNS BESIDE A CLOSE-UP
// ============================================================================

/// The probes every state is asked: dense searches unfiltered, under the
/// declared field and under the mixed filter, a count under the mixed
/// filter, sparse searches unfiltered and under the declared field, and the
/// first page of `list`. Each answer as text.
struct Probes {
    queries: Vec<Vec<f32>>,
    sparse: SparseVector,
    declared: Filter,
    mixed: Filter,
}

impl Probes {
    fn new() -> Self {
        Probes {
            queries: (0..2).map(|q| record(1_000_000 + q).vector).collect(),
            sparse: SparseVector {
                dims: vec![1, 3, 21, 24],
                values: vec![1.0, 2.0, 1.0, 3.0],
            },
            declared: filter(json!({"cat": "b"})),
            mixed: mixed(),
        }
    }

    fn len(&self) -> usize {
        8
    }

    fn ask(&self, collection: &Collection, probe: usize) -> String {
        let params = || collection.search_params(K, Some(EF), false, None).unwrap();
        let dense = |query: &[f32], filter: Option<&Filter>| {
            let hits = collection.search_one(query, filter, params()).unwrap();
            format!(
                "{:?}",
                hits.iter()
                    .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                    .collect::<Vec<_>>()
            )
        };
        let sparse = |filter: Option<&Filter>| {
            let hits = collection
                .search_sparse(self.sparse.as_ref(), filter, K, IdfScope::Corpus)
                .unwrap();
            format!(
                "{:?}",
                hits.iter()
                    .map(|(id, score)| (id.clone(), score.to_bits()))
                    .collect::<Vec<_>>()
            )
        };
        match probe {
            0 => dense(&self.queries[0], None),
            1 => dense(&self.queries[1], None),
            2 => dense(&self.queries[0], Some(&self.declared)),
            3 => dense(&self.queries[1], Some(&self.mixed)),
            4 => collection.count(Some(&self.mixed)).to_string(),
            5 => sparse(None),
            6 => sparse(Some(&self.declared)),
            _ => {
                // Each record's fields in key order, since a map's own order
                // differs from one call to the next.
                let page: Vec<(String, BTreeMap<String, Value>)> = collection
                    .list(10, 0, None)
                    .unwrap()
                    .into_iter()
                    .map(|(id, fields)| (id, fields.into_iter().collect()))
                    .collect();
                format!("{page:?}")
            }
        }
    }

    fn all(&self, collection: &Collection) -> Vec<String> {
        (0..self.len()).map(|p| self.ask(collection, p)).collect()
    }
}

/// When something ran, from its start to its end.
type Span = (Instant, Instant);

/// One answer a thread beside the close-ups gave: the probe, the epochs it
/// started and ended in, when it ran, and what it answered.
struct Asked {
    probe: usize,
    epochs: (usize, usize),
    span: Span,
    answer: String,
}

/// The states a writer moves a collection through, as epochs a thread
/// beside it reads. An even epoch `2s` is state `s` and an odd one is a
/// change from it to the next in progress. A change that is not one step
/// for a reader, being an `add` of many records, runs with the threads
/// beside it paused, and moves the epoch two at once.
struct Epochs {
    epoch: AtomicUsize,
    paused: AtomicBool,
    active: AtomicUsize,
    stop: AtomicBool,
}

impl Epochs {
    fn new() -> Self {
        Epochs {
            epoch: AtomicUsize::new(0),
            paused: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        }
    }

    /// One change that is one step for a reader.
    fn change(&self, run: impl FnOnce()) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        run();
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// One change with every thread beside it paused.
    fn paused(&self, run: impl FnOnce()) {
        self.paused.store(true, Ordering::SeqCst);
        while self.active.load(Ordering::SeqCst) > 0 {
            std::thread::yield_now();
        }
        run();
        self.epoch.fetch_add(2, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Ask until stopped, each answer with its epochs and its span, or
    /// `None` where the writer has paused this thread.
    fn ask(&self, ask: impl Fn() -> String) -> Option<(usize, usize, Instant, Instant, String)> {
        let active = Active::enter(&self.active);
        if self.paused.load(Ordering::SeqCst) {
            drop(active);
            std::thread::sleep(Duration::from_micros(100));
            return None;
        }
        let first = self.epoch.load(Ordering::SeqCst);
        let start = Instant::now();
        let answer = ask();
        let end = Instant::now();
        let last = self.epoch.load(Ordering::SeqCst);
        drop(active);
        Some((first, last, start, end, answer))
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Stop the threads beside the writer once the guard returned is
    /// dropped, the writer's panic included, so a check the writer fails
    /// fails the test and never leaves the threads beside it asking.
    fn stop_on_drop(&self) -> StopOnDrop<'_> {
        StopOnDrop(&self.stop)
    }
}

/// One thread asking, from its read of the pause to its last read of the
/// epoch, let go when dropped, its panic included, so a thread that fails a
/// check never holds the writer's pause open.
struct Active<'a>(&'a AtomicUsize);

impl<'a> Active<'a> {
    fn enter(active: &'a AtomicUsize) -> Self {
        active.fetch_add(1, Ordering::SeqCst);
        Active(active)
    }
}

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Stops the threads beside a writer when dropped; see
/// [`Epochs::stop_on_drop`].
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Whether `asked` answered for one of the states its epochs span.
fn answered_a_state(asked: &Asked, states: &[Vec<String>]) -> bool {
    let first = asked.epochs.0 / 2;
    let last = asked.epochs.1.div_ceil(2);
    (first..=last).any(|state| states[state][asked.probe] == asked.answer)
}

/// Whether `span` overlaps any of `phases`.
fn overlaps(span: Span, phases: &[Span]) -> bool {
    phases
        .iter()
        .any(|&(start, end)| span.0 < end && span.1 > start)
}

/// The points this thread's close-ups reached, each with when.
type Marks = Rc<RefCell<Vec<(CloseUpPoint, Instant)>>>;

/// The build phases and the swap phases of this thread's close-ups, recorded
/// by the hook, each held open for a little while so the threads beside it
/// overlap it.
struct Phases {
    marks: Marks,
}

impl Phases {
    fn record(hold: Duration) -> Self {
        let marks: Marks = Rc::default();
        let kept = marks.clone();
        CLOSE_UP_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |point| {
                kept.borrow_mut().push((point, Instant::now()));
                if matches!(point, CloseUpPoint::Building | CloseUpPoint::Swapping) {
                    std::thread::sleep(hold);
                }
            }));
        });
        Phases { marks }
    }

    /// The build phases and the swap phases, from the point each begins to
    /// the point it ends.
    fn spans(&self) -> (Vec<Span>, Vec<Span>) {
        let marks = self.marks.borrow();
        let pairs = |from: CloseUpPoint, to: CloseUpPoint| {
            let starts = marks.iter().filter(|(point, _)| *point == from);
            let ends = marks.iter().filter(|(point, _)| *point == to);
            starts
                .zip(ends)
                .map(|((_, start), (_, end))| (*start, *end))
                .collect::<Vec<_>>()
        };
        (
            pairs(CloseUpPoint::Building, CloseUpPoint::Built),
            pairs(CloseUpPoint::Swapping, CloseUpPoint::Swapped),
        )
    }
}

impl Drop for Phases {
    fn drop(&mut self) {
        CLOSE_UP_HOOK.with(|hook| *hook.borrow_mut() = None);
    }
}

/// Run the close-ups of `kind` with three threads searching and one asking
/// for the statistics beside them, and hold every answer to the states
/// before and after the change it overlapped. Returns the close-ups run,
/// the answers that overlapped a build phase, and those that overlapped a
/// swap phase.
fn beside(kind: Kind, rounds: usize) -> (usize, usize, usize) {
    let collection = collection(kind, 100, &["cat"], Unlink::Lazy, Weighting::Dot);
    let probes = Probes::new();
    let epochs = Epochs::new();
    add(&collection, 0..2_000);
    assert!(narrowed(&collection, &probes.mixed), "{kind:?}");
    let mut states = vec![probes.all(&collection)];
    let phases = Phases::record(Duration::from_millis(5));
    let mut held: Vec<usize> = (0..2_000).collect();
    let mut next = 2_000usize;
    let mut close_ups = 0usize;
    let mut m = collection.m();

    let asked: Vec<Asked> = std::thread::scope(|scope| {
        let stop = epochs.stop_on_drop();
        let searchers: Vec<_> = (0..3)
            .map(|thread| {
                let (collection, probes, epochs) = (&collection, &probes, &epochs);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut probe = thread;
                    while !epochs.stopped() {
                        if let Some((first, last, start, end, answer)) =
                            epochs.ask(|| probes.ask(collection, probe))
                        {
                            out.push(Asked {
                                probe,
                                epochs: (first, last),
                                span: (start, end),
                                answer,
                            });
                            probe = (probe + 1) % probes.len();
                        }
                    }
                    out
                })
            })
            .collect();
        // Every check a debug build makes when the statistics are read, the
        // bound on holes left out, which a removal batch passes between its
        // own guards and its close-up's swap.
        let statistics = {
            let (collection, epochs) = (&collection, &epochs);
            scope.spawn(move || {
                let mut read = 0usize;
                while !epochs.stopped() {
                    if epochs
                        .ask(|| collection.stats().len().to_string())
                        .is_some()
                    {
                        read += 1;
                    }
                }
                read
            })
        };

        for round in 0..rounds {
            let fresh: Vec<usize> = (next..next + 4_200).collect();
            next += 4_200;
            epochs.paused(|| add(&collection, fresh.iter().copied()));
            held.extend(&fresh);
            states.push(probes.all(&collection));

            // One batch of the oldest 4,200 held, which passes the bound
            // and closes the positions up once it has let its guards go.
            let gone: Vec<usize> = held.drain(..4_200).collect();
            let before = holes(&collection);
            epochs.change(|| remove(&collection, gone.iter().copied()));
            assert_eq!(holes(&collection), 0, "{kind:?} round {round}");
            assert!(before + gone.len() > ALLOWANCE);
            close_ups += 1;
            states.push(probes.all(&collection));

            if round % 3 == 2 {
                epochs.change(|| {
                    collection.compact().unwrap();
                });
                close_ups += 1;
                states.push(probes.all(&collection));
            }
            if round % 4 == 1 {
                m = if m == 4 { 6 } else { 4 };
                let plan = collection.plan_rebuild(Some(m), None, None).unwrap();
                epochs.change(|| {
                    collection.rebuild(plan).unwrap();
                });
                close_ups += 1;
                states.push(probes.all(&collection));
            }
            if kind != Kind::Raw && round % 5 == 3 {
                epochs.change(|| assert!(collection.rebuild_with_quantization().unwrap()));
                close_ups += 1;
                states.push(probes.all(&collection));
            }
        }
        drop(stop);
        let statistics = statistics
            .join()
            .expect("the statistics never fail a check");
        assert!(statistics > 0, "{kind:?}");
        searchers
            .into_iter()
            .flat_map(|searcher| searcher.join().expect("no search fails"))
            .collect()
    });

    let (builds, swaps) = phases.spans();
    assert_eq!(
        builds.len(),
        close_ups,
        "{kind:?}: a build phase a close-up"
    );
    assert_eq!(swaps.len(), close_ups, "{kind:?}: a swap phase a close-up");
    let mut over_builds = 0usize;
    let mut over_swaps = 0usize;
    for asked in &asked {
        assert!(
            answered_a_state(asked, &states),
            "{kind:?}: probe {} in epochs {:?} answered {} where the states it spans answer {:?}",
            asked.probe,
            asked.epochs,
            asked.answer,
            (asked.epochs.0 / 2..=asked.epochs.1.div_ceil(2))
                .map(|state| &states[state][asked.probe])
                .collect::<Vec<_>>()
        );
        over_builds += usize::from(overlaps(asked.span, &builds));
        over_swaps += usize::from(overlaps(asked.span, &swaps));
    }
    eprintln!(
        "{kind:?}: {close_ups} close-ups, {} answers, {over_builds} overlapping a build phase, \
         {over_swaps} overlapping a swap phase",
        asked.len()
    );
    (close_ups, over_builds, over_swaps)
}

/// Searches run on three threads, and the statistics on a fourth, while the
/// writer runs close-ups on removal, compactions, rebuilds at another
/// degree and the training rebuilds, under raw, product quantized and
/// scalar quantized declarations with a sparse space. Every search, every
/// count under the filter the columns only bound and every page of `list`
/// answers what the collection answers at rest in the state before the
/// change it overlapped or the state after it, and never anything else, and
/// every read of the statistics passes every check but the bound on holes.
/// The build and the swap of every close-up are held open for a little
/// while, so thousands of answers overlap a build and hundreds a swap.
#[test]
fn searches_beside_every_close_up_see_the_state_before_or_after_it() {
    let mut totals = (0usize, 0usize, 0usize);
    for kind in [Kind::Raw, Kind::PqWithRaw, Kind::Int8] {
        let (close_ups, builds, swaps) = beside(kind, 24);
        totals = (totals.0 + close_ups, totals.1 + builds, totals.2 + swaps);
    }
    eprintln!(
        "every kind: {} close-ups, {} answers overlapping a build phase, {} overlapping a swap \
         phase",
        totals.0, totals.1, totals.2
    );
    assert!(totals.1 >= 2_000, "thousands of answers overlap a build");
    assert!(totals.2 >= 200, "hundreds of answers overlap a swap");
}

// ============================================================================
// WHAT WAITS FOR A CLOSE-UP
// ============================================================================

/// Searches complete while a close-up's build is held open, since the build
/// holds read guards alone. While its swap is held open, which holds every
/// write guard, a search waits and completes once the swap lets them go.
/// A second writer waits for the whole close-up, build and swap, since the
/// close-up runs under the `writers` the removal took. The search that
/// waited answers the closed-up collection, which answers what it answered
/// before the close-up.
#[test]
fn searches_run_while_the_build_is_held_and_wait_only_for_the_swap() {
    let collection = collection(Kind::Raw, 100, &["cat"], Unlink::Lazy, Weighting::Dot);
    add(&collection, 0..6_000);
    remove(&collection, 0..ALLOWANCE);
    assert_eq!(
        holes(&collection),
        ALLOWANCE,
        "the allowance reached and not passed"
    );
    let probes = Probes::new();
    let wait = Duration::from_millis(300);

    std::thread::scope(|scope| {
        let (reached_tx, reached) = mpsc::channel::<CloseUpPoint>();
        let (go, go_rx) = mpsc::channel::<()>();
        let (swapped_tx, swapped) = mpsc::channel::<Instant>();
        let remover = scope.spawn(|| {
            CLOSE_UP_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move |point| {
                    if matches!(point, CloseUpPoint::Building | CloseUpPoint::Swapping) {
                        reached_tx.send(point).unwrap();
                        go_rx.recv().unwrap();
                    }
                    if point == CloseUpPoint::Swapped {
                        swapped_tx.send(Instant::now()).unwrap();
                    }
                }));
            });
            let removed = collection.remove_point(format!("r{ALLOWANCE}")).unwrap();
            CLOSE_UP_HOOK.with(|hook| *hook.borrow_mut() = None);
            removed
        });

        // The build is held open, with its read guards.
        assert_eq!(reached.recv().unwrap(), CloseUpPoint::Building);
        let (answers_tx, answers) = mpsc::channel();
        {
            let (collection, probes) = (&collection, &probes);
            scope.spawn(move || {
                let started = Instant::now();
                let all: Vec<Vec<String>> = (0..5).map(|_| probes.all(collection)).collect();
                answers_tx.send((all, started.elapsed())).unwrap();
            });
        }
        let (during_build, took) = answers
            .recv_timeout(Duration::from_secs(20))
            .expect("searches complete while the build is held open");
        let (writer_tx, writer) = mpsc::channel();
        {
            let collection = &collection;
            scope.spawn(move || {
                assert!(collection.detach_sink().is_none());
                writer_tx.send(Instant::now()).unwrap();
            });
        }
        assert!(
            writer.recv_timeout(wait).is_err(),
            "a second writer waits while the build is held"
        );
        go.send(()).unwrap();

        // The swap is held open, with every write guard.
        assert_eq!(reached.recv().unwrap(), CloseUpPoint::Swapping);
        let (waited_tx, waited) = mpsc::channel();
        {
            let (collection, probes) = (&collection, &probes);
            scope.spawn(move || {
                let started = Instant::now();
                let answer = probes.all(collection);
                waited_tx.send((answer, started, Instant::now())).unwrap();
            });
        }
        assert!(
            waited.recv_timeout(wait).is_err(),
            "a search waits while the swap holds the write guards"
        );
        assert!(
            writer.recv_timeout(Duration::from_millis(50)).is_err(),
            "a second writer waits while the swap is held"
        );
        let released = Instant::now();
        go.send(()).unwrap();

        assert!(remover.join().unwrap());
        let (after_swap, started, ended) = waited.recv().unwrap();
        assert!(started < released && ended > released);
        // The close-up reaches its last point under the `writers` the
        // removal holds, so the second writer, which takes them, goes after.
        let wrote = writer.recv().unwrap();
        assert!(
            wrote >= swapped.recv().unwrap(),
            "the second writer went once the close-up was done"
        );
        assert_eq!(holes(&collection), 0, "the close-up ran");
        assert_eq!(after_swap, probes.all(&collection));
        for answers in &during_build {
            assert_eq!(answers, &after_swap, "a close-up moves no answer");
        }
        eprintln!(
            "five rounds of every probe took {took:?} while the build was held open, and the \
             search that met the swap waited {:?}",
            ended - started
        );
    });
}

// ============================================================================
// COUNT UNDER A FILTER THE COLUMNS ONLY BOUND
// ============================================================================

/// A count under a filter naming a declared field and an undeclared one
/// selects the bound under the metadata's guard and the columns', and walks
/// it under both. A removal batch that passes the bound and closes up, sent
/// while the count is between its selection and its walk, waits for the
/// count, so the count is the one before the removal; and once the count is
/// done, the removal closes up and a second count is the one after it.
#[test]
fn a_count_holds_its_bound_while_a_close_up_waits_for_it() {
    let collection = collection(Kind::Raw, 100, &["cat"], Unlink::Lazy, Weighting::Dot);
    add(&collection, 0..12_000);
    let filter = mixed();
    assert!(narrowed(&collection, &filter));
    let model = |records: std::ops::Range<usize>| {
        records
            .map(record)
            .filter(|record| matches_filter(&record.metadata, &filter))
            .count()
    };
    let (before, after) = (model(0..12_000), model(6_001..12_000));
    assert!(before > after && after > 0);
    assert_eq!(collection.count(Some(&filter)), before);

    std::thread::scope(|scope| {
        let (selected_tx, selected) = mpsc::channel::<()>();
        let (closed, closed_rx) = mpsc::channel::<()>();
        let counter = scope.spawn(|| {
            let (walking_tx, walking) = mpsc::channel::<Instant>();
            COUNT_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    selected_tx.send(()).unwrap();
                    // Wait for the close-up to be done, or give up: the
                    // removal waits for this count's guards, so it is never
                    // done while they are held.
                    let _ = closed_rx.recv_timeout(Duration::from_millis(300));
                    walking_tx.send(Instant::now()).unwrap();
                }));
            });
            let counted = collection.count(Some(&filter));
            COUNT_HOOK.with(|hook| *hook.borrow_mut() = None);
            (counted, walking.recv().unwrap())
        });
        selected.recv().unwrap();
        let remover = {
            let collection = &collection;
            scope.spawn(move || {
                let (building_tx, building) = mpsc::channel::<Instant>();
                CLOSE_UP_HOOK.with(|hook| {
                    *hook.borrow_mut() = Some(Box::new(move |point| {
                        if point == CloseUpPoint::Building {
                            building_tx.send(Instant::now()).unwrap();
                        }
                    }));
                });
                remove(collection, 0..6_001);
                CLOSE_UP_HOOK.with(|hook| *hook.borrow_mut() = None);
                let _ = closed.send(());
                building.recv().unwrap()
            })
        };
        let (counted, walking_at) = counter.join().unwrap();
        let building_at = remover.join().unwrap();
        assert_eq!(counted, before, "the count read its bound under one hold");
        // The count walks its bound under guards the removal's batch takes,
        // so the close-up after the batch begins once the walk is done.
        assert!(building_at > walking_at, "the removal waited for the count");
    });
    assert_eq!(holes(&collection), 0, "the removal closed up");
    assert_eq!(collection.count(Some(&filter)), after);
}

/// Counts run on one thread, under the filter the columns only bound,
/// while the writer runs removal batches that close up, compactions,
/// rebuilds at another degree and a clear. Every count is the model's count
/// for the state before the change it overlapped or the state after it.
#[test]
fn counts_beside_every_change_see_the_state_before_or_after_it() {
    let collection = collection(Kind::Raw, 100, &["cat"], Unlink::Lazy, Weighting::Dot);
    let filter = mixed();
    let epochs = Epochs::new();
    let mut held: Vec<usize> = Vec::new();
    let mut next = 0usize;
    let model = |held: &[usize]| {
        held.iter()
            .filter(|&&i| matches_filter(&record(i).metadata, &filter))
            .count()
            .to_string()
    };
    let mut states: Vec<Vec<String>> = vec![vec![model(&held)]];
    let mut changes = 0usize;
    let mut m = collection.m();
    let phases = Phases::record(Duration::from_millis(2));

    let asked: Vec<Asked> = std::thread::scope(|scope| {
        let stop = epochs.stop_on_drop();
        let counter = {
            let (collection, epochs, filter) = (&collection, &epochs, &filter);
            scope.spawn(move || {
                let mut out = Vec::new();
                while !epochs.stopped() {
                    if let Some((first, last, start, end, answer)) =
                        epochs.ask(|| collection.count(Some(filter)).to_string())
                    {
                        out.push(Asked {
                            probe: 0,
                            epochs: (first, last),
                            span: (start, end),
                            answer,
                        });
                    }
                }
                out
            })
        };
        for round in 0..16 {
            let fresh: Vec<usize> = (next..next + 4_500).collect();
            next += 4_500;
            epochs.paused(|| add(&collection, fresh.iter().copied()));
            held.extend(&fresh);
            states.push(vec![model(&held)]);
            assert!(narrowed(&collection, &filter), "round {round}");

            let gone: Vec<usize> = held.drain(..4_300).collect();
            epochs.change(|| remove(&collection, gone.iter().copied()));
            assert_eq!(holes(&collection), 0, "round {round}: the close-up ran");
            states.push(vec![model(&held)]);
            changes += 1;

            match round % 4 {
                1 => {
                    epochs.change(|| {
                        collection.compact().unwrap();
                    });
                    states.push(vec![model(&held)]);
                    changes += 1;
                }
                2 => {
                    m = if m == 4 { 6 } else { 4 };
                    let plan = collection.plan_rebuild(Some(m), None, None).unwrap();
                    epochs.change(|| {
                        collection.rebuild(plan).unwrap();
                    });
                    states.push(vec![model(&held)]);
                    changes += 1;
                }
                3 if round == 15 => {
                    epochs.change(|| {
                        collection.clear().unwrap();
                    });
                    held.clear();
                    states.push(vec![model(&held)]);
                    changes += 1;
                }
                _ => {}
            }
        }
        drop(stop);
        counter.join().expect("no count fails")
    });

    let (builds, swaps) = phases.spans();
    let mut overlapping = 0usize;
    let mut over_builds = 0usize;
    let mut over_swaps = 0usize;
    let mut values: Vec<&str> = Vec::new();
    for asked in &asked {
        assert!(
            answered_a_state(asked, &states),
            "a count in epochs {:?} was {} where the states it spans count {:?}",
            asked.epochs,
            asked.answer,
            (asked.epochs.0 / 2..=asked.epochs.1.div_ceil(2))
                .map(|state| &states[state][0])
                .collect::<Vec<_>>()
        );
        overlapping += usize::from(asked.epochs.0 != asked.epochs.1 || asked.epochs.0 % 2 == 1);
        over_builds += usize::from(overlaps(asked.span, &builds));
        over_swaps += usize::from(overlaps(asked.span, &swaps));
        if !values.contains(&asked.answer.as_str()) {
            values.push(&asked.answer);
        }
    }
    eprintln!(
        "{} counts beside {changes} changes and {} close-ups, {overlapping} overlapping a \
         change, {over_builds} a build phase and {over_swaps} a swap phase, {} values counted",
        asked.len(),
        builds.len(),
        values.len()
    );
    assert!(over_builds >= 100, "counts overlap the build phases");
    assert!(over_swaps >= 10, "counts overlap the swap phases");
}
