//! Dense positions, held by a property test.
//!
//! A record's position names it in every per-record structure. Positions run
//! from zero in internal id order, with no gap but the holes removals leave,
//! and a compaction, a rebuild, the training rebuild, a load and a clear
//! close them, as does a removal batch that leaves more holes than 4,096 and
//! the records held. One seeded generator drives random sequences of writes
//! on a collection of up to 2,000 records at dimension 8, beside a reference
//! model that holds, for each external id, the internal id, the vector, the
//! fields and the sparse half the record was given, and the holes each
//! removal batch leaves. Each run declares one field and a sparse space, and
//! the runs cover raw, product quantized with and without raw vectors, and
//! scalar quantized declarations, each with a training size the sequence
//! crosses. The writes include churn, overwrites of most of the records call
//! after call, so the close-up on removal runs in every sequence and in the
//! replay of a journal.
//!
//! After every step the collection answers what the model answers. Its
//! positions agree, its holes and its close-ups on removal are the model's,
//! its records, ids, fields and sparse halves are the model's, `count`,
//! `list` and `get_records` are the model's, every exact path is brute force
//! over the model, and every traversal returns the page the same graph
//! returns under a test by internal id that the model answers, read through
//! the nodes' origin ids. After every save the artefacts list the model's
//! records in increasing internal id, the graph dump holds each record's row
//! once, a load answers the same, and a second save writes the same bytes.
//! After a recovery from a checkpoint and a journal, the recovered collection
//! holds the holes the model replays and answers and saves what the live one
//! does. After every compaction, rebuild and load, every per-record structure
//! reaches the records held and no further.

#![allow(clippy::disallowed_types)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use zeusdb_vector_core::{
    compile_filter, matches_filter, unframe, Error, Filter, FrameKind, IdfScope, JournalRecord,
    Operation, OperationKind, Position, SparseVector, VectorIndex,
};
use zeusdb_vector_sparse::{SparseConfig, Unlink, Weighting};

use super::{
    Collection, Declaration, Int8Scale, OperationSink, ParsedRecord, SparseHalf, StorageMode,
};
use crate::journal::Durability;

/// A directory under the system's temporary directory, removed on drop
/// together with the journals beside anything in it.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-positions-tests-{}-{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ============================================================================
// THE GENERATOR AND THE MODEL
// ============================================================================

/// The seeded generator, splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    /// True `percent` times in a hundred.
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

thread_local! {
    /// What each run exercised, by name, for the run to hold to its floor.
    static TALLY: std::cell::RefCell<BTreeMap<String, usize>> =
        const { std::cell::RefCell::new(BTreeMap::new()) };
}

fn tally(what: &str) {
    TALLY.with(|tally| *tally.borrow_mut().entry(what.to_string()).or_default() += 1);
}

/// Keep the largest of the values `what` reached.
fn tally_most(what: &str, value: usize) {
    TALLY.with(|tally| {
        let mut tally = tally.borrow_mut();
        let most = tally.entry(what.to_string()).or_default();
        *most = (*most).max(value);
    });
}

/// Dimension of every vector.
const DIM: usize = 8;
/// The most records a sequence holds.
const MOST: usize = 2_000;
/// The training size every quantized declaration names.
const TRAINING: usize = 1_000;
/// Dimensions a sparse half draws from.
const TERMS: u32 = 48;
/// The page size every search asks for, and the traversal width.
const K: usize = 10;
const EF: usize = 40;
/// The holes a collection keeps whatever its size. A removal batch that
/// leaves more than this and more than the records held closes them up.
const ALLOWANCE: usize = 4_096;

/// The close-ups on removal this thread has run, live and in replays.
fn close_ups() -> usize {
    super::insert::CLOSE_UPS_ON_REMOVAL.with(|count| count.get())
}

/// The positions below the extent that hold no record.
fn holes(collection: &Collection) -> usize {
    let ids = collection.ids.read().unwrap();
    ids.extent() - ids.len()
}

/// What a run declares besides the field and the sparse space.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Raw,
    PqWithRaw,
    PqOnly,
    Int8,
}

/// One record as the model holds it.
#[derive(Clone, Debug)]
struct Held {
    internal: usize,
    vector: Vec<f32>,
    fields: HashMap<String, Value>,
    sparse: Option<(Vec<u32>, Vec<f32>)>,
}

/// What the collection must hold, by external id, and the counters the
/// collection keeps beside it.
#[derive(Clone, Debug, Default)]
struct Model {
    held: BTreeMap<String, Held>,
    counter: usize,
    /// One past the largest internal id a sparse half was inserted under
    /// since the space was made or cleared, which the artefact records.
    sparse_slots: usize,
    /// The positions the collection holds no record at. A removal batch
    /// adds one for each record it removes and closes them all up where
    /// they pass the allowance and the records held, and a compaction, a
    /// rebuild, a training, a load and a clear close them up.
    holes: usize,
    /// The close-ups on removal the collection has run.
    passes: usize,
    /// While a journal is attached, what a recovery from its last
    /// checkpoint replays: the checkpoint loads with no holes, and the
    /// journal's removal batches follow the same rule.
    replay: Option<Replayed>,
}

/// The holes and the close-ups on removal of a replay since its checkpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Replayed {
    holes: usize,
    passes: usize,
}

impl Model {
    fn len(&self) -> usize {
        self.held.len()
    }

    /// Every record's external id and fields, in increasing internal id.
    fn by_internal(&self) -> Vec<(&str, &Held)> {
        let mut out: Vec<(&str, &Held)> = self
            .held
            .iter()
            .map(|(name, held)| (name.as_str(), held))
            .collect();
        out.sort_by_key(|(_, held)| held.internal);
        out
    }

    fn name_of_internal(&self) -> HashMap<usize, String> {
        self.held
            .iter()
            .map(|(name, held)| (held.internal, name.clone()))
            .collect()
    }

    /// The model's side of one insert, overwrite or not: a held id loses
    /// its record and every record takes the next internal id, in batch
    /// order.
    fn insert(&mut self, record: &ParsedRecord) {
        self.counter += 1;
        let sparse = match &record.sparse {
            Some(SparseHalf::Vector(vector)) => Some((vector.dims.clone(), vector.values.clone())),
            Some(SparseHalf::Terms(_)) => unreachable!("no text layer is declared"),
            None => None,
        };
        if sparse.is_some() {
            self.sparse_slots = self.sparse_slots.max(self.counter + 1);
        }
        self.held.insert(
            record.id.clone(),
            Held {
                internal: self.counter,
                vector: record.vector.clone(),
                fields: record.metadata.clone(),
                sparse,
            },
        );
    }

    fn clear(&mut self) {
        self.held.clear();
        self.counter = 0;
        self.sparse_slots = 0;
        self.closed_up();
    }

    /// One removal batch, which removed `removed` records the model has
    /// already let go of, live and in a replay of the journal. A batch that
    /// leaves more holes than the allowance and the records held closes
    /// them up once it is done.
    fn removal_batch(&mut self, removed: usize) {
        let held = self.len();
        let batch = |holes: &mut usize, passes: &mut usize| {
            *holes += removed;
            let pass = *holes > ALLOWANCE.max(held);
            if pass {
                *holes = 0;
                *passes += 1;
            }
            pass
        };
        if batch(&mut self.holes, &mut self.passes) {
            tally("close-up on removal");
        }
        if let Some(replay) = &mut self.replay {
            if batch(&mut replay.holes, &mut replay.passes) {
                tally("close-up on removal in a journal");
            }
        }
    }

    /// Every position closed up, which a compaction, a rebuild, a training
    /// and a clear do, live and in a replay.
    fn closed_up(&mut self) {
        self.holes = 0;
        if let Some(replay) = &mut self.replay {
            replay.holes = 0;
        }
    }
}

/// A vector of `DIM` values on a grid of a quarter, so some distances tie.
fn vector(rng: &mut Rng) -> Vec<f32> {
    (0..DIM)
        .map(|_| (rng.below(41) as f32 - 20.0) * 0.25)
        .collect()
}

/// Fields for one record: the declared `cat` most of the time, the
/// undeclared `i` and `tag` some of the time, and now and then none.
fn fields(rng: &mut Rng) -> HashMap<String, Value> {
    let mut fields = HashMap::new();
    if rng.chance(5) {
        return fields;
    }
    if rng.chance(90) {
        fields.insert("cat".to_string(), json!(["a", "b", "c", "d"][rng.below(4)]));
    }
    if rng.chance(80) {
        fields.insert("i".to_string(), json!(rng.below(1_000)));
    }
    if rng.chance(50) {
        fields.insert("tag".to_string(), json!(["x", "y"][rng.below(2)]));
    }
    fields
}

/// A sparse half seven times in ten, of one to five terms with values
/// that are whole numbers, so every length and total is exact.
fn sparse_half(rng: &mut Rng) -> Option<SparseHalf> {
    if !rng.chance(70) {
        return None;
    }
    let mut dims: Vec<u32> = (0..1 + rng.below(5))
        .map(|_| rng.below(TERMS as usize) as u32)
        .collect();
    dims.sort_unstable();
    dims.dedup();
    let values = dims.iter().map(|_| 1.0 + rng.below(3) as f32).collect();
    Some(SparseHalf::Vector(SparseVector { dims, values }))
}

fn record(rng: &mut Rng, id: String) -> ParsedRecord {
    ParsedRecord {
        id,
        vector: vector(rng),
        sparse: sparse_half(rng),
        metadata: fields(rng),
    }
}

// ============================================================================
// WHAT IS ASKED AFTER EVERY STEP
// ============================================================================

/// The queries and filters every check asks.
struct Probe {
    queries: Vec<Vec<f32>>,
    sparse: Vec<SparseVector>,
    /// A filter on the declared field alone, one on undeclared fields alone,
    /// and one on both.
    filters: Vec<(&'static str, Filter)>,
    idf: Vec<IdfScope>,
}

impl Probe {
    fn new(rng: &mut Rng, weighting: &Weighting) -> Self {
        let filter = |value: Value| {
            let map: HashMap<String, Value> = serde_json::from_value(value).unwrap();
            compile_filter(&map).unwrap()
        };
        let sparse = (0..3)
            .map(|q| {
                let dims: Vec<u32> = vec![q, 7 + q * 5, 20 + q * 3, 31 + q];
                let values = dims.iter().map(|&d| 1.0 + (d % 3) as f32).collect();
                SparseVector { dims, values }
            })
            .collect();
        Probe {
            queries: (0..3).map(|_| vector(rng)).collect(),
            sparse,
            filters: vec![
                ("declared", filter(json!({"cat": "b"}))),
                ("undeclared", filter(json!({"i": {"lt": 400}}))),
                (
                    "mixed",
                    filter(json!({"cat": {"in": ["a", "c"]}, "tag": "x"})),
                ),
            ],
            idf: match weighting {
                Weighting::Bm25 { .. } => vec![IdfScope::Corpus, IdfScope::Global],
                _ => vec![IdfScope::Corpus],
            },
        }
    }
}

/// One page, as external ids and score bits.
type Page = Vec<(String, u32)>;

fn bits(page: Vec<(String, f32)>) -> Page {
    page.into_iter()
        .map(|(name, score)| (name, score.to_bits()))
        .collect()
}

/// Whether the model's record passes `filter`, where one is given.
fn admits(held: &Held, filter: Option<&Filter>) -> bool {
    filter.is_none_or(|filter| matches_filter(&held.fields, filter))
}

/// The vector `get_records` returns for a record the model holds: the one
/// it was given where the collection keeps raw vectors, and the
/// reconstruction of its code where it keeps codes alone.
fn expected_vector(collection: &Collection, kind: Kind, held: &Held) -> Vec<f32> {
    if !collection.is_quantized() || kind == Kind::PqWithRaw || kind == Kind::Raw {
        return held.vector.clone();
    }
    match kind {
        Kind::PqOnly => {
            let pq = collection.pq().unwrap();
            pq.reconstruct(&pq.quantize(&held.vector).unwrap()).unwrap()
        }
        Kind::Int8 => {
            let index = collection.dense().index.read().unwrap();
            let graph = index.graph();
            let row = graph.int8_encode(&held.vector).unwrap();
            graph.int8_codec().unwrap().reconstruct(&row).unwrap()
        }
        _ => unreachable!(),
    }
}

/// The node each internal id names in the graph, read off the nodes'
/// origin ids, so the reference never reads a position.
fn nodes_by_id(collection: &Collection) -> HashMap<usize, u32> {
    let index = collection.dense().index.read().unwrap();
    let graph = index.graph();
    let mut out = HashMap::new();
    for node in 0..graph.nb_points() as u32 {
        let previous = out.insert(graph.origin_id_at(node), node);
        assert!(previous.is_none(), "two nodes share an internal id");
    }
    out
}

/// The exact page brute force over the model gives: every admitted record
/// scored from the node its internal id names, ordered by score and then by
/// external id, cut to the page.
fn brute_dense(collection: &Collection, model: &Model, query: &[f32], filter: &Filter) -> Page {
    let nodes = nodes_by_id(collection);
    let index = collection.dense().index.read().unwrap();
    let mut scored: Vec<(String, f32)> = model
        .held
        .iter()
        .filter(|(_, held)| admits(held, Some(filter)))
        .map(|(name, held)| {
            let node = nodes[&held.internal];
            (name.clone(), index.exact_distance_of_node(query, node))
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(K);
    bits(scored)
}

/// The sparse page brute force over the model gives under `weighting`:
/// every admitted record holding a sparse half scored term by term in
/// dimension order, the records scoring nothing dropped, ordered by score
/// downward and then by external id, cut to the page.
fn brute_sparse(
    model: &Model,
    weighting: &Weighting,
    query: &SparseVector,
    filter: Option<&Filter>,
    idf: IdfScope,
) -> Page {
    let with_sparse: Vec<(&String, &Held)> = model
        .held
        .iter()
        .filter(|(_, held)| held.sparse.is_some())
        .collect();
    let length = |held: &Held| -> f32 {
        let (_, values) = held.sparse.as_ref().unwrap();
        values.iter().map(|&v| v as f64).sum::<f64>() as f32
    };
    // The query as the scan weights it.
    let (dims, values, c0, c1): (Vec<u32>, Vec<f32>, f32, f32) = match weighting {
        Weighting::Bm25 { k1, b } => {
            let corpus: Vec<&Held> = with_sparse
                .iter()
                .filter(|(_, held)| match idf {
                    IdfScope::Global => true,
                    IdfScope::Corpus => admits(held, filter),
                })
                .map(|(_, held)| *held)
                .collect();
            let n = corpus.len() as f64;
            let numerator = *k1 as f64 + 1.0;
            let mut dims = Vec::new();
            let mut values = Vec::new();
            for (&d, &qw) in query.dims.iter().zip(&query.values) {
                let df = corpus
                    .iter()
                    .filter(|held| held.sparse.as_ref().unwrap().0.contains(&d))
                    .count();
                if df == 0 {
                    continue;
                }
                let df = df as f64;
                let rarity = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                dims.push(d);
                values.push((qw as f64 * rarity * numerator) as f32);
            }
            let total: f64 = with_sparse
                .iter()
                .map(|(_, held)| length(held) as f64)
                .sum();
            let mean = if with_sparse.is_empty() {
                0.0
            } else {
                total / with_sparse.len() as f64
            };
            let mean = if mean > 0.0 { mean } else { 1.0 };
            (
                dims,
                values,
                k1 * (1.0 - b),
                (*k1 as f64 * *b as f64 / mean) as f32,
            )
        }
        _ => (query.dims.clone(), query.values.clone(), 0.0, 0.0),
    };
    let bm25 = matches!(weighting, Weighting::Bm25 { .. });
    let mut scored: Vec<(String, f32)> = Vec::new();
    for (name, held) in &with_sparse {
        if !admits(held, filter) {
            continue;
        }
        let (rdims, rvalues) = held.sparse.as_ref().unwrap();
        let mut sum: Option<f32> = None;
        for (&d, &qw) in dims.iter().zip(&values) {
            let Ok(at) = rdims.binary_search(&d) else {
                continue;
            };
            let stored = rvalues[at];
            let s = if bm25 {
                qw * stored / (stored + c0 + c1 * length(held))
            } else {
                stored * qw
            };
            sum = Some(match sum {
                None => s,
                Some(total) => total + s,
            });
        }
        if let Some(score) = sum {
            if score != 0.0 {
                scored.push(((*name).clone(), score));
            }
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(K);
    bits(scored)
}

/// The collection's own traversal under each admit set its filters give,
/// against the same graph traversed under the model's test by internal id.
fn check_traversals(collection: &Collection, model: &Model, probe: &Probe, step: &str) {
    let names = model.name_of_internal();
    let mut filters: Vec<(&str, Option<&Filter>)> = vec![("none", None)];
    filters.extend(
        probe
            .filters
            .iter()
            .map(|(name, filter)| (*name, Some(filter))),
    );
    for query in &probe.queries {
        for (label, filter) in &filters {
            let page: Page = {
                let ids = collection.ids.read().unwrap();
                let index = collection.dense().index.read().unwrap();
                let metadata = collection.vector_metadata.read().unwrap();
                let columns = collection.columns.read().unwrap();
                let plan = collection.admit_plan(*filter, &columns, &ids, &metadata);
                tally(match (&plan, index.stranded() > 0) {
                    (super::search::AdmitPlan::All, false) => "traversal with no predicate",
                    (super::search::AdmitPlan::All, true) => "traversal past stranded nodes",
                    (super::search::AdmitPlan::Bitmap(_), _) => "traversal under a bitmap",
                    _ => "traversal under a table call",
                });
                let hits = plan.run(|admit| index.traverse_under(query, K, EF, admit));
                hits.items
                    .iter()
                    .map(|hit| {
                        let name = ids
                            .name(hit.position.slot())
                            .unwrap_or_else(|| panic!("{step}: a hit names no record"));
                        (name.to_string(), hit.score.to_bits())
                    })
                    .collect()
            };
            let reference: Page = {
                let mut index = collection.dense().index.write().unwrap();
                let admitted = |id: usize| {
                    names
                        .get(&id)
                        .is_some_and(|name| admits(&model.held[name], *filter))
                };
                index
                    .traverse_by_id(query, K, EF, &admitted)
                    .into_iter()
                    .map(|(id, score)| (names[&id].clone(), score.to_bits()))
                    .collect()
            };
            assert_eq!(page, reference, "{step}: the traversal under {label}");
        }
    }
}

/// Every page a collection answers through its public calls: the
/// unfiltered and each filtered dense search, and the sparse arm under each
/// filter and scope.
fn pages(collection: &Collection, probe: &Probe) -> Vec<Page> {
    let mut out = Vec::new();
    let params = || collection.search_params(K, Some(EF), false, None).unwrap();
    for query in &probe.queries {
        let hits = collection.search_one(query, None, params()).unwrap();
        out.push(
            hits.iter()
                .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                .collect(),
        );
        for (_, filter) in &probe.filters {
            let hits = collection
                .search_one(query, Some(filter), params())
                .unwrap();
            out.push(
                hits.iter()
                    .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                    .collect(),
            );
        }
    }
    for query in &probe.sparse {
        for filter in std::iter::once(None).chain(probe.filters.iter().map(|(_, f)| Some(f))) {
            for &idf in &probe.idf {
                let hits = collection
                    .search_sparse(query.as_ref(), filter, K, idf)
                    .unwrap();
                out.push(bits(hits));
            }
        }
    }
    out
}

/// Everything asked after every step, of a collection that holds `holes`
/// holes.
fn check(
    collection: &Collection,
    model: &Model,
    holes: usize,
    kind: Kind,
    weighting: &Weighting,
    probe: &Probe,
    step: &str,
) {
    assert!(collection.positions_agree(), "{step}: the positions agree");
    let (extent, held) = {
        let ids = collection.ids.read().unwrap();
        (ids.extent(), ids.len())
    };
    assert_eq!(extent - held, holes, "{step}: the holes");
    assert!(
        extent - held <= ALLOWANCE.max(held),
        "{step}: the holes are within the allowance or the records held"
    );
    if extent > model.len() {
        tally("a state with holes");
    }
    tally_most("most records held", model.len());
    assert_eq!(collection.len(), model.len(), "{step}: the record count");
    assert_eq!(collection.count(None), model.len(), "{step}: count");
    assert_eq!(
        collection.id_counter(),
        model.counter,
        "{step}: the counter"
    );

    // The ids, in increasing internal id, which is position order.
    let expected: Vec<(usize, String)> = model
        .by_internal()
        .into_iter()
        .map(|(name, held)| (held.internal, name.to_string()))
        .collect();
    assert_eq!(collection.internal_ids(), expected, "{step}: the ids");

    // Every record's fields and vector.
    let names: Vec<String> = model.held.keys().cloned().collect();
    let views = collection.records(names.clone(), true, true).unwrap();
    assert_eq!(views.len(), names.len(), "{step}");
    for view in &views {
        let held = &model.held[&view.id];
        assert_eq!(
            view.metadata,
            held.fields.clone().into_iter().collect::<BTreeMap<_, _>>(),
            "{step}: {}'s fields",
            view.id
        );
        let want = expected_vector(collection, kind, held);
        let got = view.vector.as_ref().unwrap();
        assert!(
            got.iter()
                .map(|v| v.to_bits())
                .eq(want.iter().map(|v| v.to_bits())),
            "{step}: {}'s vector",
            view.id
        );
    }

    // The codes the collection keeps by external id, where it keeps any.
    if collection.is_quantized() && matches!(kind, Kind::PqWithRaw | Kind::PqOnly) {
        let pq = collection.pq().unwrap();
        let codes = collection.pq_codes();
        assert_eq!(codes.len(), model.len(), "{step}: one code a record");
        for (name, held) in &model.held {
            assert_eq!(
                codes.get(name),
                Some(&pq.quantize(&held.vector).unwrap()),
                "{step}: {name}'s code"
            );
        }
    }

    // Every sparse half, by its length at the record's position, and the
    // document frequency of every term.
    {
        let ids = collection.ids.read().unwrap();
        let sparse = collection.sparse().unwrap().index.read().unwrap();
        assert!(sparse.keys_agree(), "{step}: the sparse keys");
        let mut df = vec![0usize; TERMS as usize];
        let mut held_sparse = 0usize;
        for (name, held) in &model.held {
            let at = ids.slot_of(name).unwrap();
            let length = sparse.length_of(Position::from_slot(at));
            match &held.sparse {
                Some((dims, values)) => {
                    held_sparse += 1;
                    let want = values.iter().map(|&v| v as f64).sum::<f64>() as f32;
                    assert_eq!(length, Some(want), "{step}: {name}'s sparse length");
                    for &d in dims {
                        df[d as usize] += 1;
                    }
                }
                None => assert_eq!(length, None, "{step}: {name} holds no sparse half"),
            }
        }
        assert_eq!(sparse.len(), held_sparse, "{step}: sparse records");
        for (d, &count) in df.iter().enumerate() {
            assert_eq!(sparse.df(d as u32), count, "{step}: df of {d}");
        }
    }

    // `list`, in windows and after a cursor.
    let ordered = model.by_internal();
    let window = |from: usize, number: usize| -> Vec<(String, BTreeMap<String, Value>)> {
        ordered
            .iter()
            .skip(from)
            .take(number)
            .map(|(name, held)| (name.to_string(), held.fields.clone().into_iter().collect()))
            .collect()
    };
    for (number, offset) in [
        (7usize, 0usize),
        (50, 37),
        (MOST + 5, 0),
        (3, model.len().saturating_sub(2)),
    ] {
        assert_eq!(
            collection.list(number, offset, None).unwrap(),
            window(offset, number),
            "{step}: list {number} from {offset}"
        );
    }
    if !ordered.is_empty() {
        for at in [0, ordered.len() / 3, ordered.len() - 1] {
            let cursor = ordered[at].0;
            assert_eq!(
                collection.list(25, 0, Some(cursor)).unwrap(),
                window(at + 1, 25),
                "{step}: list after {cursor}"
            );
        }
    }

    // `count` under each filter.
    for (label, filter) in &probe.filters {
        let want = model
            .held
            .values()
            .filter(|held| admits(held, Some(filter)))
            .count();
        assert_eq!(
            collection.count(Some(filter)),
            want,
            "{step}: count under {label}"
        );
    }

    // The exact paths: each filter selects fewer records than the scan
    // threshold, so each dense search under it scores every match.
    let params = || collection.search_params(K, Some(EF), false, None).unwrap();
    for query in &probe.queries {
        for (label, filter) in &probe.filters {
            let page: Page = collection
                .search_one(query, Some(filter), params())
                .unwrap()
                .iter()
                .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                .collect();
            assert_eq!(
                page,
                brute_dense(collection, model, query, filter),
                "{step}: the exact scan under {label}"
            );
        }
    }
    for query in &probe.sparse {
        for filter in std::iter::once(None).chain(probe.filters.iter().map(|(_, f)| Some(f))) {
            for &idf in &probe.idf {
                let page = bits(
                    collection
                        .search_sparse(query.as_ref(), filter, K, idf)
                        .unwrap(),
                );
                assert_eq!(
                    page,
                    brute_sparse(model, weighting, query, filter, idf),
                    "{step}: the sparse arm under {:?} {:?}",
                    filter.is_some(),
                    idf
                );
            }
        }
    }

    check_traversals(collection, model, probe, step);
}

/// Every per-record structure reaches the records held and no further,
/// which a compaction, a rebuild, a load and a clear leave.
fn check_extents(collection: &Collection, records: usize, graph_too: bool, step: &str) {
    let ids = collection.ids.read().unwrap();
    let index = collection.dense().index.read().unwrap();
    let sparse = collection.sparse().unwrap().index.read().unwrap();
    let metadata = collection.vector_metadata.read().unwrap();
    let columns = collection.columns.read().unwrap();
    assert_eq!(ids.extent(), records, "{step}: the id store's extent");
    assert_eq!(index.node_at().len(), records, "{step}: node_at's extent");
    assert!(
        metadata.extent() <= records,
        "{step}: the metadata's extent"
    );
    assert_eq!(columns.extent(), records, "{step}: the columns' extent");
    assert!(
        sparse.extent() <= records,
        "{step}: the sparse table's extent"
    );
    if graph_too {
        assert_eq!(
            index.graph().nb_points(),
            records,
            "{step}: the graph's nodes"
        );
    }
}

// ============================================================================
// WHAT A SAVE WRITES
// ============================================================================

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// Every node of a version 2 graph dump as its origin id and its row's
/// bytes, read by the layout its header names.
fn dump_rows(blob: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let long = |at: usize| u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(word(blob, 8), 2, "a version 2 dump");
    let nodes = word(blob, 32) as usize;
    let (identity, rows) = (long(48), long(56));
    let width = rows / nodes;
    let ids_at = 128 + 64;
    let rows_at = 128 + identity;
    (0..nodes)
        .map(|node| {
            let row = &blob[rows_at + node * width..rows_at + (node + 1) * width];
            (word(blob, ids_at + 4 * node), row.to_vec())
        })
        .collect()
}

/// The artefacts list the model's records in increasing internal id, each
/// with the model's content.
fn check_artefacts(dir: &Path, collection: &Collection, model: &Model, kind: Kind, step: &str) {
    let ordered = model.by_internal();
    let internal: Vec<u32> = ordered
        .iter()
        .map(|(_, held)| held.internal as u32)
        .collect();

    // mappings.bin: the ids, then their lengths, then their text.
    let bytes = std::fs::read(dir.join("mappings.bin")).unwrap();
    let framed = unframe(&bytes, FrameKind::IdMappings, "mappings.bin").unwrap();
    let n = framed.entries as usize;
    assert_eq!(n, ordered.len(), "{step}: mappings entries");
    let payload = framed.payload;
    let ids: Vec<u32> = (0..n).map(|i| word(payload, 4 * i)).collect();
    assert_eq!(ids, internal, "{step}: mappings in increasing internal id");
    let mut text = 8 * n;
    for (i, (name, _)) in ordered.iter().enumerate() {
        let len = word(payload, 4 * (n + i)) as usize;
        assert_eq!(
            &payload[text..text + len],
            name.as_bytes(),
            "{step}: mapping {i}"
        );
        text += len;
    }

    // vectors.bin, where a product quantized graph keeps raw vectors.
    let path = dir.join("vectors.bin");
    if collection.writes_raw_vectors() && !ordered.is_empty() {
        let bytes = std::fs::read(&path).unwrap();
        let framed = unframe(&bytes, FrameKind::RawVectors, "vectors.bin").unwrap();
        let payload = framed.payload;
        assert_eq!(word(payload, 0) as usize, DIM);
        let stride = 4 + 4 * DIM;
        assert_eq!(framed.entries as usize, ordered.len(), "{step}: vectors");
        for (i, (_, held)) in ordered.iter().enumerate() {
            let row = &payload[4 + i * stride..4 + (i + 1) * stride];
            assert_eq!(
                word(row, 0) as usize,
                held.internal,
                "{step}: vector {i}'s id"
            );
            let values: Vec<u32> = (0..DIM).map(|d| word(row, 4 + 4 * d)).collect();
            let want: Vec<u32> = held.vector.iter().map(|v| v.to_bits()).collect();
            assert_eq!(values, want, "{step}: vector {i}");
        }
    } else {
        assert!(
            !path.exists(),
            "{step}: no raw vectors beside codes, so no vectors.bin"
        );
    }

    // No separate row store: the graph dump holds every record's row.
    for name in ["pq_codes.bin", "int8_rows.zdbint8"] {
        assert!(!dir.join(name).exists(), "{step}: no {name}");
    }

    // The graph dump's rows: each record the model holds once, its row the
    // vector it holds, its codes or its scalar row, by the graph's kind, and
    // any other node stranded.
    let path = dir.join("hnsw_index.zdbgraph");
    if ordered.is_empty() {
        assert!(!path.exists(), "{step}: no record, so no graph dump");
    } else {
        let quantized = collection.is_quantized();
        let index = collection.dense().index.read().unwrap();
        let expected_row = |vector: &[f32]| -> Vec<u8> {
            match kind {
                Kind::PqWithRaw | Kind::PqOnly if quantized => {
                    collection.pq().unwrap().quantize(vector).unwrap()
                }
                Kind::Int8 if quantized => index
                    .graph()
                    .int8_encode(vector)
                    .unwrap()
                    .iter()
                    .map(|&b| b as u8)
                    .collect(),
                _ => vector.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }
        };
        let by_internal: HashMap<usize, &Held> = ordered
            .iter()
            .map(|(_, held)| (held.internal, *held))
            .collect();
        let mut found = 0usize;
        for (origin, row) in dump_rows(&std::fs::read(&path).unwrap()) {
            if let Some(held) = by_internal.get(&(origin as usize)) {
                assert_eq!(
                    row,
                    expected_row(&held.vector),
                    "{step}: the row of {origin}"
                );
                found += 1;
            }
        }
        assert_eq!(
            found,
            ordered.len(),
            "{step}: every record has its row once"
        );
    }

    // The sparse postings: the slot count, the live count, then every live
    // record's id and vector.
    let path = dir.join("spaces").join("terms").join("postings.zdbsparse");
    let with_sparse: Vec<&(&str, &Held)> = ordered
        .iter()
        .filter(|(_, held)| held.sparse.is_some())
        .collect();
    if path.exists() {
        let bytes = std::fs::read(&path).unwrap();
        let framed = unframe(&bytes, FrameKind::SparsePostings, "postings").unwrap();
        let payload = framed.payload;
        assert_eq!(
            word(payload, 0) as usize,
            model.sparse_slots,
            "{step}: the slot count"
        );
        assert_eq!(
            word(payload, 4) as usize,
            with_sparse.len(),
            "{step}: sparse records"
        );
        let mut at = 8;
        for (name, held) in with_sparse {
            let (dims, values) = held.sparse.as_ref().unwrap();
            assert_eq!(
                word(payload, at) as usize,
                held.internal,
                "{step}: {name}'s sparse id"
            );
            let nnz = word(payload, at + 4) as usize;
            assert_eq!(nnz, dims.len(), "{step}: {name}'s nonzeros");
            at += 8;
            let got: Vec<u32> = (0..nnz).map(|j| word(payload, at + 4 * j)).collect();
            assert_eq!(&got, dims, "{step}: {name}'s dimensions");
            at += 4 * nnz;
            let got: Vec<u32> = (0..nnz).map(|j| word(payload, at + 4 * j)).collect();
            let want: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{step}: {name}'s values");
            at += 4 * nnz;
        }
        assert_eq!(at, payload.len(), "{step}: nothing past the last record");
    } else {
        assert!(
            with_sparse.is_empty(),
            "{step}: sparse halves held but none written"
        );
    }

    // metadata.json, compact, keyed by external id, in increasing internal
    // id, naming every held record that holds a field and no other.
    let text = std::fs::read_to_string(dir.join("metadata.json")).unwrap();
    let parsed: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&text).unwrap();
    let with_fields: Vec<&(&str, &Held)> = ordered
        .iter()
        .filter(|(_, held)| !held.fields.is_empty())
        .collect();
    assert_eq!(
        with_fields.len(),
        parsed.len(),
        "{step}: metadata names the held records that hold a field"
    );
    let mut last = 0usize;
    for (name, held) in with_fields {
        assert_eq!(
            parsed.get(*name),
            Some(&held.fields),
            "{step}: {name}'s saved fields"
        );
        let found = text
            .find(&format!("\"{name}\":"))
            .unwrap_or_else(|| panic!("{step}: {name} in metadata.json"));
        assert!(
            found >= last,
            "{step}: metadata.json in increasing internal id"
        );
        last = found;
    }
}

/// Every file a save wrote, by its path under the directory, except the
/// manifest, which records when the save ran.
fn saved_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, out);
            } else {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if name != "manifest.json" {
                    out.insert(name, std::fs::read(&path).unwrap());
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

// ============================================================================
// THE SEQUENCE
// ============================================================================

/// One run's state: the collection, the model, and the names a new record
/// takes.
struct Run {
    kind: Kind,
    weighting: Weighting,
    rng: Rng,
    model: Model,
    names: usize,
    removed: Vec<String>,
    temp: TempDir,
    probe: Probe,
    step: usize,
    /// The thread's close-ups on removal when the run began.
    close_ups_before: usize,
    /// The close-ups on removal the recoveries replayed.
    replayed: usize,
    /// The journals that churned.
    journal_churns: usize,
}

impl Run {
    fn label(&self, what: &str) -> String {
        format!(
            "{:?} {:?} step {} ({what})",
            self.kind, self.weighting, self.step
        )
    }

    fn declaration(&self) -> (Declaration, Option<super::QuantizationConfig>) {
        let declaration = Declaration::validate(DIM, "l2", 8, 40, 256, vec!["cat".to_string()])
            .unwrap()
            .with_sparse(
                "terms",
                SparseConfig {
                    weighting: self.weighting,
                    ..SparseConfig::default()
                },
            )
            .unwrap();
        let quantization = match self.kind {
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
        (declaration, quantization)
    }

    /// The slot count a load restores the sparse space to, being one past
    /// the largest internal id a held record's sparse half is under, since
    /// the load raises it by each record it files and by nothing else.
    fn restored_slots(&self) -> usize {
        self.model
            .held
            .values()
            .filter(|held| held.sparse.is_some())
            .map(|held| held.internal + 1)
            .max()
            .unwrap_or(0)
    }

    /// Whether the slot count a save would record now is the one a load of
    /// it restores, so a collection loaded or recovered from it writes the
    /// postings the live one does. The artefact records one past the largest
    /// internal id the space took a record under since it was made or
    /// cleared, which a removal does not lower.
    fn slot_count_held(&self) -> bool {
        self.model.sparse_slots == self.restored_slots()
    }

    /// Add one record with a sparse half, which takes the next internal id
    /// and so makes the slot count one a load restores, where there is room
    /// for it.
    fn hold_slot_count(&mut self, collection: &Collection) {
        if self.model.len() == MOST {
            let name = self.held_names(1).pop().unwrap();
            assert!(collection.remove_point(name.clone()).unwrap());
            self.forget(&name);
            self.model.removal_batch(1);
        }
        let name = self.fresh_name();
        let mut record = record(&mut self.rng, name);
        record.sparse = Some(SparseHalf::Vector(SparseVector {
            dims: vec![self.rng.below(TERMS as usize) as u32],
            values: vec![2.0],
        }));
        let was = collection.is_quantized();
        let added = collection.add_records(vec![record.clone()], vec![], false);
        assert_eq!(
            added.total_errors,
            0,
            "{}",
            self.label("hold the slot count")
        );
        self.model.insert(&record);
        self.trained_if(was, collection);
        assert!(self.slot_count_held());
        self.check(collection, "slot count held");
    }

    /// Close the model's positions up where the collection trained during
    /// the last write, since the training rebuild closes them up.
    fn trained_if(&mut self, was: bool, collection: &Collection) -> bool {
        let trained = !was && collection.is_quantized();
        if trained {
            self.model.closed_up();
        }
        trained
    }

    /// A name no record holds: a new one, or now and then one a removal
    /// freed.
    fn fresh_name(&mut self) -> String {
        if !self.removed.is_empty() && self.rng.chance(20) {
            let at = self.rng.below(self.removed.len());
            let name = self.removed.swap_remove(at);
            if !self.model.held.contains_key(&name) {
                return name;
            }
        }
        self.names += 1;
        format!("r{}", self.names)
    }

    fn held_names(&mut self, most: usize) -> Vec<String> {
        let all: Vec<String> = self.model.held.keys().cloned().collect();
        let mut picked: Vec<String> = Vec::new();
        if all.is_empty() {
            return picked;
        }
        let wanted = 1 + self.rng.below(most);
        for _ in 0..wanted {
            let name = all[self.rng.below(all.len())].clone();
            if !picked.contains(&name) {
                picked.push(name);
            }
        }
        picked
    }

    fn insert(&mut self, collection: &Collection) {
        let room = MOST - self.model.len();
        if room == 0 {
            return;
        }
        let size = (1 + self.rng.below(200)).min(room);
        let mut records = Vec::with_capacity(size);
        for _ in 0..size {
            let name = self.fresh_name();
            if records.iter().any(|r: &ParsedRecord| r.id == name) {
                continue;
            }
            records.push(record(&mut self.rng, name));
        }
        let added = collection.add_records(records.clone(), vec![], false);
        assert_eq!(
            added.total_errors,
            0,
            "{}: {:?}",
            self.label("insert"),
            added.errors
        );
        for record in &records {
            self.model.insert(record);
        }
    }

    fn overwrite(&mut self, collection: &Collection) {
        let mut names = self.held_names(40);
        if self.model.len() < MOST && self.rng.chance(50) {
            names.push(self.fresh_name());
        }
        self.overwrite_names(collection, names, "overwrite");
    }

    /// Add a record under each of `names` with overwrite set: one removal
    /// batch of the names held, then the inserts.
    fn overwrite_names(&mut self, collection: &Collection, names: Vec<String>, what: &str) {
        let records: Vec<ParsedRecord> = names
            .into_iter()
            .map(|name| record(&mut self.rng, name))
            .collect();
        let was = collection.is_quantized();
        let added = collection.add_records(records.clone(), vec![], true);
        assert_eq!(
            added.total_errors,
            0,
            "{}: {:?}",
            self.label(what),
            added.errors
        );
        let removed = records
            .iter()
            .filter(|record| self.model.held.remove(&record.id).is_some())
            .count();
        self.model.removal_batch(removed);
        for record in &records {
            self.model.insert(record);
        }
        self.trained_if(was, collection);
    }

    /// Churn: overwrite most of the held records, call after call, until a
    /// removal batch has closed the positions up, and where a journal is
    /// attached until a replay of it would close them up too. A collection
    /// holding fewer than a thousand records takes fresh ones first, so a
    /// few calls pass the allowance. The holes are held to the model's
    /// after every call.
    fn churn(&mut self, collection: &Collection) {
        if self.model.len() < 1_000 {
            let size = 1_000 - self.model.len();
            let mut records: Vec<ParsedRecord> = Vec::with_capacity(size);
            for _ in 0..size {
                let name = self.fresh_name();
                if records.iter().any(|r| r.id == name) {
                    continue;
                }
                records.push(record(&mut self.rng, name));
            }
            let was = collection.is_quantized();
            let added = collection.add_records(records.clone(), vec![], false);
            assert_eq!(added.total_errors, 0, "{}", self.label("churn"));
            for record in &records {
                self.model.insert(record);
            }
            self.trained_if(was, collection);
        }
        if self.model.replay.is_some() {
            self.journal_churns += 1;
        }
        let passes = self.model.passes;
        let replayed = self.model.replay.map(|replay| replay.passes);
        loop {
            let all: Vec<String> = self.model.held.keys().cloned().collect();
            let names: Vec<String> = all.into_iter().filter(|_| self.rng.chance(85)).collect();
            self.overwrite_names(collection, names, "churn");
            assert_eq!(
                holes(collection),
                self.model.holes,
                "{}: the holes after a churning call",
                self.label("churn")
            );
            let live = self.model.passes > passes;
            let replay = match (replayed, self.model.replay) {
                (Some(before), Some(now)) => now.passes > before,
                _ => true,
            };
            if live && replay {
                break;
            }
        }
    }

    fn forget(&mut self, name: &str) {
        if self.model.held.remove(name).is_some() {
            self.removed.push(name.to_string());
        }
    }

    fn remove(&mut self, collection: &Collection) {
        match self.rng.below(3) {
            0 => {
                let names = self.held_names(1);
                let name = names
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| "absent".to_string());
                let held = self.model.held.contains_key(&name);
                assert_eq!(collection.remove_point(name.clone()).unwrap(), held);
                self.forget(&name);
                self.model.removal_batch(usize::from(held));
            }
            1 => {
                let mut names = self.held_names(100);
                names.push(format!("absent{}", self.step));
                let missing = collection.remove_points(&names).unwrap();
                assert_eq!(
                    missing,
                    vec![format!("absent{}", self.step)],
                    "{}",
                    self.label("remove")
                );
                for name in &names {
                    self.forget(name);
                }
                self.model.removal_batch(names.len() - 1);
            }
            _ => {
                let filter = match self.rng.below(3) {
                    0 => {
                        let cat = ["a", "b", "c", "d"][self.rng.below(4)];
                        json!({"cat": cat, "i": {"lt": 500}})
                    }
                    1 => json!({"i": {"gte": 900}}),
                    _ => json!({"tag": "y", "cat": "c"}),
                };
                let map: HashMap<String, Value> = serde_json::from_value(filter).unwrap();
                let filter = compile_filter(&map).unwrap();
                let gone: Vec<String> = self
                    .model
                    .held
                    .iter()
                    .filter(|(_, held)| admits(held, Some(&filter)))
                    .map(|(name, _)| name.clone())
                    .collect();
                assert_eq!(collection.remove_where(&filter).unwrap(), gone.len());
                for name in &gone {
                    self.forget(name);
                }
                self.model.removal_batch(gone.len());
            }
        }
    }

    fn update(&mut self, collection: &Collection) {
        let mut names = self.held_names(5);
        names.push(format!("absent{}", self.step));
        for name in names {
            let fields = fields(&mut self.rng);
            let held = self.model.held.contains_key(&name);
            assert_eq!(
                collection.update_metadata(&name, fields.clone()).unwrap(),
                held
            );
            if let Some(record) = self.model.held.get_mut(&name) {
                record.fields = fields;
            }
        }
    }

    fn rebuild(&mut self, collection: &Collection) {
        let m = collection.m();
        let degrees: Vec<usize> = [4usize, 6, 8, 12].into_iter().filter(|&d| d != m).collect();
        let degree = degrees[self.rng.below(degrees.len())];
        let plan = collection.plan_rebuild(Some(degree), None, None).unwrap();
        collection.rebuild(plan).unwrap();
        assert_eq!(collection.m(), degree);
        self.model.closed_up();
    }

    /// One write, chosen by weight, applied to the collection and the model.
    /// Returns what it was, and whether every structure's extent is then the
    /// records held.
    fn write(&mut self, collection: &Collection, journaled: bool) -> (&'static str, Option<bool>) {
        let untrained = self.kind != Kind::Raw && !collection.is_quantized();
        let insert_weight = if untrained { 55 } else { 30 };
        let roll = self.rng.below(100);
        if roll < insert_weight {
            let was = collection.is_quantized();
            self.insert(collection);
            let trained = self.trained_if(was, collection);
            return ("insert", trained.then_some(false));
        }
        match self.rng.below(100) {
            0..=16 => {
                self.overwrite(collection);
                ("overwrite", None)
            }
            17..=24 => {
                self.churn(collection);
                ("churn", (self.model.holes == 0).then_some(false))
            }
            25..=49 => {
                self.remove(collection);
                ("remove", None)
            }
            50..=64 => {
                self.update(collection);
                ("update_metadata", None)
            }
            65..=79 => {
                let stranded = collection.dense().index.read().unwrap().stranded();
                collection.compact().unwrap();
                self.model.closed_up();
                ("compact", Some(stranded > 0))
            }
            80..=89 => {
                self.rebuild(collection);
                ("rebuild", Some(true))
            }
            90..=93 if !journaled || self.rng.chance(50) => {
                if self.rng.chance(30) {
                    collection.clear().unwrap();
                    self.model.clear();
                    ("clear", Some(true))
                } else {
                    let _ = collection.stats();
                    ("stats", None)
                }
            }
            _ => {
                collection.shrink_to_fit();
                ("shrink_to_fit", None)
            }
        }
    }

    fn check(&self, collection: &Collection, what: &str) {
        assert_eq!(
            close_ups() - self.close_ups_before,
            self.model.passes + self.replayed,
            "{}: the close-ups on removal",
            self.label(what)
        );
        check(
            collection,
            &self.model,
            self.model.holes,
            self.kind,
            &self.weighting,
            &self.probe,
            &self.label(what),
        );
    }

    /// Save, check the artefacts, load, check the load and save it again.
    ///
    /// A save whose sparse slot count reaches past every record the space
    /// holds, the newest having gone, loads too. The load restores the slot
    /// count from the records it files, so the second save records the
    /// lower count, and its postings differ from the first save's in that
    /// word and the frame's checksum alone, which its own check reads.
    fn save_and_load(&mut self, collection: Collection) -> Collection {
        let label = self.label("save");
        let dir = self.temp.at(&format!("save-{}", self.step));
        collection.save(dir.to_str().unwrap()).unwrap();
        check_artefacts(&dir, &collection, &self.model, self.kind, &label);
        let restored = self.restored_slots();
        let moved = restored != self.model.sparse_slots;
        if moved {
            tally("load past the newest sparse record");
        }
        let before = pages(&collection, &self.probe);
        let loaded = Collection::load(dir.to_str().unwrap()).unwrap();
        tally("load");
        // The load files the records at the first positions, with no hole,
        // and counts the sparse slots of the records it files.
        self.model.closed_up();
        self.model.sparse_slots = restored;
        self.check(&loaded, "loaded");
        check_extents(&loaded, self.model.len(), false, &self.label("loaded"));
        assert_eq!(
            pages(&loaded, &self.probe),
            before,
            "{label}: the pages after a load"
        );
        let again = self.temp.at(&format!("save-{}-again", self.step));
        loaded.save(again.to_str().unwrap()).unwrap();
        let (first, second) = (saved_files(&dir), saved_files(&again));
        assert_eq!(
            first.keys().collect::<Vec<_>>(),
            second.keys().collect::<Vec<_>>(),
            "{label}"
        );
        for (name, bytes) in &first {
            if moved && name.ends_with("postings.zdbsparse") {
                continue;
            }
            assert!(
                bytes == &second[name],
                "{label}: {name} differs on the second save"
            );
        }
        check_artefacts(
            &again,
            &loaded,
            &self.model,
            self.kind,
            &self.label("saved again"),
        );
        loaded
    }

    /// Journal from a checkpoint, run further writes, and recover from the
    /// checkpoint and the journal into a collection that answers and saves
    /// what the live one does.
    fn journal(&mut self, collection: &Collection) {
        // The checkpoint's slot count is one the recovery restores, so the
        // recovered collection writes the postings the live one does.
        if !self.slot_count_held() {
            self.hold_slot_count(collection);
        }
        let dir = self.temp.at(&format!("journal-{}", self.step));
        collection
            .journal_to(dir.to_str().unwrap(), Durability::default())
            .unwrap();
        // The checkpoint loads with no hole, so a replay starts from none.
        self.model.replay = Some(Replayed::default());
        let writes = 2 + self.rng.below(6);
        // A churn after the first write, and so after any checkpoint, in
        // the first journal of a sequence and now and then in a later one,
        // so a recovery closes up on removal in every sequence.
        let churn_at = (self.journal_churns == 0 || self.rng.chance(40))
            .then(|| 1 + self.rng.below(writes - 1));
        for k in 0..writes {
            self.step += 1;
            let what = if Some(k) == churn_at {
                self.churn(collection);
                "churn"
            } else {
                self.write(collection, true).0
            };
            tally(what);
            self.check(collection, what);
            if k == 0 && self.slot_count_held() && self.rng.chance(30) {
                collection.checkpoint().unwrap();
                self.model.replay = Some(Replayed::default());
            }
        }
        assert!(collection.detach_sink().is_some());
        let replay = self.model.replay.take().unwrap();
        let label = self.label("recovered");
        let live_pages = pages(collection, &self.probe);
        let live_dir = self.temp.at(&format!("journal-{}-live", self.step));
        collection.save(live_dir.to_str().unwrap()).unwrap();
        let before = close_ups();
        let (recovered, _) =
            Collection::recover(dir.to_str().unwrap(), None, Durability::default()).unwrap();
        tally("recovery");
        assert_eq!(
            close_ups() - before,
            replay.passes,
            "{label}: the close-ups on removal the replay ran"
        );
        self.replayed += replay.passes;
        if replay.passes > 0 {
            tally("close-up on removal in a recovery");
        }
        check(
            &recovered,
            &self.model,
            replay.holes,
            self.kind,
            &self.weighting,
            &self.probe,
            &label,
        );
        assert_eq!(
            pages(&recovered, &self.probe),
            live_pages,
            "{label}: the pages"
        );
        assert!(recovered.detach_sink().is_some());
        let recovered_dir = self.temp.at(&format!("journal-{}-recovered", self.step));
        recovered.save(recovered_dir.to_str().unwrap()).unwrap();
        let (live, again) = (saved_files(&live_dir), saved_files(&recovered_dir));
        assert_eq!(
            live.keys().collect::<Vec<_>>(),
            again.keys().collect::<Vec<_>>(),
            "{label}"
        );
        for (name, bytes) in &live {
            assert!(
                bytes == &again[name],
                "{label}: {name} differs from the live save"
            );
        }
    }
}

/// One seeded sequence of `steps` operations on a collection of `kind`
/// with a sparse space under `weighting`.
fn run(kind: Kind, weighting: Weighting, seed: u64, steps: usize) {
    let mut rng = Rng(seed);
    let probe = Probe::new(&mut rng, &weighting);
    let mut run = Run {
        kind,
        weighting,
        rng,
        model: Model::default(),
        names: 0,
        removed: Vec::new(),
        temp: TempDir::new(),
        probe,
        step: 0,
        close_ups_before: close_ups(),
        replayed: 0,
        journal_churns: 0,
    };
    let (declaration, quantization) = run.declaration();
    let mut collection = Collection::build(declaration, quantization);
    let mut trained = false;
    let mut saves = 0;
    let mut journals = 0;
    while run.step < steps {
        run.step += 1;
        let roll = run.rng.below(100);
        if roll < 7 && run.model.len() > 0 {
            // Now and then the slot count is held first, so the sequence
            // saves both a collection a load restores whole and one whose
            // newest sparse record has gone.
            if !run.slot_count_held() && run.rng.chance(60) {
                run.hold_slot_count(&collection);
            }
            collection = run.save_and_load(collection);
            saves += 1;
            continue;
        }
        if roll < 11 && run.model.len() > 0 {
            run.journal(&collection);
            journals += 1;
            continue;
        }
        let (what, closed) = run.write(&collection, false);
        tally(what);
        run.check(&collection, what);
        if let Some(graph_too) = closed {
            check_extents(&collection, run.model.len(), graph_too, &run.label(what));
        }
        trained |= collection.is_quantized();
    }
    assert!(
        saves > 0 && journals > 0,
        "{kind:?}: the sequence saved and journaled"
    );
    assert_eq!(
        close_ups() - run.close_ups_before,
        run.model.passes + run.replayed,
        "{kind:?}: every close-up on removal was one the model ran"
    );
    if kind != Kind::Raw {
        assert!(trained, "{kind:?}: the sequence crossed the training size");
    }
    let tallied = TALLY.with(|tally| std::mem::take(&mut *tally.borrow_mut()));
    eprintln!("{kind:?} {:?}: {tallied:?}", run.weighting);
    for what in [
        "insert",
        "overwrite",
        "remove",
        "update_metadata",
        "compact",
        "rebuild",
        "load",
        "recovery",
        "churn",
        "close-up on removal",
        "close-up on removal in a journal",
        "close-up on removal in a recovery",
        "a state with holes",
        "traversal with no predicate",
        "traversal past stranded nodes",
        "traversal under a bitmap",
        "traversal under a table call",
    ] {
        assert!(
            tallied.contains_key(what),
            "{kind:?}: the sequence never ran {what}"
        );
    }
}

#[test]
fn positions_follow_every_write_on_a_raw_collection() {
    run(Kind::Raw, Weighting::Dot, 0x5eed_0001, 90);
    run(
        Kind::Raw,
        Weighting::Bm25 { k1: 1.2, b: 0.75 },
        0x5eed_0002,
        90,
    );
}

#[test]
fn positions_follow_every_write_on_a_product_quantized_collection_with_raw_vectors() {
    run(
        Kind::PqWithRaw,
        Weighting::Bm25 { k1: 1.2, b: 0.75 },
        0x5eed_0003,
        90,
    );
}

#[test]
fn positions_follow_every_write_on_a_product_quantized_collection_without_raw_vectors() {
    run(Kind::PqOnly, Weighting::Dot, 0x5eed_0004, 90);
}

#[test]
fn positions_follow_every_write_on_a_scalar_quantized_collection() {
    run(
        Kind::Int8,
        Weighting::Bm25 { k1: 1.2, b: 0.75 },
        0x5eed_0005,
        90,
    );
}

// ============================================================================
// EACH WRITE, ONE AT A TIME
// ============================================================================

/// A collection of `n` records at dimension 8 with the field `cat` declared
/// and a sparse space, records `r0` onward, each with a sparse half, under
/// product quantization where `quantization` names a storage mode.
fn small(n: usize, quantization: Option<StorageMode>) -> Collection {
    let declaration = Declaration::validate(DIM, "l2", 8, 40, 64, vec!["cat".to_string()])
        .unwrap()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let quantization = quantization.map(|mode| {
        declaration
            .quantization(4, 4, TRAINING, None, mode)
            .unwrap()
    });
    let collection = Collection::build(declaration, quantization);
    put(&collection, 0..n);
    collection
}

fn numbered(i: usize) -> ParsedRecord {
    let mut rng = Rng(0x00c1_05e0 + i as u64);
    ParsedRecord {
        id: format!("r{i}"),
        vector: vector(&mut rng),
        sparse: Some(SparseHalf::Vector(SparseVector {
            dims: vec![(i % 11) as u32, 20 + (i % 7) as u32],
            values: vec![1.0, 1.0 + (i % 3) as f32],
        })),
        metadata: HashMap::from([
            ("cat".to_string(), json!(["a", "b", "c"][i % 3])),
            ("i".to_string(), json!(i)),
        ]),
    }
}

fn put(collection: &Collection, records: impl Iterator<Item = usize>) {
    let added = collection.add_records(records.map(numbered).collect(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

/// Each structure's answer for one position, as a test reads it.
#[derive(Debug, PartialEq)]
struct Slot {
    name: Option<String>,
    node: Option<u32>,
    fields: Option<BTreeMap<String, Value>>,
    column: bool,
    sparse: Option<f32>,
}

fn slot(collection: &Collection, at: usize) -> Slot {
    let ids = collection.ids.read().unwrap();
    let index = collection.dense().index.read().unwrap();
    let sparse = collection.sparse().unwrap().index.read().unwrap();
    let metadata = collection.vector_metadata.read().unwrap();
    let columns = collection.columns.read().unwrap();
    Slot {
        name: ids.name(at).map(str::to_string),
        node: index.node_of(Position::from_slot(at)),
        fields: metadata.get(at).map(|fields| fields.to_map()),
        column: columns.holds(at),
        sparse: sparse.length_of(Position::from_slot(at)),
    }
}

/// Every insertion takes the position after the last one assigned, holes
/// included, so a hole is never filled again. An insertion the index refuses
/// after the id store named the record leaves nothing behind: the store takes
/// the position back, and the metadata, the columns and the sparse space are
/// never written there.
#[test]
fn an_insert_takes_the_next_position_and_a_refused_one_writes_nothing() {
    let collection = small(10, None);
    for i in 0..10 {
        assert_eq!(collection.ids().slot_of(&format!("r{i}")), Some(i));
        assert_eq!(collection.internal_id_of(&format!("r{i}")), Some(i + 1));
    }
    assert!(collection
        .remove_points(&["r3".to_string(), "r7".to_string()])
        .unwrap()
        .is_empty());
    put(&collection, 10..11);
    assert_eq!(
        collection.ids().slot_of("r10"),
        Some(10),
        "a hole is never filled"
    );
    assert_eq!(collection.ids().extent(), 11);
    assert!(collection.positions_agree());

    // A node filed at the next position behind the id store's back, so the
    // index refuses the next record there.
    {
        use zeusdb_vector_core::{RecordId, VectorIndex};
        let mut index = collection.dense().index.write().unwrap();
        let planted = vec![0.5; DIM];
        let prepared = index.prepare(RecordId(9_999), &planted).unwrap();
        index
            .insert(Position(11), RecordId(9_999), &planted, prepared)
            .unwrap();
    }
    let before = slot(&collection, 11);
    let added = collection.add_records(vec![numbered(11)], vec![], false);
    assert_eq!(added.total_errors, 1, "the index refuses the record");
    assert!(!collection.contains("r11"));
    assert_eq!(collection.ids().extent(), 11, "the position is taken back");
    assert_eq!(
        slot(&collection, 11),
        before,
        "nothing is written at the position"
    );
    assert_eq!(
        before,
        Slot {
            name: None,
            node: before.node,
            fields: None,
            column: false,
            sparse: None,
        }
    );
    assert!(before.node.is_some());
    let metadata = collection.vector_metadata.read().unwrap().extent();
    let columns = collection.columns.read().unwrap().extent();
    let sparse = collection.sparse().unwrap().index.read().unwrap().extent();
    assert_eq!((metadata, columns, sparse), (11, 11, 11));
}

/// A removal leaves a hole every per-record structure reports absent, and
/// strands the record's node, which keeps its place in the graph.
#[test]
fn a_removal_leaves_a_hole_every_structure_reports_absent() {
    let collection = small(12, None);
    let held = slot(&collection, 5);
    assert_eq!(held.name.as_deref(), Some("r5"));
    let node = held.node.unwrap();
    assert!(held.fields.is_some() && held.column && held.sparse.is_some());
    assert!(collection.remove_point("r5".to_string()).unwrap());
    assert_eq!(
        slot(&collection, 5),
        Slot {
            name: None,
            node: None,
            fields: None,
            column: false,
            sparse: None,
        }
    );
    {
        let ids = collection.ids.read().unwrap();
        let index = collection.dense().index.read().unwrap();
        let sparse = collection.sparse().unwrap().index.read().unwrap();
        assert!(!ids.contains_slot(5));
        assert_eq!(ids.extent(), 12, "the hole keeps its place");
        assert_eq!(index.node_at()[5], Position::NONE);
        assert_eq!(index.graph().record_of_at(node), Position::NONE);
        assert_eq!(index.graph().nb_points(), 12, "the node stays in the graph");
        assert!(!sparse.live_set().contains(5));
        use zeusdb_vector_core::VectorIndex;
        assert_eq!(index.stranded(), 1);
    }
    assert!(collection.positions_agree());
    assert_eq!(slot(&collection, 6).name.as_deref(), Some("r6"));
}

/// The close-up a rebuild at a new degree, a compaction and the training
/// rebuild run moves every per-record structure to the positions below the
/// record count in internal id order: the id store's table names each record
/// at its new position, the metadata, the columns and the sparse space hold
/// each record's own entry there, every posting is keyed by the new position
/// and the postings of the records removed are gone, and each record's node
/// is the one at its new position.
#[test]
fn a_close_up_moves_every_structure_and_keeps_order() {
    for (label, quantization, records) in [
        ("rebuild", None, 400usize),
        ("compact", None, 400),
        (
            "training",
            Some(StorageMode::QuantizedWithRaw),
            TRAINING - 1,
        ),
    ] {
        let collection = small(records, quantization);
        let gone: Vec<String> = (0..records)
            .filter(|i| i % 3 != 1)
            .map(|i| format!("r{i}"))
            .collect();
        assert!(collection.remove_points(&gone).unwrap().is_empty());
        {
            let sparse = collection.sparse().unwrap().index.read().unwrap();
            assert!(
                sparse.dead_records() > 0,
                "{label}: a removal leaves dead records"
            );
        }
        let expected: Vec<(usize, String)> = collection.internal_ids();
        match label {
            "rebuild" => {
                let plan = collection.plan_rebuild(Some(6), None, None).unwrap();
                collection.rebuild(plan).unwrap();
            }
            "compact" => {
                collection.compact().unwrap();
            }
            _ => {
                // The records that fill the training set again, the last of
                // which sets off the training rebuild.
                let refill = TRAINING - collection.len();
                put(&collection, records..records + refill);
                assert!(collection.is_quantized(), "{label}");
            }
        }
        let mut expected = expected;
        if label == "training" {
            let refill = TRAINING - expected.len();
            for k in 0..refill {
                expected.push((records + 1 + k, format!("r{}", records + k)));
            }
        }
        assert_eq!(collection.internal_ids(), expected, "{label}: order kept");
        assert!(collection.positions_agree(), "{label}");
        let ids = collection.ids.read().unwrap();
        let index = collection.dense().index.read().unwrap();
        let sparse = collection.sparse().unwrap().index.read().unwrap();
        let metadata = collection.vector_metadata.read().unwrap();
        let columns = collection.columns.read().unwrap();
        assert_eq!(ids.extent(), expected.len(), "{label}");
        assert!(ids.table_agrees(), "{label}");
        assert_eq!(sparse.dead_records(), 0, "{label}: dead postings dropped");
        assert_eq!(sparse.dead_postings(), 0, "{label}");
        assert!(sparse.keys_agree(), "{label}");
        let mut postings = 0usize;
        for (at, (internal, name)) in expected.iter().enumerate() {
            let i: usize = name[1..].parse().unwrap();
            let record = numbered(i);
            assert_eq!(ids.slot_of(name), Some(at), "{label}: {name}");
            assert_eq!(index.internal_id_at(at), Some(*internal), "{label}: {name}");
            let node = index.node_of(Position::from_slot(at)).unwrap();
            assert_eq!(
                node as usize, at,
                "{label}: the replacement's node is the position"
            );
            assert_eq!(
                metadata.get(at).unwrap().to_map(),
                record
                    .metadata
                    .clone()
                    .into_iter()
                    .collect::<BTreeMap<_, _>>(),
                "{label}"
            );
            assert!(columns.agrees_with(at, &record.metadata), "{label}");
            let Some(SparseHalf::Vector(half)) = &record.sparse else {
                unreachable!()
            };
            postings += half.dims.len();
            assert_eq!(
                sparse.length_of(Position::from_slot(at)),
                Some(half.values.iter().sum::<f32>()),
                "{label}: {name}"
            );
        }
        assert_eq!(sparse.postings_total(), postings, "{label}");
        for d in 0..32u32 {
            let want = expected
                .iter()
                .filter(|(_, name)| {
                    let i: usize = name[1..].parse().unwrap();
                    i % 11 == d as usize || 20 + i % 7 == d as usize
                })
                .count();
            assert_eq!(sparse.df(d), want, "{label}: df of {d}");
        }
    }
}

/// Each rebuild refuses a record it finds no source data for before it
/// swaps anything: a compaction, a rebuild at a new degree, the rebuild from
/// stored codes, and a load whose dump is gone and whose codes miss a record
/// the mappings hold.
#[test]
fn each_rebuild_refuses_a_record_without_source_data() {
    let collection = small(TRAINING + 20, Some(StorageMode::QuantizedOnly));
    assert!(collection.is_quantized());
    assert!(collection.remove_point("r3".to_string()).unwrap());
    let before = collection.internal_ids();
    let nodes = collection.dense().index.read().unwrap().graph().nb_points();
    assert!(collection
        .dense()
        .pq_codes
        .write()
        .unwrap()
        .remove("r40")
        .is_some());

    let refused = collection.compact().unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("1 of 1019 live records have no stored quantized codes"),
        "{refused}"
    );
    let plan = collection.plan_rebuild(Some(6), None, None).unwrap();
    let refused = collection.rebuild(plan).unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("have no stored quantized codes"),
        "{refused}"
    );
    let refused = collection.rebuild_with_quantization().unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("built 1018 nodes for the 1019 records the index holds"),
        "{refused}"
    );
    assert_eq!(collection.internal_ids(), before, "nothing was swapped");
    assert_eq!(
        collection.dense().index.read().unwrap().graph().nb_points(),
        nodes
    );
    assert!(collection.positions_agree());

    // A directory before format 5.0.0 holds each record's codes in
    // pq_codes.bin beside the dump, and its rebuild from that file meets the
    // record the file lacks. At 5.0.0 the dump holds every node's codes.
    let temp = TempDir::new();
    let path = temp.at("codes-short.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    crate::persistence::older::rewrite_before_five(&path, "4.0.0");
    let lacking = before
        .iter()
        .find(|(_, name)| name == "r40")
        .map(|(internal, _)| *internal as u32)
        .unwrap();
    let codes = std::fs::read(path.join("pq_codes.bin")).unwrap();
    let framed = zeusdb_vector_core::unframe(&codes, FrameKind::PqCodes, "pq_codes.bin").unwrap();
    let width = u32::from_le_bytes(framed.payload[..4].try_into().unwrap()) as usize;
    let mut payload = framed.payload[..4].to_vec();
    for row in framed.payload[4..].chunks_exact(4 + width) {
        if u32::from_le_bytes(row[..4].try_into().unwrap()) != lacking {
            payload.extend_from_slice(row);
        }
    }
    let codes = zeusdb_vector_core::frame(
        FrameKind::PqCodes,
        zeusdb_vector_core::FrameEncoding::Engine,
        framed.entries - 1,
        &payload,
    );
    std::fs::write(path.join("pq_codes.bin"), &codes).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
    manifest["file_digests"]["pq_codes.bin"] = json!({ "bytes": codes.len() });
    std::fs::write(
        path.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::remove_file(path.join(zeusdb_vector_core::DUMP_FILENAME)).unwrap();
    let refused = Collection::load(path.to_str().unwrap()).err().unwrap();
    assert!(
        refused.to_string().contains(
            "the rebuild from codes built 1018 nodes for the 1019 records the index holds"
        ),
        "{refused}"
    );
}

// ============================================================================
// THE CLOSE-UP ON REMOVAL
// ============================================================================

/// A collection of `n` records `r0` onward, each with a sparse half under
/// `weighting` and the sparse policy `unlink`, at a small degree, so the
/// thousands of records a close-up on removal needs build quickly.
fn many(n: usize, unlink: Unlink, weighting: Weighting) -> Collection {
    let declaration = Declaration::validate(DIM, "l2", 4, 16, 100, vec!["cat".to_string()])
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
    let collection = Collection::build(declaration, None);
    put(&collection, 0..n);
    collection
}

/// The external ids of the records `records` names.
fn named(records: impl Iterator<Item = usize>) -> Vec<String> {
    records.map(|i| format!("r{i}")).collect()
}

/// Remove the records `records` names in one batch, every one of them held.
fn remove_batch(collection: &Collection, records: impl Iterator<Item = usize>) {
    let missing = collection.remove_points(&named(records)).unwrap();
    assert!(missing.is_empty(), "{missing:?}");
}

/// The extent and the records held.
fn reach_and_held(collection: &Collection) -> (usize, usize) {
    let ids = collection.ids.read().unwrap();
    (ids.extent(), ids.len())
}

/// The close-up on removal runs at the removal batch that first leaves more
/// holes than 4,096 and more than the records held, and at no batch before.
/// Below 4,096 records held the allowance decides: 5,000 records close up at
/// the 4,097th removal, with 903 held, and the 4,096th does not. Above it the
/// records held decide: 10,000 close up at the 5,001st, with 4,999 held, and
/// the 5,000th, which leaves as many holes as records, does not, and those
/// 4,999 close up again 4,097 removals later. Batches walk to a hundred
/// short of each boundary and single removals cross it, with the holes read
/// after every call. The records keep their order at the first positions,
/// the graph keeps every node, and every structure reaches the records held.
#[test]
fn the_close_up_on_removal_runs_once_the_holes_pass_the_bound_and_not_before() {
    for (records, batch, crossing) in [(5_000usize, 100usize, 4_097usize), (10_000, 500, 5_001)] {
        let collection = many(records, Unlink::Lazy, Weighting::Dot);
        let start = close_ups();
        let mut removed = 0;
        while removed < crossing - 100 {
            let to = (removed + batch).min(crossing - 100);
            remove_batch(&collection, removed..to);
            removed = to;
            assert_eq!(holes(&collection), removed, "{records}: after {removed}");
            assert_eq!(close_ups(), start, "{records}: no close-up after {removed}");
        }
        while removed + 1 < crossing {
            assert!(collection.remove_point(format!("r{removed}")).unwrap());
            removed += 1;
            assert_eq!(holes(&collection), removed, "{records}: after {removed}");
            assert_eq!(close_ups(), start, "{records}: no close-up after {removed}");
        }
        assert!(collection.remove_point(format!("r{removed}")).unwrap());
        let held = records - crossing;
        assert_eq!(
            close_ups(),
            start + 1,
            "{records}: the close-up after {crossing}"
        );
        assert_eq!(reach_and_held(&collection), (held, held));
        assert!(collection.positions_agree());
        check_extents(&collection, held, false, "closed up on removal");
        {
            let index = collection.dense().index.read().unwrap();
            assert_eq!(
                index.graph().nb_points(),
                records,
                "the graph keeps every node"
            );
            assert_eq!(index.stranded(), crossing);
        }
        for (at, (internal, name)) in collection.internal_ids().iter().enumerate() {
            assert_eq!(*internal, crossing + at + 1, "the order is kept");
            assert_eq!(collection.ids().slot_of(name), Some(at), "{name}");
        }
        if held > ALLOWANCE {
            remove_batch(&collection, crossing..crossing + ALLOWANCE);
            assert_eq!(
                (holes(&collection), close_ups()),
                (ALLOWANCE, start + 1),
                "{records}: the allowance reached again and not passed"
            );
            let next = format!("r{}", crossing + ALLOWANCE);
            assert!(collection.remove_point(next).unwrap());
            assert_eq!(
                (holes(&collection), close_ups()),
                (0, start + 2),
                "{records}: the second close-up"
            );
            assert!(collection.positions_agree());
        }
    }
}

/// One removal batch that passes the bound twice part way through closes
/// up once, after it. Removing 11,000 of 12,000 records one record a batch
/// would have closed up after the 6,001st removal and again after the
/// 10,098th. As one batch it leaves 11,000 holes against 1,000 records,
/// which the one close-up at its end takes away.
#[test]
fn a_removal_batch_that_passes_the_bound_twice_closes_up_once_after_it() {
    let (records, gone) = (12_000usize, 11_000usize);
    let mut part_way = Vec::new();
    let mut open = 0usize;
    for removed in 1..=gone {
        open += 1;
        if open > ALLOWANCE.max(records - removed) {
            part_way.push(removed);
            open = 0;
        }
    }
    assert_eq!(part_way, vec![6_001, 10_098]);
    let collection = many(records, Unlink::Strand, Weighting::Dot);
    let start = close_ups();
    remove_batch(&collection, 0..gone);
    assert_eq!(
        close_ups(),
        start + 1,
        "one close-up, once the batch is done"
    );
    let held = records - gone;
    assert_eq!(reach_and_held(&collection), (held, held));
    check_extents(&collection, held, false, "closed up after the batch");
    assert!(collection.positions_agree());
    let sparse = collection.sparse().unwrap().index.read().unwrap();
    assert_eq!((sparse.dead_records(), sparse.dead_postings()), (0, 0));
}

/// Every path that removes records closes up once its batch leaves more
/// holes than the bound: a removal by id, a batch of ids, a delete by ids,
/// `remove_where` and an overwrite. Each starts from the same 5,000 records
/// read back from one save, removes 4,096 in one batch, which reaches the
/// allowance and does not pass it, and then removes one record more by the
/// path. After the close-up the sparse space holds no removed record, and
/// an overwrite's new record takes the position after the records held.
#[test]
fn every_removal_path_closes_up_once_its_batch_passes_the_bound() {
    let temp = TempDir::new();
    let dir = temp.at("five-thousand");
    many(5_000, Unlink::Strand, Weighting::Dot)
        .save(dir.to_str().unwrap())
        .unwrap();
    for path in [
        "remove_point",
        "remove_points",
        "delete",
        "remove_where",
        "overwrite",
    ] {
        let collection = Collection::load(dir.to_str().unwrap()).unwrap();
        let start = close_ups();
        remove_batch(&collection, 0..ALLOWANCE);
        assert_eq!(
            (holes(&collection), close_ups()),
            (ALLOWANCE, start),
            "{path}"
        );
        let next = format!("r{ALLOWANCE}");
        match path {
            "remove_point" => assert!(collection.remove_point(next.clone()).unwrap()),
            "remove_points" => assert_eq!(
                collection
                    .remove_points(&[next.clone(), "absent".to_string()])
                    .unwrap(),
                vec!["absent".to_string()]
            ),
            "delete" => assert_eq!(
                collection
                    .delete_ids(&[next.clone(), next.clone()])
                    .unwrap(),
                1
            ),
            "remove_where" => {
                let filter =
                    compile_filter(&HashMap::from([("i".to_string(), json!(ALLOWANCE))])).unwrap();
                assert_eq!(collection.remove_where(&filter).unwrap(), 1);
            }
            _ => {
                let mut record = numbered(ALLOWANCE);
                record.vector = vec![0.5; DIM];
                let added = collection.add_records(vec![record], vec![], true);
                assert_eq!(added.total_errors, 0, "{:?}", added.errors);
            }
        }
        let held = 5_000 - ALLOWANCE - usize::from(path != "overwrite");
        assert_eq!(close_ups(), start + 1, "{path}: one close-up");
        assert_eq!(reach_and_held(&collection), (held, held), "{path}");
        assert!(collection.positions_agree(), "{path}");
        check_extents(&collection, held, false, path);
        {
            let sparse = collection.sparse().unwrap().index.read().unwrap();
            assert_eq!(
                (sparse.dead_records(), sparse.dead_postings(), sparse.len()),
                (0, 0, held),
                "{path}"
            );
        }
        if path == "overwrite" {
            assert_eq!(
                collection.ids().slot_of(&next),
                Some(held - 1),
                "the new record takes the last position"
            );
            assert_eq!(collection.internal_id_of(&next), Some(5_001));
        } else {
            assert!(!collection.contains(&next), "{path}");
        }
    }
}

/// The close-up on removal drops the postings of every record removed since
/// the last close-up, under each sparse policy and weighting, and empties the
/// dead set, which `get_stats` reports: the postings left are the records
/// held, each term's document frequency counts them, and the arm answers
/// what a collection built from the records held alone answers. The forward
/// arena keeps the removed spans until a compaction rewrites it.
#[test]
fn the_close_up_on_removal_drops_the_postings_of_the_records_removed() {
    let queries = [
        SparseVector {
            dims: vec![1, 3, 21, 24],
            values: vec![1.0, 2.0, 1.0, 3.0],
        },
        SparseVector {
            dims: vec![0, 5, 10, 20, 26],
            values: vec![2.0, 1.0, 1.0, 1.0, 2.0],
        },
    ];
    for unlink in [Unlink::Strand, Unlink::Lazy, Unlink::Eager] {
        for weighting in [Weighting::Dot, Weighting::Bm25 { k1: 1.2, b: 0.75 }] {
            let label = format!("{unlink:?} {weighting:?}");
            let collection = many(6_000, unlink, weighting);
            let gone: Vec<usize> = (0..6_000)
                .filter(|i| i % 6 != 0)
                .take(ALLOWANCE + 1)
                .collect();
            remove_batch(&collection, gone[..ALLOWANCE].iter().copied());
            let (dead_postings, arena) = {
                let sparse = collection.sparse().unwrap().index.read().unwrap();
                assert_eq!(sparse.dead_records(), ALLOWANCE, "{label}");
                (sparse.dead_postings(), sparse.heap_bytes().forward)
            };
            match unlink {
                Unlink::Strand => assert_eq!(dead_postings, 2 * ALLOWANCE, "{label}"),
                Unlink::Eager => assert_eq!(dead_postings, 0, "{label}"),
                Unlink::Lazy => {}
            }
            assert_eq!(
                collection.stats()["sparse_dead_postings"],
                dead_postings.to_string(),
                "{label}"
            );

            let start = close_ups();
            let last = format!("r{}", gone[ALLOWANCE]);
            assert!(collection.remove_point(last).unwrap());
            assert_eq!(close_ups(), start + 1, "{label}");
            let mut removed = vec![false; 6_000];
            for &i in &gone {
                removed[i] = true;
            }
            let kept: Vec<usize> = (0..6_000).filter(|&i| !removed[i]).collect();
            {
                let sparse = collection.sparse().unwrap().index.read().unwrap();
                assert_eq!(
                    (sparse.dead_records(), sparse.dead_postings()),
                    (0, 0),
                    "{label}"
                );
                assert_eq!(sparse.len(), kept.len(), "{label}");
                assert_eq!(
                    sparse.postings_total(),
                    2 * kept.len(),
                    "{label}: the postings of the records held"
                );
                for d in 0..27u32 {
                    let want = kept
                        .iter()
                        .filter(|&&i| i % 11 == d as usize || 20 + i % 7 == d as usize)
                        .count();
                    assert_eq!(sparse.df(d), want, "{label}: df of {d}");
                }
                assert!(sparse.keys_agree(), "{label}");
                assert_eq!(
                    sparse.heap_bytes().forward,
                    arena,
                    "{label}: the arena keeps the removed spans"
                );
            }
            let stats = collection.stats();
            assert_eq!(stats["sparse_dead_postings"], "0", "{label}");
            assert_eq!(
                stats["sparse_postings"],
                (2 * kept.len()).to_string(),
                "{label}"
            );

            let twin = many(0, unlink, weighting);
            put(&twin, kept.iter().copied());
            let scopes = match weighting {
                Weighting::Bm25 { .. } => vec![IdfScope::Corpus, IdfScope::Global],
                _ => vec![IdfScope::Corpus],
            };
            let page = |collection: &Collection, query: &SparseVector, idf: IdfScope| {
                bits(
                    collection
                        .search_sparse(query.as_ref(), None, K, idf)
                        .unwrap(),
                )
            };
            for query in &queries {
                for &idf in &scopes {
                    assert_eq!(
                        page(&collection, query, idf),
                        page(&twin, query, idf),
                        "{label}: {idf:?}"
                    );
                }
            }

            collection.compact().unwrap();
            let sparse = collection.sparse().unwrap().index.read().unwrap();
            assert!(
                sparse.heap_bytes().forward < arena,
                "{label}: a compaction rewrites the arena"
            );
        }
    }
}

/// Every record a sink has been handed, as its kind and its bytes.
type Recorded = Vec<(OperationKind, Vec<u8>)>;

/// A sink that keeps every record it is handed, shared with the test.
#[derive(Clone, Debug, Default)]
struct Recording(Arc<Mutex<Recorded>>);

impl Recording {
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl OperationSink for Recording {
    fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<(), Error> {
        self.0.lock().unwrap().push((kind, payload.to_vec()));
        Ok(())
    }

    fn commit(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// A replay closes up at the points the live collection did. A collection
/// of 5,000 records is saved, and then passes the bound three times, by a
/// removal by id, by an overwrite and by a delete, with a batch, inserts
/// and `remove_where` between, while a sink keeps every record. The save
/// loaded again, with every record applied to it one at a time, holds at the
/// end of each step the extent, the records and the close-ups the live
/// collection held, and in the end every record at the position the live
/// collection holds it at, and answers the same pages.
#[test]
fn a_replay_closes_up_at_the_points_the_live_collection_did() {
    let temp = TempDir::new();
    let checkpoint = temp.at("checkpoint");
    let bm25 = Weighting::Bm25 { k1: 1.2, b: 0.75 };
    let live = many(5_000, Unlink::Strand, bm25);
    live.save(checkpoint.to_str().unwrap()).unwrap();
    let recording = Recording::default();
    assert!(live.attach_sink(Box::new(recording.clone())).is_none());
    let start = close_ups();
    let mut steps: Vec<(&str, usize, (usize, usize), usize)> = Vec::new();
    let mut mark = |what: &'static str| {
        steps.push((
            what,
            recording.len(),
            reach_and_held(&live),
            close_ups() - start,
        ));
    };
    let overwrite = |records: std::ops::Range<usize>| {
        let records: Vec<ParsedRecord> = records
            .map(|i| {
                let mut record = numbered(i);
                record.vector[0] += 0.125;
                record
            })
            .collect();
        let added = live.add_records(records, vec![], true);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    };

    remove_batch(&live, 0..ALLOWANCE);
    mark("a batch reaching the allowance");
    assert!(live.remove_point(format!("r{ALLOWANCE}")).unwrap());
    mark("a removal by id passing it");
    put(&live, 5_000..9_500);
    mark("4,500 new records");
    overwrite(5_000..7_500);
    mark("an overwrite");
    overwrite(5_000..7_500);
    mark("an overwrite passing the bound");
    let filter =
        compile_filter(&HashMap::from([("i".to_string(), json!({"gte": 7_500}))])).unwrap();
    assert_eq!(live.remove_where(&filter).unwrap(), 2_000);
    mark("remove_where");
    assert_eq!(live.delete_ids(&named(5_000..7_500)).unwrap(), 2_500);
    mark("a delete passing the bound");
    assert!(live.detach_sink().is_some());
    assert_eq!(
        steps.iter().map(|step| step.3).collect::<Vec<_>>(),
        vec![0, 1, 1, 1, 2, 2, 3]
    );
    assert_eq!(
        steps.iter().map(|step| step.2).collect::<Vec<_>>(),
        vec![
            (5_000, 904),
            (903, 903),
            (5_403, 5_403),
            (7_903, 5_403),
            (5_403, 5_403),
            (5_403, 3_403),
            (903, 903),
        ]
    );

    let replayed = Collection::load(checkpoint.to_str().unwrap()).unwrap();
    let records = recording.0.lock().unwrap().clone();
    let replay_start = close_ups();
    let mut applied = 0;
    for &(what, upto, reach, passes) in &steps {
        while applied < upto {
            let (kind, payload) = &records[applied];
            let record = JournalRecord {
                sequence: applied as u64 + 1,
                kind: *kind,
                offset: 0,
                payload,
            };
            replayed
                .apply(Operation::decode(&record, DIM, "recorded").unwrap())
                .unwrap();
            applied += 1;
        }
        assert_eq!(
            (reach_and_held(&replayed), close_ups() - replay_start),
            (reach, passes),
            "{what}"
        );
    }
    // Each collection's guard taken alone, since a thread holds one id store
    // guard at a time.
    for (_, name) in live.internal_ids() {
        let replayed_at = replayed.ids().slot_of(&name);
        let live_at = live.ids().slot_of(&name);
        assert_eq!(replayed_at, live_at, "{name}");
    }
    let probe = Probe::new(&mut Rng(0x5eed_0205), &bm25);
    assert_eq!(pages(&replayed, &probe), pages(&live, &probe));
}

/// A recovery replays a journal onto a checkpoint that loads with no hole,
/// so where the live collection held holes when it wrote the checkpoint, the
/// two close up at different removals and hold their records at different
/// positions. Nothing depends on a position: the recovered collection holds
/// the same records under the same internal ids, answers the same pages and
/// writes the same artefacts.
#[test]
fn positions_after_a_recovery_may_differ_and_nothing_depends_on_them() {
    let temp = TempDir::new();
    let dir = temp.at("journaled");
    let bm25 = Weighting::Bm25 { k1: 1.2, b: 0.75 };
    let live = many(5_000, Unlink::Strand, bm25);
    remove_batch(&live, 0..3_000);
    assert_eq!(holes(&live), 3_000);
    live.journal_to(dir.to_str().unwrap(), Durability::default())
        .unwrap();
    let start = close_ups();
    // 4,100 holes live, past the bound, and 1,100 in a replay from the
    // checkpoint, within it.
    remove_batch(&live, 3_000..4_100);
    assert_eq!((holes(&live), close_ups()), (0, start + 1));
    put(&live, 5_000..5_100);
    assert!(live.detach_sink().is_some());

    let replay_start = close_ups();
    let (recovered, report) =
        Collection::recover(dir.to_str().unwrap(), None, Durability::default()).unwrap();
    assert_eq!(report.replayed, 101, "one removal and a hundred inserts");
    assert_eq!(
        close_ups(),
        replay_start,
        "the replay stays within the bound"
    );
    assert_eq!(holes(&recovered), 1_100);
    assert!(recovered.positions_agree());
    assert_eq!(live.ids().slot_of("r4100"), Some(0));
    assert_eq!(recovered.ids().slot_of("r4100"), Some(1_100));
    assert_eq!(recovered.internal_ids(), live.internal_ids());
    let probe = Probe::new(&mut Rng(0x5eed_0206), &bm25);
    assert_eq!(pages(&recovered, &probe), pages(&live, &probe));
    assert!(recovered.detach_sink().is_some());
    let (a, b) = (temp.at("live"), temp.at("recovered"));
    live.save(a.to_str().unwrap()).unwrap();
    recovered.save(b.to_str().unwrap()).unwrap();
    let (a, b) = (saved_files(&a), saved_files(&b));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (name, bytes) in &a {
        assert!(bytes == &b[name], "{name} differs from the live save");
    }
}
