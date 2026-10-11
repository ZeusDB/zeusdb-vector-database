//! Dense positions, held by a property test.
//!
//! A record's position names it in every per-record structure. Positions run
//! from zero in internal id order, with no gap but the holes removals leave,
//! and a compaction, a rebuild, the training rebuild, a load and a clear
//! close them. One seeded generator drives random sequences of writes on a
//! collection of up to 2,000 records at dimension 8, beside a reference model
//! that holds, for each external id, the internal id, the vector, the fields
//! and the sparse half the record was given. Each run declares one field and
//! a sparse space, and the runs cover raw, product quantized with and without
//! raw vectors, and scalar quantized declarations, each with a training size
//! the sequence crosses.
//!
//! After every step the collection answers what the model answers. Its
//! positions agree, its records, ids, fields and sparse halves are the
//! model's, `count`, `list` and `get_records` are the model's, every exact
//! path is brute force over the model, and every traversal returns the page
//! the same graph returns under a test by internal id that the model answers,
//! read through the nodes' origin ids. After every save the artefacts list
//! the model's records in increasing internal id, a load answers the same,
//! and a second save writes the same bytes. After a recovery from a
//! checkpoint and a journal, the recovered collection answers and saves what
//! the live one does. After every compaction, rebuild and load, every
//! per-record structure reaches the records held and no further.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use zeusdb_vector_core::{
    compile_filter, matches_filter, unframe, Filter, FrameKind, IdfScope, Position, SparseVector,
    VectorIndex,
};
use zeusdb_vector_sparse::{SparseConfig, Weighting};

use super::{Collection, Declaration, Int8Scale, ParsedRecord, SparseHalf, StorageMode};
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

/// Everything asked after every step.
fn check(
    collection: &Collection,
    model: &Model,
    kind: Kind,
    weighting: &Weighting,
    probe: &Probe,
    step: &str,
) {
    assert!(collection.positions_agree(), "{step}: the positions agree");
    if collection.ids.read().unwrap().extent() > model.len() {
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
        assert_eq!(view.metadata, held.fields, "{step}: {}'s fields", view.id);
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
    let window = |from: usize, number: usize| -> Vec<(String, HashMap<String, Value>)> {
        ordered
            .iter()
            .skip(from)
            .take(number)
            .map(|(name, held)| (name.to_string(), held.fields.clone()))
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

    // vectors.bin, where the collection holds raw vectors.
    let path = dir.join("vectors.bin");
    if collection.holds_raw_vectors() && !ordered.is_empty() {
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
        assert!(!path.exists(), "{step}: no raw vectors, so no vectors.bin");
    }

    // pq_codes.bin, where the collection holds codes.
    let path = dir.join("pq_codes.bin");
    if collection.is_quantized()
        && matches!(kind, Kind::PqWithRaw | Kind::PqOnly)
        && !ordered.is_empty()
    {
        let pq = collection.pq().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let framed = unframe(&bytes, FrameKind::PqCodes, "pq_codes.bin").unwrap();
        let payload = framed.payload;
        let width = word(payload, 0) as usize;
        assert_eq!(framed.entries as usize, ordered.len(), "{step}: codes");
        for (i, (_, held)) in ordered.iter().enumerate() {
            let row = &payload[4 + i * (4 + width)..4 + (i + 1) * (4 + width)];
            assert_eq!(
                word(row, 0) as usize,
                held.internal,
                "{step}: code {i}'s id"
            );
            assert_eq!(
                &row[4..],
                pq.quantize(&held.vector).unwrap().as_slice(),
                "{step}: code {i}"
            );
        }
    } else {
        assert!(!path.exists(), "{step}: no codes, so no pq_codes.bin");
    }

    // The scalar rows, where the collection holds them.
    let path = dir.join("int8_rows.zdbint8");
    if collection.is_quantized() && kind == Kind::Int8 && !ordered.is_empty() {
        let bytes = std::fs::read(&path).unwrap();
        let framed = unframe(&bytes, FrameKind::Int8Rows, "int8_rows.zdbint8").unwrap();
        let payload = framed.payload;
        assert_eq!(framed.entries as usize, ordered.len(), "{step}: rows");
        let index = collection.dense().index.read().unwrap();
        let stride = payload.len() / ordered.len();
        for (i, (_, held)) in ordered.iter().enumerate() {
            let row = &payload[i * stride..(i + 1) * stride];
            assert_eq!(word(row, 0) as usize, held.internal, "{step}: row {i}'s id");
            let want: Vec<u8> = index
                .graph()
                .int8_encode(&held.vector)
                .unwrap()
                .iter()
                .map(|&b| b as u8)
                .collect();
            assert_eq!(&row[4..4 + want.len()], want.as_slice(), "{step}: row {i}");
        }
    } else {
        assert!(!path.exists(), "{step}: no rows, so no row artefact");
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

    // metadata.json, keyed by external id, in increasing internal id.
    let text = std::fs::read_to_string(dir.join("metadata.json")).unwrap();
    let parsed: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&text).unwrap();
    let with_fields: Vec<&(&str, &Held)> = ordered
        .iter()
        .filter(|(name, _)| parsed.contains_key(*name))
        .collect();
    assert_eq!(
        with_fields.len(),
        parsed.len(),
        "{step}: metadata names held records"
    );
    let mut last = 0usize;
    for (name, held) in with_fields {
        assert_eq!(
            &parsed[*name], &held.fields,
            "{step}: {name}'s saved fields"
        );
        let found = text
            .find(&format!("\n  \"{name}\": "))
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
    /// Saves whose load was refused for the sparse slot count.
    refusals: usize,
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

    /// Whether a load reads the sparse artefact a save would write now. The
    /// artefact records one past the largest internal id the space took a
    /// record under since it was made or cleared, and a load refuses a slot
    /// count past the largest id the mappings hold, plus one.
    fn loadable(&self) -> bool {
        let largest = self.model.held.values().map(|held| held.internal).max();
        self.model.sparse_slots <= largest.unwrap_or(0) + 1
    }

    /// Add one record with a sparse half, which takes the next internal id
    /// and so makes the next save loadable, where there is room for it.
    fn make_loadable(&mut self, collection: &Collection) {
        if self.model.len() == MOST {
            let name = self.held_names(1).pop().unwrap();
            assert!(collection.remove_point(name.clone()).unwrap());
            self.forget(&name);
        }
        let name = self.fresh_name();
        let mut record = record(&mut self.rng, name);
        record.sparse = Some(SparseHalf::Vector(SparseVector {
            dims: vec![self.rng.below(TERMS as usize) as u32],
            values: vec![2.0],
        }));
        let added = collection.add_records(vec![record.clone()], vec![], false);
        assert_eq!(added.total_errors, 0, "{}", self.label("make loadable"));
        self.model.insert(&record);
        assert!(self.loadable());
        self.check(collection, "made loadable");
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
        let records: Vec<ParsedRecord> = names
            .into_iter()
            .map(|name| record(&mut self.rng, name))
            .collect();
        let added = collection.add_records(records.clone(), vec![], true);
        assert_eq!(
            added.total_errors,
            0,
            "{}: {:?}",
            self.label("overwrite"),
            added.errors
        );
        for record in &records {
            self.model.insert(record);
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
            let trained = !was && collection.is_quantized();
            return ("insert", trained.then_some(false));
        }
        match self.rng.below(100) {
            0..=19 => {
                self.overwrite(collection);
                ("overwrite", None)
            }
            20..=49 => {
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
        check(
            collection,
            &self.model,
            self.kind,
            &self.weighting,
            &self.probe,
            &self.label(what),
        );
    }

    /// Save, check the artefacts, load, check the load and save it again.
    fn save_and_load(&mut self, collection: Collection) -> Collection {
        let label = self.label("save");
        let dir = self.temp.at(&format!("save-{}", self.step));
        let loadable = self.loadable();
        collection.save(dir.to_str().unwrap()).unwrap();
        check_artefacts(&dir, &collection, &self.model, self.kind, &label);
        if !loadable {
            // The sparse artefact's slot count reaches past the largest id
            // the mappings hold, which a load refuses before it files a
            // record, and the collection carries on as it was.
            let refused = Collection::load(dir.to_str().unwrap()).err();
            let refused = refused.unwrap_or_else(|| panic!("{label}: the load was refused"));
            assert!(
                refused.to_string().contains("record slots"),
                "{label}: {refused}"
            );
            self.refusals += 1;
            tally("load refused for the slot count");
            return collection;
        }
        let before = pages(&collection, &self.probe);
        let loaded = Collection::load(dir.to_str().unwrap()).unwrap();
        tally("load");
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
            assert!(
                bytes == &second[name],
                "{label}: {name} differs on the second save"
            );
        }
        loaded
    }

    /// Journal from a checkpoint, run further writes, and recover from the
    /// checkpoint and the journal into a collection that answers and saves
    /// what the live one does.
    fn journal(&mut self, collection: &Collection) {
        if !self.loadable() {
            self.make_loadable(collection);
        }
        let dir = self.temp.at(&format!("journal-{}", self.step));
        collection
            .journal_to(dir.to_str().unwrap(), Durability::default())
            .unwrap();
        let writes = 2 + self.rng.below(6);
        for k in 0..writes {
            self.step += 1;
            let (what, _) = self.write(collection, true);
            tally(what);
            self.check(collection, what);
            if k == 0 && self.loadable() && self.rng.chance(30) {
                collection.checkpoint().unwrap();
            }
        }
        assert!(collection.detach_sink().is_some());
        let label = self.label("recovered");
        let live_pages = pages(collection, &self.probe);
        let live_dir = self.temp.at(&format!("journal-{}-live", self.step));
        collection.save(live_dir.to_str().unwrap()).unwrap();
        let (recovered, _) =
            Collection::recover(dir.to_str().unwrap(), None, Durability::default()).unwrap();
        tally("recovery");
        self.check(&recovered, "recovered");
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
        refusals: 0,
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
            if !run.loadable() && run.rng.chance(60) {
                run.make_loadable(&collection);
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
    fields: Option<HashMap<String, Value>>,
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
                record.metadata,
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

    let temp = TempDir::new();
    let path = temp.at("codes-short.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(path.join(zeusdb_vector_core::DUMP_FILENAME)).unwrap();
    let refused = Collection::load(path.to_str().unwrap()).err().unwrap();
    assert!(
        refused.to_string().contains(
            "the rebuild from codes built 1018 nodes for the 1019 records the index holds"
        ),
        "{refused}"
    );
}
