//! Proof that the flat traversal returns the vendored page, bit for bit.
//!
//! The bar is identical results on identical topology, so every comparison
//! here feeds one parsed dump to both constructors, `Hnsw::from_loaded_points`
//! on the vendored side and `FlatGraph::from_loaded` on the flat side, and
//! then runs the same queries through `Hnsw::search_filter` and
//! `FlatGraph::search`. A page matches only if it has the same length, the
//! same ids in the same order, and the same score bits at every position.
//!
//! The small tests build their graphs in process and run on every `cargo
//! test`. The `#[ignore]` tests are the relay harness: they read the artifact
//! directory named by `ZEUSDB_RELAY77_DIR`, which holds indexes built and
//! saved by the shipped Python path over the relay 55 datasets, and they run
//! the full grid, the latency measurement, the build timing and the memory
//! holds. They are ignored because they need those artifacts, not because they
//! are optional.

use super::dump::{parse_dump, write_dump, DumpElement, Expected, GraphKind, ParsedDump};
use super::flat::FlatGraph;
use super::levels::{LevelGenerator, DEFAULT_LEVEL_SEED};
use super::mutable::MutableGraph;
use super::traverse::Topology;
use super::{Distance, GraphHit};
use crate::distance::{CosineDist, L1Dist, L2Dist};
use crate::hnsw_index::DistPQ;
use crate::pq::PQ;
use hnsw_rs::hnsw::{LoadedEdge, LoadedPoint, PointId, NB_LAYER_MAX};
use hnsw_rs::prelude::{FilterT, Hnsw, Neighbour};
use std::io::Write as _;
use std::sync::Arc;

type BoxedFilter = Box<dyn Fn(&usize) -> bool>;

// ============================================================================
// COMPARISON CORE
// ============================================================================

/// One cell's outcome: pages compared, hits compared, and every difference.
#[derive(Default)]
struct CellOutcome {
    pages: usize,
    hits: usize,
    mismatched_pages: usize,
    /// At most the first few differences, each with its magnitude.
    detail: Vec<String>,
    /// The largest absolute score difference seen on any mismatched position.
    worst_score_gap: f32,
}

impl CellOutcome {
    fn absorb(&mut self, other: CellOutcome) {
        self.pages += other.pages;
        self.hits += other.hits;
        self.mismatched_pages += other.mismatched_pages;
        self.worst_score_gap = self.worst_score_gap.max(other.worst_score_gap);
        for line in other.detail {
            if self.detail.len() < 8 {
                self.detail.push(line);
            }
        }
    }
}

/// Compare one vendored page against one flat page, id and score bits both.
fn compare_pages(
    label: &str,
    vendored: &[Neighbour],
    flat: &[GraphHit],
    outcome: &mut CellOutcome,
) {
    outcome.pages += 1;
    let mut differs = false;
    if vendored.len() != flat.len() {
        differs = true;
        if outcome.detail.len() < 8 {
            outcome.detail.push(format!(
                "{}: page length {} vendored against {} flat",
                label,
                vendored.len(),
                flat.len()
            ));
        }
    }
    for (at, (v, f)) in vendored.iter().zip(flat.iter()).enumerate() {
        outcome.hits += 1;
        if v.d_id != f.internal_id {
            differs = true;
            if outcome.detail.len() < 8 {
                outcome.detail.push(format!(
                    "{}: position {} id {} vendored against {} flat",
                    label, at, v.d_id, f.internal_id
                ));
            }
        }
        if v.distance.to_bits() != f.distance.to_bits() {
            differs = true;
            let gap = (v.distance - f.distance).abs();
            outcome.worst_score_gap = outcome.worst_score_gap.max(gap);
            if outcome.detail.len() < 8 {
                outcome.detail.push(format!(
                    "{}: position {} score {:e} ({:08x}) vendored against {:e} ({:08x}) flat",
                    label,
                    at,
                    v.distance,
                    v.distance.to_bits(),
                    f.distance,
                    f.distance.to_bits()
                ));
            }
        }
    }
    if differs {
        outcome.mismatched_pages += 1;
    }
}

/// One of ZeusDB's two structures, as the comparison sees it.
///
/// Both run the same traversal, so this is not an abstraction over two
/// searches. It is what lets one grid be run twice, once per layout, so that
/// the mutable structure is held to the vendored page by exactly the cells the
/// CSR was held to.
trait Candidate<T> {
    fn page(
        &self,
        query: &[T],
        knbn: usize,
        ef: usize,
        filter: Option<&BoxedFilter>,
    ) -> Vec<GraphHit>;
    fn nodes(&self) -> usize;
    fn stored_vector(&self, node: u32) -> Vec<T>;
    fn stored_edges(&self) -> usize;
    fn above_level(&self) -> usize;
    fn bytes(&self) -> usize;
    fn layout(&self) -> &'static str;
}

impl<T, D> Candidate<T> for FlatGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    fn page(
        &self,
        query: &[T],
        knbn: usize,
        ef: usize,
        filter: Option<&BoxedFilter>,
    ) -> Vec<GraphHit> {
        self.search(query, knbn, ef, filter)
    }
    fn nodes(&self) -> usize {
        FlatGraph::nb_points(self)
    }
    fn stored_vector(&self, node: u32) -> Vec<T> {
        FlatGraph::vector(self, node).to_vec()
    }
    fn stored_edges(&self) -> usize {
        FlatGraph::nb_edges(self)
    }
    fn above_level(&self) -> usize {
        FlatGraph::above_level_edges(self)
    }
    fn bytes(&self) -> usize {
        FlatGraph::memory_bytes(self)
    }
    fn layout(&self) -> &'static str {
        "flat"
    }
}

impl<T, D> Candidate<T> for MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    fn page(
        &self,
        query: &[T],
        knbn: usize,
        ef: usize,
        filter: Option<&BoxedFilter>,
    ) -> Vec<GraphHit> {
        self.search(query, knbn, ef, filter)
    }
    fn nodes(&self) -> usize {
        MutableGraph::nb_points(self)
    }
    fn stored_vector(&self, node: u32) -> Vec<T> {
        MutableGraph::vector(self, node).to_vec()
    }
    fn stored_edges(&self) -> usize {
        MutableGraph::nb_edges(self)
    }
    fn above_level(&self) -> usize {
        MutableGraph::above_level_edges(self)
    }
    fn bytes(&self) -> usize {
        MutableGraph::memory_bytes(self)
    }
    fn layout(&self) -> &'static str {
        "mutable"
    }
}

/// Run one query through the vendored graph and one candidate, and compare.
#[allow(clippy::too_many_arguments)]
fn compare_one<T, D, C>(
    hnsw: &Hnsw<'static, T, D>,
    candidate: &C,
    query: &[T],
    knbn: usize,
    ef: usize,
    filter: Option<&BoxedFilter>,
    label: &str,
    outcome: &mut CellOutcome,
) where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
    C: Candidate<T> + ?Sized,
{
    let vendored = hnsw.search_filter(query, knbn, ef, filter.map(|f| f as &dyn FilterT));
    let hits = candidate.page(query, knbn, ef, filter);
    compare_pages(label, &vendored, &hits, outcome);
}

/// The predicate a filter kind stands for, over origin ids.
fn predicate(kind: &str) -> Option<BoxedFilter> {
    match kind {
        "none" => None,
        "all" => Some(Box::new(|_: &usize| true)),
        "half" => Some(Box::new(|id: &usize| id.is_multiple_of(2))),
        "sparse" => Some(Box::new(|id: &usize| id.is_multiple_of(1000))),
        "nothing" => Some(Box::new(|_: &usize| false)),
        other => panic!("unknown filter kind {}", other),
    }
}

/// The `ef_search` the shipped search resolves when the caller names none.
fn default_ef(space: &str, top_k: usize) -> usize {
    match space {
        "l1" | "l2" => (2 * top_k).max(150),
        _ => (2 * top_k).max(100),
    }
}

// ============================================================================
// SMALL GRAPH FIXTURES
// ============================================================================

/// Deterministic pseudo-random vectors, the generator the dump tests use.
fn sample_vectors(records: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    (0..records)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 16_777_216.0 - 0.5
                })
                .collect()
        })
        .collect()
}

/// Build a graph by sequential insertion, which is the shipped build path.
fn build_raw<D>(
    data: &[Vec<f32>],
    m: usize,
    ef_construction: usize,
    dist: D,
) -> Hnsw<'static, f32, D>
where
    D: Distance<f32> + Send + Sync,
{
    let hnsw = Hnsw::new(
        m,
        data.len().max(1),
        NB_LAYER_MAX as usize,
        ef_construction,
        dist,
    );
    for (id, vector) in data.iter().enumerate() {
        hnsw.insert((vector.as_slice(), id));
    }
    hnsw
}

/// Round a built graph through the dump writer and reader, which is the
/// topology source the production loader uses, and hand the parse back.
fn parsed_topology<T, D>(hnsw: &Hnsw<'static, T, D>, kind: GraphKind) -> ParsedDump<T>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let dir = tempfile::tempdir().unwrap();
    write_dump(hnsw, kind, dir.path()).unwrap();
    let expected = Expected {
        kind,
        dimension: hnsw.get_point_indexation().get_data_dimension(),
        m: hnsw.get_max_nb_connection_full(),
        ef_construction: hnsw.get_ef_construction(),
        min_nodes: 0,
    };
    parse_dump::<T>(dir.path(), &expected).unwrap()
}

/// Clone a parse, so one file read feeds both constructors.
fn clone_parse<T: Clone>(parsed: &ParsedDump<T>) -> ParsedDump<T> {
    ParsedDump {
        points_by_layer: parsed.points_by_layer.clone(),
        entry: parsed.entry,
        m: parsed.m,
        ef_construction: parsed.ef_construction,
        level_scale: parsed.level_scale,
        nb_point: parsed.nb_point,
    }
}

/// All three implementations, fed the identical parsed topology.
struct Trio<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Distance<T> + Send + Sync,
{
    vendored: Hnsw<'static, T, D>,
    flat: FlatGraph<T, D>,
    mutable: MutableGraph<T, D>,
}

fn trio_from<T, D>(parsed: ParsedDump<T>, dist: impl Fn() -> D) -> Trio<T, D>
where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
{
    let first = clone_parse(&parsed);
    let second = clone_parse(&parsed);
    let vendored = Hnsw::from_loaded_points(
        first.points_by_layer,
        first.entry,
        first.m,
        first.ef_construction,
        first.level_scale,
        dist(),
    )
    .unwrap();
    let flat = FlatGraph::from_loaded(
        second.points_by_layer,
        second.entry,
        second.m,
        second.ef_construction,
        second.level_scale,
        dist(),
    )
    .unwrap();
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        dist(),
    )
    .unwrap();
    Trio {
        vendored,
        flat,
        mutable,
    }
}

/// Run the standard small grid over both candidates and assert both clean.
fn check_trio<D>(name: &str, space: &str, trio: &Trio<f32, D>, queries: &[Vec<f32>])
where
    D: Distance<f32> + Send + Sync,
{
    assert_clean(
        &format!("{} flat", name),
        run_small_grid(&trio.vendored, &trio.flat, space, queries),
    );
    assert_clean(
        &format!("{} mutable", name),
        run_small_grid(&trio.vendored, &trio.mutable, space, queries),
    );
}

/// The standard small grid: every `top_k`, width and filter combination the
/// relay names, with held-out queries and self queries both.
fn run_small_grid<D, C>(
    hnsw: &Hnsw<'static, f32, D>,
    candidate: &C,
    space: &str,
    queries: &[Vec<f32>],
) -> CellOutcome
where
    D: Distance<f32> + Send + Sync,
    C: Candidate<f32> + ?Sized,
{
    let nb = candidate.nodes();
    let self_queries: Vec<Vec<f32>> = (0..20)
        .map(|i| candidate.stored_vector((i * nb / 20) as u32))
        .collect();

    let mut outcome = CellOutcome::default();
    for &top_k in &[1usize, 10, 100] {
        for &ef in &[default_ef(space, top_k), 10, 7, 1] {
            for kind in ["none", "all", "half"] {
                let filter = predicate(kind);
                let label = format!("k{} ef{} {}", top_k, ef, kind);
                for query in queries.iter().chain(self_queries.iter()) {
                    compare_one(
                        hnsw,
                        candidate,
                        query,
                        top_k,
                        ef,
                        filter.as_ref(),
                        &label,
                        &mut outcome,
                    );
                }
            }
        }
    }
    // The short page, the empty filter, and the degenerate width.
    for (top_k, ef, kind) in [
        (10usize, 100usize, "sparse"),
        (100, 100, "sparse"),
        (10, 100, "nothing"),
        (1, 1, "half"),
    ] {
        let filter = predicate(kind);
        let label = format!("k{} ef{} {}", top_k, ef, kind);
        for query in queries.iter().take(10).chain(self_queries.iter().take(5)) {
            compare_one(
                hnsw,
                candidate,
                query,
                top_k,
                ef,
                filter.as_ref(),
                &label,
                &mut outcome,
            );
        }
    }
    outcome
}

fn assert_clean(name: &str, outcome: CellOutcome) {
    println!(
        "parity {}: {} pages, {} hits, {} mismatched",
        name, outcome.pages, outcome.hits, outcome.mismatched_pages
    );
    assert_eq!(
        outcome.mismatched_pages, 0,
        "{} pages differed; first differences: {:?}; worst score gap {:e}",
        outcome.mismatched_pages, outcome.detail, outcome.worst_score_gap
    );
}

// ============================================================================
// SMALL TESTS, RUN ON EVERY `cargo test`
// ============================================================================

#[test]
fn flat_matches_vendored_on_cosine() {
    let data = sample_vectors(1500, 24, 0x77_01);
    let hnsw = build_raw(&data, 16, 64, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let trio = trio_from(parsed, || CosineDist {});
    let queries = sample_vectors(60, 24, 0x77_02);
    check_trio("cosine", "cosine", &trio, &queries);
}

#[test]
fn flat_matches_vendored_on_l2() {
    let data = sample_vectors(900, 16, 0x77_03);
    let hnsw = build_raw(&data, 16, 48, L2Dist {});
    let parsed = parsed_topology(&hnsw, GraphKind::L2);
    let trio = trio_from(parsed, || L2Dist {});
    let queries = sample_vectors(40, 16, 0x77_04);
    check_trio("l2", "l2", &trio, &queries);
}

#[test]
fn flat_matches_vendored_on_l1() {
    let data = sample_vectors(700, 12, 0x77_05);
    let hnsw = build_raw(&data, 8, 48, L1Dist {});
    let parsed = parsed_topology(&hnsw, GraphKind::L1);
    let trio = trio_from(parsed, || L1Dist {});
    let queries = sample_vectors(40, 12, 0x77_06);
    check_trio("l1", "l1", &trio, &queries);
}

/// Ties are where a merely plausible traversal drifts: with many identical
/// vectors the heaps hold runs of equal distances, and only an identical
/// comparison sequence returns the identical page.
#[test]
fn flat_matches_vendored_under_ties() {
    let distinct = sample_vectors(25, 16, 0x77_07);
    let data: Vec<Vec<f32>> = (0..600).map(|i| distinct[i % 25].clone()).collect();
    let hnsw = build_raw(&data, 16, 48, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let trio = trio_from(parsed, || CosineDist {});

    let tie_grid = |candidate: &dyn Candidate<f32>| {
        let mut outcome = CellOutcome::default();
        for &top_k in &[1usize, 10, 100] {
            for &ef in &[100usize, 10] {
                for kind in ["none", "half"] {
                    let filter = predicate(kind);
                    let label = format!("ties {} k{} ef{} {}", candidate.layout(), top_k, ef, kind);
                    for query in distinct.iter() {
                        compare_one(
                            &trio.vendored,
                            candidate,
                            query,
                            top_k,
                            ef,
                            filter.as_ref(),
                            &label,
                            &mut outcome,
                        );
                    }
                }
            }
        }
        outcome
    };
    assert_clean("ties flat", tie_grid(&trio.flat));
    assert_clean("ties mutable", tie_grid(&trio.mutable));
}

/// The quantized ADC path: codes in the graph, the query living in the thread
/// local table, and the dummy code slice standing in for it.
#[test]
fn flat_matches_vendored_on_quantized_adc() {
    const N: usize = 800;
    const DIM: usize = 32;
    const SUBVECTORS: usize = 8;

    let all = crate::test_vectors::clustered(N + 40, DIM, 0x77_08);
    let data = &all[..N];
    let queries = &all[N..];

    let pq = Arc::new(PQ::new(DIM, SUBVECTORS, 8, 500, None));
    pq.train(data).unwrap();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let codes = pq.quantize_batch(&refs).unwrap();

    let built: Hnsw<'static, u8, DistPQ> =
        Hnsw::new(16, N, NB_LAYER_MAX as usize, 100, DistPQ::new(pq.clone()));
    for (id, code) in codes.iter().enumerate() {
        built.insert((code.as_slice(), id));
    }

    let parsed = parsed_topology(&built, GraphKind::CosinePq);
    let trio = trio_from(parsed, || DistPQ::new(pq.clone()));

    let dummy = vec![0u8; SUBVECTORS];
    let adc_grid = |candidate: &dyn Candidate<u8>| {
        let mut outcome = CellOutcome::default();
        for &top_k in &[1usize, 10, 50] {
            for &ef in &[100usize, 10] {
                for kind in ["none", "half", "sparse", "nothing"] {
                    let filter = predicate(kind);
                    let label = format!("adc {} k{} ef{} {}", candidate.layout(), top_k, ef, kind);
                    for query in queries.iter().chain(data.iter().step_by(40)) {
                        let _lut = trio
                            .vendored
                            .get_distance()
                            .install_query_lut(query)
                            .unwrap();
                        compare_one(
                            &trio.vendored,
                            candidate,
                            &dummy,
                            top_k,
                            ef,
                            filter.as_ref(),
                            &label,
                            &mut outcome,
                        );
                    }
                }
            }
        }
        outcome
    };
    assert_clean("quantized adc flat", adc_grid(&trio.flat));
    assert_clean("quantized adc mutable", adc_grid(&trio.mutable));
}

/// What a non-finite query does if one ever reaches the traversal. Every
/// ZeusDB entry point rejects it first, so this documents the behaviour
/// rather than supporting it: an infinite query traverses and scores its page
/// identically on both structures, and a NaN query panics on both at the same
/// place with the same message. The place is the candidate assertion rather
/// than the heap ordering, because the first heap holds one element and a
/// one element heap never compares, so `NaN <= 0.` fails first.
#[test]
fn non_finite_queries_behave_identically() {
    let data = sample_vectors(400, 8, 0x77_09);
    let hnsw = build_raw(&data, 16, 48, L2Dist {});
    let parsed = parsed_topology(&hnsw, GraphKind::L2);
    let trio = trio_from(parsed, || L2Dist {});

    let infinite = vec![f32::INFINITY; 8];
    let mut outcome = CellOutcome::default();
    compare_one(
        &trio.vendored,
        &trio.flat,
        &infinite,
        10,
        20,
        None,
        "inf flat",
        &mut outcome,
    );
    compare_one(
        &trio.vendored,
        &trio.mutable,
        &infinite,
        10,
        20,
        None,
        "inf mutable",
        &mut outcome,
    );
    assert_clean("infinite query", outcome);

    let nan = vec![f32::NAN; 8];
    let vendored_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trio.vendored.search_filter(&nan, 10, 20, None)
    }));
    let mutable_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trio.mutable.search(&nan, 10, 20, None::<&BoxedFilter>)
    }));
    let flat_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trio.flat.search(&nan, 10, 20, None::<&BoxedFilter>)
    }));
    match mutable_panic {
        Err(any) => {
            let text = any
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| any.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            assert_eq!(text, "assertion failed: c.dist_to_ref <= 0.");
        }
        Ok(_) => panic!("a NaN query should panic on the mutable structure too"),
    }
    let message = |any: Box<dyn std::any::Any + Send>| -> String {
        any.downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| any.downcast_ref::<String>().cloned())
            .unwrap_or_default()
    };
    match (vendored_panic, flat_panic) {
        (Err(v), Err(f)) => {
            assert_eq!(message(v), "assertion failed: c.dist_to_ref <= 0.");
            assert_eq!(message(f), "assertion failed: c.dist_to_ref <= 0.");
        }
        (v, f) => panic!(
            "a NaN query should panic on both structures, vendored {:?} flat {:?}",
            v.is_ok(),
            f.is_ok()
        ),
    }
}

/// The parameters construction needs survive the load unchanged.
#[test]
fn the_flat_graph_keeps_the_construction_parameters() {
    let data = sample_vectors(300, 8, 0x77_0a);
    let hnsw = build_raw(&data, 24, 80, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let entry = parsed.entry;
    let scale = parsed.level_scale;
    let flat = FlatGraph::from_loaded(
        parsed.points_by_layer,
        entry,
        parsed.m,
        parsed.ef_construction,
        scale,
        CosineDist {},
    )
    .unwrap();
    assert_eq!(flat.nb_points(), 300);
    assert_eq!(flat.m(), 24);
    assert_eq!(flat.ef_construction(), 80);
    assert_eq!(flat.level_scale(), scale);
    assert_eq!(flat.entry_point_id(), entry);
    assert_eq!(flat.dim(), 8);
}

/// Every rejection `from_loaded_points` makes, made here too, plus the one
/// stricter rule the flat layout adds.
#[test]
fn the_loader_refuses_malformed_topology() {
    let point = |neighbours: Vec<Vec<LoadedEdge>>| LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours,
    };
    let refuse = |points: Vec<Vec<LoadedPoint<f32>>>, entry: PointId, needle: &str| {
        let error = FlatGraph::from_loaded(points, entry, 16, 64, 0.36, CosineDist {})
            .err()
            .expect("a malformed topology must be refused");
        assert!(error.contains(needle), "{}", error);
    };

    refuse(vec![], PointId(0, 0), "between 1 and");
    refuse(vec![vec![]], PointId(0, 0), "no entry point");
    refuse(
        vec![vec![point(vec![])]],
        PointId(0, 5),
        "no point is there",
    );
    refuse(
        vec![vec![point(vec![])]],
        PointId(3, 0),
        "no point is there",
    );
    // A width disagreement between two points.
    refuse(
        vec![vec![
            point(vec![]),
            LoadedPoint {
                origin_id: 1,
                data: vec![0.0f32; 3],
                neighbours: vec![],
            },
        ]],
        PointId(0, 0),
        "holds 3 values",
    );
    // An edge out of range, by layer and by rank.
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(9, 0),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 7),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    // A non-finite stored distance.
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 0),
            distance: f32::NAN,
        }]])]],
        PointId(0, 0),
        "carries a distance",
    );
    // A list above the point's own level: the descent residue the vendored
    // insert leaves behind. Both constructors accept it; the flat loader
    // validates it, counts it, and holds no edge for it, because no traversal
    // reaches it.
    let above = vec![vec![point(vec![
        vec![],
        vec![LoadedEdge {
            target: PointId(0, 0),
            distance: 0.5,
        }],
    ])]];
    assert!(
        Hnsw::<f32, CosineDist>::from_loaded_points(
            above.clone(),
            PointId(0, 0),
            16,
            64,
            0.36,
            CosineDist {}
        )
        .is_ok(),
        "the vendored constructor accepts the descent residue"
    );
    let flat = FlatGraph::from_loaded(above, PointId(0, 0), 16, 64, 0.36, CosineDist {}).unwrap();
    assert_eq!(flat.nb_edges(), 0);
    assert_eq!(flat.above_level_edges(), 1);
    // And a malformed edge inside such a list is still refused.
    refuse(
        vec![vec![point(vec![
            vec![],
            vec![LoadedEdge {
                target: PointId(0, 3),
                distance: 0.5,
            }],
        ])]],
        PointId(0, 0),
        "that layer holds",
    );

    // Parameter bounds.
    let ok = || vec![vec![point(vec![])]];
    assert!(FlatGraph::from_loaded(ok(), PointId(0, 0), 0, 64, 0.36, CosineDist {}).is_err());
    assert!(FlatGraph::from_loaded(ok(), PointId(0, 0), 300, 64, 0.36, CosineDist {}).is_err());
    assert!(FlatGraph::from_loaded(ok(), PointId(0, 0), 16, 64, f64::NAN, CosineDist {}).is_err());
    assert!(FlatGraph::from_loaded(ok(), PointId(0, 0), 16, 64, -1.0, CosineDist {}).is_err());
}

/// The memory figure is exact arithmetic over the buffers, so it can be
/// checked against the counts rather than sampled.
#[test]
fn the_memory_figure_is_exact() {
    let data = sample_vectors(500, 12, 0x77_0b);
    let hnsw = build_raw(&data, 8, 32, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let layer_lens: Vec<usize> = parsed.points_by_layer.iter().map(Vec::len).collect();
    let flat = FlatGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        CosineDist {},
    )
    .unwrap();

    let nodes = flat.nb_points();
    let mut expected = std::mem::size_of_val(&flat);
    expected += nodes * std::mem::size_of::<usize>();
    expected += nodes * 12 * std::mem::size_of::<f32>();
    expected += flat.nb_edges() * std::mem::size_of::<u32>();
    let mut before = 0usize;
    for len in layer_lens.iter().take(NB_LAYER_MAX as usize) {
        expected += (nodes - before + 1) * std::mem::size_of::<u32>();
        before += len;
    }
    assert_eq!(flat.memory_bytes(), expected);
}

/// The concurrency shape the design states: plain data, shareable across the
/// threads a released-GIL search runs on.
#[test]
fn the_flat_graph_is_send_and_sync() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<FlatGraph<f32, CosineDist>>();
    assert_send_sync::<FlatGraph<f32, L2Dist>>();
    assert_send_sync::<FlatGraph<f32, L1Dist>>();
    assert_send_sync::<FlatGraph<u8, DistPQ>>();
}

// ============================================================================
// THE MUTABLE STRUCTURE
// ============================================================================

/// The parameters construction needs survive the load unchanged here too.
#[test]
fn the_mutable_graph_keeps_the_construction_parameters() {
    let data = sample_vectors(300, 8, 0x78_0a);
    let hnsw = build_raw(&data, 24, 80, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let entry = parsed.entry;
    let scale = parsed.level_scale;
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        entry,
        parsed.m,
        parsed.ef_construction,
        scale,
        CosineDist {},
    )
    .unwrap();
    assert_eq!(mutable.nb_points(), 300);
    assert_eq!(mutable.m(), 24);
    assert_eq!(mutable.ef_construction(), 80);
    assert_eq!(mutable.level_scale(), scale);
    assert_eq!(mutable.entry_point_id(), entry);
    assert_eq!(mutable.dim(), 8);
    // A loaded graph numbers its nodes in dump order, which is layer major, so
    // every node's level is the layer it arrived in and the levels are
    // non-decreasing across the arena.
    let mut previous = 0u8;
    for node in 0..mutable.nb_points() as u32 {
        let level = mutable.level(node);
        assert!(level >= previous);
        previous = level;
    }
}

/// Every rejection the other two constructors make, made here too, plus the
/// one rule this layout adds and the residue rule it does not share.
#[test]
fn the_mutable_loader_refuses_malformed_topology() {
    let point = |neighbours: Vec<Vec<LoadedEdge>>| LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours,
    };
    let refuse = |points: Vec<Vec<LoadedPoint<f32>>>, entry: PointId, needle: &str| {
        let error = MutableGraph::from_loaded(points, entry, 16, 64, 0.36, CosineDist {})
            .err()
            .expect("a malformed topology must be refused");
        assert!(error.contains(needle), "{}", error);
    };

    refuse(vec![], PointId(0, 0), "between 1 and");
    refuse(vec![vec![]], PointId(0, 0), "no entry point");
    refuse(
        vec![vec![point(vec![])]],
        PointId(0, 5),
        "no point is there",
    );
    refuse(
        vec![vec![point(vec![])]],
        PointId(3, 0),
        "no point is there",
    );
    refuse(
        vec![vec![
            point(vec![]),
            LoadedPoint {
                origin_id: 1,
                data: vec![0.0f32; 3],
                neighbours: vec![],
            },
        ]],
        PointId(0, 0),
        "holds 3 values",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(9, 0),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 7),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 0),
            distance: f32::NAN,
        }]])]],
        PointId(0, 0),
        "carries a distance",
    );

    // The rule this layout adds: a list longer than its slab. The vendored
    // builder never produces one, since it shrinks past the cap under the same
    // guard that grew it, so this is a dump no ZeusDB save wrote.
    let long: Vec<Vec<LoadedPoint<f32>>> = vec![vec![LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours: vec![(0..6)
            .map(|_| LoadedEdge {
                target: PointId(0, 0),
                distance: 0.5,
            })
            .collect()],
    }]];
    let error = MutableGraph::from_loaded(long, PointId(0, 0), 2, 64, 0.36, CosineDist {})
        .err()
        .expect("a list longer than its slab must be refused");
    assert!(error.contains("and a list holds 5"), "{}", error);

    // And the overflow slot itself is representable, which is the state patch
    // 3's guarded pop works in: `2 * m + 1` entries at layer zero.
    let full: Vec<Vec<LoadedPoint<f32>>> = vec![vec![LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours: vec![(0..5)
            .map(|_| LoadedEdge {
                target: PointId(0, 0),
                distance: 0.5,
            })
            .collect()],
    }]];
    let held = MutableGraph::from_loaded(full, PointId(0, 0), 2, 64, 0.36, CosineDist {}).unwrap();
    assert_eq!(held.nb_edges(), 5);

    // The descent residue is kept rather than dropped, which is the one place
    // this constructor differs from the flat one.
    let above = vec![vec![point(vec![
        vec![],
        vec![LoadedEdge {
            target: PointId(0, 0),
            distance: 0.5,
        }],
    ])]];
    let mutable =
        MutableGraph::from_loaded(above, PointId(0, 0), 16, 64, 0.36, CosineDist {}).unwrap();
    assert_eq!(mutable.nb_edges(), 0);
    assert_eq!(mutable.above_level_edges(), 1);

    let ok = || vec![vec![point(vec![])]];
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 0, 64, 0.36, CosineDist {}).is_err());
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 300, 64, 0.36, CosineDist {}).is_err());
    assert!(
        MutableGraph::from_loaded(ok(), PointId(0, 0), 16, 64, f64::NAN, CosineDist {}).is_err()
    );
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 16, 64, -1.0, CosineDist {}).is_err());
}

/// The memory figure is exact arithmetic over the buffers here too, so the
/// per-node arithmetic the relay states can be checked rather than believed.
#[test]
fn the_mutable_memory_figure_is_exact() {
    const M: usize = 8;
    const DIM: usize = 12;
    let data = sample_vectors(500, DIM, 0x78_0b);
    let hnsw = build_raw(&data, M, 32, CosineDist {});
    let parsed = parsed_topology(&hnsw, GraphKind::Cosine);
    let layer_lens: Vec<usize> = parsed.points_by_layer.iter().map(Vec::len).collect();
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        CosineDist {},
    )
    .unwrap();

    let nodes = mutable.nb_points();
    let upper_lists: usize = layer_lens
        .iter()
        .enumerate()
        .map(|(layer, len)| layer * len)
        .sum();
    let mut expected = std::mem::size_of_val(&mutable);
    // Per node, beside the vector: the origin id, the level, the layer zero
    // length, the upper base and the residue start.
    expected += nodes * std::mem::size_of::<usize>();
    expected += nodes;
    expected += nodes * std::mem::size_of::<u16>();
    expected += nodes * std::mem::size_of::<u32>();
    expected += (nodes + 1) * std::mem::size_of::<u32>();
    // The vectors.
    expected += nodes * DIM * std::mem::size_of::<f32>();
    // The layer zero slabs, `2 * m + 1` targets and distances each.
    expected += nodes * (2 * M + 1) * 2 * std::mem::size_of::<u32>();
    // The upper slabs, `m + 1` each, and one length per upper list.
    expected += upper_lists * (M + 1) * 2 * std::mem::size_of::<u32>();
    expected += upper_lists * std::mem::size_of::<u16>();
    // The residue, nine bytes an edge.
    expected += mutable.above_level_edges() * (1 + 2 * std::mem::size_of::<u32>());
    assert_eq!(mutable.memory_bytes(), expected);

    println!(
        "mutable memory nodes {} edges {} residue {} bytes {} per_node {:.2}",
        nodes,
        mutable.nb_edges(),
        mutable.above_level_edges(),
        mutable.memory_bytes(),
        (mutable.memory_bytes() - nodes * DIM * 4) as f64 / nodes as f64
    );
}

/// The concurrency shape the design states, for the structure that will hold
/// the graph after the cutover.
#[test]
fn the_mutable_graph_is_send_and_sync() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<MutableGraph<f32, CosineDist>>();
    assert_send_sync::<MutableGraph<f32, L2Dist>>();
    assert_send_sync::<MutableGraph<f32, L1Dist>>();
    assert_send_sync::<MutableGraph<u8, DistPQ>>();
}

/// Write a graph out, load it into the mutable structure, write it out again,
/// and compare the two files byte for byte.
fn round_trip<T, D>(
    built: &Hnsw<'static, T, D>,
    kind: GraphKind,
    dist: impl Fn() -> D,
) -> (usize, usize)
where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
{
    let first = tempfile::tempdir().unwrap();
    write_dump(built, kind, first.path()).unwrap();
    let expected = Expected {
        kind,
        dimension: built.get_point_indexation().get_data_dimension(),
        m: built.get_max_nb_connection_full(),
        ef_construction: built.get_ef_construction(),
        min_nodes: 0,
    };
    let parsed = parse_dump::<T>(first.path(), &expected).unwrap();
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        dist(),
    )
    .unwrap();
    let second = tempfile::tempdir().unwrap();
    write_dump(&mutable.dump_view(), kind, second.path()).unwrap();

    let before = std::fs::read(first.path().join(super::dump::DUMP_FILENAME)).unwrap();
    let after = std::fs::read(second.path().join(super::dump::DUMP_FILENAME)).unwrap();
    assert_eq!(
        before.len(),
        after.len(),
        "the dump changed length, {} against {}",
        before.len(),
        after.len()
    );
    let differing = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();
    let first_difference = before.iter().zip(after.iter()).position(|(a, b)| a != b);
    assert_eq!(
        differing, 0,
        "{} bytes differ, first at offset {:?}",
        differing, first_difference
    );
    (before.len(), mutable.above_level_edges())
}

/// A dump loaded into the mutable structure and written back out is the same
/// file, byte for byte, on every configuration the small tests cover.
///
/// This is what makes the structure a lossless image of the file rather than a
/// lossy one, and it is why the descent residue is stored. Without it the
/// second file would be shorter by exactly the residue edges.
#[test]
fn a_dump_round_trips_through_the_mutable_graph() {
    let cosine = build_raw(&sample_vectors(1500, 24, 0x78_01), 16, 64, CosineDist {});
    let (bytes, residue) = round_trip(&cosine, GraphKind::Cosine, || CosineDist {});
    println!("round trip cosine bytes {} residue {}", bytes, residue);
    assert!(residue > 0, "the fixture must carry descent residue");

    let l2 = build_raw(&sample_vectors(900, 16, 0x78_02), 16, 48, L2Dist {});
    let (bytes, residue) = round_trip(&l2, GraphKind::L2, || L2Dist {});
    println!("round trip l2 bytes {} residue {}", bytes, residue);

    let l1 = build_raw(&sample_vectors(700, 12, 0x78_03), 8, 48, L1Dist {});
    let (bytes, residue) = round_trip(&l1, GraphKind::L1, || L1Dist {});
    println!("round trip l1 bytes {} residue {}", bytes, residue);

    // Ties, where a list holds runs of equal stored distances and only a
    // stable ordering reproduces the file.
    let distinct = sample_vectors(25, 16, 0x78_04);
    let repeated: Vec<Vec<f32>> = (0..600).map(|i| distinct[i % 25].clone()).collect();
    let ties = build_raw(&repeated, 16, 48, CosineDist {});
    let (bytes, residue) = round_trip(&ties, GraphKind::Cosine, || CosineDist {});
    println!("round trip ties bytes {} residue {}", bytes, residue);

    // The quantized graph, whose elements are `u8` codes.
    const N: usize = 800;
    const DIM: usize = 32;
    const SUBVECTORS: usize = 8;
    let all = crate::test_vectors::clustered(N, DIM, 0x78_05);
    let pq = Arc::new(PQ::new(DIM, SUBVECTORS, 8, 500, None));
    pq.train(&all).unwrap();
    let refs: Vec<&[f32]> = all.iter().map(|v| v.as_slice()).collect();
    let codes = pq.quantize_batch(&refs).unwrap();
    let quantized: Hnsw<'static, u8, DistPQ> =
        Hnsw::new(16, N, NB_LAYER_MAX as usize, 100, DistPQ::new(pq.clone()));
    for (id, code) in codes.iter().enumerate() {
        quantized.insert((code.as_slice(), id));
    }
    let (bytes, residue) = round_trip(&quantized, GraphKind::CosinePq, || DistPQ::new(pq.clone()));
    println!("round trip quantized bytes {} residue {}", bytes, residue);
}

// ============================================================================
// THE LEVEL GENERATOR
// ============================================================================

/// The vendored level stream, read out through the only public route to it.
///
/// `LayerGenerator::generate` is private to the crate, so the stream is read
/// where it lands. `generate_new_point` draws exactly one level per insertion
/// and files the point in the layer of that level, so the level of the point
/// inserted under id `i` is the `i`th draw. The graph is started from a one
/// point loaded topology rather than from `Hnsw::new`, because
/// `from_loaded_points` is the constructor that takes the scale absolutely and
/// so is the only one that can be given a scale that makes the cap bind.
fn vendored_levels(scale: f64, draws: usize, warmup: usize, reseed: Option<u64>) -> Vec<u8> {
    let seed_point = LoadedPoint {
        origin_id: usize::MAX,
        data: vec![0.0f32, 0.0],
        neighbours: vec![],
    };
    let mut hnsw = Hnsw::<f32, L2Dist>::from_loaded_points(
        vec![vec![seed_point]],
        PointId(0, 0),
        4,
        4,
        scale,
        L2Dist {},
    )
    .unwrap();
    // Distinct points, because a graph of identical vectors never satisfies the
    // traversal's stopping bound and every insertion would walk the whole
    // component.
    let data = sample_vectors(warmup + draws, 2, 0x78_11);
    for j in 0..warmup {
        hnsw.insert((data[draws + j].as_slice(), draws + 1 + j));
    }
    if let Some(seed) = reseed {
        hnsw.set_level_seed(seed);
    }
    for (id, vector) in data.iter().take(draws).enumerate() {
        hnsw.insert((vector.as_slice(), id));
    }

    let mut levels = vec![u8::MAX; draws];
    for layer in 0..NB_LAYER_MAX as usize {
        for point in hnsw.get_point_indexation().get_layer_iterator(layer) {
            let id = point.get_origin_id();
            if id < draws {
                levels[id] = layer as u8;
            }
        }
    }
    assert!(
        levels.iter().all(|&l| l != u8::MAX),
        "every inserted point must be found in some layer"
    );
    levels
}

/// The same stream from ZeusDB's own generator.
fn own_levels(scale: f64, draws: usize, warmup: usize, reseed: Option<u64>) -> (Vec<u8>, usize) {
    let mut generator = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    assert_eq!(generator.scale(), scale, "the scale is installed as given");
    for _ in 0..warmup {
        generator.generate();
    }
    if let Some(seed) = reseed {
        generator.set_seed(seed);
    }
    let before = generator.redraws();
    let levels: Vec<u8> = (0..draws).map(|_| generator.generate() as u8).collect();
    (levels, generator.redraws() - before)
}

/// Compare the two streams draw for draw and report.
fn compare_streams(name: &str, scale: f64, draws: usize, warmup: usize, reseed: Option<u64>) {
    let (own, redraws) = own_levels(scale, draws, warmup, reseed);
    let vendored = vendored_levels(scale, draws, warmup, reseed);
    assert_eq!(own.len(), vendored.len());
    let differing = own
        .iter()
        .zip(vendored.iter())
        .filter(|(a, b)| a != b)
        .count();
    let first = own.iter().zip(vendored.iter()).position(|(a, b)| a != b);
    let mut histogram = [0usize; NB_LAYER_MAX as usize];
    for &level in &own {
        histogram[level as usize] += 1;
    }
    println!(
        "levels {} scale {} draws {} warmup {} reseed {:?} differing {} redraws {} histogram {:?}",
        name, scale, draws, warmup, reseed, differing, redraws, histogram
    );
    if let Some(at) = first {
        panic!(
            "the streams diverge at draw {}: own {} against vendored {}",
            at, own[at], vendored[at]
        );
    }
    assert_eq!(differing, 0);
}

/// The level streams are identical draw for draw, including where the cap
/// binds and the redraw consumes a second value from the same stream.
#[test]
fn the_level_stream_matches_the_vendored_one() {
    // The default scale at `m` 16, which is what every shipped index draws
    // with. The cap never binds here: `P(level >= 16)` is `16^-16`.
    compare_streams(
        "default",
        LevelGenerator::default_scale(16),
        100_000,
        0,
        None,
    );
    // A scale where the cap binds. `P(level >= 16)` is `exp(-16 / 4)`, about
    // one draw in fifty five, so the redraw path is exercised thousands of
    // times and every draw after the first one has to have consumed the same
    // amount of the stream.
    compare_streams("capped", 4.0, 100_000, 0, None);
}

/// `set_level_seed` resets both generators to the same place, whatever either
/// had drawn before.
#[test]
fn the_level_seed_resets_both_generators() {
    // Reseeding to the default after a warm up reproduces the cold stream.
    let (cold, _) = own_levels(LevelGenerator::default_scale(16), 2_000, 0, None);
    let (warm, _) = own_levels(
        LevelGenerator::default_scale(16),
        2_000,
        500,
        Some(DEFAULT_LEVEL_SEED),
    );
    assert_eq!(cold, warm);

    // And both generators agree after a reseed to a chosen value, with the
    // stream advanced by a different amount first.
    compare_streams("reseeded", 4.0, 5_000, 137, Some(0x0102_0304_0506_0708));
}

// ============================================================================
// THE RELAY HARNESS OVER REAL DATA, RUN BY NAME WITH THE ARTIFACT DIRECTORY
// ============================================================================

/// One artifact configuration, as the manifest records it.
struct ArtifactConfig {
    name: String,
    space: String,
    dim: usize,
    quantized: bool,
    subvectors: usize,
    index_dir: std::path::PathBuf,
    queries: std::path::PathBuf,
    m: usize,
    ef_construction: usize,
}

fn artifact_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("ZEUSDB_RELAY77_DIR").expect("set ZEUSDB_RELAY77_DIR to the artifact root"),
    )
}

fn read_manifest(root: &std::path::Path) -> Vec<ArtifactConfig> {
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    manifest["configs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let index_dir = std::path::PathBuf::from(entry["index_dir"].as_str().unwrap());
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(index_dir.join("config.json")).unwrap(),
            )
            .unwrap();
            ArtifactConfig {
                name: entry["name"].as_str().unwrap().to_string(),
                space: entry["space"].as_str().unwrap().to_string(),
                dim: entry["dim"].as_u64().unwrap() as usize,
                quantized: entry["quantized"].as_bool().unwrap(),
                subvectors: entry["subvectors"].as_u64().unwrap_or(0) as usize,
                index_dir,
                queries: std::path::PathBuf::from(entry["queries"].as_str().unwrap()),
                m: config["m"].as_u64().unwrap() as usize,
                ef_construction: config["ef_construction"].as_u64().unwrap() as usize,
            }
        })
        .collect()
}

/// The f32 query slab the Python side exported.
fn read_queries(path: &std::path::Path) -> Vec<Vec<f32>> {
    let raw = std::fs::read(path).unwrap();
    let count = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let dim = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    assert_eq!(raw.len(), 16 + count * dim * 4);
    (0..count)
        .map(|q| {
            let base = 16 + q * dim * 4;
            raw[base..base + dim * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
        .collect()
}

fn expected_for(config: &ArtifactConfig) -> Expected {
    let kind = match (config.space.as_str(), config.quantized) {
        ("l2", false) => GraphKind::L2,
        ("l1", false) => GraphKind::L1,
        (_, false) => GraphKind::Cosine,
        ("l2", true) => GraphKind::L2Pq,
        ("l1", true) => GraphKind::L1Pq,
        (_, true) => GraphKind::CosinePq,
    };
    Expected {
        kind,
        dimension: if config.quantized {
            config.subvectors
        } else {
            config.dim
        },
        m: config.m,
        ef_construction: config.ef_construction,
        min_nodes: 0,
    }
}

/// The trained PQ a quantized artifact was saved with, rebuilt from its
/// codebook exactly as the persistence loader rebuilds it.
fn load_pq(config: &ArtifactConfig) -> Arc<PQ> {
    let raw = std::fs::read(config.index_dir.join("pq_centroids.bin")).unwrap();
    let (centroids, _): (Vec<Vec<Vec<f32>>>, usize) =
        bincode::decode_from_slice(&raw, bincode::config::standard()).unwrap();
    let quant: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(config.index_dir.join("quantization.json")).unwrap(),
    )
    .unwrap();
    let pq = PQ::new(
        config.dim,
        config.subvectors,
        quant["bits"].as_u64().unwrap() as usize,
        quant["training_size"].as_u64().unwrap() as usize,
        None,
    );
    pq.set_centroids(centroids).unwrap();
    pq.set_trained(true);
    Arc::new(pq)
}

/// The raw vector store and the id mappings of a saved index, for the rerank
/// comparison.
fn load_raw_store(
    config: &ArtifactConfig,
) -> (
    std::collections::HashMap<String, Vec<f32>>,
    std::collections::HashMap<usize, String>,
) {
    let raw = std::fs::read(config.index_dir.join("vectors.bin")).unwrap();
    let (vectors, _): (std::collections::HashMap<String, Vec<f32>>, usize) =
        bincode::decode_from_slice(&raw, bincode::config::standard()).unwrap();
    let raw = std::fs::read(config.index_dir.join("mappings.bin")).unwrap();
    let (mappings, _): (crate::persistence::IdMappings, usize) =
        bincode::decode_from_slice(&raw, bincode::config::standard()).unwrap();
    (vectors, mappings.rev_map)
}

/// Self queries: stored vectors of evenly spaced nodes. On a raw graph they
/// come straight out of the arena; on a quantized one the raw vector store
/// supplies them, since the graph holds codes.
fn self_queries_raw<C>(candidate: &C, count: usize) -> Vec<Vec<f32>>
where
    C: Candidate<f32> + ?Sized,
{
    let nb = candidate.nodes();
    (0..count)
        .map(|i| candidate.stored_vector((i * nb / count) as u32))
        .collect()
}

/// The grid body, shared by every raw space through monomorphisation.
fn grid_over<D, C>(
    config: &ArtifactConfig,
    vendored: &Hnsw<'static, f32, D>,
    flat: &C,
) -> CellOutcome
where
    D: Distance<f32> + Send + Sync,
    C: Candidate<f32> + ?Sized,
{
    println!(
        "structure {} {} nodes {} edges {} above_level_edges {} bytes {}",
        config.name,
        flat.layout(),
        flat.nodes(),
        flat.stored_edges(),
        flat.above_level(),
        flat.bytes()
    );
    let queries = read_queries(&config.queries);
    let self_queries = self_queries_raw(flat, 50);
    let mut total = CellOutcome::default();

    for &top_k in &[1usize, 10, 100] {
        for &ef in &[default_ef(&config.space, top_k), 200, 50, 7] {
            for kind in ["none", "all", "half"] {
                let filter = predicate(kind);
                let mut outcome = CellOutcome::default();
                let label = format!(
                    "{} {} k{} ef{} {}",
                    config.name,
                    flat.layout(),
                    top_k,
                    ef,
                    kind
                );
                for query in queries.iter().take(250).chain(self_queries.iter()) {
                    compare_one(
                        vendored,
                        flat,
                        query,
                        top_k,
                        ef,
                        filter.as_ref(),
                        &label,
                        &mut outcome,
                    );
                }
                println!(
                    "cell {} pages {} hits {} mismatched {}",
                    label, outcome.pages, outcome.hits, outcome.mismatched_pages
                );
                total.absorb(outcome);
            }
        }
    }
    for (top_k, kind, take) in [
        (10usize, "sparse", 25usize),
        (100, "sparse", 25),
        (10, "nothing", 25),
    ] {
        let ef = default_ef(&config.space, top_k);
        let filter = predicate(kind);
        let mut outcome = CellOutcome::default();
        let label = format!(
            "{} {} k{} ef{} {}",
            config.name,
            flat.layout(),
            top_k,
            ef,
            kind
        );
        for query in queries.iter().take(take).chain(self_queries.iter().take(5)) {
            compare_one(
                vendored,
                flat,
                query,
                top_k,
                ef,
                filter.as_ref(),
                &label,
                &mut outcome,
            );
        }
        println!(
            "cell {} pages {} hits {} mismatched {}",
            label, outcome.pages, outcome.hits, outcome.mismatched_pages
        );
        total.absorb(outcome);
    }
    total
}

/// The quantized grid: the ADC cells and the reranked cells.
fn run_quantized_artifact(config: &ArtifactConfig) -> CellOutcome {
    let pq = load_pq(config);
    let parsed = parse_dump::<u8>(&config.index_dir, &expected_for(config)).unwrap();
    let trio = trio_from(parsed, || DistPQ::new(pq.clone()));
    let vendored = &trio.vendored;
    let flat = &trio.flat;
    let (raw_vectors, rev_map) = load_raw_store(config);

    let queries = read_queries(&config.queries);
    // Self queries resolve through the graph's own origin ids, because the
    // internal id sequence starts wherever `get_next_id` starts it.
    let self_queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let node = (i * flat.nb_points() / 50) as u32;
            let internal = Topology::origin_id(flat, node);
            let ext = rev_map.get(&internal).unwrap_or_else(|| {
                panic!(
                    "internal id {} is not in rev_map ({} entries)",
                    internal,
                    rev_map.len()
                )
            });
            raw_vectors
                .get(ext)
                .unwrap_or_else(|| {
                    panic!(
                        "external id {:?} is not in vectors.bin ({} entries)",
                        ext,
                        raw_vectors.len()
                    )
                })
                .clone()
        })
        .collect();
    let dummy = vec![0u8; config.subvectors];
    let mut total = CellOutcome::default();
    println!(
        "structure {} flat nodes {} edges {} above_level_edges {} bytes {}",
        config.name,
        flat.nb_points(),
        flat.nb_edges(),
        flat.above_level_edges(),
        flat.memory_bytes()
    );
    println!(
        "structure {} mutable nodes {} edges {} above_level_edges {} bytes {}",
        config.name,
        trio.mutable.nb_points(),
        trio.mutable.nb_edges(),
        trio.mutable.above_level_edges(),
        trio.mutable.memory_bytes()
    );

    // The ADC cells, over both structures.
    for &top_k in &[1usize, 10, 100] {
        for &ef in &[default_ef(&config.space, top_k), 200, 50, 7] {
            for kind in ["none", "all", "half"] {
                let filter = predicate(kind);
                let mut outcome = CellOutcome::default();
                let label = format!("{} adc k{} ef{} {}", config.name, top_k, ef, kind);
                for query in queries.iter().take(250).chain(self_queries.iter()) {
                    let _lut = vendored.get_distance().install_query_lut(query).unwrap();
                    compare_one(
                        vendored,
                        flat,
                        &dummy,
                        top_k,
                        ef,
                        filter.as_ref(),
                        &label,
                        &mut outcome,
                    );
                    compare_one(
                        vendored,
                        &trio.mutable,
                        &dummy,
                        top_k,
                        ef,
                        filter.as_ref(),
                        &label,
                        &mut outcome,
                    );
                }
                println!(
                    "cell {} pages {} hits {} mismatched {}",
                    label, outcome.pages, outcome.hits, outcome.mismatched_pages
                );
                total.absorb(outcome);
            }
        }
    }
    for (top_k, kind, take) in [
        (10usize, "sparse", 25usize),
        (100, "sparse", 25),
        (10, "nothing", 25),
    ] {
        let ef = default_ef(&config.space, top_k);
        let filter = predicate(kind);
        let mut outcome = CellOutcome::default();
        let label = format!("{} adc k{} ef{} {}", config.name, top_k, ef, kind);
        for query in queries.iter().take(take) {
            let _lut = vendored.get_distance().install_query_lut(query).unwrap();
            compare_one(
                vendored,
                flat,
                &dummy,
                top_k,
                ef,
                filter.as_ref(),
                &label,
                &mut outcome,
            );
            compare_one(
                vendored,
                &trio.mutable,
                &dummy,
                top_k,
                ef,
                filter.as_ref(),
                &label,
                &mut outcome,
            );
        }
        println!(
            "cell {} pages {} hits {} mismatched {}",
            label, outcome.pages, outcome.hits, outcome.mismatched_pages
        );
        total.absorb(outcome);
    }

    // The reranked cells: the seam page at the over-fetch, then the rescoring
    // both pages go through, which is the shipped `collect_hits` arithmetic.
    let rescore = crate::rerank::raw_distance_fn(&config.space);
    for &top_k in &[10usize, 100] {
        for &factor in &[10usize, 25] {
            let fetch = (top_k * factor).min(flat.nb_points());
            for kind in ["none", "half"] {
                let filter = predicate(kind);
                let mut outcome = CellOutcome::default();
                let label = format!("{} rerank k{} f{} {}", config.name, top_k, factor, kind);
                for query in queries.iter().take(100).chain(self_queries.iter().take(20)) {
                    let ef = default_ef(&config.space, top_k);
                    let (vendored_page, flat_page, mutable_page) = {
                        let _lut = vendored.get_distance().install_query_lut(query).unwrap();
                        (
                            vendored.search_filter(
                                &dummy,
                                fetch,
                                ef,
                                filter.as_ref().map(|f| f as &dyn FilterT),
                            ),
                            flat.search(&dummy, fetch, ef, filter.as_ref()),
                            trio.mutable.search(&dummy, fetch, ef, filter.as_ref()),
                        )
                    };
                    let flat_hits = flat_page;
                    compare_pages(&label, &vendored_page, &flat_hits, &mut outcome);
                    compare_pages(&label, &vendored_page, &mutable_page, &mut outcome);

                    // Rescore both pages exactly as `collect_hits` does and
                    // compare the final reranked results.
                    let rescored = |ids: Vec<usize>, scores_len: usize| -> Vec<(String, u32)> {
                        assert_eq!(ids.len(), scores_len);
                        let mut scored: Vec<(String, f32)> = ids
                            .into_iter()
                            .map(|internal| {
                                let ext = rev_map[&internal].clone();
                                let score = rescore(query, &raw_vectors[&ext]);
                                (ext, score)
                            })
                            .collect();
                        crate::rerank::take_best(&mut scored, top_k);
                        scored
                            .into_iter()
                            .map(|(id, s)| (id, s.to_bits()))
                            .collect()
                    };
                    let vendored_len = vendored_page.len();
                    let flat_len = flat_hits.len();
                    let vendored_final =
                        rescored(vendored_page.iter().map(|n| n.d_id).collect(), vendored_len);
                    let flat_final =
                        rescored(flat_hits.iter().map(|h| h.internal_id).collect(), flat_len);
                    outcome.pages += 1;
                    outcome.hits += vendored_final.len();
                    if vendored_final != flat_final {
                        outcome.mismatched_pages += 1;
                        if outcome.detail.len() < 8 {
                            outcome
                                .detail
                                .push(format!("{}: reranked pages differ", label));
                        }
                    }
                }
                println!(
                    "cell {} pages {} hits {} mismatched {}",
                    label, outcome.pages, outcome.hits, outcome.mismatched_pages
                );
                total.absorb(outcome);
            }
        }
    }
    total
}

/// The whole real-data parity run. Needs `ZEUSDB_RELAY77_DIR`.
#[test]
#[ignore = "needs the relay 77 artifacts; run by name with ZEUSDB_RELAY77_DIR set"]
fn real_data_parity() {
    let root = artifact_dir();
    let only = std::env::var("ZEUSDB_ONLY").ok();
    let mut grand = CellOutcome::default();
    for config in read_manifest(&root) {
        if let Some(only) = &only {
            if &config.name != only {
                continue;
            }
        }
        let outcome = if config.quantized {
            run_quantized_artifact(&config)
        } else {
            let parsed = parse_dump::<f32>(&config.index_dir, &expected_for(&config)).unwrap();
            match config.space.as_str() {
                "l2" => {
                    let trio = trio_from(parsed, || L2Dist {});
                    let mut outcome = grid_over(&config, &trio.vendored, &trio.flat);
                    outcome.absorb(grid_over(&config, &trio.vendored, &trio.mutable));
                    outcome
                }
                "l1" => {
                    let trio = trio_from(parsed, || L1Dist {});
                    let mut outcome = grid_over(&config, &trio.vendored, &trio.flat);
                    outcome.absorb(grid_over(&config, &trio.vendored, &trio.mutable));
                    outcome
                }
                _ => {
                    let trio = trio_from(parsed, || CosineDist {});
                    let mut outcome = grid_over(&config, &trio.vendored, &trio.flat);
                    outcome.absorb(grid_over(&config, &trio.vendored, &trio.mutable));
                    outcome
                }
            }
        };
        println!(
            "config {} pages {} hits {} mismatched {} worst_gap {:e}",
            config.name,
            outcome.pages,
            outcome.hits,
            outcome.mismatched_pages,
            outcome.worst_score_gap
        );
        for line in &outcome.detail {
            println!("difference {}", line);
        }
        grand.absorb(outcome);
    }
    println!(
        "grand total pages {} hits {} mismatched {}",
        grand.pages, grand.hits, grand.mismatched_pages
    );
    assert_eq!(
        grand.mismatched_pages, 0,
        "pages differed: {:?}",
        grand.detail
    );
}

// ============================================================================
// MEASUREMENT, RUN BY NAME WITH THE ARTIFACT DIRECTORY
// ============================================================================

fn nanos_of<R>(work: impl FnOnce() -> R) -> (u128, R) {
    let start = std::time::Instant::now();
    let result = work();
    (start.elapsed().as_nanos(), result)
}

fn mean_p95(mut nanos: Vec<u128>) -> (f64, f64) {
    nanos.sort_unstable();
    let mean = nanos.iter().sum::<u128>() as f64 / nanos.len() as f64;
    let p95 = nanos[(nanos.len() * 95) / 100 - 1] as f64;
    (mean / 1.0e6, p95 / 1.0e6)
}

/// Search latency, vendored against flat, on the standard cell.
#[test]
#[ignore = "measurement; run by name with ZEUSDB_RELAY77_DIR set"]
fn measure_search_latency() {
    let root = artifact_dir();
    for config in read_manifest(&root) {
        if config.quantized {
            latency_over_quantized(&config);
            continue;
        }
        let parsed = parse_dump::<f32>(&config.index_dir, &expected_for(&config)).unwrap();
        match config.space.as_str() {
            "l2" => latency_over(&config, trio_from(parsed, || L2Dist {})),
            "l1" => latency_over(&config, trio_from(parsed, || L1Dist {})),
            _ => latency_over(&config, trio_from(parsed, || CosineDist {})),
        }
    }
}

/// The quantized cell: ADC over the dummy query, the table installed per
/// query exactly as the shipped path installs it.
fn latency_over_quantized(config: &ArtifactConfig) {
    let pq = load_pq(config);
    let parsed = parse_dump::<u8>(&config.index_dir, &expected_for(config)).unwrap();
    let trio = trio_from(parsed, || DistPQ::new(pq.clone()));
    let vendored = &trio.vendored;
    let flat = &trio.flat;
    let queries: Vec<Vec<f32>> = read_queries(&config.queries)
        .into_iter()
        .take(250)
        .collect();
    let dummy = vec![0u8; config.subvectors];
    let top_k = 10usize;
    let ef = default_ef(&config.space, top_k);
    let live: BoxedFilter = Box::new(|_: &usize| true);

    for query in queries.iter().take(20) {
        let _lut = vendored.get_distance().install_query_lut(query).unwrap();
        let _ = vendored.search_filter(&dummy, top_k, ef, Some(&live as &dyn FilterT));
        let _ = flat.search(&dummy, top_k, ef, Some(&live));
        let _ = trio.mutable.search(&dummy, top_k, ef, Some(&live));
    }

    for repeat in 0..3 {
        for which in ["vendored", "flat", "mutable"] {
            let mut per_query = Vec::with_capacity(queries.len());
            for query in &queries {
                let (nanos, _hits) = match which {
                    "vendored" => nanos_of(|| {
                        let _lut = vendored.get_distance().install_query_lut(query).unwrap();
                        std::hint::black_box(
                            vendored
                                .search_filter(&dummy, top_k, ef, Some(&live as &dyn FilterT))
                                .len(),
                        )
                    }),
                    "flat" => nanos_of(|| {
                        let _lut = vendored.get_distance().install_query_lut(query).unwrap();
                        std::hint::black_box(flat.search(&dummy, top_k, ef, Some(&live)).len())
                    }),
                    _ => nanos_of(|| {
                        let _lut = vendored.get_distance().install_query_lut(query).unwrap();
                        std::hint::black_box(
                            trio.mutable.search(&dummy, top_k, ef, Some(&live)).len(),
                        )
                    }),
                };
                per_query.push(nanos);
            }
            let (mean, p95) = mean_p95(per_query);
            println!(
                "latency {} {} repeat {} mean_ms {:.4} p95_ms {:.4} queries {} k {} ef {} filtered adc",
                config.name, which, repeat, mean, p95, queries.len(), top_k, ef
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

fn latency_over<D>(config: &ArtifactConfig, trio: Trio<f32, D>)
where
    D: Distance<f32> + Send + Sync,
{
    let vendored = &trio.vendored;
    let flat = &trio.flat;
    let queries: Vec<Vec<f32>> = read_queries(&config.queries)
        .into_iter()
        .take(250)
        .collect();
    let top_k = 10usize;
    let ef = default_ef(&config.space, top_k);
    let live: BoxedFilter = Box::new(|_: &usize| true);

    // Warm both once.
    for query in queries.iter().take(20) {
        let _ = vendored.search_filter(query, top_k, ef, Some(&live as &dyn FilterT));
        let _ = flat.search(query, top_k, ef, Some(&live));
        let _ = trio.mutable.search(query, top_k, ef, Some(&live));
    }

    for repeat in 0..3 {
        for which in ["vendored", "flat", "mutable"] {
            let mut per_query = Vec::with_capacity(queries.len());
            for query in &queries {
                let (nanos, _hits) = match which {
                    "vendored" => nanos_of(|| {
                        std::hint::black_box(
                            vendored
                                .search_filter(query, top_k, ef, Some(&live as &dyn FilterT))
                                .len(),
                        )
                    }),
                    "flat" => nanos_of(|| {
                        std::hint::black_box(flat.search(query, top_k, ef, Some(&live)).len())
                    }),
                    _ => nanos_of(|| {
                        std::hint::black_box(
                            trio.mutable.search(query, top_k, ef, Some(&live)).len(),
                        )
                    }),
                };
                per_query.push(nanos);
            }
            let (mean, p95) = mean_p95(per_query);
            println!(
                "latency {} {} repeat {} mean_ms {:.4} p95_ms {:.4} queries {} k {} ef {} filtered",
                config.name,
                which,
                repeat,
                mean,
                p95,
                queries.len(),
                top_k,
                ef
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

/// Time to build each structure from one parsed topology.
#[test]
#[ignore = "measurement; run by name with ZEUSDB_RELAY77_DIR set"]
fn measure_build_time() {
    let root = artifact_dir();
    for config in read_manifest(&root) {
        if config.quantized {
            let pq = load_pq(&config);
            time_builds::<u8, _>(&config, || DistPQ::new(pq.clone()));
        } else {
            time_builds::<f32, _>(&config, || CosineDist {});
        }
    }
}

/// Time both constructors from one parse, three repeats each, clone excluded.
/// The distance value never evaluates during construction, so raw configs of
/// every space share the cosine value; only the element type matters.
fn time_builds<T, D>(config: &ArtifactConfig, dist: impl Fn() -> D)
where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
{
    let expected = expected_for(config);
    let (parse_nanos, parsed) = nanos_of(|| parse_dump::<T>(&config.index_dir, &expected).unwrap());
    println!(
        "build {} parse_ms {:.1}",
        config.name,
        parse_nanos as f64 / 1.0e6
    );
    for repeat in 0..3 {
        let copy = clone_parse(&parsed);
        let (nanos, vendored) = nanos_of(|| {
            Hnsw::from_loaded_points(
                copy.points_by_layer,
                copy.entry,
                copy.m,
                copy.ef_construction,
                copy.level_scale,
                dist(),
            )
            .unwrap()
        });
        println!(
            "build {} vendored repeat {} ms {:.1}",
            config.name,
            repeat,
            nanos as f64 / 1.0e6
        );
        drop(vendored);

        let copy = clone_parse(&parsed);
        let (nanos, flat) = nanos_of(|| {
            FlatGraph::from_loaded(
                copy.points_by_layer,
                copy.entry,
                copy.m,
                copy.ef_construction,
                copy.level_scale,
                dist(),
            )
            .unwrap()
        });
        println!(
            "build {} flat repeat {} ms {:.1} nodes {} edges {} bytes {}",
            config.name,
            repeat,
            nanos as f64 / 1.0e6,
            flat.nb_points(),
            flat.nb_edges(),
            flat.memory_bytes()
        );
        drop(flat);
        std::io::stdout().flush().unwrap();
    }
}

/// Hold one structure in memory so an outside sampler can read the resident
/// set. `ZEUSDB_HOLD` picks what is held: `vendored`, `flat`, or `parse`,
/// which parses and drops everything as the allocator-retention control.
#[test]
#[ignore = "measurement; run by name with ZEUSDB_RELAY77_DIR and ZEUSDB_HOLD set"]
fn hold_structure_for_sampling() {
    let root = artifact_dir();
    let which = std::env::var("ZEUSDB_HOLD").expect("set ZEUSDB_HOLD");
    let config_name = std::env::var("ZEUSDB_HOLD_CONFIG").expect("set ZEUSDB_HOLD_CONFIG");
    let config = read_manifest(&root)
        .into_iter()
        .find(|c| c.name == config_name)
        .expect("the named config is not in the manifest");
    assert!(
        !config.quantized,
        "the hold measurement runs on the raw configs"
    );

    let parsed = parse_dump::<f32>(&config.index_dir, &expected_for(&config)).unwrap();
    let held_vendored;
    let held_flat;
    match which.as_str() {
        "vendored" => {
            held_vendored = Some(
                Hnsw::from_loaded_points(
                    parsed.points_by_layer,
                    parsed.entry,
                    parsed.m,
                    parsed.ef_construction,
                    parsed.level_scale,
                    CosineDist {},
                )
                .unwrap(),
            );
            held_flat = None;
        }
        "flat" => {
            held_flat = Some(
                FlatGraph::from_loaded(
                    parsed.points_by_layer,
                    parsed.entry,
                    parsed.m,
                    parsed.ef_construction,
                    parsed.level_scale,
                    CosineDist {},
                )
                .unwrap(),
            );
            held_vendored = None;
        }
        "parse" => {
            drop(parsed);
            held_vendored = None;
            held_flat = None;
        }
        other => panic!("unknown hold {}", other),
    }

    println!(
        "HOLDING {} {} pid {} nodes {}",
        which,
        config.name,
        std::process::id(),
        held_vendored
            .as_ref()
            .map(|h| h.get_nb_point())
            .or_else(|| held_flat.as_ref().map(|f| f.nb_points()))
            .unwrap_or(0)
    );
    if let Some(flat) = held_flat.as_ref() {
        println!("flat accounted bytes {}", flat.memory_bytes());
    }
    std::io::stdout().flush().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(40));
    println!("HOLD DONE");
}
