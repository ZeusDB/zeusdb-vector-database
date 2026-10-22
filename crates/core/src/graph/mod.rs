//! The boundary between the index and the graph it is built on.
//!
//! Everything the index does to the graph goes through [`VectorGraph`], and
//! everything the graph hands back arrives as a [`GraphHit`], so the crate's
//! own types stop here rather than travelling into `hnsw_index.rs`.
//!
//! The on-disk format is [`dump`], which is ZeusDB's own. The vendored crate's
//! two file dump is no longer written or read.
//!
//! # The graph is ZeusDB's own
//!
//! Every variant holds a [`mutable::MutableGraph`] with a
//! [`levels::LevelGenerator`] beside it. The structure, the traversal, the
//! insert, the level stream, the distance trait and the dump are all in this
//! module, and nothing outside the crate is involved in any of them. The
//! vendored graph crate the first six releases were built on is gone, source
//! and dependency both.
//!
//! # What the seam does not cover
//!
//! One thing cannot be hidden, because ZeusDB implements it rather than calling
//! it. [`Distance`] is the trait `CosineDist`, `L1Dist`, `L2Dist`, `DotDist`,
//! `DistPQ` and `Int8Dist` implement, and a trait has to be nameable where the
//! implementation is written. It is declared here so `distance.rs` and
//! `hnsw_index.rs` import it from the seam and the set of implementors stays
//! countable from one line.
//!
//! # The distance types are no longer pinned to their modules
//!
//! `Hnsw::file_dump` wrote `std::any::type_name::<D>()` into the dump header
//! and the loader compared the saved string against the same call. `type_name`
//! is the full module path of the declaration, so moving `CosineDist`,
//! `L1Dist`, `L2Dist` or `DistPQ` to another module changed what a save wrote
//! and stopped every previously saved index from loading. That is why `DistPQ`
//! was declared in `hnsw_index` and the three raw distances in `distance.rs`,
//! even though the graph is what uses them.
//!
//! ZeusDB's header carries a [`dump::GraphKind`] discriminant instead, which is
//! a number ZeusDB chose rather than a fact about where a type is declared. The
//! four types can now be moved, renamed or replaced without a saved index
//! becoming unreadable. `DistPQ` has been, and now sits in `distance.rs` with
//! the other four, which is what removed this module's dependency on
//! `hnsw_index`.

// `levels: Mutex<LevelGenerator>` is not a field of the index, so it is
// outside the registry in `zeusdb_vector_hnsw::locks` and outside the order it
// enforces. Every path that draws a level holds the graph's own guard or the
// graph outright, so the generator is reached under one lock and never held
// across another. See `clippy.toml`.
#![allow(clippy::disallowed_types)]

use crate::distance::{
    CosineDist, DistPQ, DotDist, Int8Dist, Int8Metric, L1Dist, L2Dist, PqMetric,
};
use crate::int8::Int8Codec;
use crate::pq::PQ;
use crate::space::Position;
use crate::storage::Dir;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, trace, warn};

/// The target every record this module emits carries.
///
/// It is this module's path under `zeusdb_vector_database`, the package a
/// user configures logging by, rather than `module_path!()`, which would name
/// this crate. The filter directive the binding installs and a `RUST_LOG`
/// directive both match a target by prefix, so a record carrying this crate's
/// name would fall outside both and be dropped. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::graph";

pub(crate) mod dump;
pub(crate) mod store;

// The level generator, which draws a new point's top level.
#[cfg_attr(not(test), allow(dead_code))]
mod levels;
// The structure every shipped graph is, and the insert that builds it.
#[cfg_attr(not(test), allow(dead_code))]
mod mutable;
#[cfg(test)]
mod structure;
#[cfg(test)]
pub(crate) mod test_graph;
// A seeded mutator over valid dumps of both versions, driving the readers with
// generated input rather than with the damage cases written by hand.
#[cfg(test)]
mod fuzz;
// The graph and ADC guards, which build a graph directly rather than through
// an index.
#[cfg(test)]
mod guard_tests;
// The traversal, written once against an accessor.
#[cfg_attr(not(test), allow(dead_code))]
mod traverse;

pub use dump::{DumpRefusal, DumpRegion};
use dump::{Expected, GraphKind};
use levels::LevelGenerator;
use mutable::MutableGraph;
/// What phase one of an insertion hands to phase two. Opaque outside the
/// graph, and nameable so a caller can carry it between the two guards.
pub use mutable::Planned;
use store::VectorStore;
use traverse::LAYERS;

/// How far apart two points of type `T` are.
///
/// One method, and every distance in the crate is one expression. It is
/// declared here rather than in `distance.rs` because the graph is what calls
/// it and `distance.rs` is one of the modules that implements it.
///
/// The implementors are `CosineDist`, `L2Dist`, `L1Dist`, `DotDist`, `DistPQ`
/// and `Int8Dist`, all six in `distance.rs`. Every one imports the name from this
/// module, so the set is countable from the `use` sites of this line.
///
/// The shape is the one the trait it replaced had, unchanged, because changing
/// it would have been a second edit landing at the same time as the deletion
/// and there is nothing wrong with it.
pub trait Distance<T> {
    /// The distance from `va` to `vb`. Both slices are the graph's width.
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}

/// How far `sum(x * x)` may sit from one before a vector is not unit length.
///
/// The quantity is the squared norm rather than the norm, because that is what
/// the check computes and a square root would only add error to it.
///
/// The residual a correct normalisation leaves is the whole of what this has to
/// absorb. `Space::normalize_vector` divides by an `f32` norm accumulated in
/// `f32`, and the worst `|sum(x * x) - 1|` that leaves, measured over the real
/// 128, 768 and 1,536 dimensional sets and over adversarial input of four
/// thousand values spanning eight orders of magnitude, is 3.576e-7, which is
/// three `f32` steps at one. This is 1e-3, so it sits about 2,800 times above
/// the worst residual observed and about 20 times below what a vector whose
/// norm is wrong by one percent would produce. Nothing legitimate lands between
/// those.
///
/// Both uses sit inside `assert_unit_for_cosine`'s `cfg(debug_assertions)`
/// block, so a release build reads this constant nowhere and `dead_code` fires
/// on it. That warning was the only one a release build of this crate emitted.
/// The allow is conditioned the same way the uses are, so a debug build still
/// reports it if the assertion is ever deleted.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
const COSINE_UNIT_TOLERANCE: f32 = 1e-3;

/// Assert, in debug builds only, that a vector reaching a cosine graph is unit
/// length.
///
/// The check belongs here rather than in [`Distance::eval`] for two reasons.
/// The seam sees a vector once per insertion and once per search where the
/// kernel sees it several thousand times per search, so this costs one pass
/// over the vector where the kernel would cost one per evaluation. And
/// `rerank::rescore_candidate` hands `CosineDist` a PQ reconstruction, which is
/// deliberately not unit length, so a kernel level assertion could not be
/// written without a tolerance loose enough to admit an unnormalised vector.
///
/// A zero vector passes. `normalize_vector` returns it unchanged, since there
/// is nothing to divide by, and `CosineDist` scores it one against everything,
/// which the departure recorded on that type describes. It is a vector of no
/// direction rather than a violated precondition.
///
/// **This compiles to nothing in release.** The body is behind
/// `cfg(debug_assertions)` rather than inside `debug_assert!`, so the sum is
/// not merely unevaluated but absent.
#[inline]
fn assert_unit_for_cosine(_vector: &[f32], _site: &str) {
    #[cfg(debug_assertions)]
    {
        let squared: f32 = _vector.iter().map(|x| x * x).sum();
        assert!(
            squared == 0.0 || (squared - 1.0).abs() <= COSINE_UNIT_TOLERANCE,
            "a vector reaching the cosine graph at {} is not unit length: \
             sum(x * x) is {}, which is {} from one and past the {} tolerance. \
             CosineDist is 1 - dot and assumes normalisation, so this page or \
             this insertion would be scored on projection rather than on \
             direction. Every path into a cosine graph must normalise first, \
             which process_vector_for_space does.",
            _site,
            squared,
            (squared - 1.0).abs(),
            COSINE_UNIT_TOLERANCE
        );
    }
}

/// One result of a graph traversal, in ZeusDB's own terms.
///
/// The vendored `Neighbour` carries a third field, `p_id`, which locates the
/// point inside the layer structure. No ZeusDB caller reads it. Every search
/// path takes the record's position, resolves it through the id store, and
/// takes the distance as the score when no rerank is in play, so those two
/// fields are what the seam carries and nothing else.
#[derive(Debug, Clone, Copy)]
pub struct GraphHit {
    /// The position of the record the node holds, which the id store
    /// resolves.
    pub position: usize,
    /// Distance from the query on whatever scale the graph was built with.
    pub distance: f32,
}

/// One graph and the level generator that draws for it.
///
/// The two belong together. A graph's level stream is part of what decides its
/// edges, so a generator that outlived one graph and was handed to the next
/// would make the second graph a function of the first one's history. Every
/// path that replaces the graph replaces the pair.
///
/// The generator is behind a mutex because the draw happens under the index's
/// graph **read** guard, taken for the draw alone ahead of phase one. It is a
/// leaf: nothing inside it takes another lock, and no other lock is taken while
/// it is held. It is also uncontended, since the index's `writers` mutex admits
/// one mutator at a time.
pub(crate) struct Backend<T, D> {
    graph: MutableGraph<T, D>,
    /// The vectors this graph scores against, addressed by node index.
    ///
    /// `f32` on a raw graph and one `u8` per subvector on a quantized one. The
    /// graph itself holds links, levels and origin ids and no vectors at all.
    store: VectorStore<T>,
    /// The raw vectors a quantized graph keeps beside its codes, where the
    /// storage mode keeps them.
    ///
    /// `None` on every raw graph, where the store above already is the raw
    /// vectors, and on a `quantized_only` graph, which holds none. `Some` on a
    /// `quantized_with_raw` graph, addressed by the same node index the codes
    /// are, so rerank and `get_records` reach a raw vector with one
    /// multiplication and no hashing.
    raw: Option<VectorStore<f32>>,
    levels: Mutex<LevelGenerator>,
}

impl<T, D> Backend<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// An empty graph at the default scale for its degree, with the reservation
    /// `expected_size` asks for.
    ///
    /// The arguments are clamped rather than validated, because every caller has
    /// validated them already: `Declaration::validate` rejects a zero dimension, a
    /// zero `expected_size` and an `m` outside 2 to 256, and the loader's
    /// constructor is fed a directory this crate wrote. The clamp is what makes
    /// the `expect` below unreachable rather than merely unlikely, and it is
    /// what lets this constructor stay infallible for the call sites that have
    /// no error path to return one on.
    fn sized(
        dim: usize,
        m: usize,
        max_layer: usize,
        ef_construction: usize,
        expected_size: usize,
        dist_f: D,
    ) -> Self {
        let dim = dim.max(1);
        let m = m.clamp(2, 256);
        let expected_size = expected_size.max(1);
        let maxlevel = max_layer.clamp(1, LAYERS);
        let scale = LevelGenerator::default_scale(m);
        let (graph, store) =
            MutableGraph::new(dim, m, ef_construction, scale, expected_size, dist_f)
                .expect("the arguments were clamped into the range MutableGraph::new accepts");
        Backend {
            graph,
            store,
            raw: None,
            levels: Mutex::new(LevelGenerator::new(scale, maxlevel)),
        }
    }

    /// An empty graph like [`Backend::sized`] builds, at the default scale
    /// for its degree, that has asked the allocator for nothing.
    /// [`Backend::reserve_declared`] gives it the reservation `sized` takes.
    fn unreserved(
        dim: usize,
        m: usize,
        max_layer: usize,
        ef_construction: usize,
        dist_f: D,
    ) -> Self {
        let dim = dim.max(1);
        let m = m.clamp(2, 256);
        let maxlevel = max_layer.clamp(1, LAYERS);
        let scale = LevelGenerator::default_scale(m);
        let (graph, store) = MutableGraph::unreserved(dim, m, ef_construction, scale, dist_f)
            .expect("the arguments were clamped into the range MutableGraph::unreserved accepts");
        Backend {
            graph,
            store,
            raw: None,
            levels: Mutex::new(LevelGenerator::new(scale, maxlevel)),
        }
    }

    /// Take the reservation [`Backend::sized`] takes for `expected_size`, on
    /// a graph that holds no node.
    fn reserve_declared(&mut self, expected_size: usize) {
        self.graph
            .reserve_declared(&mut self.store, expected_size.max(1));
    }

    /// A graph read back from a dump, with a generator at the scale the dump
    /// recorded.
    ///
    /// The generator is seeded at `DEFAULT_LEVEL_SEED`, so a loaded index that
    /// keeps inserting starts the level stream over. That is what
    /// `Hnsw::from_loaded_points` does through `new_with_absolute_scale`, so it
    /// is the behaviour this restores rather than a new one.
    fn restored(loaded: (MutableGraph<T, D>, VectorStore<T>)) -> Self {
        let (graph, store) = loaded;
        let scale = graph.level_scale();
        Backend {
            graph,
            store,
            raw: None,
            levels: Mutex::new(LevelGenerator::new(scale, LAYERS)),
        }
    }

    /// Draw the level the next node takes, advancing the stream by one draw.
    ///
    /// Separate from phase one so that a caller holds the level before the
    /// insertion begins and can record it, and so that a caller replaying a
    /// recorded level can plan at it without drawing. The stream advances
    /// once per insertion either way, in insertion order, which is what makes
    /// two builds of the same records the same graph.
    fn draw_level(&self) -> usize {
        self.levels
            .lock()
            .expect("the level generator is a leaf and no path panics holding it")
            .generate()
    }

    /// Phase one at `level`. Decides the insertion, reading only and drawing
    /// nothing.
    fn plan_at_level(&self, data: &[T], level: usize) -> Planned {
        self.graph.plan_insertion(&self.store, data, level)
    }

    /// Phase two. Writes what phase one decided.
    fn install(&mut self, data: &[T], origin_id: usize, position: u32, planned: Planned) {
        self.graph
            .install_insertion(&mut self.store, data, origin_id, position, planned);
    }

    /// Append one raw vector to the side store a `quantized_with_raw` graph
    /// keeps, so it stays in step with the node just installed.
    ///
    /// A caller that hands nothing where the store exists, or something where
    /// it does not, has a record the mode does not describe. Both are treated
    /// the same way the rest of the seam treats an element type mismatch, being
    /// a logged refusal rather than a panic, and the invariant that follows is
    /// asserted in debug builds.
    fn push_raw(&mut self, raw: Option<&[f32]>) {
        match (self.raw.as_mut(), raw) {
            (Some(store), Some(values)) => store.push(values),
            (None, None) => {}
            (Some(_), None) => error!(
                target: LOG_TARGET,
                operation = "vector_insert",
                error = "invalid_operation",
                reason = "quantized_with_raw_insert_carried_no_raw_vector",
                "A quantized graph keeping raw vectors was handed a record without one"
            ),
            (None, Some(_)) => error!(
                target: LOG_TARGET,
                operation = "vector_insert",
                error = "invalid_operation",
                reason = "graph_keeps_no_raw_vectors",
                "A raw vector was offered to a graph that keeps none beside its codes"
            ),
        }
        debug_assert!(
            self.raw
                .as_ref()
                .is_none_or(|store| store.len() == self.graph.nb_points()),
            "the raw side store is addressed by node index, so it holds one vector per node"
        );
    }

    /// The draw and both phases, for a caller holding the graph outright.
    fn insert(&mut self, data: &[T], origin_id: usize, position: u32) {
        let level = self.draw_level();
        let planned = self.plan_at_level(data, level);
        self.install(data, origin_id, position, planned);
    }
}

/// The graph, in whichever of the eight shapes this index built it
///
/// Four raw variants holding `f32` points, three product quantized variants
/// holding `u8` codes and one scalar quantized variant holding `i8` codes.
/// They differ in the distance they were built with, which the
/// graph takes as a type parameter, so the enum is what stands in for a single
/// graph type.
// The variants carry `Backend`, which is `pub(crate)`. A caller outside the
// crate can bind one by matching a variant and can do nothing with it, since
// every method on it is crate-private, so the type stays crate-private rather
// than widening for an interface nothing outside uses. The binding never
// matches a variant; it goes through the methods below.
#[allow(private_interfaces)]
pub enum VectorGraph {
    // Raw vector variants
    Cosine(Backend<f32, CosineDist>),
    L2(Backend<f32, L2Dist>),
    L1(Backend<f32, L1Dist>),
    Dot(Backend<f32, DotDist>),

    // PQ variants, holding one `u8` per subvector
    CosinePQ(Backend<u8, DistPQ>),
    L2PQ(Backend<u8, DistPQ>),
    L1PQ(Backend<u8, DistPQ>),

    /// The scalar quantized graph, holding one `i8` a value.
    ///
    /// One variant for all four spaces rather than one per space, because
    /// the metric travels inside [`Int8Dist`] and nothing outside it has to
    /// match on the space: the dump header takes its kind from the metric
    /// through [`VectorGraph::kind`] and the loader hands the metric back
    /// through [`restore_int8_graph`]. The product quantized graphs carry
    /// one variant per space because their first dump format pinned each to
    /// a type name, which this one never did.
    ///
    /// It keeps no raw side store. A record is one code and nothing else,
    /// which is what makes the memory case: the store holds a quarter of
    /// what a raw store holds and there is no second store beside it.
    Int8(Backend<i8, Int8Dist>),
}

/// One record on its way into the graph, in whichever form that graph holds.
///
/// The six variants split two ways at once, by distance and by element type,
/// and only the second decides what an insertion carries. Naming the payload
/// rather than writing one method per element type means the mismatch between a
/// raw record and a quantized graph is stated once per phase instead of four
/// times.
#[derive(Clone, Copy)]
pub enum Record<'a> {
    /// A raw vector, already normalized for the index space.
    Raw(&'a [f32]),
    /// The quantized codes of one record, one byte per subvector, and the raw
    /// vector where the storage mode keeps one.
    ///
    /// The raw travels with the codes because the node the codes are installed
    /// at is the node the raw has to sit at, and that node is decided by the
    /// install. Passing it separately would mean a second call that could be
    /// skipped, and a skipped call is a raw store one short of its graph.
    Codes {
        codes: &'a [u8],
        raw: Option<&'a [f32]>,
    },
    /// The scalar codes of one record, one signed byte a value, already
    /// encoded through the graph's own codec.
    Int8(&'a [i8]),
}

/// The stride the timing kernels take across the store, a prime large
/// enough that successive records share no cache line and no page, and the
/// offset each round starts from, another prime, so a round touches records
/// the last one did not.
const SCATTER_STRIDE: usize = 7919;
const SCATTER_OFFSET: usize = 104_729;

impl VectorGraph {
    pub fn new_raw(
        space: &str,
        dim: usize,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        Self::raw(
            space,
            dim,
            m,
            expected_size,
            max_layer,
            ef_construction,
            true,
        )
    }

    /// A raw graph built and logged as [`VectorGraph::new_raw`] builds and
    /// logs one, holding no reservation until [`VectorGraph::reserve_declared`]
    /// gives it one.
    ///
    /// The loader builds the collection it fills around one, since a load
    /// replaces the graph whenever the directory holds a record.
    pub fn new_raw_unreserved(
        space: &str,
        dim: usize,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        Self::raw(
            space,
            dim,
            m,
            expected_size,
            max_layer,
            ef_construction,
            false,
        )
    }

    /// The raw graph for `space`, reserved for `expected_size` where `reserve`
    /// is set.
    fn raw(
        space: &str,
        dim: usize,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
        reserve: bool,
    ) -> Self {
        info!(
            target: LOG_TARGET,
            operation = "hnsw_creation",
            space = space,
            dim = dim,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "raw",
            "Creating raw HNSW index"
        );

        macro_rules! raw {
            ($variant:ident, $dist:expr) => {
                VectorGraph::$variant(if reserve {
                    Backend::sized(dim, m, max_layer, ef_construction, expected_size, $dist)
                } else {
                    Backend::unreserved(dim, m, max_layer, ef_construction, $dist)
                })
            };
        }
        match space {
            "cosine" => raw!(Cosine, CosineDist {}),
            "l2" => raw!(L2, L2Dist {}),
            "l1" => raw!(L1, L1Dist {}),
            "dot" => raw!(Dot, DotDist {}),
            _ => {
                error!(
                    target: LOG_TARGET,
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided"
                );
                // This is a programming error that should be caught earlier.
                // Defaulting to cosine rather than panicking, which is what the
                // load path also does, so the two construction paths agree on
                // what an unrecognised space means.
                warn!(
                    target: LOG_TARGET,
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                raw!(Cosine, CosineDist {})
            }
        }
    }

    pub fn new_pq(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
        pq: Arc<PQ>,
    ) -> Self {
        // A quantized graph holds one byte per subvector, so the codebook is
        // what states its width rather than the index's declared dimension.
        let dim = pq.subvectors();
        info!(
            target: LOG_TARGET,
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "quantized",
            subvectors = pq.subvectors(),
            bits = pq.bits(),
            "Creating PQ-enabled HNSW index"
        );

        // The metric travels with the variant, because the two are the same
        // choice. A quantized graph orders and scores by what its space
        // declared, and `DistPQ` is told which rather than inferring it.
        macro_rules! quantized {
            ($variant:ident, $metric:expr) => {
                VectorGraph::$variant(Backend::sized(
                    dim,
                    m,
                    max_layer,
                    ef_construction,
                    expected_size,
                    DistPQ::new(pq, $metric),
                ))
            };
        }
        match space {
            "cosine" => quantized!(CosinePQ, PqMetric::Cosine),
            "l2" => quantized!(L2PQ, PqMetric::SquaredL2),
            // Unreachable. `validate_space_supports_quantization` refuses the
            // pair at create() and at load, and records the L1 tables and the
            // k-medians codebook that were measured before keeping it refused.
            // Squared L2 is what it was built with while the pair was served.
            "l1" => quantized!(L1PQ, PqMetric::SquaredL2),
            // "dot" is not here and cannot reach here. The same function
            // refuses that pair at both doors and records the measurement
            // behind the refusal.
            _ => {
                error!(
                    target: LOG_TARGET,
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided for PQ"
                );
                warn!(
                    target: LOG_TARGET,
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                quantized!(CosinePQ, PqMetric::Cosine)
            }
        }
    }

    /// A scalar quantized graph over `codec`, for `space`.
    ///
    /// The graph holds one row a record, being one `i8` a value and, on a
    /// cosine graph, four bytes of the record's inverse norm after them, so
    /// the store's width is [`Int8Dist::row_width`] rather than the index's
    /// declared dimension. The metric is read off the space name the way the
    /// two other constructors read it, and an unrecognised name falls back
    /// to cosine as they do.
    pub fn new_int8(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
        codec: Arc<Int8Codec>,
    ) -> Self {
        let metric = Int8Metric::from_space(space);
        let dist = Int8Dist::new(codec, metric);
        let dim = dist.row_width();
        info!(
            target: LOG_TARGET,
            operation = "hnsw_creation",
            space = space,
            dim = dim,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "int8",
            metric = metric.space(),
            "Creating scalar quantized HNSW index"
        );
        VectorGraph::Int8(Backend::sized(
            dim,
            m,
            max_layer,
            ef_construction,
            expected_size,
            dist,
        ))
    }

    /// Search the graph, admitting only the internal ids the filter accepts.
    ///
    /// The filter runs inside the traversal, before the fixed `top_k` cut, so a
    /// node the caller rejects routes the search but never consumes a result
    /// slot. Removal and overwrite both leave a node behind that no longer
    /// resolves to a record, and without the filter each such node inside a
    /// query's `top_k` costs one result. Passing `None` restores the previous
    /// unfiltered behaviour.
    ///
    /// It is taken by generic reference rather than as a trait object so the
    /// closure the caller passes is monomorphised into the traversal.
    pub fn search<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&F>,
    ) -> Result<Vec<GraphHit>, String>
    where
        F: Fn(&usize) -> bool,
    {
        match self {
            // Raw vector search
            VectorGraph::Cosine(b) => {
                assert_unit_for_cosine(query, "search");
                Ok(b.graph.search(&b.store, query, k, ef, filter))
            }
            VectorGraph::L2(b) => Ok(b.graph.search(&b.store, query, k, ef, filter)),
            VectorGraph::L1(b) => Ok(b.graph.search(&b.store, query, k, ef, filter)),
            VectorGraph::Dot(b) => Ok(b.graph.search(&b.store, query, k, ef, filter)),

            // PQ-based search with ADC
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                // This query's ADC table, installed on this thread alone. The
                // guard is named so it lives to the end of the arm rather than
                // dropping at the end of the statement, and it releases the
                // table once the traversal is done.
                let _query_lut = b.graph.distance().install_query_lut(query)?;

                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; b.graph.distance().subvectors()];

                Ok(b.graph.search(&b.store, &dummy_query, k, ef, filter))
            }

            // Scalar codes. The query is installed on this thread at full
            // width and the traversal is handed a dummy code, exactly as the
            // product quantized search is.
            VectorGraph::Int8(b) => {
                if b.graph.distance().metric() == Int8Metric::Cosine {
                    assert_unit_for_cosine(query, "search");
                }
                let _query = b.graph.distance().install_query(query)?;
                let dummy_query = vec![0i8; b.graph.distance().dim()];
                Ok(b.graph.search(&b.store, &dummy_query, k, ef, filter))
            }
        }
    }

    /// Number of nodes the graph holds, which is the number of insertions it has
    /// taken. It exceeds the live record count by exactly the number of nodes
    /// that removal and overwrite have stranded.
    pub fn nb_points(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.nb_points(),
            VectorGraph::L2(b) => b.graph.nb_points(),
            VectorGraph::L1(b) => b.graph.nb_points(),
            VectorGraph::Dot(b) => b.graph.nb_points(),
            VectorGraph::CosinePQ(b) => b.graph.nb_points(),
            VectorGraph::L2PQ(b) => b.graph.nb_points(),
            VectorGraph::L1PQ(b) => b.graph.nb_points(),
            VectorGraph::Int8(b) => b.graph.nb_points(),
        }
    }

    /// Bytes the graph asks the allocator for.
    ///
    /// Exact rather than sampled. Every buffer the structure holds has a known
    /// capacity and there is no per node or per edge allocation to estimate, so
    /// this is arithmetic over the capacities rather than a measurement over a
    /// sample of the points. The figure it replaced had to sample, because the
    /// vendored structure allocated six blocks per point and the adjacency count
    /// is a property of the data rather than of `m`. What the figure now covers
    /// is on `MutableGraph::memory_bytes`.
    ///
    /// It is a request count and not a commitment. The allocator's headers, its
    /// rounding and its fragmentation sit outside it, and a process holding this
    /// graph commits more than this names.
    /// Bytes the adjacency, the levels, the origin ids and their inverse ask
    /// for, with no vector counted.
    pub fn links_memory_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.memory_bytes(),
            VectorGraph::L2(b) => b.graph.memory_bytes(),
            VectorGraph::L1(b) => b.graph.memory_bytes(),
            VectorGraph::Dot(b) => b.graph.memory_bytes(),
            VectorGraph::CosinePQ(b) => b.graph.memory_bytes(),
            VectorGraph::L2PQ(b) => b.graph.memory_bytes(),
            VectorGraph::L1PQ(b) => b.graph.memory_bytes(),
            VectorGraph::Int8(b) => b.graph.memory_bytes(),
        }
    }

    /// Bytes the store the graph scores against asks for, being the raw vectors
    /// on a raw graph and the codes on a quantized one.
    pub fn store_memory_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.store.memory_bytes(),
            VectorGraph::L2(b) => b.store.memory_bytes(),
            VectorGraph::L1(b) => b.store.memory_bytes(),
            VectorGraph::Dot(b) => b.store.memory_bytes(),
            VectorGraph::CosinePQ(b) => b.store.memory_bytes(),
            VectorGraph::L2PQ(b) => b.store.memory_bytes(),
            VectorGraph::L1PQ(b) => b.store.memory_bytes(),
            VectorGraph::Int8(b) => b.store.memory_bytes(),
        }
    }

    /// Bytes the raw side store asks for, which is zero unless this is a
    /// `quantized_with_raw` graph.
    pub(crate) fn raw_memory_bytes(&self) -> usize {
        self.raw_store().map_or(0, VectorStore::memory_bytes)
    }

    /// Bytes the store holding this graph's raw vectors asks for, wherever it
    /// sits, and zero on a graph that keeps none.
    ///
    /// A raw graph scores against the vectors themselves, so the store the
    /// graph was handed is the raw store. A quantized graph scores against
    /// codes, and keeps the vectors in a side store or not at all. One name for
    /// the same quantity, so a caller pricing the raw vectors does not have to
    /// know which kind of graph it is holding.
    ///
    /// It is the capacity and not the payload. A store filled by insertion
    /// carries the slack of its last doubling, and that slack is real memory
    /// the process asked for; see [`VectorGraph::raw_vectors_reserved_bytes`].
    pub fn raw_vectors_memory_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.store.memory_bytes(),
            VectorGraph::L2(b) => b.store.memory_bytes(),
            VectorGraph::L1(b) => b.store.memory_bytes(),
            VectorGraph::Dot(b) => b.store.memory_bytes(),
            VectorGraph::CosinePQ(_) | VectorGraph::L2PQ(_) | VectorGraph::L1PQ(_) => {
                self.raw_memory_bytes()
            }
            VectorGraph::Int8(_) => 0,
        }
    }

    /// Bytes of the adjacency's request that no node has been written into.
    ///
    /// See `MutableGraph::reserved_bytes`. This is the links alone, on the same
    /// division `links_memory_bytes` makes.
    pub fn links_reserved_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.reserved_bytes(),
            VectorGraph::L2(b) => b.graph.reserved_bytes(),
            VectorGraph::L1(b) => b.graph.reserved_bytes(),
            VectorGraph::Dot(b) => b.graph.reserved_bytes(),
            VectorGraph::CosinePQ(b) => b.graph.reserved_bytes(),
            VectorGraph::L2PQ(b) => b.graph.reserved_bytes(),
            VectorGraph::L1PQ(b) => b.graph.reserved_bytes(),
            VectorGraph::Int8(b) => b.graph.reserved_bytes(),
        }
    }

    /// Bytes of the scored store's request that no vector has been written
    /// into, on the same division `store_memory_bytes` makes.
    pub fn store_reserved_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.store.reserved_bytes(),
            VectorGraph::L2(b) => b.store.reserved_bytes(),
            VectorGraph::L1(b) => b.store.reserved_bytes(),
            VectorGraph::Dot(b) => b.store.reserved_bytes(),
            VectorGraph::CosinePQ(b) => b.store.reserved_bytes(),
            VectorGraph::L2PQ(b) => b.store.reserved_bytes(),
            VectorGraph::L1PQ(b) => b.store.reserved_bytes(),
            VectorGraph::Int8(b) => b.store.reserved_bytes(),
        }
    }

    /// Bytes of the raw vector store's request that no vector has been written
    /// into, on the same division `raw_vectors_memory_bytes` makes.
    pub fn raw_vectors_reserved_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.store.reserved_bytes(),
            VectorGraph::L2(b) => b.store.reserved_bytes(),
            VectorGraph::L1(b) => b.store.reserved_bytes(),
            VectorGraph::Dot(b) => b.store.reserved_bytes(),
            VectorGraph::CosinePQ(_) | VectorGraph::L2PQ(_) | VectorGraph::L1PQ(_) => {
                self.raw_store().map_or(0, VectorStore::reserved_bytes)
            }
            VectorGraph::Int8(_) => 0,
        }
    }

    /// The raw side store, where this graph keeps one.
    fn raw_store(&self) -> Option<&VectorStore<f32>> {
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_)
            | VectorGraph::Int8(_) => None,
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw.as_ref()
            }
        }
    }

    /// Every node's position, indexed by node, being the place of the record
    /// the node holds, or [`Position::NONE`] for a stranded node.
    pub fn records(&self) -> &[u32] {
        match self {
            VectorGraph::Cosine(b) => b.graph.records(),
            VectorGraph::L2(b) => b.graph.records(),
            VectorGraph::L1(b) => b.graph.records(),
            VectorGraph::Dot(b) => b.graph.records(),
            VectorGraph::CosinePQ(b) => b.graph.records(),
            VectorGraph::L2PQ(b) => b.graph.records(),
            VectorGraph::L1PQ(b) => b.graph.records(),
            VectorGraph::Int8(b) => b.graph.records(),
        }
    }

    /// Nodes the graph's positions by node have room for, which is the
    /// reservation a declared size took, so the index can reserve its own
    /// vector by position to match.
    pub fn record_capacity(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.record_capacity(),
            VectorGraph::L2(b) => b.graph.record_capacity(),
            VectorGraph::L1(b) => b.graph.record_capacity(),
            VectorGraph::Dot(b) => b.graph.record_capacity(),
            VectorGraph::CosinePQ(b) => b.graph.record_capacity(),
            VectorGraph::L2PQ(b) => b.graph.record_capacity(),
            VectorGraph::L1PQ(b) => b.graph.record_capacity(),
            VectorGraph::Int8(b) => b.graph.record_capacity(),
        }
    }

    /// The position of the record one node holds, or [`Position::NONE`]
    /// where the node is stranded.
    pub fn record_of_at(&self, node: u32) -> u32 {
        self.records()[node as usize]
    }

    /// Every node's position to write, indexed by node, for a pass that
    /// files the records a load holds.
    pub fn records_mut(&mut self) -> &mut [u32] {
        match self {
            VectorGraph::Cosine(b) => b.graph.records_mut(),
            VectorGraph::L2(b) => b.graph.records_mut(),
            VectorGraph::L1(b) => b.graph.records_mut(),
            VectorGraph::Dot(b) => b.graph.records_mut(),
            VectorGraph::CosinePQ(b) => b.graph.records_mut(),
            VectorGraph::L2PQ(b) => b.graph.records_mut(),
            VectorGraph::L1PQ(b) => b.graph.records_mut(),
            VectorGraph::Int8(b) => b.graph.records_mut(),
        }
    }

    /// Trade every node's position for `records`, one a node, which a
    /// close-up built off to the side while the graph kept its nodes, so the
    /// graph files each node's record where `records` does and `records`
    /// holds what the graph held. A move alone.
    pub fn swap_records(&mut self, records: &mut Vec<u32>) {
        match self {
            VectorGraph::Cosine(b) => b.graph.swap_records(records),
            VectorGraph::L2(b) => b.graph.swap_records(records),
            VectorGraph::L1(b) => b.graph.swap_records(records),
            VectorGraph::Dot(b) => b.graph.swap_records(records),
            VectorGraph::CosinePQ(b) => b.graph.swap_records(records),
            VectorGraph::L2PQ(b) => b.graph.swap_records(records),
            VectorGraph::L1PQ(b) => b.graph.swap_records(records),
            VectorGraph::Int8(b) => b.graph.swap_records(records),
        }
    }

    /// Nodes the reservation for `expected_size` covers, being what a graph
    /// of this kind, width and degree declared at that size reserves for
    /// each per-node array, under the byte budget its arenas are held to.
    pub fn declared_records(&self, expected_size: usize) -> usize {
        let expected_size = expected_size.max(1);
        match self {
            VectorGraph::Cosine(b) => b.graph.declared_records(expected_size),
            VectorGraph::L2(b) => b.graph.declared_records(expected_size),
            VectorGraph::L1(b) => b.graph.declared_records(expected_size),
            VectorGraph::Dot(b) => b.graph.declared_records(expected_size),
            VectorGraph::CosinePQ(b) => b.graph.declared_records(expected_size),
            VectorGraph::L2PQ(b) => b.graph.declared_records(expected_size),
            VectorGraph::L1PQ(b) => b.graph.declared_records(expected_size),
            VectorGraph::Int8(b) => b.graph.declared_records(expected_size),
        }
    }

    /// Every node's internal id, indexed by node, for a pass over many
    /// nodes that reads each with one plain read.
    pub fn origin_ids(&self) -> &[usize] {
        match self {
            VectorGraph::Cosine(b) => b.graph.origin_ids(),
            VectorGraph::L2(b) => b.graph.origin_ids(),
            VectorGraph::L1(b) => b.graph.origin_ids(),
            VectorGraph::Dot(b) => b.graph.origin_ids(),
            VectorGraph::CosinePQ(b) => b.graph.origin_ids(),
            VectorGraph::L2PQ(b) => b.graph.origin_ids(),
            VectorGraph::L1PQ(b) => b.graph.origin_ids(),
            VectorGraph::Int8(b) => b.graph.origin_ids(),
        }
    }

    /// The internal id one node was inserted under.
    pub fn origin_id_at(&self, node: u32) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.origin_id_of(node),
            VectorGraph::L2(b) => b.graph.origin_id_of(node),
            VectorGraph::L1(b) => b.graph.origin_id_of(node),
            VectorGraph::Dot(b) => b.graph.origin_id_of(node),
            VectorGraph::CosinePQ(b) => b.graph.origin_id_of(node),
            VectorGraph::L2PQ(b) => b.graph.origin_id_of(node),
            VectorGraph::L1PQ(b) => b.graph.origin_id_of(node),
            VectorGraph::Int8(b) => b.graph.origin_id_of(node),
        }
    }

    /// One node's raw vector.
    ///
    /// One array read, the store addressed by node, and no hashing at all. The
    /// dense index turns a record's position into its node first. `None` where
    /// the index keeps no raw vector for that node, which is every node of a
    /// trained `quantized_only` index, and for a node this graph does not hold.
    pub fn raw_vector_at(&self, node: u32) -> Option<&[f32]> {
        match self {
            VectorGraph::Cosine(b) => b.store.try_get(node),
            VectorGraph::L2(b) => b.store.try_get(node),
            VectorGraph::L1(b) => b.store.try_get(node),
            VectorGraph::Dot(b) => b.store.try_get(node),
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw.as_ref()?.try_get(node)
            }
            VectorGraph::Int8(_) => None,
        }
    }

    /// One node's quantized codes.
    ///
    /// The store a quantized graph scores against holds one code per node, so
    /// this is the same read `raw_vector_at` makes on a raw graph. `None` on a
    /// raw graph, which holds no codes, and for a node this graph does not
    /// hold.
    pub fn codes_at(&self, node: u32) -> Option<&[u8]> {
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_)
            | VectorGraph::Int8(_) => None,
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.store.try_get(node)
            }
        }
    }

    /// One node's stored row, being its scalar codes and, on a cosine graph,
    /// the four bytes of its inverse norm after them. What a rebuild re-inserts
    /// without re-encoding.
    ///
    /// The same read `codes_at` makes on a product quantized graph. `None` on
    /// every other graph, and for a node this graph does not hold.
    pub fn int8_row_at(&self, node: u32) -> Option<&[i8]> {
        match self {
            VectorGraph::Int8(b) => b.store.try_get(node),
            _ => None,
        }
    }

    /// One node's scalar codes alone, one a value. `None` on every other
    /// graph, and for a node this graph does not hold.
    pub fn int8_codes_at(&self, node: u32) -> Option<&[i8]> {
        match self {
            VectorGraph::Int8(b) => {
                let row = b.store.try_get(node)?;
                Some(b.graph.distance().codes_of_row(row))
            }
            _ => None,
        }
    }

    /// Encode one vector as the row this graph stores, through its own
    /// codec and metric. Refused on every other graph.
    pub fn int8_encode(&self, vector: &[f32]) -> Result<Vec<i8>, String> {
        match self {
            VectorGraph::Int8(b) => b.graph.distance().encode(vector),
            _ => Err("this graph holds no scalar codes".to_string()),
        }
    }

    /// The codec a scalar quantized graph decodes through, and `None` on
    /// every other graph.
    pub fn int8_codec(&self) -> Option<&Arc<Int8Codec>> {
        match self {
            VectorGraph::Int8(b) => Some(b.graph.distance().codec()),
            _ => None,
        }
    }

    /// Bytes one stored row holds on a scalar quantized graph, being the
    /// codec's width plus the norm tail on a cosine graph, and `None` on
    /// every other graph. The stride of the rows artefact.
    pub fn int8_row_width(&self) -> Option<usize> {
        match self {
            VectorGraph::Int8(b) => Some(b.graph.distance().row_width()),
            _ => None,
        }
    }

    /// The distance from `query` to one node's row on the metric this graph
    /// declared, being the number a traversal scores the node with, so an
    /// exact scan over scalar rows scores what a traversal scores bit for
    /// bit. `None` on every other graph, and for a node this graph does not
    /// hold.
    pub fn int8_distance_at(&self, query: &[f32], node: u32) -> Option<f32> {
        match self {
            VectorGraph::Int8(b) => {
                let row = b.store.try_get(node)?;
                Some(b.graph.distance().query_distance(query, row))
            }
            _ => None,
        }
    }

    /// The vector one node's row stands for, which on a cosine graph is the
    /// decoded vector at unit length, being the vector the graph scores as;
    /// see [`Int8Dist::reconstruct`]. `None` on every other graph, and for a
    /// node this graph does not hold.
    pub fn int8_reconstruct_at(&self, node: u32) -> Option<Vec<f32>> {
        match self {
            VectorGraph::Int8(b) => {
                let row = b.store.try_get(node)?;
                b.graph.distance().reconstruct(row).ok()
            }
            _ => None,
        }
    }

    /// Time one distance evaluation on this graph's own store, in
    /// nanoseconds, as the median of `rounds` rounds of `evaluations`
    /// evaluations between stored points.
    ///
    /// What a search costs is a number of these, so an index prices a
    /// search by multiplying a count by this figure. It is measured rather
    /// than tabulated because it moves with the machine and the build, and
    /// it is never persisted for the same reason. `None` where the store
    /// holds fewer than two points, in which case the caller falls back to
    /// its compiled-in floor.
    ///
    /// A quantized graph evaluates against a table computed for one query of
    /// zeros, which costs the same table reads a real query does.
    pub fn time_distance_ns(
        &self,
        evaluations: usize,
        rounds: usize,
        admits: &dyn Fn(usize) -> bool,
    ) -> Option<f64> {
        // One query against records scattered across the store, each first
        // tested by the predicate, which is what an exact scan does per
        // admitted record. The records are taken at a stride of a large
        // prime from an offset that moves with the round, so a round touches
        // records the last one did not and the figure is what the kernel
        // costs over memory rather than over a line the last evaluation
        // left in cache. Measured at width 1,536 over 50,000 points, a
        // sequential walk priced the exact scan at 1.8 to 2.4 times its
        // measured cost between 250 and 1,000 admitted and this walk at 1.2
        // to 1.8; at width 100 the two agree within 30 percent. Timed against
        // one of the graph's own points as the query, where a query vector
        // is a stored point's twin in every way the kernel can see.
        fn time<T, D: Distance<T>>(
            dist: &D,
            store: &VectorStore<T>,
            evaluations: usize,
            rounds: usize,
            admits: &dyn Fn(usize) -> bool,
        ) -> Option<f64> {
            let n = store.len();
            if n < 2 || evaluations == 0 || rounds == 0 {
                return None;
            }
            let query = store.get(1);
            let mut samples = Vec::with_capacity(rounds);
            for round in 0..rounds {
                let mut acc = 0f32;
                let started = std::time::Instant::now();
                for k in 0..evaluations {
                    let i = k
                        .wrapping_mul(SCATTER_STRIDE)
                        .wrapping_add(round.wrapping_mul(SCATTER_OFFSET))
                        % n;
                    if admits(i) {
                        acc += dist.eval(query, store.get(i as u32));
                    }
                }
                let elapsed = started.elapsed().as_secs_f64();
                std::hint::black_box(acc);
                samples.push(elapsed * 1e9 / evaluations as f64);
            }
            samples.sort_by(|a, b| a.total_cmp(b));
            Some(samples[samples.len() / 2])
        }
        match self {
            VectorGraph::Cosine(b) => {
                time(b.graph.distance(), &b.store, evaluations, rounds, admits)
            }
            VectorGraph::L2(b) => time(b.graph.distance(), &b.store, evaluations, rounds, admits),
            VectorGraph::L1(b) => time(b.graph.distance(), &b.store, evaluations, rounds, admits),
            VectorGraph::Dot(b) => time(b.graph.distance(), &b.store, evaluations, rounds, admits),
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                let query = vec![0f32; b.graph.distance().dim()];
                let _query_lut = b.graph.distance().install_query_lut(&query).ok()?;
                time(b.graph.distance(), &b.store, evaluations, rounds, admits)
            }
            // The query is the decoded twin of the point the raw timing takes
            // as its query, installed on this thread, so the kernel timed is
            // the one a search runs, being query against decoded code.
            VectorGraph::Int8(b) => {
                let query = b.graph.distance().reconstruct(b.store.try_get(1)?).ok()?;
                let _query = b.graph.distance().install_query(&query).ok()?;
                time(b.graph.distance(), &b.store, evaluations, rounds, admits)
            }
        }
    }

    /// Time one whole search on this graph, in nanoseconds, as the median of
    /// `rounds` searches at `k` and `ef` under `admits`, each asked with a
    /// query built from two of the graph's own points. `admits` is asked what
    /// a search's predicate is asked, being the position each candidate's
    /// node holds its record at.
    ///
    /// What a traversal costs is not what a distance evaluation costs
    /// multiplied by a count. A traversal's time is the memory it touches,
    /// being the neighbour lists and the vectors of nodes scattered across
    /// the store, where a kernel timed in a loop runs from cache. Measured
    /// at width 100 over 50,000 points, a search at `ef` 200 took as long as
    /// eleven thousand kernel evaluations and visited about a thousand
    /// nodes. So an index prices a traversal from this figure and a scan
    /// from [`VectorGraph::time_distance_ns`]. `None` where the graph holds
    /// fewer than two points.
    ///
    /// The query is the midpoint of two points scattered across the store,
    /// normalised on a cosine graph, rather than a stored point itself, so
    /// the search is for a vector the graph does not hold, which is what a
    /// query is. The predicate is in place because a filtered search and a
    /// search over a graph holding a removed record's node carry one, and
    /// the test is paid once per candidate. Measured at width 100 over 50,000
    /// points, this figure sits within 3 percent of a search the collection
    /// runs on the same graph; a stored point as the query, with no
    /// predicate, sat within 10 percent of it, so the two changes are for
    /// the search's shape rather than for a gap they close. On a quantized
    /// graph the two points are reconstructed from their codes, so the
    /// table the query installs is one a real query would.
    pub fn time_search_ns(
        &self,
        k: usize,
        ef: usize,
        rounds: usize,
        admits: &dyn Fn(usize) -> bool,
    ) -> Option<f64> {
        let n = self.nb_points();
        if n < 2 || rounds == 0 {
            return None;
        }
        let point = |node: u32| -> Option<Vec<f32>> {
            Some(match self {
                VectorGraph::Cosine(b) => b.store.get(node).to_vec(),
                VectorGraph::L2(b) => b.store.get(node).to_vec(),
                VectorGraph::L1(b) => b.store.get(node).to_vec(),
                VectorGraph::Dot(b) => b.store.get(node).to_vec(),
                VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                    b.graph.distance().reconstruct(b.store.get(node)).ok()?
                }
                VectorGraph::Int8(b) => b.graph.distance().reconstruct(b.store.get(node)).ok()?,
            })
        };
        let cosine = match self {
            VectorGraph::Cosine(_) | VectorGraph::CosinePQ(_) => true,
            VectorGraph::Int8(b) => b.graph.distance().metric() == Int8Metric::Cosine,
            _ => false,
        };
        let mut samples = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let first = (round.wrapping_mul(SCATTER_STRIDE).wrapping_add(1) % n) as u32;
            let second = (round
                .wrapping_mul(SCATTER_STRIDE)
                .wrapping_add(SCATTER_OFFSET)
                % n) as u32;
            let a = point(first)?;
            let b = point(second)?;
            let mut query: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
            if cosine {
                let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    query.iter_mut().for_each(|v| *v /= norm);
                }
            }
            let predicate = |at: &usize| admits(*at);
            let started = std::time::Instant::now();
            let hits = self.search(&query, k, ef, Some(&predicate)).ok()?;
            let elapsed = started.elapsed();
            std::hint::black_box(hits);
            samples.push(elapsed.as_secs_f64() * 1e9);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        Some(samples[samples.len() / 2])
    }

    /// Whether this graph holds a raw vector for every node it carries.
    pub fn holds_raw(&self) -> bool {
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_) => true,
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw.is_some()
            }
            VectorGraph::Int8(_) => false,
        }
    }

    /// Records a raw side store may be reserved for, given a declaration.
    ///
    /// `expected_size` or as many vectors as `RESERVE_BYTES` holds at this
    /// width, whichever is smaller, which is the rule
    /// `mutable::reserved_records` applies to the graph's own arenas and the
    /// same budget. A caller passing a record count it has already counted
    /// wants that count and does not come through here.
    ///
    /// What it is for: `expected_size` is validated at 100 million and a
    /// `Vec::with_capacity` that cannot be served aborts the process rather
    /// than unwinding, so a declaration of 100 million records at dimension
    /// 1,536 asks a raw store for 614 GB and takes the interpreter with it. No
    /// exception can be raised after that. The graph's own arenas have been
    /// held to a byte budget since they started reserving a vector per declared
    /// record; the side store a quantized graph opens was reserved from the
    /// declaration directly and was not.
    pub fn reserved_raw_records(dim: usize, expected_size: usize) -> usize {
        let per_record = dim.saturating_mul(std::mem::size_of::<f32>()).max(1);
        expected_size.min((mutable::RESERVE_BYTES / per_record).max(1))
    }

    /// Open a raw side store on a quantized graph, sized for `records`.
    ///
    /// A raw graph refuses, because its own store already is the raw vectors
    /// and a second one would be the duplication this replaced.
    pub fn open_raw_store(&mut self, dim: usize, records: usize) -> Result<(), String> {
        // The one construction site of a store that does not clamp its width,
        // because the two graph constructors do and this takes the index's
        // declared dimension instead. That dimension is validated where an
        // index is created and is not validated where one is loaded, so a
        // config.json naming a zero reaches here rather than being stopped
        // earlier. Every such directory is refused before this by the checks
        // that compare the dump header and the codebook against the declared
        // dimension, so this is the belt rather than the braces, and it is
        // here because a refusal costs nothing and a store of no width can
        // address no node.
        if dim == 0 {
            return Err("a raw store holds vectors of at least one value".to_string());
        }
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_) => Err("a raw graph already holds its raw vectors".to_string()),
            VectorGraph::Int8(_) => {
                Err("a scalar quantized graph keeps no raw vectors beside its codes".to_string())
            }
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw = Some(VectorStore::with_capacity(dim, records));
                Ok(())
            }
        }
    }

    /// Reserve the raw side store for a declared record count.
    ///
    /// A store built by pushing grows geometrically, so a store that starts at
    /// the record count present when it was opened and then takes the rest of
    /// the corpus ends holding close to twice what it uses. The training
    /// transition opens one at the training sample's size, which is a fraction
    /// of what the index expects, and every record after that is a push into a
    /// block sized for the sample. Reserving here for what the index was
    /// declared for makes the block the size the caller asked for, exactly as
    /// the graph's own arenas are at creation.
    ///
    /// The declaration goes through [`VectorGraph::reserved_raw_records`], so
    /// this cannot ask for more than the byte budget allows. It never shrinks:
    /// a store already past the declaration keeps what it has, since the
    /// declaration is a hint and the records are real.
    ///
    /// Nothing on a graph that keeps no raw store, which is a raw graph, where
    /// the store is the graph's own and is reserved at construction, and a
    /// `quantized_only` graph, which has none.
    pub fn reserve_raw_for(&mut self, expected_size: usize) {
        let Some(dim) = self.raw_dim() else {
            return;
        };
        let want = Self::reserved_raw_records(dim, expected_size);
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_)
            | VectorGraph::Int8(_) => {}
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                if let Some(raw) = b.raw.as_mut() {
                    raw.reserve_for(want);
                }
            }
        }
    }

    /// Carry every raw vector `source` holds into this graph, in this graph's
    /// node order.
    ///
    /// This is the training transition and the compaction of a
    /// `quantized_with_raw` index, both of which build a replacement graph
    /// whose node numbering is its own. The raws are re-addressed rather than
    /// re-derived, so nothing is quantized, reconstructed or lost, and a record
    /// the source cannot supply is an error rather than a gap. Each node of
    /// this graph names the position it files its record at, and
    /// `source_node_at` names, for each such position, the node of the same
    /// record in `source`.
    pub fn adopt_raw_from(
        &mut self,
        source: &VectorGraph,
        source_node_at: &[u32],
        dim: usize,
    ) -> Result<usize, String> {
        let nodes = self.nb_points();
        self.open_raw_store(dim, nodes)?;
        for node in 0..nodes as u32 {
            let position = self.record_of_at(node);
            let vector = source_node_at
                .get(position as usize)
                .and_then(|&old| source.raw_vector_at(old))
                .ok_or_else(|| {
                    format!(
                        "record with internal id {} has no raw vector to carry over",
                        self.origin_id_at(node)
                    )
                })?;
            match self {
                VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                    b.raw
                        .as_mut()
                        .expect("the store was opened above")
                        .push(vector);
                }
                _ => unreachable!("open_raw_store refused every raw graph"),
            }
        }
        Ok(nodes)
    }

    /// Append one raw vector to the side store, which becomes the next node's.
    ///
    /// For the loader, which fills the store in node order after the graph has
    /// come back from its dump. A raw graph refuses, for the reason
    /// `open_raw_store` refuses.
    pub fn push_raw_vector(&mut self, values: &[f32]) -> Result<(), String> {
        match self {
            VectorGraph::Cosine(_)
            | VectorGraph::L2(_)
            | VectorGraph::L1(_)
            | VectorGraph::Dot(_) => Err("a raw graph already holds its raw vectors".to_string()),
            VectorGraph::Int8(_) => {
                Err("a scalar quantized graph keeps no raw vectors beside its codes".to_string())
            }
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                match b.raw.as_mut() {
                    Some(store) => {
                        store.push(values);
                        Ok(())
                    }
                    None => Err("this graph keeps no raw vectors".to_string()),
                }
            }
        }
    }

    /// Raw vectors this graph holds, which is its node count where it holds
    /// them at all.
    pub fn raw_count(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.store.len(),
            VectorGraph::L2(b) => b.store.len(),
            VectorGraph::L1(b) => b.store.len(),
            VectorGraph::Dot(b) => b.store.len(),
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw.as_ref().map_or(0, VectorStore::len)
            }
            VectorGraph::Int8(_) => 0,
        }
    }

    /// Values a stored raw vector holds, where this graph holds any.
    pub fn raw_dim(&self) -> Option<usize> {
        match self {
            VectorGraph::Cosine(b) => Some(b.store.dim()),
            VectorGraph::L2(b) => Some(b.store.dim()),
            VectorGraph::L1(b) => Some(b.store.dim()),
            VectorGraph::Dot(b) => Some(b.store.dim()),
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.raw.as_ref().map(VectorStore::dim)
            }
            VectorGraph::Int8(_) => None,
        }
    }

    /// Return every buffer's spare capacity to the allocator, and report the
    /// bytes released.
    ///
    /// What the slack is and why a built graph carries it where a loaded one
    /// does not is on `MutableGraph::shrink_to_fit`.
    pub fn shrink_to_fit(&mut self) -> usize {
        let before = self.store_memory_bytes() + self.raw_memory_bytes();
        let links = match self {
            VectorGraph::Cosine(b) => {
                b.store.shrink_to_fit();
                b.graph.shrink_to_fit()
            }
            VectorGraph::L2(b) => {
                b.store.shrink_to_fit();
                b.graph.shrink_to_fit()
            }
            VectorGraph::L1(b) => {
                b.store.shrink_to_fit();
                b.graph.shrink_to_fit()
            }
            VectorGraph::Dot(b) => {
                b.store.shrink_to_fit();
                b.graph.shrink_to_fit()
            }
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.store.shrink_to_fit();
                if let Some(raw) = b.raw.as_mut() {
                    raw.shrink_to_fit();
                }
                b.graph.shrink_to_fit()
            }
            VectorGraph::Int8(b) => {
                b.store.shrink_to_fit();
                b.graph.shrink_to_fit()
            }
        };
        links + before.saturating_sub(self.store_memory_bytes() + self.raw_memory_bytes())
    }

    /// Take the reservation `expected_size` declares, which is what
    /// [`VectorGraph::new_raw`] takes at construction, on a graph
    /// [`VectorGraph::new_raw_unreserved`] built and nothing has been
    /// inserted into.
    pub fn reserve_declared(&mut self, expected_size: usize) {
        match self {
            VectorGraph::Cosine(b) => b.reserve_declared(expected_size),
            VectorGraph::L2(b) => b.reserve_declared(expected_size),
            VectorGraph::L1(b) => b.reserve_declared(expected_size),
            VectorGraph::Dot(b) => b.reserve_declared(expected_size),
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                b.reserve_declared(expected_size)
            }
            VectorGraph::Int8(b) => b.reserve_declared(expected_size),
        }
    }

    /// Whether this graph scores against codes rather than against the
    /// vectors themselves, which is true of a product quantized graph and
    /// of a scalar quantized one. A caller that needs to know which asks
    /// [`VectorGraph::is_pq`] or [`VectorGraph::is_int8`].
    pub fn is_quantized(&self) -> bool {
        self.is_pq() || self.is_int8()
    }

    /// Whether this graph scores against product quantized codes through a
    /// codebook.
    pub fn is_pq(&self) -> bool {
        matches!(
            self,
            VectorGraph::CosinePQ(_) | VectorGraph::L2PQ(_) | VectorGraph::L1PQ(_)
        )
    }

    /// Whether this graph scores against scalar codes, one signed byte a
    /// value, decoded through a per dimension scale.
    pub fn is_int8(&self) -> bool {
        matches!(self, VectorGraph::Int8(_))
    }

    /// Reseed the level stream. Resets it rather than extending it, so a caller
    /// that wants a chosen seed calls this before the first insertion.
    ///
    /// No shipped path calls this. Every graph draws from `DEFAULT_LEVEL_SEED`,
    /// which is what makes a build a function of its data and its parameters
    /// alone. It exists under `cfg(test)` so that the capability the vendored
    /// `Hnsw::set_level_seed` carried is still reachable where a test needs two
    /// graphs drawn from one chosen stream.
    #[cfg(test)]
    pub(crate) fn set_level_seed(&mut self, seed: u64) {
        let reseed = |levels: &Mutex<LevelGenerator>| {
            levels.lock().unwrap().set_seed(seed);
        };
        match self {
            VectorGraph::Cosine(b) => reseed(&b.levels),
            VectorGraph::L2(b) => reseed(&b.levels),
            VectorGraph::L1(b) => reseed(&b.levels),
            VectorGraph::Dot(b) => reseed(&b.levels),
            VectorGraph::CosinePQ(b) => reseed(&b.levels),
            VectorGraph::L2PQ(b) => reseed(&b.levels),
            VectorGraph::L1PQ(b) => reseed(&b.levels),
            VectorGraph::Int8(b) => reseed(&b.levels),
        }
    }

    /// Draw the level the next insertion's node takes, advancing the level
    /// stream by one draw.
    ///
    /// Reads the graph and writes nothing but the stream's own position, so it
    /// runs under a read guard, taken for the draw alone. The draw sits ahead
    /// of [`Self::plan_at_level`] rather than inside it so that a caller holds
    /// the level before the insertion begins: one that records the level can
    /// record it then, and one replaying a recorded level plans at it without
    /// drawing. Exactly one draw per insertion, in insertion order, is what
    /// makes two builds of the same records the same graph, so a caller that
    /// draws and then installs nothing has moved the stream.
    pub fn draw_level(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.draw_level(),
            VectorGraph::L2(b) => b.draw_level(),
            VectorGraph::L1(b) => b.draw_level(),
            VectorGraph::Dot(b) => b.draw_level(),
            VectorGraph::CosinePQ(b) => b.draw_level(),
            VectorGraph::L2PQ(b) => b.draw_level(),
            VectorGraph::L1PQ(b) => b.draw_level(),
            VectorGraph::Int8(b) => b.draw_level(),
        }
    }

    /// Phase one of one insertion at `level`: descend and choose the lists,
    /// drawing nothing.
    ///
    /// Reads the graph and writes nothing, so it runs under a read guard. What
    /// comes back owns everything it holds, so the guard is dropped before
    /// [`Self::install`] takes the write guard. `None` means the record does not
    /// belong in this graph, which is a programming error rather than a state,
    /// and the caller then installs nothing. `level` is what
    /// [`Self::draw_level`] handed out, or what a record of that draw carries;
    /// the install asserts it below the layer count.
    ///
    /// The soundness of the split is the index's `writers` mutex: it admits one
    /// mutator at a time and every mutating entry point takes it, so nothing can
    /// change the graph between the two phases. `install` asserts the node count
    /// it planned against rather than trusting that.
    pub fn plan_at_level(&self, record: Record<'_>, level: usize) -> Option<Planned> {
        match (self, record) {
            (VectorGraph::Cosine(b), Record::Raw(v)) => {
                assert_unit_for_cosine(v, "insert");
                Some(b.plan_at_level(v, level))
            }
            (VectorGraph::L2(b), Record::Raw(v)) => Some(b.plan_at_level(v, level)),
            (VectorGraph::L1(b), Record::Raw(v)) => Some(b.plan_at_level(v, level)),
            (VectorGraph::Dot(b), Record::Raw(v)) => Some(b.plan_at_level(v, level)),
            (VectorGraph::CosinePQ(b), Record::Codes { codes, .. })
            | (VectorGraph::L2PQ(b), Record::Codes { codes, .. })
            | (VectorGraph::L1PQ(b), Record::Codes { codes, .. }) => {
                Some(b.plan_at_level(codes, level))
            }
            (VectorGraph::Int8(b), Record::Int8(codes)) => Some(b.plan_at_level(codes, level)),
            // A scalar graph handed anything but scalar codes, or scalar
            // codes handed to any other graph, is the same programming error
            // the two arms below name for the other element types.
            (VectorGraph::Int8(_), _) | (_, Record::Int8(_)) => {
                error!(
                    target: LOG_TARGET,
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "scalar_codes_and_graph_disagree",
                    "A scalar quantized graph takes scalar codes and nothing else takes them"
                );
                None
            }
            (_, Record::Raw(_)) => {
                error!(
                    target: LOG_TARGET,
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_raw_vectors_into_pq_index",
                    "Cannot insert raw vectors into PQ index"
                );
                None
            }
            (_, Record::Codes { .. }) => {
                error!(
                    target: LOG_TARGET,
                    operation = "pq_codes_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_pq_codes_into_raw_index",
                    "Cannot insert PQ codes into raw index"
                );
                None
            }
        }
    }

    /// Phase two of one insertion: append the node and write its edges.
    ///
    /// Runs under the write guard, and takes the plan the read guarded phase
    /// produced. The record must be the one that was planned. The node takes
    /// `id` as the internal id the dump records and `at` as the position of
    /// the record it holds.
    pub fn install(&mut self, record: Record<'_>, id: usize, at: Position, planned: Planned) {
        let position = at.0;
        match (self, record) {
            (VectorGraph::Cosine(b), Record::Raw(v)) => b.install(v, id, position, planned),
            (VectorGraph::L2(b), Record::Raw(v)) => b.install(v, id, position, planned),
            (VectorGraph::L1(b), Record::Raw(v)) => b.install(v, id, position, planned),
            (VectorGraph::Dot(b), Record::Raw(v)) => b.install(v, id, position, planned),
            (VectorGraph::CosinePQ(b), Record::Codes { codes, raw })
            | (VectorGraph::L2PQ(b), Record::Codes { codes, raw })
            | (VectorGraph::L1PQ(b), Record::Codes { codes, raw }) => {
                b.install(codes, id, position, planned);
                b.push_raw(raw);
            }
            (VectorGraph::Int8(b), Record::Int8(codes)) => b.install(codes, id, position, planned),
            // Unreachable, because a plan the element type refused is `None`
            // and the caller then installs nothing.
            _ => error!(
                target: LOG_TARGET,
                operation = "vector_insert",
                error = "invalid_operation",
                reason = "element_type_mismatch_at_install",
                "A planned insertion reached a graph of the other element type"
            ),
        }
    }

    /// Insert one raw vector under internal id `id` at position `at`, the
    /// draw and both phases, for a caller holding the graph outright.
    ///
    /// The three rebuild paths, being `compact`, the quantization rebuild and
    /// the persistence rebuild, each build a fresh graph off to the side and
    /// swap it in under one write guard, so the graph they insert into is a
    /// local nobody else can reach. They need no phase split and no lock.
    pub fn insert(&mut self, vector: &[f32], id: usize, at: Position) {
        let level = self.draw_level();
        if let Some(planned) = self.plan_at_level(Record::Raw(vector), level) {
            self.install(Record::Raw(vector), id, at, planned);
        }
    }

    /// Insertion is sequential whatever the batch size. Every caller is a
    /// one-time structural rebuild, at training completion, in `compact` or in
    /// the persistence loader, and each of them sorts its batch by internal id
    /// so that two rebuilds of the same records wire the same graph. Each entry
    /// is the codes, the internal id and the record's position.
    pub fn insert_batch_pq(&mut self, data: &[(&Vec<u8>, usize, Position)]) -> Result<(), String> {
        debug!(
            target: LOG_TARGET,
            operation = "batch_insert_pq",
            batch_size = data.len(),
            "Starting PQ batch insertion"
        );

        match self {
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                for (codes, id, at) in data {
                    b.insert(codes.as_slice(), *id, at.0);
                }

                Ok(())
            }
            _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
        }
    }

    /// Insert scalar rows, as [`VectorGraph::int8_encode`] produced them,
    /// one record at a time in the order given, for the rebuild paths that
    /// build a fresh graph off to the side. Every caller sorts its batch by
    /// internal id so that two rebuilds of the same records wire the same
    /// graph, as [`VectorGraph::insert_batch_pq`] requires of its callers.
    /// Each entry is the row, the internal id and the record's position.
    pub fn insert_batch_int8(&mut self, data: &[(&[i8], usize, Position)]) -> Result<(), String> {
        debug!(
            target: LOG_TARGET,
            operation = "batch_insert_int8",
            batch_size = data.len(),
            "Starting scalar code batch insertion"
        );
        match self {
            VectorGraph::Int8(b) => {
                for (codes, id, at) in data {
                    b.insert(codes, *id, at.0);
                }
                Ok(())
            }
            _ => Err("Cannot insert scalar codes into a graph that does not hold them".to_string()),
        }
    }

    /// Which of the graphs this is, as the dump header records it.
    fn kind(&self) -> GraphKind {
        match self {
            VectorGraph::Cosine(_) => GraphKind::Cosine,
            VectorGraph::L2(_) => GraphKind::L2,
            VectorGraph::L1(_) => GraphKind::L1,
            VectorGraph::Dot(_) => GraphKind::Dot,
            VectorGraph::CosinePQ(_) => GraphKind::CosinePq,
            VectorGraph::L2PQ(_) => GraphKind::L2Pq,
            VectorGraph::L1PQ(_) => GraphKind::L1Pq,
            VectorGraph::Int8(b) => match b.graph.distance().metric() {
                Int8Metric::Cosine => GraphKind::CosineInt8,
                Int8Metric::L2 => GraphKind::L2Int8,
                Int8Metric::L1 => GraphKind::L1Int8,
                Int8Metric::Dot => GraphKind::DotInt8,
            },
        }
    }

    /// Write the graph as a version 1 dump, which a directory saved before
    /// format 5.0.0 holds, for a test that makes such a directory.
    #[cfg(any(test, feature = "test-support"))]
    pub fn dump_version_1(&self, dir: &dyn Dir, name: &str) -> Result<(), String> {
        let kind = self.kind();
        match self {
            VectorGraph::Cosine(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L2(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L1(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::Dot(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::CosinePQ(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L2PQ(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L1PQ(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::Int8(b) => {
                dump::v1::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
        }
    }

    /// Write the graph in ZeusDB's own format, at version 2, as the artefact
    /// `name` in `dir`, replacing any artefact of that name.
    pub fn dump(&self, dir: &dyn Dir, name: &str) -> Result<(), String> {
        let kind = self.kind();
        trace!(
            target: LOG_TARGET,
            operation = "save_hnsw_graph",
            distance_type = kind.label(),
            "Writing the graph dump"
        );
        match self {
            VectorGraph::Cosine(b) => {
                dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L2(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
            VectorGraph::L1(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
            VectorGraph::Dot(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
            VectorGraph::CosinePQ(b) => {
                dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name)
            }
            VectorGraph::L2PQ(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
            VectorGraph::L1PQ(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
            VectorGraph::Int8(b) => dump::write_dump(&b.graph.dump_view(&b.store), kind, dir, name),
        }
    }
}

// ============================================================================
// RESTORING THE SAVED GRAPH
// ============================================================================

/// What the directory says the dump has to sit within, the records its nodes
/// are filed for, and how it is to be read.
///
/// The two bounds are read from the directory rather than from the dump, and
/// both are checked before the reader allocates from anything the file
/// declares.
#[derive(Clone, Copy)]
pub struct DumpBounds<'a> {
    /// The live record count. The graph holds at least this many nodes and
    /// holds more whenever a removal or an overwrite has stranded one.
    pub min_nodes: usize,
    /// `config.json`'s `id_counter`, being the largest internal id the index
    /// has ever issued.
    ///
    /// See `dump::Expected::max_origin_id`, which is where it is checked and
    /// why.
    pub max_origin_id: usize,
    /// The internal id of the record at each position, in increasing order,
    /// being the records the mappings hold. A node whose origin id it holds
    /// takes that id's position, any other node is stranded, and an id it
    /// holds that no node carries refuses the dump. `None` files every node
    /// at the rank of its origin id, which is the graph whose every node
    /// holds a record.
    pub positions: Option<&'a [u32]>,
    /// The dump version the directory's format requires: 1 below format
    /// 5.0.0 and 2 from it.
    pub version: u32,
    /// At version 2, read and verify the identity and the rows and hand the
    /// rows back without building a graph, for a rebuild the caller asked
    /// for. Version 1 reads the whole dump whatever this says.
    pub rows_only: bool,
}

/// What restoring a graph from its dump gave.
#[allow(private_interfaces)]
pub enum Restored {
    /// The graph the dump holds, boxed so the restore moves a pointer to it.
    Graph(Box<VectorGraph>),
    /// The rows of a version 2 dump, each region verified, where no graph was
    /// built from it: its topology fails, which `reason` names, or the caller
    /// asked for the rows alone.
    Rows { rows: DumpRows, reason: String },
}

/// The rows of a version 2 dump, in the dump's node order, with what each
/// node is filed by.
pub struct DumpRows {
    /// Each node's origin id, being the internal id it was inserted under.
    pub origin_ids: Vec<usize>,
    /// Each node's position among the records the mappings hold, or
    /// `Position::NONE` for a stranded node.
    pub positions: Vec<u32>,
    /// Values in one row.
    pub width: usize,
    /// Every node's row, end to end.
    pub values: RowValues,
}

/// The values of every row a dump holds, in the element type its graph
/// stores.
pub enum RowValues {
    /// A raw graph's vectors.
    Raw(Vec<f32>),
    /// A product quantized graph's codes, one byte a subvector.
    Codes(Vec<u8>),
    /// A scalar quantized graph's rows, one signed byte a value.
    Int8(Vec<i8>),
}

/// The rows of `read` as the type a caller rebuilds from.
fn erased_rows<T: dump::DumpElement>(read: dump::ReadRows<T>) -> DumpRows {
    let width = read.store.dim();
    DumpRows {
        origin_ids: read.origin_ids,
        positions: read.record_of,
        width,
        values: T::into_row_values(read.store.into_values()),
    }
}

/// Read a dump of element type `T` and hand back what it gave, the graph
/// wrapped by `wrap` where it gave one.
fn restore_with<T, D>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
    dist: D,
    bounds: &DumpBounds<'_>,
    wrap: impl FnOnce(Backend<T, D>) -> VectorGraph,
) -> Result<Restored, DumpRefusal>
where
    T: dump::DumpElement,
    D: Distance<T> + Send + Sync,
{
    match dump::read_dump_at::<T, D>(
        dir,
        name,
        expected,
        dist,
        bounds.positions,
        bounds.rows_only,
    )? {
        dump::DumpRead::Graph(graph, store) => Ok(Restored::Graph(Box::new(wrap(
            Backend::restored((*graph, store)),
        )))),
        dump::DumpRead::Rows { rows, reason } => Ok(Restored::Rows {
            rows: erased_rows(rows),
            reason,
        }),
    }
}

/// Restore the saved graph for one index configuration.
///
/// `pq` present means the saved graph was a product quantized one, which is
/// the condition the loader branches on, since training is what replaces the
/// raw graph with a quantized one. A dump whose element type or kind
/// disagrees with that is refused on its header.
///
/// A refusal names its region. Below format 5.0.0 the caller rebuilds the
/// graph from the directory's separate row stores on any refusal. At 5.0.0 a
/// refusal refuses the load, and the rows handed back where the topology
/// fails are what the caller rebuilds from.
///
/// A restored graph draws levels at the scale the dump recorded, if the index
/// goes on inserting. See [`Backend::restored`].
///
/// The dump is the artefact `name` in `dir`.
#[allow(clippy::too_many_arguments)]
pub fn restore_graph(
    dir: &dyn Dir,
    name: &str,
    space: &str,
    m: usize,
    ef_construction: usize,
    dim: usize,
    pq: Option<Arc<PQ>>,
    bounds: DumpBounds<'_>,
) -> Result<Restored, DumpRefusal> {
    let DumpBounds {
        min_nodes,
        max_origin_id,
        version,
        ..
    } = bounds;
    match pq {
        Some(pq) => {
            let kind = match space {
                "l2" => GraphKind::L2Pq,
                "l1" => GraphKind::L1Pq,
                // `new_pq` also falls back to cosine on an unrecognised space,
                // so the two construction paths agree on what a bad space means.
                _ => GraphKind::CosinePq,
            };
            let expected = Expected {
                kind,
                dimension: pq.subvectors(),
                m,
                ef_construction,
                min_nodes,
                max_origin_id,
                version,
            };
            // The metric follows the discriminant the dump carries, so a
            // directory scores the way the graph inside it was wired.
            let metric = match kind {
                GraphKind::CosinePq => PqMetric::Cosine,
                _ => PqMetric::SquaredL2,
            };
            restore_with::<u8, DistPQ>(
                dir,
                name,
                &expected,
                DistPQ::new(pq, metric),
                &bounds,
                |restored| match kind {
                    GraphKind::L2Pq => VectorGraph::L2PQ(restored),
                    GraphKind::L1Pq => VectorGraph::L1PQ(restored),
                    _ => VectorGraph::CosinePQ(restored),
                },
            )
        }
        None => {
            // The raw graphs differ only in their distance type, so each arm
            // states the discriminant the dump must carry and the value the
            // reload needs, and nothing else about them differs.
            macro_rules! raw {
                ($kind:expr, $dist:ty, $value:expr, $variant:ident) => {{
                    let expected = Expected {
                        kind: $kind,
                        dimension: dim,
                        m,
                        ef_construction,
                        min_nodes,
                        max_origin_id,
                        version,
                    };
                    restore_with::<f32, $dist>(
                        dir,
                        name,
                        &expected,
                        $value,
                        &bounds,
                        VectorGraph::$variant,
                    )
                }};
            }
            match space {
                "l2" => raw!(GraphKind::L2, L2Dist, L2Dist {}, L2),
                "l1" => raw!(GraphKind::L1, L1Dist, L1Dist {}, L1),
                "dot" => raw!(GraphKind::Dot, DotDist, DotDist {}, Dot),
                // `new_raw` also falls back to cosine on an unrecognised space,
                // so the two construction paths agree on what a bad space means.
                _ => raw!(GraphKind::Cosine, CosineDist, CosineDist {}, Cosine),
            }
        }
    }
}

/// The scalar quantized kind a space name maps to, which is the kind a dump
/// of such a graph carries and the kind `restore_int8_graph` expects.
fn int8_kind(space: &str) -> GraphKind {
    match Int8Metric::from_space(space) {
        Int8Metric::Cosine => GraphKind::CosineInt8,
        Int8Metric::L2 => GraphKind::L2Int8,
        Int8Metric::L1 => GraphKind::L1Int8,
        Int8Metric::Dot => GraphKind::DotInt8,
    }
}

/// Restore the saved graph of a scalar quantized index.
///
/// The counterpart of [`restore_graph`] for a graph holding one `i8` a
/// value. The codec and the space together decide the row width the dump
/// must carry, being the codec's width plus the norm tail on a cosine graph,
/// and the space decides the kind the header must carry. A refusal and the
/// rows handed back mean what they mean for [`restore_graph`]. The dump is
/// the artefact `name` in `dir`.
pub fn restore_int8_graph(
    dir: &dyn Dir,
    name: &str,
    space: &str,
    m: usize,
    ef_construction: usize,
    codec: Arc<Int8Codec>,
    bounds: DumpBounds<'_>,
) -> Result<Restored, DumpRefusal> {
    let kind = int8_kind(space);
    let dist = Int8Dist::new(codec, Int8Metric::from_space(space));
    let expected = Expected {
        kind,
        dimension: dist.row_width(),
        m,
        ef_construction,
        min_nodes: bounds.min_nodes,
        max_origin_id: bounds.max_origin_id,
        version: bounds.version,
    };
    restore_with::<i8, Int8Dist>(dir, name, &expected, dist, &bounds, VectorGraph::Int8)
}
