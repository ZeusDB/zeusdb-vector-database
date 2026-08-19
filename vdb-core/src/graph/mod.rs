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
//! it. [`Distance`] is the trait `CosineDist`, `L1Dist`, `L2Dist`, `DotDist`
//! and `DistPQ` implement, and a trait has to be nameable where the
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
//! becoming unreadable. None of them is moved here.

use crate::distance::{CosineDist, DistPQ, L1Dist, L2Dist};
use crate::pq::PQ;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, trace, warn};

pub(crate) mod dump;

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
// The traversal, written once against an accessor.
#[cfg_attr(not(test), allow(dead_code))]
mod traverse;

use dump::{Expected, GraphKind};
use levels::LevelGenerator;
use mutable::{MutableGraph, Planned};
use traverse::{Topology, LAYERS};

/// How far apart two points of type `T` are.
///
/// One method, and every distance in the crate is one expression. It is
/// declared here rather than in `distance.rs` because the graph is what calls
/// it and `distance.rs` is one of the modules that implements it.
///
/// The implementors are `CosineDist`, `L2Dist`, `L1Dist` and `DotDist` in
/// `distance.rs` and `DistPQ` in `hnsw_index`. Every one imports the name from
/// this module, so the set is countable from the `use` sites of this line.
///
/// The shape is the one the trait it replaced had, unchanged, because changing
/// it would have been a second edit landing at the same time as the deletion
/// and there is nothing wrong with it.
pub(crate) trait Distance<T> {
    /// The distance from `va` to `vb`. Both slices are the graph's width.
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}

/// How far `sum(x * x)` may sit from one before a vector is not unit length.
///
/// The quantity is the squared norm rather than the norm, because that is what
/// the check computes and a square root would only add error to it.
///
/// The residual a correct normalisation leaves is the whole of what this has to
/// absorb. `HNSWIndex::normalize_vector` divides by an `f32` norm accumulated in
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
/// path takes the origin id, resolves it through `rev_map`, and takes the
/// distance as the score when no rerank is in play, so those two fields are
/// what the seam carries and nothing else.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphHit {
    /// The id the record was inserted under, which `rev_map` resolves.
    pub internal_id: usize,
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
/// The generator is behind a mutex because the draw happens in phase one, which
/// runs under the index's graph **read** guard. It is a leaf: nothing inside it
/// takes another lock, and no other lock is taken while it is held. It is also
/// uncontended, since the index's `writers` mutex admits one mutator at a time.
pub(crate) struct Backend<T, D> {
    graph: MutableGraph<T, D>,
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
    /// validated them already: `HNSWIndex::build` rejects a zero dimension, a
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
        let graph = MutableGraph::new(dim, m, ef_construction, scale, expected_size, dist_f)
            .expect("the arguments were clamped into the range MutableGraph::new accepts");
        Backend {
            graph,
            levels: Mutex::new(LevelGenerator::new(scale, maxlevel)),
        }
    }

    /// A graph read back from a dump, with a generator at the scale the dump
    /// recorded.
    ///
    /// The generator is seeded at `DEFAULT_LEVEL_SEED`, so a loaded index that
    /// keeps inserting starts the level stream over. That is what
    /// `Hnsw::from_loaded_points` does through `new_with_absolute_scale`, so it
    /// is the behaviour this restores rather than a new one.
    fn restored(graph: MutableGraph<T, D>) -> Self {
        let scale = graph.level_scale();
        Backend {
            graph,
            levels: Mutex::new(LevelGenerator::new(scale, LAYERS)),
        }
    }

    /// Phase one. Draws the level and decides the insertion, reading only.
    fn plan(&self, data: &[T]) -> Planned {
        let level = self
            .levels
            .lock()
            .expect("the level generator is a leaf and no path panics holding it")
            .generate();
        self.graph.plan_insertion(data, level)
    }

    /// Phase two. Writes what phase one decided.
    fn install(&mut self, data: &[T], origin_id: usize, planned: Planned) {
        self.graph.install_insertion(data, origin_id, planned);
    }

    /// Both phases, for a caller holding the graph outright.
    fn insert(&mut self, data: &[T], origin_id: usize) {
        let planned = self.plan(data);
        self.install(data, origin_id, planned);
    }
}

/// The graph, in whichever of the six shapes this index built it
///
/// Three raw variants holding `f32` points and three quantized variants holding
/// `u8` codes. They differ in the distance they were built with, which the
/// graph takes as a type parameter, so the enum is what stands in for a single
/// graph type.
pub(crate) enum VectorGraph {
    // Raw vector variants
    Cosine(Backend<f32, CosineDist>),
    L2(Backend<f32, L2Dist>),
    L1(Backend<f32, L1Dist>),

    // PQ variants, holding one `u8` per subvector
    CosinePQ(Backend<u8, DistPQ>),
    L2PQ(Backend<u8, DistPQ>),
    L1PQ(Backend<u8, DistPQ>),
}

/// One record on its way into the graph, in whichever form that graph holds.
///
/// The six variants split two ways at once, by distance and by element type,
/// and only the second decides what an insertion carries. Naming the payload
/// rather than writing one method per element type means the mismatch between a
/// raw record and a quantized graph is stated once per phase instead of four
/// times.
#[derive(Clone, Copy)]
pub(crate) enum Record<'a> {
    /// A raw vector, already normalized for the index space.
    Raw(&'a [f32]),
    /// The quantized codes of one record, one byte per subvector.
    Codes(&'a [u8]),
}

impl VectorGraph {
    pub(crate) fn new_raw(
        space: &str,
        dim: usize,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        info!(
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
                VectorGraph::$variant(Backend::sized(
                    dim,
                    m,
                    max_layer,
                    ef_construction,
                    expected_size,
                    $dist,
                ))
            };
        }
        match space {
            "cosine" => raw!(Cosine, CosineDist {}),
            "l2" => raw!(L2, L2Dist {}),
            "l1" => raw!(L1, L1Dist {}),
            _ => {
                error!(
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
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                raw!(Cosine, CosineDist {})
            }
        }
    }

    pub(crate) fn new_pq(
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

        macro_rules! quantized {
            ($variant:ident) => {
                VectorGraph::$variant(Backend::sized(
                    dim,
                    m,
                    max_layer,
                    ef_construction,
                    expected_size,
                    DistPQ::new(pq),
                ))
            };
        }
        match space {
            "cosine" => quantized!(CosinePQ),
            "l2" => quantized!(L2PQ),
            "l1" => quantized!(L1PQ),
            _ => {
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided for PQ"
                );
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                quantized!(CosinePQ)
            }
        }
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
    pub(crate) fn search<F>(
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
                Ok(b.graph.search(query, k, ef, filter))
            }
            VectorGraph::L2(b) => Ok(b.graph.search(query, k, ef, filter)),
            VectorGraph::L1(b) => Ok(b.graph.search(query, k, ef, filter)),

            // PQ-based search with ADC
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                // This query's ADC table, installed on this thread alone. The
                // guard is named so it lives to the end of the arm rather than
                // dropping at the end of the statement, and it releases the
                // table once the traversal is done.
                let _query_lut = b.graph.distance().install_query_lut(query)?;

                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; b.graph.distance().subvectors()];

                Ok(b.graph.search(&dummy_query, k, ef, filter))
            }
        }
    }

    /// Number of nodes the graph holds, which is the number of insertions it has
    /// taken. It exceeds the live record count by exactly the number of nodes
    /// that removal and overwrite have stranded.
    pub(crate) fn nb_points(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.nb_points(),
            VectorGraph::L2(b) => b.graph.nb_points(),
            VectorGraph::L1(b) => b.graph.nb_points(),
            VectorGraph::CosinePQ(b) => b.graph.nb_points(),
            VectorGraph::L2PQ(b) => b.graph.nb_points(),
            VectorGraph::L1PQ(b) => b.graph.nb_points(),
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
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.memory_bytes(),
            VectorGraph::L2(b) => b.graph.memory_bytes(),
            VectorGraph::L1(b) => b.graph.memory_bytes(),
            VectorGraph::CosinePQ(b) => b.graph.memory_bytes(),
            VectorGraph::L2PQ(b) => b.graph.memory_bytes(),
            VectorGraph::L1PQ(b) => b.graph.memory_bytes(),
        }
    }

    /// Return every buffer's spare capacity to the allocator, and report the
    /// bytes released.
    ///
    /// What the slack is and why a built graph carries it where a loaded one
    /// does not is on `MutableGraph::shrink_to_fit`.
    pub(crate) fn shrink_to_fit(&mut self) -> usize {
        match self {
            VectorGraph::Cosine(b) => b.graph.shrink_to_fit(),
            VectorGraph::L2(b) => b.graph.shrink_to_fit(),
            VectorGraph::L1(b) => b.graph.shrink_to_fit(),
            VectorGraph::CosinePQ(b) => b.graph.shrink_to_fit(),
            VectorGraph::L2PQ(b) => b.graph.shrink_to_fit(),
            VectorGraph::L1PQ(b) => b.graph.shrink_to_fit(),
        }
    }

    pub(crate) fn is_quantized(&self) -> bool {
        matches!(
            self,
            VectorGraph::CosinePQ(_) | VectorGraph::L2PQ(_) | VectorGraph::L1PQ(_)
        )
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
            VectorGraph::CosinePQ(b) => reseed(&b.levels),
            VectorGraph::L2PQ(b) => reseed(&b.levels),
            VectorGraph::L1PQ(b) => reseed(&b.levels),
        }
    }

    /// Phase one of one insertion: draw the level, descend, choose the lists.
    ///
    /// Reads the graph and writes nothing, so it runs under a read guard. What
    /// comes back owns everything it holds, so the guard is dropped before
    /// [`Self::install`] takes the write guard. `None` means the record does not
    /// belong in this graph, which is a programming error rather than a state,
    /// and the caller then installs nothing.
    ///
    /// The soundness of the split is the index's `writers` mutex: it admits one
    /// mutator at a time and every mutating entry point takes it, so nothing can
    /// change the graph between the two phases. `install` asserts the node count
    /// it planned against rather than trusting that.
    pub(crate) fn plan(&self, record: Record<'_>) -> Option<Planned> {
        match (self, record) {
            (VectorGraph::Cosine(b), Record::Raw(v)) => {
                assert_unit_for_cosine(v, "insert");
                Some(b.plan(v))
            }
            (VectorGraph::L2(b), Record::Raw(v)) => Some(b.plan(v)),
            (VectorGraph::L1(b), Record::Raw(v)) => Some(b.plan(v)),
            (VectorGraph::CosinePQ(b), Record::Codes(c))
            | (VectorGraph::L2PQ(b), Record::Codes(c))
            | (VectorGraph::L1PQ(b), Record::Codes(c)) => Some(b.plan(c)),
            (_, Record::Raw(_)) => {
                error!(
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_raw_vectors_into_pq_index",
                    "Cannot insert raw vectors into PQ index"
                );
                None
            }
            (_, Record::Codes(_)) => {
                error!(
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
    /// produced. The record must be the one that was planned.
    pub(crate) fn install(&mut self, record: Record<'_>, id: usize, planned: Planned) {
        match (self, record) {
            (VectorGraph::Cosine(b), Record::Raw(v)) => b.install(v, id, planned),
            (VectorGraph::L2(b), Record::Raw(v)) => b.install(v, id, planned),
            (VectorGraph::L1(b), Record::Raw(v)) => b.install(v, id, planned),
            (VectorGraph::CosinePQ(b), Record::Codes(c))
            | (VectorGraph::L2PQ(b), Record::Codes(c))
            | (VectorGraph::L1PQ(b), Record::Codes(c)) => b.install(c, id, planned),
            // Unreachable, because a plan the element type refused is `None`
            // and the caller then installs nothing.
            _ => error!(
                operation = "vector_insert",
                error = "invalid_operation",
                reason = "element_type_mismatch_at_install",
                "A planned insertion reached a graph of the other element type"
            ),
        }
    }

    /// Insert one raw vector, both phases, for a caller holding the graph
    /// outright.
    ///
    /// The three rebuild paths, being `compact`, the quantization rebuild and
    /// the persistence rebuild, each build a fresh graph off to the side and
    /// swap it in under one write guard, so the graph they insert into is a
    /// local nobody else can reach. They need no phase split and no lock.
    pub(crate) fn insert(&mut self, vector: &[f32], id: usize) {
        if let Some(planned) = self.plan(Record::Raw(vector)) {
            self.install(Record::Raw(vector), id, planned);
        }
    }

    /// Insertion is sequential whatever the batch size. Every caller is a
    /// one-time structural rebuild, at training completion, in `compact` or in
    /// the persistence loader, and each of them sorts its batch by internal id
    /// so that two rebuilds of the same records wire the same graph.
    pub(crate) fn insert_batch_pq(&mut self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        debug!(
            operation = "batch_insert_pq",
            batch_size = data.len(),
            "Starting PQ batch insertion"
        );

        match self {
            VectorGraph::CosinePQ(b) | VectorGraph::L2PQ(b) | VectorGraph::L1PQ(b) => {
                for (codes, id) in data {
                    b.insert(codes.as_slice(), *id);
                }

                Ok(())
            }
            _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
        }
    }

    /// Which of the six graphs this is, as the dump header records it.
    fn kind(&self) -> GraphKind {
        match self {
            VectorGraph::Cosine(_) => GraphKind::Cosine,
            VectorGraph::L2(_) => GraphKind::L2,
            VectorGraph::L1(_) => GraphKind::L1,
            VectorGraph::CosinePQ(_) => GraphKind::CosinePq,
            VectorGraph::L2PQ(_) => GraphKind::L2Pq,
            VectorGraph::L1PQ(_) => GraphKind::L1Pq,
        }
    }

    /// Write the graph to `dir` in ZeusDB's own format.
    ///
    /// Returns the name of the file written, which is fixed. The vendored
    /// writer returned a basename that was not always the one asked for,
    /// because it appended a random suffix rather than overwriting when a
    /// memory mapped data file was active. Nothing maps anything here and the
    /// file is replaced outright, so the name is a constant.
    pub(crate) fn dump(&self, dir: &Path) -> Result<String, String> {
        let kind = self.kind();
        trace!(
            operation = "save_hnsw_graph",
            distance_type = kind.label(),
            "Writing the graph dump"
        );
        match self {
            VectorGraph::Cosine(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
            VectorGraph::L2(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
            VectorGraph::L1(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
            VectorGraph::CosinePQ(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
            VectorGraph::L2PQ(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
            VectorGraph::L1PQ(b) => dump::write_dump(&b.graph.dump_view(), kind, dir),
        }?;
        Ok(dump::DUMP_FILENAME.to_string())
    }
}

// ============================================================================
// RESTORING THE SAVED GRAPH
// ============================================================================

/// Restore the saved graph for one index configuration
///
/// `pq` present means the saved graph was a quantized one, which is exactly the
/// condition the loader branches on, since training is what replaces the raw
/// graph with a PQ graph. A directory whose dump disagrees is caught by the
/// element type and the distance discriminant in the header and falls back.
///
/// Every reason this returns an error is a reason to rebuild rather than a
/// reason to fail, and the caller treats it that way. A dump written by 0.6.0
/// or earlier is one of them: it carries the vendored magic, this reader does
/// not recognise it, and the index rebuilds its graph once and writes the new
/// format on the next save. There is deliberately no reader for the old format.
///
/// # What this no longer does
///
/// It used to inspect the dump's header, compare the data file's length against
/// what the point count implied, and only then enter the vendored reload inside
/// a `catch_unwind`, because that reload panics on a malformed header and calls
/// `std::process::exit(1)` on a short data file. It also leaked the vendored
/// loader, because the returned graph borrowed it. [`dump::read_dump`] returns
/// an error on every malformed input and owns nothing borrowed, so all four are
/// gone.
///
/// It also used to reset `extend_candidates`, which the vendored reload set
/// true where `Hnsw::new` sets it false. Neither construction path here has such
/// a flag, since the branches those two parameters guarded are absent rather
/// than present behind a setter nothing calls.
///
/// # The level scale comes from the dump
///
/// A restored graph keeps drawing levels if the index goes on inserting, and it
/// draws at the scale the dump recorded rather than at the default for its
/// degree. That is what the vendored reload did through
/// `new_with_absolute_scale`, so it is a match rather than a change. See
/// [`Backend::restored`].
pub(crate) fn restore_graph(
    dir: &Path,
    space: &str,
    m: usize,
    ef_construction: usize,
    dim: usize,
    pq: Option<Arc<PQ>>,
    min_nodes: usize,
) -> Result<(VectorGraph, usize), String> {
    let graph = match pq {
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
            };
            let restored = Backend::restored(dump::read_dump::<u8, DistPQ>(
                dir,
                &expected,
                DistPQ::new(pq),
            )?);
            match kind {
                GraphKind::L2Pq => VectorGraph::L2PQ(restored),
                GraphKind::L1Pq => VectorGraph::L1PQ(restored),
                _ => VectorGraph::CosinePQ(restored),
            }
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
                    };
                    VectorGraph::$variant(Backend::restored(dump::read_dump::<f32, $dist>(
                        dir, &expected, $value,
                    )?))
                }};
            }
            match space {
                "l2" => raw!(GraphKind::L2, L2Dist, L2Dist {}, L2),
                "l1" => raw!(GraphKind::L1, L1Dist, L1Dist {}, L1),
                // `new_raw` also falls back to cosine on an unrecognised space,
                // so the two construction paths agree on what a bad space means.
                _ => raw!(GraphKind::Cosine, CosineDist, CosineDist {}, Cosine),
            }
        }
    };

    let nodes = graph.nb_points();
    Ok((graph, nodes))
}
