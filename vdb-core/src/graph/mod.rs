//! The boundary between ZeusDB and the vendored `hnsw_rs` graph.
//!
//! This module is the only place in the crate that names `hnsw_rs` outside a
//! test module. Everything the index does to the graph goes through
//! [`VectorGraph`], and everything the graph hands back arrives as a
//! [`GraphHit`], so the crate's own types stop here rather than travelling into
//! `hnsw_index.rs`.
//!
//! The on-disk format is [`dump`], which is ZeusDB's own. The vendored crate's
//! two file dump is no longer written or read.
//!
//! # What the seam does not cover
//!
//! One thing cannot be hidden, because ZeusDB implements it rather than calling
//! it. [`Distance`] is the vendored crate's trait, `CosineDist`, `L1Dist`,
//! `L2Dist` and `DistPQ` implement it, and a trait implemented for a foreign
//! crate has to be nameable where the implementation is written. It is
//! re-exported here so `distance.rs` and `hnsw_index.rs` import it from the
//! seam and one line changes if the graph is ever replaced. The coupling is
//! real and this re-export does not remove it.
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
use hnsw_rs::hnsw::Point;
use hnsw_rs::prelude::{FilterT, Hnsw};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

pub(crate) mod dump;

use dump::{Expected, GraphKind};

/// The vendored crate's distance trait, re-exported at the seam.
///
/// ZeusDB implements this for its own distance types, so the name has to be
/// visible where those implementations are written. Importing it from here
/// rather than from `hnsw_rs::prelude` keeps the crate's name in one file and
/// makes the set of implementors countable from this line.
pub(crate) use hnsw_rs::prelude::Distance;

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

/// Bytes an `Arc<T>` allocation carries beyond `T`, being the strong and the
/// weak count.
const ARC_COUNTS_BYTES: usize = 2 * std::mem::size_of::<usize>();

/// Bytes a `Vec<T>` header occupies, being a pointer, a capacity and a length.
const VEC_HEADER_BYTES: usize = 3 * std::mem::size_of::<usize>();

/// Bytes `parking_lot::RwLock<()>` occupies, being one `AtomicUsize`.
const PARKING_LOT_LOCK_BYTES: usize = std::mem::size_of::<usize>();

/// The capacity `Vec::push` gives a buffer it has just allocated for the first
/// time. `RawVec::MIN_NON_ZERO_CAP` is 4 for an element of 8 bytes.
const MIN_VEC_CAP: usize = 4;

/// Points whose neighbour lists the graph memory figure is measured over.
///
/// The adjacency count is a property of the data rather than of `m`, so it is
/// sampled rather than derived; see `graph_memory_bytes`. The sample is taken
/// by striding the point enumeration, which is insertion order within a layer,
/// because a prefix would be all early records and an early record has taken
/// more reverse links than a late one.
const GRAPH_SAMPLE_POINTS: usize = 4096;

/// Layer indices `graph_memory_bytes` asks the graph about.
///
/// The vendored crate fixes the layer count at `NB_LAYER_MAX`, which is 16 and
/// is `pub(crate)`, and `get_layer_nb_point` answers zero for an index it does
/// not have. Probing past the end therefore costs one lock and no correctness.
const GRAPH_LAYER_PROBE: usize = 32;

/// Layer `Vec` headers a point carries when nothing was sampled to count them.
///
/// Only reachable on a graph that reports points and holds none in any layer,
/// which no path produces. It is the crate's `NB_LAYER_MAX`.
const GRAPH_LAYERS_FALLBACK: usize = 16;

/// What the HNSW graph holds, in bytes it has asked the allocator for
///
/// `get_stats` used to report the storage maps and the two quantization tables
/// and stop there, which on a trained `quantized_only` index at 50,000 records
/// of dimension 1,536 named 9.77 MB against a measured 231 MiB resident. The
/// graph is the rest of it and this is what it holds.
///
/// # Per point
///
/// The graph owns a second copy of every point, separate from the storage map,
/// and it is `dim * 4` bytes in a raw graph and `subvectors` bytes in a
/// quantized one. That copy is one allocation. Around it the vendored crate
/// carries five more, all of them fixed and none of them proportional to the
/// dimension.
///
/// ```text
///   Arc<Point<T>>                              16 + size_of::<Point<T>>()
///   the point's own data vector                dim * 4, or subvectors
///   Arc<RwLock<Vec<Vec<Arc<PointWithOrder>>>>>  16 + 8 + 24
///   sixteen layer Vec headers                  16 * 24
///   its Arc slot in points_by_layer            8
/// ```
///
/// `Point` is 112 bytes on a 64 bit target, being a 24 byte `PointData` enum,
/// a `DataId`, a `PointId`, the `Arc` to the neighbour lists and a 64 byte
/// `[AtomicU32; 16]` of in-degree counters. `size_of` is taken rather than
/// written down. The sixteen layer headers are allocated for every point
/// whatever level it was drawn at, because `Point::new` fills the outer `Vec`
/// to `NB_LAYER_MAX` before it knows anything about the point.
///
/// # Per adjacency entry
///
/// Every entry in a neighbour list is an `Arc<PointWithOrder>`, which is 16
/// bytes of `Arc` counts around a pointer to the target and an `f32` distance,
/// and a pointer slot in the list itself.
///
/// **The number of entries is a property of the data and not of `m`.** Layer
/// zero caps a list at `2 * m` and the crate does fill it on data with no
/// structure, measured at exactly 32.000 entries per point at `m` 16 and
/// exactly 64.000 at `m` 32 over 40,000 uniform points on the sphere. Real
/// embeddings do not fill it, because `select_neighbours` prunes a candidate
/// that sits closer to an already chosen neighbour than to the query and
/// clustered data gives it far more to prune. The same measurement over 50,000
/// dbpedia-openai records at `m` 32 reads 29.95 at the full 1,536 dimensions
/// and 36.75 over their first 128, and over 10,000 of them at `m` 16 it reads
/// 24.81. **A count derived from `m` alone is 2.03 times the truth** at 50,000
/// records of dimension 1,536, being 3,401,398 entries against the 1,677,300
/// the saved graph dump holds. So the entry count is measured over
/// `GRAPH_SAMPLE_POINTS` points and scaled.
///
/// A list holds more slots than entries. It is filled once by `clone_from`,
/// which sizes it exactly, and grown afterwards by the reverse link updates,
/// which double it. `2 * len` is the capacity that produces, and it is what the
/// measurement on the uniform corpus asks for: at `m` 32 the live bytes exceed
/// a length based count by 506 per point where doubling a full layer zero list
/// predicts 512.
///
/// # What it does not cover
///
/// The allocator. Every block above carries a header and is rounded up, and the
/// process commits more than the sum of the blocks. Measured on this platform a
/// 32 byte request occupies 52 bytes of commit and a 512 byte request occupies
/// 551, and the whole graph commits between 1.4 and 1.7 times what this figure
/// names. That is allocator behaviour rather than something the graph holds,
/// and folding a platform factor into a reported number would state it as
/// though the structure carried it.
fn graph_memory_bytes<T, D>(hnsw: &Hnsw<'_, T, D>) -> usize
where
    T: Clone + Send + Sync + 'static,
    D: Distance<T> + Send + Sync,
{
    let indexation = hnsw.get_point_indexation();
    let nb_point = indexation.get_nb_point();
    if nb_point == 0 {
        return 0;
    }

    let element_bytes = indexation.get_data_dimension() * std::mem::size_of::<T>();
    let point_bytes = ARC_COUNTS_BYTES + std::mem::size_of::<Point<'static, T>>();
    let neighbour_cell_bytes = ARC_COUNTS_BYTES + PARKING_LOT_LOCK_BYTES + VEC_HEADER_BYTES;
    // `PointWithOrder` is a pointer to the target and an `f32` distance, and it
    // is padded to the pointer's alignment, so it is two words rather than one
    // and a half. It is `pub(crate)` in the vendored crate, so its size is
    // written out rather than taken.
    let entry_bytes = ARC_COUNTS_BYTES + 2 * std::mem::size_of::<usize>();
    let slot_bytes = std::mem::size_of::<usize>();

    // The adjacency, over a strided sample. `get_neighborhood_id` is the only
    // way out of the crate and it reallocates, so it is not called on every
    // point of a large graph. The stride runs across the concatenation of the
    // layers rather than within one, so the sample holds upper layer points in
    // the proportion the graph does, and a point at an upper layer carries more
    // adjacency than one at layer zero.
    //
    // One layer at a time, and never two iterators at once. Each iterator holds
    // a read guard on `points_by_layer` for its whole life, `parking_lot` does
    // not admit a recursive read while a writer is queued, and a concurrent
    // `add` queues exactly that writer. `get_layer_nb_point` takes the same
    // guard, so the counts are read before the first iterator exists.
    // Every layer is probed rather than stopping at the first empty one,
    // because a level is drawn independently per point and an empty layer below
    // an occupied one is legal. A layer index the graph does not have returns
    // zero rather than raising, so the probe is bounded by the crate's own
    // `NB_LAYER_MAX` without naming it.
    let layer_counts: Vec<usize> = (0..GRAPH_LAYER_PROBE)
        .map(|layer| indexation.get_layer_nb_point(layer))
        .collect();

    let stride = nb_point.div_ceil(GRAPH_SAMPLE_POINTS).max(1);
    let mut seen = 0usize;
    let mut sampled = 0usize;
    let mut adjacency = 0usize;
    let mut layers = 0usize;
    for (index, count) in layer_counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        for point in indexation.get_layer_iterator(index) {
            let take = seen.is_multiple_of(stride);
            seen += 1;
            if !take {
                continue;
            }
            let neighbourhood = point.get_neighborhood_id();
            layers = layers.max(neighbourhood.len());
            for list in &neighbourhood {
                if list.is_empty() {
                    continue;
                }
                let capacity = (2 * list.len()).max(MIN_VEC_CAP);
                adjacency += capacity * slot_bytes + list.len() * entry_bytes;
            }
            sampled += 1;
        }
    }

    if layers == 0 {
        layers = GRAPH_LAYERS_FALLBACK;
    }
    let fixed =
        point_bytes + element_bytes + neighbour_cell_bytes + layers * VEC_HEADER_BYTES + slot_bytes;
    let mut total = nb_point * fixed;
    if sampled > 0 {
        total += ((adjacency as f64 / sampled as f64) * nb_point as f64).round() as usize;
    }
    total
}

/// Turn a page of the crate's neighbours into ZeusDB's own hits.
///
/// One allocation of `k` entries per search, each entry a copy of two fields
/// out of three. It is the whole cost of the seam on the read path.
fn hits(neighbours: Vec<hnsw_rs::prelude::Neighbour>) -> Vec<GraphHit> {
    neighbours
        .into_iter()
        .map(|n| GraphHit {
            internal_id: n.d_id,
            distance: n.distance,
        })
        .collect()
}

/// The graph, in whichever of the six shapes this index built it
///
/// Three raw variants holding `f32` points and three quantized variants holding
/// `u8` codes. They differ in the distance they were built with, which the
/// vendored crate takes as a type parameter, so the enum is what stands in for
/// a single graph type.
pub(crate) enum VectorGraph {
    // Raw vector variants
    Cosine(Hnsw<'static, f32, CosineDist>),
    L2(Hnsw<'static, f32, L2Dist>),
    L1(Hnsw<'static, f32, L1Dist>),

    // PQ variants - corrected to use u8 element type
    CosinePQ(Hnsw<'static, u8, DistPQ>),
    L2PQ(Hnsw<'static, u8, DistPQ>),
    L1PQ(Hnsw<'static, u8, DistPQ>),
}

impl VectorGraph {
    pub(crate) fn new_raw(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "raw",
            "Creating raw HNSW index"
        );

        match space {
            "cosine" => VectorGraph::Cosine(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                CosineDist {},
            )),
            "l2" => VectorGraph::L2(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                L2Dist {},
            )),
            "l1" => VectorGraph::L1(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                L1Dist {},
            )),
            _ => {
                // âœ… ENTERPRISE: Replace panic with graceful error
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided"
                );
                // This is a programming error that should be caught earlier
                // For now, default to cosine to prevent panic
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                VectorGraph::Cosine(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    CosineDist {},
                ))
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
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "quantized",
            subvectors = pq.subvectors,
            bits = pq.bits,
            "Creating PQ-enabled HNSW index"
        );

        match space {
            "cosine" => {
                let dist_pq = DistPQ::new(pq);
                VectorGraph::CosinePQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l2" => {
                let dist_pq = DistPQ::new(pq);
                VectorGraph::L2PQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l1" => {
                let dist_pq = DistPQ::new(pq);
                VectorGraph::L1PQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            _ => {
                // âœ… ENTERPRISE: Replace panic with graceful error
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
                let dist_pq = DistPQ::new(pq);
                VectorGraph::CosinePQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
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
    /// The filter is a plain predicate over an internal id rather than the
    /// vendored crate's `FilterT`, which the seam satisfies on the caller's
    /// behalf. It is taken by generic reference rather than as a trait object
    /// so the closure the caller passes is monomorphised into the traversal
    /// exactly as it was before the seam existed.
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
        let filter = filter.map(|f| f as &dyn FilterT);
        match self {
            // Raw vector search
            VectorGraph::Cosine(hnsw) => Ok(hits(hnsw.search_filter(query, k, ef, filter))),
            VectorGraph::L2(hnsw) => Ok(hits(hnsw.search_filter(query, k, ef, filter))),
            VectorGraph::L1(hnsw) => Ok(hits(hnsw.search_filter(query, k, ef, filter))),

            // PQ-based search with ADC
            VectorGraph::CosinePQ(hnsw) | VectorGraph::L2PQ(hnsw) | VectorGraph::L1PQ(hnsw) => {
                // This query's ADC table, installed on this thread alone. The
                // guard is named so it lives to the end of the arm rather than
                // dropping at the end of the statement, and it releases the
                // table once the traversal is done.
                let _query_lut = hnsw.get_distance().install_query_lut(query)?;

                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; self.code_size()];

                Ok(hits(hnsw.search_filter(&dummy_query, k, ef, filter)))
            }
        }
    }

    /// Number of nodes the graph holds, which is the number of insertions it has
    /// taken. It exceeds the live record count by exactly the number of nodes
    /// that removal and overwrite have stranded.
    pub(crate) fn nb_points(&self) -> usize {
        match self {
            VectorGraph::Cosine(hnsw) => hnsw.get_nb_point(),
            VectorGraph::L2(hnsw) => hnsw.get_nb_point(),
            VectorGraph::L1(hnsw) => hnsw.get_nb_point(),
            VectorGraph::CosinePQ(hnsw) => hnsw.get_nb_point(),
            VectorGraph::L2PQ(hnsw) => hnsw.get_nb_point(),
            VectorGraph::L1PQ(hnsw) => hnsw.get_nb_point(),
        }
    }

    /// Bytes one stored code occupies, and zero on a raw graph. Internal to the
    /// seam, since the only caller is the dummy query `search` builds.
    fn code_size(&self) -> usize {
        match self {
            VectorGraph::CosinePQ(hnsw) => hnsw.get_distance().subvectors(),
            VectorGraph::L2PQ(hnsw) => hnsw.get_distance().subvectors(),
            VectorGraph::L1PQ(hnsw) => hnsw.get_distance().subvectors(),
            _ => 0,
        }
    }

    /// Bytes the graph asks the allocator for. See `graph_memory_bytes`.
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            VectorGraph::Cosine(hnsw) => graph_memory_bytes(hnsw),
            VectorGraph::L2(hnsw) => graph_memory_bytes(hnsw),
            VectorGraph::L1(hnsw) => graph_memory_bytes(hnsw),
            VectorGraph::CosinePQ(hnsw) => graph_memory_bytes(hnsw),
            VectorGraph::L2PQ(hnsw) => graph_memory_bytes(hnsw),
            VectorGraph::L1PQ(hnsw) => graph_memory_bytes(hnsw),
        }
    }

    pub(crate) fn is_quantized(&self) -> bool {
        matches!(
            self,
            VectorGraph::CosinePQ(_) | VectorGraph::L2PQ(_) | VectorGraph::L1PQ(_)
        )
    }

    pub(crate) fn insert(&self, vector: &[f32], id: usize) {
        match self {
            VectorGraph::Cosine(hnsw) => hnsw.insert((vector, id)),
            VectorGraph::L2(hnsw) => hnsw.insert((vector, id)),
            VectorGraph::L1(hnsw) => hnsw.insert((vector, id)),
            _ => {
                // âœ… ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_raw_vectors_into_pq_index",
                    "Cannot insert raw vectors into PQ index"
                );
            }
        }
    }

    /// Insert PQ codes into the index
    pub(crate) fn insert_pq_codes(&self, codes: &[u8], id: usize) {
        match self {
            VectorGraph::CosinePQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            VectorGraph::L2PQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            VectorGraph::L1PQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            _ => {
                // âœ… ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "pq_codes_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_pq_codes_into_raw_index",
                    "Cannot insert PQ codes into raw index"
                );
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn insert_batch(&self, data: &[(&Vec<f32>, usize)]) {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting batch insertion"
        );

        if data.len() >= threshold {
            match self {
                VectorGraph::Cosine(hnsw) => hnsw.parallel_insert(data),
                VectorGraph::L2(hnsw) => hnsw.parallel_insert(data),
                VectorGraph::L1(hnsw) => hnsw.parallel_insert(data),
                _ => {
                    // âœ… ENTERPRISE: Replace panic with graceful error
                    error!(
                        operation = "batch_insert",
                        error = "invalid_operation",
                        reason = "cannot_batch_insert_raw_vectors_into_pq_index",
                        "Cannot batch insert raw vectors into PQ index"
                    );
                }
            }
        } else {
            for (vector, id) in data {
                self.insert(vector.as_slice(), *id);
            }
        }
    }

    /// Insertion is sequential whatever the batch size. Every caller is a
    /// one-time structural rebuild, at training completion, in `compact` or in
    /// the persistence loader, and each of them sorts its batch by internal id
    /// so that two rebuilds of the same records wire the same graph. Above
    /// 1,000 insertions per thread this used to fork to `parallel_insert`,
    /// which draws levels from the shared seeded generator in thread arrival
    /// order and interleaves the neighbour list updates, so the sort bought
    /// nothing at exactly the sizes where reproducibility costs the most.
    pub(crate) fn insert_batch_pq(&self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        debug!(
            operation = "batch_insert_pq",
            batch_size = data.len(),
            "Starting PQ batch insertion"
        );

        match self {
            VectorGraph::CosinePQ(hnsw) | VectorGraph::L2PQ(hnsw) | VectorGraph::L1PQ(hnsw) => {
                for (codes, id) in data {
                    hnsw.insert((codes.as_slice(), *id));
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
            VectorGraph::Cosine(hnsw) => dump::write_dump(hnsw, kind, dir),
            VectorGraph::L2(hnsw) => dump::write_dump(hnsw, kind, dir),
            VectorGraph::L1(hnsw) => dump::write_dump(hnsw, kind, dir),
            VectorGraph::CosinePQ(hnsw) => dump::write_dump(hnsw, kind, dir),
            VectorGraph::L2PQ(hnsw) => dump::write_dump(hnsw, kind, dir),
            VectorGraph::L1PQ(hnsw) => dump::write_dump(hnsw, kind, dir),
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
/// true where `Hnsw::new` sets it false. `Hnsw::from_loaded_points` sets the
/// insertion flags to what `new` sets, so a restored graph and a fresh one
/// insert alike without anything being put back afterwards.
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
                dimension: pq.subvectors,
                m,
                ef_construction,
                min_nodes,
            };
            let hnsw = dump::read_dump::<u8, DistPQ>(dir, &expected, DistPQ::new(pq))?;
            match kind {
                GraphKind::L2Pq => VectorGraph::L2PQ(hnsw),
                GraphKind::L1Pq => VectorGraph::L1PQ(hnsw),
                _ => VectorGraph::CosinePQ(hnsw),
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
                    VectorGraph::$variant(dump::read_dump::<f32, $dist>(dir, &expected, $value)?)
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
