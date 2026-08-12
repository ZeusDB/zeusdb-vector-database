//! The boundary between ZeusDB and the vendored `hnsw_rs` graph.
//!
//! This is the only file in the crate that names `hnsw_rs` outside a test
//! module. Everything the index does to the graph goes through [`VectorGraph`],
//! and everything the graph hands back arrives as a [`GraphHit`], so the crate's
//! own types stop here rather than travelling into `hnsw_index.rs`.
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
//! # The distance types are pinned to their modules
//!
//! `Hnsw::file_dump` writes `std::any::type_name::<D>()` into the dump header
//! and [`inspect_graph_dump`] compares the saved string against the same call.
//! `type_name` is a full module path, so moving `CosineDist`, `L1Dist`,
//! `L2Dist` or `DistPQ` to another module changes what a save writes and stops
//! every previously saved index from loading. That is why `DistPQ` stays in
//! `hnsw_index.rs` and the three raw distances stay in `distance.rs` even
//! though the graph is what uses them.

use crate::distance::{CosineDist, L1Dist, L2Dist};
use crate::hnsw_index::DistPQ;
use crate::pq::PQ;
use hnsw_rs::api::AnnT; // This provides the file_dump method
use hnsw_rs::hnsw::Point;
use hnsw_rs::prelude::{FilterT, Hnsw};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

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
                // ✅ ENTERPRISE: Replace panic with graceful error
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
                // ✅ ENTERPRISE: Replace panic with graceful error
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
                // ✅ ENTERPRISE: Replace panic with graceful error logging
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
                // ✅ ENTERPRISE: Replace panic with graceful error logging
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
                    // ✅ ENTERPRISE: Replace panic with graceful error
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

    pub(crate) fn insert_batch_pq(&self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert_pq",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting PQ batch insertion"
        );

        match self {
            VectorGraph::CosinePQ(hnsw) | VectorGraph::L2PQ(hnsw) | VectorGraph::L1PQ(hnsw) => {
                if data.len() >= threshold {
                    hnsw.parallel_insert(data);
                } else {
                    for (codes, id) in data {
                        hnsw.insert((codes.as_slice(), *id));
                    }
                }

                Ok(())
            }
            _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
        }
    }

    /// Match a freshly constructed graph's insertion settings.
    ///
    /// `Hnsw::new` starts with `extend_candidates` false and the vendored
    /// reload sets it true, so a restored graph would build the neighbourhood
    /// of every record added after the load differently from the same record
    /// added before the save. Nothing else `load_hnsw_with_dist` fills in
    /// differs from what `new` sets.
    fn settle_after_reload(&mut self) {
        match self {
            VectorGraph::Cosine(hnsw) => hnsw.set_extend_candidates(false),
            VectorGraph::L2(hnsw) => hnsw.set_extend_candidates(false),
            VectorGraph::L1(hnsw) => hnsw.set_extend_candidates(false),
            VectorGraph::CosinePQ(hnsw) => hnsw.set_extend_candidates(false),
            VectorGraph::L2PQ(hnsw) => hnsw.set_extend_candidates(false),
            VectorGraph::L1PQ(hnsw) => hnsw.set_extend_candidates(false),
        }
    }

    /// Write the graph to `dir` under the basename the loader reads back.
    ///
    /// Returns the basename the dump was written under, which the vendored
    /// crate reports and which is not always the one asked for: it appends a
    /// suffix rather than overwriting when a memory mapped data file is active.
    /// See `reload_graph` for why this index never reaches that path.
    pub(crate) fn dump(&self, dir: &Path) -> Result<String, String> {
        let dumped = match self {
            VectorGraph::Cosine(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine",
                    "Using Cosine distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
            VectorGraph::L2(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2",
                    "Using L2 distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
            VectorGraph::L1(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1",
                    "Using L1 distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
            VectorGraph::CosinePQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine_pq",
                    "Using Cosine-PQ distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
            VectorGraph::L2PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2_pq",
                    "Using L2-PQ distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
            VectorGraph::L1PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1_pq",
                    "Using L1-PQ distance HNSW"
                );
                hnsw.file_dump(dir, HNSW_DUMP_BASENAME)
            }
        };

        dumped.map_err(|e| e.to_string())
    }
}

// ============================================================================
// RESTORING THE SAVED GRAPH
// ============================================================================

/// Basename `VectorGraph::dump` writes under and the loader reads back.
///
/// Private to the seam. The save path used to pass the literal in from
/// `hnsw_index.rs` while the load path read this constant, so the two halves of
/// the on-disk contract were written in two places. They are one now.
const HNSW_DUMP_BASENAME: &str = "hnsw_index";

/// Header of the dumped data file, being one magic and the data dimension.
const DUMP_DATA_HEADER_BYTES: usize = 4 + 8;

/// What the dumped data file spends per point before the vector itself, being
/// one magic, the origin id and the serialized byte length.
const DUMP_DATA_POINT_BYTES: usize = 4 + 8 + 8;

/// Layers the vendored crate always dumps, being its `NB_LAYER_MAX`.
const DUMP_LAYERS: u8 = 16;

/// Read the dump's own description and judge it against what this index expects
///
/// Everything here runs before the vendored reload is entered, because that
/// reload reaches `std::process::exit(1)` when the data file is short. The data
/// file's length is fully determined by the point count and the dimension, so
/// an exact size comparison closes that path. Every other malformed dump the
/// vendored reader meets raises a panic it can unwind from, which the caller
/// catches.
///
/// Returns the node count the dump declares.
#[allow(clippy::too_many_arguments)]
fn inspect_graph_dump(
    dir: &Path,
    dimension: usize,
    element_bytes: usize,
    t_name: &str,
    dist_name: &str,
    m: usize,
    ef_construction: usize,
    min_nodes: usize,
) -> Result<usize, String> {
    let graph_path = dir.join(format!("{}.hnsw.graph", HNSW_DUMP_BASENAME));
    let data_path = dir.join(format!("{}.hnsw.data", HNSW_DUMP_BASENAME));

    if !graph_path.exists() || !data_path.exists() {
        return Err("the directory holds no HNSW graph dump".to_string());
    }

    let file = std::fs::File::open(&graph_path)
        .map_err(|e| format!("the graph dump could not be opened: {}", e))?;
    let mut reader = std::io::BufReader::new(file);

    // `load_description` unwraps a UTF-8 conversion on the two names it reads,
    // so a dump whose header is garbage panics here rather than returning.
    let described = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_rs::hnswio::load_description(&mut reader)
    }));
    let descr = match described {
        Ok(Ok(descr)) => descr,
        Ok(Err(e)) => return Err(format!("the graph dump has no readable header: {}", e)),
        Err(_) => return Err("the graph dump has an unreadable header".to_string()),
    };

    if descr.t_name != t_name {
        return Err(format!(
            "the dump stores {} points where this index holds {}",
            descr.t_name, t_name
        ));
    }
    if descr.distname != dist_name {
        return Err(format!(
            "the dump was written under distance {} and this build uses {}",
            descr.distname, dist_name
        ));
    }
    if descr.dimension != dimension {
        return Err(format!(
            "the dump stores {} values per point where this index expects {}",
            descr.dimension, dimension
        ));
    }
    if descr.nb_layer != DUMP_LAYERS {
        return Err(format!(
            "the dump declares {} layers where this build uses {}",
            descr.nb_layer, DUMP_LAYERS
        ));
    }
    if descr.max_nb_connection as usize != m {
        return Err(format!(
            "the dump was written at m {} and config.json declares {}",
            descr.max_nb_connection, m
        ));
    }
    if descr.ef != ef_construction {
        return Err(format!(
            "the dump was written at ef_construction {} and config.json declares {}",
            descr.ef, ef_construction
        ));
    }
    if descr.nb_point < min_nodes {
        return Err(format!(
            "the dump holds {} graph nodes and the index holds {} records",
            descr.nb_point, min_nodes
        ));
    }

    let expected = DUMP_DATA_HEADER_BYTES
        + descr
            .nb_point
            .saturating_mul(DUMP_DATA_POINT_BYTES + dimension * element_bytes);
    let actual = std::fs::metadata(&data_path)
        .map_err(|e| format!("the data dump could not be measured: {}", e))?
        .len();
    if actual != expected as u64 {
        return Err(format!(
            "the data dump is {} bytes where {} nodes of {} values need {}",
            actual, descr.nb_point, dimension, expected
        ));
    }

    Ok(descr.nb_point)
}

/// Reload one graph of a known element type and distance
///
/// `load_hnsw_with_dist` rather than `load_hnsw`, for two reasons. It takes the
/// distance by value, which is the only way to restore a PQ graph, since
/// `DistPQ` carries the codebook and cannot be produced by `Default`. And it
/// leaves `datamap_opt` false, where `load_hnsw` sets it true and a later
/// `file_dump` then refuses to overwrite its own files and writes
/// `hnsw_index-4173.hnsw.graph` beside them instead. Measured on the vendored
/// crate.
///
/// The reader is leaked because the vendored signature ties the returned graph's
/// lifetime to it, so that a graph reading a memory mapped data file cannot
/// outlive the mapping. Nothing is mapped here, since neither the default
/// options nor this entry point ever construct a `DataMap`, so the leak is 280
/// bytes per successful load and holds no file open.
fn reload_graph<T, D>(dir: &Path, dist: D) -> Result<Hnsw<'static, T, D>, String>
where
    T: 'static
        + Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Sized
        + Send
        + Sync
        + std::fmt::Debug,
    D: Distance<T> + Send + Sync,
{
    let reader = Box::leak(Box::new(hnsw_rs::hnswio::HnswIo::new(
        dir,
        HNSW_DUMP_BASENAME,
    )));

    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader.load_hnsw_with_dist::<T, D>(dist)
    }));

    match loaded {
        Ok(Ok(hnsw)) => Ok(hnsw),
        Ok(Err(e)) => Err(format!("the graph dump could not be read: {}", e)),
        Err(_) => Err("the graph dump is malformed and reading it panicked".to_string()),
    }
}

/// Restore the saved graph for one index configuration
///
/// `pq` present means the saved graph was a quantized one, which is exactly the
/// condition the loader branches on, since training is what replaces the raw
/// graph with a PQ graph. A directory whose dump disagrees is caught by the
/// element type and distance name in `inspect_graph_dump` and falls back.
pub(crate) fn restore_graph(
    dir: &Path,
    space: &str,
    m: usize,
    ef_construction: usize,
    dim: usize,
    pq: Option<Arc<PQ>>,
    min_nodes: usize,
) -> Result<(VectorGraph, usize), String> {
    let mut graph = match pq {
        Some(pq) => {
            let nodes = inspect_graph_dump(
                dir,
                pq.subvectors,
                std::mem::size_of::<u8>(),
                std::any::type_name::<u8>(),
                std::any::type_name::<DistPQ>(),
                m,
                ef_construction,
                min_nodes,
            )?;
            let hnsw = reload_graph::<u8, DistPQ>(dir, DistPQ::new(pq))?;
            let restored = hnsw.get_nb_point();
            if restored != nodes {
                return Err(format!(
                    "the dump declares {} graph nodes and yielded {}",
                    nodes, restored
                ));
            }
            match space {
                "l2" => VectorGraph::L2PQ(hnsw),
                "l1" => VectorGraph::L1PQ(hnsw),
                _ => VectorGraph::CosinePQ(hnsw),
            }
        }
        None => {
            // The raw graphs differ only in their distance type, so each arm
            // states the name the dump must carry and the value the reload
            // needs, and nothing else about them differs.
            macro_rules! raw {
                ($dist:ty, $value:expr, $variant:ident) => {{
                    let nodes = inspect_graph_dump(
                        dir,
                        dim,
                        std::mem::size_of::<f32>(),
                        std::any::type_name::<f32>(),
                        std::any::type_name::<$dist>(),
                        m,
                        ef_construction,
                        min_nodes,
                    )?;
                    let hnsw = reload_graph::<f32, $dist>(dir, $value)?;
                    let restored = hnsw.get_nb_point();
                    if restored != nodes {
                        return Err(format!(
                            "the dump declares {} graph nodes and yielded {}",
                            nodes, restored
                        ));
                    }
                    VectorGraph::$variant(hnsw)
                }};
            }
            match space {
                "l2" => raw!(L2Dist, L2Dist {}, L2),
                "l1" => raw!(L1Dist, L1Dist {}, L1),
                // `new_raw` also falls back to cosine on an unrecognised space,
                // so the two construction paths agree on what a bad space means.
                _ => raw!(CosineDist, CosineDist {}, Cosine),
            }
        }
    };

    graph.settle_after_reload();
    let nodes = graph.nb_points();
    Ok((graph, nodes))
}
