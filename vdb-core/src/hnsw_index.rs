use crate::distance::{CosineDist, L1Dist, L2Dist};
use chrono::Utc;
use hnsw_rs::api::AnnT; // This provides the file_dump method
use hnsw_rs::prelude::{Distance, FilterT, Hnsw};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
// Aliased because `std::sync::atomic::Ordering` already holds the bare name.
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ✅ ENTERPRISE: Structured logging imports
use tracing::{debug, error, info, instrument, trace, warn};

// Import PQ module
use crate::pq::PQ;

/// Largest `expected_size` a caller may declare.
///
/// `expected_size` reaches `PointIndexation::new` in the vendored graph crate,
/// which reserves capacity for it across the 16 layers at creation. Since the
/// layer reservation was corrected the reservation is one `Arc` slot per
/// declared record, measured at 8.02 bytes per declared record and flat across
/// declarations of 10 million through 4 billion. This bound therefore caps the
/// creation-time reservation at 764 MB, which was measured rather than derived.
///
/// The bound exists because the reservation is not fallible. `Vec::with_capacity`
/// aborts the process on allocation failure rather than unwinding, so a
/// declaration too large for the machine cannot be turned into a Python
/// exception after the fact. A declared 20 billion asks for 155 GB in the layer
/// zero reservation alone and the process dies with no traceback. Making that
/// path fallible means `try_reserve` inside the vendored crate, which is not a
/// change this package makes.
///
/// One hundred million is far above anything this index holds. A real 100,000
/// record build at 768 dimensions measured 10,617 bytes per record, so a hundred
/// million records is roughly a terabyte of process memory for the data alone.
/// Declaring less than the truth is safe, because a layer that receives more
/// points than reserved grows through the ordinary `Vec::push` path, so capping
/// the declaration costs a caller nothing it can observe.
///
/// The bound is not a guarantee. A machine whose commit limit is below the
/// reservation can still abort at a declaration under it.
const MAX_EXPECTED_SIZE: usize = 100_000_000;

/// Multiple of `expected_size` at which an index warns that it has outgrown its
/// declaration. Fires once per index.
const EXPECTED_SIZE_OVERGROWTH_FACTOR: usize = 2;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageMode {
    #[default]
    #[serde(rename = "quantized_only")]
    QuantizedOnly,

    #[serde(rename = "quantized_with_raw")]
    QuantizedWithRaw,
}

impl StorageMode {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s {
            "quantized_only" => Ok(StorageMode::QuantizedOnly),
            "quantized_with_raw" => Ok(StorageMode::QuantizedWithRaw),
            _ => Err(format!(
                "Invalid storage_mode: '{}'. Supported: quantized_only, quantized_with_raw",
                s
            )),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            StorageMode::QuantizedOnly => "quantized_only",
            StorageMode::QuantizedWithRaw => "quantized_with_raw",
        }
    }
}

// Updated QuantizationConfig structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: StorageMode,
}

/// Candidates pulled from the graph per requested result when a quantized
/// search reranks and the caller named no factor of its own.
///
/// Measured at 10,000 records of dimension 768, 48 subvectors at 8 bits, over
/// two data draws of 200 held out queries. Recall at `top_k` 10 runs 0.105 and
/// 0.115 with no rerank, 0.404 and 0.415 at factor 5, 0.690 and 0.693 at 10,
/// 0.889 at 15, 0.995 on both draws at 20, and 1.000 from 25 upward, against
/// 1.000 for the raw path on the same data. Latency at factor 20 is 0.76 and
/// 0.67 milliseconds against the raw path's 0.68, rising to 0.98 at factor 25
/// and 1.7 at 50 for at most five thousandths more recall.
///
/// The curve turns here. Twenty is the last point where recall is still buying
/// its latency.
///
/// It is a factor rather than a candidate count, so a large `top_k` multiplies
/// it. The candidate count that saturates recall is close to 200 whatever the
/// page size, so `top_k` 100 reaches 0.986 at factor 2 and 1.000 at factor 5,
/// and paying factor 20 there costs 7.4 milliseconds for nothing. A caller
/// asking for a large page should lower it.
pub const DEFAULT_RERANK_FACTOR: usize = 20;

/// The raw vector distance for a space
///
/// These are the same `crate::distance` implementations `DistanceType::new_raw`
/// hands to a raw graph, so a rescored score is the number a raw index would
/// have reported for the same pair rather than a second implementation of the
/// same formula.
fn raw_distance_fn(space: &str) -> fn(&[f32], &[f32]) -> f32 {
    match space {
        "l2" => |a: &[f32], b: &[f32]| L2Dist {}.eval(a, b),
        "l1" => |a: &[f32], b: &[f32]| L1Dist {}.eval(a, b),
        _ => |a: &[f32], b: &[f32]| CosineDist {}.eval(a, b),
    }
}

/// How a quantized search over-fetches and rescores
///
/// Present only when the index is quantized, its storage mode keeps raw
/// vectors, and the caller has not turned rerank off. `HNSWIndex::rerank_plan`
/// is the single place that decides, and all three search paths take it from
/// there.
#[derive(Clone, Copy)]
struct RerankPlan {
    /// Candidates to pull from the graph per requested result.
    factor: usize,
    /// The space's raw vector distance.
    distance: fn(&[f32], &[f32]) -> f32,
}

/// The settings a search carries once its input has been parsed
///
/// Bundled rather than threaded through one at a time, because the three batch
/// entry points forward every one of them unchanged and the page size, the
/// traversal breadth and the rerank plan are read together wherever they are
/// read at all.
#[derive(Clone, Copy)]
struct SearchParams {
    top_k: usize,
    ef: usize,
    return_vector: bool,
    rerank: Option<RerankPlan>,
}

impl SearchParams {
    /// Candidates to ask the graph for
    ///
    /// The requested page unless the page is going to be reordered, and the
    /// over-fetch capped at the live record count when it is. The cap means a
    /// large factor degrades to a full scan rather than to an allocation the
    /// size of the factor.
    fn fetch_k(&self, live_records: usize) -> usize {
        match self.rerank {
            Some(plan) => self
                .top_k
                .saturating_mul(plan.factor)
                .min(live_records.max(self.top_k)),
            None => self.top_k,
        }
    }
}

/// Score one candidate against the query on the raw vector scale
///
/// The stored raw vector where the index holds one, which under
/// `quantized_with_raw` is every record, and the reconstruction from its codes
/// otherwise. Both are vectors of the index's own dimension, so a page mixing
/// them is still ordered by one distance.
///
/// `None` means the candidate holds neither a raw vector nor codes, which no
/// record resolving through `rev_map` can be. The callers sort such a candidate
/// last rather than letting an unscored one displace a scored one.
fn rescore_candidate(
    plan: &RerankPlan,
    query: &[f32],
    ext_id: &str,
    vectors: &HashMap<String, Vec<f32>>,
    pq: Option<&Arc<PQ>>,
    pq_codes: &HashMap<String, Vec<u8>>,
) -> Option<f32> {
    if let Some(stored) = vectors.get(ext_id) {
        return Some((plan.distance)(query, stored));
    }
    let reconstructed = pq?.reconstruct(pq_codes.get(ext_id)?).ok()?;
    Some((plan.distance)(query, &reconstructed))
}

/// Order a rescored page and cut it to the requested size
///
/// `total_cmp` rather than `partial_cmp`, so a non-finite score orders rather
/// than panicking the sort.
fn take_best<T>(scored: &mut Vec<(T, f32)>, top_k: usize) {
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(top_k);
}

thread_local! {
    /// The ADC lookup table for the query the calling thread is running.
    ///
    /// The table belongs to a query, not to an index. It used to live on
    /// `DistPQ`, one per index, which meant two searches overlapping would each
    /// overwrite the other's table and score candidates against a query they
    /// were never given. An exclusive lock on the graph was the only thing
    /// preventing that, so the table had to move before the lock could be
    /// relaxed.
    ///
    /// `Distance::eval` takes `&self` and has no parameter to carry per query
    /// state, so the table cannot be threaded through as an argument. Thread
    /// local storage is the way to give it to `eval` without giving it to the
    /// index, and it needs no change to the vendored crate.
    ///
    /// The invariant this rests on is that one query's traversal runs entirely
    /// on the thread that installed its table. `Hnsw::search` is sequential
    /// within a single query, so that holds. `batch_search_parallel` splits
    /// across queries rather than within one, and each query installs its own
    /// table on the worker that runs it. Adopting `Hnsw::parallel_search`, which
    /// would fan one query's distance evaluations out across the pool, would
    /// break this and must not be done without replacing the mechanism.
    static QUERY_LUT: RefCell<Option<Vec<Vec<f32>>>> = const { RefCell::new(None) };
}

/// Holds a query's ADC table on the calling thread and removes it on drop.
///
/// Drop rather than an explicit clear, so an early return or a panic inside the
/// traversal cannot leave a stale table behind for the next query this thread
/// runs. A leftover table would be read as if it belonged to that next query.
struct QueryLut;

impl Drop for QueryLut {
    fn drop(&mut self) {
        QUERY_LUT.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Custom distance function for Product Quantization using ADC
#[derive(Clone)]
pub struct DistPQ {
    /// Reference to the PQ instance for accessing centroids
    pq: Arc<PQ>,
}

impl DistPQ {
    pub fn new(pq: Arc<PQ>) -> Self {
        DistPQ { pq }
    }

    /// Compute this query's ADC table and install it for the calling thread.
    ///
    /// The returned guard must be held for the whole traversal. Dropping it
    /// early returns the thread to graph construction mode, where `eval` reads
    /// the codebook's symmetric table instead.
    fn install_query_lut(&self, query: &[f32]) -> Result<QueryLut, String> {
        if !self.pq.is_trained() {
            return Err("PQ must be trained before ADC computation".to_string());
        }

        let lut = self.pq.compute_adc_lut(query)?;
        QUERY_LUT.with(|slot| *slot.borrow_mut() = Some(lut));
        Ok(QueryLut)
    }
}

impl Distance<u8> for DistPQ {
    /// Distance between two points the graph holds, both of which are PQ codes
    ///
    /// A query table on this thread means a search is running. `a` is then the
    /// dummy code vector `DistanceType::search` passes, the real query lives in
    /// the table, and the distance is asymmetric: query subvector against stored
    /// centroid.
    ///
    /// No query table means graph construction, where there is no query and
    /// both `a` and `b` are stored codes. The distance is then symmetric,
    /// centroid against centroid, read from the table the codebook carries.
    /// Returning infinity here, which is what this did until the symmetric
    /// table existed, made every candidate tie in the neighbour selection
    /// heuristic and left the graph with one edge per node.
    ///
    /// Both branches return a sum of squared L2 distances, so they are on the
    /// same scale and neither takes a square root.
    ///
    /// Choosing the branch on the table rather than on `a` is deliberate. The
    /// dummy query is a valid code slice and cannot be told apart from real
    /// codes by inspection. It is sound because the table is thread local, so an
    /// insertion can never observe a query table it did not install itself, no
    /// matter what any other thread is doing at the time. That used to depend on
    /// the graph mutex serialising searches against insertions, which is the
    /// dependency this removes.
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        QUERY_LUT.with(|slot| {
            let slot = slot.borrow();
            let Some(lut) = slot.as_ref() else {
                return self.pq.symmetric_distance(a, b);
            };

            // b.len() should equal pq.subvectors
            let mut sum = 0.0f32;
            for (sv, &code) in b.iter().enumerate() {
                // lut[sv][code]
                let distance_component = lut
                    .get(sv)
                    .and_then(|row| row.get(code as usize))
                    .copied()
                    .unwrap_or(f32::INFINITY);
                sum += distance_component;
            }
            sum
        })
    }
}

// Enhanced DistanceType enum to support PQ variants
enum DistanceType {
    // Raw vector variants
    Cosine(Hnsw<'static, f32, CosineDist>),
    L2(Hnsw<'static, f32, L2Dist>),
    L1(Hnsw<'static, f32, L1Dist>),

    // PQ variants - corrected to use u8 element type
    CosinePQ(Hnsw<'static, u8, DistPQ>),
    L2PQ(Hnsw<'static, u8, DistPQ>),
    L1PQ(Hnsw<'static, u8, DistPQ>),
}

impl DistanceType {
    fn new_raw(
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
            "cosine" => DistanceType::Cosine(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                CosineDist {},
            )),
            "l2" => DistanceType::L2(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                L2Dist {},
            )),
            "l1" => DistanceType::L1(Hnsw::new(
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
                DistanceType::Cosine(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    CosineDist {},
                ))
            }
        }
    }

    fn new_pq(
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
                DistanceType::CosinePQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l2" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L2PQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l1" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L1PQ(Hnsw::new(
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
                DistanceType::CosinePQ(Hnsw::new(
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
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn FilterT>,
    ) -> Result<Vec<hnsw_rs::prelude::Neighbour>, String> {
        match self {
            // Raw vector search
            DistanceType::Cosine(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),
            DistanceType::L2(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),
            DistanceType::L1(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),

            // PQ-based search with ADC
            DistanceType::CosinePQ(hnsw) | DistanceType::L2PQ(hnsw) | DistanceType::L1PQ(hnsw) => {
                // This query's ADC table, installed on this thread alone. The
                // guard is named so it lives to the end of the arm rather than
                // dropping at the end of the statement, and it releases the
                // table once the traversal is done.
                let _query_lut = hnsw.get_distance().install_query_lut(query)?;

                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; self.get_code_size()];

                Ok(hnsw.search_filter(&dummy_query, k, ef, filter))
            }
        }
    }

    /// Number of nodes the graph holds, which is the number of insertions it has
    /// taken. It exceeds the live record count by exactly the number of nodes
    /// that removal and overwrite have stranded.
    fn nb_points(&self) -> usize {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.get_nb_point(),
            DistanceType::L2(hnsw) => hnsw.get_nb_point(),
            DistanceType::L1(hnsw) => hnsw.get_nb_point(),
            DistanceType::CosinePQ(hnsw) => hnsw.get_nb_point(),
            DistanceType::L2PQ(hnsw) => hnsw.get_nb_point(),
            DistanceType::L1PQ(hnsw) => hnsw.get_nb_point(),
        }
    }

    fn get_code_size(&self) -> usize {
        match self {
            DistanceType::CosinePQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L2PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L1PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            _ => 0,
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(
            self,
            DistanceType::CosinePQ(_) | DistanceType::L2PQ(_) | DistanceType::L1PQ(_)
        )
    }

    fn insert(&self, vector: &[f32], id: usize) {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L2(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L1(hnsw) => hnsw.insert((vector, id)),
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
    fn insert_pq_codes(&self, codes: &[u8], id: usize) {
        match self {
            DistanceType::CosinePQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            DistanceType::L2PQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            DistanceType::L1PQ(hnsw) => {
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
    fn insert_batch(&self, data: &[(&Vec<f32>, usize)]) {
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
                DistanceType::Cosine(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L2(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L1(hnsw) => hnsw.parallel_insert(data),
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

    fn insert_batch_pq(&self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
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
            DistanceType::CosinePQ(hnsw) | DistanceType::L2PQ(hnsw) | DistanceType::L1PQ(hnsw) => {
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
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AddResult {
    #[pyo3(get)]
    pub total_inserted: usize,
    #[pyo3(get)]
    pub total_errors: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub vector_shape: Option<(usize, usize)>,
}

#[pymethods]
impl AddResult {
    fn __repr__(&self) -> String {
        format!(
            "AddResult(inserted={}, errors={}, shape={:?})",
            self.total_inserted, self.total_errors, self.vector_shape
        )
    }

    pub fn is_success(&self) -> bool {
        self.total_errors == 0
    }

    /// One line human-readable summary of the insertion
    ///
    /// ASCII only, deliberately. This used to open with a check mark and carry a
    /// cross before the error count, and it is the first thing the README and the
    /// documentation site tell a new user to print. `print()` encodes through the
    /// console's code page, so on a Windows console still using the legacy one
    /// that first statement raised `UnicodeEncodeError` before the reader had
    /// added a second record.
    ///
    /// The counts stay available as `total_inserted` and `total_errors`, so the
    /// alternative of returning them as structured data would only duplicate two
    /// attributes that already exist while breaking every caller that prints
    /// this. The numbers and the words around them are unchanged, so a substring
    /// test or a `(\d+) inserted` match still holds. What no longer holds is a
    /// parse keyed on the emoji themselves or on a fixed character offset.
    pub fn summary(&self) -> String {
        format!(
            "{} inserted, {} errors",
            self.total_inserted, self.total_errors
        )
    }
}

/// Lock acquisition order for `HNSWIndex`
///
/// Every path that holds two of these guards at once acquires them in this
/// order, top to bottom. Releasing may happen in any order.
///
/// ```text
/// id_map < rev_map < hnsw < vectors < pq_codes < vector_metadata
///        < training_ids < metadata < id_counter < vector_count
/// ```
///
/// This exists because search and mutation now overlap. Until the receivers
/// were relaxed, PyO3's exclusive borrow kept every mutating method away from
/// every search, so no reader and no writer were ever in flight together and
/// the acquisition order could not matter. It matters now. A search holds
/// `rev_map` for its whole traversal and takes `vectors` afterwards, so a
/// removal taking `vectors` before `rev_map`, which is what it used to do,
/// deadlocks against it on the first interleaving that lands.
///
/// One further rule, which the order alone does not express. No path forks to
/// rayon while holding a write guard. Mutations are serialised against each
/// other by `writers`, so a read guard held across a fork can only ever be
/// blocked by that one writer, and a fork under a write guard is exactly the
/// case where the pool's workers can all end up waiting on the forking thread.
#[pyclass]
pub struct HNSWIndex {
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,

    // Quantization configuration and PQ instance
    quantization_config: Option<QuantizationConfig>,
    pq: Option<Arc<PQ>>,
    pq_codes: RwLock<HashMap<String, Vec<u8>>>, // PQ codes storage

    // Index-level metadata (simple, infrequently accessed)
    metadata: Mutex<HashMap<String, String>>,

    // Thread-safe vector store with RwLock for concurrent reads
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    vector_metadata: RwLock<HashMap<String, HashMap<String, Value>>>,
    id_map: RwLock<HashMap<String, usize>>,
    rev_map: RwLock<HashMap<usize, String>>,

    // Mutex for write-only fields
    id_counter: Mutex<usize>,
    vector_count: Mutex<usize>, // Track total vectors for training trigger

    /// The graph. A read guard covers a traversal and a single record insertion,
    /// because `hnsw_rs` takes `&self` on both and does its own interior locking.
    /// A write guard covers replacing the whole backend, which `compact`,
    /// `rebuild_with_quantization` and the persistence rebuild each do once.
    hnsw: RwLock<DistanceType>,

    /// Serialises the mutating operations against each other, not against reads.
    ///
    /// `add`, `remove_point`, `compact` and `rebuild_with_quantization` were
    /// serialised against everything by PyO3's exclusive borrow. Relaxing the
    /// receivers removes that, and their internals are not written to interleave
    /// with each other. Id allocation, the training trigger and the overwrite
    /// path each read state and then act on it, so two of them in flight would
    /// race. This restores exactly the mutual exclusion the borrow flag gave
    /// them and nothing more, which leaves searches free to run throughout.
    ///
    /// Held by the Python entry points only. An internal caller reaching a
    /// mutating helper is already inside the guard, so the helpers never take
    /// it and cannot deadlock against the caller that owns it.
    writers: Mutex<()>,

    // ID-based training collection
    training_ids: RwLock<Vec<String>>,      // Just IDs, not vectors
    training_threshold_reached: AtomicBool, // Atomic flag for safety

    // Timestamp when the index was created
    created_at: String,

    // NEW: Flag to prevent training ID collection during persistence rebuild
    pub rebuilding_from_persistence: AtomicBool,

    /// Set once the index has warned that it holds materially more records than
    /// `expected_size` declared, so the warning fires once rather than on every
    /// subsequent `add`.
    overgrowth_warned: AtomicBool,
}

/// Build an `HNSWIndex`
///
/// The only way to construct an index from Python other than loading one from
/// disk. `HNSWIndex` carries no `#[new]`, so the class is importable for
/// `isinstance` checks and type annotations while direct construction raises
/// `TypeError`. Every rule that governs a valid index is enforced here, which
/// is what makes the Python factory and this function agree.
#[pyfunction]
#[pyo3(name = "_create_hnsw_index")]
#[pyo3(signature = (dim, space, m, ef_construction, expected_size, quantization_config = None))]
pub fn create_hnsw_index(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
) -> PyResult<HNSWIndex> {
    HNSWIndex::build(
        dim,
        space,
        m,
        ef_construction,
        expected_size,
        quantization_config,
    )
}

impl HNSWIndex {
    #[instrument(level = "info", skip(quantization_config), fields(
        dim = dim,
        space = %space,
        m = m,
        ef_construction = ef_construction,
        expected_size = expected_size,
        has_quantization = quantization_config.is_some()
    ))]
    pub(crate) fn build(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        quantization_config: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        let start_time = Instant::now();

        // Validation of parameters
        if dim == 0 {
            error!(
                operation = "validation",
                field = "dim",
                value = dim,
                "Invalid dimension"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dim must be positive",
            ));
        }
        if ef_construction == 0 {
            error!(
                operation = "validation",
                field = "ef_construction",
                value = ef_construction,
                "Invalid ef_construction"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ef_construction must be positive",
            ));
        }
        if expected_size == 0 {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                "Invalid expected_size"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "expected_size must be positive",
            ));
        }
        if expected_size > MAX_EXPECTED_SIZE {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                max_allowed = MAX_EXPECTED_SIZE,
                "expected_size exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected_size must be at most {}, got {}. The graph reserves one \
                 slot per declared record at creation, 8 bytes each, so this \
                 declaration would ask for {:.1} GB before a single record is \
                 added. That allocation is not fallible: above this bound the \
                 process aborts rather than raising. expected_size is a capacity \
                 hint and not a limit, and under-declaring only costs some \
                 reallocation, so declare what you expect to hold.",
                MAX_EXPECTED_SIZE,
                expected_size,
                (expected_size as f64 * 8.0) / 1_000_000_000.0
            )));
        }
        if m < 2 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                min_allowed = 2,
                "m below minimum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "m must be at least 2, got {}. Layer assignment samples from a \
                 scale of 1 / ln(m), which is infinity at m 1, so every point \
                 overflows the layer cap and is redispatched uniformly across all \
                 16 layers instead of following the exponential distribution the \
                 graph depends on. Measured on 3,000 records of 32 dimensions, \
                 recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and 1.0000 \
                 at m 16.",
                m
            )));
        }
        if m > 256 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                max_allowed = 256,
                "m exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "m must be less than or equal to 256",
            ));
        }

        // Early space validation with user-friendly error
        let space_normalized = space.to_lowercase();
        match space_normalized.as_str() {
            "cosine" | "l2" | "l1" => {
                debug!(operation = "validation", space = %space_normalized, "Distance space validated");
            }
            _ => {
                error!(operation = "validation", field = "space", value = %space, "Unsupported distance space");
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1'",
                    space
                )));
            }
        }

        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            let qtype = config
                .get_item("type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'type' in quantization_config",
                    )
                })?
                .extract::<String>()?;

            if qtype != "pq" {
                error!(operation = "validation", field = "quantization_type", value = %qtype, "Unsupported quantization type");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported quantization type: '{}'. Only 'pq' is currently supported.",
                    qtype
                )));
            }

            // Extract PQ parameters
            let subvectors = config
                .get_item("subvectors")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'subvectors' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let bits = config
                .get_item("bits")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'bits' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let training_size = config
                .get_item("training_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'training_size' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let max_training_vectors = config
                .get_item("max_training_vectors")?
                .map(|v| v.extract::<usize>())
                .transpose()?;

            // Extract storage_mode
            let storage_mode_str = config
                .get_item("storage_mode")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "quantized_only".to_string());

            let storage_mode = StorageMode::from_string(&storage_mode_str)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            // Validate PQ parameters
            if subvectors == 0 {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    value = subvectors,
                    "Subvectors must be positive"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "subvectors must be a positive integer, got 0",
                ));
            }

            if subvectors > dim {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors exceed dimension"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) cannot exceed dimension ({})",
                    subvectors, dim
                )));
            }

            if !dim.is_multiple_of(subvectors) {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors must divide dimension evenly"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) must divide dimension ({}) evenly",
                    subvectors, dim
                )));
            }

            if !(1..=8).contains(&bits) {
                error!(
                    operation = "validation",
                    field = "bits",
                    value = bits,
                    min = 1,
                    max = 8,
                    "Bits out of range"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bits must be between 1 and 8, got {}",
                    bits
                )));
            }

            if training_size < 1000 {
                error!(
                    operation = "validation",
                    field = "training_size",
                    value = training_size,
                    min = 1000,
                    "Training size too small"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "training_size must be at least 1000, got {}",
                    training_size
                )));
            }

            // A max below the threshold produces an index that reaches its
            // training threshold and then fails training on every record from
            // then on, because the cap is already exceeded by the time the
            // trigger fires. Enforced here so it holds on every construction
            // path rather than only the Python factory.
            if let Some(max_training) = max_training_vectors {
                if max_training < training_size {
                    error!(
                        operation = "validation",
                        field = "max_training_vectors",
                        value = max_training,
                        training_size = training_size,
                        "max_training_vectors below training_size"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "max_training_vectors ({}) must be >= training_size ({})",
                        max_training, training_size
                    )));
                }
            }

            let config = QuantizationConfig {
                subvectors,
                bits,
                training_size,
                max_training_vectors,
                storage_mode,
            };

            debug!(
                operation = "pq_configuration",
                subvectors = subvectors,
                bits = bits,
                training_size = training_size,
                storage_mode = %storage_mode_str,
                sub_dim = dim / subvectors,
                num_centroids = 1 << bits,
                "Product Quantization configured"
            );

            // Create PQ instance
            let pq = Arc::new(PQ::new(
                dim,
                subvectors,
                bits,
                training_size,
                max_training_vectors,
            ));

            (Some(config), Some(pq))
        } else {
            (None, None)
        };

        let max_layer = 16; // Always use NB_LAYER_MAX for hnsw-rs compatibility
        trace!(
            operation = "hnsw_config",
            max_layer = max_layer,
            reason = "hnsw-rs compatibility",
            "Using fixed max_layer"
        );

        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = DistanceType::new_raw(
            &space_normalized,
            m,
            expected_size,
            max_layer,
            ef_construction,
        );

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "index_creation_complete",
            dim = dim,
            space = %space_normalized,
            m = m,
            ef_construction = ef_construction,
            expected_size = expected_size,
            has_quantization = quantization_params.is_some(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: quantization_params,
            pq: pq_instance,
            pq_codes: RwLock::new(HashMap::new()),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        })
    }
}

#[pymethods]
impl HNSWIndex {
    /// Get quantization configuration and status
    pub fn get_quantization_info(&self) -> Option<PyObject> {
        Python::with_gil(|py| {
            if let Some(config) = &self.quantization_config {
                let dict = PyDict::new(py);
                dict.set_item("type", "pq").ok()?;
                dict.set_item("subvectors", config.subvectors).ok()?;
                dict.set_item("bits", config.bits).ok()?;
                dict.set_item("training_size", config.training_size).ok()?;

                if let Some(max_training) = config.max_training_vectors {
                    dict.set_item("max_training_vectors", max_training).ok()?;
                }

                if let Some(pq) = &self.pq {
                    dict.set_item("is_trained", pq.is_trained()).ok()?;

                    // Use enhanced PQ methods
                    let (memory_mb, total_centroids) = pq.get_memory_stats();
                    dict.set_item("memory_mb", memory_mb).ok()?;
                    dict.set_item("total_centroids", total_centroids).ok()?;

                    // The symmetric distance table graph construction reads.
                    // Reported separately because it is derived from the
                    // codebook rather than part of it, and because it scales
                    // with subvectors and bits alone while memory_mb scales
                    // with the dimension too.
                    dict.set_item(
                        "sdc_memory_mb",
                        pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0),
                    )
                    .ok()?;

                    // Calculate compression ratio using cached values
                    let original_bytes = pq.dim * 4; // f32
                    let compressed_bytes = pq.subvectors; // u8 per subvector
                    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
                    dict.set_item("compression_ratio", compression_ratio).ok()?;
                }

                Some(dict.into())
            } else {
                None
            }
        })
    }

    /// Check if quantization is enabled
    pub fn has_quantization(&self) -> bool {
        self.quantization_config.is_some()
    }

    /// Get current vector count (for monitoring training trigger)
    pub fn get_vector_count(&self) -> usize {
        *self.vector_count.lock().unwrap()
    }

    /// Get the distance space configuration
    pub fn get_space(&self) -> String {
        self.space.clone()
    }

    /// Rebuild the HNSW index to use PQ codes after training is complete
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization()
    ), err)]
    pub fn rebuild_with_quantization(&self, py: Python<'_>) -> PyResult<bool> {
        // The whole rebuild runs with the interpreter lock released, the mutation
        // guard included. Waiting for another writer while holding the lock would
        // stall every Python thread in the process for the length of that writer,
        // which is the failure `add` releasing the lock would otherwise create.
        py.allow_threads(|| {
            let _writers = self.writers.lock().unwrap();
            self.rebuild_with_quantization_locked()
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Check if the index is using quantized search
    pub fn is_quantized(&self) -> bool {
        if let Some(pq) = &self.pq {
            if pq.is_trained() {
                let hnsw_guard = self.hnsw.read().unwrap();
                return hnsw_guard.is_quantized();
            }
        }
        false
    }

    /// Check if quantization can be used (PQ is trained)
    pub fn can_use_quantization(&self) -> bool {
        if let Some(pq) = &self.pq {
            pq.is_trained()
        } else {
            false
        }
    }

    /// Enhanced add method that properly handles PQ overwrite scenarios
    #[pyo3(signature = (data, overwrite = true))]
    #[instrument(level = "info", skip(self, data), fields(
        overwrite = overwrite,
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn add(&self, data: Bound<PyAny>, overwrite: bool) -> PyResult<AddResult> {
        let start_time = Instant::now();

        // Input validation
        if data.is_none() {
            error!(
                operation = "add_vectors",
                error = "data_is_none",
                "Data cannot be None"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data cannot be None",
            ));
        }

        // Use error-collecting parsing
        let (parsed_data, parse_errors) = self.parse_input_data(&data);

        let mut total_inserted = 0;
        let mut total_errors = 0;
        let mut errors = Vec::new();

        // Add parse errors to the collection
        for parse_error in parse_errors {
            errors.push(parse_error);
            total_errors += 1;
        }

        if parsed_data.is_empty() && errors.is_empty() {
            trace!(
                operation = "add_vectors",
                result = "empty_input",
                "No vectors to process"
            );
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        let total_input_count = parsed_data.len() + total_errors;
        let vector_shape = Some((total_input_count, self.dim));

        debug!(
            operation = "add_vectors_start",
            total_vectors = parsed_data.len(),
            parse_errors = total_errors,
            overwrite = overwrite,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            storage_mode = self.get_storage_mode(),
            "Starting vector addition"
        );

        // Parsing is the whole of what reads Python objects, and it is done.
        // Everything below works on `parsed_data`, which is owned Rust, so the
        // insertion phase runs with the interpreter lock released. The mutation
        // guard is taken inside that region rather than above it, so a caller
        // waiting for another writer waits without the lock. Holding it while
        // waiting would stall every Python thread in the process for the length
        // of the writer ahead, which is the failure this change would otherwise
        // introduce in place of the one it removes.
        //
        // `insert_parsed_records` carries the proof that nothing inside touches
        // Python.
        let py = data.py();
        let (inserted, insert_errors) = py.allow_threads(|| {
            let _writers = self.writers.lock().unwrap();
            self.insert_parsed_records(parsed_data, overwrite)
        });
        total_inserted += inserted;

        // The errors come back in the order they happened. Two of the three
        // variants carry a message Rust already built. The third carries a
        // `PyErr`, which is formatted here because `PyErr`'s `Display` acquires
        // the interpreter lock and so could not run above.
        for insert_error in insert_errors {
            match insert_error {
                InsertError::Counted(message) => {
                    errors.push(message);
                    total_errors += 1;
                }
                InsertError::Training(message) => {
                    errors.push(message);
                }
                InsertError::Vector { id, err } => {
                    trace!(
                        operation = "add_vector_error",
                        vector_id = %id,
                        error = %err,
                        "Vector addition failed"
                    );
                    errors.push(format!("Vector {}: {}", id, err));
                    total_errors += 1;
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "add_vectors_complete",
            total_inserted = total_inserted,
            total_errors = total_errors,
            success_rate = if total_input_count > 0 {
                total_inserted as f64 / total_input_count as f64 * 100.0
            } else {
                100.0
            },
            duration_ms = duration_ms,
            overwrite_mode = overwrite,
            final_storage_mode = self.get_storage_mode(),
            "Vector addition completed"
        );

        self.warn_if_outgrown_expected_size();

        Ok(AddResult {
            total_inserted,
            total_errors,
            errors,
            vector_shape,
        })
    }

    pub fn get_training_progress(&self) -> f32 {
        if let Some(config) = &self.quantization_config {
            // If PQ is trained, always return 100%
            if let Some(pq) = &self.pq {
                if pq.is_trained() {
                    return 100.0;
                }
            }
            let training_ids = self.training_ids.read().unwrap();
            (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        if let Some(config) = &self.quantization_config {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                0
            } else {
                let training_ids = self.training_ids.read().unwrap();
                config.training_size.saturating_sub(training_ids.len())
            }
        } else {
            0
        }
    }

    /// Check if training is ready to be triggered
    pub fn is_training_ready(&self) -> bool {
        self.training_threshold_reached.load(Ordering::Acquire)
    }

    /// Get current storage mode description
    pub fn get_storage_mode(&self) -> String {
        if !self.has_quantization() {
            "raw_only".to_string()
        } else if !self.can_use_quantization() {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                "raw_ready_for_training".to_string()
            } else {
                "raw_collecting_for_training".to_string()
            }
        } else if self.is_quantized() {
            "quantized_active".to_string()
        } else {
            "raw_trained_not_rebuilt".to_string()
        }
    }

    /// Enhanced search method with automatic ADC usage
    ///
    /// `rerank` controls how far a quantized search over-fetches before it
    /// rescores the candidates against raw vectors. Omitted, it uses
    /// `DEFAULT_RERANK_FACTOR`. An integer of 1 or more pulls that many
    /// candidates per requested result. Zero turns rerank off and restores the
    /// ADC scores and ordering. It has no effect on a raw index or on a
    /// `quantized_only` one, both of which never rerank; see `rerank_plan`.
    // The argument list is the Python signature, so it is not free to be
    // bundled the way the internal batch paths bundle theirs.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false, rerank=None))]
    #[instrument(level = "debug", skip(self, py, vector, filter), fields(
        top_k = top_k,
        ef_search = ef_search,
        return_vector = return_vector,
        rerank = rerank,
        is_quantized = self.is_quantized()
    ), err)]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
        rerank: Option<usize>,
    ) -> PyResult<PyObject> {
        let start_time = Instant::now();

        let ef = ef_search.unwrap_or_else(|| match self.space.to_lowercase().as_str() {
            "l1" | "l2" => std::cmp::max(2 * top_k, 150),
            _ => std::cmp::max(2 * top_k, 100),
        });

        // Resolved once here rather than per query, because it locks the graph
        // to read whether the index is quantized and the batch paths take that
        // lock themselves.
        let params = SearchParams {
            top_k,
            ef,
            return_vector,
            rerank: self.rerank_plan(rerank),
        };

        trace!(
            operation = "search_config",
            ef = ef,
            space = %self.space,
            rerank_factor = params.rerank.map(|plan| plan.factor),
            "Search parameters configured"
        );

        let filter_conditions = filter
            .map(|f| self.python_dict_to_value_map(f))
            .transpose()?;

        // Reject an unrecognised operator before the search runs. Checking it
        // per record would make the error depend on the data, because a record
        // that lacks the field never reaches the operator at all.
        if let Some(conditions) = filter_conditions.as_ref() {
            self.validate_filter_conditions(conditions)?;
        }

        // Detect batch vs single query with comprehensive input support
        let result: PyObject = if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]

            // Validation for empty batch or empty vectors in batch
            if list_vec.is_empty() {
                error!(
                    operation = "search",
                    error = "empty_batch",
                    "Batch cannot be empty"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Batch cannot be empty",
                ));
            }

            // Check for empty vectors within the batch
            for (i, vec) in list_vec.iter().enumerate() {
                if vec.is_empty() {
                    error!(
                        operation = "search",
                        error = "empty_vector_in_batch",
                        vector_index = i,
                        "Vector in batch cannot be empty"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Vector {} in batch cannot be empty",
                        i
                    )));
                }
            }

            debug!(
                operation = "batch_search",
                batch_size = list_vec.len(),
                "Starting batch search"
            );
            let results =
                self.batch_search_internal(&list_vec, filter_conditions.as_ref(), params, py)?;
            PyList::new(py, results)?.into()
        } else if let Ok(np_array) = vector.downcast::<PyArray2<f32>>() {
            // Format: NumPy 2D array (N, dims)
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            if shape.len() != 2 || shape[1] != self.dim {
                error!(
                    operation = "search",
                    error = "shape_mismatch",
                    expected_shape = format!("(N, {})", self.dim),
                    actual_shape = format!("{:?}", shape),
                    "NumPy array shape mismatch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "NumPy array must have shape (N, {}), got {:?}",
                    self.dim, shape
                )));
            }

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat.chunks(self.dim).map(|chunk| chunk.to_vec()).collect();
            debug!(
                operation = "batch_search_numpy",
                batch_size = batch.len(),
                "Starting NumPy batch search"
            );
            let results =
                self.batch_search_internal(&batch, filter_conditions.as_ref(), params, py)?;
            PyList::new(py, results)?.into()
        } else {
            // Single vector path - enhanced with NumPy 1D support
            let query_vector = if let Ok(array1d) = vector.downcast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else {
                vector.extract::<Vec<f32>>()?
            };

            // PROCESS HERE using extract_single_vector logic
            let processed_query = self.validate_and_process_query_vector(query_vector)?;

            trace!(
                operation = "single_search",
                query_dim = processed_query.len(),
                "Starting single vector search"
            );

            let search_results = py.allow_threads(|| -> PyResult<QueryHits> {
                // Check if we should use quantized search
                let use_quantized = self.is_quantized();

                trace!(
                    operation = "search_method",
                    use_quantized = use_quantized,
                    "Selected search method"
                );

                // One read guard for the whole search, taken before the graph lock
                // and held across it. The predicate runs once per candidate the
                // traversal visits, so acquiring the guard inside it would put a
                // lock acquisition on the hot path. See the same pattern in the
                // two batch paths.
                let rev_map = self.rev_map.read().unwrap();
                let live = |internal_id: &usize| rev_map.contains_key(internal_id);

                let fetch_k = params.fetch_k(rev_map.len());

                let hnsw_results = {
                    let hnsw_guard = self.hnsw.read().unwrap();

                    if use_quantized {
                        // Use ADC search for quantized index
                        hnsw_guard
                            .search(&processed_query, fetch_k, params.ef, Some(&live))
                            .unwrap_or_else(|e| {
                                error!(operation = "adc_search", error = %e, "ADC search failed");
                                Vec::new()
                            })
                    } else {
                        // Use raw vector search
                        match hnsw_guard.search(&processed_query, fetch_k, params.ef, Some(&live)) {
                            Ok(results) => results,
                            Err(e) => {
                                error!(operation = "raw_search", error = %e, "Raw search failed");
                                Vec::new()
                            }
                        }
                    }
                };

                // Process results with enhanced vector retrieval
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();
                let vector_metadata = self.vector_metadata.read().unwrap();

                // Resolve, filter and score first, holding only a borrowed id
                // and a float per candidate. Metadata and vectors are cloned
                // after the cut, so an over-fetched page pays for the results
                // it returns rather than for every candidate it considered.
                let mut scored: Vec<(&String, f32)> = Vec::with_capacity(hnsw_results.len());
                let has_filter = filter_conditions.is_some();

                for neighbor in hnsw_results {
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        if has_filter {
                            if let Some(meta) = vector_metadata.get(ext_id) {
                                let filter_conds = filter_conditions.as_ref().unwrap();
                                if !self.matches_filter(meta, filter_conds)? {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let score = match params.rerank.as_ref() {
                            Some(plan) => rescore_candidate(
                                plan,
                                &processed_query,
                                ext_id,
                                &vectors,
                                self.pq.as_ref(),
                                &pq_codes,
                            )
                            .unwrap_or(f32::INFINITY),
                            None => neighbor.distance,
                        };

                        scored.push((ext_id, score));
                    }
                }

                if params.rerank.is_some() {
                    take_best(&mut scored, top_k);
                }

                let mut results = Vec::with_capacity(scored.len());
                for (ext_id, score) in scored {
                    let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
                    let vector_data = if return_vector {
                        // Try raw vector first, then PQ reconstruction
                        vectors.get(ext_id).cloned().or_else(|| {
                            if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(ext_id)) {
                                pq.reconstruct(codes).ok()
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };

                    results.push((ext_id.clone(), score, metadata, vector_data));
                }

                Ok(results)
            })?;

            // Convert to Python objects
            let mut output: Vec<Py<PyDict>> = Vec::with_capacity(search_results.len());
            for (id, score, metadata, vector_data) in search_results {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;
                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }
                output.push(dict.into());
            }

            PyList::new(py, output)?.into()
        };

        // ✅ ENTERPRISE: Add duration timing to hot path with actual result count
        let duration_ms = start_time.elapsed().as_millis();
        let results_count = {
            let any = result.bind(py);
            match any.downcast::<PyList>() {
                Ok(list) => list.len(),
                Err(_) => 0,
            }
        };

        debug!(
            operation = "search_complete",
            results_count = results_count,
            duration_ms = duration_ms,
            "Search completed"
        );

        Ok(result)
    }

    /// Enhanced Save method to include HNSW Graph
    ///
    /// The whole save runs with the interpreter lock released. `save_index`
    /// reaches `save_config`, `save_mappings`, `save_metadata`,
    /// `save_quantization_config`, `save_pq_centroids`, `save_pq_codes`,
    /// `save_vectors` and `save_manifest`, and every one of them speaks only to
    /// `serde_json`, `bincode` and `std::fs`. Every Python token in
    /// `persistence.rs` sits in the load path, in `rebuild_using_add_method` and
    /// `convert_json_value_to_python`. `save_hnsw_graph` calls the vendored
    /// crate's `file_dump`, which names PyO3 nowhere.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.allow_threads(|| self.save_locked(path))
    }

    /// Python property: `index.dim`
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get records by ID(s) with PQ reconstruction support and storage mode awareness
    ///
    /// Looks the ids up in the union of the raw vectors and the quantized codes,
    /// so it already saw every record before the other accessors did. An id that
    /// resolves to no record is dropped from the result rather than reported, so
    /// the returned list can be shorter than the list of ids asked for.
    ///
    /// `return_vector` is served from the raw vector where one exists and from a
    /// reconstruction of the code where one does not, which is the case for a
    /// record added after training under `quantized_only`. The returned value is
    /// then an approximation rather than the value supplied. Measured on 16
    /// dimensional data with 4 subvectors and 8 bits, a reconstructed vector
    /// differed from the stored unit vector by 0.066 at the worst component and
    /// sat at cosine similarity 0.991 to it, while a raw-backed record returned
    /// exactly. `get_stats()["raw_vectors_stored"]` is what tells the two apart
    /// in aggregate.
    #[pyo3(signature = (input, return_vector = true))]
    pub fn get_records(
        &self,
        py: Python<'_>,
        input: &Bound<PyAny>,
        return_vector: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)",
            ));
        };

        trace!(
            operation = "get_records",
            record_count = ids.len(),
            return_vector = return_vector,
            "Retrieving records"
        );

        let mut records = Vec::with_capacity(ids.len());

        // Use read locks for concurrent access
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        for id in ids {
            // Check if this ID exists in either storage
            let exists = vectors.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id.clone())?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    let vector_data = if let Some(raw_vector) = vectors.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.clone())
                    } else if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(&id)) {
                        // Case 2: Only quantized codes available (QuantizedOnly mode)
                        match pq.reconstruct(codes) {
                            Ok(reconstructed) => Some(reconstructed),
                            Err(e) => {
                                warn!(operation = "vector_reconstruction", vector_id = %id, error = %e, "Failed to reconstruct vector");
                                None
                            }
                        }
                    } else {
                        // Case 3: No vector data available
                        None
                    };

                    if let Some(vec) = vector_data {
                        dict.set_item("vector", vec)?;
                    }
                }

                records.push(dict.into());
            }
        }

        trace!(
            operation = "get_records_complete",
            found_records = records.len(),
            "Records retrieval completed"
        );
        Ok(records)
    }

    /// Enhanced get_stats with storage mode information
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        // Nodes the graph holds, which exceeds the live record count by exactly the
        // number of nodes removal and overwrite have stranded. `compact` reclaims the
        // difference. Read first, because the declared lock order puts the graph
        // above every map read below it.
        let graph_nodes = self.hnsw.read().unwrap().nb_points();

        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let training_ids = self.training_ids.read().unwrap();
        let vector_count = *self.vector_count.lock().unwrap();

        // Basic stats
        stats.insert("total_vectors".to_string(), vector_count.to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("index_type".to_string(), "HNSW".to_string());

        stats.insert("m".to_string(), self.m.to_string());
        stats.insert(
            "ef_construction".to_string(),
            self.ef_construction.to_string(),
        );
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());

        stats.insert("graph_nodes".to_string(), graph_nodes.to_string());
        stats.insert(
            "stranded_graph_nodes".to_string(),
            graph_nodes.saturating_sub(vector_count).to_string(),
        );

        // Storage breakdown
        stats.insert("raw_vectors_stored".to_string(), vectors.len().to_string());
        stats.insert(
            "quantized_codes_stored".to_string(),
            pq_codes.len().to_string(),
        );

        // Training info
        if let Some(config) = &self.quantization_config {
            stats.insert("quantization_type".to_string(), "pq".to_string());
            stats.insert(
                "quantization_training_size".to_string(),
                config.training_size.to_string(),
            );

            // Storage mode information
            stats.insert(
                "storage_mode".to_string(),
                config.storage_mode.to_string().to_string(),
            );

            // Calculate actual memory usage based on storage mode
            let raw_memory_mb = (vectors.len() * self.dim * 4) as f64 / (1024.0 * 1024.0);
            let quantized_memory_mb =
                (pq_codes.len() * config.subvectors) as f64 / (1024.0 * 1024.0);

            stats.insert(
                "raw_vectors_memory_mb".to_string(),
                format!("{:.2}", raw_memory_mb),
            );
            stats.insert(
                "quantized_codes_memory_mb".to_string(),
                format!("{:.2}", quantized_memory_mb),
            );

            // `memory_savings` used to sit beside `storage_strategy`, reading
            // "maximum" under QuantizedOnly. That mode is the smaller of the two
            // and it is not a maximum of anything. Measured at 3,000 records of
            // 64 dimensions it held more resident memory than the same index
            // unquantized, because the centroid distance table is a fixed 1 MB.
            // What replaces it is the fact the figures above can be checked
            // against, being which records still have a raw vector.
            match config.storage_mode {
                StorageMode::QuantizedOnly => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "memory_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "training_records_only".to_string(),
                    );
                }
                StorageMode::QuantizedWithRaw => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "quality_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "all_records".to_string(),
                    );
                }
            }

            let collected_count = training_ids.len();
            let progress = self.get_training_progress();
            stats.insert(
                "training_progress".to_string(),
                format!(
                    "{}/{} ({:.1}%)",
                    collected_count, config.training_size, progress
                ),
            );

            let vectors_needed = self.training_vectors_needed();
            stats.insert(
                "training_vectors_needed".to_string(),
                vectors_needed.to_string(),
            );
            stats.insert(
                "training_threshold_reached".to_string(),
                self.training_threshold_reached
                    .load(Ordering::Acquire)
                    .to_string(),
            );

            if let Some(pq) = &self.pq {
                let is_trained = pq.is_trained();
                stats.insert("quantization_trained".to_string(), is_trained.to_string());
                stats.insert(
                    "quantization_active".to_string(),
                    self.is_quantized().to_string(),
                );

                // The two fixed costs, reported here so that the whole memory
                // question can be answered from one call. Both are independent
                // of the record count, and at small record counts the table is
                // the largest single thing a quantized index holds. They were
                // only on `get_quantization_info`, which is where a caller
                // reading the storage breakdown above would not look.
                let (centroid_mb, _) = pq.get_memory_stats();
                stats.insert(
                    "codebook_memory_mb".to_string(),
                    format!("{:.2}", centroid_mb),
                );
                stats.insert(
                    "sdc_table_memory_mb".to_string(),
                    format!("{:.2}", pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0)),
                );

                if is_trained {
                    let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
                    stats.insert(
                        "quantization_compression_ratio".to_string(),
                        format!("{:.1}x", compression_ratio),
                    );
                }
            }
        } else {
            stats.insert("quantization_type".to_string(), "none".to_string());
            stats.insert("storage_mode".to_string(), "raw_only".to_string());
        }

        stats.insert(
            "storage_mode_description".to_string(),
            self.get_storage_mode(),
        );

        stats
    }

    /// List the first number of records in the index (ID and metadata)
    ///
    /// Enumerates `id_map`, which holds every live record. It used to enumerate
    /// `vectors`, which under `quantized_only` holds only the records collected
    /// before training, so every record added afterwards was missing from the
    /// listing while search still returned it.
    ///
    /// Iteration order is a hash map's and is not stable between calls, so
    /// `number` takes an arbitrary N rather than a defined page.
    #[pyo3(signature = (number=10))]
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, PyObject)>> {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        let mut results = Vec::new();
        for id in id_map.keys().take(number) {
            let metadata = vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = self.value_map_to_python(&metadata, py)?;
            results.push((id.clone(), py_metadata));
        }
        Ok(results)
    }

    /// Check whether a record with this id is in the index
    ///
    /// Reads `id_map`, which is the record set. Every insertion path writes it,
    /// `remove_point_internal` keys its removal on it, `add(overwrite=True)`
    /// keys its collision test on it, and `compact` rebuilds the graph from it.
    /// It used to read `vectors`, which under `quantized_only` holds only the
    /// records collected before training, so this returned `false` for a record
    /// that search returned and `remove_point` removed.
    pub fn contains(&self, id: String) -> bool {
        let id_map = self.id_map.read().unwrap();
        id_map.contains_key(&id)
    }

    /// Add index-level metadata
    pub fn add_metadata(&self, metadata: HashMap<String, String>) {
        let mut meta_lock = self.metadata.lock().unwrap();
        for (key, value) in metadata {
            meta_lock.insert(key, value);
        }
    }

    /// Get index-level metadata value
    pub fn get_metadata(&self, key: String) -> Option<String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.get(&key).cloned()
    }

    /// Get all index-level metadata
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.clone()
    }

    /// Get a human-readable info string
    ///
    /// `vectors=` is the live record count, taken from `id_map`. It used to be
    /// `vectors.len()`, which under `quantized_only` counts only the records
    /// that still hold a raw vector and therefore reported fewer records than
    /// the index contains.
    ///
    /// The count is read into a local and the guard dropped before anything
    /// else is touched. `is_quantized()` below takes the graph lock, and the
    /// declared order puts the graph above `vectors`, so holding that guard
    /// across the call was an inversion.
    pub fn info(&self) -> String {
        let record_count = self.id_map.read().unwrap().len();
        let base_info = format!(
            "HNSWIndex(dim={}, space={}, m={}, ef_construction={}, expected_size={}, vectors={}",
            self.dim, self.space, self.m, self.ef_construction, self.expected_size, record_count
        );

        if let Some(config) = &self.quantization_config {
            let trained_status = self
                .pq
                .as_ref()
                .map(|pq| {
                    if pq.is_trained() {
                        "trained"
                    } else {
                        "untrained"
                    }
                })
                .unwrap_or("unknown");

            let active_status = if self.is_quantized() {
                "active"
            } else {
                "inactive"
            };

            // Use cached compression ratio calculation with proper float division
            let compression_info = self
                .pq
                .as_ref()
                .map(|pq| format!("{:.1}x", (pq.dim as f64 * 4.0) / pq.subvectors as f64))
                .unwrap_or_else(|| "unknown".to_string());

            format!(
                "{}, quantization=pq(subvectors={}, bits={}, {}, {}, compression={}))",
                base_info,
                config.subvectors,
                config.bits,
                trained_status,
                active_status,
                compression_info
            )
        } else {
            format!("{}, quantization=none)", base_info)
        }
    }

    /// Remove vector by ID
    /// Public remove_point method (unchanged for API compatibility)
    /// This code delegates to remove_point_internal() which handles all the complex logic
    pub fn remove_point(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        // `id` arrives already converted, and `remove_point_internal` is in the
        // set `insert_parsed_records` verifies, so the whole body is Rust. The
        // removal itself is short, but the wait for the mutation guard is not,
        // because `add` can now hold it for a long insert with the lock released.
        // Waiting here with the lock held would stall every Python thread.
        py.allow_threads(|| {
            let _writers = self.writers.lock().unwrap();
            self.remove_point_internal(id)
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Rebuild the graph in memory and reclaim the nodes removal and overwrite strand.
    ///
    /// `remove_point` clears a record from every storage map but cannot delete its
    /// graph node, and `add(overwrite=True)` is a removal followed by an insertion,
    /// so both leave behind a node that still holds a copy of the vector and both
    /// directions of adjacency while resolving to no record. Search already excludes
    /// those nodes, so this is a resource operation and not a correctness one. What it
    /// reclaims is their memory, their edge slots in live neighbour lists, and the
    /// traversal steps they cost.
    ///
    /// Returns the number of nodes reclaimed. Zero means the graph held no stranded
    /// nodes, in which case nothing is rebuilt and the call is a no-op.
    ///
    /// The cost is a full sequential rebuild, proportional to the live record count
    /// rather than to the amount of debris. Nothing outside the graph is touched.
    /// Internal ids, external ids, metadata, stored vectors, quantized codes, PQ
    /// training state and the id counter all survive unchanged, so the record any
    /// given id resolves to is the same before and after.
    ///
    /// The replacement graph is built in full before the old one is dropped, so peak
    /// memory holds both for the duration and a failure part way through leaves the
    /// index exactly as it was.
    ///
    /// This is never automatic. Calling it is a decision a deployment can schedule.
    ///
    /// The rebuild runs with the interpreter lock released. Every function it
    /// reaches is in the set `insert_parsed_records` verifies, plus
    /// `DistanceType::new_raw` and `insert_batch`, which are the same shape as
    /// the quantized pair already listed there.
    pub fn compact(&self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.compact_locked())
    }

    /// Get performance characteristics and limitations
    ///
    /// Reports only what the code does. Insertion runs one record at a time
    /// through `add_single_vector`, so the fields that described a parallel
    /// insert path were removed rather than corrected.
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("search_speedup_expected".to_string(), "1.2x-2x".to_string());
        info.insert(
            "search_bottleneck".to_string(),
            "hnsw_mutex_serialization".to_string(),
        );
        info.insert(
            "benefits".to_string(),
            "gil_release_concurrent_metadata_processing_batched_search".to_string(),
        );
        info.insert("insertion_path".to_string(), "sequential".to_string());

        // Add quantization performance info
        //
        // `quantization_memory_savings` used to sit here, reporting
        // 1 - 1/compression_ratio as a percentage. That is the share of a
        // vector a code replaces, not the memory an index saves, and under
        // QuantizedWithRaw the index saves nothing on the vectors because it
        // keeps every one of them. It carried no information
        // `quantization_compression` does not, so it is gone rather than
        // qualified. Memory belongs to `get_stats`, which measures rather than
        // projects.
        if let Some(config) = &self.quantization_config {
            let original_bytes = self.dim * 4; // f32
            let compressed_bytes = config.subvectors; // u8 per subvector
            let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

            info.insert(
                "quantization_compression".to_string(),
                format!("{:.1}x", compression_ratio),
            );
            // Measured at 0.16 recall at 10 against 1.00 for the same data
            // unquantized, so "slight" was wrong by a factor the word cannot
            // carry. Rerank recovers it in full and only QuantizedWithRaw can
            // rerank, so the two modes get different answers.
            info.insert(
                "quantization_accuracy_impact".to_string(),
                match config.storage_mode {
                    StorageMode::QuantizedOnly => "large_recall_loss_no_rerank_available",
                    StorageMode::QuantizedWithRaw => "large_recall_loss_unless_reranked",
                }
                .to_string(),
            );
        }

        info
    }

    /// Concurrent benchmark for search performance
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(
        &self,
        query_count: usize,
        max_threads: Option<usize>,
    ) -> PyResult<HashMap<String, f64>> {
        use rand::random; // Import for random number generation

        let start_time = Instant::now();

        debug!(
            operation = "benchmark_start",
            query_count = query_count,
            max_threads = max_threads,
            "Starting concurrent read benchmark"
        );

        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| random::<f32>()).collect())
            .collect();

        let mut results = HashMap::new();

        // Sequential benchmark
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        results.insert("sequential_time".to_string(), sequential_time);
        results.insert(
            "sequential_qps".to_string(),
            queries.len() as f64 / sequential_time,
        );

        // Parallel benchmark
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads
            .unwrap_or(available_threads)
            .min(available_threads);

        let start = Instant::now();
        let _: Vec<_> = queries
            .par_iter()
            .map(|query| self.raw_search_no_gil(query))
            .collect();

        let parallel_time = start.elapsed().as_secs_f64();
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert(
            "parallel_qps".to_string(),
            queries.len() as f64 / parallel_time,
        );
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);

        let total_duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "benchmark_complete",
            sequential_qps = queries.len() as f64 / sequential_time,
            parallel_qps = queries.len() as f64 / parallel_time,
            speedup = sequential_time / parallel_time,
            duration_ms = total_duration_ms,
            "Benchmark completed"
        );

        Ok(results)
    }
}

/// Records accepted by `add`, after parsing and before insertion, as
/// (external id, vector, metadata). The vector is still in its input form and
/// has not been normalized for the index space yet.
type ParsedRecords = Vec<(String, Vec<f32>, HashMap<String, Value>)>;

/// An error raised inside `add`'s insertion phase, carried out to be recorded
///
/// The insertion phase runs with the interpreter lock released, so it cannot
/// build the message for an error that arrives as a `PyErr`. `PyErr`'s
/// `Display` implementation calls `Python::with_gil`, which would reacquire the
/// lock in the middle of the region that exists to have released it, and would
/// do so while the mutation guard and possibly a storage guard are held.
/// `add` formats those once the lock is back.
enum InsertError {
    /// A message Rust already holds, counted against `total_errors`
    Counted(String),

    /// A training failure. Recorded but not counted, which is what the training
    /// path has always done, because a training failure is not a rejected record
    Training(String),

    /// A `PyErr` from one of the three insert paths, with the id it belongs to.
    /// Counted against `total_errors` once formatted
    Vector { id: String, err: PyErr },
}

/// Search hits for one query vector, as (external id, distance, metadata,
/// optional raw vector). The raw vector is present only when the caller asked
/// for it and the index still holds one.
type QueryHits = Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>;

/// A JSON number as either an exact integer or a float. `serde_json` stores an
/// integer and a float in different variants, and comparing those variants is
/// what made 10 and 10.0 unequal under some operators and equal under others.
enum NumericValue {
    Integer(i128),
    Float(f64),
}

// INTERNAL METHODS, HELPERS AND IMPLEMENTATIONS
impl HNSWIndex {
    /// Count the records the index actually holds
    ///
    /// The union of the raw vectors and the PQ codes, because `quantized_only`
    /// keeps a record added after training in the codes alone. This is derived
    /// from the stored data rather than from the counter, so it is what the
    /// counter is checked against after a load. Not exposed to Python, since
    /// the only caller is the load path in `persistence`.
    pub fn count_stored_records(&self) -> usize {
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let code_only = pq_codes
            .keys()
            .filter(|id| !vectors.contains_key(*id))
            .count();
        vectors.len() + code_only
    }

    /// Warn once when the index holds materially more records than it declared
    ///
    /// `expected_size` is a capacity hint rather than a limit, so exceeding it is
    /// legal and the index keeps working. What it costs is not the reservation,
    /// which grows through the ordinary `Vec::push` path, but the graph degree.
    /// The Python factory derives the default `m` from `expected_size`, and `m`
    /// is fixed at construction, so an index that has outgrown its declaration by
    /// a wide margin is running at a degree chosen for a smaller index and no
    /// later `add` revises it. Nothing else tells a caller that.
    ///
    /// A warning rather than an error, because the index is correct and the only
    /// remedy is to rebuild at an honest declaration, which is the caller's call.
    ///
    /// Fires once per index. The flag is claimed with a compare and exchange, so
    /// two writers crossing the threshold together produce one line and not two.
    fn warn_if_outgrown_expected_size(&self) {
        if self.overgrowth_warned.load(Ordering::Acquire) {
            return;
        }

        let threshold = self
            .expected_size
            .saturating_mul(EXPECTED_SIZE_OVERGROWTH_FACTOR);
        let live_records = self.id_map.read().unwrap().len();
        if live_records <= threshold {
            return;
        }

        if self
            .overgrowth_warned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        warn!(
            operation = "expected_size_exceeded",
            live_records = live_records,
            expected_size = self.expected_size,
            m = self.m,
            "Index holds more than {}x the records its expected_size declared. \
             expected_size is a hint and not a limit, so nothing is broken, but m \
             is fixed at construction and was sized for the declaration. Recreate \
             the index with an expected_size matching what it actually holds if \
             recall matters. This warning fires once.",
            EXPECTED_SIZE_OVERGROWTH_FACTOR
        );
    }

    /// Get next available internal ID
    ///
    /// Not exposed to Python. Every call takes the counter mutex and
    /// increments it, so a call from outside the insertion path burns an
    /// internal id that no record will ever hold.
    fn get_next_id(&self) -> usize {
        let mut counter = self.id_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Pure function for vector normalization
    fn normalize_vector(&self, vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector // Return unchanged for zero vectors
        }
    }

    /// Process vector according to distance space
    fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        match self.space.to_lowercase().as_str() {
            "cosine" => self.normalize_vector(vector),
            // Future extensions:
            // "l2" => self.preprocess_l2(vector),
            // "l1" => self.preprocess_l1(vector),
            _ => vector,
        }
    }

    /// Helper for query processing (mirrors extract_single_vector validation)
    fn validate_and_process_query_vector(&self, vector: Vec<f32>) -> PyResult<Vec<f32>> {
        // Same validation as extract_single_vector
        if vector.is_empty() {
            error!(
                operation = "query_validation",
                error = "empty_vector",
                "Search vector cannot be empty"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Search vector cannot be empty",
            ));
        }
        if vector.len() != self.dim {
            error!(
                operation = "query_validation",
                error = "dimension_mismatch",
                expected = self.dim,
                actual = vector.len(),
                "Search vector dimension mismatch"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Search vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                error!(
                    operation = "query_validation",
                    error = "invalid_value",
                    index = i,
                    value = val,
                    "Search vector contains invalid value"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Search vector contains invalid value at index {}: {}",
                    i, val
                )));
            }
        }

        // Apply same processing as storage vectors
        Ok(self.process_vector_for_space(vector))
    }

    /// Internal remove_point method that can be called without Python bindings
    /// This is the core method that properly removes all traces of a document
    /// Enhanced internal remove_point method with comprehensive PQ support
    fn remove_point_internal(&self, id: String) -> Result<bool, String> {
        // Read before the guards are taken, because it reaches the graph lock and
        // the declared order puts the graph above every map held below.
        let storage_mode = self.get_storage_mode();

        // Every write guard, in the order declared on `HNSWIndex`. This used to
        // take `vectors` before `rev_map`, which is the reverse of what a search
        // does, and a search holds `rev_map` for its whole traversal. That pair
        // could not overlap while the receivers were exclusive. It can now.
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        let mut pq_codes = self.pq_codes.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();

        // Check if the document exists
        if let Some(internal_id) = id_map.remove(&id) {
            // Track what we're removing for logging
            let had_raw_vector = vectors.contains_key(&id);
            let had_pq_codes = pq_codes.contains_key(&id);

            // Remove from all data structures
            vectors.remove(&id); // Remove raw vectors (if present)
            vector_metadata.remove(&id); // Remove metadata
            pq_codes.remove(&id); // Remove PQ codes (if present)
            rev_map.remove(&internal_id); // Remove ID mapping

            // Enhanced training state cleanup for quantization
            if self.has_quantization() {
                // Remove from training IDs if present and not yet trained
                if !self.can_use_quantization() {
                    let mut training_ids = self.training_ids.write().unwrap();
                    let original_len = training_ids.len();
                    training_ids.retain(|training_id| training_id != &id);

                    if training_ids.len() != original_len {
                        trace!(
                            operation = "training_cleanup",
                            vector_id = %id,
                            remaining_training_vectors = training_ids.len(),
                            "Removed vector from training set"
                        );

                        // Update threshold status if we dropped below training size
                        if let Some(config) = &self.quantization_config {
                            if training_ids.len() < config.training_size {
                                self.training_threshold_reached
                                    .store(false, std::sync::atomic::Ordering::Release);
                                debug!(
                                    operation = "training_threshold_reset",
                                    remaining_vectors = training_ids.len(),
                                    required = config.training_size,
                                    "Training threshold reset due to removal"
                                );
                            }
                        }
                    }
                }
            }

            // Decrement vector count since we removed a vector
            {
                let mut count = self.vector_count.lock().unwrap();
                if *count > 0 {
                    *count -= 1;
                }
            }

            debug!(
                operation = "remove_point_internal",
                vector_id = %id,
                internal_id = internal_id,
                had_raw_vector = had_raw_vector,
                had_pq_codes = had_pq_codes,
                storage_mode = storage_mode,
                note = "hnsw_graph_entries_remain_unreachable",
                "Vector completely removed from index (HNSW graph entries become unreachable)"
            );
            Ok(true)
        } else {
            trace!(
                operation = "remove_point_internal",
                vector_id = %id,
                "Vector not found for removal"
            );
            Ok(false)
        }
    }

    /// The body of `rebuild_with_quantization`, with the writers guard already held
    ///
    /// Training reaches this from inside `add`, which owns the guard for the whole
    /// call, so the two entry points are separate rather than one taking the guard
    /// twice and deadlocking on itself.
    ///
    /// Errors are `String` rather than `PyErr` because both callers reach this with
    /// the interpreter lock released, and `PyErr`'s `Display` acquires it. The
    /// entry point above turns the message back into the `PyRuntimeError` it always
    /// raised.
    fn rebuild_with_quantization_locked(&self) -> Result<bool, String> {
        let start_time = Instant::now();

        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                warn!(
                    operation = "rebuild_quantization",
                    reason = "pq_not_trained",
                    "Cannot rebuild: PQ not trained"
                );
                return Ok(false);
            }
        };

        // Create new PQ-based HNSW index
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        trace!(
            operation = "rebuild_quantization",
            max_layer = max_layer,
            "Creating new PQ HNSW index"
        );

        // Quantize every stored vector and record the codes, then release the
        // storage guards. Nothing below this block holds one, which is what lets
        // the graph work take its own guards in the declared order rather than
        // under a `vectors` guard taken first.
        let (vector_count, retained) = {
            let vectors = self.vectors.read().unwrap();
            if vectors.is_empty() {
                warn!(
                    operation = "rebuild_quantization",
                    reason = "no_vectors",
                    "Cannot rebuild: no vectors available"
                );
                return Ok(false);
            }

            info!(
                operation = "quantization_rebuild_start",
                vector_count = vectors.len(),
                "Starting quantization rebuild"
            );

            let vector_refs: Vec<&[f32]> = vectors.values().map(|v| v.as_slice()).collect();
            let quantized_codes = pq.quantize_batch(&vector_refs).map_err(|e| {
                error!(operation = "quantization_rebuild", error = %e, "Failed to quantize vectors");
                format!("Failed to quantize vectors: {}", e)
            })?;

            // Store quantized codes. Codes for records that have no raw vector are
            // kept rather than cleared, because under QuantizedOnly they are the
            // only copy of every record added after training completed and there is
            // nothing left to re-quantize them from. Clearing dropped those records
            // from the index outright. Removal already deletes an id's codes, so
            // nothing stale can survive here.
            let mut pq_codes = self.pq_codes.write().unwrap();
            let retained = pq_codes
                .keys()
                .filter(|id| !vectors.contains_key(*id))
                .count();

            for (i, (id, _)) in vectors.iter().enumerate() {
                if i < quantized_codes.len() {
                    pq_codes.insert(id.clone(), quantized_codes[i].clone());
                }
            }
            debug!(
                operation = "quantization_rebuild",
                codes_stored = pq_codes.len(),
                codes_retained = retained,
                "Quantized codes stored"
            );
            (vectors.len(), retained)
        };

        // The codes are copied out so the graph is built with no lock held at
        // all. A large batch insert forks to rayon, and a fork under the graph's
        // write guard can leave every worker in the pool waiting on the thread
        // that holds it. Copying costs one byte per subvector per record.
        let batch_data: Vec<(Vec<u8>, usize)> = {
            let id_map = self.id_map.read().unwrap();
            let pq_codes = self.pq_codes.read().unwrap();
            pq_codes
                .iter()
                .filter_map(|(id, codes)| {
                    id_map
                        .get(id)
                        .map(|&internal_id| (codes.clone(), internal_id))
                })
                .collect()
        };

        let new_hnsw = DistanceType::new_pq(
            &self.space,
            self.m,
            self.expected_size,
            max_layer,
            self.ef_construction,
            pq.clone(),
        );

        if !batch_data.is_empty() {
            let batch: Vec<(&Vec<u8>, usize)> = batch_data
                .iter()
                .map(|(codes, internal_id)| (codes, *internal_id))
                .collect();
            new_hnsw.insert_batch_pq(&batch)
                .map_err(|e| {
                    error!(operation = "quantization_rebuild", error = %e, "Failed to insert quantized vectors");
                    format!("Failed to insert quantized vectors: {}", e)
                })?;
        }

        // The replacement is built in full before it is installed, so a search
        // running alongside this sees the old graph or the new one and never a
        // partly filled one. It used to see the empty new graph for as long as
        // the insertions took.
        //
        // The old graph is moved out and dropped after the guard is released.
        // See `replace_graph`.
        self.replace_graph(new_hnsw);

        // ✅ ENTERPRISE: Add duration timing with fixed compression ratio calculation
        let duration_ms = start_time.elapsed().as_millis();
        let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
        info!(
            operation = "quantization_rebuild_complete",
            vector_count = vector_count,
            codes_inserted = batch_data.len(),
            codes_retained = retained,
            compression_ratio = compression_ratio,
            duration_ms = duration_ms,
            "Quantization rebuild completed successfully"
        );

        Ok(true)
    }

    /// The body of `compact`, with the interpreter lock already released
    fn compact_locked(&self) -> PyResult<usize> {
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();

        let live_count = self.id_map.read().unwrap().len();
        let nodes_before = self.hnsw.read().unwrap().nb_points();

        if nodes_before <= live_count {
            debug!(
                operation = "compact",
                graph_nodes = nodes_before,
                live_records = live_count,
                "No stranded nodes, compact is a no-op"
            );
            return Ok(0);
        }

        let quantized = self.is_quantized();
        // NB_LAYER_MAX, matching every other construction site in this file.
        let max_layer = 16;

        let new_hnsw = if quantized {
            let pq = self.pq.as_ref().cloned().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Index reports a quantized graph but holds no product quantizer",
                )
            })?;
            DistanceType::new_pq(
                &self.space,
                self.m,
                self.expected_size,
                max_layer,
                self.ef_construction,
                pq,
            )
        } else {
            DistanceType::new_raw(
                &self.space,
                self.m,
                self.expected_size,
                max_layer,
                self.ef_construction,
            )
        };

        // Re-insert every live record under the internal id it already holds, so the
        // two id maps stay correct without being rewritten. A record whose source data
        // is missing is collected rather than skipped, because skipping it would drop
        // it from the index silently.
        let missing: Vec<String> = {
            let id_map = self.id_map.read().unwrap();

            if quantized {
                let pq_codes = self.pq_codes.read().unwrap();
                let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(id_map.len());
                let mut missing = Vec::new();

                for (ext_id, &internal_id) in id_map.iter() {
                    match pq_codes.get(ext_id) {
                        Some(codes) => batch.push((codes, internal_id)),
                        None => missing.push(ext_id.clone()),
                    }
                }

                if missing.is_empty() && !batch.is_empty() {
                    new_hnsw.insert_batch_pq(&batch).map_err(|e| {
                        error!(operation = "compact", error = %e, "Failed to re-insert quantized codes");
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to re-insert quantized codes during compact: {}",
                            e
                        ))
                    })?;
                }

                missing
            } else {
                let vectors = self.vectors.read().unwrap();
                let mut missing = Vec::new();

                for (ext_id, &internal_id) in id_map.iter() {
                    match vectors.get(ext_id) {
                        Some(vector) => new_hnsw.insert(vector, internal_id),
                        None => missing.push(ext_id.clone()),
                    }
                }

                missing
            }
        };

        if !missing.is_empty() {
            error!(
                operation = "compact",
                missing_records = missing.len(),
                live_records = live_count,
                quantized = quantized,
                "Refusing to compact, some live records have no source data to rebuild from"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Refusing to compact: {} of {} live records have no stored {} to rebuild \
                 the graph from, so compacting would drop them. The index is unchanged.",
                missing.len(),
                live_count,
                if quantized {
                    "quantized codes"
                } else {
                    "vector"
                }
            )));
        }

        let nodes_after = new_hnsw.nb_points();
        self.replace_graph(new_hnsw);

        let reclaimed = nodes_before - nodes_after;
        info!(
            operation = "compact_complete",
            nodes_before = nodes_before,
            nodes_after = nodes_after,
            nodes_reclaimed = reclaimed,
            live_records = live_count,
            quantized = quantized,
            duration_ms = start_time.elapsed().as_millis(),
            "Graph compacted"
        );

        Ok(reclaimed)
    }

    /// The body of `save`, with the interpreter lock already released
    fn save_locked(&self, path: &str) -> PyResult<()> {
        // A save reads the mappings, the metadata, the codes, the vectors and
        // the graph in five separate passes, so it needs the index to hold
        // still. PyO3's exclusive borrow used to guarantee that by keeping every
        // mutating method away from it. Relaxing the receivers removes that, and
        // a save overlapping an add would write a directory whose mappings and
        // vectors came from different instants. This takes the mutation lock
        // instead, which blocks a concurrent write and no reader at all.
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();
        info!(operation = "save_start", path = path, "Starting index save");

        let path_buf = Path::new(path);

        // Phase 1: Save all ZeusDB components (already tested to work)
        debug!(operation = "save_phase1", "Saving ZeusDB components");
        crate::persistence::save_index(self, path)?;

        // Phase 2: Save HNSW graph using hnsw-rs native dump
        debug!(operation = "save_phase2", "Saving HNSW graph");
        self.save_hnsw_graph(path_buf)?;

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "save_complete",
            path = path,
            duration_ms = duration_ms,
            "Index save completed successfully"
        );
        Ok(())
    }

    /// Install a replacement graph, and drop the old one outside the guard
    ///
    /// The three paths that replace the whole backend, `compact`, the
    /// quantization rebuild and the persistence rebuild, all used to write
    /// `*hnsw_guard = new_hnsw` directly. That assignment drops the old graph
    /// while the write guard is still held, and dropping a graph is not a quiet
    /// operation. `PointIndexation::drop` in the vendored crate clears each
    /// layer with `into_par_iter().for_each(...)`, so the drop forks to rayon.
    ///
    /// A rayon fork under the graph's write guard deadlocks whenever the pool is
    /// occupied by search tasks. `batch_search_parallel` fans a batch of more
    /// than five queries across the pool and each task takes a read guard, so
    /// once a writer is queued every worker blocks behind it. The fork then has
    /// no worker to run on and the writer never reaches the point of releasing.
    /// Relay 36 wrote the rule that no path forks to rayon while holding a write
    /// guard, and this is the fork that rule missed, because it is hidden inside
    /// an assignment rather than written as a call.
    ///
    /// Moving the old value out and dropping it after the guard is released
    /// keeps the swap to a pointer move under the guard.
    fn replace_graph(&self, new_hnsw: DistanceType) {
        let old = {
            let mut hnsw_guard = self.hnsw.write().unwrap();
            std::mem::replace(&mut *hnsw_guard, new_hnsw)
        };
        drop(old);
    }

    /// The insertion phase of `add`, run with the interpreter lock released
    ///
    /// Everything here operates on `ParsedRecords`, which is
    /// `Vec<(String, Vec<f32>, HashMap<String, Value>)>` and holds no Python
    /// object, no `Py<T>`, no `PyObject` and no borrow of anything Python owns.
    /// The caller holds the mutation guard.
    ///
    /// The complete set of functions reachable from here, verified by reading
    /// each one rather than by inference:
    ///
    /// - `remove_point_internal`, and through it `get_storage_mode`,
    ///   `has_quantization` and `can_use_quantization`
    /// - `add_single_vector`, and the three paths below it, `add_raw_vector`,
    ///   `add_with_id_collection` and `add_quantized_vector`
    /// - `get_next_id`, `is_quantized`
    /// - `maybe_trigger_training`, `train_quantization_from_ids` and
    ///   `rebuild_with_quantization_locked`
    /// - `PQ::is_trained`, `quantize`, `quantize_batch` and `train`, plus the
    ///   k-means below `train`
    /// - `DistanceType::insert`, `insert_pq_codes`, `insert_batch_pq`,
    ///   `new_pq` and `nb_points`, and the vendored `hnsw_rs` graph below them
    ///
    /// None of them takes a `Python` token, and none of them calls into the
    /// interpreter. `pq.rs`, `distance.rs` and the vendored crate name PyO3
    /// nowhere at all. The two places that did reach Python were both removed
    /// rather than worked around. `rebuild_with_quantization_locked` returned a
    /// `PyResult` whose error the training path formatted into a message, and it
    /// now returns `Result<bool, String>`. The per-record errors are carried out
    /// as `InsertError` values instead of being formatted here.
    ///
    /// Training completing mid-insert is the longest thing this can run, since it
    /// fires k-means and then rebuilds the whole graph from quantized codes, and
    /// it is entirely Rust.
    ///
    /// Logging is safe. The `tracing` subscriber this crate installs writes to
    /// stdout, to stderr, or to a rotating file through `tracing-appender`. No
    /// layer bridges to Python's `logging`, and the Python layer only ever sets
    /// environment variables that the Rust initialiser reads at import.
    ///
    /// A panic in here is safe too. `Python::allow_threads` restores the
    /// interpreter lock from a `Drop` guard, so an unwind reacquires it before it
    /// reaches PyO3's boundary.
    fn insert_parsed_records(
        &self,
        parsed_data: ParsedRecords,
        overwrite: bool,
    ) -> (usize, Vec<InsertError>) {
        let mut total_inserted = 0;
        let mut errors: Vec<InsertError> = Vec::new();

        // ENHANCED FIX: Handle overwrites properly for ALL paths (Raw, Training, PQ)
        if overwrite {
            // Phase 1: Batch identify and remove existing documents
            let (ids_to_remove, storage_analysis) = {
                let id_map = self.id_map.read().unwrap();
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();

                let mut ids_to_remove = Vec::new();
                let mut has_raw = 0;
                let mut has_pq = 0;
                let mut has_both = 0;

                for (id, _, _) in &parsed_data {
                    if id_map.contains_key(id) {
                        ids_to_remove.push(id.clone());

                        // Analyze what's being replaced for logging
                        let has_raw_vector = vectors.contains_key(id);
                        let has_pq_codes = pq_codes.contains_key(id);

                        match (has_raw_vector, has_pq_codes) {
                            (true, true) => has_both += 1,
                            (true, false) => has_raw += 1,
                            (false, true) => has_pq += 1,
                            (false, false) => {} // Shouldn't happen, but handle gracefully
                        }
                    }
                }

                (ids_to_remove, (has_raw, has_pq, has_both))
            }; // Release all read locks here

            if !ids_to_remove.is_empty() {
                info!(
                    operation = "overwrite_preparation",
                    documents_to_remove = ids_to_remove.len(),
                    storage_analysis = format!(
                        "raw_only: {}, pq_only: {}, both: {}",
                        storage_analysis.0, storage_analysis.1, storage_analysis.2
                    ),
                    "Removing existing documents for overwrite"
                );

                // Batch remove existing documents (handles both raw and PQ data)
                let mut removed_count = 0;
                let mut removal_errors = 0;

                for id in ids_to_remove {
                    match self.remove_point_internal(id.clone()) {
                        Ok(was_removed) => {
                            if was_removed {
                                removed_count += 1;
                                trace!(
                                    operation = "overwrite_removal",
                                    vector_id = %id,
                                    "Removed existing vector/codes for overwrite"
                                );
                            }
                        }
                        Err(e) => {
                            removal_errors += 1;
                            warn!(
                                operation = "overwrite_removal",
                                vector_id = %id,
                                error = %e,
                                "Failed to remove existing vector for overwrite"
                            );
                            errors.push(InsertError::Counted(format!(
                                "Failed to remove existing {}: {}",
                                id, e
                            )));
                        }
                    }
                }

                info!(
                    operation = "overwrite_removal_complete",
                    removed_count = removed_count,
                    removal_errors = removal_errors,
                    "Completed removal phase for overwrite"
                );
            }
        }

        // Phase 2: Add new vectors using the correct path based on current PQ state
        debug!(
            operation = "add_vectors_insertion_phase",
            current_state = self.get_storage_mode(),
            "Starting insertion phase"
        );

        for (id, vector, metadata) in parsed_data {
            let id_for_error = id.clone();

            // Use overwrite=false since we already handled removals above
            // The add_single_vector method will route to the correct path based on current PQ state
            match self.add_single_vector(id, vector, metadata, false) {
                Ok(inserted_new) => {
                    total_inserted += 1;
                    if inserted_new {
                        let mut count = self.vector_count.lock().unwrap();
                        *count += 1;
                    }

                    // Check training trigger (graceful failure handling)
                    if let Err(training_error) = self.maybe_trigger_training() {
                        warn!(
                            operation = "training_trigger",
                            error = %training_error,
                            vector_id = %id_for_error,
                            "Training trigger failed"
                        );
                        errors.push(InsertError::Training(format!(
                            "Training failed: {}",
                            training_error
                        )));
                    }
                }
                Err(e) => {
                    errors.push(InsertError::Vector {
                        id: id_for_error,
                        err: e,
                    });
                }
            }
        }

        (total_inserted, errors)
    }

    // 1. CORE VECTOR OPERATIONS (6 methods)
    /// 3-PATH ARCHITECTURE - Main router
    fn add_single_vector(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, Value>,
        overwrite: bool,
    ) -> PyResult<bool> {
        // Check if this is a new vector or an overwrite
        let is_new = {
            let id_map = self.id_map.read().unwrap();
            !id_map.contains_key(&id)
        };

        if !overwrite && !is_new {
            warn!(
                operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector with ID '{}' already exists",
                id
            )));
        }

        trace!(
            operation = "add_single_vector",
            vector_id = %id,
            is_new = is_new,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(id, vector, metadata)?;
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(id, vector, metadata)?;
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(id, vector, metadata)?;
        }

        Ok(is_new)
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Store processed vector directly (no additional processing)
        {
            let mut vectors = self.vectors.write().unwrap();
            vectors.insert(id.clone(), vector.clone()); // Already normalized
        }

        // Insert processed vector into HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert(&vector, internal_id); // Already normalized
        }

        trace!(
            operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(())
    }

    /// Path B: ID collection for consistent training
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        // 1. Store vector normally (single storage)
        self.add_raw_vector(id.clone(), vector, metadata)?;

        // SKIP TRAINING ID COLLECTION DURING PERSISTENCE REBUILD
        if self
            .rebuilding_from_persistence
            .load(std::sync::atomic::Ordering::Acquire)
        {
            trace!(
                operation = "add_with_id_collection",
                vector_id = %id,
                reason = "rebuilding_from_persistence",
                "Skipping training ID collection during rebuild"
            );
            return Ok(());
        }

        // 2. Collect ID for training (minimal memory overhead)
        if let Some(config) = &self.quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.clone());
                    let progress = (training_ids.len() as f32 / config.training_size as f32
                        * 100.0)
                        .min(100.0);

                    trace!(
                        operation = "training_id_collection",
                        vector_id = %id,
                        collected_count = training_ids.len(),
                        target_size = config.training_size,
                        progress_percent = progress,
                        "Training ID collected"
                    );

                    // Check if we've reached the threshold
                    if training_ids.len() >= config.training_size {
                        self.training_threshold_reached
                            .store(true, Ordering::Release);
                        info!(
                            operation = "training_threshold_reached",
                            collected_count = training_ids.len(),
                            target_size = config.training_size,
                            "Training threshold reached - ready for PQ training"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Quantize the vector
        let pq = self.pq.as_ref().unwrap();
        let codes = pq.quantize(&vector).map_err(|e| {
            error!(
                operation = "add_quantized_vector",
                vector_id = %id,
                error = %e,
                "Failed to quantize vector"
            );
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to quantize vector: {}",
                e
            ))
        })?;

        // Store quantized codes (always)
        {
            let mut pq_codes = self.pq_codes.write().unwrap();
            pq_codes.insert(id.clone(), codes.clone());
        }

        // Store raw vector only if configured to keep them
        if let Some(config) = &self.quantization_config {
            if config.storage_mode == StorageMode::QuantizedWithRaw {
                let mut vectors = self.vectors.write().unwrap();
                vectors.insert(id.clone(), vector.clone());
            }
            // If QuantizedOnly mode, we don't store raw vectors (saves memory)
        }

        // Insert codes into quantized HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert_pq_codes(&codes, internal_id);
        }

        trace!(
            operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes.len(),
            "Quantized vector added successfully"
        );

        Ok(())
    }

    /// TRAINING TRIGGER: Uses threshold flag for race condition safety
    #[instrument(level = "info", skip(self), fields(
        threshold_reached = self.training_threshold_reached.load(Ordering::Acquire),
        has_quantization = self.has_quantization()
    ))]
    fn maybe_trigger_training(&self) -> Result<(), String> {
        // Check atomic flag first (fast path)
        if !self.training_threshold_reached.load(Ordering::Acquire) {
            return Ok(());
        }

        // Only proceed if we have quantization config and aren't already trained
        if let Some(_config) = &self.quantization_config {
            if let Some(pq) = &self.pq {
                if !pq.is_trained() {
                    info!(
                        operation = "training_trigger",
                        "Training threshold reached - starting PQ training"
                    );
                    return self.train_quantization_from_ids();
                }
            }
        }

        Ok(())
    }

    /// TRAINING EXECUTION: Uses collected IDs for deterministic training set
    #[instrument(level = "info", skip(self), fields(
        has_pq = self.pq.is_some(),
        has_config = self.quantization_config.is_some()
    ))]
    fn train_quantization_from_ids(&self) -> Result<(), String> {
        let start_time = Instant::now();

        let pq = self.pq.as_ref().ok_or("PQ not available")?.clone();
        let config = self
            .quantization_config
            .as_ref()
            .ok_or("Config not available")?
            .clone();

        // Get consistent training set using collected IDs. `vectors` is taken
        // first because the declared lock order puts it above `training_ids`,
        // and every reader that holds both takes them that way round.
        let training_vectors = {
            let vectors = self.vectors.read().unwrap();
            let training_ids = self.training_ids.read().unwrap();

            // ADD EARLY CHECK:
            if training_ids.is_empty() {
                warn!(
                    operation = "pq_training",
                    reason = "no_training_ids",
                    "No training IDs available"
                );
                // Reset threshold to prevent repeated attempts
                self.training_threshold_reached
                    .store(false, Ordering::Release);
                return Err("No training IDs available for training".to_string());
            }

            let mut training_data = Vec::new();
            let mut missing_vectors = 0;

            for id in training_ids.iter() {
                if let Some(vector) = vectors.get(id) {
                    training_data.push(vector.clone());
                } else {
                    missing_vectors += 1;
                }
            }

            if missing_vectors > 0 {
                warn!(
                    operation = "pq_training",
                    missing_vectors = missing_vectors,
                    available_vectors = training_data.len(),
                    "Some training vectors were removed before training"
                );
            }

            debug!(
                operation = "pq_training_dataset",
                collected_ids = training_ids.len(),
                available_vectors = training_data.len(),
                target_size = config.training_size,
                "Training dataset prepared"
            );

            training_data
        };

        if training_vectors.len() < config.training_size {
            error!(
                operation = "pq_training",
                available = training_vectors.len(),
                required = config.training_size,
                "Insufficient vectors for training"
            );
            return Err(format!(
                "Insufficient vectors for training: need {}, have {} (some may have been removed)",
                config.training_size,
                training_vectors.len()
            ));
        }

        // Respect max_training_vectors limit
        let final_training_set = if let Some(max_training) = config.max_training_vectors {
            if training_vectors.len() > max_training {
                // Take first max_training vectors (deterministic)
                debug!(
                    operation = "pq_training_limit",
                    available = training_vectors.len(),
                    using = max_training,
                    "Limiting training set size"
                );
                training_vectors.into_iter().take(max_training).collect()
            } else {
                training_vectors
            }
        } else {
            training_vectors
        };

        info!(
            operation = "pq_training_start",
            training_vectors = final_training_set.len(),
            subvectors = config.subvectors,
            bits = config.bits,
            "Starting PQ training"
        );

        // Train the PQ model
        let training_start = Instant::now();
        pq.train(&final_training_set)?;
        let training_duration = training_start.elapsed();

        info!(
            operation = "pq_training_complete",
            training_vectors = final_training_set.len(),
            duration_ms = training_duration.as_millis(),
            "PQ training completed successfully"
        );

        // Clear training IDs (no longer needed)
        {
            let mut training_ids = self.training_ids.write().unwrap();
            training_ids.clear();
        }

        // Rebuild index with quantization
        debug!(
            operation = "pq_rebuild_start",
            "Rebuilding index with quantization"
        );
        let rebuild_start = Instant::now();
        let rebuild_success = self
            .rebuild_with_quantization_locked()
            .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
        let rebuild_duration = rebuild_start.elapsed();

        if rebuild_success {
            // The code size against the vector size. A `memory_savings_percent`
            // field used to sit beside it, carrying 1 - 1/compression_ratio,
            // which is the same number in another form and was labelled as a
            // saving the index does not make.
            let compression_ratio = (self.dim as f64 * 4.0) / pq.subvectors as f64;

            let total_duration_ms = start_time.elapsed().as_millis();
            info!(
                operation = "pq_complete",
                rebuild_duration_ms = rebuild_duration.as_millis(),
                compression_ratio = compression_ratio,
                total_duration_ms = total_duration_ms,
                "Index successfully rebuilt with quantization"
            );
        } else {
            error!(operation = "pq_rebuild", "Index rebuild returned false");
            return Err("Index rebuild returned false".to_string());
        }

        Ok(())
    }

    // 2. SEARCH OPERATIONS (2 methods)

    /// Decide whether a search reranks, and how far it over-fetches
    ///
    /// Rerank rescores the candidates the graph returns against raw vectors,
    /// so it needs a raw vector for every candidate. Three cases resolve to no
    /// rerank.
    ///
    /// A raw index already ranks by the raw distance, so over-fetching and
    /// rescoring would return the same page at a higher cost.
    ///
    /// A `quantized_only` index has no raw vector for the records added after
    /// training, and rescoring those against their reconstruction gains
    /// nothing, because the reconstruction carries exactly the information the
    /// ADC distance already used. Measured at 10,000 records of dimension 768,
    /// recall at `top_k` 10 over the code held records moved from 0.1320 to
    /// 0.1330 across one data seed and from 0.1440 to 0.1400 across another,
    /// which is noise in both directions. The mode keeps raw vectors for the
    /// records collected before training, one in ten at these settings, and
    /// rescoring only those would put two accuracies in one ranking for a
    /// tenth of the page.
    ///
    /// `rerank = 0` from the caller turns it off and restores the ADC scores.
    fn rerank_plan(&self, rerank: Option<usize>) -> Option<RerankPlan> {
        if rerank == Some(0) || !self.is_quantized() {
            return None;
        }

        let keeps_raw = self
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        Some(RerankPlan {
            factor: rerank.unwrap_or(DEFAULT_RERANK_FACTOR).max(1),
            distance: raw_distance_fn(&self.space),
        })
    }

    /// Raw search without Python objects (for benchmarking)
    fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // Concurrent read access to ID mapping, taken before the graph lock so the
        // traversal predicate can consult it without acquiring anything itself.
        let rev_map = self.rev_map.read().unwrap();
        let live = |internal_id: &usize| rev_map.contains_key(internal_id);

        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard
                .search(query, 10, 100, Some(&live))
                .unwrap_or_else(|_| Vec::new())
        }; // Lock released immediately

        hnsw_results
            .into_iter()
            .filter_map(|neighbor| {
                rev_map
                    .get(&neighbor.get_origin_id())
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }

    /// Parse input data into (id, vector, metadata) tuples with error collection
    fn parse_input_data(&self, data: &Bound<PyAny>) -> (ParsedRecords, Vec<String>) {
        let mut parsed_vectors = Vec::new();
        let mut errors = Vec::new();

        if let Ok(dict) = data.downcast::<PyDict>() {
            self.parse_dict_input_safe(dict, &mut parsed_vectors, &mut errors);
        } else if let Ok(list) = data.downcast::<PyList>() {
            self.parse_list_input_safe(list, &mut parsed_vectors, &mut errors);
        } else if let Ok(np_array) = data.downcast::<PyArray2<f32>>() {
            if let Err(e) = self.parse_numpy_input_safe(np_array, &mut parsed_vectors, &mut errors)
            {
                errors.push(format!("NumPy parsing error: {}", e));
            }
        } else {
            // Single vector
            match self.extract_single_vector_safe(data) {
                Ok(vector) => {
                    let id = self.generate_id();
                    parsed_vectors.push((id, vector, HashMap::new()));
                }
                Err(e) => {
                    errors.push(format!("Single vector error: {}", e));
                }
            }
        }

        (parsed_vectors, errors)
    }

    /// Safe dictionary parsing that collects errors
    fn parse_dict_input_safe(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) {
        // Check for single object format
        if dict.contains("id").unwrap_or(false)
            && (dict.contains("values").unwrap_or(false)
                || dict.contains("vector").unwrap_or(false))
        {
            // Single object format
            let vector_result = if let Ok(Some(values_item)) = dict.get_item("values") {
                self.extract_single_vector_safe(&values_item)
            } else if let Ok(Some(vector_item)) = dict.get_item("vector") {
                self.extract_single_vector_safe(&vector_item)
            } else {
                Err("Missing 'vector' or 'values' key".to_string())
            };

            match vector_result {
                Ok(vector) => {
                    let id = match dict.get_item("id") {
                        Ok(Some(id_item)) => id_item
                            .extract::<String>()
                            .unwrap_or_else(|_| self.generate_id()),
                        _ => self.generate_id(),
                    };

                    let metadata = match dict.get_item("metadata") {
                        Ok(Some(meta_item)) => {
                            if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                                self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                            } else {
                                HashMap::new()
                            }
                        }
                        _ => HashMap::new(),
                    };

                    parsed_vectors.push((id, vector, metadata));
                }
                Err(e) => {
                    let id = dict
                        .get_item("id")
                        .ok()
                        .flatten()
                        .and_then(|id_item| id_item.extract::<String>().ok())
                        .unwrap_or_else(|| "single_object".to_string());

                    errors.push(format!("Vector {}: {}", id, e));
                }
            }
        } else {
            // Batch format - try the existing parse_batch_format
            if let Err(e) = self.parse_batch_format(dict, parsed_vectors, errors) {
                errors.push(format!("Batch parsing error: {}", e));
            }
        }
    }

    /// Handle Format 3 & 5: Batch format - WORKING SOLUTION
    fn parse_batch_format(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        // Process each key path immediately without storing references

        // Try "vectors" key
        if let Some(vectors_item) = dict.get_item("vectors")? {
            if let Ok(list) = vectors_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = vectors_item.downcast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors, errors);
            }
        }

        // Try "embeddings" key
        if let Some(embeddings_item) = dict.get_item("embeddings")? {
            if let Ok(list) = embeddings_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = embeddings_item.downcast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors, errors);
            }
        }

        // Try "values" key
        if let Some(values_item) = dict.get_item("values")? {
            if let Ok(list) = values_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "values field must be a list in batch format",
                ));
            }
        }

        // No valid vector data found
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Missing vector data. Expected one of: 'vectors', 'embeddings', or 'values' key",
        ))
    }

    /// Helper method to process vector list (extracted to avoid code duplication)
    fn process_vector_list(
        &self,
        vectors: &Bound<PyList>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
    ) -> PyResult<()> {
        // Process each vector in the batch
        for (i, vector_item) in vectors.iter().enumerate() {
            let vector = self.extract_single_vector(&vector_item)?;

            // Extract ID from "ids" array
            let id = match dict.get_item("ids")? {
                Some(item) => {
                    let ids_list = item.downcast::<PyList>()?;
                    if i < ids_list.len() {
                        ids_list.get_item(i)?.extract::<String>()?
                    } else {
                        self.generate_id()
                    }
                }
                None => self.generate_id(),
            };

            // Extract metadata from "metadatas" or "metadata" arrays
            let meta = match dict
                .get_item("metadatas")?
                .or_else(|| dict.get_item("metadata").ok().flatten())
            {
                Some(item) => {
                    if let Ok(meta_list) = item.downcast::<PyList>() {
                        if i < meta_list.len() {
                            let metadata_item = meta_list.get_item(i)?;
                            if let Ok(meta_dict) = metadata_item.downcast::<PyDict>() {
                                self.python_dict_to_value_map(meta_dict)?
                            } else if metadata_item.is_none() {
                                HashMap::new()
                            } else {
                                let mut map = HashMap::new();
                                let value = Self::python_object_to_value(&metadata_item)?;
                                let key = if value.is_string() { "text" } else { "value" };
                                map.insert(key.to_string(), value);
                                map
                            }
                        } else {
                            HashMap::new()
                        }
                    } else {
                        HashMap::new()
                    }
                }
                None => HashMap::new(),
            };

            parsed_vectors.push((id, vector, meta));
        }

        Ok(())
    }

    /// Parse NumPy array with context (IDs and metadata from dict)
    fn parse_numpy_with_context(
        &self,
        np_array: &Bound<PyArray2<f32>>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        trace!(operation = "parse_numpy_context", shape = ?shape, "Processing NumPy array with context");

        if shape.len() != 2 || shape[1] != self.dim {
            error!(
                operation = "parse_numpy_context",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", self.dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape validation failed"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "NumPy array must have shape (N, {}), got {:?}",
                self.dim, shape
            )));
        }

        let flat = readonly.as_slice()?;
        let num_vectors = shape[0];

        // Extract IDs array
        let ids_list = dict
            .get_item("ids")?
            .and_then(|item| item.downcast::<PyList>().ok().cloned());

        // Extract metadata array
        let metadatas_list = dict
            .get_item("metadatas")?
            .or_else(|| dict.get_item("metadata").ok().flatten())
            .and_then(|item| item.downcast::<PyList>().ok().cloned());

        trace!(
            operation = "parse_numpy_context",
            num_vectors = num_vectors,
            has_ids = ids_list.is_some(),
            has_metadata = metadatas_list.is_some(),
            "Processing vectors with context"
        );

        let mut rejected = 0usize;

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

            // Resolve the caller's id before validating, so a rejected row can
            // be named without advancing the internal id counter for a record
            // that is never stored.
            let provided_id = match &ids_list {
                Some(ids) if i < ids.len() => ids.get_item(i)?.extract::<String>().ok(),
                _ => None,
            };

            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                let label = provided_id.clone().unwrap_or_else(|| format!("row_{}", i));
                error!(
                    operation = "parse_numpy_context",
                    error = "invalid_value",
                    vector_id = %label,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector {}: contains invalid value at index {}: {} (must be finite)",
                    label, component, value
                ));
                rejected += 1;
                continue;
            }

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());

            // Get ID from provided IDs or generate
            let id = match provided_id {
                Some(id) => id,
                None => self.generate_id(),
            };

            // Get metadata from provided metadata or use empty
            let metadata = if let Some(metas) = &metadatas_list {
                if i < metas.len() {
                    let meta_item = metas.get_item(i)?;
                    if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                        self.python_dict_to_value_map(meta_dict)?
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

            trace!(
                operation = "parse_numpy_vector",
                vector_index = i,
                vector_id = %id,
                metadata_keys = metadata.keys().len(),
                "Parsed NumPy vector with context"
            );

            parsed_vectors.push((id, processed_vector, metadata));
        }

        trace!(
            operation = "parse_numpy_context_complete",
            parsed_count = num_vectors - rejected,
            rejected_count = rejected,
            "NumPy parsing completed"
        );
        Ok(())
    }

    /// Safe list parsing that collects errors instead of failing immediately
    fn parse_list_input_safe(
        &self,
        list: &Bound<PyList>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) {
        for (item_index, item) in list.iter().enumerate() {
            if let Ok(item_dict) = item.downcast::<PyDict>() {
                // Extract vector safely
                let vector_result = if let Ok(Some(vector_item)) = item_dict.get_item("vector") {
                    self.extract_single_vector_safe(&vector_item)
                } else if let Ok(Some(values_item)) = item_dict.get_item("values") {
                    self.extract_single_vector_safe(&values_item)
                } else {
                    Err("Missing 'vector' or 'values' key in item".to_string())
                };

                match vector_result {
                    Ok(vector) => {
                        // Extract ID
                        let id = match item_dict.get_item("id") {
                            Ok(Some(id_item)) => id_item
                                .extract::<String>()
                                .unwrap_or_else(|_| self.generate_id()),
                            _ => self.generate_id(),
                        };

                        // Extract metadata
                        let metadata = match item_dict.get_item("metadata") {
                            Ok(Some(meta_item)) => {
                                if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                                    self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                                } else {
                                    // Handle non-dict metadata
                                    let mut map = HashMap::new();
                                    if let Ok(value) = Self::python_object_to_value(&meta_item) {
                                        let key = if value.is_string() { "text" } else { "value" };
                                        map.insert(key.to_string(), value);
                                    }
                                    map
                                }
                            }
                            _ => HashMap::new(),
                        };

                        parsed_vectors.push((id, vector, metadata));
                    }
                    Err(e) => {
                        // Collect error with item index and ID for context
                        let id = item_dict
                            .get_item("id")
                            .ok()
                            .flatten()
                            .and_then(|id_item| id_item.extract::<String>().ok())
                            .unwrap_or_else(|| format!("item_{}", item_index));

                        errors.push(format!("Vector {}: {}", id, e));
                    }
                }
            } else {
                // Direct vector item
                match self.extract_single_vector_safe(&item) {
                    Ok(vector) => {
                        let id = self.generate_id();
                        parsed_vectors.push((id, vector, HashMap::new()));
                    }
                    Err(e) => {
                        errors.push(format!("Item {}: {}", item_index, e));
                    }
                }
            }
        }
    }

    /// Safe NumPy parsing for error collection
    fn parse_numpy_input_safe(
        &self,
        np_array: &Bound<PyArray2<f32>>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> Result<(), String> {
        // This is the same as your current parse_numpy_input but returns Result<(), String>
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        if shape.len() != 2 || shape[1] != self.dim {
            return Err(format!(
                "NumPy array must have shape (N, {}), got {:?}",
                self.dim, shape
            ));
        }

        let flat = readonly
            .as_slice()
            .map_err(|e| format!("NumPy access error: {}", e))?;
        let num_vectors = shape[0];

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

            // A bare array carries no ids, so a rejected row is named by its
            // position and no id is generated for it.
            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                error!(
                    operation = "parse_numpy",
                    error = "invalid_value",
                    row = i,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector row_{}: contains invalid value at index {}: {} (must be finite)",
                    i, component, value
                ));
                continue;
            }

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());
            let id = self.generate_id();
            parsed_vectors.push((id, processed_vector, HashMap::new()));
        }

        Ok(())
    }

    /// Report the first non-finite component of a vector, if there is one
    ///
    /// The two NumPy branches read their rows straight out of the buffer, so
    /// they skip the per-value check that `extract_single_vector` runs. A NaN
    /// that reaches the graph degrades every later query rather than only the
    /// one that carried it, so both branches route through this.
    fn first_non_finite(vector: &[f32]) -> Option<(usize, f32)> {
        vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(index, value)| (index, *value))
    }

    /// Extract a single vector from various Python types (enhanced)
    fn extract_single_vector(&self, data: &Bound<PyAny>) -> PyResult<Vec<f32>> {
        let vector = if let Ok(array1d) = data.downcast::<PyArray1<f32>>() {
            // NumPy 1D array
            array1d.readonly().as_slice()?.to_vec()
        } else if let Ok(list) = data.downcast::<PyList>() {
            // Python list
            list.iter()
                .map(|item| item.extract::<f32>())
                .collect::<PyResult<Vec<f32>>>()?
        } else {
            // Direct extraction (e.g., from other numeric arrays)
            data.extract::<Vec<f32>>()?
        };

        // Comprehensive validation
        if vector.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Vector cannot be empty",
            ));
        }

        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }

        // Check for invalid values
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector contains invalid value at index {}: {} (must be finite)",
                    i, val
                )));
            }
        }

        // ✅ Apply space-specific processing
        Ok(self.process_vector_for_space(vector))
    }

    /// Generate a unique ID for a vector
    fn generate_id(&self) -> String {
        let id = self.get_next_id();
        format!("vec_{}", id)
    }

    /// Safe version of extract_single_vector that returns String errors instead of PyErr
    fn extract_single_vector_safe(&self, data: &Bound<PyAny>) -> Result<Vec<f32>, String> {
        let vector = if let Ok(array1d) = data.downcast::<PyArray1<f32>>() {
            array1d
                .readonly()
                .as_slice()
                .map_err(|e| format!("NumPy access error: {}", e))?
                .to_vec()
        } else if let Ok(list) = data.downcast::<PyList>() {
            list.iter()
                .map(|item| {
                    item.extract::<f32>()
                        .map_err(|e| format!("List item error: {}", e))
                })
                .collect::<Result<Vec<f32>, String>>()?
        } else {
            data.extract::<Vec<f32>>()
                .map_err(|e| format!("Vector extraction error: {}", e))?
        };

        // Same validation as extract_single_vector, but with String errors
        if vector.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }
        if vector.len() != self.dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "Vector contains invalid value at index {}: {}",
                    i, val
                ));
            }
        }

        Ok(self.process_vector_for_space(vector))
    }

    // 4. DATA CONVERSION & FILTERING (12 methods)
    // Helper methods for data conversion and filtering
    fn python_dict_to_value_map(
        &self,
        py_dict: &Bound<PyDict>,
    ) -> PyResult<HashMap<String, Value>> {
        let mut map = HashMap::new();

        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = Self::python_object_to_value(&value)?;
            map.insert(string_key, json_value);
        }

        Ok(map)
    }

    fn python_object_to_value(py_obj: &Bound<PyAny>) -> PyResult<Value> {
        if py_obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = py_obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = py_obj.extract::<i64>() {
            Ok(Value::Number(serde_json::Number::from(i)))
        } else if let Ok(f) = py_obj.extract::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(f) {
                Ok(Value::Number(num))
            } else {
                Ok(Value::String(f.to_string()))
            }
        } else if let Ok(s) = py_obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(py_list) = py_obj.downcast::<PyList>() {
            let mut vec = Vec::new();
            for item in py_list.iter() {
                vec.push(Self::python_object_to_value(&item)?);
            }
            Ok(Value::Array(vec))
        } else if let Ok(py_dict) = py_obj.downcast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in py_dict.iter() {
                let string_key = key.extract::<String>()?;
                let json_value = Self::python_object_to_value(&value)?;
                map.insert(string_key, json_value);
            }
            Ok(Value::Object(map))
        } else {
            Ok(Value::String(py_obj.to_string()))
        }
    }

    fn matches_filter(
        &self,
        metadata: &HashMap<String, Value>,
        filter: &HashMap<String, Value>,
    ) -> PyResult<bool> {
        for (field, condition) in filter {
            if !self.field_matches(metadata, field, condition)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn field_matches(
        &self,
        metadata: &HashMap<String, Value>,
        field: &str,
        condition: &Value,
    ) -> PyResult<bool> {
        let field_value = match metadata.get(field) {
            Some(value) => value,
            None => return Ok(false),
        };

        match condition {
            // A map is always the operator form. Direct equality against a
            // nested object has no syntax of its own, because the two forms
            // would be indistinguishable, so it is written {"eq": {...}}.
            Value::Object(ops) => self.evaluate_value_conditions(field_value, ops),
            _ => Ok(Self::values_equal(field_value, condition)),
        }
    }

    /// Reject an operator the engine does not implement, before any record is
    /// examined. Checking during evaluation is not enough on its own, because a
    /// record that lacks the field never reaches the operator and a filter that
    /// fails an earlier field short circuits, so whether the typo is noticed
    /// would depend on the data. `evaluate_operator` is the only list of
    /// operator names, so validation cannot disagree with dispatch about what
    /// is known. The field value here is a placeholder, which is sound because
    /// every operator helper is total and the unknown operator arm is the one
    /// error the dispatch can produce.
    fn validate_filter_conditions(&self, filter: &HashMap<String, Value>) -> PyResult<()> {
        for condition in filter.values() {
            if let Value::Object(operations) = condition {
                for (op, target_value) in operations {
                    self.evaluate_operator(&Value::Null, op, target_value)?;
                }
            }
        }
        Ok(())
    }

    fn evaluate_value_conditions(
        &self,
        field_value: &Value,
        operations: &serde_json::Map<String, Value>,
    ) -> PyResult<bool> {
        for (op, target_value) in operations {
            if !self.evaluate_operator(field_value, op, target_value)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn evaluate_operator(
        &self,
        field_value: &Value,
        op: &str,
        target_value: &Value,
    ) -> PyResult<bool> {
        match op {
            "eq" => Ok(Self::values_equal(field_value, target_value)),
            "ne" => Ok(!Self::values_equal(field_value, target_value)),
            "gt" => self.compare_values(field_value, target_value, CmpOrdering::is_gt),
            "gte" => self.compare_values(field_value, target_value, CmpOrdering::is_ge),
            "lt" => self.compare_values(field_value, target_value, CmpOrdering::is_lt),
            "lte" => self.compare_values(field_value, target_value, CmpOrdering::is_le),
            "contains" => self.value_contains(field_value, target_value),
            "startswith" => self.value_starts_with(field_value, target_value),
            "endswith" => self.value_ends_with(field_value, target_value),
            "in" => self.value_in_array(field_value, target_value),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown filter operation: {}",
                op
            ))),
        }
    }

    /// Equality over the whole value tree. Numbers compare by magnitude, so a
    /// stored integer matches an equal float, and arrays and objects compare
    /// element by element so their numbers do too. Every other pairing keeps
    /// `serde_json` equality, which is why a boolean is not equal to a number
    /// and a numeric string is not equal to a number.
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(left), Value::Number(right)) => {
                Self::compare_numbers(left, right) == Some(CmpOrdering::Equal)
            }
            (Value::Array(left), Value::Array(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right.iter())
                        .all(|(item, other)| Self::values_equal(item, other))
            }
            (Value::Object(left), Value::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, item)| {
                        right
                            .get(key)
                            .is_some_and(|other| Self::values_equal(item, other))
                    })
            }
            _ => a == b,
        }
    }

    /// Order two JSON numbers by magnitude. Integers compare as integers, so
    /// two values above 2^53 that share an f64 representation stay distinct,
    /// and a mixed pair compares exactly rather than through a lossy cast.
    fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<CmpOrdering> {
        match (Self::numeric_value(a)?, Self::numeric_value(b)?) {
            (NumericValue::Integer(left), NumericValue::Integer(right)) => Some(left.cmp(&right)),
            (NumericValue::Float(left), NumericValue::Float(right)) => left.partial_cmp(&right),
            (NumericValue::Integer(left), NumericValue::Float(right)) => {
                Self::compare_integer_to_float(left, right)
            }
            (NumericValue::Float(left), NumericValue::Integer(right)) => {
                Self::compare_integer_to_float(right, left).map(CmpOrdering::reverse)
            }
        }
    }

    /// `i128` holds every `serde_json` integer, which is an `i64` or a `u64`,
    /// so the widening is lossless.
    fn numeric_value(number: &serde_json::Number) -> Option<NumericValue> {
        if let Some(value) = number.as_i64() {
            Some(NumericValue::Integer(value as i128))
        } else if let Some(value) = number.as_u64() {
            Some(NumericValue::Integer(value as i128))
        } else {
            number.as_f64().map(NumericValue::Float)
        }
    }

    /// Order an integer against a float without casting the integer to f64.
    /// The float splits into a truncated part, which converts to an integer
    /// exactly, and a fraction that breaks the tie when the integer parts are
    /// equal. A float outside the `i128` range saturates on conversion, and
    /// the comparison still lands on the correct side because every integer
    /// reaching this point fits in a `u64`.
    fn compare_integer_to_float(integer: i128, float: f64) -> Option<CmpOrdering> {
        if float.is_nan() {
            return None;
        }
        if float.is_infinite() {
            return Some(if float.is_sign_positive() {
                CmpOrdering::Less
            } else {
                CmpOrdering::Greater
            });
        }

        let truncated = float.trunc();
        let integer_part = truncated as i128;
        Some(match integer.cmp(&integer_part) {
            CmpOrdering::Equal => truncated.partial_cmp(&float)?,
            ordering => ordering,
        })
    }

    fn compare_values<F>(&self, a: &Value, b: &Value, op: F) -> PyResult<bool>
    where
        F: Fn(CmpOrdering) -> bool,
    {
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                Ok(Self::compare_numbers(n1, n2).is_some_and(op))
            }
            _ => Ok(false),
        }
    }

    fn value_contains(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.contains(s2)),
            (Value::Array(arr), val) => Ok(arr.iter().any(|item| Self::values_equal(item, val))),
            _ => Ok(false),
        }
    }

    fn value_starts_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.starts_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_ends_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.ends_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_in_array(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match target {
            Value::Array(arr) => Ok(arr.iter().any(|item| Self::values_equal(item, field))),
            _ => Ok(false),
        }
    }

    fn value_map_to_python(
        &self,
        value_map: &HashMap<String, Value>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        for (key, value) in value_map {
            let py_value = Self::value_to_python_object(value, py)?;
            dict.set_item(key, py_value)?;
        }

        Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
    }

    fn value_to_python_object(value: &Value, py: Python<'_>) -> PyResult<PyObject> {
        let py_obj = match value {
            Value::Null => py.None(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into_any(),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.into_pyobject(py)?.to_owned().unbind().into_any()
                } else if let Some(f) = n.as_f64() {
                    f.into_pyobject(py)?.to_owned().unbind().into_any()
                } else {
                    n.to_string()
                        .into_pyobject(py)?
                        .to_owned()
                        .unbind()
                        .into_any()
                }
            }
            Value::String(s) => s.clone().into_pyobject(py)?.unbind().into_any(),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(Self::value_to_python_object(item, py)?)?;
                }
                py_list.unbind().into_any()
            }
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (k, v) in obj {
                    py_dict.set_item(k, Self::value_to_python_object(v, py)?)?;
                }
                py_dict.unbind().into_any()
            }
        };

        Ok(py_obj)
    }

    // 5. BATCH SEARCH METHODS (3 methods)
    /// Internal batch search method for multiple query vectors
    #[instrument(level = "debug", skip(self, vectors, filter_conditions, params, py), fields(
        batch_size = vectors.len(),
        top_k = params.top_k,
        ef = params.ef,
        return_vector = params.return_vector,
        has_filter = filter_conditions.is_some(),
        rerank_factor = params.rerank.map(|plan| plan.factor)
    ), err)]
    fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let start_time = Instant::now();

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                error!(
                    operation = "batch_search_validation",
                    vector_index = i,
                    expected_dim = self.dim,
                    actual_dim = vector.len(),
                    "Vector dimension mismatch in batch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector {}: dimension mismatch: expected {}, got {}",
                    i,
                    self.dim,
                    vector.len()
                )));
            }

            // The same value check the single query path applies. A non-finite
            // component survives normalization, because the norm of a vector
            // containing one is not greater than zero, and the search then
            // returns hits whose scores carry no distance information. The
            // message names the batch entry as well as the component, so one
            // bad vector is findable in a batch of thousands.
            for (component, &value) in vector.iter().enumerate() {
                if !value.is_finite() {
                    error!(
                        operation = "batch_search_validation",
                        vector_index = i,
                        value_index = component,
                        value = value,
                        "Vector in batch contains invalid value"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Vector {} in batch contains invalid value at index {}: {} (must be finite)",
                        i, component, value
                    )));
                }
            }
        }

        // Choose strategy based on batch size
        let result = if vectors.len() <= 5 {
            trace!(
                operation = "batch_search_strategy",
                strategy = "sequential",
                "Using sequential processing"
            );
            self.batch_search_sequential(vectors, filter_conditions, params, py)
        } else {
            trace!(
                operation = "batch_search_strategy",
                strategy = "parallel",
                "Using parallel processing"
            );
            self.batch_search_parallel(vectors, filter_conditions, params, py)
        };

        // ✅ ENTERPRISE: Add duration timing to hot path
        let duration_ms = start_time.elapsed().as_millis();
        debug!(
            operation = "batch_search_complete",
            batch_size = vectors.len(),
            duration_ms = duration_ms,
            "Batch search completed"
        );

        result
    }

    /// Sequential batch processing (for small batches)
    fn batch_search_sequential(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.allow_threads(|| -> PyResult<Vec<QueryHits>> {
            // The read guard is taken before the graph lock and held across every
            // query in the batch, so the traversal predicate below is a hash lookup
            // rather than a lock acquisition.
            let rev_map = self.rev_map.read().unwrap();
            let live = |internal_id: &usize| rev_map.contains_key(internal_id);

            let hnsw_guard = self.hnsw.read().unwrap();
            let vector_store = self.vectors.read().unwrap();
            let code_store = self.pq_codes.read().unwrap();
            let metadata_store = self.vector_metadata.read().unwrap();

            // The same over-fetch the single query path applies, so a batch of
            // one query returns what that query returns on its own.
            let fetch_k = params.fetch_k(rev_map.len());

            let mut all_results = Vec::with_capacity(vectors.len());

            for vector in vectors {
                // FIX: Process each query vector for space
                let processed_query = self.process_vector_for_space(vector.clone());

                let neighbors = hnsw_guard
                    .search(&processed_query, fetch_k, params.ef, Some(&live))
                    .unwrap_or_else(|_| Vec::new());

                let mut scored: Vec<(&String, f32)> = Vec::with_capacity(neighbors.len());

                for neighbor in neighbors {
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        // Apply filter if specified
                        if let Some(filter_conds) = filter_conditions {
                            if let Some(meta) = metadata_store.get(ext_id) {
                                if !self.matches_filter(meta, filter_conds)? {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let score = match params.rerank.as_ref() {
                            Some(plan) => rescore_candidate(
                                plan,
                                &processed_query,
                                ext_id,
                                &vector_store,
                                self.pq.as_ref(),
                                &code_store,
                            )
                            .unwrap_or(f32::INFINITY),
                            None => neighbor.distance,
                        };

                        scored.push((ext_id, score));
                    }
                }

                if params.rerank.is_some() {
                    take_best(&mut scored, params.top_k);
                }

                let mut query_results = Vec::with_capacity(scored.len());
                for (ext_id, score) in scored {
                    let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                    let vector_data = if params.return_vector {
                        vector_store.get(ext_id).cloned()
                    } else {
                        None
                    };

                    query_results.push((ext_id.clone(), score, metadata, vector_data));
                }

                all_results.push(query_results);
            }

            Ok(all_results)
        })?;

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }

            output.push(py_batch);
        }

        Ok(output)
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let span = tracing::Span::current();
        let rust_results = py.allow_threads(|| -> PyResult<Vec<QueryHits>> {
            let results: PyResult<Vec<QueryHits>> = vectors
                .par_iter()
                .map(|vector| -> PyResult<QueryHits> {
                    let _entered = span.clone().entered();
                    // FIX: Process each query vector for space
                    let processed_query = self.process_vector_for_space(vector.clone());

                    // Taken before the graph lock, as in the other two search paths,
                    // so every path acquires these two locks in the same order.
                    let rev_map = self.rev_map.read().unwrap();
                    let live = |internal_id: &usize| rev_map.contains_key(internal_id);

                    // The same over-fetch the other two search paths apply.
                    let fetch_k = params.fetch_k(rev_map.len());

                    // Brief HNSW search (individual lock per query)
                    let neighbors = {
                        let hnsw_guard = self.hnsw.read().unwrap();
                        hnsw_guard
                            .search(&processed_query, fetch_k, params.ef, Some(&live))
                            .unwrap_or_else(|_| Vec::new())
                    };

                    // Concurrent data lookup
                    let vector_store = self.vectors.read().unwrap();
                    let code_store = self.pq_codes.read().unwrap();
                    let metadata_store = self.vector_metadata.read().unwrap();

                    let mut scored: Vec<(&String, f32)> = Vec::with_capacity(neighbors.len());

                    for neighbor in neighbors {
                        let internal_id = neighbor.get_origin_id();

                        if let Some(ext_id) = rev_map.get(&internal_id) {
                            // Apply filter if specified
                            if let Some(filter_conds) = filter_conditions {
                                if let Some(meta) = metadata_store.get(ext_id) {
                                    if !self.matches_filter(meta, filter_conds)? {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }

                            let score = match params.rerank.as_ref() {
                                Some(plan) => rescore_candidate(
                                    plan,
                                    &processed_query,
                                    ext_id,
                                    &vector_store,
                                    self.pq.as_ref(),
                                    &code_store,
                                )
                                .unwrap_or(f32::INFINITY),
                                None => neighbor.distance,
                            };

                            scored.push((ext_id, score));
                        }
                    }

                    if params.rerank.is_some() {
                        take_best(&mut scored, params.top_k);
                    }

                    let mut query_results = Vec::with_capacity(scored.len());
                    for (ext_id, score) in scored {
                        let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                        let vector_data = if params.return_vector {
                            vector_store.get(ext_id).cloned()
                        } else {
                            None
                        };

                        query_results.push((ext_id.clone(), score, metadata, vector_data));
                    }

                    Ok(query_results)
                })
                .collect();

            results
        })?;

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }

            output.push(py_batch);
        }

        Ok(output)
    }

    // 6. PERSISTENCE INTEGRATION METHODS (2 methods)

    /// Load an index from a .zdb directory structure (Phase 2)
    pub fn load(path: &str) -> PyResult<Self> {
        crate::persistence::load_index(path)
    }

    /// Save HNSW graph using hnsw-rs native file_dump
    #[instrument(level = "info", skip(self), fields(
        vector_count = self.get_vector_count(),
        path = %path.display()
    ))]
    fn save_hnsw_graph(&self, path: &Path) -> PyResult<()> {
        debug!(
            operation = "save_hnsw_graph_start",
            "Starting HNSW graph save"
        );

        // EMPTY INDEX CHECK:
        let vector_count = self.get_vector_count();
        if vector_count == 0 {
            debug!(
                operation = "save_hnsw_graph",
                reason = "empty_index",
                "Skipping HNSW graph dump - index is empty"
            );
            return Ok(());
        }

        let hnsw_guard = self.hnsw.read().unwrap();

        let dump_result = match &*hnsw_guard {
            DistanceType::Cosine(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine",
                    "Using Cosine distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L2(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2",
                    "Using L2 distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L1(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1",
                    "Using L1 distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::CosinePQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine_pq",
                    "Using Cosine-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L2PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2_pq",
                    "Using L2-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L1PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1_pq",
                    "Using L1-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
        };

        match dump_result {
            Ok(basename) => {
                debug!(
                    operation = "save_hnsw_graph_complete",
                    basename = %basename,
                    files_created = %["hnsw.graph", "hnsw.data"].iter()
                        .map(|ext| format!("{}.{}", basename, ext))
                        .collect::<Vec<_>>()
                        .join(", "),
                    "HNSW graph saved successfully"
                );
                Ok(())
            }
            Err(e) => {
                error!(operation = "save_hnsw_graph", error = %e, "HNSW graph dump failed");
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "HNSW graph dump failed: {}",
                    e
                )))
            }
        }
    }

    // ============================================================================
    // PERSISTENCE Minimal Empty Constructor and SETTERS
    // ============================================================================
    /// Minimal constructor for persistence loading - creates empty index with config
    /// No validation needed since config comes from trusted saved state
    pub fn new_empty(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
    ) -> Self {
        let space_normalized = space.to_lowercase();
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        let hnsw = DistanceType::new_raw(
            &space_normalized,
            m,
            expected_size,
            max_layer,
            ef_construction,
        );

        HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: None,
            pq: None,
            pq_codes: RwLock::new(HashMap::new()),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: chrono::Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }

    /// Set ID mappings (for persistence loading only)
    pub(crate) fn set_id_mappings(
        &mut self,
        id_map: HashMap<String, usize>,
        rev_map: HashMap<usize, String>,
    ) {
        *self.id_map.write().unwrap() = id_map;
        *self.rev_map.write().unwrap() = rev_map;
    }

    /// Set counters (for persistence loading only)
    pub(crate) fn set_counters(&mut self, id_counter: usize, vector_count: usize) {
        *self.id_counter.lock().unwrap() = id_counter;
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Set the vector count alone (for persistence loading only)
    ///
    /// Separate from `set_counters` because the id counter must keep whatever
    /// the graph rebuild advanced it to. Rewinding it would hand out internal
    /// ids the rebuild has already used.
    pub(crate) fn set_vector_count(&mut self, vector_count: usize) {
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Replace the stored record data with what was read from disk
    ///
    /// For persistence loading only, and only after the graph rebuild. The
    /// rebuild routes every record through add(), which stores whatever vector
    /// it was handed, so a record that was reconstructed from PQ codes would
    /// otherwise be kept at full width. Writing the three maps back leaves the
    /// loaded index holding exactly what was saved.
    pub(crate) fn restore_storage_maps(
        &mut self,
        vectors: HashMap<String, Vec<f32>>,
        pq_codes: HashMap<String, Vec<u8>>,
        vector_metadata: HashMap<String, HashMap<String, Value>>,
    ) {
        *self.vectors.write().unwrap() = vectors;
        *self.pq_codes.write().unwrap() = pq_codes;
        *self.vector_metadata.write().unwrap() = vector_metadata;
    }

    /// Rebuild the graph from the stored PQ codes (for persistence loading only)
    ///
    /// Requires a trained product quantizer, which the loader installs before
    /// any rebuild runs. Replaces the raw graph `new_empty` built with a fresh
    /// PQ graph and inserts every record's codes under the internal id restored
    /// from mappings.bin, so the loaded index is quantized exactly as the saved
    /// one was and no vector is reconstructed to full width on the way.
    ///
    /// A record that has a raw vector but no stored codes is quantized through
    /// the loaded codebook rather than dropped. An intact directory saved by a
    /// trained index holds codes for every record, so that path only runs on a
    /// directory that lost pq_codes.bin while keeping its raw vectors. A record
    /// missing from mappings.bin is assigned a fresh internal id for the same
    /// reason: every record must come back.
    ///
    /// Returns (records inserted, quantized from raw, remapped).
    pub(crate) fn rebuild_graph_from_codes(
        &mut self,
        pq_codes: &HashMap<String, Vec<u8>>,
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<(usize, usize, usize), String> {
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                return Err(
                    "the quantized graph rebuild requires a trained product quantizer".to_string(),
                )
            }
        };

        let mut extra: Vec<(String, Vec<u8>)> = Vec::new();
        for (id, vector) in vectors {
            if !pq_codes.contains_key(id) {
                let codes = pq.quantize(vector).map_err(|e| {
                    format!(
                        "record '{}' has a raw vector but no stored PQ codes, and quantizing \
                         it through the loaded codebook failed: {}",
                        id, e
                    )
                })?;
                extra.push((id.clone(), codes));
            }
        }

        // NB_LAYER_MAX, matching every other construction site in this file.
        let max_layer = 16;
        let new_hnsw = DistanceType::new_pq(
            &self.space,
            self.m,
            self.expected_size,
            max_layer,
            self.ef_construction,
            pq,
        );

        let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(pq_codes.len() + extra.len());
        let mut lost: Vec<(&String, &Vec<u8>)> = Vec::new();
        {
            let id_map = self.id_map.read().unwrap();
            for (id, codes) in pq_codes
                .iter()
                .chain(extra.iter().map(|(id, codes)| (id, codes)))
            {
                match id_map.get(id) {
                    Some(&internal_id) => batch.push((codes, internal_id)),
                    None => lost.push((id, codes)),
                }
            }
        }
        let remapped = lost.len();
        for (id, codes) in lost {
            let internal_id = self.get_next_id();
            self.id_map.write().unwrap().insert(id.clone(), internal_id);
            self.rev_map
                .write()
                .unwrap()
                .insert(internal_id, id.clone());
            batch.push((codes, internal_id));
        }

        // Filled before it is installed, and installed under one write guard, so
        // the graph the index holds is never a partly rebuilt one. The batch
        // insert forks to rayon when it is large, which must not happen while
        // the graph's write guard is held.
        if !batch.is_empty() {
            new_hnsw.insert_batch_pq(&batch)?;
        }
        self.replace_graph(new_hnsw);

        Ok((batch.len(), extra.len(), remapped))
    }

    /// Set quantization config (for persistence loading only)
    pub(crate) fn set_quantization_config(&mut self, config: Option<QuantizationConfig>) {
        self.quantization_config = config;
    }

    /// Set PQ instance (for persistence loading only)
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<crate::pq::PQ>>) {
        self.pq = pq;
    }

    /// Set training threshold reached flag (for persistence loading only)
    pub(crate) fn set_training_threshold_reached(&mut self, value: bool) {
        self.training_threshold_reached
            .store(value, std::sync::atomic::Ordering::Release);
    }

    // ============================================================================
    // PERSISTENCE GETTERS - For accessing private fields from persistence module
    // ============================================================================

    /// Get the vector dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// Get the distance space (cosine, l2, l1) - changed to a more idiomatic getter
    pub fn space(&self) -> &str {
        // Changed from get_space to space
        &self.space
    }

    /// Get the maximum number of bidirectional links per node
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Get the construction parameter ef_construction
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Get the expected size parameter
    pub fn get_expected_size(&self) -> usize {
        self.expected_size
    }

    /// Get the current ID counter value (thread-safe)
    pub fn get_id_counter(&self) -> usize {
        *self.id_counter.lock().unwrap()
    }

    /// Get read access to the vectors HashMap (thread-safe)
    pub fn get_vectors(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<f32>>> {
        self.vectors.read().unwrap()
    }

    /// Get read access to the PQ codes HashMap (thread-safe)
    pub fn get_pq_codes(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<u8>>> {
        self.pq_codes.read().unwrap()
    }

    /// Get read access to the vector metadata HashMap (thread-safe)
    pub fn get_vector_metadata(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<String, HashMap<String, Value>>> {
        self.vector_metadata.read().unwrap()
    }

    /// Get read access to the ID map (external ID -> internal ID)
    pub fn get_id_map(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, usize>> {
        self.id_map.read().unwrap()
    }

    /// Get read access to the reverse ID map (internal ID -> external ID)
    pub fn get_rev_map(&self) -> std::sync::RwLockReadGuard<'_, HashMap<usize, String>> {
        self.rev_map.read().unwrap()
    }

    /// Get reference to the quantization configuration
    pub fn get_quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    /// Get reference to the PQ instance
    pub fn get_pq(&self) -> Option<&Arc<crate::pq::PQ>> {
        self.pq.as_ref()
    }

    /// Helper to get quantization subvectors count
    pub fn get_quantization_subvectors(&self) -> usize {
        self.quantization_config
            .as_ref()
            .map(|config| config.subvectors)
            .unwrap_or(1)
    }

    /// Get the index creation timestamp
    pub fn get_created_at(&self) -> &str {
        &self.created_at
    }

    /// Get read access to training IDs (for persistence)
    pub fn get_training_ids(&self) -> std::sync::RwLockReadGuard<'_, Vec<String>> {
        self.training_ids.read().unwrap()
    }

    /// Get training threshold reached flag (for persistence)
    pub fn get_training_threshold_reached(&self) -> bool {
        self.training_threshold_reached
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Set training IDs (for persistence loading only)
    pub(crate) fn set_training_ids(&mut self, ids: Vec<String>) {
        *self.training_ids.write().unwrap() = ids;
    }
}

#[cfg(test)]
mod tests {
    use super::{raw_distance_fn, rescore_candidate, take_best, DistPQ, RerankPlan, SearchParams};
    use crate::distance::{CosineDist, L1Dist, L2Dist};
    use crate::pq::PQ;
    // `DistCosine` is the `anndists` implementation these distances replaced.
    // The two graph guard tests keep it on purpose. They guard patches in the
    // vendored crate rather than anything about the distance, their data is
    // deliberately unnormalised, and holding the distance fixed keeps the
    // orphan counts their comments record comparable across relays.
    use hnsw_rs::prelude::{DistCosine, Distance, Hnsw};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, OnceLock};

    // Scale for the quantized graph tests. Small enough for CI and large
    // enough that the neighbour selection heuristic runs, which needs
    // `search_layer` to return more than `2 * M` candidates.
    //
    // Eight bits is the setting the README recommends and it is not
    // negotiable here for speed. Six was tried, and on data with this cluster
    // structure a 64 centroid codebook is coarse enough that every record in a
    // cluster quantizes to the same codes in every subvector. Their distance is
    // then genuinely zero, the diversity heuristic ties for real, and roughly
    // 45 percent of nodes come out with one neighbour. That is the quantizer
    // being too coarse for the data rather than the defect these tests guard,
    // but it is indistinguishable from it at the assertion, so the tests run at
    // the width real indexes use. k-means over 256 centroids is what makes them
    // the slowest in the crate.
    const PQ_N: usize = 1200;
    const PQ_NQ: usize = 100;
    const PQ_DIM: usize = 32;
    const PQ_SUBVECTORS: usize = 8;
    const PQ_BITS: usize = 8;
    const PQ_M: usize = 16;
    const PQ_EF_C: usize = 200;

    /// Clustered unit vectors. Fifty Gaussian centres, points drawn as a centre
    /// plus 0.15 times a Gaussian perturbation, then L2 normalised, with the
    /// centres deliberately left unnormalised. This is the shape real
    /// embeddings have and the specification every quantized measurement in
    /// this project uses, so the figures in the relay reports and the
    /// thresholds below describe the same data.
    ///
    /// Uniform noise was tried and rejected. Its spread is too small relative
    /// to the centre separation, so records within a cluster quantize to the
    /// same codes, their distance is genuinely zero, and the graph partially
    /// collapses for a reason that has nothing to do with what these tests
    /// assert.
    fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gauss = |rng: &mut StdRng| {
            let u: f32 = rng.random::<f32>().max(1e-12);
            let v: f32 = rng.random::<f32>();
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        };
        let centres: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| gauss(&mut rng)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % 50];
                let mut v: Vec<f32> = (0..dim).map(|d| c[d] + 0.15 * gauss(&mut rng)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    /// The rescored score has to be the number a raw index would report, not a
    /// second implementation of the same formula that happens to agree today.
    #[test]
    fn rerank_scores_come_from_the_raw_distances() {
        let a = vec![0.6f32, 0.8, 0.0, -0.5];
        let b = vec![-0.2f32, 0.3, 0.9, 0.4];

        assert_eq!(
            raw_distance_fn("cosine")(&a, &b),
            CosineDist {}.eval(&a, &b)
        );
        assert_eq!(raw_distance_fn("l2")(&a, &b), L2Dist {}.eval(&a, &b));
        assert_eq!(raw_distance_fn("l1")(&a, &b), L1Dist {}.eval(&a, &b));

        // An unrecognised space falls back the way `DistanceType::new_raw`
        // does, so the score still matches the graph that was built.
        assert_eq!(
            raw_distance_fn("nonsense")(&a, &b),
            CosineDist {}.eval(&a, &b)
        );
    }

    fn plan(factor: usize) -> RerankPlan {
        RerankPlan {
            factor,
            distance: raw_distance_fn("cosine"),
        }
    }

    fn params(top_k: usize, rerank: Option<RerankPlan>) -> SearchParams {
        SearchParams {
            top_k,
            ef: 100,
            return_vector: false,
            rerank,
        }
    }

    /// The over-fetch is the factor the caller asked for, and it cannot ask the
    /// graph for more nodes than the index holds.
    #[test]
    fn fetch_k_over_fetches_and_stays_bounded() {
        assert_eq!(params(10, None).fetch_k(10_000), 10);
        assert_eq!(params(10, Some(plan(1))).fetch_k(10_000), 10);
        assert_eq!(params(10, Some(plan(20))).fetch_k(10_000), 200);

        // Capped at the live record count rather than at the product.
        assert_eq!(params(10, Some(plan(20))).fetch_k(50), 50);

        // A page larger than the index still asks for the page, so a short
        // result stays short rather than being cut further.
        assert_eq!(params(500, Some(plan(20))).fetch_k(50), 500);

        // A factor big enough to overflow the multiply degrades to a full scan.
        assert_eq!(params(usize::MAX, Some(plan(2))).fetch_k(50), usize::MAX);
        assert_eq!(params(10, Some(plan(usize::MAX))).fetch_k(50), 50);
    }

    /// The page comes back ascending and cut to size, and a non-finite score
    /// orders rather than panicking the sort.
    #[test]
    fn take_best_orders_ascending_and_cuts() {
        let mut scored = vec![
            ("d", 0.9f32),
            ("a", 0.1),
            ("e", f32::INFINITY),
            ("c", 0.5),
            ("b", 0.2),
        ];
        take_best(&mut scored, 3);
        assert_eq!(
            scored.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );

        let mut short = vec![("a", 2.0f32), ("b", 1.0)];
        take_best(&mut short, 10);
        assert_eq!(short.len(), 2);
        assert_eq!(short[0].0, "b");
    }

    /// A candidate is scored from its raw vector where the index kept one and
    /// from its reconstruction otherwise, and both land on the space's own
    /// distance scale.
    #[test]
    fn rescore_prefers_the_raw_vector_and_falls_back_to_the_codes() {
        let data = clustered(64, 8, 99);
        let pq = Arc::new(PQ::new(8, 4, 4, 16, None));
        pq.train(&data).unwrap();

        let query = data[0].clone();
        let stored = data[7].clone();
        let codes = pq.quantize(&stored).unwrap();

        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut pq_codes: HashMap<String, Vec<u8>> = HashMap::new();
        vectors.insert("kept".to_string(), stored.clone());
        pq_codes.insert("kept".to_string(), codes.clone());
        pq_codes.insert("coded".to_string(), codes.clone());

        let p = plan(1);

        // The raw vector wins where there is one, so the score is exact.
        let exact = rescore_candidate(&p, &query, "kept", &vectors, Some(&pq), &pq_codes);
        assert_eq!(exact, Some(CosineDist {}.eval(&query, &stored)));

        // Codes alone still score, against the reconstruction.
        let approximate =
            rescore_candidate(&p, &query, "coded", &vectors, Some(&pq), &pq_codes).unwrap();
        let reconstructed = pq.reconstruct(&codes).unwrap();
        assert_eq!(approximate, CosineDist {}.eval(&query, &reconstructed));

        // Neither is unscoreable rather than silently zero, which is what keeps
        // an unscored candidate from displacing a scored one.
        assert!(rescore_candidate(&p, &query, "absent", &vectors, Some(&pq), &pq_codes).is_none());
        assert!(rescore_candidate(&p, &query, "coded", &vectors, None, &pq_codes).is_none());
    }

    struct PqFixture {
        data: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
        pq: Arc<PQ>,
        codes: Vec<Vec<u8>>,
    }

    /// One trained codebook shared by every quantized graph test, because
    /// k-means over 256 centroids is the expensive part and it is the same
    /// codebook each time. The graphs themselves are built per test, since
    /// that is what is under assertion.
    fn fixture() -> &'static PqFixture {
        static FIXTURE: OnceLock<PqFixture> = OnceLock::new();
        FIXTURE.get_or_init(|| {
            let all = clustered(PQ_N + PQ_NQ, PQ_DIM, 42);
            let data: Vec<Vec<f32>> = all[..PQ_N].to_vec();
            let queries: Vec<Vec<f32>> = all[PQ_N..].to_vec();

            let pq = Arc::new(PQ::new(PQ_DIM, PQ_SUBVECTORS, PQ_BITS, 1000, None));
            pq.train(&data).expect("pq training");
            let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
            let codes = pq.quantize_batch(&refs).expect("quantization");

            PqFixture {
                data,
                queries,
                pq,
                codes,
            }
        })
    }

    /// Build the quantized graph exactly as `insert_pq_codes` does, one code
    /// vector at a time with no query table set.
    fn build_pq_graph(pq: Arc<PQ>, codes: &[Vec<u8>]) -> Hnsw<'static, u8, DistPQ> {
        let hnsw = Hnsw::new(PQ_M, codes.len(), 16, PQ_EF_C, DistPQ::new(pq));
        for (i, c) in codes.iter().enumerate() {
            hnsw.insert((c.as_slice(), i));
        }
        hnsw
    }

    /// Layer zero adjacency keyed by origin id, each list sorted.
    fn layer_zero_adjacency(hnsw: &Hnsw<'static, u8, DistPQ>) -> Vec<(usize, Vec<usize>)> {
        let mut adj: Vec<(usize, Vec<usize>)> = hnsw
            .get_point_indexation()
            .into_iter()
            .map(|p| {
                let mut v: Vec<usize> = p.get_neighborhood_id()[0].iter().map(|x| x.d_id).collect();
                v.sort_unstable();
                (p.get_origin_id(), v)
            })
            .collect();
        adj.sort_unstable_by_key(|(id, _)| *id);
        adj
    }

    /// The strongest assertion available about the quantized graph: that the
    /// data it is built from has any effect on it at all.
    ///
    /// Build the same graph twice, in the same insertion order so the level
    /// sequence is identical, once with each id holding its own codes and once
    /// with every id holding a different record's codes. A graph built on real
    /// distances comes out different. A graph built on a constant comes out
    /// byte for byte identical, which is what `DistPQ::eval` produced for every
    /// release that shipped quantization: it returned infinity whenever no
    /// query table was set, and no insertion path sets one.
    ///
    /// Measured on the shipped v0.4.1 code at 10,000 records, layer zero
    /// adjacency was identical for 10,000 of 10,000 nodes.
    #[test]
    fn quantized_graph_depends_on_the_data() {
        let f = fixture();

        let own = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

        let mut shuffled = f.codes.clone();
        shuffled.reverse();
        let other = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &shuffled));

        assert_eq!(own.len(), PQ_N);
        assert_eq!(other.len(), PQ_N);

        let identical = own
            .iter()
            .zip(other.iter())
            .filter(|((id_a, a), (id_b, b))| id_a == id_b && a == b)
            .count();

        assert!(
            identical * 20 < PQ_N,
            "layer zero adjacency is identical for {} of {} nodes when every id is given a \
             different record's codes, so the graph is not being built on the codes. \
             DistPQ::eval is returning a constant on the insertion path.",
            identical,
            PQ_N
        );
    }

    /// A graph whose distances all tie leaves every node with one neighbour,
    /// because the diversity heuristic in `select_neighbours` rejects a
    /// candidate that is at least as close to an already chosen neighbour as it
    /// is to the new point, and under a total tie that is every candidate after
    /// the first. Measured on the shipped code, layer zero out-degree was
    /// exactly one for 99.64 percent of nodes and a traversal reached 33 of
    /// 10,000.
    #[test]
    fn quantized_graph_layer_zero_out_degree() {
        let f = fixture();
        let adj = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

        let degenerate = adj.iter().filter(|(_, n)| n.len() <= 1).count();
        assert!(
            degenerate * 100 < PQ_N,
            "{} of {} nodes have layer zero out-degree of one or less; the quantized graph \
             has collapsed to a star",
            degenerate,
            PQ_N
        );

        let total: usize = adj.iter().map(|(_, n)| n.len()).sum();
        let mean = total as f64 / PQ_N as f64;
        assert!(
            mean > (PQ_M as f64) / 2.0,
            "mean layer zero out-degree is {:.2}, expected well above {} for m = {}",
            mean,
            PQ_M / 2,
            PQ_M
        );
    }

    /// Quantized search has to find the right answers, not merely return the
    /// right number of them. Measured on the shipped code at this scale the
    /// graph reached 33 nodes of 1,200 and recall was under one percent, so
    /// the threshold below fails against the old behaviour by a wide margin.
    ///
    /// The ceiling is the quantizer rather than the graph, so this asserts a
    /// floor and not equality with the raw path.
    #[test]
    fn quantized_graph_recall_against_brute_force() {
        const K: usize = 10;

        let f = fixture();
        let (data, queries) = (&f.data, &f.queries);
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        let mut hits = 0usize;
        let mut returned = usize::MAX;
        for q in queries.iter() {
            let found = {
                let _lut = hnsw.get_distance().install_query_lut(q).expect("query lut");
                let dummy = vec![0u8; PQ_SUBVECTORS];
                hnsw.search(&dummy, K, 100)
            };
            returned = returned.min(found.len());

            let mut truth: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    (
                        v.iter().zip(q.iter()).map(|(x, y)| (x - y) * (x - y)).sum(),
                        j,
                    )
                })
                .collect();
            truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let truth: HashSet<usize> = truth[..K].iter().map(|x| x.1).collect();

            hits += found.iter().filter(|n| truth.contains(&n.d_id)).count();
        }

        assert_eq!(returned, K, "a top {} request came back short", K);

        let recall = hits as f64 / (K * PQ_NQ) as f64;
        assert!(
            recall > 0.30,
            "quantized recall at top {} is {:.4}, which is far below what these codes support",
            K,
            recall
        );
    }

    /// The whole graph has to be reachable, which is the property the shipped
    /// code lost most visibly. Asking for more results than the graph can reach
    /// is what exposed it: a request for 1,000 came back with 34.
    #[test]
    fn quantized_graph_is_fully_reachable() {
        let f = fixture();
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        let found = {
            let _lut = hnsw
                .get_distance()
                .install_query_lut(&f.data[0])
                .expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            hnsw.search(&dummy, PQ_N, PQ_N)
        };

        assert_eq!(
            found.len(),
            PQ_N,
            "a request for all {} records returned {}, so the traversal cannot reach the \
             whole graph",
            PQ_N,
            found.len()
        );
    }

    /// The symmetric distance must not leak into search. With a query table set
    /// the distance has to be the asymmetric one, byte for byte as before, so
    /// a quantized graph and a raw graph rank a query's own record the same way.
    #[test]
    fn quantized_search_still_uses_the_query_table() {
        let f = fixture();
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        // The ADC distance from a query to a stored code, computed directly.
        let query = &f.data[7];
        let lut = f.pq.compute_adc_lut(query).expect("adc lut");
        let expected: f32 = f.codes[7]
            .iter()
            .enumerate()
            .map(|(sv, &c)| lut[sv][c as usize])
            .sum();

        let found = {
            let _lut = hnsw
                .get_distance()
                .install_query_lut(query)
                .expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            hnsw.search(&dummy, 10, 200)
        };

        assert!(
            found.iter().any(|n| n.d_id == 7),
            "a record could not find itself"
        );

        // Every distance the search reports must be the asymmetric one. The
        // symmetric table would give a different number here, since it compares
        // the dummy query's all-zero codes rather than the query itself.
        for n in found.iter() {
            let adc: f32 = f.codes[n.d_id]
                .iter()
                .enumerate()
                .map(|(sv, &c)| lut[sv][c as usize])
                .sum();
            assert!(
                (n.distance - adc).abs() <= 1e-4 * adc.max(1.0),
                "search reported {} for record {} where its asymmetric distance is {}, \
                 so the query path is no longer using the query table",
                n.distance,
                n.d_id,
                adc
            );
        }

        // The record's own ADC distance is the smallest of the ten returned,
        // which is what the ranking has to produce.
        assert!(
            found[0].distance <= expected + 1e-4,
            "the top hit scored {} against the query's own record at {}",
            found[0].distance,
            expected
        );
    }

    /// Guards the vendored hnsw_rs patch that files reverse links at the
    /// layer being processed instead of at the inserting point's own top
    /// layer. Without the patch, points assigned a level above zero lose
    /// their layer-zero inbound adjacency and can become unreachable to
    /// similarity search, and at this index size roughly one to two
    /// percent of self-queries fail. A failure here means the patch was
    /// lost, most likely during an hnsw_rs upgrade. See
    /// vendor/hnsw_rs/ZEUSDB-PATCH.md.
    ///
    /// Insertion is sequential on purpose. `parallel_insert` assigns levels
    /// in whatever order threads reach the level generator, so the graph
    /// varies between runs even under the fixed seed and the test was
    /// intermittently red. Building one vector at a time makes the graph a
    /// function of the data and the parameters alone, which is also the
    /// path every non-quantized index takes through `add()`. The defect
    /// this test guards is not specific to the parallel path.
    #[test]
    fn self_query_reachability() {
        const N: usize = 3000;
        const DIM: usize = 32;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let hnsw = Hnsw::new(16, N, 16, 200, DistCosine {});
        for (i, v) in data.iter().enumerate() {
            hnsw.insert((v.as_slice(), i));
        }

        let failures: Vec<usize> = (0..N)
            .filter(|&i| hnsw.search(&data[i], 1, 64).first().map(|n| n.d_id) != Some(i))
            .collect();

        assert!(
            failures.is_empty(),
            "{} of {} points cannot find themselves by self-query (first: {:?}); \
             the hnsw_rs reverse link layer patch is missing",
            failures.len(),
            N,
            &failures[..failures.len().min(10)]
        );
    }

    /// Guards the vendored hnsw_rs patch that stops the layer-zero overflow
    /// pop from evicting a point's last inbound link. Without the patch the
    /// pop always discards the farthest entry, so a point whose only inbound
    /// link happens to be that entry is left with no layer-zero in-edge and
    /// becomes an orphan that no search can reach through the graph. A
    /// failure here means the patch was lost, most likely during an hnsw_rs
    /// upgrade. See vendor/hnsw_rs/ZEUSDB-PATCH.md.
    ///
    /// In-degree is counted from the adjacency lists rather than from the
    /// patch's own counters, so the assertion holds against the graph itself
    /// and stays meaningful if the counters are ever wrong.
    ///
    /// `M` is 4 rather than the shipped 16 on purpose. The layer-zero
    /// neighbour cap is `2 * M`, so a small `M` fills lists early and makes
    /// the overflow pop frequent, which is the only site that can strand a
    /// point. Whether the shipped `M` of 16 strands points depends on the
    /// data model rather than on index size. On uniform vectors like these
    /// it strands none up to 30,000 points, while on clustered data, 50
    /// Gaussian clusters at sigma 0.15 in 768 dimensions, it strands 6 of
    /// 10,000. A small `M` lets this uniform generator fail fast instead,
    /// and the unpatched crate strands 24 of these 5,000 points.
    ///
    /// Insertion is sequential for the same reason as `self_query_reachability`.
    #[test]
    fn layer_zero_in_degree() {
        const N: usize = 5000;
        const DIM: usize = 128;
        const M: usize = 4;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let hnsw = Hnsw::new(M, N, 16, 200, DistCosine {});
        for (i, v) in data.iter().enumerate() {
            hnsw.insert((v.as_slice(), i));
        }

        let mut in_degree = vec![0usize; N];
        let mut nb_seen = 0usize;
        for point in hnsw.get_point_indexation() {
            nb_seen += 1;
            for neighbour in &point.get_neighborhood_id()[0] {
                in_degree[neighbour.d_id] += 1;
            }
        }
        assert_eq!(nb_seen, N, "walked {} points, expected {}", nb_seen, N);

        let orphans: Vec<usize> = (0..N).filter(|&i| in_degree[i] == 0).collect();

        assert!(
            orphans.is_empty(),
            "{} of {} points have zero layer-zero in-degree (first: {:?}); \
             the hnsw_rs overflow pop guard is missing",
            orphans.len(),
            N,
            &orphans[..orphans.len().min(10)]
        );
    }
}
