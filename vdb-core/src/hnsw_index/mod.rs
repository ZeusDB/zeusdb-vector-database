//! The vector index: its state, the locks that protect it, and the Python
//! surface it presents.
//!
//! # What lives here and what does not
//!
//! This file holds the `HNSWIndex` struct, the one `#[pymethods]` block, and the
//! two types the dump header pins in place. Everything else is in a child
//! module, and a child can read the private fields because it is a descendant of
//! this one. The `#[pymethods]` block cannot be split: PyO3 accepts a second one
//! only under its `multiple-pymethods` feature, which would add a dependency
//! this crate has removed.
//!
//! | module | what it covers |
//! |---|---|
//! | `construct` | building an index and validating the declaration |
//! | `input` | turning Python input into records and query vectors |
//! | `insert` | insertion, replacement, removal, compaction |
//! | `search` | the four paths that reach the graph |
//! | `training` | fitting the codebook and rebuilding over the codes |
//! | `stats` | what the index reports about itself |
//! | `persist` | the accessors and setters `persistence.rs` speaks to |
//!
//! # Why `DistPQ` is declared here
//!
//! It used to be unable to live anywhere else. `Hnsw::file_dump` wrote
//! `std::any::type_name::<D>()` of the distance into the dump header, and both
//! the loader and the vendored `load_hnsw_with_dist` compared it by exact
//! equality. `type_name` is the full module path of the **declaration**, so
//! declaring `DistPQ` anywhere else changed what every save wrote and stopped
//! every saved quantized index from loading. That is also why this module is a
//! directory rather than a rename.
//!
//! ZeusDB's format carries a `graph::dump::GraphKind` discriminant instead, so
//! the pin is gone and the declaration is free to move. It stays here for now,
//! since moving it is a change to make on its own.
//!
//! `distance.rs` re-exports the name so that every call site imports its
//! distances from one place.

mod construct;
#[cfg(test)]
mod graph_guard_tests;
mod input;
mod insert;
mod persist;
mod search;
mod stats;
mod training;

use crate::conversion::{python_dict_to_value_map, value_map_to_python};
use crate::filter::validate_filter_conditions;
// The graph and everything the graph crate supplies arrive through the seam.
// `Distance` is the one name from that crate this file still needs, because
// `DistPQ` below implements it. See the note at the top of `graph.rs`.
use crate::graph::{Distance, VectorGraph};
use crate::pq::PQ;
use crate::rerank::{RerankCalibration, SearchParams};
use insert::InsertError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ✅ ENTERPRISE: Structured logging imports
use tracing::{debug, error, info, instrument, trace, warn};

/// Records accepted by `add`, after parsing and before insertion, as
/// (external id, vector, metadata). The vector is still in its input form and
/// has not been normalized for the index space yet.
///
/// It holds no Python object, which is what makes it the boundary the
/// interpreter lock is released across. `input` produces it and `insert`
/// consumes it.
pub(crate) type ParsedRecords = Vec<(String, Vec<f32>, HashMap<String, Value>)>;
/// Layers every graph this crate builds is created with.
///
/// It is the vendored crate's `NB_LAYER_MAX`, which is `pub(crate)` there and so
/// cannot be named from here. Every construction site passed the literal 16 with
/// a comment saying it matched the others, which was five statements of one
/// fact and a claim a reader had to check by searching. A dump written at any
/// other layer count is refused on load, so this is part of the on-disk contract
/// rather than a tuning knob; see `DUMP_LAYERS` in `graph.rs`.
const MAX_LAYER: usize = 16;
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
pub(crate) struct QueryLut;
impl Drop for QueryLut {
    fn drop(&mut self) {
        QUERY_LUT.with(|slot| *slot.borrow_mut() = None);
    }
}
/// Custom distance function for Product Quantization using ADC
///
/// This lives here rather than beside the raw distances in `distance.rs`
/// because `std::any::type_name::<DistPQ>()` used to be written into every
/// saved graph dump and checked on load, so moving the type to another module
/// changed that string and stopped every previously saved quantized index from
/// loading. ZeusDB's format records a discriminant instead, so the constraint
/// is gone and only the habit remains.
#[derive(Clone)]
pub struct DistPQ {
    /// Reference to the PQ instance for accessing centroids
    pq: Arc<PQ>,
}
impl DistPQ {
    pub fn new(pq: Arc<PQ>) -> Self {
        DistPQ { pq }
    }

    /// Bytes one code occupies, which is the length of the dummy query the
    /// seam hands the traversal. The codebook is private to this type, so the
    /// seam asks rather than reaching into it.
    pub(crate) fn subvectors(&self) -> usize {
        self.pq.subvectors
    }

    /// Compute this query's ADC table and install it for the calling thread.
    ///
    /// The returned guard must be held for the whole traversal. Dropping it
    /// early returns the thread to graph construction mode, where `eval` reads
    /// the codebook's symmetric table instead.
    pub(crate) fn install_query_lut(&self, query: &[f32]) -> Result<QueryLut, String> {
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
    /// dummy code vector `VectorGraph::search` passes, the real query lives in
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
/// `skip_from_py_object` because nothing extracts an `AddResult`. It is the
/// return type of `add` and appears in no argument position, in this crate or
/// in the Python layer. PyO3 0.29 derives `FromPyObject` for a `#[pyclass]`
/// that is `Clone` and warns that the derive becomes opt-in, so the choice has
/// to be stated. Opting in would generate an extraction path no caller reaches.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
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
///
/// Two locks sit outside the order. `writers` is taken by the mutating Python
/// entry points before any guard and never by an internal helper; see the
/// field. `rerank_calibration` is never held together with any other guard:
/// training and the loader write it with nothing held, and both readers take
/// it alone. The locks inside `PQ` are leaves, since nothing in `pq.rs` can
/// name an index guard, so they may be taken under any of the above but no
/// index guard may be taken under them, which no path does.
///
/// Taking the same guard twice on one thread is forbidden even for reads.
/// The standard library queues readers behind a waiting writer, so a second
/// read on the thread already holding one deadlocks the moment a writer lands
/// between them, which is how `get_stats` used to hang against training id
/// collection.
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

    /// What training measured about how deep this index's codes bury a true
    /// neighbour, which is what the default rerank fetch is derived from.
    ///
    /// Written once by `calibrate_rerank` at training completion and by the
    /// loader from `quantization.json`. `None` on an unquantized index, on a
    /// `quantized_only` one, before training, and on an index trained before
    /// the calibration existed. See `RerankCalibration`.
    rerank_calibration: RwLock<Option<RerankCalibration>>,

    // Index-level metadata (simple, infrequently accessed)
    metadata: Mutex<HashMap<String, String>>,

    /// The raw vector store.
    ///
    /// Holds every record for an unquantized index and under
    /// `quantized_with_raw`. Under `quantized_only` it holds the records
    /// collected before training and nothing after: the quantization rebuild
    /// releases them once their codes are stored, and the loader drops them
    /// from a directory written before that was true. A trained
    /// `quantized_only` index therefore holds no raw vector anywhere.
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
    hnsw: RwLock<VectorGraph>,

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

    /// Set while a load rebuilds the graph, so the rebuild does not refill the
    /// training collection with the ids it is replaying.
    ///
    /// Private, and written only through `set_rebuilding_from_persistence`.
    /// It was the one field of this struct that `persistence.rs` named, and a
    /// field the storage layer can reach is a field the storage layer can leave
    /// set, which would suppress training collection for the life of the index.
    rebuilding_from_persistence: AtomicBool,

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
    pub(super) fn replace_graph(&self, new_hnsw: VectorGraph) {
        let old = {
            let mut hnsw_guard = self.hnsw.write().unwrap();
            std::mem::replace(&mut *hnsw_guard, new_hnsw)
        };
        drop(old);
    }
}
#[pymethods]
impl HNSWIndex {
    /// Get quantization configuration and status
    pub fn get_quantization_info(&self) -> Option<Py<PyAny>> {
        self.quantization_info()
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
    ///
    /// Re-encodes whatever raw vectors the index still holds through the
    /// trained codebook and rebuilds the graph from the stored codes. It never
    /// retrains the codebook; training runs exactly once, on the `add` that
    /// reaches `training_size`. A trained `quantized_only` index holds no raw
    /// vectors, so there the rebuild proceeds from the codes alone, and under
    /// either mode nothing is lost by calling it. Returns false when there is
    /// no trained quantizer or nothing stored to rebuild from.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization()
    ), err)]
    pub fn rebuild_with_quantization(&self, py: Python<'_>) -> PyResult<bool> {
        // The whole rebuild runs with the interpreter lock released, the mutation
        // guard included. Waiting for another writer while holding the lock would
        // stall every Python thread in the process for the length of that writer,
        // which is the failure `add` releasing the lock would otherwise create.
        py.detach(|| {
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
        let (inserted, insert_errors) = py.detach(|| {
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
    /// rescores the candidates against raw vectors. Omitted, the fetch is
    /// derived from the live record count; see `SearchParams::fetch_k`. An
    /// integer of 1 or more pulls that many candidates per requested result,
    /// which is a fixed multiple of the page and does not move with the corpus.
    /// Zero turns rerank off and restores the ADC scores and ordering. It has
    /// no effect on a raw index or on a `quantized_only` one, both of which
    /// never rerank; see `rerank_plan`.
    ///
    /// `ef_search` has no effect on a reranked quantized search. Below the
    /// fetch it is discarded, because `Hnsw::search_filter` raises the
    /// traversal width to the number of neighbours asked for and cannot return
    /// more results than its candidate list holds. Above the fetch it buys no
    /// recall, because the candidates a fetch returns are limited by the ADC
    /// ordering rather than by the traversal, and quadrupling it moves recall
    /// at 10 by at most 0.008. The default fetch is at least 250 and the
    /// default `ef_search` is 100, so changing `ef_search` alone changes
    /// nothing on a reranked search at the defaults.
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
    ) -> PyResult<Py<PyAny>> {
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
            rerank_factor = params.rerank.and_then(|plan| plan.factor),
            "Search parameters configured"
        );

        let filter_conditions = filter.map(python_dict_to_value_map).transpose()?;

        // Reject an unrecognised operator before the search runs. Checking it
        // per record would make the error depend on the data, because a record
        // that lacks the field never reaches the operator at all.
        if let Some(conditions) = filter_conditions.as_ref() {
            validate_filter_conditions(conditions)?;
        }

        // Detect batch vs single query with comprehensive input support
        let result: Py<PyAny> = if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
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
        } else if let Ok(np_array) = vector.cast::<PyArray2<f32>>() {
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
            let query_vector = if let Ok(array1d) = vector.cast::<PyArray1<f32>>() {
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

            let results = self.single_search_internal(
                &processed_query,
                filter_conditions.as_ref(),
                params,
                py,
            )?;
            PyList::new(py, results)?.into()
        };

        // ✅ ENTERPRISE: Add duration timing to hot path with actual result count
        let duration_ms = start_time.elapsed().as_millis();
        let results_count = {
            let any = result.bind(py);
            match any.cast::<PyList>() {
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
    /// the `conversion` module it calls. `save_hnsw_graph` reaches
    /// `graph::dump::write_dump`, which names PyO3 nowhere.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.save_locked(path))
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
    /// reconstruction of the code where one does not. Under `quantized_only`
    /// that is every record once training completes, the training records
    /// included, since the rebuild releases their raw vectors the moment their
    /// codes are stored. The returned value is then an approximation rather
    /// than the value supplied. Measured on 16 dimensional data with 4
    /// subvectors and 8 bits, a reconstructed vector differed from the stored
    /// unit vector by 0.066 at the worst component and sat at cosine similarity
    /// 0.991 to it. Under `quantized_with_raw` every record keeps its raw
    /// vector and returns exactly. `get_stats()["raw_vectors_stored"]` is what
    /// tells the two apart in aggregate.
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
                dict.set_item("metadata", value_map_to_python(&metadata, py)?)?;

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
    ///
    /// The figures, and why `total_memory_mb` is not the resident set, are on
    /// `collect_stats`.
    pub fn get_stats(&self) -> HashMap<String, String> {
        self.collect_stats()
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
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        let mut results = Vec::new();
        for id in id_map.keys().take(number) {
            let metadata = vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = value_map_to_python(&metadata, py)?;
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
    /// `vectors=` is the live record count in every storage mode. See
    /// `info_string`.
    pub fn info(&self) -> String {
        self.info_string()
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
        py.detach(|| {
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
    /// `VectorGraph::new_raw` and `insert_batch`, which are the same shape as
    /// the quantized pair already listed there.
    pub fn compact(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.compact_locked())
    }

    /// Get performance characteristics and limitations
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        self.performance_info()
    }

    /// Concurrent benchmark for search performance
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(
        &self,
        query_count: usize,
        max_threads: Option<usize>,
    ) -> PyResult<HashMap<String, f64>> {
        self.benchmark_reads(query_count, max_threads)
    }
}
