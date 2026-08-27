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
//! # Why this module is a directory rather than a rename
//!
//! `Hnsw::file_dump` wrote `std::any::type_name::<D>()` of the distance into
//! the dump header, and both the loader and the vendored `load_hnsw_with_dist`
//! compared it by exact equality. `type_name` is the full module path of the
//! **declaration**, so while `DistPQ` was declared here, renaming this module
//! changed what every save wrote and stopped every saved quantized index from
//! loading. ZeusDB's format carries a `graph::dump::GraphKind` discriminant
//! instead, and `DistPQ` now lives in `distance.rs` beside the other four
//! implementors, so neither constraint remains.

mod construct;
// The declaration rules, so that `persistence::load_config` applies the same
// ones to `config.json` that `build` applies to a caller's arguments.
pub(crate) use construct::{
    validate_index_parameters, validate_space_supports_quantization, warn_if_selection_disabled,
};
#[cfg(test)]
mod graph_guard_tests;
mod input;
mod insert;
/// Every lock on this type, with its place in the declared acquisition order
/// asserted on a debug build.
pub(crate) mod locks;
mod persist;
mod search;
mod stats;
mod training;

use crate::columns::{ColumnStore, Selection};
use crate::conversion::{python_dict_to_value_map, value_map_to_python};
use crate::filter::{compile_filter, matches_filter};
// The graph and everything the graph crate supplies arrive through the seam.
// See the note at the top of `graph.rs`.
use crate::error::Error;
use crate::graph::VectorGraph;
use crate::pq::PQ;
use crate::rerank::{RawVectors, RerankCalibration, SearchParams};
use insert::InsertError;
use locks::{order, MutexAt, RwLockAt};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
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

/// Largest `top_k` a search may ask for.
///
/// `top_k` sizes the candidate search through the default `ef_search` of
/// twice `top_k`, and `search_layer` sizes its two candidate heaps from that
/// width, 8 bytes a slot, before it visits a node. The allocation is not
/// fallible, so `search(top_k=2**40)` asked for 17,592,186,044,416 bytes and
/// **aborted the process** with exit status 3221226505, and `top_k=2**33`
/// died the same way asking for 137 GB. Nothing checked either argument.
///
/// The ceiling is four times the largest page any comparable engine serves,
/// Milvus at 16,384, and six times Elasticsearch's 10,000, so no real caller
/// is refused. At the ceiling the heaps are 2 MiB and the result list is
/// 65,536 Python dicts, which is slow and is what was asked for.
const MAX_TOP_K: usize = 65_536;

/// Largest `ef_search` a search may pass, being the default `ef_search` at
/// the largest `top_k`.
///
/// The same two heaps as `MAX_TOP_K`, reached directly. `search(ef_search=2**40)`
/// asked for 8,796,093,022,208 bytes and aborted. At the ceiling the heaps are
/// 2 MiB, and a search at that width on a corpus smaller than it is an
/// exhaustive scan, which is the slowest thing a search can be and is bounded
/// by the corpus.
const MAX_EF_SEARCH: usize = 2 * MAX_TOP_K;
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

    /// The id of every record this call put in the index, in insertion order.
    ///
    /// **It lines up with `total_inserted` and with nothing else.**
    /// `len(ids) == total_inserted` on every result this crate produces. A
    /// record rejected by parsing contributes no id, because none was ever
    /// allocated for it, and a record rejected by insertion contributes none
    /// either, because it is not in the index. So this is not positionally
    /// aligned with the input, and a caller who needs that alignment reads
    /// `errors`, which names each rejection by its id or by its position.
    ///
    /// Why it exists. An `add` that is given no ids generates them, as
    /// `vec_{counter}`, and until this field the generated ids were
    /// unreachable: the caller had to list the index and guess which entries
    /// were new. The llama-index adapter probes for exactly this attribute and,
    /// not finding it, returns the ids it supplied, which on a batch where any
    /// node lacked an id is a shorter list of the wrong values.
    ///
    /// An overwrite reports the id it accepted, since a replacement is an
    /// insertion as far as this counts.
    #[pyo3(get)]
    pub ids: Vec<String>,
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
/// id_map < rev_map < hnsw < pq_codes < vector_metadata < columns
///        < training_ids < metadata < id_counter < vector_count
/// ```
///
/// **This order is checked rather than believed.** Every field below is a
/// [`locks::RwLockAt`] or a [`locks::MutexAt`] carrying its rank as a const
/// generic, and on a debug build each acquisition asserts that the thread holds
/// none of the same lock and nothing ranked above it. See [`locks`] for what
/// that catches, what it costs and what it misses. In release the wrappers are
/// the standard types by another name.
///
/// `columns` sits directly below `vector_metadata` because every path that
/// writes one writes the other, and a filtered search holds both: the columns
/// to decide which records match and the metadata to fill the page it returns.
///
/// This exists because search and mutation now overlap. Until the receivers
/// were relaxed, PyO3's exclusive borrow kept every mutating method away from
/// every search, so no reader and no writer were ever in flight together and
/// the acquisition order could not matter. It matters now. A search holds
/// `rev_map` for its whole traversal and reads the graph under it, so a removal
/// taking the graph before `rev_map`, which is what it used to do, deadlocks
/// against it on the first interleaving that lands.
///
/// `vectors` used to sit between `hnsw` and `pq_codes` here. The lock went with
/// the field when the raw vectors moved into the graph's own store, which the
/// graph guard already covers, so the order is one shorter than it was.
///
/// One further rule, which the order alone does not express. No path forks to
/// rayon while holding a write guard. Mutations are serialised against each
/// other by `writers`, so a read guard held across a fork can only ever be
/// blocked by that one writer, and a fork under a write guard is exactly the
/// case where the pool's workers can all end up waiting on the forking thread.
///
/// Four locks sit outside the order. `writers` is taken by the mutating Python
/// entry points before any guard and never by an internal helper; see the
/// field. `rerank_calibration`, `training_completed_at` and `created_at` are
/// never held together with any other guard: training and the loader write them
/// with nothing held, and every reader takes them alone. The registry ranks
/// `writers` above everything and the other three below everything, which is
/// the half of that claim a rank can state. The locks inside `PQ` are leaves,
/// since nothing in `pq.rs` can name an index guard, so they may be taken under
/// any of the above but no index guard may be taken under them, which no path
/// does.
///
/// Taking the same guard twice on one thread is forbidden even for reads.
/// The standard library queues readers behind a waiting writer, so a second
/// read on the thread already holding one deadlocks the moment a writer lands
/// between them, which is how `get_stats` used to hang against training id
/// collection. The registry asserts this on every acquisition in a debug build,
/// so it fires in an ordinary single threaded test rather than waiting for a
/// writer to land in the window.
#[pyclass]
pub struct HNSWIndex {
    dim: usize,
    space: String,
    /// The graph degree, written by `rebuild` and read everywhere else.
    ///
    /// An atomic rather than a plain field because `rebuild` takes `&self`, as
    /// every other mutating entry point on this type does. It is written only
    /// under `writers`, which every mutating entry point takes first, and read
    /// by the saver, the stats and the three rebuild paths. A search never
    /// reads it. `expected_size` is the same for the same reason.
    m: AtomicUsize,
    ef_construction: AtomicUsize,
    expected_size: AtomicUsize,

    // Quantization configuration and PQ instance
    quantization_config: Option<QuantizationConfig>,
    pq: Option<Arc<PQ>>,
    pq_codes: RwLockAt<{ order::PQ_CODES }, HashMap<String, Vec<u8>>>, // PQ codes storage

    /// What training measured about how deep this index's codes bury a true
    /// neighbour, which is what the default rerank fetch is derived from.
    ///
    /// Written once by `calibrate_rerank` at training completion and by the
    /// loader from `quantization.json`. `None` on an unquantized index, on a
    /// `quantized_only` one, before training, and on an index trained before
    /// the calibration existed. See `RerankCalibration`.
    rerank_calibration: RwLockAt<{ order::RERANK_CALIBRATION }, Option<RerankCalibration>>,

    // Index-level metadata (simple, infrequently accessed)
    metadata: MutexAt<{ order::METADATA }, HashMap<String, String>>,

    vector_metadata: RwLockAt<{ order::VECTOR_METADATA }, HashMap<String, HashMap<String, Value>>>,

    /// One column per field declared at `create()`, addressed by internal id.
    ///
    /// **This is what a filtered search reads instead of walking every
    /// record.** A filter naming only declared fields compiles to a bitmap over
    /// internal ids, which both the exact scan and the graph traversal consume,
    /// and `vector_metadata` is then read only for the records the page
    /// returns. A filter naming an undeclared field cannot be answered here.
    /// Where the declared fields still bound which records could match, the
    /// metadata is read for those alone; where they bound nothing, and on an
    /// index with no declaration, it falls back to the walk, which is what
    /// every index did before this existed.
    ///
    /// It supplements the metadata map rather than replacing it. `get_records`,
    /// `list`, the result page and the saver all read the map, and a column
    /// store is the wrong shape to reassemble a record from. What the columns
    /// hold is a code per record and one copy of each distinct value, so a
    /// declared field with few distinct values costs four bytes a record. A
    /// declared field whose value differs on every record is held in full a
    /// second time; see `columns::Column`.
    columns: RwLockAt<{ order::COLUMNS }, ColumnStore>,

    /// Set once a filtered search has warned that it named a field this index
    /// did not declare, so the warning fires once rather than per search.
    ///
    /// Silent on an index that declared nothing, because there the walk is not
    /// a surprise: it is what the index has always done and what its
    /// declaration asked for.
    undeclared_filter_warned: AtomicBool,

    id_map: RwLockAt<{ order::ID_MAP }, HashMap<String, usize>>,
    rev_map: RwLockAt<{ order::REV_MAP }, HashMap<usize, String>>,

    // Mutex for write-only fields
    id_counter: MutexAt<{ order::ID_COUNTER }, usize>,

    /// The counter behind a generated external id, being `vec_N`.
    ///
    /// **Separate from `id_counter`, and it is not reset by `clear`.** It used
    /// to be the same counter, which meant two things at once. `clear` resets
    /// `id_counter` deliberately, because the graph's id-to-node array is one
    /// dense slot per internal id issued and an index cleared and refilled
    /// repeatedly would grow it without bound. That reset handed out `vec_1` a
    /// second time, so an external reference to the first record now named a
    /// different one and nothing said so.
    ///
    /// Splitting them lets each keep the property it needs. `id_counter` still
    /// resets, so the dense array still shrinks. This one never goes backwards,
    /// so a generated id is issued once in the life of an index and survives a
    /// save and load. See `config.json`'s `generated_ids`.
    ///
    /// It also stops a generated id burning an internal one. `generate_id` used
    /// to call `get_next_id`, so a batch of three records with no ids of their
    /// own consumed six internal ids and the fourth record added afterwards was
    /// `vec_7`. `list`'s ordering is unaffected either way, since it reads the
    /// internal ids the records actually hold.
    generated_ids: MutexAt<{ order::GENERATED_IDS }, usize>,
    vector_count: MutexAt<{ order::VECTOR_COUNT }, usize>, // Track total vectors for training trigger

    /// The graph.
    ///
    /// A read guard covers a traversal and the compute phase of a single record
    /// insertion. A write guard covers the install phase of that insertion, and
    /// covers replacing the whole backend, which `compact`,
    /// `rebuild_with_quantization` and the persistence rebuild each do once.
    ///
    /// The insertion is what takes this lock twice for one operation, a read
    /// guard for the phase that decides and a write guard for the phase that
    /// writes. It used to take a read guard alone, because the vendored graph
    /// took `&self` on an insert and did its own interior locking per neighbour
    /// list. ZeusDB's structure is a set of slabs and a mutator takes `&mut`,
    /// so the exclusion moved from inside the graph to this lock. See
    /// `insert_one` for the sequence and for what makes the gap between the two
    /// phases safe.
    /// The graph, and the raw vector store addressed by its node indices.
    ///
    /// **There is one copy of every raw vector and it is in here.** The index
    /// used to hold a second, in a `HashMap<String, Vec<f32>>` keyed by
    /// external id, written from the same local on the same insertion as the
    /// graph's. That map is gone. A raw vector is reached by
    /// `id_map[ext] -> VectorGraph::raw_vector`, which is one hash lookup the
    /// caller was already making and then two array reads.
    ///
    /// Which store holds the raws depends on the graph. On a raw graph they
    /// are the store the traversal scores against. On a `quantized_with_raw`
    /// graph they are a second store beside the codes, carried over node by
    /// node when training replaced the graph. On a trained `quantized_only`
    /// graph there are none.
    hnsw: RwLockAt<{ order::HNSW }, VectorGraph>,

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
    writers: MutexAt<{ order::WRITERS }, ()>,

    // ID-based training collection
    training_ids: RwLockAt<{ order::TRAINING_IDS }, Vec<String>>, // Just IDs, not vectors
    training_threshold_reached: AtomicBool,                       // Atomic flag for safety

    /// When the codebook was fitted, in RFC 3339, or `None` on an index that
    /// has never trained.
    ///
    /// Stamped once, by the `add` that reaches `training_size`, and carried
    /// through a save and a load unchanged.
    ///
    /// A directory written by a release that stamped this at save time instead
    /// carries a save time under the name, and there is no way to recover the
    /// real one. The loader restores what is there rather than restamping it,
    /// so the wrong value stops moving instead of being replaced by a newer
    /// wrong value.
    training_completed_at: RwLockAt<{ order::TRAINING_COMPLETED_AT }, Option<String>>,

    /// Timestamp when the index was created, in RFC 3339.
    ///
    /// Restored from `manifest.json` by the loader. `new_empty` stamps
    /// `Utc::now()` because it has nothing better to start from, and until the
    /// loader wrote the saved value back over it a load reset the field, so a
    /// save of a loaded index recorded the load as the creation.
    created_at: RwLockAt<{ order::CREATED_AT }, String>,

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
#[pyo3(signature = (dim, space, m, ef_construction, expected_size, quantization_config = None, indexed_fields = None))]
pub fn create_hnsw_index(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
    indexed_fields: Option<Vec<String>>,
) -> PyResult<HNSWIndex> {
    HNSWIndex::build(
        dim,
        space,
        m,
        ef_construction,
        expected_size,
        quantization_config,
        indexed_fields.unwrap_or_default(),
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
    /// The rule is that no path forks to rayon while holding a write guard, and
    /// this is a fork that rule is easy to miss on, because it is hidden inside
    /// an assignment rather than written as a call.
    ///
    /// Moving the old value out and dropping it after the guard is released
    /// keeps the swap to a pointer move under the guard.
    /// The shape rule for a 2-D query array
    ///
    /// Shared by the `f32` and `f64` batch arms because a query is wrong in the
    /// same way whatever it is made of. Both arms only became reachable when
    /// `cast` was moved above `extract`, so until then the list arm below them
    /// answered for every array and these messages were never seen. They are
    /// worded to match it: an empty batch is refused in the same words, and a
    /// row of the wrong width says "dimension mismatch" rather than describing
    /// a shape, because that is what a caller who passed a list would read.
    fn validate_batch_array_shape(shape: &[usize], dim: usize) -> Result<(), Error> {
        if shape.len() != 2 {
            error!(
                operation = "search",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape mismatch"
            );
            return Err(Error::BatchArrayShape {
                dim,
                shape: shape.to_vec(),
            });
        }

        if shape[0] == 0 {
            error!(
                operation = "search",
                error = "empty_batch",
                "Batch cannot be empty"
            );
            return Err(Error::BatchEmpty);
        }

        if shape[1] != dim {
            error!(
                operation = "search",
                error = "dimension_mismatch",
                expected = dim,
                actual = shape[1],
                "NumPy batch row width does not match the index"
            );
            return Err(Error::SearchVectorDimension {
                expected: dim,
                got: shape[1],
            });
        }

        Ok(())
    }

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
        Ok(py
            .detach(|| {
                let _writers = self.writers.lock().unwrap();
                self.rebuild_with_quantization_locked()
            })
            .map_err(Error::Engine)?)
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
        let (parsed_data, parse_errors) = self.parse_input_data(&data)?;

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
                ids: vec![],
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
        let (inserted_ids, insert_errors) = py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.insert_parsed_records(parsed_data, overwrite)
        });
        let total_inserted = inserted_ids.len();

        // The errors come back in the order they happened. Two of the three
        // variants carry a message Rust already built. The third carries the
        // `Error` the record's insertion raised, formatted here against its id.
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
                    errors.push(format!(
                        "Vector {}: {}: {}",
                        id,
                        err.exception().name(),
                        err
                    ));
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
            ids: inserted_ids,
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

        // Both arguments size the candidate heaps the traversal allocates
        // before it visits a node, and neither allocation is fallible; see
        // `MAX_TOP_K`. Checked first, so a bad argument is a ValueError and
        // not a dead interpreter, and before `ef` is derived so the derivation
        // cannot overflow on a value the check would have refused.
        if top_k > MAX_TOP_K {
            return Err(Error::TopKTooLarge {
                max: MAX_TOP_K,
                top_k,
            }
            .into());
        }
        if let Some(requested) = ef_search {
            if requested > MAX_EF_SEARCH {
                return Err(Error::EfSearchTooLarge {
                    max: MAX_EF_SEARCH,
                    ef_search: requested,
                }
                .into());
            }
        }

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

        // Compiled once per search, however many queries the call carries, and
        // before any record is examined. Rejecting an unrecognised operator or a
        // malformed group per record would make the error depend on the data,
        // because a record that lacks the field never reaches the operator at
        // all. What comes back cannot fail against any record, which is why the
        // traversal predicate has no error channel.
        let filter_conditions = filter
            .map(python_dict_to_value_map)
            .transpose()?
            .as_ref()
            .map(compile_filter)
            .transpose()?;

        // Detect batch vs single query with comprehensive input support.
        //
        // # Why `cast` runs before `extract`
        //
        // `extract::<Vec<Vec<f32>>>` succeeds on a 2-D NumPy array, because an
        // array satisfies the sequence protocol and each row satisfies it in
        // turn. Tried first, it therefore consumed exactly the input the
        // zero-copy arm below it was written for, and that arm was unreachable
        // for every array of every dtype. Measured on a one record index at
        // dimension 1,536 with a batch of 32, where the traversal is negligible
        // and the marshalling is the whole cost, it took 101.96 microseconds a
        // query against 21.54 for a list of lists, so the general path was
        // paying 4.7 times the list path to read the one input it could read
        // without copying at all.
        //
        // `cast` is a type check rather than a conversion attempt, so it
        // cannot claim an input that is not an array of that dtype, and it is
        // safe above `extract` in a way `extract` is not above it. `add`
        // already dispatches this way at every arm.
        //
        // The `f64` arm sits between them. NumPy's default dtype is `f64`, so
        // `np.random.rand(32, 1536)` is the common shape rather than an unusual
        // one, and without an arm of its own it falls to `extract` and is read
        // one Python float at a time. The conversion is the same rounding
        // `extract::<f32>` performs per element, so the values are identical.
        let result: Py<PyAny> = if let Ok(np_array) = vector.cast::<PyArray2<f32>>() {
            // Format: NumPy 2-D array (N, dims), read without copying.
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            Self::validate_batch_array_shape(shape, self.dim)?;

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
        } else if let Ok(np_array) = vector.cast::<PyArray2<f64>>() {
            // Format: NumPy 2-D array of f64, narrowed in one pass.
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            Self::validate_batch_array_shape(shape, self.dim)?;

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat
                .chunks(self.dim)
                .map(|chunk| chunk.iter().map(|&value| value as f32).collect())
                .collect();
            debug!(
                operation = "batch_search_numpy",
                batch_size = batch.len(),
                dtype = "f64",
                "Starting NumPy batch search"
            );
            let results =
                self.batch_search_internal(&batch, filter_conditions.as_ref(), params, py)?;
            PyList::new(py, results)?.into()
        } else if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]

            // Validation for empty batch or empty vectors in batch
            if list_vec.is_empty() {
                error!(
                    operation = "search",
                    error = "empty_batch",
                    "Batch cannot be empty"
                );
                return Err(Error::BatchEmpty.into());
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
                    return Err(Error::BatchVectorEmpty { position: i }.into());
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
        } else {
            // Single vector path - enhanced with NumPy 1D support.
            //
            // Already `cast` before `extract`, so there is no reordering to
            // make here. The `f64` arm is the same addition as on the batch
            // path and for the same reason: without it a `float64` query, which
            // is what NumPy hands back by default, was read one Python float at
            // a time. Measured on a one record index at dimension 1,536 that
            // was 84.41 microseconds a query against 8.43 for `float32`.
            let query_vector = if let Ok(array1d) = vector.cast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else if let Ok(array1d) = vector.cast::<PyArray1<f64>>() {
                array1d
                    .readonly()
                    .as_slice()?
                    .iter()
                    .map(|&value| value as f32)
                    .collect()
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
    /// `save_quantization_config`, `save_pq_centroids`, `save_pq_codes` and
    /// `save_vectors`, and every one of them speaks only to `serde_json`,
    /// `bincode` and `std::fs`. Every Python token in `persistence.rs` sits in
    /// the load path, in `rebuild_using_add_method` and the `conversion` module
    /// it calls. `save_hnsw_graph` reaches `graph::dump::write_dump`, which
    /// names PyO3 nowhere, and `save_manifest` and `StagingDir::commit` after it
    /// speak only to `serde_json` and `std::fs`.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        Ok(py.detach(|| self.save_locked(path))?)
    }

    /// Python property: `index.dim`
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Python property: `index.space`
    ///
    /// The same value `get_space()` returns, which stays. The two exist because
    /// `dim` is a property and `space` was not, so anything reading the index's
    /// configuration by attribute found half of it. The langchain adapter reads
    /// `getattr(index, "space", None)` to choose how it normalises a distance
    /// into a relevance score, and finding nothing it took the literal default
    /// of cosine, so an L2 or L1 index was scored by the cosine rule.
    ///
    /// It is a property rather than a second method because the caller that
    /// needed it probes for an attribute and calls it only if it turns out to be
    /// callable, so a property satisfies it and a method reads as configuration
    /// rather than as an action.
    ///
    /// `m`, `ef_construction` and `expected_size` are properties too, added
    /// after this one and for a different reason. Neither adapter reads them;
    /// their gap was that a caller wanting them had to parse `get_stats()`,
    /// which returns them as text.
    ///
    /// Named `space_property` in Rust because PyO3 derives the symbol it
    /// generates from the Rust name, and a getter's symbol takes a `get_`
    /// prefix, so a getter written as `space` collides with the existing
    /// `get_space` method. `#[getter(space)]` names the Python property.
    #[getter(space)]
    pub fn space_property(&self) -> String {
        self.space.clone()
    }

    /// Python property: `index.m`
    ///
    /// The graph degree, and the one creation parameter `rebuild` can change.
    ///
    /// Reachable before this only as `int(index.get_stats()["m"])`, which is a
    /// number formatted into a string and parsed back. hnswlib exposes `M`,
    /// `ef_construction` and `max_elements` as read-only properties by those
    /// names, and every other comparator exposes the equivalent typed.
    ///
    /// Read-only, because `m` is what the graph was built with. Assigning it
    /// would describe a graph that does not exist. Changing it for real is
    /// `rebuild(m=...)`, which builds the graph again at the new degree.
    #[getter]
    pub fn m(&self) -> usize {
        self.get_m()
    }

    /// Python property: `index.ef_construction`
    ///
    /// The candidate width each insertion searched at. It has no effect on a
    /// search, so a caller wanting a wider search passes `ef_search`.
    ///
    /// Read-only, because it describes work already done. Changing it for real
    /// is `rebuild(ef_construction=...)`, which re-inserts every record at the
    /// new width and so makes the number true again.
    #[getter]
    pub fn ef_construction(&self) -> usize {
        self.get_ef_construction()
    }

    /// Python property: `index.expected_size`
    ///
    /// The record count declared at creation. A capacity hint rather than a
    /// cap, unlike hnswlib's `max_elements`, so an index that grows past it
    /// grows the graph rather than raising. It selected the default `m` and it
    /// sized the initial reservation, which is why it is worth reading back.
    ///
    /// `len(index)` is the actual count and this is the declaration. The two
    /// disagreeing is ordinary.
    #[getter]
    pub fn expected_size(&self) -> usize {
        self.get_expected_size()
    }

    /// `len(index)`, the number of live records.
    ///
    /// Reads `id_map`, which is the record set: every insertion path writes it,
    /// removal keys on it, and `contains`, `list` and `count` all read the same
    /// map, so none of them can disagree with this. It equals
    /// `get_vector_count()` and `get_stats()["total_vectors"]`, which are
    /// maintained separately as a counter.
    ///
    /// Zero on an empty index, which is the only edge case a length has.
    pub fn __len__(&self) -> usize {
        self.id_map.read().unwrap().len()
    }

    /// `id in index`, which is `contains(id)`.
    ///
    /// `contains()` stays, because removing it would break every caller using
    /// it. This is the same read of the same map and cannot answer differently.
    pub fn __contains__(&self, id: String) -> bool {
        self.contains(id)
    }

    /// Live records matching a filter, or every live record when none is given.
    ///
    /// **Exact, and therefore a complete walk.** With a filter this evaluates
    /// every record's metadata and counts the matches. It cannot stop early:
    /// a count is a statement about the whole index, so the first record it
    /// skipped would make the answer a lower bound rather than a count.
    /// `scan_candidates` stops at `FULL_SCAN_THRESHOLD` because a search only
    /// needs to know whether the matching set is small enough to rank directly,
    /// which is a question an early exit answers and this one is not.
    ///
    /// What it reuses is `matches_filter` and `validate_filter_conditions`,
    /// which are the whole of the filter language. What it does not reuse is
    /// the scan's give-up point, its distance evaluation and its sort, none of
    /// which a count reads.
    ///
    /// Without a filter it is `len(index)` and reads one map length.
    ///
    /// An unknown operator raises `ValueError` before any record is examined,
    /// which is what a search does with the same filter. An empty index counts
    /// zero, and a filter matching nothing counts zero.
    #[pyo3(signature = (filter=None))]
    pub fn count(&self, py: Python<'_>, filter: Option<&Bound<PyDict>>) -> PyResult<usize> {
        let Some(filter) = filter else {
            return Ok(self.id_map.read().unwrap().len());
        };
        let conditions = compile_filter(&python_dict_to_value_map(filter)?)?;
        if conditions.matches_every_record() {
            return Ok(self.id_map.read().unwrap().len());
        }

        // The walk runs with the interpreter lock released. It reads every
        // record, which at 100,000 records is tens of milliseconds, and holding
        // the lock for that would stall every Python thread in the process.
        //
        // There is no error channel inside and none is needed, which is the
        // argument `Filtered::judge` makes for the search path. The filter is
        // compiled, so `matches_filter` returns `bool` and there is no error
        // arm to explain.
        Ok(py.detach(|| {
            // The columns answer this outright where every field is declared,
            // as a population count over the bitmap rather than a walk. Where
            // one field has no column the declared ones bound the candidates
            // and the metadata decides among them, which is the same count over
            // fewer reads. The guards below are taken in the declared order,
            // and the columns guard is released before the other two, since the
            // selection owns its bitmap.
            let selection = {
                let columns = self.columns.read().unwrap();
                columns.select(&conditions)
            };
            let rev_map = self.rev_map.read().unwrap();
            match selection {
                Selection::Exact(selected) => selected.count(),
                Selection::Narrowed(bound, _) => {
                    let vector_metadata = self.vector_metadata.read().unwrap();
                    let mut counted = 0;
                    bound.for_each(|slot| {
                        if let Some(id) = rev_map.get(&slot) {
                            if vector_metadata
                                .get(id)
                                .is_some_and(|meta| matches_filter(meta, &conditions))
                            {
                                counted += 1;
                            }
                        }
                    });
                    counted
                }
                Selection::Whole(_) => {
                    let vector_metadata = self.vector_metadata.read().unwrap();
                    vector_metadata
                        .values()
                        .filter(|meta| matches_filter(meta, &conditions))
                        .count()
                }
            }
        }))
    }

    /// The metadata fields this index built a column for, in the order they
    /// were declared.
    ///
    /// Empty on an index created without `indexed_fields` and on every
    /// directory saved before the declaration existed. A filter naming a field
    /// not in this list still works and still returns the same records; what it
    /// costs is a walk of every record's metadata.
    #[getter]
    pub fn indexed_fields(&self) -> Vec<String> {
        self.columns.read().unwrap().declared().to_vec()
    }

    /// Return the graph's spare buffer capacity to the allocator.
    ///
    /// A graph built by insertion grows its arenas geometrically, so the last
    /// growth leaves the largest of them holding close to twice what it uses. A
    /// graph read back from a saved dump has none of that slack, because the
    /// node count is known before the first write. That is the whole of why the
    /// same index reports a smaller graph after a save and load round trip than
    /// it did when it was built.
    ///
    /// `compact()` does not reclaim it. Compaction rebuilds by inserting, so the
    /// replacement graph regrows exactly the same slack, which is why this is
    /// its own operation. `compact()` calls it on the graph it built, so a
    /// caller who runs compaction gets both.
    ///
    /// **No node, edge or distance is touched.** Fourteen buffers are
    /// reallocated at their current lengths and their contents copied, so the
    /// topology after the call is the topology before it and every search
    /// returns the same page with the same scores.
    ///
    /// **The index stays writable.** Every buffer is a growable vector, so the
    /// next `add()` reallocates the per node arenas once and then proceeds as
    /// before. Shrinking an index that is still being written to therefore
    /// trades one regrowth for the memory, which is why this is never automatic.
    /// On an index that is finished, or one about to be searched for a long
    /// time, there is nothing to trade.
    ///
    /// Returns the bytes released, which is zero only when the buffers are
    /// already tight, in which case nothing is reallocated either.
    ///
    /// **On an empty index it releases the whole creation reservation**, which
    /// is what `expected_size` bought. Calling it before inserting is therefore
    /// not free: it hands back the pre-allocation and every subsequent
    /// insertion regrows the arenas from nothing. Call it on an index that
    /// holds its records, not on one about to receive them.
    pub fn shrink_to_fit(&self, py: Python<'_>) -> usize {
        py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            let mut hnsw = self.hnsw.write().unwrap();
            hnsw.shrink_to_fit()
        })
    }

    /// Get records by ID(s) with PQ reconstruction support and storage mode awareness
    ///
    /// Looks the ids up in the union of the raw vectors and the quantized codes,
    /// so it already saw every record before the other accessors did.
    ///
    /// **An absent id is dropped by default and raised with `strict=True`.** The
    /// default is silence because that is what every existing caller is written
    /// against: asking for ten ids and receiving nine has always been how this
    /// reports an id the index does not hold, and changing the return shape
    /// would break every one of them. What the default cannot do is say *which*
    /// nine, since the result is not aligned with the input. `strict=True`
    /// raises a `KeyError` naming the absent ids, which is the form a caller who
    /// cares can act on, and it is recommended over the alternative of returning
    /// a structure carrying the misses: a caller who passes ids it believes are
    /// present wants the failure, and one who does not can filter with
    /// `contains`.
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
    #[pyo3(signature = (input, return_vector = true, strict = false))]
    pub fn get_records(
        &self,
        py: Python<'_>,
        input: &Bound<PyAny>,
        return_vector: bool,
        strict: bool,
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
        let mut absent: Vec<String> = Vec::new();

        // Use read locks for concurrent access. `id_map` is the record set,
        // and the graph is where the raw vectors live, so both are taken here
        // and in that order.
        let id_map = self.id_map.read().unwrap();
        let hnsw = self.hnsw.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let raws = RawVectors {
            id_map: &id_map,
            graph: &hnsw,
        };

        for id in ids {
            // Check if this ID exists in either storage
            let exists = id_map.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id.clone())?;
                dict.set_item("metadata", value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    let vector_data = if let Some(raw_vector) = raws.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.to_vec())
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
                        // A list, matching `search`.
                        dict.set_item("vector", vec)?;
                    }
                }

                records.push(dict.into());
            } else if strict {
                absent.push(id);
            }
        }

        if !absent.is_empty() {
            // Every absent id rather than the first, because a caller correcting
            // a list wants the whole list. Sorted so the message does not depend
            // on the order the ids were asked in.
            absent.sort();
            return Err(Error::RecordsAbsent { absent }.into());
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

    /// One page of records, as (id, metadata), in the order they were added.
    ///
    /// Enumerates `id_map`, which holds every live record. It used to enumerate
    /// `vectors`, which under `quantized_only` holds only the records collected
    /// before training, so every record added afterwards was missing from the
    /// listing while search still returned it.
    ///
    /// # The order, which is what makes `offset` mean anything
    ///
    /// **Ascending internal id, which is arrival order.** This used to hand back
    /// `id_map.keys()` directly, and a `HashMap` iterates in an order its hasher
    /// reseeds in every process, so two calls in one process agreed and two
    /// processes did not. An offset over an order like that is not a page: it
    /// can return a record twice and miss another entirely, which is worse than
    /// having no paging at all. The scan path meets the same problem, where two
    /// equally distant records come back in hasher order, and pins a tie break
    /// for it.
    ///
    /// Arrival order rather than the external id's own order, for two reasons.
    /// A record added while a caller is paging appends at the end, so it cannot
    /// push a record the caller has not reached yet across a page boundary,
    /// which sorting by external id would do for any id that sorts low. And it
    /// is the order `compact` already rebuilds in, for the same reason: a
    /// property of the data rather than of where a hasher put a key. Internal
    /// ids are unique and are never reissued, so the order is total, and they
    /// survive a save and load, so it is the same order in the next process.
    ///
    /// # Paging by cursor rather than by count
    ///
    /// **`offset` shifts under deletion and `after` does not.** Removing a
    /// record ahead of an offset moves everything behind it up by one, so the
    /// next page skips a record, and no ordering fixes that: it is what paging
    /// by a count means. `after` names the last record the caller saw, and the
    /// next page is every record whose internal id is above that one's, so a
    /// deletion anywhere ahead of the cursor changes nothing about where the
    /// page starts.
    ///
    /// It is cheap because internal ids are what the order already rests on.
    /// They are unique, never reissued, and survive `compact`, `rebuild` and a
    /// save and load, so a cursor taken in one process is a cursor in the next.
    ///
    /// The one case it cannot absorb is the cursor record itself being removed,
    /// since its internal id goes with it. That raises, naming the id, rather
    /// than silently returning a page from somewhere else.
    ///
    /// `offset` stays as a convenience and the two are not combined: passing
    /// both raises, because a caller mixing them has two ideas about where the
    /// page starts.
    ///
    /// An offset at or past the record count returns an empty list rather than
    /// raising, and `number` of zero returns an empty list. Neither is an error.
    #[pyo3(signature = (number=10, offset=0, after=None))]
    pub fn list(
        &self,
        py: Python<'_>,
        number: usize,
        offset: usize,
        after: Option<String>,
    ) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        let cursor = match after {
            None => None,
            Some(ref id) => {
                if offset != 0 {
                    return Err(Error::ListAfterWithOffset {
                        after: id.clone(),
                        offset,
                    }
                    .into());
                }
                let Some(&internal) = id_map.get(id.as_str()) else {
                    return Err(Error::ListCursorMissing { after: id.clone() }.into());
                };
                Some(internal)
            }
        };

        // Only the records up to the end of the requested page need ordering, so
        // the tail is partitioned away in linear time and the sort runs over the
        // prefix. Paging a large index reads small pages many times, and sorting
        // the whole record set on each of them would be the dominant cost.
        let mut ordered: Vec<(usize, &String)> = id_map
            .iter()
            .map(|(id, &internal)| (internal, id))
            .filter(|&(internal, _)| cursor.is_none_or(|from| internal > from))
            .collect();
        let end = offset.saturating_add(number).min(ordered.len());
        if end < ordered.len() {
            ordered.select_nth_unstable_by_key(end, |&(internal, _)| internal);
        }
        let window = &mut ordered[..end];
        window.sort_unstable_by_key(|&(internal, _)| internal);

        let page = window.get(offset.min(end)..).unwrap_or(&[]);
        let mut results = Vec::with_capacity(page.len());
        for &(_, id) in page.iter() {
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
        Ok(py
            .detach(|| {
                let _writers = self.writers.lock().unwrap();
                self.remove_point_internal(id)
            })
            .map_err(Error::Engine)?)
    }

    /// Remove a batch of records, taking the mutation lock once.
    ///
    /// **Returns the ids that were not in the index**, in the order they were
    /// given, so an empty list means every id was removed. That is what the
    /// caller needs and a count is not: both shipped adapters loop
    /// `remove_point` today precisely to learn which ids failed, and both then
    /// throw away everything except a total.
    ///
    /// A repeated id is removed on its first occurrence and skipped afterwards,
    /// so it is never reported missing. The caller asked for the record to be
    /// gone and it is.
    ///
    /// An empty list of ids removes nothing and returns an empty list. Every id
    /// being absent returns all of them, which is not an error: removing a
    /// record that is not there is the state the caller asked for.
    ///
    /// What it saves against the loop is the locking. One acquisition of the
    /// mutation guard and one of each of the five storage guards for the whole
    /// batch, rather than one of each per id. It also makes the batch atomic
    /// against every search, where the loop let a search land between any two
    /// removals.
    ///
    /// Stranded graph nodes are left exactly as `remove_point` leaves them, one
    /// per record removed. Search already excludes them. `compact()` reclaims
    /// them and this does not call it, because compaction costs a full rebuild
    /// and the caller is the one who knows whether the debris is worth it.
    pub fn remove_points(&self, py: Python<'_>, ids: Vec<String>) -> Vec<String> {
        py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.remove_points_internal(&ids)
        })
    }

    /// Remove records by id or by filter, and report how many went.
    ///
    /// An alias. `delete(ids=...)` is `remove_points`, `delete(where=...)` is
    /// `remove_where`, and both of those stay. It exists because `delete` is
    /// what four of the five comparators call the operation, so a caller
    /// arriving from any of them reaches for `index.delete(...)`, gets an
    /// `AttributeError`, and has to go looking. The existing family compounds
    /// that: `remove_point` and `remove_points` differ by one character.
    ///
    /// **Both arguments is an error.** Two selections do not compose into one
    /// without a rule, and either rule is a guess. The union deletes records
    /// the filter did not choose and the intersection deletes fewer records
    /// than the ids name, so neither is what a caller writing both meant.
    ///
    /// **Neither argument is an error.** A `delete()` that deleted everything
    /// is the hazard `remove_where({})` already refuses, arrived at by leaving
    /// two optional arguments unset rather than by passing an empty mapping,
    /// which is easier to do by accident and not harder.
    ///
    /// **Returns the number of records removed**, in both cases, so the return
    /// type does not depend on which argument was given. `remove_points`
    /// returns the ids it could not find, which is more than a count and is
    /// what a caller needing that detail should keep calling. A repeated id
    /// counts once, because one record was removed.
    ///
    /// `ids` takes a single string or a list of strings, matching
    /// `get_records`. An empty list removes nothing and returns zero, which is
    /// not an error: it is a delete of nothing, not a delete of everything.
    #[pyo3(signature = (ids=None, r#where=None))]
    pub fn delete(
        &self,
        py: Python<'_>,
        ids: Option<&Bound<PyAny>>,
        r#where: Option<&Bound<PyDict>>,
    ) -> PyResult<usize> {
        match (ids, r#where) {
            (Some(_), Some(_)) => Err(Error::DeleteBothSelectors.into()),
            (None, None) => Err(Error::DeleteNoSelector.into()),
            (Some(ids), None) => {
                let requested: Vec<String> = if let Ok(single) = ids.extract::<String>() {
                    vec![single]
                } else if let Ok(many) = ids.extract::<Vec<String>>() {
                    many
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "delete(ids=) expects a string or a list of strings",
                    ));
                };

                // A repeated id names one record, so it counts once whether or
                // not it was there. `remove_points_internal` already skips a
                // repeat rather than reporting it missing.
                let distinct: std::collections::HashSet<&String> = requested.iter().collect();
                let distinct_count = distinct.len();

                let missing = py.detach(|| {
                    let _writers = self.writers.lock().unwrap();
                    self.remove_points_internal(&requested)
                });

                Ok(distinct_count - missing.len())
            }
            (None, Some(filter)) => self.remove_where(py, filter),
        }
    }

    /// Remove every record whose metadata matches the filter, and report how
    /// many were removed.
    ///
    /// The filter is the language `search` takes, evaluated by the same
    /// function against the same metadata, so a filter that selects a set here
    /// selects the same set there. An unknown operator raises `ValueError`
    /// before any record is examined.
    ///
    /// A filter matching nothing removes nothing and returns zero, which is not
    /// an error. An empty index returns zero.
    ///
    /// **An empty filter is refused.** Everywhere else in this language an empty
    /// filter matches every record, and `search(filter={})` returns the whole
    /// index for exactly that reason. This is the one method where following
    /// that rule destroys every record, and an empty mapping reaches it far more
    /// often from a caller that built its filter and got nothing than from a
    /// caller that meant it. Consistency is worth less here than the failure it
    /// would permit, because a search is repeatable and this is not. A caller
    /// who does mean it names the records, through `remove_points`.
    ///
    /// Stranded graph nodes are left behind, one per record removed, exactly as
    /// `remove_point` leaves them. This does not call `compact()` either; see
    /// `remove_points`.
    pub fn remove_where(&self, py: Python<'_>, filter: &Bound<PyDict>) -> PyResult<usize> {
        let conditions = compile_filter(&python_dict_to_value_map(filter)?)?;
        // Asked of the compiled tree rather than of the caller's mapping.
        // Emptiness was the whole test while `{}` was the only way to write
        // "everything", and boolean composition adds `{"$and": []}` and
        // `{"$not": {"$or": []}}` to it. Both are refused here for the same
        // reason the empty mapping is.
        if conditions.matches_every_record() {
            return Err(Error::RemoveWhereMatchesEverything.into());
        }
        Ok(py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.remove_where_locked(&conditions)
        }))
    }

    /// Empty the index, keeping its configuration, and report how many
    /// records went.
    ///
    /// **A fresh graph and empty maps, not `remove_points` over every id.**
    /// Both were considered. Removing every record one at a time is linear in
    /// the record count and leaves one stranded graph node per record, so an
    /// index of a million records would spend a long time producing a graph
    /// holding a million dead nodes and no live ones, which `compact()` would
    /// then have to walk again. Replacing the graph reclaims all of it at once
    /// and leaves `get_stats()["stranded_graph_nodes"]` at zero, which is what
    /// an empty index should report.
    ///
    /// What it keeps is the index: `dim`, `space`, `m`, `ef_construction`,
    /// `expected_size`, the index-level metadata `add_metadata` wrote, and the
    /// quantization configuration including a fitted codebook. **Training is
    /// not undone.** A codebook is fitted from data that is now gone and cannot
    /// be refitted from an empty index, so a trained quantized index stays
    /// trained and its replacement graph is a quantized graph. An untrained one
    /// returns to collecting, since the records it had collected are gone.
    ///
    /// The internal id counter restarts, because nothing is left for a reissued
    /// id to collide with and restarting keeps the internal ids in step with
    /// the fresh graph's node indices, which is the invariant `list()`'s
    /// ordering and `compact()` both rest on.
    ///
    /// Clearing an index that is already empty is not an error and returns
    /// zero. It still replaces the graph, so it also returns the arena.
    ///
    /// llama-index probes `hasattr(index, "clear")` and raised
    /// `NotImplementedError` when it found nothing.
    pub fn clear(&self, py: Python<'_>) -> PyResult<usize> {
        let quantized = self.is_quantized();
        let pq = self.pq.as_ref().cloned();

        if quantized && pq.is_none() {
            return Err(Error::NoQuantizer.into());
        }

        py.detach(|| {
            let _writers = self.writers.lock().unwrap();

            // Built before any guard is taken, so the allocation happens
            // outside the write guard exactly as `compact` arranges it.
            let fresh = if let (true, Some(pq)) = (quantized, pq) {
                let mut graph = VectorGraph::new_pq(
                    &self.space,
                    self.get_m(),
                    self.get_expected_size(),
                    MAX_LAYER,
                    self.get_ef_construction(),
                    pq,
                );
                // A cleared `quantized_with_raw` index goes on keeping raw
                // vectors, so its replacement graph opens the store the next
                // insertion writes into. Without this the store is absent and
                // every record added after a clear would lose its raw vector.
                if self
                    .quantization_config
                    .as_ref()
                    .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw)
                {
                    graph
                        .open_raw_store(self.dim, self.get_expected_size())
                        .expect("a quantized graph accepts a raw side store");
                }
                graph
            } else {
                VectorGraph::new_raw(
                    &self.space,
                    self.dim,
                    self.get_m(),
                    self.get_expected_size(),
                    MAX_LAYER,
                    self.get_ef_construction(),
                )
            };

            // The storage guards in the order declared on the struct, which is
            // the order every other multi-guard path here takes them in.
            let removed = {
                let mut id_map = self.id_map.write().unwrap();
                let mut rev_map = self.rev_map.write().unwrap();
                let mut pq_codes = self.pq_codes.write().unwrap();
                let mut vector_metadata = self.vector_metadata.write().unwrap();
                let mut columns = self.columns.write().unwrap();
                let mut training_ids = self.training_ids.write().unwrap();
                let mut id_counter = self.id_counter.lock().unwrap();
                let mut vector_count = self.vector_count.lock().unwrap();

                let removed = id_map.len();
                id_map.clear();
                rev_map.clear();
                pq_codes.clear();
                vector_metadata.clear();
                // Keeps the declaration and drops every record, which is what
                // `clear` does to the index itself. The reservation comes back
                // too, since a cleared index is about to be filled again.
                columns.clear(self.get_expected_size());
                training_ids.clear();
                *id_counter = 0;
                *vector_count = 0;
                removed
            };

            self.replace_graph(fresh);

            // An index still collecting for training starts collecting again,
            // since what it had collected is gone. A trained one is left alone,
            // because the flag records that training happened and it did.
            if !quantized {
                self.training_threshold_reached
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }
            self.overgrowth_warned
                .store(false, std::sync::atomic::Ordering::SeqCst);

            info!(
                operation = "clear",
                records_removed = removed,
                quantized = quantized,
                "Index cleared"
            );

            Ok(removed)
        })
    }

    /// Replace one record's metadata without resupplying its vector.
    ///
    /// **Wholesale, not a merge.** The supplied mapping becomes the record's
    /// metadata and any key not in it is gone. That is what `add(overwrite=True)`
    /// already does, since it removes the record outright before inserting the
    /// replacement, so the two ways of re-tagging a record agree.
    ///
    /// Returns `false` for an id the index does not hold, and writes nothing in
    /// that case. `true` when the metadata was replaced. Passing an empty
    /// mapping clears the record's metadata, which is a write and returns
    /// `true`.
    ///
    /// **It touches `vector_metadata` and nothing else.** No graph work, no
    /// vector work, no id allocation, no training. The record keeps its internal
    /// id, its graph node, its stored vector and its quantized codes.
    ///
    /// That is the reason it exists. Re-tagging a document used to mean reading
    /// it back with `get_records` and adding it again with `overwrite=True`, and
    /// under `quantized_only` `get_records` returns a reconstruction rather than
    /// the vector supplied, so the round trip silently replaced the record's
    /// vector with an approximation of itself. It also stranded a graph node and
    /// re-ran the insertion. None of that happens here.
    pub fn update_metadata(
        &self,
        py: Python<'_>,
        id: String,
        metadata: &Bound<PyDict>,
    ) -> PyResult<bool> {
        let fields = python_dict_to_value_map(metadata)?;
        Ok(py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.update_metadata_locked(&id, fields)
        }))
    }

    /// Rebuild the graph at a new degree, in place.
    ///
    /// **`m` is the one creation parameter nothing else can correct.** It is
    /// chosen from `expected_size` at `create()` and fixed there, so an index
    /// declared for 10,000 records and given a million runs at a degree meant
    /// for the smaller one, and no search width recovers the recall that costs.
    /// `dim` and `space` cannot be wrong that way, since an index at the wrong
    /// width or the wrong metric rejects or misranks its records outright, and
    /// `ef_construction` describes work already done.
    ///
    /// At least one of the three has to be given. Rebuilding the graph as it
    /// stands is `compact()`, which reclaims the nodes removals and overwrites
    /// leave behind.
    ///
    /// `expected_size` moves with `m` because the two are related. It selects
    /// the default `m`, it sizes the replacement graph's reservation, and it is
    /// what the overgrowth warning compares against, so a caller correcting one
    /// usually wants the other. Passing it also re-arms that warning.
    ///
    /// `ef_construction` moves with them because **the warning below names it as
    /// one of two remedies and a caller has to be able to take either.** Raising
    /// `m` past half of `ef_construction` switches the neighbour selection
    /// heuristic off, and the message says to raise `ef_construction` or lower
    /// `m`. It is honest to report afterwards for the same reason the rebuild is
    /// honest at all: every record really was re-inserted at the new width.
    ///
    /// **Everything except the graph survives untouched.** Each record is
    /// re-inserted under the internal id it already holds, so `id_map`,
    /// `rev_map`, the metadata store and every declared field's column stay
    /// correct without being rewritten, and the record any given id resolves to
    /// is the same before and after. A quantized index is rebuilt from its
    /// stored codes rather than re-encoded, so the codebook is not retrained and
    /// no record's code changes; a `quantized_with_raw` index carries its raw
    /// store over node by node.
    ///
    /// The three are held to the rules `create()` applies, on the same five
    /// values, and an invalid one raises the message `create()` raises for it.
    /// The `ef_construction` against `2 * m` pair is checked too, and warns where
    /// the pair puts the neighbour selection heuristic out of reach.
    ///
    /// Returns the node count of the graph it built, which equals the live
    /// record count.
    ///
    /// **The cost is a full rebuild by insertion**, the same cost as
    /// `compact()`, proportional to the live record count. The replacement is
    /// built in full before the old graph is dropped, so peak memory holds both
    /// and a failure part way through leaves the index exactly as it was.
    #[pyo3(signature = (m=None, expected_size=None, ef_construction=None))]
    pub fn rebuild(
        &self,
        py: Python<'_>,
        m: Option<usize>,
        expected_size: Option<usize>,
        ef_construction: Option<usize>,
    ) -> PyResult<usize> {
        if m.is_none() && expected_size.is_none() && ef_construction.is_none() {
            return Err(Error::RebuildWithoutChanges.into());
        }
        let new_m = m.unwrap_or_else(|| self.get_m());
        let new_expected_size = expected_size.unwrap_or_else(|| self.get_expected_size());
        let new_ef_construction = ef_construction.unwrap_or_else(|| self.get_ef_construction());
        // The rules `create()` applies, on the same five values, so a degree
        // this build would refuse at creation is refused here with the message
        // creation raises for it.
        validate_index_parameters(
            self.dim,
            &self.space,
            new_m,
            new_ef_construction,
            new_expected_size,
            "",
        )?;
        // Before the interpreter lock is released, because a Python warning
        // needs it. The pair is what decides it, exactly as at `create()`, and
        // **both of the remedies it names are reachable from here**: this takes
        // `ef_construction` as well as `m`, so a caller told to raise one or
        // lower the other can do either.
        warn_if_selection_disabled(py, new_m, new_ef_construction)?;
        if expected_size.is_some() {
            // A raised declaration is a new bar for the overgrowth warning, and
            // the old one has already been claimed if it fired.
            self.overgrowth_warned.store(false, Ordering::Release);
        }
        Ok(py.detach(|| self.rebuild_locked(new_m, new_expected_size, new_ef_construction))?)
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
    /// `VectorGraph::new_raw` and `VectorGraph::insert`, which are the same
    /// shape as the quantized pair already listed there.
    pub fn compact(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(py.detach(|| self.compact_locked())?)
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
        Ok(self.benchmark_reads(query_count, max_threads)?)
    }
}
