//! The index: its state, the locks that protect it, and every operation the
//! binding calls.
//!
//! # The two structs
//!
//! [`Collection`] is the record set and everything addressed by a record: the
//! two id maps and the live set, the metadata and its columns, the counters,
//! the training buffer, the timestamps, the mutation lock and the warning
//! flags. [`Space`] is one vector space over those records: the graph, the
//! quantizer with its codes and calibration, the metric, the width and the two
//! graph tunables. A collection holds one space today, as a plain field. The
//! fields are divided so that a map of named spaces is an addition later
//! rather than a second division, and nothing here assumes there will only
//! ever be one.
//!
//! The one path that crosses the two is training. It reads the training ids
//! from the collection, fetches their vectors through the space's graph, fits
//! the space's quantizer, writes the codes and the calibration on the space
//! and clears the buffer on the collection. It is a method on the collection
//! that borrows its space, which is what every operation here is: the
//! collection owns the space and reaches into it as `self.space`.
//!
//! # What lives here and what does not
//!
//! This file holds the two structs, the live record set, the quantization
//! configuration and the operations that are a read of one or two guards. The
//! rest is in a child module, and a child can read the private fields because
//! it is a descendant of this one.
//!
//! | module | what it covers |
//! |---|---|
//! | `construct` | building a collection and validating the declaration |
//! | `input` | what a vector becomes once it is out of Python |
//! | `insert` | insertion, replacement, removal, compaction, rebuild, clear |
//! | `search` | the four paths that reach the graph, and the page they build |
//! | `training` | fitting the codebook and rebuilding over the codes |
//! | `stats` | what the index reports about itself |
//! | `persist` | the accessors and setters `persistence.rs` speaks to |
//!
//! Nothing here names Python. The binding holds the `#[pyclass]` that wraps a
//! `Collection` by value, parses every argument into the owned Rust these
//! methods take, releases the interpreter lock around the call, and converts
//! what comes back. The error type is [`Error`] throughout, and the binding
//! maps it to an exception class.
//!
//! # Log records
//!
//! Every record emitted from this module tree carries the target the module
//! carried when it lived in the binding, `zeusdb_vector_database::hnsw_index`
//! and its children, rather than `module_path!()`, which would name this
//! crate. The filter directive the binding installs and a `RUST_LOG`
//! directive both match a target by prefix, so a record carrying this crate's
//! name would fall outside both and be dropped. Each file names its target
//! once, as `LOG_TARGET`, and every macro call and every `#[instrument]`
//! attribute in it passes that constant.

mod construct;
mod input;
mod insert;
mod persist;
mod search;
mod stats;
mod training;

// The declaration rules, so that `persistence::load_config` applies the same
// ones to `config.json` that `Declaration::validate` applies to a caller's
// arguments.
pub use construct::Declaration;
pub(crate) use construct::{validate_index_parameters, validate_space_supports_quantization};
pub use insert::{Added, RebuildPlan};
pub use search::QueryHits;
pub use stats::{QuantizationReport, QuantizerReport};

use crate::locks::{order, MutexAt, RwLockAt};
use crate::RerankCalibration;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{trace, warn};
use zeusdb_vector_core::{
    matches_filter, Bitmap, ColumnStore, Error, Filter, Selection, VectorGraph, PQ,
};

/// The target every record this file emits carries. See the module
/// documentation.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index";

/// Records accepted by `add`, after parsing and before insertion, as
/// (external id, vector, metadata). The vector has been processed for the
/// index space already; see `Space::process_vector_for_space`.
///
/// It holds no Python object, which is what makes it the boundary the
/// interpreter lock is released across. The binding produces it and `insert`
/// consumes it.
pub type ParsedRecords = Vec<(String, Vec<f32>, HashMap<String, Value>)>;

/// Layers every graph this crate builds is created with.
///
/// It is the dump format's layer count, `NB_LAYER_MAX` in the engine's floor,
/// read from there rather than restated, because a dump written at any other
/// layer count is refused on load, so this is part of the on-disk contract
/// rather than a tuning knob.
const MAX_LAYER: usize = zeusdb_vector_core::NB_LAYER_MAX as usize;

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
/// The product quantization a space was declared with.
///
/// Built through [`Declaration::quantization`], which holds the five values to
/// the rules `create()` applies, or read back from `quantization.json` by the
/// loader, which holds them to the same rules through
/// `validate_quantization_fields`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: StorageMode,
}

/// One page of `list`, as (id, metadata) in ascending internal id.
pub type Listing = Vec<(String, HashMap<String, Value>)>;

/// One record as `get_records` returns it: its id, its metadata, and its
/// vector where one was asked for and the index could supply one.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordView {
    pub id: String,
    pub metadata: HashMap<String, Value>,
    pub vector: Option<Vec<f32>>,
}

// ============================================================================
// THE LIVE RECORD SET
// ============================================================================

/// The internal id of every live record, resolved to its external id, and the
/// same set of ids as a bitmap.
///
/// **The bitmap is what an unfiltered search runs under.** The traversal asks
/// its predicate once for every six to eight distance evaluations, and the
/// predicate used to be `HashMap::contains_key` on the reverse map, which at
/// 50,000 records cost 46 of a 179 microsecond search on SIFT and 41 of 189 on
/// GloVe. A bit test costs nothing the traversal can distinguish from no
/// predicate at all. The page cannot move, because the bitmap holds exactly
/// the set of keys the map holds: every write to one is a write to the other,
/// through the four methods below, and there is no `DerefMut`.
///
/// Reads go through `Deref` to the map, so every path that resolved a node
/// through `rev_map.get` still does.
///
/// A slot is an internal id, and internal ids are never reused, so the bitmap
/// grows with the id counter rather than with the live count. That is one bit
/// per id ever issued, which `get_stats` leaves out of
/// `index_bookkeeping_memory_mb` so that figure is unchanged by the set.
pub(crate) struct LiveRecords {
    by_internal: HashMap<usize, String>,
    live: Bitmap,
}

impl LiveRecords {
    fn new() -> Self {
        LiveRecords {
            by_internal: HashMap::new(),
            live: Bitmap::default(),
        }
    }

    /// Record that `internal` resolves to `external`.
    pub(crate) fn insert(&mut self, internal: usize, external: String) {
        self.by_internal.insert(internal, external);
        self.live.insert(internal);
    }

    /// Forget `internal`, returning the external id it resolved to.
    pub(crate) fn remove(&mut self, internal: usize) -> Option<String> {
        self.live.remove(internal);
        self.by_internal.remove(&internal)
    }

    /// Forget every record.
    pub(crate) fn clear(&mut self) {
        self.by_internal.clear();
        self.live.clear();
    }

    /// Replace the whole set with a map read back from disk.
    pub(crate) fn replace(&mut self, by_internal: HashMap<usize, String>) {
        let mut live = Bitmap::default();
        for &internal in by_internal.keys() {
            live.insert(internal);
        }
        self.by_internal = by_internal;
        self.live = live;
    }

    /// The set as bits, for the traversal predicate.
    #[inline]
    pub(crate) fn live(&self) -> &Bitmap {
        &self.live
    }

    /// The map itself, for the saver, which writes it whole.
    pub(crate) fn map(&self) -> &HashMap<usize, String> {
        &self.by_internal
    }
}

impl std::ops::Deref for LiveRecords {
    type Target = HashMap<usize, String>;
    #[inline]
    fn deref(&self) -> &HashMap<usize, String> {
        &self.by_internal
    }
}

// ============================================================================
// THE SPACE
// ============================================================================

/// One vector space over the collection's records.
///
/// The graph, the quantizer with its codes and its calibration, the metric,
/// the width and the two graph tunables. Everything that would be repeated
/// per named space if there were more than one, and nothing that would not.
/// The record set, the metadata and the training buffer are the collection's,
/// because a record is one record however many spaces hold a vector for it.
///
/// Its locks keep the ranks they had on the undivided struct, `hnsw` at 3,
/// `pq_codes` at 4, `rerank_calibration` at 11 and `training_completed_at` at
/// 12, so a search that takes `id_map` at 1 and then `hnsw` at 3 through two
/// structs is the same acquisition it was through one. See the order on
/// [`Collection`].
pub(crate) struct Space {
    dim: usize,
    /// The distance space, normalised to lower case: cosine, l2, l1 or dot.
    metric: String,
    /// The graph degree, written by `rebuild` and read everywhere else.
    ///
    /// An atomic rather than a plain field because `rebuild` takes `&self`, as
    /// every other mutating operation does. It is written only under
    /// `writers`, which every mutating entry point takes first, and read by
    /// the saver, the stats and the three rebuild paths. A search never reads
    /// it. `ef_construction` is the same for the same reason.
    m: AtomicUsize,
    ef_construction: AtomicUsize,

    // Quantization configuration and PQ instance
    quantization_config: Option<QuantizationConfig>,
    pq: Option<Arc<PQ>>,
    pq_codes: RwLockAt<{ order::PQ_CODES }, HashMap<String, Vec<u8>>>, // PQ codes storage

    /// What training measured about how deep this space's codes bury a true
    /// neighbour, which is what the default rerank fetch is derived from.
    ///
    /// Written once by `calibrate_rerank` at training completion and by the
    /// loader from `quantization.json`. `None` on an unquantized space, on a
    /// `quantized_only` one, before training, and on a space trained before
    /// the calibration existed. See `RerankCalibration`.
    rerank_calibration: RwLockAt<{ order::RERANK_CALIBRATION }, Option<RerankCalibration>>,

    /// The graph, and the raw vector store addressed by its node indices.
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

    /// When the codebook was fitted, in RFC 3339, or `None` on a space that
    /// has never trained.
    ///
    /// Stamped once, by the `add` that reaches `training_size`, and carried
    /// through a save and a load unchanged. It is the quantizer's own date, so
    /// it sits beside the quantizer rather than with the collection's
    /// creation stamp.
    ///
    /// A directory written by a release that stamped this at save time instead
    /// carries a save time under the name, and there is no way to recover the
    /// real one. The loader restores what is there rather than restamping it,
    /// so the wrong value stops moving instead of being replaced by a newer
    /// wrong value.
    training_completed_at: RwLockAt<{ order::TRAINING_COMPLETED_AT }, Option<String>>,
}

impl Space {
    /// The graph degree.
    ///
    /// **The one read of `m` in the crate**, so that `rebuild` writing it has a
    /// single place to be seen from. Acquire against the release the write
    /// makes, though every writer also holds `writers` and every reader that
    /// matters runs after it.
    pub(crate) fn m(&self) -> usize {
        self.m.load(Ordering::Acquire)
    }

    /// The construction width. The one read of it, for the reason `m` gives.
    pub(crate) fn ef_construction(&self) -> usize {
        self.ef_construction.load(Ordering::Acquire)
    }

    /// Whether the space was declared with quantization.
    pub(crate) fn has_quantization(&self) -> bool {
        self.quantization_config.is_some()
    }

    /// Whether the codebook is fitted.
    pub(crate) fn can_use_quantization(&self) -> bool {
        self.pq.as_ref().is_some_and(|pq| pq.is_trained())
    }

    /// Whether the graph scores against codes.
    ///
    /// Takes the graph's read guard, so it cannot be asked by a path already
    /// holding one. `search_candidates` asks the guard it was handed instead.
    pub(crate) fn is_quantized(&self) -> bool {
        if let Some(pq) = &self.pq {
            if pq.is_trained() {
                let hnsw_guard = self.hnsw.read().unwrap();
                return hnsw_guard.is_quantized();
            }
        }
        false
    }

    /// Whether the storage mode keeps a raw vector beside every code.
    pub(crate) fn keeps_raw(&self) -> bool {
        self.quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw)
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
    /// The rule is that no path forks to rayon while holding a write guard, and
    /// this is a fork that rule is easy to miss on, because it is hidden inside
    /// an assignment rather than written as a call.
    ///
    /// Moving the old value out and dropping it after the guard is released
    /// keeps the swap to a pointer move under the guard.
    pub(crate) fn replace_graph(&self, new_hnsw: VectorGraph) {
        let old = {
            let mut hnsw_guard = self.hnsw.write().unwrap();
            std::mem::replace(&mut *hnsw_guard, new_hnsw)
        };
        drop(old);
    }

    /// What training measured, where it ran
    pub(crate) fn rerank_calibration(&self) -> Option<RerankCalibration> {
        *self.rerank_calibration.read().unwrap()
    }

    /// Install a calibration read back from a saved index
    pub(crate) fn set_rerank_calibration(&self, calibration: Option<RerankCalibration>) {
        *self.rerank_calibration.write().unwrap() = calibration;
    }
}

// ============================================================================
// THE COLLECTION
// ============================================================================

/// The record set, with one vector space over it.
///
/// # Lock acquisition order
///
/// Every path that holds two of these guards at once acquires them in this
/// order, top to bottom. Releasing may happen in any order.
///
/// ```text
/// id_map < rev_map < hnsw < pq_codes < vector_metadata < columns
///        < training_ids < metadata < id_counter < vector_count
/// ```
///
/// **This order is checked rather than believed.** Every lock below, and every
/// lock on [`Space`], is a [`RwLockAt`] or a [`MutexAt`] carrying its rank as
/// a const generic, and on a debug build each acquisition asserts that the
/// thread holds none of the same lock and nothing ranked above it. See
/// [`crate::locks`] for what that catches, what it costs and what it misses.
/// In release the wrappers are the standard types by another name. A rank is
/// a number and the registry does not care which struct holds the lock, so
/// dividing the fields between two structs changed nothing it checks.
///
/// `columns` sits directly below `vector_metadata` because every path that
/// writes one writes the other, and a filtered search holds both: the columns
/// to decide which records match and the metadata to fill the page it returns.
///
/// This exists because search and mutation overlap. Until the receivers were
/// relaxed, PyO3's exclusive borrow kept every mutating method away from every
/// search, so no reader and no writer were ever in flight together and the
/// acquisition order could not matter. It matters now. A search holds
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
/// Four locks sit outside the order. `writers` is taken by the mutating
/// operations before any guard and never by an internal helper; see the
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
pub struct Collection {
    /// The one vector space. See [`Space`] for what it holds and why.
    space: Space,

    /// The record count declared at creation.
    ///
    /// A capacity hint rather than a cap. It sized the column store's
    /// reservation and the graph's, it selected the default `m`, and it is
    /// what the overgrowth warning compares the live count against. It is the
    /// collection's declaration rather than the space's because it is a count
    /// of records, and a second space would be sized from the same number.
    /// An atomic because `rebuild` rewrites it through `&self`.
    expected_size: AtomicUsize,

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
    /// Internal id to external id, and the live set as bits. See
    /// [`LiveRecords`].
    rev_map: RwLockAt<{ order::REV_MAP }, LiveRecords>,

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
    /// Held by the operations the binding calls, and by `save`, which holds it
    /// across all four of its phases. An internal caller reaching a mutating
    /// helper is already inside the guard, so the helpers never take it and
    /// cannot deadlock against the caller that owns it.
    writers: MutexAt<{ order::WRITERS }, ()>,

    // ID-based training collection
    training_ids: RwLockAt<{ order::TRAINING_IDS }, Vec<String>>, // Just IDs, not vectors
    training_threshold_reached: AtomicBool,                       // Atomic flag for safety

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

impl Collection {
    /// The vector width.
    pub fn dim(&self) -> usize {
        self.space.dim
    }

    /// The distance space, normalised: cosine, l2, l1 or dot.
    pub fn metric(&self) -> &str {
        &self.space.metric
    }

    /// The graph degree. See `Space::m`.
    pub fn m(&self) -> usize {
        self.space.m()
    }

    /// The construction width. See `Space::ef_construction`.
    pub fn ef_construction(&self) -> usize {
        self.space.ef_construction()
    }

    /// The record count declared at creation. The one read of it, for the
    /// reason `Space::m` gives.
    pub fn expected_size(&self) -> usize {
        self.expected_size.load(Ordering::Acquire)
    }

    /// The counter `add` maintains, which `get_stats` reports as
    /// `total_vectors`.
    pub fn vector_count(&self) -> usize {
        *self.vector_count.lock().unwrap()
    }

    /// Whether the space was declared with quantization.
    pub fn has_quantization(&self) -> bool {
        self.space.has_quantization()
    }

    /// Whether the codebook is fitted.
    pub fn can_use_quantization(&self) -> bool {
        self.space.can_use_quantization()
    }

    /// Whether the graph scores against codes. Takes the graph's read guard.
    pub fn is_quantized(&self) -> bool {
        self.space.is_quantized()
    }

    /// The storage mode as `get_storage_mode` reports it.
    ///
    /// Reaches the graph's read guard through `is_quantized`, so it cannot be
    /// asked under a storage guard; `remove_point_internal` reads it before it
    /// takes any.
    pub fn storage_mode(&self) -> String {
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

    /// The number of live records, which is `len(index)`.
    ///
    /// Reads `id_map`, which is the record set: every insertion path writes it,
    /// removal keys on it, and `contains`, `list` and `count` all read the same
    /// map, so none of them can disagree with this.
    pub fn len(&self) -> usize {
        self.id_map.read().unwrap().len()
    }

    /// Whether the index holds no live record.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a record with this id is in the index
    ///
    /// Reads `id_map`, which is the record set. Every insertion path writes it,
    /// `remove_point_internal` keys its removal on it, `add(overwrite=True)`
    /// keys its collision test on it, and `compact` rebuilds the graph from it.
    /// It used to read `vectors`, which under `quantized_only` holds only the
    /// records collected before training, so this returned `false` for a record
    /// that search returned and `remove_point` removed.
    pub fn contains(&self, id: &str) -> bool {
        let id_map = self.id_map.read().unwrap();
        id_map.contains_key(id)
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
    /// The filter arrives compiled, so nothing here can fail on it and there is
    /// no error channel, which is the argument `Filtered::judge` makes for the
    /// search path. A filter matching every record is one map length, as is no
    /// filter at all.
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        let Some(conditions) = filter else {
            return self.id_map.read().unwrap().len();
        };
        if conditions.matches_every_record() {
            return self.id_map.read().unwrap().len();
        }

        // The columns answer this outright where every field is declared,
        // as a population count over the bitmap rather than a walk. Where
        // one field has no column the declared ones bound the candidates
        // and the metadata decides among them, which is the same count over
        // fewer reads. The guards below are taken in the declared order,
        // and the columns guard is released before the other two, since the
        // selection owns its bitmap.
        let selection = {
            let columns = self.columns.read().unwrap();
            columns.select(conditions)
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
                            .is_some_and(|meta| matches_filter(meta, conditions))
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
                    .filter(|meta| matches_filter(meta, conditions))
                    .count()
            }
        }
    }

    /// The metadata fields this index built a column for, in the order they
    /// were declared. Empty on an index created without `indexed_fields`.
    pub fn indexed_fields(&self) -> Vec<String> {
        self.columns.read().unwrap().declared().to_vec()
    }

    /// Records by id, with the vector where one was asked for.
    ///
    /// Looks the ids up in the union of the raw vectors and the quantized
    /// codes. An absent id is dropped, or reported through
    /// `Error::RecordsAbsent` naming every absent id in sorted order when
    /// `strict` is set. The vector is the raw one where the index holds one
    /// and the reconstruction from the codes where it does not, which under
    /// `quantized_only` is every record once training completes. The contract
    /// behind those choices is documented on the Python entry point.
    pub fn records(
        &self,
        ids: Vec<String>,
        return_vector: bool,
        strict: bool,
    ) -> Result<Vec<RecordView>, Error> {
        trace!(
            target: LOG_TARGET,
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
        let hnsw = self.space.hnsw.read().unwrap();
        let pq_codes = self.space.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let raws = crate::RawVectors {
            id_map: &id_map,
            graph: &hnsw,
        };

        for id in ids {
            // Check if this ID exists in either storage
            let exists = id_map.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let vector = if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    if let Some(raw_vector) = raws.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.to_vec())
                    } else if let (Some(pq), Some(codes)) = (&self.space.pq, pq_codes.get(&id)) {
                        // Case 2: Only quantized codes available (QuantizedOnly mode)
                        match pq.reconstruct(codes) {
                            Ok(reconstructed) => Some(reconstructed),
                            Err(e) => {
                                warn!(target: LOG_TARGET, operation = "vector_reconstruction", vector_id = %id, error = %e, "Failed to reconstruct vector");
                                None
                            }
                        }
                    } else {
                        // Case 3: No vector data available
                        None
                    }
                } else {
                    None
                };

                records.push(RecordView {
                    id,
                    metadata,
                    vector,
                });
            } else if strict {
                absent.push(id);
            }
        }

        if !absent.is_empty() {
            // Every absent id rather than the first, because a caller correcting
            // a list wants the whole list. Sorted so the message does not depend
            // on the order the ids were asked in.
            absent.sort();
            return Err(Error::RecordsAbsent { absent });
        }

        trace!(
            target: LOG_TARGET,
            operation = "get_records_complete",
            found_records = records.len(),
            "Records retrieval completed"
        );
        Ok(records)
    }

    /// One page of records, as (id, metadata), in ascending internal id,
    /// which is arrival order.
    ///
    /// `after` names the last record the caller saw and the page is every
    /// record above it, which holds still under deletion where `offset` does
    /// not. The two are not combined: both given raises, and a cursor the index
    /// no longer holds raises rather than paging from somewhere else. The
    /// reasoning is documented on the Python entry point.
    pub fn list(
        &self,
        number: usize,
        offset: usize,
        after: Option<&str>,
    ) -> Result<Listing, Error> {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        let cursor = match after {
            None => None,
            Some(id) => {
                if offset != 0 {
                    return Err(Error::ListAfterWithOffset {
                        after: id.to_string(),
                        offset,
                    });
                }
                let Some(&internal) = id_map.get(id) else {
                    return Err(Error::ListCursorMissing {
                        after: id.to_string(),
                    });
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
            results.push((id.clone(), metadata));
        }
        Ok(results)
    }

    /// Add index-level metadata
    pub fn add_metadata(&self, metadata: HashMap<String, String>) {
        let mut meta_lock = self.metadata.lock().unwrap();
        for (key, value) in metadata {
            meta_lock.insert(key, value);
        }
    }

    /// Get index-level metadata value
    pub fn metadata(&self, key: &str) -> Option<String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.get(key).cloned()
    }

    /// Get all index-level metadata
    pub fn all_metadata(&self) -> HashMap<String, String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.clone()
    }

    /// Generate a unique ID for a vector
    ///
    /// From a counter of its own rather than from the internal id counter. See
    /// `generated_ids` for what each of the two has to guarantee and why one
    /// counter could not do both.
    pub fn generate_id(&self) -> String {
        let mut counter = self.generated_ids.lock().unwrap();
        *counter += 1;
        format!("vec_{}", *counter)
    }
}

#[cfg(test)]
mod tests {
    use super::LiveRecords;
    use std::collections::HashMap;

    /// The bitmap admits exactly the ids the map holds, after every kind of
    /// write, which is what makes the unfiltered traversal's page the same
    /// page it was under `contains_key`.
    #[test]
    fn the_live_set_agrees_with_the_map_through_every_write() {
        let agrees = |records: &LiveRecords| {
            (0..256).all(|id| records.live().contains(id) == records.contains_key(&id))
        };
        let mut records = LiveRecords::new();
        assert!(agrees(&records));

        // Across a word boundary, since the words grow on demand.
        for id in [1usize, 2, 3, 64, 65, 130] {
            records.insert(id, format!("r{id}"));
        }
        assert!(agrees(&records));
        assert_eq!(records.live().count(), 6);
        assert_eq!(records.get(&130).map(String::as_str), Some("r130"));

        assert_eq!(records.remove(64), Some("r64".to_string()));
        assert_eq!(records.remove(64), None);
        assert!(agrees(&records));
        assert!(!records.live().contains(64));

        let mut saved = HashMap::new();
        saved.insert(7usize, "a".to_string());
        saved.insert(190usize, "b".to_string());
        records.replace(saved);
        assert!(agrees(&records));
        assert_eq!(records.live().count(), 2);
        assert_eq!(records.map().len(), 2);

        records.clear();
        assert!(agrees(&records));
        assert_eq!(records.live().count(), 0);
        assert!(records.is_empty());
    }
}
