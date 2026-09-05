//! The index: its state, the locks that protect it, and every operation the
//! binding calls.
//!
//! # The two structs
//!
//! [`Collection`] is the record set and everything addressed by a record: the
//! two id maps and the live set, the metadata and its columns, the counters,
//! the training buffer, the timestamps, the mutation lock and the warning
//! flags. A [`Space`] is one vector space over those records. The dense one,
//! [`DenseSpace`], holds the graph behind its guard as a [`DenseIndex`], the
//! quantizer with its codes and calibration, the metric, the width and the
//! two graph tunables. The sparse one, [`SparseSpace`], holds a postings index
//! behind one guard. A collection holds its spaces in a list in declaration
//! order, which is the lock order, and the first is always the dense space
//! the binding declares, reached as `self.dense()`.
//!
//! Each index implements the seam in `zeusdb_vector_core`, so an insertion is
//! `prepare` under the space's read guard and `insert` under its write guard,
//! a removal is `remove`, and a search is `search` under an admit set the
//! collection builds from its live set, its columns and its metadata. The
//! collection never reaches past the seam into a structure, except to read
//! the graph for the rerank rule, the persistence and the statistics, which
//! is what `DenseIndex::graph` exists for.
//!
//! The one path that crosses the two is training. It reads the training ids
//! from the collection, fetches their vectors through the space's graph, fits
//! the space's quantizer, writes the codes and the calibration on the space
//! and clears the buffer on the collection. The training buffer is the
//! collection's and `training_size` is the space's, which holds while one
//! space can be quantized; a second such space would need a buffer of its
//! own.
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
//! | `apply` | a journaled operation applied at the values its record names |
//! | `construct` | building a collection and validating the declaration |
//! | `crash_tests` | what a killed process leaves, and what opening it recovers |
//! | `durability_tests` | the three policies, the interval thread, and what a failed commit leaves |
//! | `input` | what a vector becomes once it is out of Python |
//! | `insert` | insertion, replacement, removal, compaction, rebuild, clear |
//! | `int8_tests` | a scalar quantized space through the collection, its artefacts and its bounds |
//! | `journal_tests` | a directory and the journal beside it |
//! | `query` | a query over one or more arms, its plan and its fused page |
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

mod apply;
mod construct;
#[cfg(test)]
mod crash_tests;
mod dense;
#[cfg(test)]
mod durability_tests;
mod input;
mod insert;
#[cfg(test)]
mod int8_tests;
#[cfg(test)]
mod journal_tests;
#[cfg(test)]
mod mutation_tests;
mod persist;
#[cfg(test)]
mod persist_tests;
mod query;
#[cfg(test)]
mod query_tests;
#[cfg(test)]
mod replay_tests;
mod search;
#[cfg(test)]
mod spaces_tests;
mod stats;
mod training;

// The declaration rules, so that `persistence::load_config` applies the same
// ones to `config.json` that `Declaration::validate` applies to a caller's
// arguments.
pub use construct::Declaration;
pub(crate) use construct::{
    validate_index_parameters, validate_space_supports_quantization, SparseDeclaration,
};
pub(crate) use dense::{DenseIndex, DenseOpen};
pub use insert::{Added, RebuildPlan};
pub use query::{
    AdmitShape, Arm, ArmPlan, Page, Plan, Query, QueryHit, DEFAULT_FETCH_PER_K, MAX_ARMS,
};
pub use search::{QueryHits, SparseHits};
pub use stats::{QuantizationReport, QuantizerReport};

use crate::journal::Durability;
use crate::locks::{MutexAt, RwLockAt};
use crate::RerankCalibration;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tracing::{trace, warn};
use zeusdb_vector_core::{
    matches_filter, Bitmap, ColumnStore, Error, Filter, Int8Codec, MetadataStore, Operation,
    OperationKind, Persist, Selection, SpaceName, SparseVector, VectorGraph, VectorIndex, PQ,
};
use zeusdb_vector_sparse::{PostingsIndex, SparseConfig};
use zeusdb_vector_text::{count_record_with, TermDictionary, Tokenizer, TokenizerConfig};

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

/// What a record hands the sparse space, where it hands anything.
///
/// A space that takes term ids takes a vector of them. A space with a text
/// layer takes the terms its tokenizer split, which the collection counts
/// into term ids and term frequencies as it inserts the record, under the
/// mutation guard, issuing an id to every term not seen before. The terms
/// travel as strings rather than as ids so that nothing between the
/// tokenizer and the insertion holds an id the dictionary issued. Ids issued
/// outside the mutation guard were ids a `clear` could reissue: the
/// dictionary was emptied under a caller's ids, and the caller's postings
/// then named terms other records had been given.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseHalf {
    /// Term ids and weights, on a space that takes term ids.
    Vector(SparseVector),
    /// Terms already split, on a space with a text layer. Counted under the
    /// mutation guard, so the ids are issued as the record is inserted.
    Terms(Vec<String>),
}

/// One record on its way in, with a vector for every space it fills.
///
/// The dense vector is always present, since the binding's surface declares
/// one dense space and every record fills it. The sparse half fills the
/// collection's sparse space where one is declared, and a record may leave
/// it empty. A `ParsedRecords` tuple is a record with no sparse half.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub sparse: Option<SparseHalf>,
    pub metadata: HashMap<String, Value>,
}

impl From<(String, Vec<f32>, HashMap<String, Value>)> for ParsedRecord {
    fn from((id, vector, metadata): (String, Vec<f32>, HashMap<String, Value>)) -> Self {
        ParsedRecord {
            id,
            vector,
            sparse: None,
            metadata,
        }
    }
}

/// Where a mutation's record goes, once something is attached to take it.
///
/// Every durable mutation is encoded as an [`Operation`] record and handed
/// here before it runs, under `writers` and before any storage guard, one
/// `append` per record and one `commit` per entry point call, so that
/// nothing a search can see was left unrecorded. A term interned for a
/// record is handed over under the text layer's dictionary guard at the
/// moment its id is issued, so the records arrive in id order. The engine
/// holds one implementation, being [`crate::JournalSink`], a
/// write-ahead journal beside the collection's directory. A collection is
/// built with no sink, and then the insert path skips the encode and every
/// mutation runs as it did. A collection applying recorded operations back,
/// through [`Collection::apply`], hands nothing here.
///
/// Beyond the two a mutation uses, the trait carries what a checkpoint asks
/// of the sink, being the sync, the sequence it has reached, the truncation
/// once the directory holds those records, and the two values a manifest
/// records so that a directory and its records are paired by content. They
/// are on the trait rather than reached by detaching the sink around the
/// save, because detaching and attaching each take the mutation guard and
/// release it, and a mutation running in the window between the save and the
/// reattachment would land in neither the directory nor the records.
pub trait OperationSink: Send + std::fmt::Debug {
    /// Take one record's bytes. Called with `writers` held on every
    /// mutation path and with the dictionary's write guard held for an
    /// interning, and with nothing else. A refusal refuses the mutation
    /// before it runs, with the error carried out to the caller.
    fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<(), Error>;

    /// Make everything appended so far durable, after the last record of a
    /// call and before the call mutates anything a search can see.
    fn commit(&mut self) -> Result<(), Error>;

    /// Make everything appended so far durable whatever the policy, which a
    /// checkpoint does before it records the sequence it is about to hold.
    ///
    /// The default does what [`OperationSink::commit`] does, which is right
    /// for a sink that is already durable at every commit and wrong for one
    /// that defers.
    fn sync(&mut self) -> Result<(), Error> {
        self.commit()
    }

    /// The last sequence handed over, or the first less one where nothing
    /// has been. A checkpoint records this after it syncs, and a recovery
    /// replays every record above it.
    ///
    /// A sink that carries no sequence reports zero, which is the value that
    /// means no record was ever appended.
    fn sequence_reached(&self) -> u64 {
        0
    }

    /// Drop every record handed over so far, which the checkpoint that just
    /// ran now holds. Called after the save has committed its directory and
    /// never before.
    fn truncate(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// The name a manifest records for this sink's file, where it has one.
    ///
    /// A sink that is not a file records nothing, and a directory saved by a
    /// collection holding one is a directory with no journal.
    fn journal_file(&self) -> Option<&str> {
        None
    }

    /// The collection this sink's records belong to, where it names one, so
    /// the manifest and the file's own header can be held against each
    /// other.
    fn journal_collection_id(&self) -> Option<u128> {
        None
    }

    /// Where this sink's file sits, where it is a file, so a checkpoint can
    /// be held to the directory the file belongs beside.
    fn journal_path(&self) -> Option<&std::path::Path> {
        None
    }

    /// The durability policy this sink keeps, where it keeps one. What a
    /// caller reads back, and nothing the collection acts on.
    fn durability(&self) -> Option<Durability> {
        None
    }

    /// The size of this sink's file, where it is a file, so a caller can
    /// see what a replay would read.
    fn journal_bytes(&self) -> Option<u64> {
        None
    }

    /// The interval policy's thread, for the tests that hold it to its
    /// lifetime and its cost. Compiled away outside them.
    #[cfg(test)]
    fn flusher_watch(&self) -> Option<crate::flusher::Watch> {
        None
    }
}

/// The sink a collection hands its records to, and the fault a failed
/// commit left on it.
///
/// A commit that fails leaves the records it was to make durable in the
/// sink and out of the collection, since the call refuses them, and they
/// may or may not be on the device. A replay would install them. So a
/// mutation recorded after them would replay onto a collection holding
/// records this one refused, and an insert under the same id, which is what
/// a retry is, would refuse the whole open as a record the collection
/// already holds. The fault stops that: every record is refused with it
/// until a checkpoint syncs the journal, records a sequence at or above the
/// refused records so a replay skips them, and truncates them.
#[derive(Debug, Default)]
pub(crate) struct SinkSlot {
    attached: Option<Box<dyn OperationSink>>,
    fault: Option<SinkFault>,
}

/// What the commit that failed said, and the sequence it had reached.
#[derive(Clone, Debug)]
struct SinkFault {
    sequence: u64,
    error: String,
}

impl SinkSlot {
    /// The refusal every record meets while the fault stands.
    fn refuse_if_faulted(&self) -> Result<(), Error> {
        match &self.fault {
            Some(fault) => Err(Error::JournalCommitFailed {
                sequence: fault.sequence,
                error: fault.error.clone(),
            }),
            None => Ok(()),
        }
    }
}

/// The journal a collection hands its records to, as a caller sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalStatus {
    /// The file.
    pub path: std::path::PathBuf,
    /// The policy every commit keeps.
    pub durability: Durability,
    /// The sequence the checkpoint in the directory holds.
    pub checkpoint_sequence: u64,
    /// The last sequence handed to the journal.
    pub sequence_reached: u64,
    /// The file's length, where the filesystem answered.
    pub bytes: Option<u64>,
}

impl JournalStatus {
    /// Records a replay of the journal would apply, being every record
    /// above the checkpoint's sequence.
    pub fn records_since_checkpoint(&self) -> u64 {
        self.sequence_reached
            .saturating_sub(self.checkpoint_sequence)
    }
}

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
/// How a scalar quantized space fits its scales.
///
/// One value today. It is an enum rather than a flag so that `quantization.json`
/// names the rule by a word a later release can add to, and so that a caller
/// reads the word back from `get_stats()` and `get_quantization_info()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Int8Scale {
    /// One scale a dimension, being the largest magnitude seen in that
    /// dimension over the training sample divided by 127.
    #[serde(rename = "per_dimension")]
    PerDimension,
}

impl Int8Scale {
    /// The name `quantization_config` and `quantization.json` spell it by.
    pub const PER_DIMENSION: &'static str = "per_dimension";

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            Self::PER_DIMENSION => Some(Int8Scale::PerDimension),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Int8Scale::PerDimension => Self::PER_DIMENSION,
        }
    }
}

/// The scheme a quantized space encodes its records under, with the values
/// that scheme alone takes. What the two schemes share, being the training
/// size, its cap and the storage mode, sits beside this on
/// [`QuantizationConfig`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Product quantization: one byte a subvector, each decoded through a
    /// codebook of `2^bits` centroids fitted to the training sample.
    Pq { subvectors: usize, bits: usize },
    /// Scalar quantization: one signed byte a value, each decoded through
    /// one scale a dimension fitted to the training sample.
    Int8 { scale: Int8Scale },
}

impl QuantizationScheme {
    /// The word `quantization_config`'s `type` key and `quantization.json`'s
    /// `type` field carry.
    pub fn type_name(&self) -> &'static str {
        match self {
            QuantizationScheme::Pq { .. } => "pq",
            QuantizationScheme::Int8 { .. } => "int8",
        }
    }

    pub fn is_int8(&self) -> bool {
        matches!(self, QuantizationScheme::Int8 { .. })
    }

    /// The subvector count and the bits of a product quantized scheme, and
    /// `None` on a scalar one.
    pub fn pq_shape(&self) -> Option<(usize, usize)> {
        match self {
            QuantizationScheme::Pq { subvectors, bits } => Some((*subvectors, *bits)),
            QuantizationScheme::Int8 { .. } => None,
        }
    }

    /// The scale rule of a scalar scheme, and `None` on a product quantized
    /// one.
    pub fn int8_scale(&self) -> Option<Int8Scale> {
        match self {
            QuantizationScheme::Int8 { scale } => Some(*scale),
            QuantizationScheme::Pq { .. } => None,
        }
    }
}

/// The quantization a space was declared with.
///
/// Built through [`Declaration::quantization`] for a product quantized
/// space and [`Declaration::scalar_quantization`] for a scalar one, each of
/// which holds its values to the rules `create()` applies, or read back from
/// `quantization.json` by the loader, which holds them to the same rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub scheme: QuantizationScheme,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: StorageMode,
}

impl QuantizationConfig {
    pub fn is_int8(&self) -> bool {
        self.scheme.is_int8()
    }
}

/// One page of `list`, as (id, metadata) in ascending internal id.
pub type Listing = Vec<(String, HashMap<String, Value>)>;

/// The name of the dense space the binding declares.
///
/// An ordinary name to everything below the binding. A collection built
/// through `Declaration` always holds a dense space under it, first.
pub const DEFAULT_SPACE: &str = "default";

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
/// per id ever issued, being a third of a byte a record beside the sixty-odd
/// the tables hold, and `index_bookkeeping_memory_mb` counts it. It used to
/// leave it out on the grounds that the figure should be unchanged by the set,
/// which made the figure a sum with a term missing from it.
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

    /// The set as bits.
    ///
    /// The traversal used to run under this bitmap. It runs under the dense
    /// index's own live set now, which is maintained on the same writes, and
    /// this one is what that set is checked against on a debug build; see
    /// [`Collection::live_sets_agree`].
    #[cfg(test)]
    pub(crate) fn live(&self) -> &Bitmap {
        &self.live
    }

    /// Whether `other` admits every live record, by a word walk of the
    /// intersection against the live count. What decides that a filter's
    /// bitmap is no filter at all; see `Collection::admit_plan`.
    pub(crate) fn admits_every_live(&self, other: &Bitmap) -> bool {
        other.count_and(&self.live) == self.by_internal.len()
    }

    /// Whether `other` holds exactly the ids this set holds.
    pub(crate) fn agrees_with(&self, other: &Bitmap) -> bool {
        if self.live.count() != other.count() {
            return false;
        }
        let mut agrees = true;
        self.live.for_each_while(|slot| {
            agrees = other.contains(slot);
            agrees
        });
        agrees
    }

    /// The map itself, for the saver, which writes it whole.
    pub(crate) fn map(&self) -> &HashMap<usize, String> {
        &self.by_internal
    }

    /// Bytes the set's bitmap asks the allocator for, for `Collection::stats`.
    ///
    /// One word per sixty-four internal ids, which is a third of a byte a
    /// record. Small, and the report is a sum, so a structure left out of it
    /// makes the sum wrong by however little it holds.
    pub(crate) fn live_heap_bytes(&self) -> usize {
        self.live.heap_bytes()
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
// THE SPACES
// ============================================================================

/// How one space is declared, as the collection reports it back.
///
/// Each variant carries the index's own configuration type, and the crate
/// root re-exports every one of them, so a caller declaring a space depends
/// on this crate alone. Adding a kind of space adds a variant here.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SpaceConfig {
    Dense(DenseConfig),
    /// A sparse space that takes term ids and weights alone.
    Sparse(SparseConfig),
    /// A sparse space with a text layer, which takes text as well.
    Text(TextConfig),
}

/// The declaration of a sparse space with a text layer, being the index's
/// own configuration and the tokenizer as a value.
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub index: SparseConfig,
    pub tokenizer: TokenizerConfig,
}

/// The declaration of a dense space, being what `create()` takes for it.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseConfig {
    pub dim: usize,
    /// The distance space, normalised to lower case: cosine, l2, l1 or dot.
    pub metric: String,
    pub m: usize,
    pub ef_construction: usize,
    pub quantization: Option<QuantizationConfig>,
}

/// One vector space over the collection's records, of whichever kind.
///
/// The dense variant is several times the sparse one, since it carries the
/// declaration and four guards where the sparse carries one. The enum is
/// held once per space in a list of at most a few entries, and boxing the
/// dense variant would put an indirection on every read of the dense space,
/// which is every operation, so the size difference is accepted.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Space {
    Dense(DenseSpace),
    Sparse(SparseSpace),
}

/// A space with the name it was declared under.
pub(crate) struct NamedSpace {
    pub(crate) name: SpaceName,
    pub(crate) space: Space,
}

/// The dense vector space over the collection's records.
///
/// The graph, behind its guard as a [`DenseIndex`], the quantizer with its
/// codes and its calibration, the metric, the width and the two graph
/// tunables. Everything that would be repeated per dense space if there were
/// more than one, and nothing that would not. The record set, the metadata
/// and the training buffer are the collection's, because a record is one
/// record however many spaces hold a vector for it.
///
/// Its four locks take their ranks from the space's position in the
/// collection's declaration, so the first dense space's index guard sits at
/// the rank the graph guard has always had and its codes guard one above,
/// and a second space's sit two above those. See the order on
/// [`Collection`].
pub(crate) struct DenseSpace {
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
    pq_codes: RwLockAt<HashMap<String, Vec<u8>>>, // PQ codes storage

    /// The scalar codec, once a scalar quantized space has fitted it, and
    /// its trained flag is this slot being filled.
    ///
    /// Empty when the space is declared, set once by the training trigger
    /// under `writers` or by the loader through the exclusive borrow it
    /// holds, and never cleared, since `clear` keeps a fitted codebook as it
    /// does for a product quantized space and nothing untrains one. A read
    /// is an atomic load, so `can_use_quantization` takes no guard on a
    /// scalar space and sits where `PQ::is_trained` sits in the order:
    /// readable under every guard, with nothing taken under it. The kernel
    /// never reads this slot. The graph holds its own `Arc` to the codec
    /// inside its distance, handed to it by `new_int8` and
    /// `restore_int8_graph`, so a search and an insertion read the scales
    /// through the index guard they already hold.
    int8: OnceLock<Arc<Int8Codec>>,

    /// What training measured about how deep this space's codes bury a true
    /// neighbour, which is what the default rerank fetch is derived from.
    ///
    /// Written once by `calibrate_rerank` at training completion and by the
    /// loader from `quantization.json`. `None` on an unquantized space, on a
    /// `quantized_only` one, before training, and on a space trained before
    /// the calibration existed. See `RerankCalibration`.
    rerank_calibration: RwLockAt<Option<RerankCalibration>>,

    /// The index, being the graph, the live set it is searched under, and the
    /// raw vector store addressed by its node indices.
    ///
    /// A read guard covers a traversal and `prepare`, the compute phase of a
    /// single record insertion. A write guard covers `insert`, the install
    /// phase of that insertion, a removal, and replacing the whole graph,
    /// which `compact`, `rebuild_with_quantization` and the persistence
    /// rebuild each do once.
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
    index: RwLockAt<DenseIndex>,

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
    training_completed_at: RwLockAt<Option<String>>,
}

impl DenseSpace {
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

    /// Whether the quantizer is fitted, being the codebook on a product
    /// quantized space and the scales on a scalar one.
    pub(crate) fn can_use_quantization(&self) -> bool {
        self.pq.as_ref().is_some_and(|pq| pq.is_trained()) || self.int8.get().is_some()
    }

    /// Whether the space was declared with scalar quantization, fitted or
    /// not.
    pub(crate) fn declares_int8(&self) -> bool {
        self.quantization_config
            .as_ref()
            .is_some_and(QuantizationConfig::is_int8)
    }

    /// The scalar codec, once fitted.
    pub(crate) fn int8_codec(&self) -> Option<&Arc<Int8Codec>> {
        self.int8.get()
    }

    /// Fill the codec slot. Refused, handing the codec back, where the slot
    /// is already filled, which no path reaches: the training trigger runs
    /// once, under `writers`, and the loader fills an empty collection.
    pub(crate) fn install_int8_codec(&self, codec: Arc<Int8Codec>) -> Result<(), Arc<Int8Codec>> {
        self.int8.set(codec)
    }

    /// Whether the graph scores against codes.
    ///
    /// Takes the index's read guard, so it cannot be asked by a path already
    /// holding one. The search paths ask the guard they were handed instead.
    pub(crate) fn is_quantized(&self) -> bool {
        if self.can_use_quantization() {
            let index = self.index.read().unwrap();
            return index.graph().is_quantized();
        }
        false
    }

    /// Whether the storage mode keeps a raw vector beside every code.
    pub(crate) fn keeps_raw(&self) -> bool {
        self.quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw)
    }

    /// The declaration, as the collection reports it.
    pub(crate) fn config(&self) -> DenseConfig {
        DenseConfig {
            dim: self.dim,
            metric: self.metric.clone(),
            m: self.m(),
            ef_construction: self.ef_construction(),
            quantization: self.quantization_config.clone(),
        }
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
    /// keeps the swap to a pointer move under the guard. The replacement's
    /// unit cost is timed before the guard is taken, for the same reason.
    pub(crate) fn replace_graph(&self, new_hnsw: VectorGraph) {
        let timed = DenseIndex::time_graph(&new_hnsw);
        let old = {
            let mut index = self.index.write().unwrap();
            index.replace_graph(new_hnsw, timed)
        };
        drop(old);
    }

    /// Replace the whole index, which the loader does with the index it
    /// restored from a dump. The old graph is dropped outside the guard, for
    /// the reason `replace_graph` gives.
    ///
    /// `clear` swaps the index itself rather than calling this, because it has
    /// to empty the collection's live set and the index's under one
    /// acquisition and this takes the index guard alone.
    pub(crate) fn replace_index(&self, fresh: DenseIndex) {
        let old = {
            let mut index = self.index.write().unwrap();
            std::mem::replace(&mut *index, fresh)
        };
        drop(old);
    }

    /// A fresh index over `graph` for this space, with the live set empty.
    pub(crate) fn fresh_index(&self, graph: VectorGraph) -> DenseIndex {
        DenseIndex::new(
            graph,
            &self.metric,
            self.dim,
            self.pq.clone(),
            self.keeps_raw(),
        )
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

/// The sparse vector space over the collection's records.
///
/// A postings index behind one guard, at the index rank of the space's
/// position. A search runs under the read guard and an insertion, a removal
/// and a compaction under the write guard. The index keeps its own live set,
/// as the dense one does, and a record may leave this space empty.
pub(crate) struct SparseSpace {
    config: SparseConfig,
    pub(crate) index: RwLockAt<PostingsIndex>,
    /// The text layer, where the space was declared with a tokenizer.
    pub(crate) text: Option<TextLayer>,
}

impl SparseSpace {
    pub(crate) fn config(&self) -> &SparseConfig {
        &self.config
    }

    /// The declaration as the collection reports it.
    pub(crate) fn space_config(&self) -> SpaceConfig {
        match &self.text {
            Some(text) => SpaceConfig::Text(TextConfig {
                index: self.config.clone(),
                tokenizer: text.tokenizer.config(),
            }),
            None => SpaceConfig::Sparse(self.config.clone()),
        }
    }
}

/// What turns a record's text or a query's text into the term ids the
/// postings index stores.
///
/// The tokenizer runs under no guard at all. It may be the caller's own and
/// need something of the caller's to run, such as an interpreter lock, and a
/// thread holding that while it waited for a guard the tokenizing thread
/// held would wait forever; see [`Collection::tokenize`]. The dictionary
/// sits behind the space's second guard, ranked after the index's, and the
/// terms a tokenizer split travel as strings up to the guard that counts
/// them, so nothing holds an id across a gap in which a `clear` could
/// reissue it. A record's terms are counted under the dictionary's write
/// guard, released, and the vector is then inserted under the index's write
/// guard, both under `writers`, taken by `add_records` before any record's
/// terms are counted, so nothing can empty the dictionary between a
/// record's ids being issued and its postings being written. A query's
/// terms are counted under the dictionary's read guard, taken after the
/// index's read guard and held with it until the page is made, so nothing
/// can empty the dictionary between a query's ids being counted and its
/// postings being searched; a record interning a new term waits for such a
/// query there, as it would at the record set's guard directly after. The
/// tokenizer runs before `writers` is taken, so a tokenizer that needs the
/// caller's lock costs the mutation guard nothing.
pub(crate) struct TextLayer {
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
    pub(crate) dictionary: RwLockAt<TermDictionary>,
}

// ============================================================================
// THE COLLECTION
// ============================================================================

/// The record set, with its vector spaces over it.
///
/// # Lock acquisition order
///
/// Every path that holds two of these guards at once acquires them in this
/// order, top to bottom. Releasing may happen in any order.
///
/// ```text
/// id_map < rev_map < [space 0 index < space 0 codes < space 1 index < ...]
///        < vector_metadata < columns < training_ids < metadata
///        < id_counter < vector_count
/// ```
///
/// `hnsw` and `pq_codes` in the prose elsewhere are the first space's index
/// and codes guards, which sit where they always did. A second space's guards
/// sit after them, so a search over two spaces takes both between the
/// reverse map and the metadata, in declaration order.
///
/// **This order is checked rather than believed.** Every lock below, and every
/// lock on a space, is a [`RwLockAt`] or a [`MutexAt`] given its rank at
/// construction, and on a debug build each acquisition asserts that the
/// thread holds none of the same lock and nothing ranked above it. See
/// [`crate::locks`] for what that catches, what it costs and what it misses.
/// In release the wrappers are the standard types by another name. A rank is
/// a number and the registry does not care which struct holds the lock, so
/// dividing the fields between structs changed nothing it checks.
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
/// `sink` is the last leaf, below every rank above. A mutation takes it with
/// `writers` held and nothing else, and an interning takes it with the
/// dictionary's write guard held, so anything may be held while it is taken
/// and nothing is taken under it.
///
/// Taking the same guard twice on one thread is forbidden even for reads.
/// The standard library queues readers behind a waiting writer, so a second
/// read on the thread already holding one deadlocks the moment a writer lands
/// between them, which is how `get_stats` used to hang against training id
/// collection. The registry asserts this on every acquisition in a debug build,
/// so it fires in an ordinary single threaded test rather than waiting for a
/// writer to land in the window.
pub struct Collection {
    /// The vector spaces, in declaration order, which is their lock order.
    /// The first is always the dense space the binding declares; see
    /// [`Collection::dense`].
    spaces: Vec<NamedSpace>,

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
    metadata: MutexAt<HashMap<String, String>>,

    /// Every record's metadata, indexed by internal id. See
    /// [`MetadataStore`] for what a record with and without fields costs.
    vector_metadata: RwLockAt<MetadataStore>,

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
    /// It supplements the metadata store rather than replacing it. `get_records`,
    /// `list`, the result page and the saver all read the store, and a column
    /// store is the wrong shape to reassemble a record from. What the columns
    /// hold is a code per record and one copy of each distinct value, so a
    /// declared field with few distinct values costs four bytes a record. A
    /// declared field whose value differs on every record is held in full a
    /// second time; see `columns::Column`.
    columns: RwLockAt<ColumnStore>,

    /// Set once a filtered search has warned that it named a field this index
    /// did not declare, so the warning fires once rather than per search.
    ///
    /// Silent on an index that declared nothing, because there the walk is not
    /// a surprise: it is what the index has always done and what its
    /// declaration asked for.
    undeclared_filter_warned: AtomicBool,

    id_map: RwLockAt<HashMap<String, usize>>,
    /// Internal id to external id, and the live set as bits. See
    /// [`LiveRecords`].
    rev_map: RwLockAt<LiveRecords>,

    // Mutex for write-only fields
    id_counter: MutexAt<usize>,

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
    generated_ids: MutexAt<usize>,
    vector_count: MutexAt<usize>, // Track total vectors for training trigger

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
    /// Held by the operations the binding calls, `add_metadata` among them,
    /// and by `save`, which holds it across all four of its phases, so every
    /// mutation of what a save writes is ordered against every other. An
    /// internal caller reaching a mutating helper is already inside the
    /// guard, so the helpers never take it and cannot deadlock against the
    /// caller that owns it.
    writers: MutexAt<()>,

    /// Where every mutation's record goes, once something is attached, and
    /// the fault a failed commit left; see [`OperationSink`] and
    /// [`SinkSlot`]. Empty on every collection this crate builds, so the
    /// insert path encodes nothing and no mutation waits on anything.
    /// Attached and detached under `writers`, so whether a batch is
    /// recorded is settled before its first record is admitted.
    sink: MutexAt<SinkSlot>,

    /// The collection this is, drawn once when it is built and carried for
    /// its life.
    ///
    /// It exists to pair a directory with the journal beside it. A save of a
    /// journaled collection records this in `manifest.json` and the journal's
    /// own header carries it too, so a journal from another index is refused
    /// by content rather than by name. A load takes it from the manifest
    /// where one is recorded and draws a fresh one where none is, so it is
    /// stable across a save and a load of a journaled directory and is not
    /// carried by a directory that has never held a journal.
    ///
    /// Not a lock. It is drawn once when the collection is built and
    /// replaced once, through an exclusive borrow, by a load that reads one
    /// from a manifest, both before the collection is shared with anything.
    collection_id: u128,

    /// The sequence the checkpoint in the collection's directory holds.
    ///
    /// Zero on a collection that has never been journaled and on one whose
    /// journal has had no record appended. A checkpoint syncs the sink,
    /// writes the sequence the sink had reached here, and the manifest
    /// records it, so a recovery replays every record above it and skips
    /// every record at or below it.
    journal_sequence: MutexAt<u64>,

    // ID-based training collection
    training_ids: RwLockAt<Vec<String>>, // Just IDs, not vectors
    training_threshold_reached: AtomicBool, // Atomic flag for safety

    /// Timestamp when the index was created, in RFC 3339.
    ///
    /// Restored from `manifest.json` by the loader. `new_empty` stamps
    /// `Utc::now()` because it has nothing better to start from, and until the
    /// loader wrote the saved value back over it a load reset the field, so a
    /// save of a loaded index recorded the load as the creation.
    created_at: RwLockAt<String>,

    /// Set once the index has warned that it holds materially more records than
    /// `expected_size` declared, so the warning fires once rather than on every
    /// subsequent `add`.
    overgrowth_warned: AtomicBool,
}

impl Collection {
    /// The dense space, which is always the first declared.
    ///
    /// The binding's surface declares one dense space and reaches it through
    /// this, so every operation that reads the graph, the quantizer or the
    /// width reads this space's.
    pub(crate) fn dense(&self) -> &DenseSpace {
        match &self.spaces[0].space {
            Space::Dense(dense) => dense,
            Space::Sparse(_) => unreachable!("the first space is the dense one by construction"),
        }
    }

    /// The dense space, for the loader's setters.
    pub(crate) fn dense_mut(&mut self) -> &mut DenseSpace {
        match &mut self.spaces[0].space {
            Space::Dense(dense) => dense,
            Space::Sparse(_) => unreachable!("the first space is the dense one by construction"),
        }
    }

    /// Whether the dense index's live set is the collection's, which every
    /// write keeps true and a debug build checks at the points where the
    /// two are read together.
    ///
    /// Taken in the declared order, `rev_map` then the index.
    pub(crate) fn live_sets_agree(&self) -> bool {
        let rev_map = self.rev_map.read().unwrap();
        let index = self.dense().index.read().unwrap();
        rev_map.agrees_with(index.live_set()) && rev_map.len() == index.len()
    }

    /// Bring the dense index's live set into step with `id_map`, which the
    /// loader does after it restores the mappings or replaces the graph.
    ///
    /// Taken in the declared order, `id_map` then the index, even though the
    /// loader holds `&mut self` and nothing else can be running.
    pub(crate) fn sync_dense_live(&self) {
        let id_map = self.id_map.read().unwrap();
        let mut index = self.dense().index.write().unwrap();
        index.set_live(id_map.values().copied());
    }

    /// The sparse space, where one was declared.
    pub(crate) fn sparse(&self) -> Option<&SparseSpace> {
        self.sparse_named().map(|(_, space)| space)
    }

    /// The sparse space with the name it was declared under, where one was
    /// declared. The name is the directory the space's artefacts are saved
    /// under.
    pub(crate) fn sparse_named(&self) -> Option<(&SpaceName, &SparseSpace)> {
        self.spaces.iter().find_map(|named| match &named.space {
            Space::Sparse(sparse) => Some((&named.name, sparse)),
            Space::Dense(_) => None,
        })
    }

    /// The prefix a sparse space's artefacts are written under, being
    /// `spaces/<name>/`.
    pub(crate) fn space_prefix(name: &SpaceName) -> String {
        format!("spaces/{}/", name)
    }

    /// The name of a text layer's dictionary artefact under `prefix`.
    pub(crate) fn dictionary_name(prefix: &str) -> String {
        format!("{prefix}terms.zdbdict")
    }

    /// Every artefact the sparse space writes, in the order it writes them,
    /// for the manifest's inventory. Empty where no sparse space is declared.
    pub(crate) fn space_artefact_names(&self) -> Vec<String> {
        let Some((name, space)) = self.sparse_named() else {
            return Vec::new();
        };
        let prefix = Self::space_prefix(name);
        let mut names = space.index.read().unwrap().artefact_names(&prefix);
        if space.text.is_some() {
            names.push(Self::dictionary_name(&prefix));
        }
        names
    }

    /// The text layer, where the sparse space has one. Refused where the
    /// collection declares no sparse space, or one that takes term ids
    /// alone, in that order.
    pub(crate) fn text_layer(&self) -> Result<&TextLayer, Error> {
        let space = self.sparse().ok_or(Error::NoSparseSpace)?;
        space.text.as_ref().ok_or(Error::NoTextLayer)
    }

    /// Every term of `text` as the sparse space's tokenizer hands them over,
    /// in order and repeats included, an empty term dropped.
    ///
    /// **Under no guard.** The tokenizer may be the caller's own and need
    /// something of the caller's to run, such as an interpreter lock. A
    /// thread holding that lock while it waited for the dictionary's guard,
    /// and a thread holding the dictionary's guard while its tokenizer
    /// waited for the lock, would wait for each other forever, so the
    /// engine never runs a tokenizer with a guard held: this takes none,
    /// and the terms collected here are counted under the dictionary's
    /// guard afterwards, by `add_records` for a record's
    /// [`SparseHalf::Terms`] and by a query for a text arm.
    ///
    /// A failure of the tokenizer itself comes back as the tokenizer
    /// returned it. Refused where the collection declares no sparse space,
    /// or one that takes term ids alone.
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>, Error> {
        let layer = self.text_layer()?;
        zeusdb_vector_text::tokenize(&*layer.tokenizer, text)
    }

    /// Count one record's terms, as [`Collection::tokenize`] handed them
    /// over, into the sparse space's term ids and term frequencies, issuing
    /// an id to every term not seen before, under the dictionary's guard
    /// taken alone.
    ///
    /// Reached from the insert path's admission alone, under `writers`, so a
    /// term id is issued only while the mutation guard is held and the
    /// record's postings are written under the same hold. There is
    /// deliberately no public way to count terms outside it: a caller
    /// holding ids across a gap in the guard would hold ids a `clear` in
    /// that gap could reissue. Refused where the collection declares no
    /// sparse space, or one that takes term ids alone.
    ///
    /// Every term given an id here is handed to the sink as an
    /// [`Operation::Intern`] at the moment of issue, under the dictionary's
    /// guard, so the records arrive in id order and each precedes the
    /// insert whose sparse half carries the id. A sink that refuses the
    /// record refuses the count, with the term interned, which is the state
    /// a replay of the records before it reaches.
    fn count_record_terms(&self, terms: &[String]) -> Result<SparseVector, Error> {
        let layer = self.text_layer()?;
        let mut dictionary = layer.dictionary.write().unwrap();
        count_record_with(&mut dictionary, terms, |term_id, term| {
            self.record(|| Operation::Intern {
                term_id,
                term: term.to_string(),
            })
        })
    }

    // ------------------------------------------------------------------
    // The seam
    // ------------------------------------------------------------------

    /// Attach the sink every mutation from now on hands its records to,
    /// returning the one it replaces. Under `writers`, so no batch is half
    /// recorded.
    pub fn attach_sink(&self, sink: Box<dyn OperationSink>) -> Option<Box<dyn OperationSink>> {
        let _writers = self.writers.lock().unwrap();
        self.attach_sink_locked(sink)
    }

    /// [`Collection::attach_sink`] with `writers` already held, for a caller
    /// that attaches and checkpoints under one hold so nothing mutates
    /// between the two. A fault the replaced sink left goes with it.
    pub(crate) fn attach_sink_locked(
        &self,
        sink: Box<dyn OperationSink>,
    ) -> Option<Box<dyn OperationSink>> {
        let mut slot = self.sink.lock().unwrap();
        slot.fault = None;
        slot.attached.replace(sink)
    }

    /// The id that pairs this collection with the journal beside its
    /// directory. See the field.
    pub fn collection_id(&self) -> u128 {
        self.collection_id
    }

    /// Take the id a directory's manifest recorded, so a collection loaded
    /// from a journaled directory is the collection its journal names.
    pub(crate) fn set_collection_id(&mut self, id: u128) {
        self.collection_id = id;
    }

    /// The sequence the checkpoint in the collection's directory holds.
    pub fn journal_sequence(&self) -> u64 {
        *self.journal_sequence.lock().unwrap()
    }

    /// Record the sequence a checkpoint holds, or the one a load read back
    /// from the manifest.
    pub(crate) fn set_journal_sequence(&self, sequence: u64) {
        *self.journal_sequence.lock().unwrap() = sequence;
    }

    /// The journal this collection hands its records to, as a caller sees
    /// it. `None` on a collection with no sink and on one whose sink is
    /// not a journal.
    ///
    /// The sink's guard is taken alone and released before the checkpoint
    /// sequence is read, so nothing is held across anything.
    pub fn journal_status(&self) -> Option<JournalStatus> {
        let (path, durability, sequence_reached, bytes) = {
            let slot = self.sink.lock().unwrap();
            let sink = slot.attached.as_ref()?;
            let path = sink.journal_path()?.to_path_buf();
            let durability = sink.durability()?;
            (
                path,
                durability,
                sink.sequence_reached(),
                sink.journal_bytes(),
            )
        };
        Some(JournalStatus {
            path,
            durability,
            checkpoint_sequence: self.journal_sequence(),
            sequence_reached,
            bytes,
        })
    }

    /// The interval policy's thread, where the sink runs one. Tests only.
    #[cfg(test)]
    pub(crate) fn flusher_watch(&self) -> Option<crate::flusher::Watch> {
        self.sink.lock().unwrap().attached.as_ref()?.flusher_watch()
    }

    /// Make every record handed over durable whatever the policy, and
    /// report the sequence the sink has reached, which is what a checkpoint
    /// records. `None` where no sink is attached.
    ///
    /// The mutation guard is the caller's, because a checkpoint syncs and
    /// then saves under one hold, so nothing is handed over between the two.
    /// A fault a failed commit left stands through this: the sync makes the
    /// refused records durable, and it is the truncation after the save
    /// that puts them behind the sequence the directory names.
    pub(crate) fn sync_sink_locked(&self) -> Result<Option<u64>, Error> {
        let mut slot = self.sink.lock().unwrap();
        let Some(sink) = slot.attached.as_mut() else {
            return Ok(None);
        };
        sink.sync()?;
        Ok(Some(sink.sequence_reached()))
    }

    /// Drop every record the sink holds, which the checkpoint that has just
    /// committed its directory now holds itself, and with them the fault a
    /// failed commit left, since the records it refused are now at or below
    /// the sequence the directory names and a replay skips them. Does
    /// nothing where no sink is attached. The mutation guard is the
    /// caller's.
    pub(crate) fn truncate_sink_locked(&self) -> Result<(), Error> {
        let mut slot = self.sink.lock().unwrap();
        match slot.attached.as_mut() {
            Some(sink) => {
                sink.truncate()?;
                slot.fault = None;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// The file name and the collection id the sink names, for the manifest
    /// a checkpoint writes. `None` where no sink is attached or where the
    /// sink is not a file.
    pub(crate) fn sink_journal_names(&self) -> Option<(String, u128)> {
        let slot = self.sink.lock().unwrap();
        let sink = slot.attached.as_ref()?;
        let file = sink.journal_file()?.to_string();
        let id = sink.journal_collection_id()?;
        Some((file, id))
    }

    /// The path the sink's file sits at, where it has one.
    pub(crate) fn sink_journal_path(&self) -> Option<std::path::PathBuf> {
        let slot = self.sink.lock().unwrap();
        slot.attached
            .as_ref()?
            .journal_path()
            .map(|p| p.to_path_buf())
    }

    /// Detach the sink, returning it, so that nothing is recorded from now
    /// on. The fault it left goes with it, since the fault is about its
    /// records. Under `writers`, as [`Collection::attach_sink`] is.
    pub fn detach_sink(&self) -> Option<Box<dyn OperationSink>> {
        let _writers = self.writers.lock().unwrap();
        let mut slot = self.sink.lock().unwrap();
        slot.fault = None;
        slot.attached.take()
    }

    /// Whether a sink is attached. The insert path asks once per batch,
    /// under `writers`, and skips the encode when none is.
    fn sink_attached(&self) -> bool {
        self.sink.lock().unwrap().attached.is_some()
    }

    /// Hand one operation's record to the sink, where one is attached,
    /// building the operation only then. For the kinds whose record is a
    /// few bytes; the insert path encodes its own record ahead of the id
    /// and hands the bytes to [`Collection::append_record`]. Refused with
    /// the fault while one stands.
    fn record(&self, operation: impl FnOnce() -> Operation) -> Result<(), Error> {
        let mut slot = self.sink.lock().unwrap();
        slot.refuse_if_faulted()?;
        let Some(sink) = slot.attached.as_mut() else {
            return Ok(());
        };
        let operation = operation();
        let mut payload = Vec::new();
        operation.encode(&mut payload);
        sink.append(operation.kind(), &payload)
    }

    /// Hand one record's bytes, already encoded, to the sink, where one is
    /// attached. Refused with the fault while one stands.
    fn append_record(&self, kind: OperationKind, payload: &[u8]) -> Result<(), Error> {
        let mut slot = self.sink.lock().unwrap();
        slot.refuse_if_faulted()?;
        match slot.attached.as_mut() {
            Some(sink) => sink.append(kind, payload),
            None => Ok(()),
        }
    }

    /// Commit what the sink holds, where one is attached, after the last
    /// record of a call and before the call mutates.
    ///
    /// A commit that fails is the one failure the seam holds on to. The
    /// caller refuses every record the commit was to make durable, so they
    /// are in the sink and not in the collection, and they may or may not
    /// be on the device. The failure is written onto the slot with the
    /// sequence the sink had reached, and every record after it is refused
    /// with it until a checkpoint reconciles the two; see [`SinkSlot`].
    fn commit_records(&self) -> Result<(), Error> {
        let mut slot = self.sink.lock().unwrap();
        let Some(sink) = slot.attached.as_mut() else {
            return Ok(());
        };
        match sink.commit() {
            Ok(()) => Ok(()),
            Err(error) => {
                let sequence = sink.sequence_reached();
                let message = error.to_string();
                tracing::error!(target: "zeusdb_vector_database::journal",
                    operation = "journal_commit_failed",
                    sequence = sequence,
                    error = %message,
                    "The journal's commit failed; its records are refused and nothing is \
                     recorded until a checkpoint reconciles the journal with the directory"
                );
                slot.fault = Some(SinkFault {
                    sequence,
                    error: message,
                });
                Err(error)
            }
        }
    }

    /// Distinct terms the sparse space's dictionary holds, where it has
    /// one.
    pub fn term_count(&self) -> Option<usize> {
        let layer = self.sparse()?.text.as_ref()?;
        Some(layer.dictionary.read().unwrap().len())
    }

    /// Every space's name and declaration, in declaration order.
    pub fn space_configs(&self) -> Vec<(SpaceName, SpaceConfig)> {
        self.spaces
            .iter()
            .map(|named| {
                let config = match &named.space {
                    Space::Dense(dense) => SpaceConfig::Dense(dense.config()),
                    Space::Sparse(sparse) => sparse.space_config(),
                };
                (named.name.clone(), config)
            })
            .collect()
    }

    /// The vector width.
    pub fn dim(&self) -> usize {
        self.dense().dim
    }

    /// The distance space, normalised: cosine, l2, l1 or dot.
    pub fn metric(&self) -> &str {
        &self.dense().metric
    }

    /// The graph degree. See `Space::m`.
    pub fn m(&self) -> usize {
        self.dense().m()
    }

    /// The construction width. See `Space::ef_construction`.
    pub fn ef_construction(&self) -> usize {
        self.dense().ef_construction()
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
        self.dense().has_quantization()
    }

    /// Whether the codebook is fitted.
    pub fn can_use_quantization(&self) -> bool {
        self.dense().can_use_quantization()
    }

    /// Whether the graph scores against codes. Takes the graph's read guard.
    pub fn is_quantized(&self) -> bool {
        self.dense().is_quantized()
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
        // fewer reads. The columns guard is released before the metadata
        // guard is taken, since the selection owns its bitmap, and the store
        // is indexed by the internal ids the bitmap holds, so nothing else
        // is read. A removed record holds no entry, which is the liveness
        // check.
        let selection = {
            let columns = self.columns.read().unwrap();
            columns.select(conditions)
        };
        match selection {
            Selection::Exact(selected) => selected.count(),
            Selection::Narrowed(bound, _) => {
                let vector_metadata = self.vector_metadata.read().unwrap();
                let mut counted = 0;
                bound.for_each(|slot| {
                    if vector_metadata
                        .get(slot)
                        .is_some_and(|fields| matches_filter(&fields, conditions))
                    {
                        counted += 1;
                    }
                });
                counted
            }
            Selection::Whole(_) => {
                let vector_metadata = self.vector_metadata.read().unwrap();
                vector_metadata
                    .iter()
                    .filter(|(_, fields)| matches_filter(fields, conditions))
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
        let index = self.dense().index.read().unwrap();
        let pq_codes = self.dense().pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let raws = crate::RawVectors {
            id_map: &id_map,
            graph: index.graph(),
        };

        for id in ids {
            // Check if this ID exists in either storage
            let exists = id_map.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = id_map
                    .get(&id)
                    .and_then(|&slot| vector_metadata.get(slot))
                    .map(|fields| fields.to_map())
                    .unwrap_or_default();

                let vector = if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    if let Some(raw_vector) = raws.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.to_vec())
                    } else if let (Some(pq), Some(codes)) = (&self.dense().pq, pq_codes.get(&id)) {
                        // Case 2: Only quantized codes available (QuantizedOnly mode)
                        match pq.reconstruct(codes) {
                            Ok(reconstructed) => Some(reconstructed),
                            Err(e) => {
                                warn!(target: LOG_TARGET, operation = "vector_reconstruction", vector_id = %id, error = %e, "Failed to reconstruct vector");
                                None
                            }
                        }
                    } else {
                        // Case 3: a scalar row, decoded through the scales,
                        // and nothing where the record holds no vector data
                        id_map
                            .get(&id)
                            .and_then(|&slot| index.graph().int8_reconstruct(slot))
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
        for &(internal, id) in page.iter() {
            let metadata = vector_metadata
                .get(internal)
                .map(|fields| fields.to_map())
                .unwrap_or_default();
            results.push((id.clone(), metadata));
        }
        Ok(results)
    }

    /// Add index-level metadata, merging the pairs into what is held.
    ///
    /// Under the mutation guard, taken first as every other durable mutation
    /// takes it, so a save, which holds the guard for its four phases, and
    /// every other mutation are ordered against this one. The metadata mutex
    /// alone kept the map whole; it did not order this call against the
    /// mutations a save records. The pairs are what `config.json` writes
    /// under `metadata`. The binding calls this with the interpreter lock
    /// released, since waiting for another writer with it held would stall
    /// every Python thread for the length of that writer.
    ///
    /// The pairs are handed to the sink, in key order, before they are
    /// merged, and a sink that refuses them refuses the call.
    pub fn add_metadata(&self, metadata: HashMap<String, String>) -> Result<(), Error> {
        let _writers = self.writers.lock().unwrap();
        self.record(|| {
            let mut pairs: Vec<(String, String)> = metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            pairs.sort();
            Operation::AddMetadata { pairs }
        })?;
        self.commit_records()?;
        self.add_metadata_locked(metadata);
        Ok(())
    }

    /// The body of `add_metadata`, with the mutation guard held by the
    /// caller and the record already handed over.
    pub(super) fn add_metadata_locked(&self, metadata: HashMap<String, String>) {
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
    use super::construct::MAX_DIM;
    use super::LiveRecords;
    use std::collections::HashMap;
    use zeusdb_vector_core::JOURNAL_MAX_PAYLOAD;

    /// The journal's payload ceiling is derived from this crate's ceilings,
    /// being a codebook at the widest `dim` and the most centroids `bits`
    /// admits, plus a mebibyte, so a change to either ceiling here is a
    /// change to the journal's bound as well.
    #[test]
    fn the_journal_payload_ceiling_follows_the_declaration_ceilings() {
        let most_centroids = 1usize << 8;
        assert_eq!(
            JOURNAL_MAX_PAYLOAD,
            MAX_DIM * most_centroids * 4 + (1 << 20)
        );
    }

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
