//! The index, above the engine's floor and below the binding.
//!
//! [`Collection`] is the record set with its vector spaces over it, and every
//! operation the binding exposes is a method on it: building, adding,
//! searching, removing, rebuilding, saving and loading. The dense space is
//! this crate's graph behind the seam `zeusdb_vector_core` declares, and the
//! sparse space is `zeusdb_vector_sparse` behind the same seam. Beside them
//! are the rerank rule, being how far a quantized search over-fetches before
//! it rescores against raw vectors and how that depth is measured on the
//! index's own data, the lock rank registry, being every lock the index
//! holds with its place in the declared acquisition order, the persistence
//! of a saved directory, and the write-ahead journal beside one, being the
//! sink every mutation's record reaches and the recovery that replays it
//! back. Nothing here names Python. The binding
//! holds the `#[pyclass]` that wraps a `Collection` by value, parses every
//! argument, releases the interpreter lock around the call and converts what
//! comes back.
//!
//! Every configuration type a space is declared with is re-exported here,
//! so a caller declaring a space depends on this crate alone.
//!
//! The binding takes this crate as a path dependency and reaches it through
//! the `pub use` list below and the one public module, `locks`, whose ranks
//! are named by path at every lock's declaration. Every other module is
//! private. An item a module marks `pub` that this file does not re-export
//! is unreachable, and `unreachable_pub` below makes it a warning, which the
//! lint gate turns into a failure, so the surface cannot widen without a
//! line here.
//!
//! # Log records
//!
//! Every module here that emits a `tracing` record names its target as
//! `zeusdb_vector_database::...`, the module path it carried in the binding,
//! rather than taking this crate's name from `module_path!()`, for the reason
//! the crate root of zeusdb-vector-core gives. See `LOG_TARGET` at the top of
//! each file under `collection/`, in `persistence.rs`, in `journal.rs` and in
//! `flusher.rs`.
//!
//! # Tests
//!
//! `cargo test -p zeusdb-vector-hnsw` compiles this crate, the engine's floor
//! and their dependencies alone.
#![warn(unreachable_pub)]

mod collection;
mod flusher;
mod journal;
pub mod locks;
mod persistence;
mod rerank;

pub use collection::{
    Added, AdmitShape, Arm, ArmPlan, Collection, Declaration, DenseConfig, JournalStatus, Listing,
    OperationSink, Page, ParsedRecord, ParsedRecords, Plan, QuantizationConfig, QuantizationReport,
    QuantizerReport, Query, QueryHit, QueryHits, RebuildPlan, RecordView, SpaceConfig, SparseHalf,
    SparseHits, StorageMode, TextConfig, DEFAULT_FETCH_PER_K, DEFAULT_SPACE, MAX_ARMS,
};
#[cfg(test)]
pub use flusher::Watch;
pub use journal::{
    directory_of, journal_path, Durability, JournalPolicy, JournalSink, Recovery, JOURNAL_SUFFIX,
};
pub use rerank::{
    calibrate_rerank_from_sample, default_rerank_fetch, prepare_reconstruction, raw_distance_fn,
    reconstruction_needs_unit, rescore_candidate, take_best, RawVectors, RerankCalibration,
    RerankPlan, SearchParams, DEFAULT_RERANK_CORPUS_DIVISOR, RERANK_CALIBRATION_PAGES,
    RERANK_CALIBRATION_TOP_K,
};
pub use zeusdb_vector_core::{
    kill_arm, kill_disarm, CommitMode, Contribution, Cost, Fusion, IdfScope, JournalDamage,
    Operation, OperationKind, SpaceKind, SpaceName, DEFAULT_RRF_K,
};
pub use zeusdb_vector_sparse::{SparseConfig, Unlink, Weighting};
pub use zeusdb_vector_text::{SimpleTokenizer, TermDictionary, Tokenizer, TokenizerConfig};
