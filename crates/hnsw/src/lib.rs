//! The index, above the engine's floor and below the binding.
//!
//! [`Collection`] is the record set with one vector space over it, and every
//! operation the binding exposes is a method on it: building, adding,
//! searching, removing, rebuilding, saving and loading. Beside it are the
//! rerank rule, being how far a quantized search over-fetches before it
//! rescores against raw vectors and how that depth is measured on the index's
//! own data, the lock rank registry, being every lock the index holds with its
//! place in the declared acquisition order, and the persistence of a saved
//! directory. Nothing here names Python. The binding holds the `#[pyclass]`
//! that wraps a `Collection` by value, parses every argument, releases the
//! interpreter lock around the call and converts what comes back.
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
//! each file under `collection/` and in `persistence.rs`.
//!
//! # Tests
//!
//! `cargo test -p zeusdb-vector-hnsw` compiles this crate, the engine's floor
//! and their dependencies alone.
#![warn(unreachable_pub)]

mod collection;
pub mod locks;
mod persistence;
mod rerank;

pub use collection::{
    Added, Collection, Declaration, Listing, ParsedRecords, QuantizationConfig, QuantizationReport,
    QuantizerReport, QueryHits, RebuildPlan, RecordView, StorageMode,
};
pub use rerank::{
    calibrate_rerank_from_sample, default_rerank_fetch, prepare_reconstruction, raw_distance_fn,
    reconstruction_needs_unit, rescore_candidate, take_best, RawVectors, RerankCalibration,
    RerankPlan, SearchParams, DEFAULT_RERANK_CORPUS_DIVISOR, RERANK_CALIBRATION_PAGES,
    RERANK_CALIBRATION_TOP_K,
};
