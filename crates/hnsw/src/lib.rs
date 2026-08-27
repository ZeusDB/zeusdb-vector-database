//! The graph index's own rules, above the engine's floor and below the
//! binding.
//!
//! Two things live here today, and neither names Python or reads an index.
//! The rerank rule, being how far a quantized search over-fetches before it
//! rescores against raw vectors and how that depth is measured on the index's
//! own data, and the lock rank registry, being every lock the index holds
//! with its place in the declared acquisition order. The index type itself,
//! its persistence and its Python entry points stay in the binding until
//! they can move together.
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
//! No module here emits a `tracing` record. A module that comes to emit one
//! names its target as `zeusdb_vector_database::...` rather than taking this
//! crate's name from `module_path!()`, for the reason the crate root of
//! zeusdb-vector-core gives.
//!
//! # Tests
//!
//! `cargo test -p zeusdb-vector-hnsw` compiles this crate, the engine's floor
//! and their dependencies alone.
#![warn(unreachable_pub)]

pub mod locks;
mod rerank;

pub use rerank::{
    calibrate_rerank_from_sample, default_rerank_fetch, prepare_reconstruction, raw_distance_fn,
    reconstruction_needs_unit, rescore_candidate, take_best, RawVectors, RerankCalibration,
    RerankPlan, SearchParams, DEFAULT_RERANK_CORPUS_DIVISOR, RERANK_CALIBRATION_PAGES,
    RERANK_CALIBRATION_TOP_K,
};
