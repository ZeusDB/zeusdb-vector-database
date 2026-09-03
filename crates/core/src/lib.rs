//! The engine's floor.
//!
//! What every index crate stands on, and it names nothing of Python: the
//! error type, the checksum, the seeded generator, the filter language and
//! the column store that answers it, the distance kernels, the product
//! quantizer, the graph with its dump format, the seam every index sits
//! behind, being the index trait, the admit family and the persistence traits
//! in `space` and `admit`, the fusion that combines several indexes'
//! pages into one, and the write-ahead journal's format with its reader and
//! the operations it records. The binding takes this crate as a path dependency
//! and reaches it through the re-exports below, so the crate's surface is
//! this file's `pub use` list and nothing else.
//! Every module is private. An item a module marks `pub` that this file does
//! not re-export is unreachable, and `unreachable_pub` below makes it a
//! warning, which the lint gate turns into a failure, so the surface cannot
//! widen without a line here.
//!
//! # Log records
//!
//! The two modules that emit `tracing` records name their target as
//! `zeusdb_vector_database::...`, which is the package a user configures
//! logging by, rather than taking this crate's name from `module_path!()`.
//! The filter directive the binding installs and a `RUST_LOG` directive both
//! match a target by prefix, so a module moving between crates must not move
//! its records out of that prefix. See `LOG_TARGET` in `graph/mod.rs` and
//! `graph/dump.rs`.
//!
//! # Tests
//!
//! `cargo test -p zeusdb-vector-core` compiles this crate and its
//! dependencies alone. The `test-support` feature exposes [`test_support`]
//! for a test in another crate that measures against the same data, and
//! nothing but a test target turns it on.
#![warn(unreachable_pub)]

mod admit;
mod checksum;
mod columns;
mod distance;
mod error;
mod filter;
mod frame;
mod fusion;
mod graph;
mod journal;
mod operation;
mod pq;
mod rng;
mod space;
// Test data two test modules measure against, so it is defined once.
#[cfg(any(test, feature = "test-support"))]
mod test_vectors;

pub use admit::{Admit, And, Candidates};
pub use checksum::checksum_of;
pub use columns::{validate_indexed_fields, Bitmap, ColumnStore, Selection};
pub use distance::{CosineDist, DistPQ, DotDist, L1Dist, L2Dist, PqMetric};
pub use error::{Error, Exception};
pub use filter::{compile_filter, matches_filter, Filter};
pub use frame::{
    begin as frame_begin, finish as frame_finish, frame, fuzz as frame_fuzz, unframe,
    FrameEncoding, FrameKind, Framed, FRAME_HEADER_BYTES, FRAME_MAGIC, FRAME_OVERHEAD_BYTES,
    FRAME_TRAILER_BYTES,
};
pub use fusion::{fuse, Contribution, FusedHit, Fusion, DEFAULT_RRF_K};
pub use graph::dump::{DUMP_FILENAME, LEGACY_DUMP_FILENAMES, NB_LAYER_MAX};
pub use graph::{restore_graph, Distance, DumpBounds, GraphHit, Planned, Record, VectorGraph};
pub use journal::{
    encode_journal_header, encode_journal_record, read_journal, CommitMode, JournalContents,
    JournalDamage, JournalHeader, JournalKind, JournalRecord, JournalSyncHandle, JournalWriter,
    OperationKind, JOURNAL_HEADER_BYTES, JOURNAL_MAGIC, JOURNAL_MAX_PAYLOAD,
    JOURNAL_RECORD_CHECKSUM_BYTES, JOURNAL_RECORD_HEADER_BYTES, JOURNAL_RECORD_OVERHEAD_BYTES,
};
pub use operation::{InsertParts, Operation};
pub use pq::PQ;
pub use rng::SeededRng;
pub use space::{
    read_artefact, write_artefact, ArtefactRecord, Bounds, Budget, CorpusStats, Cost, Dense, Hit,
    Hits, IdfScope, Inventory, Kind, Ledger, Persist, Prepared, RecordId, Restore, ScoreKind,
    Selectivity, SpaceKind, SpaceName, Sparse, SparseRef, SparseVector, VectorIndex,
};

/// What a test in another crate measures against.
///
/// Present under the `test-support` feature and in this crate's own tests,
/// and absent from every other build.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    pub use crate::test_vectors::clustered;
}
