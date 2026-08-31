//! The sparse index, being a mutable postings index implementing the seam in
//! `zeusdb-vector-core` for sparse vectors, scored by the sparse dot product
//! or by term frequency weighting.
//!
//! # The structure
//!
//! One `Vec` of eight byte postings per distinct dimension, reached through a
//! dimension-to-slot map because a dimension is an arbitrary `u32`, sorted by
//! record id and appended at the tail. The collection allocates internal ids
//! from a counter that never goes backwards and never reuses a value, so every
//! list receives its entries in increasing order and nothing ever inserts into
//! the middle of one. The structure keeps a binary-search insert for an id out
//! of order, so the property is its own rather than the caller's, and the
//! engine never takes that path.
//!
//! Beside the lists sits a forward arena holding every record's vector
//! contiguously, with a span per record. It costs as many bytes as the
//! postings themselves and it is load-bearing three times over. `remove`
//! walks the record's own dimensions from it rather than every list, `compact`
//! rebuilds the lists from it, and the enumerate-driven search scores every
//! admitted record from it, which is the path that wins when the admit set is
//! small.
//!
//! # Removal
//!
//! Removal is lazy. It marks the record dead, counts one dead posting on each
//! list the record sits in, and rewrites a list once its dead share crosses a
//! threshold. A stranded posting is pure waste in every scan of its list, and
//! at half the corpus removed stranding alone triples a scan where the lazy
//! rewrite holds it to a fifth over a compacted index. `compact` rewrites
//! every list and the arena at once. The two other policies, stranding
//! outright and unlinking eagerly, exist so the choice stays measurable.
//!
//! # Search
//!
//! Term-at-a-time accumulation into a dense scratch buffer with a bounded
//! top-k heap. An admit set that is a bitmap is tested by a loop
//! monomorphised over the bit, since the trait-object call is a third of the
//! scan when asked once per posting. A small admit set drives the search
//! instead, being scored record by record from the forward arena, and the
//! index chooses between the two from the unit costs it timed on itself.
//!
//! # Term frequency weighting
//!
//! A space declared with `Weighting::Bm25` reads every stored value as a term
//! frequency. Nothing derived from a corpus statistic is stored: a posting
//! carries the raw frequency, a record carries its length beside its arena
//! span, the index keeps the live count, the length total and the live
//! postings per list, and a query is weighted by the rarity of its terms at
//! the moment it is asked, over the admitted records by default. The scan
//! then reads one length per admitted posting and divides once. A record
//! arriving or leaving therefore moves every statistic at once and no stored
//! weight goes stale, at the cost of that gather and division per posting.
//!
//! # What is not here
//!
//! No tokenizer, since the index sees term ids and weights alone, no
//! block-max pruning and no segmented structure. A whole scan at fifty
//! thousand records is under a millisecond at its tail, which is the same
//! wall time as a graph search on the same records, and the structures that
//! cut it pay for themselves an order of magnitude above that.
//!
//! # Log records
//!
//! Every record this crate emits names its target as
//! `zeusdb_vector_database::sparse`, which is the package a user configures
//! logging by, rather than taking this crate's name from `module_path!()`.
//! The filter directive the binding installs and a `RUST_LOG` directive both
//! match a target by prefix, so a record carrying this crate's name would fall
//! outside both and be dropped.
#![warn(unreachable_pub)]

mod calibrate;
#[cfg(test)]
mod corpus;
mod index;
mod persist;
mod search;
#[cfg(test)]
mod verify;

pub use calibrate::UnitCosts;
pub use index::{
    HeapBytes, PostingsIndex, SparseConfig, Unlink, Weighting, DEFAULT_BM25_B, DEFAULT_BM25_K1,
    DEFAULT_LAZY_THRESHOLD_PERCENT,
};
pub use search::Mode;

/// The target every record this crate emits carries. See the crate
/// documentation.
const LOG_TARGET: &str = "zeusdb_vector_database::sparse";
