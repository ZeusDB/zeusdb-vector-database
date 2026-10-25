//! Insertion, replacement, removal, compaction, rebuild and clear.
//!
//! `add` is the whole of what the Python entry point does once parsing is
//! over, and the binding calls it with the interpreter lock released. The
//! three `add_*` paths below `insert_parsed_records` are chosen by whether the
//! index is quantized and whether it is still collecting for training. Removal
//! is logical, and `compact` reclaims the graph nodes that removal and
//! replacement leave behind. A removal batch that leaves more holes than the
//! allowance and the records held closes the positions up once it has let its
//! guards go, and the graph keeps its nodes.
//!
//! Every mutating operation the binding calls takes `writers` here, first and
//! before any guard, and the helpers under it never do. A helper that closes
//! the positions up borrows the guard from its caller, so the close-up is
//! seen to run under it.
//!
//! # The close-up
//!
//! One routine moves every per-record structure so the records held sit at
//! the positions below their count, for a removal batch that passes the bound,
//! a compaction, a rebuild at a new degree and both training rebuilds. It
//! runs in two phases under `writers`, which keeps every other writer out, so
//! nothing it reads can change between them. The first builds new copies of
//! the structures under read guards, which searches share, so a search runs
//! on the structures as they are for as long as the copies take. The
//! metadata's copy is a set of empty entries, since its blocks move rather
//! than being copied. The second takes every write guard in the declared
//! order, moves each metadata block into its new entry and trades the copies
//! for the structures, a move of each, so a search waits for the moves alone
//! and sees every structure before the close-up or every structure after it.
//! The structures it replaced are dropped once `writers` is let go, so the
//! next writer does not wait for them; see [`MutationGuard`].
//!
//! # The seam
//!
//! Every mutation hands its record to the collection's sink, where one is
//! attached, after it has taken `writers` and before it takes any storage
//! guard, then commits the sink, then runs. See [`Collection::attach_sink`]
//! and `insert_parsed_records` for the order inside a batch. The `_locked`
//! bodies below the entry points hand nothing over, so `Collection::apply`
//! can run them for a recorded operation without recording it again.

use super::{
    validate_index_parameters, Collection, DenseIndex, DenseParts, ParsedRecord, ParsedRecords,
    SparseHalf, StorageMode, HOLE_ALLOWANCE, MAX_LAYER,
};
use crate::locks::{Locked, WriteGuard};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace, warn};
use zeusdb_vector_core::{
    matches_filter, refuse_unnamable, Bitmap, ColumnParts, ColumnStore, Error, Filter, GraphRoom,
    IdParts, IdStore, InsertParts, MetadataParts, MetadataStore, Operation, OperationKind,
    Position, Prepared, RecordId, Selection, SparseVector, VectorGraph, VectorIndex,
    JOURNAL_MAX_PAYLOAD, MAX_INTERNAL_ID,
};
use zeusdb_vector_sparse::{refuse_forward, PostingsIndex, PostingsParts};

#[cfg(test)]
use zeusdb_vector_core::MetadataCopy;

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::insert";

/// The target the records of `add` and `clear` carry. Both were entry point
/// bodies in the parent module before they moved here, so their records keep
/// the parent's target and a filter directive that matched them still does.
const ENTRY_TARGET: &str = super::LOG_TARGET;
/// Multiple of `expected_size` at which an index warns that it has outgrown its
/// declaration. Fires once per index.
const EXPECTED_SIZE_OVERGROWTH_FACTOR: usize = 2;
/// An error raised inside `add`'s insertion phase, carried out to be recorded
///
/// The insertion phase runs with the interpreter lock released. The error a
/// record's insertion raises is carried out as an `Error`, whose `Display`
/// touches nothing of the interpreter, so it can be formatted anywhere. `add`
/// formats it once the lock is back, which is where the per record messages
/// are assembled.
pub(super) enum InsertError {
    /// A message Rust already holds, counted against `total_errors`
    Counted(String),

    /// A training failure. Recorded but not counted, which is what the training
    /// path has always done, because a training failure is not a rejected record
    Training(String),

    /// An error from one of the three insert paths, with the id it belongs to.
    /// Counted against `total_errors` once formatted, as
    /// `Vector <id>: <class>: <message>`, which is what the `PyErr` it used to
    /// carry displayed as
    Vector { id: String, err: Error },
}
/// The write guards a removal holds, taken in the order `Collection` declares.
///
/// A struct rather than five locals so that a batch can take them once and lend
/// them to the per record helper. They drop in field order, which is the
/// acquisition order, and releasing may happen in any order.
struct RemovalGuards<'a> {
    ids: WriteGuard<'a, IdStore>,
    /// The dense index, so a removal drops the record from its live set.
    /// Taken as a write guard between the reverse map and the codes, which
    /// is its place in the order, and held for the removal alone.
    index: WriteGuard<'a, DenseIndex>,
    pq_codes: WriteGuard<'a, HashMap<String, Vec<u8>>>,
    /// The sparse index, where the collection declares a sparse space.
    sparse: Option<WriteGuard<'a, PostingsIndex>>,
    vector_metadata: WriteGuard<'a, MetadataStore>,
    columns: WriteGuard<'a, ColumnStore>,
}

impl RemovalGuards<'_> {
    /// The records the id store holds and the holes among its positions,
    /// read as a batch's last step for the close-up's test, which runs once
    /// the batch has let its guards go.
    fn held_and_holes(&self) -> (usize, usize) {
        (self.ids.len(), self.ids.extent() - self.ids.len())
    }
}

// How many times the close-up on removal has run on this thread. Read by the
// tests that hold it to the points it must run at; nothing outside the tests
// sees it.
#[cfg(test)]
thread_local! {
    pub(super) static CLOSE_UPS_ON_REMOVAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The points a close-up passes, for a test that holds a close-up at one of
/// them or counts what overlapped it. Nothing outside the tests sees them.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CloseUpPoint {
    /// Every read guard the build takes is held, and the copies are about
    /// to be made.
    Building,
    /// The copies are made and the read guards let go.
    Built,
    /// Every write guard is held, and the copies are about to go in.
    Swapping,
    /// The copies are in and the write guards let go.
    Swapped,
    /// The structures the swap replaced are about to be dropped.
    Dropping,
    /// The structures the swap replaced are dropped.
    Dropped,
}

/// What a test sets for this thread's close-ups to call at each point.
#[cfg(test)]
pub(super) type CloseUpHook = Box<dyn FnMut(CloseUpPoint)>;

// What this thread's close-ups call at each point, where a test set it.
#[cfg(test)]
thread_local! {
    pub(super) static CLOSE_UP_HOOK: std::cell::RefCell<Option<CloseUpHook>> =
        const { std::cell::RefCell::new(None) };
}

/// Every structure a close-up moves, each as its parts with their
/// capacities, for the test that holds the close-up built off to the side
/// to the one made in place.
#[cfg(test)]
pub(super) type Layouts = Vec<(&'static str, Vec<(&'static str, usize, String)>)>;

/// The layouts of the structures the guards hold.
#[cfg(test)]
fn layouts(guards: &RemovalGuards<'_>) -> Layouts {
    let mut out = vec![
        ("ids", guards.ids.layout()),
        ("index", guards.index.layout()),
        ("metadata", guards.vector_metadata.layout()),
        ("columns", guards.columns.layout()),
    ];
    if let Some(sparse) = guards.sparse.as_ref() {
        out.push(("sparse", sparse.layout()));
    }
    out
}

/// Call this thread's close-up hook, where a test set one.
#[cfg(test)]
fn reach(point: CloseUpPoint) {
    CLOSE_UP_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow_mut().as_mut() {
            hook(point);
        }
    });
}

/// Every structure's part of one close-up, built off to the side and then
/// traded for the structures' own under every write guard.
///
/// A part is absent where the close-up leaves its structure as it is: every
/// store where no position holds a hole, the dense index where no position
/// holds a hole and no graph replaces its own, the sparse index where the
/// collection declares no sparse space, and the column store where it
/// declares no field. Once the swap has run, each part holds what its
/// structure held, for the caller to drop with no guard held.
pub(super) struct ClosedUp {
    index: Option<DenseParts>,
    sparse: Option<PostingsParts>,
    ids: Option<IdParts>,
    metadata: Option<MetadataParts>,
    columns: Option<ColumnParts>,
}

impl ClosedUp {
    /// Build every part from the structures as the read guards hold them.
    ///
    /// The positions held are listed in increasing order, `old_of_new`, and
    /// mapped to their ranks, `new_of_old`, a hole mapping to the empty
    /// marker. Both are transient, four bytes a position. Where `replacement`
    /// is given, built off to the side with the record at old position
    /// `old_of_new[k]` filed at `k`, the dense index's part is that graph and
    /// the numbering read from it; where it is not, the graph keeps every
    /// node and the index's part is its numbering moved. Each store's part
    /// is what its close-up leaves, the sparse space dropping the postings
    /// of the records it holds dead and the id store its removed text, each
    /// at the capacity the pass in place leaves it. The metadata's part is
    /// its new entries, empty until the swap moves each block into place,
    /// with `old_of_new` kept as where each block comes from.
    fn build(
        ids: &IdStore,
        index: &DenseIndex,
        sparse: Option<&PostingsIndex>,
        metadata: &MetadataStore,
        columns: &ColumnStore,
        replacement: Option<(VectorGraph, Option<(f64, f64)>)>,
        expected_size: usize,
    ) -> ClosedUp {
        let extent = ids.extent();
        let old_of_new = ids.held_positions();
        let holes = old_of_new.len() < extent;
        let index = match replacement {
            Some((graph, timed)) => Some(index.replaced(graph, timed)),
            None => holes.then(|| index.closed_up(&old_of_new, expected_size)),
        };
        if !holes {
            return ClosedUp {
                index,
                sparse: None,
                ids: None,
                metadata: None,
                columns: None,
            };
        }
        let mut new_of_old = vec![Position::NONE; extent];
        for (k, &old) in old_of_new.iter().enumerate() {
            new_of_old[old as usize] = k as u32;
        }
        ClosedUp {
            index,
            sparse: sparse.map(|sparse| sparse.closed_up(&old_of_new, &new_of_old)),
            ids: Some(ids.closed_up(&old_of_new, &new_of_old, expected_size)),
            columns: columns.closed_up(&old_of_new, expected_size),
            // Last, since the metadata's part keeps `old_of_new` as the
            // positions its swap moves the blocks from.
            metadata: Some(metadata.closed_up(old_of_new, expected_size)),
        }
    }

    /// Trade every part for its structure's own, under the write guards the
    /// caller holds. Moves alone: the metadata's trade first moves each
    /// block it keeps from its old entry into its new one; see
    /// [`MetadataStore::swap_parts`].
    fn swap_into(&mut self, guards: &mut RemovalGuards<'_>) {
        if let Some(index) = self.index.as_mut() {
            guards.index.swap_parts(index);
        }
        if let (Some(sparse), Some(parts)) = (guards.sparse.as_mut(), self.sparse.as_mut()) {
            sparse.swap_parts(parts);
        }
        if let Some(ids) = self.ids.as_mut() {
            guards.ids.swap_parts(ids);
        }
        if let Some(metadata) = self.metadata.as_mut() {
            guards.vector_metadata.swap_parts(metadata);
        }
        if let Some(columns) = self.columns.as_mut() {
            guards.columns.swap_parts(columns);
        }
    }

    /// Put every structure back as it was before [`ClosedUp::swap_into`],
    /// under the write guards the caller holds, for a test that runs a
    /// second close-up from the same state: the metadata moves each block
    /// back into its old entry, and every other part is traded again.
    #[cfg(test)]
    fn swap_back(&mut self, guards: &mut RemovalGuards<'_>) {
        if let Some(index) = self.index.as_mut() {
            guards.index.swap_parts(index);
        }
        if let (Some(sparse), Some(parts)) = (guards.sparse.as_mut(), self.sparse.as_mut()) {
            sparse.swap_parts(parts);
        }
        if let Some(ids) = self.ids.as_mut() {
            guards.ids.swap_parts(ids);
        }
        if let Some(metadata) = self.metadata.as_mut() {
            guards.vector_metadata.swap_parts_back(metadata);
        }
        if let Some(columns) = self.columns.as_mut() {
            guards.columns.swap_parts(columns);
        }
    }

    /// Bytes of id text the close-up let go of, read once the swap has run.
    fn removed_text(&self) -> usize {
        self.ids.as_ref().map_or(0, IdParts::removed_text)
    }
}

/// Every structure's part of one close-up, as the close-up off to the side
/// first built them: the positions held read from the id store's entries,
/// and every block of metadata cloned. Kept for the tests that hold
/// [`ClosedUp`] to the same result.
#[cfg(test)]
pub(super) struct ClosedUpCopied {
    index: Option<DenseParts>,
    sparse: Option<PostingsParts>,
    ids: Option<IdParts>,
    metadata: Option<MetadataCopy>,
    columns: Option<ColumnParts>,
}

#[cfg(test)]
impl ClosedUpCopied {
    /// Build every part from the structures as the read guards hold them;
    /// see [`ClosedUp::build`].
    fn build(
        ids: &IdStore,
        index: &DenseIndex,
        sparse: Option<&PostingsIndex>,
        metadata: &MetadataStore,
        columns: &ColumnStore,
        replacement: Option<(VectorGraph, Option<(f64, f64)>)>,
        expected_size: usize,
    ) -> ClosedUpCopied {
        let extent = ids.extent();
        let old_of_new: Vec<u32> = ids.slots().map(|at| at as u32).collect();
        let holes = old_of_new.len() < extent;
        let index = match replacement {
            Some((graph, timed)) => Some(index.replaced(graph, timed)),
            None => holes.then(|| index.closed_up(&old_of_new, expected_size)),
        };
        if !holes {
            return ClosedUpCopied {
                index,
                sparse: None,
                ids: None,
                metadata: None,
                columns: None,
            };
        }
        let mut new_of_old = vec![Position::NONE; extent];
        for (k, &old) in old_of_new.iter().enumerate() {
            new_of_old[old as usize] = k as u32;
        }
        ClosedUpCopied {
            index,
            sparse: sparse.map(|sparse| sparse.closed_up(&old_of_new, &new_of_old)),
            ids: Some(ids.closed_up(&old_of_new, &new_of_old, expected_size)),
            metadata: Some(metadata.closed_up_copied(&old_of_new, expected_size)),
            columns: columns.closed_up(&old_of_new, expected_size),
        }
    }

    /// Trade every part for its structure's own, under the write guards the
    /// caller holds.
    fn swap_into(&mut self, guards: &mut RemovalGuards<'_>) {
        if let Some(index) = self.index.as_mut() {
            guards.index.swap_parts(index);
        }
        if let (Some(sparse), Some(parts)) = (guards.sparse.as_mut(), self.sparse.as_mut()) {
            sparse.swap_parts(parts);
        }
        if let Some(ids) = self.ids.as_mut() {
            guards.ids.swap_parts(ids);
        }
        if let Some(metadata) = self.metadata.as_mut() {
            guards.vector_metadata.swap_copy(metadata);
        }
        if let Some(columns) = self.columns.as_mut() {
            guards.columns.swap_parts(columns);
        }
    }

    /// Bytes of id text the close-up let go of, read once the swap has run.
    fn removed_text(&self) -> usize {
        self.ids.as_ref().map_or(0, IdParts::removed_text)
    }
}

/// The parts dropped, in the tests alone, between two points a test can
/// hold open or read the guards the dropping thread holds at. A thread
/// already unwinding drops them past no point, so a hook can never panic
/// inside an unwind and abort the test run.
#[cfg(test)]
impl Drop for ClosedUp {
    fn drop(&mut self) {
        let observed = !std::thread::panicking();
        if observed {
            reach(CloseUpPoint::Dropping);
        }
        drop((
            self.index.take(),
            self.sparse.take(),
            self.ids.take(),
            self.metadata.take(),
            self.columns.take(),
        ));
        if observed {
            reach(CloseUpPoint::Dropped);
        }
    }
}

/// The mutation guard as a mutating entry point holds it, with what the
/// close-ups run under it replaced.
///
/// A close-up hands the structures its swap replaced to the holder rather
/// than dropping them under the guard. The fields drop in declaration order,
/// so the guard is let go first and those structures are dropped after it,
/// on the thread that held the guard, before the entry point returns and
/// with no guard of the collection held. A second writer therefore takes
/// the guard while the first one's old structures are freed, and a caller
/// sees the memory given back once its call returns. A helper borrows the
/// holder from its entry point, as it borrowed the guard. The list is made
/// by the first close-up, so a call that closes nothing up drops the guard
/// and finds no list.
pub(super) struct MutationGuard<'a> {
    _writers: Locked<'a, ()>,
    retired: RefCell<Option<Vec<ClosedUp>>>,
}

impl MutationGuard<'_> {
    /// Keep what a close-up's swap replaced, to drop once the guard is let
    /// go.
    fn retire(&self, parts: ClosedUp) {
        self.retired
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(parts);
    }
}

/// One record past the door and not yet installed.
///
/// What `admit` produces and `install` consumes: the record's content, the
/// internal id issued for it and the level drawn for it, both already in
/// the record the sink was handed. A replay builds one from a recorded
/// insert, at the id and the level the record names, and installs it the
/// same way.
pub(super) struct Admitted {
    pub(super) id: String,
    pub(super) vector: Vec<f32>,
    pub(super) sparse: Option<SparseVector>,
    pub(super) metadata: HashMap<String, Value>,
    pub(super) internal_id: usize,
    pub(super) level: usize,
}

/// What the records a segment has admitted and not yet installed will take
/// once they are, which the door holds each next record of the segment to.
/// Nothing installs while a segment is admitted, so what the stores hold is
/// read once, as the segment begins.
struct Taken {
    /// The text of their ids, which the id store's arena holds once they are
    /// installed.
    text: usize,
    /// The nonzeros the sparse space's forward arena held as the segment
    /// began, being zero where the collection declares no sparse space.
    forward: usize,
    /// Their sparse nonzeros, which the forward arena holds once they are
    /// installed.
    nonzeros: usize,
    /// The room they may take in the graph.
    room: GraphRoom,
}

/// A rebuild's insertion refused, logged as `what`: a refusal for the graph's
/// room as a rebuild gives it, since it names what is full, and any other as
/// the failure of the re-insertion.
fn reinsert_refused(error: Error, what: &str) -> Error {
    let error = error.in_rebuild();
    error!(target: LOG_TARGET, operation = "rebuild_graph", error = %error, "{}", what);
    match error {
        Error::GraphFull { .. } => error,
        other => Error::ReinsertCodesFailed(other.to_string()),
    }
}

impl Collection {
    /// Warn once when the index holds materially more records than it declared
    ///
    /// `expected_size` is a capacity hint rather than a limit, so exceeding it is
    /// legal and the index keeps working. What it costs is not the reservation,
    /// which grows through the ordinary `Vec::push` path, but the graph degree.
    /// The Python factory derives the default `m` from `expected_size`, and no
    /// `add` revises it, so an index that has outgrown its declaration by a wide
    /// margin is running at a degree chosen for a smaller index. Nothing else
    /// tells a caller that.
    ///
    /// A warning rather than an error, because the index is correct and the
    /// remedy is `rebuild`, which is a full pass over every record and therefore
    /// the caller's call rather than something an `add` should do to them.
    ///
    /// Fires once per index. The flag is claimed with a compare and exchange, so
    /// two writers crossing the threshold together produce one line and not two.
    pub(super) fn warn_if_outgrown_expected_size(&self) {
        if self.overgrowth_warned.load(Ordering::Acquire) {
            return;
        }

        let threshold = self
            .expected_size()
            .saturating_mul(EXPECTED_SIZE_OVERGROWTH_FACTOR);
        let live_records = self.ids.read().unwrap().len();
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

        warn!(target: LOG_TARGET, operation = "expected_size_exceeded",
            live_records = live_records,
            expected_size = self.expected_size(),
            m = self.dense().m(),
            "Index holds more than {}x the records its expected_size declared. \
             expected_size is a hint and not a limit, so nothing is broken, but m \
             was sized for the declaration. Call rebuild(m=..., expected_size=...) \
             to build the graph again at a degree matching what the index actually \
             holds, if recall matters. This warning fires once.",
            EXPECTED_SIZE_OVERGROWTH_FACTOR
        );
    }

    /// Put one record into the dense index, name it in the record set and
    /// write its metadata and columns, in the two phases the graph splits an
    /// insertion into.
    ///
    /// ```text
    /// index.read()          prepare: quantize where the graph holds codes,
    ///                       descend from the level the caller drew, choose
    ///                       the neighbour lists
    ///   drop
    /// ids.write()
    /// index.write()
    /// vector_metadata.write()
    /// columns.write()       insert: name the record at the next position,
    ///                       append the node, install its lists, update the
    ///                       reverse links, file the node at the position,
    ///                       and write the metadata and the columns there
    /// ```
    ///
    /// This is the only caller that takes the index lock twice for one
    /// operation, and it is the only one that needs to. The three rebuild paths
    /// each fill a local graph nobody else can reach and swap it in under one
    /// write guard, so they insert with no lock held at all.
    ///
    /// # Why the id store is written here
    ///
    /// The dense index keeps its own live set and the collection keeps one
    /// in the id store, and the two are the same set. Writing the store in
    /// their own block above and the live bit here left the invariant false
    /// for the whole of phase one, which is where an insertion spends its
    /// time, so a reader taking no mutation guard saw the collection holding a
    /// record the index did not. `get_stats` is such a reader, and on a build
    /// with debug assertions its check of the two sets fired. The two writes
    /// are one write, so they happen under one acquisition, which is what
    /// `remove_under_guards` already does with the same pair.
    ///
    /// It costs nothing on the search path. A search takes the id store and
    /// the index read guards in that order and holds both for its whole
    /// traversal, so a writer waiting for the index write guard was
    /// already waiting for exactly those searches. What is new is that the
    /// store is held across phase two, being the fixed `m * 2 * m` memory
    /// operations the index guard was already held across, and not across
    /// phase one.
    ///
    /// # Why the split
    ///
    /// Phase one is where an insertion spends its time. It runs the traversal at
    /// `ef_construction` per layer, which is the same work a search does several
    /// times over, and it writes nothing. Phase two is a fixed number of memory
    /// operations, roughly `m * 2 * m`. Holding the write guard across both
    /// would block every concurrent search for the whole of an insertion, which
    /// at 50,000 records of dimension 1,536 is on the order of a millisecond
    /// against a search mean near one. Holding it across phase two alone is what
    /// keeps the concurrent search figure this project measures.
    ///
    /// # Why it is sound
    ///
    /// Nothing can change the graph between the two phases. `writers` is taken
    /// by every mutating Python entry point before any guard, and this runs
    /// inside it, so the only thread that could append a node, replace the graph
    /// or rebuild it is this one. Searches run in the gap and do not mutate.
    /// What the plan carries is owned outright, so no borrow of the read guard
    /// survives it, and `VectorGraph::install` asserts the node count the plan
    /// was made against rather than taking that argument on trust.
    ///
    /// The record is named in the id store first, and the store forgets the
    /// position again if the index refuses the record, so an insertion the
    /// index refuses leaves every structure as it was and takes no position.
    /// The metadata and the columns are written only once the index has
    /// accepted the record, in the same block, so a reader sees the four
    /// structures change together.
    ///
    /// # The lock order
    ///
    /// Phase one takes the index read guard with no other guard held. Phase two
    /// takes the id store, the index, the metadata and the columns in that
    /// order, which is the order declared on `Collection` and the order
    /// `removal_guards` takes the same four in.
    ///
    /// # The level
    ///
    /// `level` is what the caller drew through [`Collection::draw_level`]
    /// after the internal id was issued and before anything was written for
    /// the record. Phase one plans at it and draws nothing, so the level is
    /// in the caller's hand before the insertion begins, which is what lets a
    /// record of the insertion carry it and a replay of that record install
    /// at it. Nothing between the draw and the plan reaches the stream, so
    /// the graph is the one a draw inside phase one built.
    ///
    /// Returns the codes the graph installed where it holds codes, so the
    /// caller can key them by external id for the paths that reach a record
    /// by name. They are read under the write guard the install took, so a
    /// search sees the node and its codes together or neither.
    ///
    /// # The position
    ///
    /// A record takes the next position, being the id store's extent, which
    /// every per-record structure files it at. Positions follow internal id
    /// order, since the record's id is the largest issued. Returns the
    /// position, which the caller files the record's sparse half at.
    fn insert_one(
        &self,
        external_id: &str,
        vector: &[f32],
        metadata: HashMap<String, Value>,
        internal_id: usize,
        level: usize,
    ) -> Result<(Position, Option<Vec<u8>>), Error> {
        let id = RecordId::from_slot(internal_id);
        let prepared = {
            let index = self.dense().index.read().unwrap();
            index.prepare_at_level(id, vector, level)?
        };
        let (at, codes, due) = {
            let mut ids = self.ids.write().unwrap();
            let mut index = self.dense().index.write().unwrap();
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            let mut columns = self.columns.write().unwrap();
            // The store first, and the position taken back if the graph
            // refuses the node, so a refused record leaves nothing behind.
            // Admission held the record to the store's refusals.
            let slot = ids.extent();
            let at = Position::from_slot(slot);
            ids.insert_admitted(slot, external_id);
            if let Err(refused) = index.insert(at, id, vector, prepared) {
                ids.retract(slot);
                return Err(refused);
            }
            // The metadata, and the declared fields of it in their columns, in
            // one block and one order, because the two are the same write seen
            // from two directions and a search that saw one without the other
            // would return a record the filter had not admitted.
            columns.write(slot, &metadata);
            debug_assert!(
                columns.agrees_with(slot, &metadata),
                "every declared field's column holds what the record's metadata holds"
            );
            vector_metadata.insert(slot, metadata);
            (
                at,
                index
                    .node_of(at)
                    .and_then(|node| index.graph().codes_at(node))
                    .map(<[u8]>::to_vec),
                index.due_for_timing(),
            )
        };
        // Each time the live count doubles, the units a search is priced
        // with are timed again, under a read guard so a concurrent search
        // is not held up, and adopted under a write guard taken for the
        // assignment alone.
        if due {
            let timed = {
                let index = self.dense().index.read().unwrap();
                DenseIndex::time_graph(index.graph())
            };
            self.dense().index.write().unwrap().set_units(timed);
        }
        Ok((at, codes))
    }

    /// Put one record into the sparse space at its position, the one the
    /// dense insertion filed it at, with its internal id beside it for the
    /// space's slot count, under the space's own guard, taken alone.
    fn insert_sparse(
        &self,
        at: Position,
        internal_id: usize,
        vector: &SparseVector,
    ) -> Result<(), Error> {
        let space = self.sparse().ok_or(Error::NoSparseSpace)?;
        let mut index = space.index.write().unwrap();
        index
            .insert(
                at,
                RecordId::from_slot(internal_id),
                vector.as_ref(),
                Prepared::none(),
            )
            .map_err(|err| self.sparse_refusal_named(err))
    }

    /// Take back the id `issue_id` just issued, because the record it
    /// was issued for was refused by the sink. Sound under `writers`, since
    /// nothing else issues an id while it is held, which the assertion
    /// states.
    fn retract_id(&self, issued: usize) {
        let mut counter = self.id_counter.lock().unwrap();
        debug_assert_eq!(
            *counter, issued,
            "no id was issued after the one taken back"
        );
        *counter = issued - 1;
    }

    /// How many internal ids the counter can issue before it passes the
    /// last one the id store holds. Every caller holds `writers`, so nothing
    /// issues an id between the read and what the caller does with it.
    fn ids_left(&self) -> usize {
        MAX_INTERNAL_ID.saturating_sub(*self.id_counter.lock().unwrap())
    }

    /// Issue the next internal id, refusing where the counter has issued the
    /// last one the id store holds: the ceiling's check and the issue in one
    /// acquisition of the counter. Every caller holds `writers`, so nothing
    /// issues an id between the two.
    fn issue_id(&self) -> Result<usize, Error> {
        let mut counter = self.id_counter.lock().unwrap();
        if *counter >= MAX_INTERNAL_ID {
            return Err(Error::InternalIdsExhausted {
                last: MAX_INTERNAL_ID,
                left: 0,
                needed: 1,
            });
        }
        *counter += 1;
        Ok(*counter)
    }

    /// Refuse a record the id store could not name or the dense space could
    /// not hold, being an id past the store's ceiling on its length, one
    /// that would take the arena past its ceiling with `arena` bytes in it
    /// and the records admitted before it named, or a vector of another
    /// width. Run where a record is admitted, so a refusal moves nothing.
    fn refuse_unplaceable(&self, id: &str, vector: &[f32], arena: usize) -> Result<(), Error> {
        refuse_unnamable(id, arena)?;
        if vector.len() != self.dense().dim {
            return Err(Error::VectorDimension {
                expected: self.dense().dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    /// What the records an overwrite replaces hold, as the records with a raw
    /// vector alone, with codes alone and with both, over every record of the
    /// batch that names a held record. For the line that reports the
    /// overwrite, and read only where that line is written.
    fn replaced_storage(&self, records: &[ParsedRecord]) -> (usize, usize, usize) {
        let ids = self.ids.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let pq_codes = self.dense().pq_codes.read().unwrap();
        let raws = index.raw_vectors();
        let (mut has_raw, mut has_pq, mut has_both) = (0, 0, 0);
        for record in records {
            if let Some(slot) = ids.slot_of(&record.id) {
                match (raws.contains(slot), pq_codes.contains_key(&record.id)) {
                    (true, true) => has_both += 1,
                    (true, false) => has_raw += 1,
                    (false, true) => has_pq += 1,
                    (false, false) => {} // Shouldn't happen, but handle gracefully
                }
            }
        }
        (has_raw, has_pq, has_both)
    }

    /// Issue the next internal id only if it is `expected`, which is what a
    /// replay does with the id a record names. The counter does not move
    /// when it is not.
    pub(super) fn issue_expected_id(&self, expected: usize) -> Result<usize, Error> {
        let mut counter = self.id_counter.lock().unwrap();
        if *counter + 1 != expected {
            return Err(Error::JournalReplayMismatch {
                detail: format!(
                    "the record names internal id {} and the collection would issue {}",
                    expected,
                    *counter + 1
                ),
            });
        }
        *counter += 1;
        Ok(*counter)
    }

    /// The ids of `requested` the index holds, each once, in the order first
    /// named. What a removal's record carries, since a record names what
    /// was removed and never a filter or a request.
    pub(super) fn held_ids(&self, requested: &[String]) -> Vec<String> {
        let ids = self.ids.read().unwrap();
        let mut seen: HashSet<&str> = HashSet::with_capacity(requested.len());
        requested
            .iter()
            .filter(|id| ids.contains_name(id.as_str()) && seen.insert(id.as_str()))
            .cloned()
            .collect()
    }

    /// Hand a removal's record to the sink and commit it, where a sink is
    /// attached and the removal names anything. Nothing is handed over for
    /// a removal of nothing, so a call naming only absent ids costs the
    /// sink nothing. The record is encoded from the ids as they are held,
    /// with none of them cloned.
    fn record_removal(&self, held: &[String]) -> Result<(), Error> {
        if held.is_empty() {
            return Ok(());
        }
        self.record_encoded(OperationKind::Remove, |out| {
            Operation::encode_remove(held, out)
        })?;
        self.commit_records()
    }

    /// Draw the level the next record's graph node takes, advancing the
    /// dense graph's level stream by one draw.
    ///
    /// Drawn after the internal id is issued and before anything is written
    /// for the record, rather than inside the graph's first phase, so that
    /// the level is in hand before the insertion begins. Under `writers`, so
    /// the stream advances once per record in insertion order, which is what
    /// makes two builds of the same records the same graph. Takes the index
    /// read guard alone and releases it, since the generator is a leaf of
    /// the graph and not part of a plan.
    pub(super) fn draw_level(&self) -> usize {
        self.dense().index.read().unwrap().draw_level()
    }

    /// Every write guard a removal takes, in the order `Collection` declares
    /// them.
    ///
    /// A batch removal holds one set of these for the whole batch rather than
    /// taking and releasing five guards per id, so the guards have to be a value
    /// a helper can borrow. The order is the declared one, which matters because
    /// a search holds the id store for its whole traversal and takes the
    /// index afterwards.
    fn removal_guards(&self) -> RemovalGuards<'_> {
        RemovalGuards {
            ids: self.ids.write().unwrap(),
            index: self.dense().index.write().unwrap(),
            pq_codes: self.dense().pq_codes.write().unwrap(),
            sparse: self.sparse().map(|space| space.index.write().unwrap()),
            vector_metadata: self.vector_metadata.write().unwrap(),
            columns: self.columns.write().unwrap(),
        }
    }

    /// Take the mutation guard for an entry point whose body may close the
    /// positions up, so what the close-up replaces is dropped once the guard
    /// is let go; see [`MutationGuard`].
    pub(super) fn hold_writers(&self) -> MutationGuard<'_> {
        MutationGuard {
            _writers: self.writers.lock().unwrap(),
            retired: RefCell::new(None),
        }
    }

    /// Close up the positions, under every write guard a removal takes: the
    /// one routine that moves every per-record structure so the records held
    /// sit at the positions below their count, in their order.
    ///
    /// `old_of_new` lists the positions the id store holds, in increasing
    /// order, and `new_of_old` maps each to its rank and each hole to the
    /// empty marker. Both are transient, four bytes a position. Where a
    /// replacement graph is handed in, built off to the side with the record
    /// at old position `old_of_new[k]` inserted at `k`, the index swaps it in
    /// and `node_at` becomes the identity over the records, which is what a
    /// compaction, a rebuild and a training transition do; the graph it
    /// replaces is handed back for the caller to drop with no guard held.
    /// Where none is, the index moves its own numbering and the graph keeps
    /// every node. Then the sparse space, which drops the postings of the
    /// records it holds dead, the id store, the metadata and the columns move
    /// their entries the same way, each giving back what it holds past the
    /// records or the reservation for `expected_size`, and the id store
    /// rewrites its text without the removed ids. With no hole there is
    /// nothing to move. Returns the graph replaced, and the bytes of id text
    /// the id store let go of.
    ///
    /// The close-up a collection runs is [`Collection::close_up`], which
    /// builds the same structures off to the side and swaps them in. This
    /// is the pass made in place under every write guard, kept for the tests
    /// that hold the two to the same result.
    #[cfg(test)]
    fn close_up_under_guards(
        guards: &mut RemovalGuards<'_>,
        replacement: Option<(VectorGraph, Option<(f64, f64)>)>,
        expected_size: usize,
    ) -> (Option<VectorGraph>, usize) {
        let extent = guards.ids.extent();
        let old_of_new: Vec<u32> = guards.ids.slots().map(|at| at as u32).collect();
        let holes = old_of_new.len() < extent;
        let replaced = match replacement {
            Some((graph, timed)) => Some(guards.index.replace_graph(graph, timed)),
            None => {
                if holes {
                    guards.index.close_up(&old_of_new);
                }
                None
            }
        };
        let mut reclaimed_text = 0;
        if holes {
            let mut new_of_old = vec![Position::NONE; extent];
            for (k, &old) in old_of_new.iter().enumerate() {
                new_of_old[old as usize] = k as u32;
            }
            if let Some(sparse) = guards.sparse.as_mut() {
                sparse.close_up(&old_of_new, &new_of_old);
            }
            reclaimed_text = guards.ids.close_up(&old_of_new, &new_of_old, expected_size);
            guards.vector_metadata.close_up(&old_of_new, expected_size);
            guards.columns.close_up(&old_of_new, expected_size);
        }
        (replaced, reclaimed_text)
    }

    /// Both close-ups from the state the collection holds, for the test that
    /// holds them to one result, under every write guard for the whole of it.
    ///
    /// The close-up built off to the side runs first: its parts are built
    /// from the structures as the guards hold them and swapped in, and every
    /// structure's layout is read. The parts then put every structure back
    /// as it was, the metadata moving each block back into its old entry and
    /// every other part traded again, since the parts hold exactly what the
    /// structures held. The close-up made in place then runs from that
    /// state, with the replacement graph the parts held, where one was
    /// given, and the layouts are read again. Both take the units `timed`
    /// gives. Returns the two layouts, aside then in place, and the bytes of
    /// id text each let go of. The collection is left as the close-up in
    /// place leaves it.
    #[cfg(test)]
    pub(super) fn close_up_both_ways(
        &self,
        replacement: Option<VectorGraph>,
        timed: Option<(f64, f64)>,
    ) -> ((Layouts, usize), (Layouts, usize)) {
        let expected_size = self.expected_size();
        let mut guards = self.removal_guards();
        let mut parts = ClosedUp::build(
            &guards.ids,
            &guards.index,
            guards.sparse.as_deref(),
            &guards.vector_metadata,
            &guards.columns,
            replacement.map(|graph| (graph, timed)),
            expected_size,
        );
        parts.swap_into(&mut guards);
        let aside = (layouts(&guards), parts.removed_text());
        parts.swap_back(&mut guards);
        let replacement = match parts.index.take() {
            Some(DenseParts::Graph { graph, .. }) => Some((*graph, timed)),
            _ => None,
        };
        let (old, removed_text) =
            Self::close_up_under_guards(&mut guards, replacement, expected_size);
        let in_place = (layouts(&guards), removed_text);
        drop(guards);
        drop(old);
        (aside, in_place)
    }

    /// The close-up and the routine that copied every part, from the state
    /// the collection holds, for the test that holds them to one result,
    /// under every write guard for the whole of it.
    ///
    /// The close-up runs first: its parts are built from the structures as
    /// the guards hold them and swapped in, every structure's layout is
    /// read, and the parts put every structure back as it was. The routine
    /// that copied every part then runs from that state, built and swapped
    /// in the same way, with the replacement graph the parts held, where one
    /// was given, and the layouts are read again. Returns the two layouts,
    /// the close-up's then the copying routine's, with the bytes of id text
    /// each let go of. The collection is left as the copying routine leaves
    /// it.
    ///
    /// Once the close-up's swap has run, each metadata entry holds the very
    /// block that the entry it moved from held, so no block was copied: the
    /// positions held are read from the id store's entries, as the copying
    /// routine reads them, and each block's place in memory before the
    /// close-up is held to its place after it.
    #[cfg(test)]
    pub(super) fn close_up_both_copies(
        &self,
        replacement: Option<VectorGraph>,
        timed: Option<(f64, f64)>,
    ) -> ((Layouts, usize), (Layouts, usize)) {
        let expected_size = self.expected_size();
        let mut guards = self.removal_guards();
        let blocks = guards.vector_metadata.blocks();
        let held: Vec<usize> = guards.ids.slots().collect();
        let mut parts = ClosedUp::build(
            &guards.ids,
            &guards.index,
            guards.sparse.as_deref(),
            &guards.vector_metadata,
            &guards.columns,
            replacement.map(|graph| (graph, timed)),
            expected_size,
        );
        parts.swap_into(&mut guards);
        let closed = (layouts(&guards), parts.removed_text());
        let moved = guards.vector_metadata.blocks();
        if parts.metadata.is_some() {
            let expected: Vec<Option<usize>> = held
                .iter()
                .take(moved.len())
                .map(|&at| blocks.get(at).copied().flatten())
                .collect();
            assert!(
                moved == expected,
                "each new metadata entry holds the block its old entry held"
            );
        } else {
            assert!(moved == blocks, "a close-up with no hole moves no block");
        }
        parts.swap_back(&mut guards);
        assert!(
            guards.vector_metadata.blocks() == blocks,
            "the blocks move back to their old entries"
        );
        let replacement = match parts.index.take() {
            Some(DenseParts::Graph { graph, .. }) => Some((*graph, timed)),
            _ => None,
        };
        let mut copied = ClosedUpCopied::build(
            &guards.ids,
            &guards.index,
            guards.sparse.as_deref(),
            &guards.vector_metadata,
            &guards.columns,
            replacement,
            expected_size,
        );
        copied.swap_into(&mut guards);
        let copies = (layouts(&guards), copied.removed_text());
        drop(guards);
        drop(parts);
        drop(copied);
        (closed, copies)
    }

    /// Every structure a close-up moves, each as its parts with their
    /// capacities, read under every write guard a removal takes.
    #[cfg(test)]
    pub(super) fn structure_layouts(&self) -> Layouts {
        layouts(&self.removal_guards())
    }

    /// Close the positions up, so the records held sit at the positions below
    /// their count in their order: the one routine a removal batch past the
    /// bound, a compaction, a rebuild at a new degree and both training
    /// rebuilds run. Returns the bytes of id text the id store let go of.
    ///
    /// `writers` is the caller's hold on the mutation guard, borrowed so the
    /// two phases are seen to run under one hold of it. Nothing else writes
    /// while it is held, so nothing the build reads can change before the
    /// swap.
    ///
    /// The build takes the read guards in the declared order, being the id
    /// store, the dense index, the sparse index, the metadata and the
    /// columns, and makes every structure's part off to the side with them
    /// held; see [`ClosedUp::build`]. A search shares those guards, so it
    /// runs throughout on the structures as they are. Where `replacement`
    /// is given, built off to the side with the record at the `k`th position
    /// held filed at `k`, the dense index's part is that graph; where it is
    /// not, the graph keeps every node.
    ///
    /// The swap takes every write guard a removal takes, in the declared
    /// order, trades every part for its structure's own, and lets the
    /// guards go. A search therefore sees every structure as it was or every
    /// structure closed up and never a mix, and waits for the trade alone.
    /// What the structures held, the graph a replacement replaced among it,
    /// is handed to `writers`, which drops it once the mutation guard is let
    /// go, so the next writer does not wait for it; see [`MutationGuard`].
    /// Dropping a graph forks to rayon, and a fork under a write guard
    /// deadlocks against a batch search holding the pool; see
    /// [`Collection::replace_dense_graph`].
    fn close_up(
        &self,
        writers: &MutationGuard<'_>,
        replacement: Option<(VectorGraph, Option<(f64, f64)>)>,
        expected_size: usize,
    ) -> usize {
        let mut parts = self.build_close_up(replacement, expected_size);
        #[cfg(test)]
        reach(CloseUpPoint::Built);
        self.swap_close_up(&mut parts);
        #[cfg(test)]
        reach(CloseUpPoint::Swapped);
        let removed_text = parts.removed_text();
        writers.retire(parts);
        removed_text
    }

    /// The build phase of [`Collection::close_up`], under read guards
    /// taken in the declared order and let go on return.
    fn build_close_up(
        &self,
        replacement: Option<(VectorGraph, Option<(f64, f64)>)>,
        expected_size: usize,
    ) -> ClosedUp {
        let ids = self.ids.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let sparse = self.sparse().map(|space| space.index.read().unwrap());
        let metadata = self.vector_metadata.read().unwrap();
        let columns = self.columns.read().unwrap();
        #[cfg(test)]
        reach(CloseUpPoint::Building);
        ClosedUp::build(
            &ids,
            &index,
            sparse.as_deref(),
            &metadata,
            &columns,
            replacement,
            expected_size,
        )
    }

    /// The swap phase of [`Collection::close_up`], under every write guard
    /// a removal takes, taken in the declared order and let go on return.
    fn swap_close_up(&self, parts: &mut ClosedUp) {
        let mut guards = self.removal_guards();
        #[cfg(test)]
        reach(CloseUpPoint::Swapping);
        parts.swap_into(&mut guards);
    }

    /// Install a replacement graph built from the records held in position
    /// order, closing the positions up, and drop the old graph once every
    /// guard is let go.
    ///
    /// The paths that replace the whole backend, `compact`, `rebuild`, both
    /// training transitions and the persistence rebuilds, all used to write
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
    /// The close-up builds the replacement's numbering and every other
    /// structure off to the side and trades them in under the guards, so the
    /// swap is a set of moves and the old graph goes out with the parts it
    /// hands to `writers`, which drops them once the mutation guard is let
    /// go. The replacement's unit cost is timed before any guard is taken,
    /// for the same reason. `writers` is the caller's. Returns the bytes of
    /// id text the close-up let go of.
    pub(super) fn replace_dense_graph(
        &self,
        writers: &MutationGuard<'_>,
        graph: VectorGraph,
    ) -> usize {
        let timed = DenseIndex::time_graph(&graph);
        let reclaimed_text = self.close_up(writers, Some((graph, timed)), self.expected_size());
        debug_assert!(
            self.positions_agree(),
            "the close-up leaves every structure at the positions the records hold"
        );
        reclaimed_text
    }

    /// The close-up on removal, run once a removal batch is done and has let
    /// its guards go, under the `writers` the caller holds. Returns whether
    /// it ran.
    ///
    /// `held_and_holes` is what the batch read under its own guards as its
    /// last step. Nothing writes the id store between the batch and this
    /// test, since every write holds `writers`, so the figures are the
    /// store's still.
    ///
    /// Where the holes exceed both [`HOLE_ALLOWANCE`] and the records held,
    /// every per-record structure is closed up by the routine a compaction
    /// runs, the sparse space dropping the postings of the records it holds
    /// dead, and the graph keeps every node. Otherwise nothing moves. The
    /// build reads every position held, every bucket and every live posting
    /// once, and it runs after at least as many removals since the last
    /// close-up as the allowance or the records held, whichever is more.
    /// Between the batch and the swap a reader sees the batch's holes, which
    /// may pass the bound; see [`Collection::positions_agree_any_holes`].
    ///
    /// A batch is what one removal's record names, so a replay of the
    /// record closes up at the same point as the removal that wrote it.
    fn close_up_after_removal(
        &self,
        writers: &MutationGuard<'_>,
        (held, holes): (usize, usize),
        expected_size: usize,
    ) -> bool {
        if holes <= HOLE_ALLOWANCE.max(held) {
            return false;
        }
        let start = Instant::now();
        let reclaimed_text = self.close_up(writers, None, expected_size);
        #[cfg(test)]
        CLOSE_UPS_ON_REMOVAL.with(|count| count.set(count.get() + 1));
        debug!(target: LOG_TARGET, operation = "close_up_on_removal",
            holes = holes,
            live_records = held,
            reclaimed_id_bytes = reclaimed_text,
            duration_ms = start.elapsed().as_millis(),
            "Positions closed up after a removal"
        );
        true
    }

    /// Remove one record with the guards already held.
    ///
    /// `storage_mode` is passed in rather than read, because reading it reaches
    /// the graph lock and the declared order puts the graph above every map the
    /// caller is holding.
    fn remove_under_guards(
        &self,
        guards: &mut RemovalGuards<'_>,
        id: &str,
        storage_mode: &str,
    ) -> bool {
        // The position the record held, which becomes a hole in every
        // per-record structure until the next close-up. A hole is never
        // filled again, since positions only grow between close-ups.
        let Some(slot) = guards.ids.remove_name(id) else {
            trace!(target: LOG_TARGET, operation = "remove_point_internal",
                vector_id = %id,
                "Vector not found for removal"
            );
            return false;
        };

        // Track what we're removing for logging.
        //
        // Whether the record had a raw vector is read off the storage mode
        // rather than off a map, because the raw vectors now live in the graph
        // and the graph lock cannot be taken here: every guard this function
        // holds sits above it in the order the crate takes them. The mode
        // answers the question exactly. Every mode except a trained
        // `quantized_only` one keeps a raw vector for every record, and that
        // one keeps none.
        let had_raw_vector = !(storage_mode == "quantized_active"
            && self
                .dense()
                .quantization_config
                .as_ref()
                .is_some_and(|config| config.storage_mode == StorageMode::QuantizedOnly));
        let had_pq_codes = guards.pq_codes.contains_key(id);

        // Remove from all data structures. The raw vector is not removed here:
        // removal strands the record's graph node rather than deleting it, and
        // the vector goes with the node when `compact` rebuilds the graph
        // without it. That is what removal already did to the node itself.
        guards.vector_metadata.remove(slot); // Remove metadata
                                             //
                                             // The columns are addressed by position, so the slot is cleared
                                             // rather than reclaimed, and a removed record leaves a hole in
                                             // every column exactly as it leaves a stranded node.
        guards.columns.erase(slot);
        guards.pq_codes.remove(id); // Remove PQ codes (if present)

        // The dense index strands the node and drops the record from its
        // live set, and hands back the node, which keeps the internal id the
        // log below names the record by. The sparse index unlinks the
        // record's postings where the record filled that space, asked first
        // whether it holds the record, since a record may leave the sparse
        // space empty and an index refuses to remove a position it does not
        // hold.
        let record = Position::from_slot(slot);
        let node = guards.index.strand(record);
        debug_assert!(
            node != Position::NONE,
            "the dense index holds every record the id store holds"
        );
        if let Some(sparse) = guards.sparse.as_mut() {
            if sparse.holds(record) {
                let _ = sparse.remove(record);
            }
        }
        debug_assert!(
            guards.columns.tracks(guards.ids.len()),
            "a column store holds one entry per live record, and a removal writes both"
        );

        // Enhanced training state cleanup for quantization
        if self.has_quantization() {
            // Remove from training IDs if present and not yet trained
            if !self.can_use_quantization() {
                let mut training_ids = self.training_ids.write().unwrap();
                let original_len = training_ids.len();
                training_ids.retain(|training_id| training_id != id);

                if training_ids.len() != original_len {
                    trace!(target: LOG_TARGET, operation = "training_cleanup",
                        vector_id = %id,
                        remaining_training_vectors = training_ids.len(),
                        "Removed vector from training set"
                    );

                    // Update threshold status if we dropped below training size
                    if let Some(config) = &self.dense().quantization_config {
                        if training_ids.len() < config.training_size {
                            self.training_threshold_reached
                                .store(false, std::sync::atomic::Ordering::Release);
                            debug!(target: LOG_TARGET, operation = "training_threshold_reset",
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

        debug!(target: LOG_TARGET, operation = "remove_point_internal",
            vector_id = %id,
            internal_id = if node == Position::NONE {
                0
            } else {
                guards.index.graph().origin_id_at(node)
            },
            had_raw_vector = had_raw_vector,
            had_pq_codes = had_pq_codes,
            storage_mode = storage_mode,
            note = "hnsw_graph_entries_remain_unreachable",
            "Vector completely removed from index (HNSW graph entries become unreachable)"
        );
        true
    }

    /// Internal remove_point method that can be called without Python bindings
    /// This is the core method that properly removes all traces of a document
    /// Enhanced internal remove_point method with comprehensive PQ support
    ///
    /// A batch of one record, so the close-up on removal runs after it where
    /// it is due, once its guards are let go.
    pub(super) fn remove_point_internal(
        &self,
        writers: &MutationGuard<'_>,
        id: String,
    ) -> Result<bool, String> {
        // Read before the guards are taken, because it reaches the graph lock and
        // the declared order puts the graph above every map held below.
        let storage_mode = self.storage_mode();
        let expected_size = self.expected_size();
        let (removed, held_and_holes) = {
            let mut guards = self.removal_guards();
            let removed = self.remove_under_guards(&mut guards, &id, &storage_mode);
            (removed, guards.held_and_holes())
        };
        let closed = self.close_up_after_removal(writers, held_and_holes, expected_size);
        debug_assert!(
            !closed || self.positions_agree(),
            "the close-up on removal leaves every structure at the positions the records hold"
        );
        Ok(removed)
    }

    /// Remove a batch of records under one set of guards.
    ///
    /// Returns the ids that were not in the index, in the order they were given.
    /// A repeated id is handled on its first occurrence and skipped afterwards,
    /// so a batch naming one id twice removes it once and reports it missing
    /// never. Reporting the second occurrence as missing would be a true
    /// statement about the index at that instant and a useless one about the
    /// request, since what the caller asked for is that the record be gone.
    ///
    /// The five guards are taken once for the batch rather than once per id,
    /// which is the whole reason this exists beside `remove_point_internal`.
    /// What that changes for a reader is that the batch is atomic against every
    /// search: none of them sees the index part way through it.
    ///
    /// The close-up on removal runs once the batch is done and its guards
    /// are let go, where it is due, and never part way through the batch.
    pub(super) fn remove_points_internal(
        &self,
        writers: &MutationGuard<'_>,
        ids: &[String],
    ) -> Vec<String> {
        let storage_mode = self.storage_mode();
        let expected_size = self.expected_size();
        let (missing, held_and_holes) = {
            let mut guards = self.removal_guards();
            let mut handled: HashSet<&str> = HashSet::with_capacity(ids.len());
            let mut missing = Vec::new();
            for id in ids {
                if !handled.insert(id.as_str()) {
                    continue;
                }
                if !self.remove_under_guards(&mut guards, id, &storage_mode) {
                    missing.push(id.clone());
                }
            }
            (missing, guards.held_and_holes())
        };
        let closed = self.close_up_after_removal(writers, held_and_holes, expected_size);
        debug_assert!(
            !closed || self.positions_agree(),
            "the close-up on removal leaves every structure at the positions the records hold"
        );
        missing
    }

    /// Remove every record whose metadata matches, and report how many.
    ///
    /// Two phases, because the matching set is read before the removal writes
    /// it. The read guards are dropped before the write guards are taken, and
    /// nothing can change the index in between because the caller holds the
    /// mutation guard.
    ///
    /// The columns answer the first phase where every field the filter names is
    /// declared. Where one field has no column they bound the candidates and
    /// the metadata decides among them, and where they bound nothing the walk
    /// answers it alone. **The walk is complete where the search's is
    /// bounded**, since a deletion has no use for a give-up point, and the
    /// columns are complete for the same reason: the bitmap holds every
    /// matching record whether there are ten of them or ten thousand.
    ///
    /// The filter arrives compiled, so nothing here can fail on it. The caller
    /// built it from the mapping it was handed and every operator name and
    /// group shape the engine cannot evaluate was rejected there, before any
    /// record was read. What can fail is the sink, which is handed the
    /// resolved ids between the two phases.
    pub(super) fn remove_where_locked(
        &self,
        writers: &MutationGuard<'_>,
        filter: &Filter,
    ) -> Result<usize, Error> {
        // The id store before `vector_metadata` before `columns`, which is
        // the declared order, because the metadata store and the bitmap hold
        // internal ids and a removal names external ones. The columns guard
        // is dropped before the metadata one is taken, since the selection
        // owns its bitmap and borrows only the filter.
        let store = self.ids.read().unwrap();
        let selection = {
            let columns = self.columns.read().unwrap();
            columns.select(filter)
        };
        let resolve = |selected: &Bitmap| {
            let mut ids = Vec::with_capacity(selected.count());
            selected.for_each(|slot| {
                if let Some(id) = store.name(slot) {
                    ids.push(id.to_string());
                }
            });
            ids
        };
        let doomed: Vec<String> = match selection {
            Selection::Exact(selected) => resolve(&selected),
            // A filter mixing a declared field with one that has no column. The
            // bound is a superset, so each candidate's metadata still decides,
            // and what the columns bought is how few candidates there are.
            Selection::Narrowed(bound, _) => {
                let vector_metadata = self.vector_metadata.read().unwrap();
                let mut ids = Vec::new();
                bound.for_each(|slot| {
                    if let Some(id) = store.name(slot) {
                        if vector_metadata
                            .get(slot)
                            .is_some_and(|fields| matches_filter(&fields, filter))
                        {
                            ids.push(id.to_string());
                        }
                    }
                });
                ids
            }
            // The store walks in increasing internal id order, so the
            // removal's record names the ids in arrival order.
            Selection::Whole(_) => {
                let vector_metadata = self.vector_metadata.read().unwrap();
                vector_metadata
                    .iter()
                    .filter(|(_, fields)| matches_filter(fields, filter))
                    .filter_map(|(slot, _)| store.name(slot).map(str::to_string))
                    .collect()
            }
        };
        drop(store);

        // The resolved ids, and never the filter, since the filter's answer
        // depends on the state it was asked against.
        self.record_removal(&doomed)?;

        let total = doomed.len();
        let missing = self.remove_points_internal(writers, &doomed);
        debug_assert!(
            missing.is_empty(),
            "every id came from the metadata store under the mutation guard, so none of them can have been absent by the time it was removed"
        );
        Ok(total - missing.len())
    }

    /// Replace one record's metadata, leaving its vector and the graph alone.
    ///
    /// Wholesale rather than a merge, which is what `add(overwrite=True)` does:
    /// that path removes the record outright, dropping the old metadata with it,
    /// and inserts the supplied metadata in its place. A merging update would
    /// mean the two ways of re-tagging a record disagreed about a key the caller
    /// left out, and no third way exists to express the other intent.
    ///
    /// `false` for an id the index does not hold, which is what `remove_point`
    /// answers for the same question, and nothing is written in that case.
    ///
    /// Existence is decided by the id store rather than by the metadata store,
    /// because the id store is the record set. Every insertion path writes a
    /// metadata entry even for a record supplied without metadata, so the two
    /// agree, and keying on the authoritative one means they cannot drift.
    pub(super) fn update_metadata_locked(
        &self,
        id: &str,
        metadata: HashMap<String, Value>,
    ) -> bool {
        let ids = self.ids.read().unwrap();
        let Some(slot) = ids.slot_of(id) else {
            trace!(target: LOG_TARGET, operation = "update_metadata",
                vector_id = %id,
                "Record not found, metadata not written"
            );
            return false;
        };
        let mut vector_metadata = self.vector_metadata.write().unwrap();
        let mut columns = self.columns.write().unwrap();
        // Wholesale here too. Every declared field is rewritten, so a key the
        // caller left out is cleared from its column rather than left holding
        // the value the record used to carry.
        columns.write(slot, &metadata);
        debug_assert!(
            columns.agrees_with(slot, &metadata),
            "every declared field's column holds what the record's metadata holds"
        );
        vector_metadata.insert(slot, metadata);
        trace!(target: LOG_TARGET, operation = "update_metadata",
            vector_id = %id,
            "Metadata replaced"
        );
        true
    }

    /// The body of `compact`, with the mutation guard held by the caller and
    /// the record already handed over.
    ///
    /// A replacement graph with no room for a record held is refused with
    /// the room's refusal as a rebuild gives it and installs nothing, so the
    /// graph and every record stay as they were. The sparse space's compaction, which runs
    /// first and is held to no room of the graph's, has run by then.
    pub(super) fn compact_locked(&self, writers: &MutationGuard<'_>) -> Result<usize, Error> {
        let start_time = Instant::now();

        // The sparse space first, where there is one, since what it has to
        // reclaim is its own whatever the graph holds. A removed record
        // leaves its span in the forward arena even after the lazy rewrite
        // has taken its postings out of every list, so the test is whether
        // any record is dead rather than whether any posting is. A close-up
        // drops the dead records' postings and keeps their spans, so the
        // test is also whether the arena holds a span no live record owns.
        // The figure `compact` reports is the graph's, as it always was.
        if let Some(space) = self.sparse() {
            let stranded = {
                let index = space.index.read().unwrap();
                index.dead_records() > 0 || index.holds_removed_spans()
            };
            if stranded {
                space.index.write().unwrap().compact();
            }
        }

        debug_assert!(
            self.positions_agree(),
            "every per-record structure agrees with the positions the records hold"
        );
        let live_count = self.ids.read().unwrap().len();
        let (nodes_before, stranded) = {
            let index = self.dense().index.read().unwrap();
            (index.graph().nb_points(), index.stranded())
        };
        debug_assert_eq!(
            stranded,
            nodes_before.saturating_sub(live_count),
            "the dense index's live set is the id store's"
        );

        if nodes_before <= live_count {
            debug!(target: LOG_TARGET, operation = "compact",
                graph_nodes = nodes_before,
                live_records = live_count,
                "No stranded nodes, compact is a no-op"
            );
            return Ok(0);
        }

        let quantized = self.is_quantized();
        let nodes_after = self.rebuild_graph(
            writers,
            self.dense().m(),
            self.expected_size(),
            self.dense().ef_construction(),
        )?;
        let reclaimed = nodes_before - nodes_after;
        info!(target: LOG_TARGET, operation = "compact_complete",
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

    /// The body of `rebuild`, with the interpreter lock already released.
    ///
    /// **`m` is the one creation parameter a caller cannot correct any other
    /// way.** It is chosen from `expected_size` at `create()` and fixed there,
    /// so an index declared for 10,000 records and given a million runs at a
    /// degree meant for the smaller one, and no search width recovers the recall
    /// that costs. This rebuilds the graph at a new degree, in place.
    ///
    /// **Everything except the graph survives untouched.** Each record is
    /// re-inserted under the internal id it already holds, which is what
    /// `compact` does, so the id store, the metadata store and every
    /// column stay correct without being rewritten. A quantized index is
    /// rebuilt from its stored codes rather than re-encoded, so the codebook is
    /// not retrained and no record's code changes; a `quantized_with_raw` index
    /// carries its raw store over node by node.
    ///
    /// `expected_size` and `ef_construction` move with it. The first selects
    /// the default `m` at `create()`, sizes the replacement graph's reservation
    /// and is what the overgrowth warning compares against. The second is one of
    /// the two remedies the neighbour selection warning names, so a caller told
    /// to raise it has to be able to.
    ///
    /// **The three are written after the rebuild has succeeded**, so an index
    /// whose records could not be rebuilt from, or whose replacement has no
    /// room for them at the new degree, still reports the configuration its
    /// graph actually has.
    ///
    /// Returns the live record count the replacement holds. The mutation
    /// guard is held by the caller and the record already handed over.
    pub(super) fn rebuild_locked(
        &self,
        writers: &MutationGuard<'_>,
        m: usize,
        expected_size: usize,
        ef_construction: usize,
    ) -> Result<usize, Error> {
        let start_time = Instant::now();

        let previous_m = self.dense().m();
        let previous_expected_size = self.expected_size();
        let previous_ef_construction = self.dense().ef_construction();
        let live_count = self.ids.read().unwrap().len();
        let nodes_before = self.dense().index.read().unwrap().graph().nb_points();

        let nodes_after = self.rebuild_graph(writers, m, expected_size, ef_construction)?;

        self.dense().m.store(m, Ordering::Release);
        self.expected_size.store(expected_size, Ordering::Release);
        self.dense()
            .ef_construction
            .store(ef_construction, Ordering::Release);

        info!(target: LOG_TARGET, operation = "rebuild_complete",
            m_before = previous_m,
            m_after = m,
            expected_size_before = previous_expected_size,
            expected_size_after = expected_size,
            ef_construction_before = previous_ef_construction,
            ef_construction_after = ef_construction,
            nodes_before = nodes_before,
            nodes_after = nodes_after,
            live_records = live_count,
            quantized = self.is_quantized(),
            duration_ms = start_time.elapsed().as_millis(),
            "Graph rebuilt"
        );

        Ok(nodes_after)
    }

    /// Build a replacement graph at one degree and install it, returning its
    /// node count.
    ///
    /// The shared body of `compact` and `rebuild`. Compaction is this at the
    /// degree the index already has, which is why the two are one function: a
    /// second copy that rebuilt by insertion would be a second place for the
    /// insertion order, the quantized branch and the raw carry-over to drift.
    ///
    /// **The three parameters are passed rather than read off the index**, so
    /// that `rebuild_locked` can write them after the rebuild has succeeded. An
    /// index whose records cannot be rebuilt from therefore still reports the
    /// configuration its graph actually has.
    ///
    /// The caller holds `writers` and lends it here.
    fn rebuild_graph(
        &self,
        writers: &MutationGuard<'_>,
        m: usize,
        expected_size: usize,
        ef_construction: usize,
    ) -> Result<usize, Error> {
        let new_hnsw = self.replacement_graph(m, expected_size, ef_construction)?;
        let nodes_after = new_hnsw.nb_points();

        // The text of every removed record, which stayed in the id store's
        // arena until now, goes with the stranded nodes: one pass over the
        // entries and one copy of the live text, which the close-up makes
        // off to the side. Nothing else reclaims it short of `clear`.
        let reclaimed_text = self.replace_dense_graph(writers, new_hnsw);
        if reclaimed_text > 0 {
            debug!(target: LOG_TARGET, operation = "rebuild_graph",
                reclaimed_id_bytes = reclaimed_text,
                "Removed records' ids reclaimed from the id store"
            );
        }

        Ok(nodes_after)
    }

    /// The replacement `rebuild_graph` installs: every record held
    /// re-inserted at one degree, in position order, under the internal id
    /// it holds, with the slack of its arenas given back. Refused, with
    /// nothing built, where a record held has no source data, and with what
    /// was built dropped where the replacement has no room for a record at
    /// the level drawn for it. The caller holds `writers`, so nothing it
    /// reads can change before the install.
    pub(super) fn replacement_graph(
        &self,
        m: usize,
        expected_size: usize,
        ef_construction: usize,
    ) -> Result<VectorGraph, Error> {
        let live_count = self.ids.read().unwrap().len();
        let quantized = self.is_quantized();
        // A scalar graph is rebuilt from its own rows, which nothing else
        // holds, under the codec the graph being replaced already carries.
        let int8 = if quantized {
            self.dense()
                .index
                .read()
                .unwrap()
                .graph()
                .int8_codec()
                .cloned()
        } else {
            None
        };

        let mut new_hnsw = if let Some(codec) = &int8 {
            VectorGraph::new_int8(
                &self.dense().metric,
                m,
                expected_size,
                MAX_LAYER,
                ef_construction,
                codec.clone(),
            )
        } else if quantized {
            let pq = self
                .dense()
                .pq
                .as_ref()
                .cloned()
                .ok_or(Error::NoQuantizer)?;
            VectorGraph::new_pq(
                &self.dense().metric,
                m,
                expected_size,
                MAX_LAYER,
                ef_construction,
                pq,
            )
        } else {
            VectorGraph::new_raw(
                &self.dense().metric,
                self.dense().dim,
                m,
                expected_size,
                MAX_LAYER,
                ef_construction,
            )
        };

        // Re-insert every live record under the internal id it already holds,
        // at its place among the records held, counting from zero, so the
        // replacement's node for each record is that place and the close-up
        // that installs it moves every per-record structure there. A record
        // whose source data is missing is collected rather than skipped,
        // because skipping it would drop it from the index silently.
        let missing: Vec<String> = {
            let ids = self.ids.read().unwrap();
            // The graph being replaced, which is where the vectors live. Taken
            // after the id store and released with it, before the close-up
            // that installs the replacement takes any guard.
            let old_index = self.dense().index.read().unwrap();
            let old_hnsw = old_index.graph();
            let origin_ids = old_hnsw.origin_ids();

            // Position order, which is internal id order, arrival order and
            // the order the store walks in, rather than the order a hash map
            // hands its entries out. Two compactions of the same index in two
            // processes otherwise wire the replacement graph differently and
            // answer the same query differently.
            let live: Vec<(&str, usize)> = ids.iter().map(|(at, id)| (id, at)).collect();

            if int8.is_some() {
                let mut batch: Vec<(&[i8], usize, Position)> = Vec::with_capacity(ids.len());
                let mut missing = Vec::new();

                for (k, &(ext_id, at)) in live.iter().enumerate() {
                    match old_index.node_of(Position::from_slot(at)).and_then(|node| {
                        old_hnsw
                            .int8_row_at(node)
                            .map(|row| (row, origin_ids[node as usize]))
                    }) {
                        Some((row, internal_id)) => {
                            batch.push((row, internal_id, Position::from_slot(k)))
                        }
                        None => missing.push(ext_id.to_string()),
                    }
                }

                if missing.is_empty() && !batch.is_empty() {
                    new_hnsw
                        .insert_batch_int8(&batch)
                        .map_err(|e| reinsert_refused(e, "Failed to re-insert scalar rows"))?;
                }

                missing
            } else if quantized {
                let pq_codes = self.dense().pq_codes.read().unwrap();
                let internal = old_index.ids_by_position();
                let mut batch: Vec<(&Vec<u8>, usize, Position)> = Vec::with_capacity(ids.len());
                let mut missing = Vec::new();

                for (k, &(ext_id, at)) in live.iter().enumerate() {
                    match (pq_codes.get(ext_id), internal.get(at)) {
                        (Some(codes), Some(internal_id)) => {
                            batch.push((codes, internal_id, Position::from_slot(k)))
                        }
                        _ => missing.push(ext_id.to_string()),
                    }
                }

                if missing.is_empty() && !batch.is_empty() {
                    new_hnsw
                        .insert_batch_pq(&batch)
                        .map_err(|e| reinsert_refused(e, "Failed to re-insert quantized codes"))?;
                }

                // The raw vectors a `quantized_with_raw` index keeps beside its
                // codes are carried over rather than rebuilt, since nothing
                // else holds them. The replacement numbers its nodes its own
                // way, so they are re-addressed node by node, each from the
                // node the record held in the graph being replaced.
                if missing.is_empty() && old_hnsw.holds_raw() {
                    if let Some(dim) = old_hnsw.raw_dim() {
                        let source: Vec<u32> = live
                            .iter()
                            .map(|&(_, at)| old_index.node_at()[at])
                            .collect();
                        new_hnsw.adopt_raw_from(old_hnsw, &source, dim).map_err(|e| {
                            error!(target: LOG_TARGET, operation = "rebuild_graph", error = %e, "Failed to carry the raw vectors over");
                            Error::AdoptRawFailed(e)
                        })?;
                    }
                }

                missing
            } else {
                let mut missing = Vec::new();

                for (k, &(ext_id, at)) in live.iter().enumerate() {
                    match old_index.node_of(Position::from_slot(at)).and_then(|node| {
                        old_hnsw
                            .raw_vector_at(node)
                            .map(|vector| (vector, origin_ids[node as usize]))
                    }) {
                        Some((vector, internal_id)) => new_hnsw
                            .try_insert(vector, internal_id, Position::from_slot(k))
                            .map_err(|e| reinsert_refused(e, "Failed to re-insert a vector"))?,
                        None => missing.push(ext_id.to_string()),
                    }
                }

                missing
            }
        };

        if !missing.is_empty() {
            error!(target: LOG_TARGET, operation = "rebuild_graph",
                missing_records = missing.len(),
                live_records = live_count,
                quantized = quantized,
                "Refusing to rebuild, some live records have no source data to rebuild from"
            );
            return Err(Error::CompactRefused {
                missing: missing.len(),
                live: live_count,
                what: if int8.is_some() {
                    "scalar codes"
                } else if quantized {
                    "quantized codes"
                } else {
                    "vector"
                },
            });
        }

        // The replacement was built by insertion, so its arenas carry the same
        // geometric slack the graph being replaced carried. Compaction exists to
        // return memory, and returning it while still holding the graph outside
        // the write guard costs one copy of the live bytes against a rebuild that
        // has just re-inserted every record.
        new_hnsw.shrink_to_fit();
        Ok(new_hnsw)
    }

    /// The insertion phase of `add`, run with the interpreter lock released
    ///
    /// Everything here operates on `ParsedRecord`s, which hold no Python
    /// object, no `Py<T>` and no borrow of anything Python owns. The caller
    /// holds the mutation guard.
    ///
    /// # The two passes
    ///
    /// A batch goes through the door first and is installed second. The
    /// first pass, `admit`, runs every check that can refuse a record,
    /// encodes the record and holds it to the sink's ceiling, issues its
    /// internal id, draws its level and hands the record over, and the sink
    /// is then committed once for the pass. The second pass, `install`,
    /// writes the metadata and the columns and puts the vector into the
    /// graph at the level the first pass drew. So every record a search can
    /// see was handed over and committed first, a record refused at the door
    /// has moved nothing, and a torn batch is a prefix of the batch, which
    /// is what the crashed process had installed.
    ///
    /// # Segments
    ///
    /// The batch is cut where an install may replace the graph under the
    /// records after it, which is training. Training rebuilds the graph
    /// with a fresh level stream and the records after it draw from that
    /// stream, so their levels cannot be drawn ahead of it. A segment ends
    /// at the record that fills the training set, training fires as that
    /// record is installed, and the next segment draws from the rebuilt
    /// graph; see `segment_limit`. On an index that is not collecting the
    /// whole batch is one segment. The levels are therefore the ones a
    /// build without a sink draws, record for record, and the graph is the
    /// same graph.
    ///
    /// The fault a failed commit left is read at the start of each segment,
    /// and while it stands the rest of the batch is refused with its words,
    /// issuing no id and drawing no level.
    ///
    /// # Replacing a held record
    ///
    /// A batch with `overwrite` that names a record the collection holds
    /// takes each of its segments through admission whole before anything
    /// but its interned terms is handed over. A held id a record names
    /// counts as free there, so a record refused at admission keeps the
    /// record it names. The segment is then handed over as one call: the
    /// removal record naming, once each, the held ids its admitted records
    /// replace, then their inserts, then the commit. The terms come first
    /// because a term is handed over as the dictionary takes it, and a
    /// removal never touches the dictionary, so a replay that applies them
    /// first reaches the same state. Only after the commit are the replaced
    /// records removed, as one batch, and the new ones installed, so a
    /// failed commit or a crash before it leaves every replaced record
    /// held. An append that fails after admission withdraws the segment and
    /// leaves the fault, as a failed commit does, since the ids and levels
    /// admission issued stay issued. The order is the same with no sink, so
    /// a journaled and an unjournaled collection build the same graph. Every
    /// other batch hands each record over as admission passes it.
    ///
    /// The complete set of functions reachable from here, verified by reading
    /// each one rather than by inference:
    ///
    /// - `remove_points_internal`, and through it `get_storage_mode`,
    ///   `has_quantization`, `can_use_quantization` and the close-up on
    ///   removal
    /// - `admit`, and through it `count_record_terms`, the text layer's
    ///   dictionary, `Operation::encode_insert`, `issue_id`,
    ///   `draw_level` and the sink's `append`
    /// - `append_segment`, and through it the sink's `append` and
    ///   `withdraw`
    /// - `install`, and the three paths below it, `add_raw_vector`,
    ///   `add_with_id_collection` and `add_quantized_vector`
    /// - `is_quantized`, `segment_limit`, `ids_left`
    /// - `maybe_trigger_training`, `fit_training`, `record_training`,
    ///   `install_training`, `rebuild_with_quantization_locked` and
    ///   `rebuild_with_int8_locked`
    /// - `PQ::is_trained`, `quantize`, `quantize_batch`, `fit` and `install`,
    ///   plus the k-means below `fit`, and `Int8Codec::fit`
    /// - `VectorGraph::draw_level`, `plan_at_level`, `install`, `insert`,
    ///   `insert_batch_pq`, `new_pq` and `nb_points`, and the graph structure
    ///   in `graph::mutable` below them
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
    /// A panic in here is safe too. `Python::detach` restores the
    /// interpreter lock from a `Drop` guard, so an unwind reacquires it before it
    /// reaches PyO3's boundary.
    pub(super) fn insert_parsed_records(
        &self,
        writers: &MutationGuard<'_>,
        parsed_data: Vec<ParsedRecord>,
        overwrite: bool,
    ) -> (Vec<String>, Vec<InsertError>) {
        let mut inserted_ids: Vec<String> = Vec::with_capacity(parsed_data.len());
        let mut errors: Vec<InsertError> = Vec::new();

        // Whether a sink is attached is read once for the batch. Attaching
        // one takes the mutation guard, so it cannot change under this call.
        // A sink that takes nothing refuses every record of the batch, an
        // overwrite's removal included, before anything is admitted.
        let (attached, read_only) = self.sink_state();
        if let Some(refusing) = read_only {
            for record in parsed_data {
                let err = refusing.error();
                errors.push(InsertError::Vector { id: record.id, err });
            }
            return (inserted_ids, errors);
        }
        let mut buffer: Vec<u8> = Vec::new();

        // Whether the batch replaces a record the collection holds, and how
        // many distinct held records it names. Such a batch goes in segment
        // by segment, each admitted whole before anything but its interned
        // terms is recorded; see `# Replacing a held record`.
        let held_named = if overwrite {
            let ids = self.ids.read().unwrap();
            let mut named: HashSet<&str> = HashSet::new();
            parsed_data
                .iter()
                .filter(|record| ids.contains_name(&record.id) && named.insert(&record.id))
                .count()
        } else {
            0
        };
        let replaces = held_named > 0;
        if replaces {
            // Every record of the batch takes a fresh internal id, the ones
            // replacing a held record included, so a batch needing more ids
            // than the collection has left is refused before anything is
            // admitted. Admitted as far as the ids went, it would remove
            // records it could not put back.
            let left = self.ids_left();
            if parsed_data.len() > left {
                for record in &parsed_data {
                    errors.push(InsertError::Vector {
                        id: record.id.clone(),
                        err: Error::InternalIdsExhausted {
                            last: MAX_INTERNAL_ID,
                            left,
                            needed: parsed_data.len(),
                        },
                    });
                }
                return (inserted_ids, errors);
            }

            // What the replaced records hold, for this line alone, so it is
            // read only where the line is written.
            if tracing::enabled!(target: LOG_TARGET, tracing::Level::INFO) {
                let storage_analysis = self.replaced_storage(&parsed_data);
                info!(target: LOG_TARGET, operation = "overwrite_preparation",
                    documents_to_remove = held_named,
                    storage_analysis = format!(
                        "raw_only: {}, pq_only: {}, both: {}",
                        storage_analysis.0, storage_analysis.1, storage_analysis.2
                    ),
                    "Removing existing documents for overwrite"
                );
            }
        }

        debug!(target: LOG_TARGET, operation = "add_vectors_insertion_phase",
            current_state = self.storage_mode(),
            "Starting insertion phase"
        );

        let mut seen: HashSet<String> = HashSet::with_capacity(parsed_data.len());
        let mut pending = parsed_data.into_iter().peekable();
        // Five of the points a crash test aborts an `add` at, each named
        // where the step it follows finishes, and a sixth where a replacing
        // segment's removal record is appended. The last is inside the sink,
        // which is what writes half a record. All of them are compiled away
        // when `debug_assertions` are off; see `zeusdb_vector_core::kill_at`.
        let mut handed_over = 0usize;
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::AddBeforeFirstAppend);
        while pending.peek().is_some() {
            // A call made while the fault a failed commit left stands is
            // refused whole, with the fault's words for every record, and
            // issues no id and draws no level.
            if let Some(fault) = self.sink_fault() {
                for record in pending {
                    errors.push(InsertError::Vector {
                        id: record.id,
                        err: fault.refusal(),
                    });
                }
                return (inserted_ids, errors);
            }
            let limit = self.segment_limit();
            let mut segment: Vec<Result<Admitted, InsertError>> = Vec::new();
            let mut admitted = 0usize;
            // What the records this segment admits take once they are
            // installed, being their ids' text, their nonzeros and their room
            // in the graph, which each next record is held to.
            let mut taken = self.taken_by_segment();
            // The held ids this segment's admitted records replace, each once,
            // since a batch admits an id once.
            let mut replaced: Vec<String> = Vec::new();
            while let Some(record) = pending.next_if(|_| limit.is_none_or(|limit| admitted < limit))
            {
                let outcome = self
                    .admit(
                        record,
                        &mut seen,
                        attached,
                        &mut buffer,
                        &mut taken,
                        overwrite,
                        replaces,
                    )
                    .map(|(admitted, held)| {
                        if held {
                            replaced.push(admitted.id.clone());
                        }
                        admitted
                    });
                if outcome.is_ok() {
                    admitted += 1;
                    if !replaces {
                        handed_over += 1;
                        if handed_over == 2 {
                            zeusdb_vector_core::kill_at(
                                zeusdb_vector_core::KillPoint::AddAfterSecondAppend,
                            );
                        }
                    }
                }
                segment.push(outcome);
            }

            // A replacing segment is recorded once its admission is over. An
            // append that fails withdraws the segment and leaves the fault,
            // and the segment and the rest of the batch are refused with it.
            let recorded = if replaces && admitted > 0 && attached {
                self.append_segment(&replaced, &segment, &mut buffer, &mut handed_over)
            } else {
                Ok(())
            };
            if recorded.is_ok() {
                zeusdb_vector_core::kill_at(
                    zeusdb_vector_core::KillPoint::AddAfterAppendBeforeCommit,
                );
            }

            // One commit for the segment, before any of it is installed. A
            // commit that fails refuses every record it was to make durable
            // and the rest of the batch with them, installing none, since
            // the call's contract is that what it returns is on the device.
            // The ids issued stay issued, because the records may have
            // reached the sink.
            let committed = recorded.and_then(|()| {
                if admitted > 0 {
                    self.commit_records()
                } else {
                    Ok(())
                }
            });
            if let Err(err) = committed {
                let class = err.exception().name();
                let message = err.to_string();
                let failed = |id: &str| {
                    InsertError::Counted(format!("Vector {}: {}: {}", id, class, message))
                };
                for outcome in segment {
                    errors.push(match outcome {
                        Ok(admitted) => failed(&admitted.id),
                        Err(e) => e,
                    });
                }
                for record in pending {
                    errors.push(failed(&record.id));
                }
                return (inserted_ids, errors);
            }
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::AddAfterCommitBeforeApply);

            // The records a replacing segment replaces go once its records
            // are durable, as one batch under one set of guards, which is how
            // a replay applies the removal's record, so the close-up on
            // removal runs at the same point in both.
            if !replaced.is_empty() {
                let missing = self.remove_points_internal(writers, &replaced);
                debug_assert!(
                    missing.is_empty(),
                    "every replaced id was held at its admission, under the mutation guard"
                );
                if tracing::enabled!(target: LOG_TARGET, tracing::Level::TRACE) {
                    for id in &replaced {
                        trace!(target: LOG_TARGET, operation = "overwrite_removal",
                            vector_id = %id,
                            "Removed existing vector/codes for overwrite"
                        );
                    }
                }
                info!(target: LOG_TARGET, operation = "overwrite_removal_complete",
                    removed_count = replaced.len() - missing.len(),
                    removal_errors = 0,
                    "Completed removal phase for overwrite"
                );
            }

            for outcome in segment {
                let admitted = match outcome {
                    Ok(admitted) => admitted,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                let id_for_error = admitted.id.clone();
                match self.install(admitted) {
                    Ok(()) => {
                        // The id is recorded here rather than counted, because the
                        // caller now needs to know which records landed and not only
                        // how many. `total_inserted` is this list's length.
                        inserted_ids.push(id_for_error.clone());
                        {
                            let mut count = self.vector_count.lock().unwrap();
                            *count += 1;
                        }

                        // Check training trigger (graceful failure handling)
                        if let Err(training_error) = self.maybe_trigger_training(writers) {
                            warn!(target: LOG_TARGET, operation = "training_trigger",
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
        }

        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::AddAfterApply);
        (inserted_ids, errors)
    }

    /// How many records the next segment of a batch may admit, or `None`
    /// for every record left.
    ///
    /// A segment ends at a record whose install may replace the graph under
    /// the records after it, which is the record that fills the training
    /// set: training fires as it is installed, rebuilds the graph with a
    /// fresh level stream, and every record after it draws from that
    /// stream, so their levels cannot be drawn ahead of it. Nothing inside
    /// `add` replaces the graph on an index declared without quantization or
    /// on one whose codebook is fitted. On one whose threshold is reached
    /// and whose codebook is not fitted, which is an index whose training
    /// failed, every install tries again, so the segment is one record.
    fn segment_limit(&self) -> Option<usize> {
        let config = self.dense().quantization_config.as_ref()?;
        if self.can_use_quantization() {
            return None;
        }
        if self.training_threshold_reached.load(Ordering::Acquire) {
            return Some(1);
        }
        let collected = self.training_ids.read().unwrap().len();
        Some(config.training_size.saturating_sub(collected).max(1))
    }

    /// What a segment's records take before any is admitted: nothing, against
    /// the graph's room and the sparse space's forward arena as they stand.
    fn taken_by_segment(&self) -> Taken {
        let room = self.dense().index.read().unwrap().graph().room();
        let forward = self
            .sparse()
            .map_or(0, |space| space.index.read().unwrap().forward_len());
        Taken {
            text: 0,
            forward,
            nonzeros: 0,
            room,
        }
    }

    /// Record one segment of a batch that replaces a held record, once its
    /// admission is over: the removal record naming the held ids `replaced`,
    /// then each admitted record's insert, encoded from what admission holds
    /// at the id and the level it issued. The terms admission interned are
    /// already in the sink ahead of these. An append that fails withdraws
    /// what the segment appended, those terms among it, and leaves the fault;
    /// see `withdraw_records`.
    fn append_segment(
        &self,
        replaced: &[String],
        segment: &[Result<Admitted, InsertError>],
        buffer: &mut Vec<u8>,
        handed_over: &mut usize,
    ) -> Result<(), Error> {
        let mut append = || -> Result<(), Error> {
            if !replaced.is_empty() {
                self.record_encoded(OperationKind::Remove, |out| {
                    Operation::encode_remove(replaced, out)
                })?;
                zeusdb_vector_core::kill_at(
                    zeusdb_vector_core::KillPoint::OverwriteAfterRemovalAppend,
                );
            }
            for admitted in segment.iter().flatten() {
                let parts = InsertParts {
                    id: &admitted.id,
                    vector: &admitted.vector,
                    metadata: &admitted.metadata,
                    sparse: admitted.sparse.as_ref().map(SparseVector::as_ref),
                };
                Operation::encode_insert(&parts, buffer);
                Operation::stamp_insert(buffer, admitted.internal_id as u64, admitted.level as u8);
                self.append_record(OperationKind::Insert, buffer)?;
                *handed_over += 1;
                if *handed_over == 2 {
                    zeusdb_vector_core::kill_at(
                        zeusdb_vector_core::KillPoint::AddAfterSecondAppend,
                    );
                }
            }
            Ok(())
        };
        append().map_err(|error| self.withdraw_records(error))
    }

    /// One record through the door.
    ///
    /// Every check that can refuse the record runs first, being the
    /// duplicate check against the index and against the batch so far, the
    /// id ceiling, the terms counted into ids where the sparse half is
    /// terms, which hands each new term to the sink as it is issued, the
    /// sparse rules under the space's weighting with the forward arena's
    /// ceiling, where a sink is attached the sink's ceiling on the encoded
    /// record, and the id store's ceilings on the id and its text with the
    /// vector's width. The id is issued at the ceiling's check, in the one
    /// acquisition of the counter, and every refusal after it takes the id
    /// back, since nothing else issues one under the mutation guard, so a
    /// refused record leaves the counter where it was. Only once every check
    /// has passed is the level drawn, so a record those refuse leaves the
    /// level stream where it was too. The graph's room is then taken at that
    /// level, since what a record can add to the graph's arenas follows from
    /// its level, and only then is the level written with the id into the
    /// encoded record and the record handed over. A record the graph's room
    /// or the sink refuses takes the id back as well; the draw it consumed
    /// moves the stream, which the next record's own draw records. Nothing
    /// the install then does can refuse a record admitted here for its id,
    /// its width or the room it takes.
    ///
    /// `taken` is what the records this segment has admitted and not yet
    /// installed take once they are: the text the id store's arena will hold
    /// before this record's id, the nonzeros the forward arena will hold
    /// before its own, and their room in the graph. Each store is the one
    /// that stands, before any record the batch replaces goes.
    ///
    /// `overwrite` says a held id the record names counts as free, and the
    /// second value returned says the record replaces a held one. `defer`
    /// says the batch replaces a held record, so the record is held to the
    /// sink's ceiling and handed over later, with its segment; see
    /// `append_segment`. A record refused here keeps the record it names
    /// held, since the removal names the ids of admitted records alone.
    ///
    /// With no sink attached nothing is encoded and the sink's ceiling does
    /// not apply. The id ceiling applies either way.
    #[allow(clippy::too_many_arguments)]
    fn admit(
        &self,
        record: ParsedRecord,
        seen: &mut HashSet<String>,
        attached: bool,
        buffer: &mut Vec<u8>,
        taken: &mut Taken,
        overwrite: bool,
        defer: bool,
    ) -> Result<(Admitted, bool), InsertError> {
        let ParsedRecord {
            id,
            vector,
            sparse,
            metadata,
        } = record;

        // The duplicate check, whether the record replaces a held one, and
        // from the same guard the length of the arena the record's id is held
        // to below.
        let (is_new, arena, held) = if seen.contains(&id) {
            (false, 0, false)
        } else {
            let ids = self.ids.read().unwrap();
            let held = ids.contains_name(&id);
            (overwrite || !held, ids.arena_len(), held)
        };
        if !is_new {
            warn!(target: LOG_TARGET, operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(InsertError::Vector {
                err: Error::DuplicateId { id: id.clone() },
                id,
            });
        }

        // The id ceiling, before anything is counted, encoded or drawn for
        // the record, and the issue with it. Issued past the last one the id
        // store holds, an id would reach the sink and then be refused at
        // install, and a replay of that record would refuse the open.
        // Refused here, the record leaves the counter, the level stream, the
        // dictionary and the sink where they were, journaled or not.
        let internal_id = match self.issue_id() {
            Ok(issued) => issued,
            Err(err) => return Err(InsertError::Vector { err, id }),
        };
        let refuse = |err: Error, id: String| -> Result<(Admitted, bool), InsertError> {
            self.retract_id(internal_id);
            Err(InsertError::Vector { id, err })
        };

        // The sparse half. Terms are counted into term ids here, under the
        // mutation guard the caller holds, so the ids a record's postings
        // carry are the dictionary's at the moment the record is inserted
        // and nothing can empty the dictionary between the two. A sparse
        // vector is then held to its rules, the weighting's included, and to
        // the collection having a sparse space, before anything is written,
        // so a record refused for its sparse half leaves nothing behind in
        // the dense one.
        let sparse = match sparse {
            None => None,
            Some(SparseHalf::Vector(vector)) => Some(vector),
            Some(SparseHalf::Terms(terms)) => match self.count_record_terms(&terms) {
                Ok(vector) => Some(vector),
                Err(err) => return refuse(err, id),
            },
        };
        if let Some(sparse) = &sparse {
            let checked = self
                .sparse()
                .ok_or(Error::NoSparseSpace)
                .and_then(|space| space.config().weighting.validate_record(sparse.as_ref()))
                .and_then(|()| {
                    refuse_forward(taken.forward, taken.nonzeros, sparse.dims.len())
                        .map_err(|err| self.sparse_refusal_named(err))
                });
            if let Err(err) = checked {
                return refuse(err, id);
            }
        }

        // The sink's ceiling. A record handed over later is encoded here
        // only where the bound does not settle it.
        if attached {
            let parts = InsertParts {
                id: &id,
                vector: &vector,
                metadata: &metadata,
                sparse: sparse.as_ref().map(SparseVector::as_ref),
            };
            if !defer || Operation::insert_size_bound(&parts) > JOURNAL_MAX_PAYLOAD {
                Operation::encode_insert(&parts, buffer);
                if buffer.len() > JOURNAL_MAX_PAYLOAD {
                    let err = Error::JournalRecordTooLarge {
                        bytes: buffer.len(),
                        ceiling: JOURNAL_MAX_PAYLOAD,
                    };
                    return refuse(err, id);
                }
            }
        }

        // The id store's refusals and the vector's width, run before the level
        // is drawn and the record handed to the sink, so the install meets none
        // of them. A refusal here takes the issued id back.
        if let Err(err) = self.refuse_unplaceable(&id, &vector, arena.saturating_add(taken.text)) {
            return refuse(err, id);
        }

        // The graph's room at the level just drawn, counting what the records
        // this segment admitted may take, before the record is recorded.
        let level = self.draw_level();
        if let Err(err) = taken.room.take(level) {
            return refuse(err, id);
        }
        if attached && !defer {
            Operation::stamp_insert(buffer, internal_id as u64, level as u8);
            if let Err(err) = self.append_record(OperationKind::Insert, buffer) {
                return refuse(err, id);
            }
        }

        seen.insert(id.clone());
        taken.text += id.len();
        taken.nonzeros += sparse.as_ref().map_or(0, |sparse| sparse.dims.len());
        Ok((
            Admitted {
                id,
                vector,
                sparse,
                metadata,
                internal_id,
                level,
            },
            held,
        ))
    }

    /// The second pass over one admitted record: the vector put into the
    /// graph at the level admission drew, the record named at the next
    /// position with its metadata and columns written there, and the sparse
    /// half into its space at the same position, all under the internal id
    /// admission issued. Routed by the storage state as `add` has always
    /// routed, being raw storage, raw storage while collecting for training,
    /// or quantized storage.
    pub(super) fn install(&self, admitted: Admitted) -> Result<(), Error> {
        let Admitted {
            id,
            vector,
            sparse,
            metadata,
            internal_id,
            level,
        } = admitted;

        trace!(target: LOG_TARGET, operation = "add_single_vector",
            vector_id = %id,
            is_new = true,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        let at = if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(&id, vector, metadata, internal_id, level)?
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(&id, vector, metadata, internal_id, level)?
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(&id, vector, metadata, internal_id, level)?
        };

        // The sparse half, at the same position and under the same internal
        // id, once the dense half and the maps are in. Training, which an
        // install can set off, runs after the record's last write, so the
        // position is the one the record still holds.
        if let Some(sparse) = sparse {
            self.insert_sparse(at, internal_id, &sparse)?;
        }

        Ok(())
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &self,
        id: &str,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>,
        internal_id: usize,
        level: usize,
    ) -> Result<Position, Error> {
        // Insert the processed vector into the graph, name the record in the
        // id store, and write its metadata and the declared fields of it in
        // their columns, in the two phases the graph splits an insertion into,
        // at the level admission drew. See `insert_one`. The four are written
        // there, under one acquisition, so that the record enters the
        // collection's live set, the index's and the columns' together. This
        // is the only copy of the vector the index keeps: the store the graph
        // is addressed against holds it, and there is no second map to write.
        let (at, _) = self.insert_one(id, &vector, metadata, internal_id, level)?; // Already normalized

        trace!(target: LOG_TARGET, operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(at)
    }

    /// Path B: ID collection for consistent training
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &self,
        id: &str,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
        internal_id: usize,
        level: usize,
    ) -> Result<Position, Error> {
        // 1. Store vector normally (single storage)
        let at = self.add_raw_vector(id, vector, metadata, internal_id, level)?;

        // 2. Collect ID for training (minimal memory overhead)
        //
        // The training set is the first `training_size` records to arrive, and
        // which records it holds cannot be drawn randomly. Training fires on the
        // record that reaches `training_size`, so the index holds exactly
        // `training_size` records at that moment and any sample of the records
        // present is the whole of them. Drawing from a wider pool would mean
        // deferring the trigger and holding more records raw, which is a change
        // of shape rather than a change of sampling.
        //
        // What the membership would buy was measured, by feeding the current
        // design its worst case instead of changing it. Three corpora at 25,000
        // records and dim 768, each built twice, once inserted in generation
        // order and once sorted so the training set is one segment.
        //
        //   corpus                              in order   sorted
        //   50 Gaussian clusters                  0.996      0.993
        //   8 sources, disjoint 48-dim subspaces  0.887      0.930
        //   8 sources, disjoint variance blocks   0.268      0.347
        //
        // Sorted trains on 2 clusters of 50 in the first row and 1 source of 8
        // in the other two, and it is no worse in any of them. The reason is
        // that a codebook is fitted per contiguous coordinate slice, so a
        // segment only looks different to it if its per-coordinate marginals
        // differ, and those are far more stable across content than the joint
        // distribution is.
        //
        // What is drawn randomly is the order the sample is held in, which is
        // what every subset of it is taken by. `training_sample` shuffles it
        // under a fixed seed; see `TRAINING_SAMPLE_SEED`.
        if let Some(config) = &self.dense().quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.to_string());
                    let progress = (training_ids.len() as f32 / config.training_size as f32
                        * 100.0)
                        .min(100.0);

                    trace!(target: LOG_TARGET, operation = "training_id_collection",
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
                        info!(target: LOG_TARGET, operation = "training_threshold_reached",
                            collected_count = training_ids.len(),
                            target_size = config.training_size,
                            "Training threshold reached - ready for PQ training"
                        );
                    }
                }
            }
        }

        Ok(at)
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &self,
        id: &str,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
        internal_id: usize,
        level: usize,
    ) -> Result<Position, Error> {
        // Quantize and insert, name the record in the id store and write its
        // metadata and columns, in the two phases the graph splits an
        // insertion into, at the level admission drew. The index quantizes in
        // the first phase, under the read guard, and installs the codes in the
        // second, carrying the raw vector beside them where the storage mode
        // keeps one, because the node the codes are installed at is the node
        // the raw has to sit at. The maps are written in the second phase, so
        // that the record enters the collection's live set, the index's and
        // the columns' under one acquisition. See `insert_one`.
        let (at, codes) = self.insert_one(id, &vector, metadata, internal_id, level)?;

        // Store the product quantized codes, keyed by external id for the
        // paths that reach a record by name. A scalar graph holds its row in
        // its own store and nowhere else, so there is nothing to keep here.
        if let Some(codes) = &codes {
            let mut pq_codes = self.dense().pq_codes.write().unwrap();
            pq_codes.insert(id.to_string(), codes.clone());
        }
        let codes_length = codes.as_ref().map_or(0, Vec::len);

        trace!(target: LOG_TARGET, operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes_length,
            "Quantized vector added successfully"
        );

        Ok(at)
    }

    /// The whole of `add` once parsing is over.
    ///
    /// The records arrive parsed and processed for the space, with the
    /// per-record parse failures already worded, and the binding calls this
    /// with the interpreter lock released. The mutation guard is taken here,
    /// around the insertion alone, so a caller waiting for another writer
    /// waits without the lock. Holding it while waiting would stall every
    /// Python thread in the process for the length of the writer ahead.
    ///
    /// `insert_parsed_records` carries the proof that nothing inside touches
    /// Python. The errors come back in the order they happened. Two of the
    /// three variants carry a message Rust already built. The third carries
    /// the `Error` the record's insertion raised, formatted here against its
    /// id.
    pub fn add(
        &self,
        parsed_data: ParsedRecords,
        parse_errors: Vec<String>,
        overwrite: bool,
    ) -> Added {
        let records = parsed_data.into_iter().map(ParsedRecord::from).collect();
        self.add_records(records, parse_errors, overwrite)
    }

    /// `add` for records that may fill the sparse space as well.
    ///
    /// What the binding's `add` becomes once parsing is over, taking the
    /// record shape a Rust caller builds. A record whose sparse half is terms
    /// has them counted into term ids inside the mutation guard this takes,
    /// as the record is inserted, so nothing can empty the dictionary between
    /// a record's ids being issued and its postings being written. The terms
    /// themselves come from `tokenize`, which runs under no guard.
    pub fn add_records(
        &self,
        parsed_data: Vec<ParsedRecord>,
        parse_errors: Vec<String>,
        overwrite: bool,
    ) -> Added {
        let start_time = Instant::now();

        let mut total_errors = 0;
        let mut errors = Vec::new();

        // Add parse errors to the collection
        for parse_error in parse_errors {
            errors.push(parse_error);
            total_errors += 1;
        }

        if parsed_data.is_empty() && errors.is_empty() {
            trace!(
                target: ENTRY_TARGET,
                operation = "add_vectors",
                result = "empty_input",
                "No vectors to process"
            );
            return Added {
                inserted: vec![],
                errors: vec![],
                total_errors: 0,
                vector_shape: Some((0, self.dense().dim)),
            };
        }

        let total_input_count = parsed_data.len() + total_errors;
        let vector_shape = Some((total_input_count, self.dense().dim));

        debug!(
            target: ENTRY_TARGET,
            operation = "add_vectors_start",
            total_vectors = parsed_data.len(),
            parse_errors = total_errors,
            overwrite = overwrite,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            storage_mode = self.storage_mode(),
            "Starting vector addition"
        );

        let (inserted_ids, insert_errors) = {
            let writers = self.hold_writers();
            self.insert_parsed_records(&writers, parsed_data, overwrite)
        };
        let total_inserted = inserted_ids.len();

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
                        target: ENTRY_TARGET,
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
            target: ENTRY_TARGET,
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
            final_storage_mode = self.storage_mode(),
            "Vector addition completed"
        );

        self.warn_if_outgrown_expected_size();

        Added {
            inserted: inserted_ids,
            errors,
            total_errors,
            vector_shape,
        }
    }

    /// Remove one record by id. `false` for an id the index does not hold.
    ///
    /// The removal's record, naming the id where the index holds it, is
    /// handed to the sink before the record goes, and a sink that refuses it
    /// refuses the call.
    pub fn remove_point(&self, id: String) -> Result<bool, Error> {
        let writers = self.hold_writers();
        let held = self.held_ids(std::slice::from_ref(&id));
        self.record_removal(&held)?;
        self.remove_point_internal(&writers, id)
            .map_err(Error::Engine)
    }

    /// Remove a batch of records under one acquisition of the mutation guard,
    /// returning the ids that were not in the index. See
    /// `remove_points_internal`. The record names the ids the index held,
    /// each once, and is handed over before any goes.
    pub fn remove_points(&self, ids: &[String]) -> Result<Vec<String>, Error> {
        let writers = self.hold_writers();
        let held = self.held_ids(ids);
        self.record_removal(&held)?;
        Ok(self.remove_points_internal(&writers, ids))
    }

    /// Remove the records these ids name and report how many went.
    ///
    /// A repeated id names one record, so it counts once whether or not it
    /// was there. `remove_points_internal` already skips a repeat rather than
    /// reporting it missing.
    pub fn delete_ids(&self, requested: &[String]) -> Result<usize, Error> {
        let distinct: HashSet<&String> = requested.iter().collect();
        let distinct_count = distinct.len();

        let missing = {
            let writers = self.hold_writers();
            let held = self.held_ids(requested);
            self.record_removal(&held)?;
            self.remove_points_internal(&writers, requested)
        };

        Ok(distinct_count - missing.len())
    }

    /// Remove every record whose metadata matches the filter, and report how
    /// many were removed.
    ///
    /// **A filter matching every record is refused.** Everywhere else in this
    /// language an empty filter matches every record, and `search(filter={})`
    /// returns the whole index for exactly that reason. This is the one
    /// operation where following that rule destroys every record, so it is
    /// asked of the compiled tree, which also refuses `{"$and": []}` and
    /// `{"$not": {"$or": []}}`. A caller who does mean it names the records.
    pub fn remove_where(&self, conditions: &Filter) -> Result<usize, Error> {
        if conditions.matches_every_record() {
            return Err(Error::RemoveWhereMatchesEverything);
        }
        let writers = self.hold_writers();
        self.remove_where_locked(&writers, conditions)
    }

    /// Replace one record's metadata without touching its vector. See
    /// `update_metadata_locked`. The record, being the id and the whole new
    /// map, is handed to the sink before anything is written, and only
    /// where the index holds the id, since nothing is written otherwise.
    pub fn update_metadata(
        &self,
        id: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<bool, Error> {
        let _writers = self.writers.lock().unwrap();
        if self.contains(id) {
            self.record(|| Operation::UpdateMetadata {
                id: id.to_string(),
                metadata: metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })?;
            self.commit_records()?;
        }
        Ok(self.update_metadata_locked(id, metadata))
    }

    /// Rebuild the graph in memory and reclaim the nodes removal and overwrite
    /// strand. See `compact_locked`. The record is handed over whether or
    /// not the graph turns out to hold a stranded node, since the decision is
    /// the body's and a replay of the record makes it the same way.
    pub fn compact(&self) -> Result<usize, Error> {
        let writers = self.hold_writers();
        self.record(|| Operation::Compact)?;
        self.commit_records()?;
        self.compact_locked(&writers)
    }

    /// Return the graph's spare buffer capacity, the dense index's
    /// numbering's, and the id store's, the metadata store's and the
    /// columns' to the allocator, reporting the bytes released, which is what
    /// `reserved_memory_mb` priced. Each store is shrunk under its own write
    /// guard, taken alone. See `VectorGraph::shrink_to_fit` and
    /// `DenseIndex::shrink_to_fit`.
    pub fn shrink_to_fit(&self) -> usize {
        let _writers = self.writers.lock().unwrap();
        let ids = self.ids.write().unwrap().shrink_to_fit();
        let dense = self.dense().index.write().unwrap().shrink_to_fit();
        let metadata = self.vector_metadata.write().unwrap().shrink_to_fit();
        let columns = self.columns.write().unwrap().shrink_to_fit();
        ids + dense + metadata + columns
    }

    /// Resolve a rebuild request against the configuration the index holds.
    ///
    /// At least one of the three has to be given, since rebuilding the graph
    /// as it stands is `compact`. The three are held to the rules `create()`
    /// applies, on the same five values, and an invalid one raises the message
    /// `create()` raises for it. The plan is returned rather than acted on so
    /// the binding can raise the neighbour selection warning, which needs the
    /// interpreter, between the check and the rebuild.
    pub fn plan_rebuild(
        &self,
        m: Option<usize>,
        expected_size: Option<usize>,
        ef_construction: Option<usize>,
    ) -> Result<RebuildPlan, Error> {
        if m.is_none() && expected_size.is_none() && ef_construction.is_none() {
            return Err(Error::RebuildWithoutChanges);
        }
        let new_m = m.unwrap_or_else(|| self.dense().m());
        let new_expected_size = expected_size.unwrap_or_else(|| self.expected_size());
        let new_ef_construction = ef_construction.unwrap_or_else(|| self.dense().ef_construction());
        validate_index_parameters(
            self.dense().dim,
            &self.dense().metric,
            new_m,
            new_ef_construction,
            new_expected_size,
            "",
        )?;
        Ok(RebuildPlan {
            m: new_m,
            expected_size: new_expected_size,
            ef_construction: new_ef_construction,
            // A raised declaration is a new bar for the overgrowth warning, and
            // the old one has already been claimed if it fired.
            rearm_overgrowth: expected_size.is_some(),
        })
    }

    /// Rebuild the graph at the degree a plan names. See `rebuild_locked`.
    /// The record carries the three values the plan resolved, never the
    /// options the caller gave, since an option's meaning depends on the
    /// configuration it was resolved against.
    pub fn rebuild(&self, plan: RebuildPlan) -> Result<usize, Error> {
        if plan.rearm_overgrowth {
            self.overgrowth_warned.store(false, Ordering::Release);
        }
        let writers = self.hold_writers();
        self.record(|| Operation::Rebuild {
            m: plan.m as u64,
            expected_size: plan.expected_size as u64,
            ef_construction: plan.ef_construction as u64,
        })?;
        self.commit_records()?;
        self.rebuild_locked(&writers, plan.m, plan.expected_size, plan.ef_construction)
    }

    /// Empty the index, keeping its configuration, and report how many
    /// records went.
    ///
    /// **A fresh graph and empty maps, not `remove_points` over every id.**
    /// Removing every record one at a time is linear in the record count and
    /// leaves one stranded graph node per record. Replacing the graph reclaims
    /// all of it at once and leaves `stranded_graph_nodes` at zero, which is
    /// what an empty index should report.
    ///
    /// What it keeps is the index: the declaration, the index-level metadata,
    /// and the quantization configuration including a fitted codebook.
    /// **Training is not undone.** A codebook is fitted from data that is now
    /// gone and cannot be refitted from an empty index, so a trained quantized
    /// index stays trained and its replacement graph is a quantized graph. An
    /// untrained one returns to collecting. The internal id counter restarts,
    /// because nothing is left for a reissued id to collide with and
    /// restarting keeps the internal ids in step with the fresh graph's node
    /// indices. The generated id counter does not; see the field.
    ///
    /// **Identity is kept whole.** The collection id, the generation and
    /// the snapshot it was last saved as or read from are the collection's,
    /// and a clear is a mutation of the collection rather than a new one: a
    /// journaled collection records it in the journal whose header names
    /// that id, so the next checkpoint must name it too. The next save is
    /// the next generation, so a directory saved before the clear reads as
    /// the earlier one.
    ///
    /// **One trade, and the old structures dropped after `writers`.** Every
    /// structure the clear empties, the sparse space's index and dictionary
    /// among them, is traded for a fresh one under one acquisition of their
    /// write guards, so a reader sees the whole state before the clear or
    /// the whole cleared state. What they held is dropped once `writers` is
    /// let go, as a close-up's retired parts are, so the next writer does
    /// not wait for it.
    pub fn clear(&self) -> Result<usize, Error> {
        let writers = self.writers.lock().unwrap();
        let replacement = self.replacement_for_clear()?;
        self.record(|| Operation::Clear)?;
        self.commit_records()?;
        let (removed, cleared) = self.clear_with(replacement);
        drop(writers);
        drop(cleared);
        Ok(removed)
    }

    /// The body of `clear`, with the mutation guard held by the caller and
    /// the record already handed over. What the clear replaced is handed
    /// back for the caller to drop once it lets `writers` go.
    pub(super) fn clear_locked(&self) -> Result<(usize, Cleared), Error> {
        let replacement = self.replacement_for_clear()?;
        Ok(self.clear_with(replacement))
    }

    /// The fresh structures a clear installs, and whether the graph it
    /// replaces scores against codes. Built before any guard is taken, so
    /// every allocation happens outside the write guards, as `compact`
    /// arranges it for its graph. The stores reserve for the declared size,
    /// under their caps, and the code map and the training list hold the
    /// capacity the old ones hold, which is what emptying them in place kept.
    fn replacement_for_clear(&self) -> Result<ClearReplacement, Error> {
        let (quantized, index) = self.fresh_index_for_clear()?;
        let names = self.columns.read().unwrap().declared().to_vec();
        let codes = self.dense().pq_codes.read().unwrap().capacity();
        let training = self.training_ids.read().unwrap().capacity();
        let sparse = self.sparse().map(|space| {
            (
                PostingsIndex::new(space.config().clone()),
                space
                    .text
                    .as_ref()
                    .map(|_| zeusdb_vector_text::TermDictionary::new()),
            )
        });
        Ok(ClearReplacement {
            quantized,
            ids: IdStore::new(self.expected_size()),
            index,
            pq_codes: HashMap::with_capacity(codes),
            sparse,
            vector_metadata: MetadataStore::new(self.expected_size()),
            columns: ColumnStore::new(names, self.expected_size()),
            training_ids: Vec::with_capacity(training),
        })
    }

    /// The fresh index a clear installs, and whether the graph it replaces
    /// scores against codes.
    fn fresh_index_for_clear(&self) -> Result<(bool, DenseIndex), Error> {
        let quantized = self.is_quantized();
        let pq = self.dense().pq.as_ref().cloned();
        let int8 = self.dense().int8_codec().cloned();

        if quantized && pq.is_none() && int8.is_none() {
            return Err(Error::NoQuantizer);
        }

        let fresh = if let (true, Some(codec)) = (quantized, int8) {
            // A cleared scalar index keeps its scales, as a cleared product
            // quantized one keeps its codebook, so its replacement is a
            // scalar graph over the same codec, and it holds nothing else.
            VectorGraph::new_int8(
                &self.dense().metric,
                self.dense().m(),
                self.expected_size(),
                MAX_LAYER,
                self.dense().ef_construction(),
                codec,
            )
        } else if let (true, Some(pq)) = (quantized, pq) {
            let mut graph = VectorGraph::new_pq(
                &self.dense().metric,
                self.dense().m(),
                self.expected_size(),
                MAX_LAYER,
                self.dense().ef_construction(),
                pq,
            );
            // A cleared `quantized_with_raw` index goes on keeping raw
            // vectors, so its replacement graph opens the store the next
            // insertion writes into. Without this the store is absent and
            // every record added after a clear would lose its raw vector.
            //
            // The declaration reaches it through the same byte budget the
            // graph's own arenas go through. It used to reach it directly, and
            // `expected_size` is validated at 100 million, so an index declared
            // at that size and dimension 1,536 asked one `Vec::with_capacity`
            // for 614 GB. That aborts the process instead of unwinding, so no
            // exception could be raised for it.
            if self.dense().keeps_raw() {
                let records =
                    VectorGraph::reserved_raw_records(self.dense().dim, self.expected_size());
                graph
                    .open_raw_store(self.dense().dim, records)
                    .expect("a quantized graph accepts a raw side store");
            }
            graph
        } else {
            VectorGraph::new_raw(
                &self.dense().metric,
                self.dense().dim,
                self.dense().m(),
                self.expected_size(),
                MAX_LAYER,
                self.dense().ef_construction(),
            )
        };

        // The replacement index, wrapping the fresh graph, with an empty live
        // set since the maps are about to be empty too. Built and timed
        // before any guard is taken, so the guarded block is a move and not
        // a construction.
        Ok((quantized, self.dense().fresh_index(fresh)))
    }

    /// Trade every structure a clear empties for the fresh one
    /// `replacement_for_clear` built, reporting how many records went and
    /// handing back what the structures held, for the caller to drop once it
    /// lets `writers` go.
    fn clear_with(&self, replacement: ClearReplacement) -> (usize, Cleared) {
        let ClearReplacement {
            quantized,
            ids: fresh_ids,
            index: fresh_index,
            pq_codes: fresh_codes,
            sparse: fresh_sparse,
            vector_metadata: fresh_metadata,
            columns: fresh_columns,
            training_ids: fresh_training,
        } = replacement;
        // The storage guards in the order declared on the struct, which is
        // the order every other multi-guard path here takes them in, the
        // sparse space's index and its dictionary among them.
        //
        // The dense index is one of them, between the id store and the codes
        // it ranks above and below. Emptying the store here and replacing the
        // index afterwards left the collection's live set empty while the
        // index's still held every record, which is the same disagreement the
        // insertion path carried and which a `get_stats` running beside a
        // `clear` observed. The sparse space was replaced after the others
        // under its own guard, so a reader between the two saw the cleared
        // record set beside the old postings. Every structure is one state,
        // so they empty under one acquisition. A record's terms are counted
        // into ids under `writers`, and a query's under the dictionary's read
        // guard held through its search, so nothing holds an id the old
        // dictionary issued once the trade is done.
        let (removed, cleared) = {
            let mut ids = self.ids.write().unwrap();
            let mut index = self.dense().index.write().unwrap();
            let mut pq_codes = self.dense().pq_codes.write().unwrap();
            let mut sparse = self.sparse().map(|space| {
                (
                    space.index.write().unwrap(),
                    space
                        .text
                        .as_ref()
                        .map(|text| text.dictionary.write().unwrap()),
                )
            });
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            let mut columns = self.columns.write().unwrap();
            let mut training_ids = self.training_ids.write().unwrap();
            let mut id_counter = self.id_counter.lock().unwrap();
            let mut vector_count = self.vector_count.lock().unwrap();

            let removed = ids.len();
            let sparse = match (sparse.as_mut(), fresh_sparse) {
                (Some((index, dictionary)), Some((fresh, fresh_dictionary))) => Some((
                    std::mem::replace(&mut **index, fresh),
                    dictionary
                        .as_mut()
                        .zip(fresh_dictionary)
                        .map(|(dictionary, fresh)| std::mem::replace(&mut **dictionary, fresh)),
                )),
                _ => None,
            };
            let cleared = Cleared {
                _parts: Some(ClearedParts {
                    _ids: std::mem::replace(&mut *ids, fresh_ids),
                    _index: std::mem::replace(&mut *index, fresh_index),
                    _pq_codes: std::mem::replace(&mut *pq_codes, fresh_codes),
                    _sparse: sparse,
                    _vector_metadata: std::mem::replace(&mut *vector_metadata, fresh_metadata),
                    _columns: std::mem::replace(&mut *columns, fresh_columns),
                    _training_ids: std::mem::replace(&mut *training_ids, fresh_training),
                }),
            };
            *id_counter = 0;
            *vector_count = 0;
            (removed, cleared)
        };
        #[cfg(test)]
        reach_clear(ClearPoint::Traded);

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
            target: ENTRY_TARGET,
            operation = "clear",
            records_removed = removed,
            quantized = quantized,
            "Index cleared"
        );

        (removed, cleared)
    }
}

/// The fresh structures a clear trades in, built by
/// `Collection::replacement_for_clear` before any guard is taken, and
/// whether the graph they replace scores against codes.
struct ClearReplacement {
    quantized: bool,
    ids: IdStore,
    index: DenseIndex,
    pq_codes: HashMap<String, Vec<u8>>,
    /// The sparse space's postings and, where it has a text layer, its
    /// dictionary.
    sparse: Option<(PostingsIndex, Option<zeusdb_vector_text::TermDictionary>)>,
    vector_metadata: MetadataStore,
    columns: ColumnStore,
    training_ids: Vec<String>,
}

/// What a clear traded out, held to be dropped once `writers` is let go.
/// Dropping the graph forks to rayon, and a fork under a write guard
/// deadlocks against a batch search holding the pool; see
/// `DenseSpace::replace_graph`.
pub(super) struct Cleared {
    _parts: Option<ClearedParts>,
}

/// The structures a clear traded out.
struct ClearedParts {
    _ids: IdStore,
    _index: DenseIndex,
    _pq_codes: HashMap<String, Vec<u8>>,
    _sparse: Option<(PostingsIndex, Option<zeusdb_vector_text::TermDictionary>)>,
    _vector_metadata: MetadataStore,
    _columns: ColumnStore,
    _training_ids: Vec<String>,
}

/// The points a clear reaches that a test can hold open.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ClearPoint {
    /// Every structure is traded and every guard let go, and `writers` is
    /// still held.
    Traded,
    /// What the clear traded out is about to be dropped.
    Dropping,
    /// What the clear traded out is dropped.
    Dropped,
}

/// What a test sets for this thread's clears to call at each point.
#[cfg(test)]
pub(super) type ClearHook = Box<dyn FnMut(ClearPoint)>;

// What this thread's clears call at each point, where a test set it.
#[cfg(test)]
thread_local! {
    pub(super) static CLEAR_HOOK: std::cell::RefCell<Option<ClearHook>> =
        const { std::cell::RefCell::new(None) };
}

/// Call this thread's clear hook, where a test set one.
#[cfg(test)]
fn reach_clear(point: ClearPoint) {
    CLEAR_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow_mut().as_mut() {
            hook(point);
        }
    });
}

/// What a clear traded out, dropped in the tests alone between two points a
/// test can hold open or read the guards the dropping thread holds at. A
/// thread already unwinding drops it past no point, as a close-up's parts
/// are.
#[cfg(test)]
impl Drop for Cleared {
    fn drop(&mut self) {
        let observed = !std::thread::panicking();
        if observed {
            reach_clear(ClearPoint::Dropping);
        }
        drop(self._parts.take());
        if observed {
            reach_clear(ClearPoint::Dropped);
        }
    }
}

/// What `add` did.
///
/// `inserted` is the id of every record the call put in the index, in
/// insertion order, and `total_errors` counts the rejected records, which the
/// `errors` list names. `vector_shape` is the input's record count against the
/// index's width. The binding turns this into the `AddResult` Python sees, and
/// the contract of each field is documented there.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Added {
    pub inserted: Vec<String>,
    pub errors: Vec<String>,
    pub total_errors: usize,
    pub vector_shape: Option<(usize, usize)>,
}

/// A rebuild request resolved against the index, ready to run.
///
/// Returned by `Collection::plan_rebuild` and consumed by `Collection::rebuild`.
/// The two fields the neighbour selection warning reads are public so the
/// binding can raise it between the two calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebuildPlan {
    pub m: usize,
    pub expected_size: usize,
    pub ef_construction: usize,
    rearm_overgrowth: bool,
}
