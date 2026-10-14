//! The dense index, being the value under a dense space's graph guard and
//! the first implementor of the seam in `zeusdb-vector-core`.
//!
//! # What it is
//!
//! The graph, the live set the graph is searched under, and what an insertion
//! or a search needs from the space's declaration, being the metric, the
//! width, the degree, the quantizer where the space has one and whether the
//! space keeps a raw vector beside every code. The guard around it is the
//! space's `index` field, so a search runs under a read guard on this value
//! and an installation under a write guard, which is the two-phase insertion
//! the graph was built for.
//!
//! # The numbering
//!
//! A record has a position, which every per-record structure is addressed by,
//! and a node, which the graph is addressed by. Two plain vectors join them.
//! The graph keeps each node's position, with an empty marker for a node a
//! removal or an overwrite has stranded, and the index keeps `node_at`, each
//! position's node, with the same marker for a position that holds no record.
//! The two are written together, under this value's write guard, at every
//! insert, removal and replacement of the graph, and `node_at` is the index's
//! live set: it is what lets the index answer `len`, `holds`, `stranded` and a
//! removal of a position it never held without reaching into the collection.
//!
//! A search under a set admitting everything runs with no predicate at all
//! while every node holds a record, and otherwise under a test of each
//! candidate's position against the empty marker, which is one comparison of
//! a value the traversal has already read.
//!
//! # The two paths
//!
//! An admit set the index can enumerate whose count is at or under
//! [`FULL_SCAN_THRESHOLD`] is scored exactly, record by record, from the
//! graph's own store or from the reconstruction of the record's codes where
//! the store holds none. Everything else traverses the graph under the admit
//! set as a predicate. The collection decides what the admit set is, being a
//! filter's bitmap, the matches of a metadata walk, or a conjunction of a
//! bound and a predicate, and the threshold is applied here, so the rule is
//! stated once whatever built the set.
//!
//! # The tie rule
//!
//! An exact page is ordered by distance and then by position, which is
//! internal id order, and it is
//! cut to `k` keeping every record tied at the distance of the last member
//! where the caller asked for that. The collection orders equal distances by
//! external id string, a rule that predates this seam and that every recorded
//! page holds, and it can apply that rule at the boundary only if the whole
//! tie group is in the page. A traversal's page is returned in the
//! traversal's own order and holds at most `k`.
//!
//! # Cost
//!
//! Two units, both timed on the graph itself when the index is opened,
//! again whenever the graph is replaced, and again each time the live count
//! doubles past the smallest graph worth timing, so a collection built by
//! insertion prices its searches on what it holds rather than on the floor
//! it started from. Neither is persisted, since both move with the machine
//! and the build. A traversal is priced per unit of
//! `ef` from a whole search timed on the graph, because a traversal's time
//! is the memory it touches and a kernel timed in a loop runs from cache:
//! at width 100 a search at `ef` 200 cost as much as eleven thousand kernel
//! evaluations and visited about a thousand nodes. Both units are timed
//! over records scattered across the store, under a bitmap test per
//! candidate as a filtered search pays, and the search is asked for a
//! vector the graph does not hold. An exact scan is priced per
//! record from the kernel. A filtered traversal grows as the admitted share
//! falls, from two measured points. An index too small to time takes a
//! compiled-in floor for each.
//!
//! What the figure prices is the arm's own work. At 50,000 records of width
//! 100 the traversal measured 106 microseconds and a filter admitting every
//! record added 70 more, being the filter's evaluation over the column
//! codes and the page's assembly, which no arm's cost includes and which
//! `ArmPlan::cost` says. A caller reading the plan against a filtered
//! query's wall time sees that difference, not a unit that is wrong.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::error;
use zeusdb_vector_core::{
    restore_graph, restore_int8_graph, Admit, ArtefactRecord, Bitmap, Bounds, Budget, Cost, Dense,
    Dir, DumpBounds, Error, Hit, Hits, Int8Codec, Inventory, Ledger, Persist, Planned, Position,
    Prepared, Record, RecordId, Restore, ScoreKind, Selectivity, VectorGraph, VectorIndex,
    DUMP_FILENAME, PQ,
};

/// The empty marker, being a position that holds no record or a node that
/// holds none.
const NONE: u32 = Position::NONE;

/// The internal id of the record at each position, borrowed from the dense
/// index: its node for each position and the graph's origin id for each
/// node. See [`DenseIndex::ids_by_position`].
pub(crate) struct IdsByPosition<'a> {
    node_at: &'a [u32],
    origin_ids: &'a [usize],
    /// Whether every position below the extent holds a record whose
    /// internal id is one past its position, which a collection holding no
    /// hole and the ids from one to its record count does.
    shifted: bool,
}

impl IdsByPosition<'_> {
    /// The internal id of the record at `at`, where the index holds one.
    ///
    /// A hole's node is [`NONE`], which lies past every node the graph holds,
    /// so the bound of the origin ids answers for it.
    pub(crate) fn get(&self, at: usize) -> Option<usize> {
        if self.shifted {
            return (at < self.node_at.len()).then_some(at + 1);
        }
        let node = *self.node_at.get(at)?;
        debug_assert!(node == NONE || (node as usize) < self.origin_ids.len());
        self.origin_ids.get(node as usize).copied()
    }
}

use super::search::FULL_SCAN_THRESHOLD;
use crate::{prepare_reconstruction, raw_distance_fn, reconstruction_needs_unit, RawVectors};

/// The target every record this file emits carries, being the search
/// module's, since the records here are the ones the traversal emitted from
/// there.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::search";

/// What the floor prices one unit of `ef` at, in kernel evaluations, where
/// the graph is too small to time a search on.
///
/// From one measured point, being 1,192 evaluations for a search at `ef` 200
/// and `k` 10 over 50,000 records at `m` 16. Linear in `ef` is an assumption
/// past that point, and it is the floor alone that uses it: an index large
/// enough to time prices a traversal from a search it ran on itself.
const FLOOR_VISITS_PER_EF: f64 = 6.0;

/// The search timed at open, being `k`, `ef` and how many searches the
/// median is taken over. A search at these settings on 50,000 points takes
/// a quarter of a millisecond at width 100 and a millisecond at 1,536, so
/// the nine together cost a few milliseconds when an index is opened.
const CALIBRATION_SEARCH_K: usize = 10;
const CALIBRATION_SEARCH_EF: usize = 100;
const CALIBRATION_SEARCHES: usize = 9;

/// How the traversal's cost grows as the admitted share falls.
///
/// A filtered traversal visits nodes the predicate rejects, and the more it
/// rejects the further it walks for each result. Fitted to two measured
/// points on 100,000 records, being four times the unfiltered cost at a tenth
/// admitted and thirty two times at a hundredth, which an exponent of three
/// quarters on the inverse of the share reproduces to within a factor of 1.4
/// at both. An assumption beyond those two points.
const SELECTIVITY_EXPONENT: f64 = 0.75;

/// Nodes the store holds before the unit cost is timed rather than taken
/// from the floor. Below this the evaluations cycle through so few vectors
/// that the figure is a cache figure rather than a kernel figure.
const CALIBRATION_MIN_NODES: usize = 64;

/// Kernel evaluations timed per round, and rounds, at open. Ten thousand
/// rather than fewer because a shorter run stays in cache and reads a
/// figure a scan over scattered records never sees: at width 100 the
/// figure moved from 14 to 58 nanoseconds between five hundred and ten
/// thousand evaluations.
const CALIBRATION_EVALUATIONS: usize = 10_000;
const CALIBRATION_ROUNDS: usize = 5;

/// The two units, in nanoseconds: one kernel evaluation over a scattered
/// record, and one unit of `ef` in a traversal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DenseUnits {
    pub(crate) distance_ns: f64,
    pub(crate) ef_ns: f64,
    pub(crate) measured: bool,
}

/// What phase one hands to phase two: the codes a quantized graph installs
/// and the plan the graph made for them, or the plan for a raw vector.
struct DensePlan {
    /// The product quantized codes, where the graph holds them.
    codes: Option<Vec<u8>>,
    /// The scalar row, where the graph holds those.
    row: Option<Vec<i8>>,
    planned: Option<Planned>,
}

/// The value under a dense space's graph guard.
pub(crate) struct DenseIndex {
    graph: VectorGraph,
    /// The node of the record at each position, and [`NONE`] for a position
    /// that holds no record. The index's live set; see the module.
    node_at: Vec<u32>,
    live_count: usize,
    /// Whether every node the graph holds holds a record this index holds,
    /// so no node a traversal can reach carries the empty marker.
    ///
    /// Computed by one pass over the nodes whenever the graph or the
    /// numbering is replaced whole. `insert` leaves it as it is, since the
    /// node it adds holds the record it is filed for, and `remove` clears it,
    /// since the removed record's node stays in the graph. It sits beside the
    /// numbering and the graph under the same guard, so a search reads the
    /// three as one.
    every_node_live: bool,
    metric: String,
    dim: usize,
    pq: Option<Arc<PQ>>,
    keeps_raw: bool,
    /// What a scan and a traversal cost per unit on this graph.
    units: DenseUnits,
    /// Values a scalar quantized graph has clipped at the edge of its code
    /// range since the scales were fitted, summed over every encode: the
    /// training rebuild, every insertion after it, and every replay. A
    /// value beyond the range a dimension's sample reached is saturated
    /// rather than refused, and this is the one place a caller can see how
    /// often that happened; `get_stats` reports it and `quantization.json`
    /// carries it. Zero on every other graph. An atomic because the encode
    /// runs under the index's read guard.
    saturated: AtomicU64,
}

// The case of `DenseIndex::traverse` the last traversal on this thread ran,
// from 1 for no predicate to 4 for the marker's test and the table call, and
// the bytes its predicate closure captures. Read by the test that holds each
// closure to what it needs; nothing outside the tests sees it.
#[cfg(test)]
thread_local! {
    static PREDICATE_BYTES: std::cell::Cell<(u8, usize)> = const { std::cell::Cell::new((0, 0)) };
}

impl DenseIndex {
    /// Wrap a graph. The index holds no record until `set_live` or an insert
    /// files one, and the unit cost is timed on the graph's store, or taken
    /// from the floor where the store is too small.
    pub(crate) fn new(
        graph: VectorGraph,
        metric: &str,
        dim: usize,
        pq: Option<Arc<PQ>>,
        keeps_raw: bool,
    ) -> Self {
        let node_at = Vec::with_capacity(reserved_positions(&graph));
        let mut index = DenseIndex {
            graph,
            node_at,
            live_count: 0,
            every_node_live: false,
            metric: metric.to_string(),
            dim,
            pq,
            keeps_raw,
            units: DenseUnits {
                distance_ns: 0.0,
                ef_ns: 0.0,
                measured: false,
            },
            saturated: AtomicU64::new(0),
        };
        index.every_node_live = every_node_live(&index.graph, &index.node_at);
        index.calibrate();
        index
    }

    /// Values a scalar quantized graph has clipped since its scales were
    /// fitted; see the field.
    pub(crate) fn saturated(&self) -> u64 {
        self.saturated.load(Ordering::Relaxed)
    }

    /// Add to the clipped value count, which the training rebuild does for
    /// the records it encodes off to the side.
    pub(crate) fn add_saturated(&self, values: u64) {
        self.saturated.fetch_add(values, Ordering::Relaxed);
    }

    /// Take the count a saved index recorded.
    pub(crate) fn set_saturated(&self, values: u64) {
        self.saturated.store(values, Ordering::Relaxed);
    }

    pub(crate) fn graph(&self) -> &VectorGraph {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut VectorGraph {
        &mut self.graph
    }

    /// Take the reservation `expected_size` declares, on an index that holds
    /// no record: the graph's, and the vector by position sized to match.
    pub(crate) fn reserve_declared(&mut self, expected_size: usize) {
        self.graph.reserve_declared(expected_size);
        let wanted = reserved_positions(&self.graph);
        self.node_at
            .reserve_exact(wanted.saturating_sub(self.node_at.len()));
    }

    /// The node of the record at each position, and [`NONE`] where a position
    /// holds no record. What a caller holding the index's guard reads a
    /// record's vector through.
    pub(crate) fn node_at(&self) -> &[u32] {
        &self.node_at
    }

    /// Where a record's raw vector is reached by its position, for a caller
    /// holding this value's guard.
    pub(crate) fn raw_vectors(&self) -> RawVectors<'_> {
        RawVectors { index: self }
    }

    /// One record's raw vector, by its position. The node `node_at` names
    /// for the position and the store's vector for that node, in one call.
    /// `None` where the position holds no record, and where the index keeps
    /// no raw vector, which is every record of a trained `quantized_only`
    /// index. A position that holds no record, and a position past the
    /// table, reach the store as the node [`NONE`], which no graph holds, so
    /// the store answers `None` for them.
    #[inline(never)]
    pub(crate) fn raw_vector(&self, at: usize) -> Option<&[f32]> {
        self.graph
            .raw_vector_at(self.node_at.get(at).copied().unwrap_or(NONE))
    }

    /// The vector one record's scalar row stands for, by its position, in
    /// one call. `None` where the position holds no record, and on every
    /// graph but a scalar quantized one. A position that holds no record, and
    /// a position past the table, reach the store as the node [`NONE`], which
    /// no graph holds, so the store answers `None` for them.
    pub(crate) fn int8_reconstruct(&self, at: usize) -> Option<Vec<f32>> {
        self.graph
            .int8_reconstruct_at(self.node_at.get(at).copied().unwrap_or(NONE))
    }

    /// Strand the record at `at`. The position and its node both take the
    /// empty marker, and the node stays in the graph, since a stranded node
    /// still routes a traversal. Answers the node the record held, or
    /// [`NONE`] where the index holds no record at `at`, in which case
    /// nothing is written.
    pub(crate) fn strand(&mut self, at: Position) -> u32 {
        let Some(entry) = self.node_at.get_mut(at.slot()) else {
            return NONE;
        };
        let node = *entry;
        // A hole's node is the marker, which lies past every node the graph
        // holds, so the bound of the graph's entries answers for it.
        let Some(record) = self.graph.records_mut().get_mut(node as usize) else {
            return NONE;
        };
        *record = NONE;
        *entry = NONE;
        self.live_count -= 1;
        self.every_node_live = false;
        node
    }

    /// The node of the record at `at`, where this index holds one.
    pub(crate) fn node_of(&self, at: Position) -> Option<u32> {
        match self.node_at.get(at.slot()) {
            Some(&node) if node != NONE => Some(node),
            _ => None,
        }
    }

    /// Bytes the numbering's vector by position asks the allocator for, for
    /// `Collection::stats`. The graph's vector by node is priced with the
    /// graph.
    pub(crate) fn node_at_heap_bytes(&self) -> usize {
        self.node_at.capacity() * std::mem::size_of::<u32>()
    }

    /// Bytes of that request no position has been written into, which
    /// [`DenseIndex::shrink_to_fit`] returns to the allocator.
    pub(crate) fn node_at_reserved_bytes(&self) -> usize {
        (self.node_at.capacity() - self.node_at.len()) * std::mem::size_of::<u32>()
    }

    /// The internal id of the record at `at`, being the id its node was
    /// inserted under, where this index holds one, for a test reading one
    /// record. A pass over many records takes
    /// [`DenseIndex::ids_by_position`] once.
    #[cfg(test)]
    pub(crate) fn internal_id_at(&self, at: usize) -> Option<usize> {
        self.ids_by_position().get(at)
    }

    /// The internal id of the record at each position, read as two plain
    /// reads a position, `node_at` and then the graph's origin ids.
    pub(crate) fn ids_by_position(&self) -> IdsByPosition<'_> {
        let origin_ids = self.graph.origin_ids();
        // No hole, the first record's id one and the last's the record
        // count: the ids rise with the positions, so they are one to the
        // count, each one past its position.
        let id_at = |at: usize| {
            self.node_at
                .get(at)
                .and_then(|&node| origin_ids.get(node as usize))
                .copied()
        };
        let held = self.live_count;
        let shifted = held > 0
            && held == self.node_at.len()
            && id_at(0) == Some(1)
            && id_at(held - 1) == Some(held);
        IdsByPosition {
            node_at: &self.node_at,
            origin_ids,
            shifted,
        }
    }

    /// Return the graph's spare capacity and the numbering's to the
    /// allocator, and report the bytes released. `node_at` gives back its
    /// slack here, as the map from an id to its node gave back its slack
    /// with the graph it sat in.
    pub(crate) fn shrink_to_fit(&mut self) -> usize {
        let before = self.node_at_heap_bytes();
        self.node_at.shrink_to_fit();
        let numbering = before - self.node_at_heap_bytes();
        self.graph.shrink_to_fit() + numbering
    }

    /// File the records the collection holds, at the positions `positions`
    /// gives, under the nodes the graph files them at, which the loader does
    /// once the id mappings are back.
    ///
    /// The graph's own numbering names each node's position, as the reader
    /// numbered a dump against the mappings or a rebuild inserted each record
    /// at its position. A node whose position is not among `positions` is
    /// stranded. Returns how many of the positions no node holds, which is
    /// zero for a graph built or read from the records and every one of them
    /// for the empty graph a load starts from.
    pub(crate) fn set_live<I: IntoIterator<Item = usize>>(&mut self, positions: I) -> usize {
        let mut held = Bitmap::default();
        let mut count = 0;
        let mut extent = 0;
        for at in positions {
            held.insert(at);
            count += 1;
            extent = extent.max(at + 1);
        }
        let mut node_at = vec![NONE; extent];
        let mut placed = 0;
        let mut every = true;
        for (node, record) in self.graph.records_mut().iter_mut().enumerate() {
            let at = *record as usize;
            if *record != NONE && held.contains(at) && node_at[at] == NONE {
                node_at[at] = node as u32;
                placed += 1;
            } else {
                *record = NONE;
                every = false;
            }
        }
        self.node_at = node_at;
        self.live_count = placed;
        self.every_node_live = every;
        count - placed
    }

    /// The index's half of a close-up, where the graph stays: the node of
    /// the record at `old_of_new[k]` moves to `k` in `node_at`, and the graph
    /// files that node's record at `k`. `old_of_new` lists the positions held
    /// in increasing order. A stranded node keeps the empty marker, so no
    /// node changes and `every_node_live` holds what it held. `node_at` gives
    /// back what it holds past the records or the reservation the graph
    /// holds for its nodes, whichever is more. The graph's positions are
    /// written through one borrow of them rather than a call a node.
    ///
    /// The close-up a collection runs is [`DenseIndex::closed_up`] and
    /// [`DenseIndex::swap_parts`]. This is the same pass made in place,
    /// kept for the tests that hold the two to the same result.
    #[cfg(test)]
    pub(crate) fn close_up(&mut self, old_of_new: &[u32]) {
        let records = self.graph.records_mut();
        for (k, &old) in old_of_new.iter().enumerate() {
            let node = self.node_at[old as usize];
            self.node_at[k] = node;
            records[node as usize] = k as u32;
        }
        self.node_at.truncate(old_of_new.len());
        self.node_at
            .shrink_to(old_of_new.len().max(reserved_by(&self.graph)));
    }

    /// The index's half of a close-up where the graph stays, built from the
    /// index and leaving it as it is, so a search goes on traversing it
    /// while the parts are made.
    ///
    /// `node_at` with the node of the record at `old_of_new[k]` at `k`, and
    /// a copy of the graph's position for each node with that node's record
    /// filed at `k`. A stranded node keeps the empty marker, so no node
    /// changes and the live count and `every_node_live` hold what they hold.
    /// The copy of the positions by node keeps the capacity the graph's
    /// holds. `node_at` keeps what it holds, or the records or the
    /// reservation a graph of this kind declared at `expected_size` takes
    /// for its nodes, whichever is more, where that is less, which is the
    /// reservation the map from an id to its node held for the declared
    /// size, one slot less, since positions count from zero.
    pub(crate) fn closed_up(&self, old_of_new: &[u32], expected_size: usize) -> DenseParts {
        let kept = old_of_new
            .len()
            .max(self.graph.declared_records(expected_size));
        let mut node_at = Vec::with_capacity(self.node_at.capacity().min(kept));
        let mut records = Vec::with_capacity(self.graph.record_capacity());
        records.extend_from_slice(self.graph.records());
        for (k, &old) in old_of_new.iter().enumerate() {
            let node = self.node_at[old as usize];
            node_at.push(node);
            records[node as usize] = k as u32;
        }
        DenseParts::Numbering { node_at, records }
    }

    /// The index's half of a close-up that installs `graph`, built off to
    /// the side: the replacement with `node_at`, the live count and the flag
    /// the unfiltered search reads, read from its own numbering as
    /// [`DenseIndex::replace_graph`] reads them, and the units `timed`
    /// gives, which [`DenseIndex::time_graph`] measured on the replacement.
    pub(crate) fn replaced(&self, graph: VectorGraph, timed: Option<(f64, f64)>) -> DenseParts {
        let (node_at, live_count, every_node_live) = numbering_of(&graph);
        let units = units_of(&graph, self.pq.as_ref(), self.dim, timed);
        DenseParts::Graph {
            graph: Box::new(graph),
            node_at,
            live_count,
            every_node_live,
            units,
        }
    }

    /// Trade this index's parts for `parts`, which
    /// [`DenseIndex::closed_up`] or [`DenseIndex::replaced`] built from it,
    /// so the index holds the new parts and `parts` its old ones, the old
    /// graph among them where a graph was replaced. Moves alone, so a
    /// caller holding the index's write guard holds it for the swap and
    /// nothing else, and drops the old parts with no guard held.
    pub(crate) fn swap_parts(&mut self, parts: &mut DenseParts) {
        match parts {
            DenseParts::Numbering { node_at, records } => {
                std::mem::swap(&mut self.node_at, node_at);
                self.graph.swap_records(records);
            }
            DenseParts::Graph {
                graph,
                node_at,
                live_count,
                every_node_live,
                units,
            } => {
                std::mem::swap(&mut self.graph, graph.as_mut());
                std::mem::swap(&mut self.node_at, node_at);
                std::mem::swap(&mut self.live_count, live_count);
                std::mem::swap(&mut self.every_node_live, every_node_live);
                std::mem::swap(&mut self.units, units);
            }
        }
    }

    /// The numbering and its capacity, the counts the index keeps and the
    /// units, for a test that holds two indexes to be the same numbering.
    #[cfg(test)]
    pub(crate) fn layout(&self) -> Vec<(&'static str, usize, String)> {
        vec![
            (
                "node_at",
                self.node_at.capacity(),
                format!("{:?}", self.node_at),
            ),
            (
                "records",
                self.graph.record_capacity(),
                format!("{:?}", self.graph.records()),
            ),
            ("nodes", 0, self.graph.nb_points().to_string()),
            ("live_count", 0, self.live_count.to_string()),
            ("every_node_live", 0, self.every_node_live.to_string()),
            ("units", 0, format!("{:?}", self.units)),
        ]
    }

    /// Swap the graph and keep the records, which is what a compaction, a
    /// rebuild and the training transition do.
    ///
    /// The replacement was built from the records the collection holds, each
    /// node filed at its record's position, so `node_at` and the live count
    /// are read back from the replacement's own numbering in the same pass
    /// that decides whether every node holds a record. `node_at` is built
    /// anew at the extent the records reach, with the reservation the
    /// replacement holds for its nodes, as the map from an id to its node was
    /// built anew with the graph it belonged to: a training rebuild keeps the
    /// declared size, and a compaction, whose graph is shrunk first, keeps
    /// the extent alone. `timed` is what [`DenseIndex::time_graph`]
    /// measured on the replacement before the caller took the write guard, so
    /// the guard is held for the swap and one pass over the replacement's
    /// nodes, and never across a timing run. `None` takes the floor.
    pub(crate) fn replace_graph(
        &mut self,
        graph: VectorGraph,
        timed: Option<(f64, f64)>,
    ) -> VectorGraph {
        let old = std::mem::replace(&mut self.graph, graph);
        let (node_at, placed, every) = numbering_of(&self.graph);
        self.node_at = node_at;
        self.live_count = placed;
        self.every_node_live = every;
        self.set_units(timed);
        old
    }

    /// Whether the two directions of the numbering agree: every position
    /// held names a node that names it back, the nodes of the positions held
    /// carry internal ids rising with their positions, every other node is
    /// stranded, the live count is the positions held, and the flag the
    /// unfiltered search reads is true exactly where no node is stranded.
    pub(crate) fn numbering_agrees(&self) -> bool {
        let records = self.graph.records();
        let mut held = 0;
        let mut last_id: Option<usize> = None;
        for (at, &node) in self.node_at.iter().enumerate() {
            if node == NONE {
                continue;
            }
            held += 1;
            if records.get(node as usize) != Some(&(at as u32)) {
                return false;
            }
            let id = self.graph.origin_id_at(node);
            if last_id.is_some_and(|last| id <= last) {
                return false;
            }
            last_id = Some(id);
        }
        let filed = records.iter().filter(|&&position| position != NONE).count();
        let every = filed == records.len();
        let back = records.iter().enumerate().all(|(node, &position)| {
            position == NONE || self.node_at.get(position as usize) == Some(&(node as u32))
        });
        back && held == self.live_count && filed == held && every == self.every_node_live
    }

    /// Time the two units on a graph that is not yet installed, so a
    /// replacement is measured with no guard held. The kernel over records
    /// scattered across the store, and a whole search per unit of `ef`,
    /// both under a bitmap test admitting every node, which is the
    /// predicate's cost without its outcome.
    pub(crate) fn time_graph(graph: &VectorGraph) -> Option<(f64, f64)> {
        let nodes = graph.nb_points();
        if nodes < CALIBRATION_MIN_NODES {
            return None;
        }
        let mut every = Bitmap::with_slots(nodes);
        for node in 0..nodes {
            every.insert(node);
        }
        let admits = |id: usize| every.contains(id);
        let distance_ns = graph
            .time_distance_ns(CALIBRATION_EVALUATIONS, CALIBRATION_ROUNDS, &admits)
            .filter(|ns| ns.is_finite() && *ns > 0.0)?;
        let search_ns = graph
            .time_search_ns(
                CALIBRATION_SEARCH_K,
                CALIBRATION_SEARCH_EF,
                CALIBRATION_SEARCHES,
                &admits,
            )
            .filter(|ns| ns.is_finite() && *ns > 0.0)?;
        Some((distance_ns, search_ns / CALIBRATION_SEARCH_EF as f64))
    }

    /// Install a quantizer, which the loader does before it restores a
    /// quantized graph.
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<PQ>>) {
        self.pq = pq;
    }

    pub(crate) fn set_keeps_raw(&mut self, keeps_raw: bool) {
        self.keeps_raw = keeps_raw;
    }

    /// The two units this index prices a search with, and whether they were
    /// timed or taken from the floor.
    #[cfg(test)]
    pub(crate) fn units(&self) -> DenseUnits {
        self.units
    }

    /// Time the two units on the graph's own store, or take the floor where
    /// the store is too small.
    ///
    /// The kernel's floor is a line through two measured points on a raw
    /// graph, 174 nanoseconds at width 100 and 1,026 at width 1,536, and one
    /// point on a quantized graph, which reads one table entry per
    /// subvector. The traversal's floor is that figure times the visits one
    /// unit of `ef` was measured to cost. Both are assumptions about this
    /// machine that a timed figure replaces.
    pub(crate) fn calibrate(&mut self) {
        let timed = Self::time_graph(&self.graph);
        self.set_units(timed);
    }

    /// Whether the index has just reached a size at which the units are
    /// timed again, being each doubling of the live count from the smallest
    /// graph worth timing. The collection times the graph under a read
    /// guard and then adopts the figures under the write guard, so no
    /// search waits on a timing run.
    pub(crate) fn due_for_timing(&self) -> bool {
        self.live_count >= CALIBRATION_MIN_NODES && self.live_count.is_power_of_two()
    }

    /// Keep the timed units, or the floor where there are none.
    pub(crate) fn set_units(&mut self, timed: Option<(f64, f64)>) {
        self.units = units_of(&self.graph, self.pq.as_ref(), self.dim, timed);
    }

    /// The traversal width a search takes when the caller names none.
    fn default_ef(&self, k: usize) -> usize {
        match self.metric.as_str() {
            "l1" | "l2" => std::cmp::max(2 * k, 150),
            _ => std::cmp::max(2 * k, 100),
        }
    }

    /// The distance from `query` to the record one node holds, on the raw
    /// scale.
    ///
    /// The stored raw vector where the graph holds one and the reconstruction
    /// from the record's codes where it does not, normalised first where the
    /// space requires it. A record holding neither cannot exist, since every
    /// insertion writes one or the other, and is sorted last.
    fn exact_distance_at(
        &self,
        query: &[f32],
        node: u32,
        distance: fn(&[f32], &[f32]) -> f32,
        needs_unit: bool,
    ) -> f32 {
        if let Some(stored) = self.graph.raw_vector_at(node) {
            return distance(query, stored);
        }
        // A scalar row is scored through the graph's own kernel, so the exact
        // scan and the traversal agree bit for bit and the page is on the
        // scale the kernel already puts it on.
        if let Some(scored) = self.graph.int8_distance_at(query, node) {
            return scored;
        }
        let reconstructed = self
            .graph
            .codes_at(node)
            .and_then(|codes| self.pq.as_ref()?.reconstruct(codes).ok());
        match reconstructed {
            Some(reconstructed) => {
                distance(query, &prepare_reconstruction(needs_unit, reconstructed))
            }
            None => f32::INFINITY,
        }
    }

    /// Score every enumerated position the index holds and cut the page.
    ///
    /// The positions arrive in increasing order. Each reads its node from
    /// `node_at`, skips a position that holds no record, and is scored from
    /// the node's vector, scalar row or codes.
    fn score_exact(
        &self,
        query: &[f32],
        positions: &[Position],
        k: usize,
        boundary_ties: bool,
    ) -> Hits {
        let distance = raw_distance_fn(&self.metric);
        let needs_unit = reconstruction_needs_unit(&self.metric);
        let node_at = &self.node_at;
        let mut scored: Vec<(f32, u32)> = positions
            .iter()
            .filter_map(|at| match node_at.get(at.slot()) {
                Some(&node) if node != NONE => Some((
                    self.exact_distance_at(query, node, distance, needs_unit),
                    at.0,
                )),
                _ => None,
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut end = k.min(scored.len());
        if boundary_ties && end > 0 && end < scored.len() {
            let boundary = scored[end - 1].0;
            while end < scored.len() && scored[end].0 == boundary {
                end += 1;
            }
        }
        scored.truncate(end);
        Hits {
            items: scored
                .into_iter()
                .map(|(score, at)| Hit {
                    position: Position(at),
                    score,
                })
                .collect(),
            kind: ScoreKind::Distance,
            exact: true,
        }
    }

    /// Run the traversal under one predicate.
    ///
    /// Three predicates, each monomorphised into the traversal by the
    /// graph's generic parameter, and each asked about the position the
    /// traversal reads for a candidate, which is the empty marker for a
    /// stranded node. A set admitting everything is the test of that marker,
    /// which is the unfiltered search, and no predicate at all while every
    /// node the graph holds holds a record. A bitmap is its bit alone, since
    /// the marker lies past every word. Anything else is the marker's test and
    /// then the table call. The marker is a constant in each closure, so a
    /// closure captures the admit set it asks and nothing else.
    fn traverse(&self, query: &[f32], k: usize, ef: usize, admit: &dyn Admit) -> Hits {
        let quantized = self.graph.is_quantized();
        let operation = if quantized {
            "adc_search"
        } else {
            "raw_search"
        };
        let searched = if admit.admits_all() {
            if self.every_node_live {
                // Every node the traversal can reach holds a record, so the
                // search runs without a predicate.
                #[cfg(test)]
                PREDICATE_BYTES.with(|bytes| bytes.set((1, 0)));
                self.graph.search(query, k, ef, None::<&fn(&usize) -> bool>)
            } else {
                // The marker is a local the closure borrows, so the closure
                // holds one reference.
                let none = NONE as usize;
                let admits = |at: &usize| *at != none;
                #[cfg(test)]
                PREDICATE_BYTES.with(|bytes| bytes.set((2, std::mem::size_of_val(&admits))));
                self.graph.search(query, k, ef, Some(&admits))
            }
        } else if let Some(bitmap) = admit.as_bitmap() {
            let admits = |at: &usize| bitmap.contains(*at);
            #[cfg(test)]
            PREDICATE_BYTES.with(|bytes| bytes.set((3, std::mem::size_of_val(&admits))));
            self.graph.search(query, k, ef, Some(&admits))
        } else {
            let admits =
                |at: &usize| *at != NONE as usize && admit.admits(Position::from_slot(*at));
            #[cfg(test)]
            PREDICATE_BYTES.with(|bytes| bytes.set((4, std::mem::size_of_val(&admits))));
            self.graph.search(query, k, ef, Some(&admits))
        };
        let graph_hits = searched.unwrap_or_else(|e| {
            error!(target: LOG_TARGET, operation = operation, error = %e, "Graph search failed");
            Vec::new()
        });
        let mut items: Vec<Hit> = graph_hits
            .into_iter()
            .map(|hit| Hit {
                position: Position::from_slot(hit.position),
                score: hit.distance,
            })
            .collect();
        // Put a product quantized l2 index's approximate scores back on the
        // scale it reports. That scorer sums a table of squared distances and
        // takes no root, because the root is monotone and would cost one per
        // evaluation to change nothing, so the root is taken once per
        // returned candidate here. Cosine arrives already on its own scale,
        // since its conversion is not monotone in the sum and the scorer has
        // to make it. A scalar l2 scorer takes the root inside the kernel,
        // reproducing the raw kernel over the decoded values, so its page is
        // already on that scale and a second root here would move every
        // score off it; the condition is the product quantized graph alone.
        if self.graph.is_pq() && self.metric == "l2" {
            for item in &mut items {
                item.score = item.score.max(0.0).sqrt();
            }
        }
        Hits {
            items,
            kind: ScoreKind::Distance,
            exact: false,
        }
    }

    /// The traversal under `admit`, whatever the admit set's size, for a
    /// test that compares each predicate's page with a reference.
    #[cfg(test)]
    pub(crate) fn traverse_under(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        admit: &dyn Admit,
    ) -> Hits {
        self.traverse(query, k, ef, admit)
    }

    /// The page this graph answers under a test of each candidate's internal
    /// id, read through the nodes' origin ids, as `(internal id, score)`.
    ///
    /// For the traversal every node names its internal id where it names
    /// its position, and the positions are put back after it, so the
    /// traversal is the one the graph ran before records had positions and
    /// `admits` is asked about internal ids alone, stranded nodes included.
    /// The scores are put on the scale `traverse` reports.
    #[cfg(test)]
    pub(crate) fn traverse_by_id(
        &mut self,
        query: &[f32],
        k: usize,
        ef: usize,
        admits: &dyn Fn(usize) -> bool,
    ) -> Vec<(usize, f32)> {
        let positions = self.graph.records().to_vec();
        let ids: Vec<u32> = (0..positions.len() as u32)
            .map(|node| self.graph.origin_id_at(node) as u32)
            .collect();
        self.graph.records_mut().copy_from_slice(&ids);
        let hits = self
            .graph
            .search(query, k, ef, Some(&|id: &usize| admits(*id)))
            .unwrap();
        self.graph.records_mut().copy_from_slice(&positions);
        let rooted = self.graph.is_pq() && self.metric == "l2";
        hits.into_iter()
            .map(|hit| {
                let score = if rooted {
                    hit.distance.max(0.0).sqrt()
                } else {
                    hit.distance
                };
                (hit.position, score)
            })
            .collect()
    }

    /// The score the exact scan gives the record one node holds.
    #[cfg(test)]
    pub(crate) fn exact_distance_of_node(&self, query: &[f32], node: u32) -> f32 {
        self.exact_distance_at(
            query,
            node,
            raw_distance_fn(&self.metric),
            reconstruction_needs_unit(&self.metric),
        )
    }

    /// Draw the level the next record's node takes, advancing the graph's
    /// level stream by one draw. See `VectorGraph::draw_level`.
    pub(crate) fn draw_level(&self) -> usize {
        self.graph.draw_level()
    }

    /// Phase one for a vector at `level`, whatever the graph holds. Quantizes
    /// first where the graph holds codes, and draws nothing.
    fn plan_at_level(
        &self,
        id: RecordId,
        vector: &[f32],
        level: usize,
    ) -> Result<DensePlan, Error> {
        if self.graph.is_int8() {
            let row = self.graph.int8_encode(vector).map_err(|e| {
                error!(target: "zeusdb_vector_database::hnsw_index::insert", operation = "add_quantized_vector",
                    internal_id = id.0,
                    error = %e,
                    "Failed to encode vector"
                );
                Error::QuantizeFailed(e)
            })?;
            if let Some(codec) = self.graph.int8_codec() {
                let clipped = codec.saturated(vector);
                if clipped > 0 {
                    self.saturated.fetch_add(clipped as u64, Ordering::Relaxed);
                }
            }
            let planned = self.graph.plan_at_level(Record::Int8(&row), level);
            return Ok(DensePlan {
                codes: None,
                row: Some(row),
                planned,
            });
        }
        if self.graph.is_quantized() {
            let pq = self.pq.as_ref().ok_or(Error::NoQuantizer)?;
            let codes = pq.quantize(vector).map_err(|e| {
                error!(target: "zeusdb_vector_database::hnsw_index::insert", operation = "add_quantized_vector",
                    internal_id = id.0,
                    error = %e,
                    "Failed to quantize vector"
                );
                Error::QuantizeFailed(e)
            })?;
            let planned = self.graph.plan_at_level(
                Record::Codes {
                    codes: &codes,
                    raw: self.keeps_raw.then_some(vector),
                },
                level,
            );
            Ok(DensePlan {
                codes: Some(codes),
                row: None,
                planned,
            })
        } else {
            Ok(DensePlan {
                codes: None,
                row: None,
                planned: self.graph.plan_at_level(Record::Raw(vector), level),
            })
        }
    }

    /// Phase one at a level the caller drew.
    ///
    /// The collection draws the level through [`DenseIndex::draw_level`]
    /// before it writes anything for the record, so the level is in hand
    /// before the insertion begins, and hands it here under the read guard it
    /// takes for the plan. The seam's own `prepare` is for a caller holding no
    /// level of its own.
    pub(crate) fn prepare_at_level(
        &self,
        id: RecordId,
        vector: &[f32],
        level: usize,
    ) -> Result<Prepared, Error> {
        Ok(Prepared::new(self.plan_at_level(id, vector, level)?))
    }
}

impl VectorIndex<Dense> for DenseIndex {
    fn len(&self) -> usize {
        self.live_count
    }

    fn holds(&self, at: Position) -> bool {
        self.node_of(at).is_some()
    }

    /// Phase one for a caller holding no level of its own: draws one and
    /// plans at it under the read guard, quantizing first where the graph
    /// holds codes. The collection draws ahead of this and calls
    /// [`DenseIndex::prepare_at_level`] instead.
    fn prepare(&self, id: RecordId, vector: &[f32]) -> Result<Prepared, Error> {
        let level = self.graph.draw_level();
        self.prepare_at_level(id, vector, level)
    }

    /// Phase two. Appends the node, files the record's position under it
    /// and the node under the position, and installs its lists under the
    /// write guard. A caller that skipped `prepare` has the draw and both
    /// phases run here.
    ///
    /// A plan the graph refused, because the record's element type is not
    /// the graph's, is a refusal: nothing is installed and the record is not
    /// held, so every record the index holds has a node. The graph logs the
    /// cause. No caller reaches it, since the plan takes the record's form
    /// from the graph.
    fn insert(
        &mut self,
        at: Position,
        id: RecordId,
        vector: &[f32],
        prepared: Prepared,
    ) -> Result<(), Error> {
        if self.holds(at) {
            return Err(Error::RecordAlreadyHeld { id: at.0 });
        }
        let plan = match prepared.take::<DensePlan>() {
            Some(plan) => plan,
            None => {
                let level = self.graph.draw_level();
                self.plan_at_level(id, vector, level)?
            }
        };
        let Some(planned) = plan.planned else {
            return Err(Error::Engine(format!(
                "the graph refused the record with internal id {}, whose form is not the graph's",
                id.0
            )));
        };
        let record = match (&plan.codes, &plan.row) {
            (Some(codes), _) => Record::Codes {
                codes,
                raw: self.keeps_raw.then_some(vector),
            },
            (None, Some(row)) => Record::Int8(row),
            (None, None) => Record::Raw(vector),
        };
        let node = self.graph.nb_points() as u32;
        self.graph.install(record, id.slot(), at, planned);
        // A record takes the next position, which `node_at` reaches, but
        // after a rebuild whose last positions were holes it reaches only the
        // last record held.
        let slot = at.slot();
        if slot == self.node_at.len() {
            self.node_at.push(node);
        } else {
            if slot > self.node_at.len() {
                self.node_at.resize(slot + 1, NONE);
            }
            self.node_at[slot] = node;
        }
        self.live_count += 1;
        Ok(())
    }

    /// Strands the node. The graph keeps it, since a stranded node still
    /// routes a traversal, and `compact` rebuilds the graph without it. The
    /// position and the node both take the empty marker.
    fn remove(&mut self, at: Position) -> Result<(), Error> {
        if self.strand(at) == NONE {
            return Err(Error::RecordNotHeld { id: at.0 });
        }
        Ok(())
    }

    /// Nodes the graph holds beyond the live records, which is what
    /// removal and overwrite strand and what a compaction reclaims.
    fn stranded(&self) -> usize {
        self.graph.nb_points().saturating_sub(self.live_count)
    }

    fn vector(&self, at: Position) -> Option<&[f32]> {
        self.graph.raw_vector_at(self.node_of(at)?)
    }

    /// The raw vector where the graph holds one and the reconstruction from
    /// the record's codes where it does not.
    fn recover(&self, at: Position) -> Option<Vec<f32>> {
        let node = self.node_of(at)?;
        if let Some(raw) = self.graph.raw_vector_at(node) {
            return Some(raw.to_vec());
        }
        if let Some(decoded) = self.graph.int8_reconstruct_at(node) {
            return Some(decoded);
        }
        let codes = self.graph.codes_at(node)?;
        self.pq.as_ref()?.reconstruct(codes).ok()
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        admit: &dyn Admit,
        budget: &Budget,
    ) -> Result<Hits, Error> {
        let ef = budget.ef.unwrap_or_else(|| self.default_ef(k));
        if let Some(admitted) = admit.len_hint() {
            if admitted <= FULL_SCAN_THRESHOLD {
                let mut positions = Vec::with_capacity(admitted);
                if admit.enumerate(&mut |at| {
                    positions.push(at);
                    true
                }) {
                    return Ok(self.score_exact(query, &positions, k, budget.boundary_ties));
                }
            }
        }
        Ok(self.traverse(query, k, ef, admit))
    }

    /// Priced from `ef` and the admitted share, and the query is ignored,
    /// since a traversal's work does not depend on which vector is asked.
    fn cost(&self, _query: &[f32], k: usize, admitted: Option<&Selectivity>) -> Cost {
        let ef = self.default_ef(k) as f64;
        match admitted {
            Some(selectivity) if (selectivity.expected as usize) <= FULL_SCAN_THRESHOLD => Cost {
                work_ns: selectivity.expected as f64 * self.units.distance_ns,
                exact: true,
            },
            Some(selectivity) => {
                let live = self.live_count.max(1) as f64;
                let share = (selectivity.expected as f64 / live).clamp(1e-3, 1.0);
                Cost {
                    work_ns: ef * self.units.ef_ns / share.powf(SELECTIVITY_EXPONENT),
                    exact: false,
                }
            }
            None => Cost {
                work_ns: ef * self.units.ef_ns,
                exact: false,
            },
        }
    }
}

/// The positions an index over `graph` reserves, being the nodes the graph
/// reserved for, since positions count from zero and every position names a
/// node. Nothing where the graph already holds a node, since the records are
/// filed from the graph then.
fn reserved_positions(graph: &VectorGraph) -> usize {
    if graph.nb_points() == 0 {
        reserved_by(graph)
    } else {
        0
    }
}

/// The positions the reservation `graph` holds for its nodes covers, being
/// one a node.
fn reserved_by(graph: &VectorGraph) -> usize {
    graph.record_capacity()
}

/// The units a search over `graph` is priced with: the timed ones, or the
/// floor where there are none.
fn units_of(
    graph: &VectorGraph,
    pq: Option<&Arc<PQ>>,
    dim: usize,
    timed: Option<(f64, f64)>,
) -> DenseUnits {
    match timed {
        Some((distance_ns, ef_ns)) => DenseUnits {
            distance_ns,
            ef_ns,
            measured: true,
        },
        None => {
            // A scalar kernel decodes every value inside the distance and
            // measured no slower than the raw kernel at every width, so it
            // takes the raw floor.
            let distance_ns = if graph.is_pq() {
                10.0 + 1.5 * pq.map_or(8, |pq| pq.subvectors()) as f64
            } else {
                115.0 + 0.6 * dim as f64
            };
            DenseUnits {
                distance_ns,
                ef_ns: distance_ns * FLOOR_VISITS_PER_EF,
                measured: false,
            }
        }
    }
}

/// `node_at`, the live count and whether every node holds a record, for a
/// graph built from the records a collection holds, each node filed at its
/// record's position. `node_at` is built at the extent the records reach,
/// with the reservation the graph holds for its nodes, as the map from an id
/// to its node was built anew with the graph it belonged to.
fn numbering_of(graph: &VectorGraph) -> (Vec<u32>, usize, bool) {
    let records = graph.records();
    let extent = records
        .iter()
        .filter(|&&position| position != NONE)
        .map(|&position| position as usize + 1)
        .max()
        .unwrap_or(0);
    let mut node_at = Vec::with_capacity(extent.max(reserved_by(graph)));
    node_at.resize(extent, NONE);
    let mut placed = 0;
    let mut every = true;
    for (node, &position) in records.iter().enumerate() {
        if position == NONE {
            every = false;
            continue;
        }
        node_at[position as usize] = node as u32;
        placed += 1;
    }
    (node_at, placed, every)
}

/// The dense index's half of a close-up built off to the side, which
/// [`DenseIndex::swap_parts`] trades for the index's own.
pub(crate) enum DenseParts {
    /// The graph keeps its nodes: `node_at` and the graph's position for
    /// each node.
    Numbering {
        node_at: Vec<u32>,
        records: Vec<u32>,
    },
    /// A replacement graph, with what the index reads from it. The graph is
    /// boxed, so the parts move a pointer to it rather than its struct.
    Graph {
        graph: Box<VectorGraph>,
        node_at: Vec<u32>,
        live_count: usize,
        every_node_live: bool,
        units: DenseUnits,
    },
}

/// Whether every node `graph` holds holds a record at a position `node_at`
/// files it under.
fn every_node_live(graph: &VectorGraph, node_at: &[u32]) -> bool {
    graph
        .records()
        .iter()
        .enumerate()
        .all(|(node, &at)| at != NONE && node_at.get(at as usize) == Some(&(node as u32)))
}

/// The dump's name under a prefix.
fn dump_name(prefix: &str) -> String {
    format!("{prefix}{DUMP_FILENAME}")
}

impl Persist for DenseIndex {
    /// Write the graph dump under `prefix` and record its length.
    ///
    /// Its length rather than a digest, because the dump streams itself and
    /// seeks back to write its header, so hashing it whole would mean reading
    /// the largest artefact in the directory back off the disk for a
    /// guarantee its own two checksums already give.
    fn write(&self, prefix: &str, dir: &dyn Dir, ledger: &mut dyn Ledger) -> Result<(), Error> {
        let name = dump_name(prefix);
        self.graph
            .dump(dir, &name)
            .map_err(Error::GraphDumpFailed)?;
        let bytes = dir
            .length(&name)
            .map_err(|e| Error::DumpLengthUnreadable(e.to_string()))?;
        ledger.record(
            &name,
            ArtefactRecord {
                bytes,
                checksum: None,
            },
        );
        Ok(())
    }

    fn artefact_names(&self, prefix: &str) -> Vec<String> {
        vec![dump_name(prefix)]
    }
}

/// What the loader knows before it opens a dense index from its dump.
pub(crate) struct DenseOpen {
    pub(crate) metric: String,
    pub(crate) dim: usize,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    /// The trained quantizer, where the saved graph was a quantized one.
    pub(crate) pq: Option<Arc<PQ>>,
    /// The fitted scalar codec, where the saved graph was a scalar one.
    pub(crate) int8: Option<Arc<Int8Codec>>,
    pub(crate) keeps_raw: bool,
}

impl Restore for DenseIndex {
    type Config = DenseOpen;

    /// Restore the graph the save wrote.
    ///
    /// The inventory's recorded length is checked before the dump is read,
    /// because a self-consistent dump from a different save of an index of
    /// the same shape would otherwise pass the dump's own checksums. Every
    /// error is a reason to rebuild rather than to fail, which the caller
    /// decides. The reader numbers the nodes against the mappings the bounds
    /// carry, so each node names its record's position or is stranded, and a
    /// dump naming no node for a record the mappings hold is refused. The
    /// index comes back holding no record and the caller files the records at
    /// those positions; see [`DenseIndex::set_live`].
    fn restore(
        config: &DenseOpen,
        prefix: &str,
        dir: &dyn Dir,
        inventory: &dyn Inventory,
        bounds: &Bounds<'_>,
    ) -> Result<Self, Error> {
        let name = dump_name(prefix);
        if let Some(recorded) = inventory.recorded(&name) {
            let found = dir.length(&name).unwrap_or(0);
            if found != recorded.bytes {
                return Err(Error::Engine(format!(
                    "the graph dump is {} bytes and manifest.json records it as {}",
                    found, recorded.bytes
                )));
            }
        }
        let dump_bounds = DumpBounds {
            min_nodes: bounds.min_records,
            max_origin_id: bounds.max_records,
            positions: Some(bounds.ids_by_position),
        };
        let (graph, _nodes) = match &config.int8 {
            Some(codec) => restore_int8_graph(
                dir,
                &name,
                &config.metric,
                config.m,
                config.ef_construction,
                codec.clone(),
                dump_bounds,
            ),
            None => {
                let pq = config.pq.as_ref().filter(|pq| pq.is_trained()).cloned();
                restore_graph(
                    dir,
                    &name,
                    &config.metric,
                    config.m,
                    config.ef_construction,
                    config.dim,
                    pq,
                    dump_bounds,
                )
            }
        }
        .map_err(Error::Engine)?;
        Ok(DenseIndex::new(
            graph,
            &config.metric,
            config.dim,
            config.pq.clone(),
            config.keeps_raw,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeusdb_vector_core::{And, Candidates, NB_LAYER_MAX};

    /// Insert one record at the position its internal id names.
    fn put(index: &mut DenseIndex, id: u32, vector: &[f32]) {
        let prepared = index.prepare(RecordId(id), vector).unwrap();
        index
            .insert(Position(id), RecordId(id), vector, prepared)
            .unwrap();
    }

    fn index_of(vectors: &[Vec<f32>]) -> DenseIndex {
        let graph = VectorGraph::new_raw("l2", 2, 4, 64, NB_LAYER_MAX as usize, 50);
        let mut index = DenseIndex::new(graph, "l2", 2, None, false);
        for (i, v) in vectors.iter().enumerate() {
            put(&mut index, i as u32 + 1, v);
        }
        index
    }

    /// The case each traversal ran and the bytes its predicate closure
    /// captures, for a set admitting everything with every node holding a
    /// record, the same once a node is stranded, a bitmap, and any other set.
    /// `CEILING` is what the four cases captured when each closure carried
    /// the live set by id beside its admit set, and `HERE` is what each
    /// captures, the closure for stranded nodes borrowing the marker and the
    /// closure for any other set naming it as a constant. A closure's
    /// captures are held across the whole traversal it is monomorphised
    /// into, so none may capture more than the ceiling in its case.
    #[test]
    fn each_predicate_captures_no_more_than_its_ceiling() {
        const CEILING: [(u8, usize); 4] = [(1, 0), (2, 8), (3, 16), (4, 24)];
        const HERE: [(u8, usize); 4] = [(1, 0), (2, 8), (3, 8), (4, 16)];
        let vectors: Vec<Vec<f32>> = (0..300)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let mut index = index_of(&vectors);
        let query = [0.1f32, 0.2];
        let measure = |index: &DenseIndex, admit: &dyn Admit| {
            PREDICATE_BYTES.with(|bytes| bytes.set((0, 0)));
            let _ = index.traverse(&query, 10, 40, admit);
            PREDICATE_BYTES.with(|bytes| bytes.get())
        };
        let every = measure(&index, &Candidates::All);
        index.remove(Position(7)).unwrap();
        let stranded = measure(&index, &Candidates::All);
        let mut bitmap = Bitmap::default();
        for at in 1..=300 {
            bitmap.insert(at);
        }
        let filtered = measure(&index, &bitmap);
        let sorted = Candidates::Sorted((1..=300u32).map(Position).collect());
        let other = measure(&index, &sorted);

        let measured = [every, stranded, filtered, other];
        assert_eq!(measured, HERE);
        for (case, (here, ceiling)) in measured.iter().zip(CEILING).enumerate() {
            assert_eq!(here.0, ceiling.0, "case {} ran", case + 1);
            assert!(
                here.1 <= ceiling.1,
                "case {} captures {} bytes, over its ceiling of {}",
                case + 1,
                here.1,
                ceiling.1
            );
        }
    }

    /// A held id is refused, neither an unknown id nor a hole can be
    /// removed, and the live and stranded counts follow every write.
    #[test]
    fn held_ids_are_refused_and_the_counts_follow_every_write() {
        let mut index = index_of(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(index.len(), 3);
        assert_eq!(index.stranded(), 0);
        assert!(matches!(
            index.insert(Position(2), RecordId(2), &[5.0, 5.0], Prepared::none()),
            Err(Error::RecordAlreadyHeld { id: 2 })
        ));
        assert!(matches!(
            index.remove(Position(9)),
            Err(Error::RecordNotHeld { id: 9 })
        ));
        index.remove(Position(2)).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.stranded(), 1);
        assert!(!index.holds(Position(2)));
        assert!(index.vector(Position(2)).is_none());
        assert_eq!(index.recover(Position(3)), Some(vec![0.0, 1.0]));
        // The node stays, so the graph still holds three.
        assert_eq!(index.graph().nb_points(), 3);
        // The hole the removal left is refused as a position never held is,
        // and the counts stay where they were.
        assert!(matches!(
            index.remove(Position(2)),
            Err(Error::RecordNotHeld { id: 2 })
        ));
        assert_eq!(index.len(), 2);
        assert_eq!(index.stranded(), 1);
    }

    /// A small enumerable admit set is scored exactly with the boundary tie
    /// group kept on request, and a set admitting everything traverses
    /// under the live set alone.
    #[test]
    fn a_small_admit_set_is_scored_exactly_and_everything_traverses() {
        let mut index = index_of(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![3.0, 0.0],
        ]);
        index.remove(Position(5)).unwrap();
        let query = [0.0f32, 0.0];

        let sorted = Candidates::Sorted(vec![Position(1), Position(2), Position(3), Position(5)]);
        let page = index
            .search(&query, 2, &sorted, &Budget::default())
            .unwrap();
        assert!(page.exact);
        let ids: Vec<u32> = page.items.iter().map(|h| h.position.0).collect();
        assert_eq!(ids, vec![1, 2]);

        let page = index
            .search(
                &query,
                2,
                &sorted,
                &Budget {
                    boundary_ties: true,
                    ..Budget::default()
                },
            )
            .unwrap();
        let ids: Vec<u32> = page.items.iter().map(|h| h.position.0).collect();
        assert_eq!(ids, vec![1, 2, 3], "the removed record 5 is never scored");

        let page = index
            .search(&query, 10, &Candidates::All, &Budget::default())
            .unwrap();
        assert!(!page.exact);
        let mut ids: Vec<u32> = page.items.iter().map(|h| h.position.0).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "the stranded node routes and is never returned"
        );
    }

    /// One page as positions and score bits.
    fn bits_of(hits: &Hits) -> Vec<(u32, u32)> {
        hits.items
            .iter()
            .map(|hit| (hit.position.0, hit.score.to_bits()))
            .collect()
    }

    /// The page a traversal returns under the live test, being whether the
    /// index holds the position, asked of the graph directly.
    fn under_the_live_bit(
        index: &DenseIndex,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<(u32, u32)> {
        index
            .graph()
            .search(
                query,
                k,
                ef,
                Some(&|at: &usize| index.holds(Position::from_slot(*at))),
            )
            .unwrap()
            .into_iter()
            .map(|hit| (hit.position as u32, hit.distance.to_bits()))
            .collect()
    }

    /// Whether any page of the graph asked with no predicate at all names a
    /// stranded node, which is a node whose record was removed.
    fn unchecked_page_names_a_stranded_node(index: &DenseIndex, queries: &[[f32; 2]]) -> bool {
        queries.iter().any(|query| {
            index
                .graph()
                .search(query, 10, 40, None::<&fn(&usize) -> bool>)
                .unwrap()
                .iter()
                .any(|hit| hit.position == NONE as usize)
        })
    }

    /// An unfiltered search leaves the live test out only while every node
    /// the graph holds holds a record, and returns the page the live test
    /// gives either way. Once a record is removed its node stays, the test
    /// is made again, and the removed record is never returned although a
    /// traversal with no predicate would reach its node.
    #[test]
    fn an_unfiltered_search_checks_the_live_bit_once_a_node_is_dead() {
        let vectors: Vec<Vec<f32>> = (0..300)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let queries: Vec<[f32; 2]> = (0..40)
            .map(|i| [(i as f32 * 0.53).cos(), (i as f32 * 0.29).sin()])
            .collect();
        let budget = Budget {
            ef: Some(40),
            ..Budget::default()
        };
        let mut index = index_of(&vectors);
        assert!(index.every_node_live, "built by insertion alone");
        for query in &queries {
            let page = index.search(query, 10, &Candidates::All, &budget).unwrap();
            assert_eq!(bits_of(&page), under_the_live_bit(&index, query, 10, 40));
        }

        let removed: Vec<u32> = (1..=300).filter(|id| id % 3 == 0).collect();
        for &id in &removed {
            index.remove(Position(id)).unwrap();
        }
        assert!(!index.every_node_live, "a removal leaves its node behind");
        assert!(
            unchecked_page_names_a_stranded_node(&index, &queries),
            "the removed records sit where these queries reach them"
        );
        for query in &queries {
            let page = index.search(query, 10, &Candidates::All, &budget).unwrap();
            assert!(page
                .items
                .iter()
                .all(|hit| !removed.contains(&hit.position.0)));
            assert_eq!(bits_of(&page), under_the_live_bit(&index, query, 10, 40));
        }
    }

    /// The flag is the fact the skip rests on, and it is computed whenever
    /// the graph or the numbering is replaced whole. A set of records naming
    /// every node but one, and one position no node carries, leaves a node
    /// stranded, reports the position no node holds, and the flag says a
    /// node is dead.
    #[test]
    fn every_node_live_is_recomputed_when_the_graph_or_the_live_set_is_replaced() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let mut index = index_of(&vectors);
        index.remove(Position(7)).unwrap();
        assert!(!index.every_node_live);

        // The same records handed back whole are read from the graph again,
        // and the removed record's node is still there.
        let held: Vec<usize> = (1..=100).filter(|&id| id != 7).collect();
        assert_eq!(index.set_live(held.iter().copied()), 0);
        assert!(!index.every_node_live);
        assert!(index.numbering_agrees());

        // As many records as nodes, one of them at a position no node holds.
        assert_eq!(
            index.set_live(held.iter().copied().chain(std::iter::once(500))),
            1,
            "position 500 has no node"
        );
        assert!(!index.every_node_live, "node 7's record is not held");
        assert!(index.numbering_agrees());

        // A graph holding the live records alone.
        index.set_live(held.iter().copied());
        let mut rebuilt = VectorGraph::new_raw("l2", 2, 4, 64, NB_LAYER_MAX as usize, 50);
        for &id in &held {
            rebuilt.insert(&vectors[id - 1], id, Position::from_slot(id));
        }
        index.replace_graph(rebuilt, None);
        assert!(index.every_node_live);
        assert!(index.numbering_agrees());
        let page = index
            .search(&[0.0, 0.0], 10, &Candidates::All, &Budget::default())
            .unwrap();
        assert_eq!(page.items.len(), 10);
        assert!(page.items.iter().all(|hit| hit.position.0 != 7));
    }

    /// The cost is priced in nanoseconds from the two units, is exact under
    /// a small admit set, and grows as the admitted share falls.
    #[test]
    fn cost_is_priced_from_the_units_and_the_admitted_share() {
        let index = index_of(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let units = index.units();
        assert!(!units.measured, "two nodes is below the calibration floor");
        assert!(units.distance_ns > 0.0 && units.ef_ns > units.distance_ns);
        let whole = index.cost(&[0.0, 0.0], 10, None);
        assert!(!whole.exact);
        assert!((whole.work_ns - 150.0 * units.ef_ns).abs() < 1e-6);
        let narrow = index.cost(&[0.0, 0.0], 10, Some(&Selectivity::exact(1)));
        assert!(narrow.exact);
        assert!((narrow.work_ns - units.distance_ns).abs() < 1e-6);
        let broad = index.cost(&[0.0, 0.0], 10, Some(&Selectivity::exact(6_000)));
        assert!(!broad.exact);
        assert!(broad.work_ns >= whole.work_ns);
    }

    /// A graph large enough to time reports measured units, and the
    /// traversal's unit is above the kernel's, since a traversal visits
    /// several nodes per unit of `ef`.
    #[test]
    fn a_large_enough_graph_times_both_units() {
        let vectors: Vec<Vec<f32>> = (0..400)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let mut index = index_of(&vectors);
        assert!(
            !index.units().measured,
            "built by insertion, so the floor until it is timed"
        );
        assert!(index.due_for_timing() || index.len() != 256);
        index.calibrate();
        let units = index.units();
        assert!(units.measured);
        assert!(units.distance_ns > 0.0 && units.distance_ns.is_finite());
        assert!(units.ef_ns > units.distance_ns);
    }

    /// An index over `vectors`, the record `k` at the position and internal
    /// id `1 + k * gap`, with every seventh record removed so its node is
    /// stranded.
    fn spread_index(
        graph: VectorGraph,
        pq: Option<Arc<PQ>>,
        keeps_raw: bool,
        vectors: &[Vec<f32>],
        gap: u32,
    ) -> DenseIndex {
        let dim = vectors[0].len();
        let mut index = DenseIndex::new(graph, "l2", dim, pq, keeps_raw);
        for (k, v) in vectors.iter().enumerate() {
            put(&mut index, 1 + k as u32 * gap, v);
        }
        for k in (0..vectors.len() as u32).step_by(7) {
            index.remove(Position(1 + k * gap)).unwrap();
        }
        index
    }

    /// The node a graph took for one internal id, found by a pass over its
    /// nodes rather than through the numbering under test.
    fn node_with_origin(graph: &VectorGraph, id: usize) -> Option<u32> {
        (0..graph.nb_points() as u32).find(|&node| graph.origin_id_at(node) == id)
    }

    /// Every position the index holds names a node that names it back, the
    /// removed and the overwritten records' nodes are stranded, and the
    /// numbering agrees after every write, a handing back of the records, a
    /// replacement of the graph, and an opening of a graph whole.
    #[test]
    fn the_numbering_follows_every_write_and_every_replacement() {
        let (count, dim) = (300usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count + 1, dim, 196);
        let layers = NB_LAYER_MAX as usize;
        let rebuilt_from = |index: &DenseIndex, vector_of: &dyn Fn(usize) -> usize| {
            let mut graph = VectorGraph::new_raw("l2", dim, 8, 64, layers, 50);
            for (at, &node) in index.node_at().iter().enumerate() {
                if node != NONE {
                    graph.insert(&vectors[vector_of(at)], at, Position::from_slot(at));
                }
            }
            graph
        };
        for gap in [1u32, 1_009] {
            let graph = VectorGraph::new_raw("l2", dim, 8, 64, layers, 50);
            let mut index = spread_index(graph, None, false, &vectors[..count], gap);
            assert!(
                !index.every_node_live && index.numbering_agrees(),
                "gap {gap}"
            );
            for k in 0..count as u32 {
                let id = 1 + k * gap;
                let node = node_with_origin(index.graph(), id as usize).unwrap();
                let held = k % 7 != 0;
                assert_eq!(index.holds(Position(id)), held);
                assert_eq!(index.node_of(Position(id)), held.then_some(node));
                let filed = if held { id } else { NONE };
                assert_eq!(index.graph().record_of_at(node), filed, "gap {gap} id {id}");
            }

            // An overwrite takes the record out at its position and puts it
            // back at a new one, and the old node stays, stranded.
            let old = 1 + 3 * gap;
            let new = 1 + count as u32 * gap;
            let stranded = index.node_of(Position(old)).unwrap();
            index.remove(Position(old)).unwrap();
            put(&mut index, new, &vectors[count]);
            let moved = index.node_of(Position(new)).unwrap();
            assert_eq!(index.graph().record_of_at(stranded), NONE);
            assert_eq!(index.graph().record_of_at(moved), new);
            assert_eq!(index.node_at()[old as usize], NONE);
            assert!(index.numbering_agrees(), "overwritten, gap {gap}");

            let vector_of = |at: usize| {
                if at == new as usize {
                    count
                } else {
                    (at - 1) / gap as usize
                }
            };
            let held: Vec<usize> = (0..index.node_at().len())
                .filter(|&at| index.node_at()[at] != NONE)
                .collect();
            assert_eq!(index.set_live(held.iter().copied()), 0);
            assert!(index.numbering_agrees(), "handed back, gap {gap}");
            assert_eq!(index.graph().record_of_at(stranded), NONE);
            assert_eq!(
                index.node_at_heap_bytes(),
                index.node_at().len() * 4,
                "gap {gap}"
            );

            index.replace_graph(rebuilt_from(&index, &vector_of), None);
            assert!(
                index.every_node_live && index.numbering_agrees(),
                "replaced, gap {gap}"
            );
            assert_eq!(index.len(), held.len());

            // A graph opened whole holds no record until the records are
            // handed over.
            let mut opened =
                DenseIndex::new(rebuilt_from(&index, &vector_of), "l2", dim, None, false);
            assert!(opened.is_empty() && !opened.every_node_live);
            assert!(held
                .iter()
                .all(|&at| !opened.holds(Position::from_slot(at))));
            assert_eq!(opened.set_live(held.iter().copied()), 0);
            assert!(opened.every_node_live && opened.numbering_agrees());
        }
    }

    /// A replacement graph hands its reservation to `node_at`, as a training
    /// rebuild's graph held the declared size for the map from an id to its
    /// node, and a shrunk replacement hands over the extent alone, as a
    /// compaction's did. Positions start at zero, so the declaration's
    /// records need no slot past it.
    #[test]
    fn a_replacement_hands_its_reservation_to_the_numbering() {
        let (count, dim, declared) = (40usize, 8usize, 500usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 202);
        let layers = NB_LAYER_MAX as usize;
        let rebuilt = || {
            let mut graph = VectorGraph::new_raw("l2", dim, 8, declared, layers, 50);
            for (k, vector) in vectors.iter().enumerate() {
                graph.insert(vector, k + 1, Position::from_slot(k));
            }
            graph
        };
        let graph = VectorGraph::new_raw("l2", dim, 8, declared, layers, 50);
        let mut index = DenseIndex::new(graph, "l2", dim, None, false);
        for (k, vector) in vectors.iter().enumerate() {
            let id = RecordId(k as u32 + 1);
            let prepared = index.prepare(id, vector).unwrap();
            index
                .insert(Position::from_slot(k), id, vector, prepared)
                .unwrap();
        }

        let graph = rebuilt();
        let reserved = graph.record_capacity();
        assert!(reserved >= declared, "the graph holds its declaration");
        index.replace_graph(graph, None);
        assert_eq!(index.node_at().len(), count);
        assert_eq!(index.node_at_heap_bytes(), reserved * 4);
        assert!(index.every_node_live && index.numbering_agrees());

        let mut shrunk = rebuilt();
        shrunk.shrink_to_fit();
        index.replace_graph(shrunk, None);
        assert_eq!(index.node_at_heap_bytes(), count * 4);
        assert!(index.every_node_live && index.numbering_agrees());
    }

    /// The index's own close-up, where the graph stays, moves the node of
    /// each record held to the record's new position in `node_at` and files
    /// the node there in the graph, in order, keeps every stranded node
    /// stranded and the flag that says so, and gives back what `node_at`
    /// held past the records or the graph's reservation. A traversal then
    /// answers what the same graph answers under the test by internal id.
    #[test]
    fn the_index_closes_up_its_numbering_where_the_graph_stays() {
        let (count, dim) = (300usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 204);
        let queries = zeusdb_vector_core::test_support::clustered(8, dim, 2_040);
        let graph = VectorGraph::new_raw("l2", dim, 8, 16, NB_LAYER_MAX as usize, 50);
        let mut index = DenseIndex::new(graph, "l2", dim, None, false);
        for (k, vector) in vectors.iter().enumerate() {
            let id = RecordId(k as u32 + 1);
            let prepared = index.prepare(id, vector).unwrap();
            index
                .insert(Position::from_slot(k), id, vector, prepared)
                .unwrap();
        }
        for k in (0..count).filter(|k| k % 4 != 2) {
            index.remove(Position::from_slot(k)).unwrap();
        }
        let kept: Vec<u32> = (0..count as u32).filter(|k| k % 4 == 2).collect();
        let capacity_before = index.node_at.capacity();
        index.close_up(&kept);

        assert_eq!(index.node_at().len(), kept.len());
        assert!(index.node_at.capacity() <= capacity_before);
        assert!(
            index.node_at.capacity() <= kept.len().max(reserved_by(index.graph())),
            "node_at keeps no more than the records or the graph's reservation"
        );
        assert!(index.numbering_agrees());
        assert!(!index.every_node_live, "the stranded nodes stay stranded");
        assert_eq!(index.len(), kept.len());
        assert_eq!(index.stranded(), count - kept.len());
        assert_eq!(index.graph().nb_points(), count, "no node moves");
        for (rank, &old) in kept.iter().enumerate() {
            let node = index.node_of(Position::from_slot(rank)).unwrap();
            assert_eq!(node, old, "the node is the one the record held");
            assert_eq!(index.graph().origin_id_at(node), old as usize + 1);
            assert_eq!(index.graph().record_of_at(node), rank as u32);
            assert_eq!(
                index.raw_vector(rank).unwrap(),
                vectors[old as usize].as_slice()
            );
        }
        assert!(index.node_of(Position::from_slot(kept.len())).is_none());

        let held: Vec<usize> = kept.iter().map(|&k| k as usize + 1).collect();
        for query in &queries {
            let page: Vec<(usize, u32)> = index
                .traverse_under(query, 10, 40, &Candidates::All)
                .items
                .iter()
                .map(|hit| {
                    (
                        index.internal_id_at(hit.position.slot()).unwrap(),
                        hit.score.to_bits(),
                    )
                })
                .collect();
            let reference: Vec<(usize, u32)> = index
                .traverse_by_id(query, 10, 40, &|id| held.binary_search(&id).is_ok())
                .into_iter()
                .map(|(id, score)| (id, score.to_bits()))
                .collect();
            assert_eq!(page, reference);
        }
    }

    fn traversed(index: &DenseIndex, query: &[f32], admit: &dyn Admit) -> Vec<(u32, u32)> {
        bits_of(&index.traverse(query, 10, 40, admit))
    }

    /// The page of a graph traversed under a test of each candidate's
    /// internal id, which a graph whose every node holds its record reports
    /// as the position.
    fn by_id(twin: &DenseIndex, query: &[f32], admits: impl Fn(usize) -> bool) -> Vec<(u32, u32)> {
        twin.graph()
            .search(query, 10, 40, Some(&|id: &usize| admits(*id)))
            .unwrap()
            .into_iter()
            .map(|hit| (hit.position as u32, hit.distance.to_bits()))
            .collect()
    }

    struct NotThird;

    impl Admit for NotThird {
        fn admits(&self, at: Position) -> bool {
            !at.0.is_multiple_of(3)
        }

        fn len_hint(&self) -> Option<usize> {
            None
        }
    }

    /// Each predicate of the traversal answers what the same graph answers
    /// under the test by internal id that a predicate made before records had
    /// positions, read through the nodes' origin ids. The reference is a twin
    /// built from the same records in the same order with nothing removed, so
    /// it is the same graph node for node and every node reports its id, and
    /// its closures are those tests: the live bit by id, a bitmap's bit and
    /// the live bit by id, and the live bit by id and the table call.
    #[test]
    fn a_traversal_through_removed_nodes_answers_what_the_test_by_id_answers() {
        let (count, dim) = (400usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 1_960);
        let queries = zeusdb_vector_core::test_support::clustered(12, dim, 1_961);
        let layers = NB_LAYER_MAX as usize;
        for gap in [1u32, 1_009] {
            let graph = VectorGraph::new_raw("l2", dim, 8, 64, layers, 50);
            let mut index = spread_index(graph, None, false, &vectors, gap);
            let mut twin = DenseIndex::new(
                VectorGraph::new_raw("l2", dim, 8, 64, layers, 50),
                "l2",
                dim,
                None,
                false,
            );
            for (k, v) in vectors.iter().enumerate() {
                put(&mut twin, 1 + k as u32 * gap, v);
            }
            for k in 0..count as u32 {
                if k % 3 != 1 && k % 7 != 0 {
                    index.remove(Position(1 + k * gap)).unwrap();
                }
            }
            let mut live = Bitmap::default();
            for k in (0..count as u32).filter(|k| k % 3 == 1 && k % 7 != 0) {
                live.insert((1 + k * gap) as usize);
            }
            assert!(twin.every_node_live && !index.every_node_live);
            assert!(
                queries.iter().any(|query| index
                    .graph()
                    .search(query, 10, 40, None::<&fn(&usize) -> bool>)
                    .unwrap()
                    .iter()
                    .any(|hit| !live.contains(hit.position))),
                "a removed record sits where these queries reach it"
            );
            let mut bitmap = Bitmap::with_slots(1 + count * gap as usize);
            for k in (0..count as u32).filter(|k| k % 2 == 0) {
                bitmap.insert(1 + (k * gap) as usize);
            }
            for query in &queries {
                let page = traversed(&index, query, &Candidates::All);
                assert_eq!(
                    page,
                    by_id(&twin, query, |id| live.contains(id)),
                    "gap {gap}"
                );
                let want = by_id(&twin, query, |id| bitmap.contains(id) && live.contains(id));
                assert_eq!(traversed(&index, query, &bitmap), want, "gap {gap}");
                let want = by_id(&twin, query, |id| {
                    live.contains(id) && bitmap.contains(id) && !id.is_multiple_of(3)
                });
                assert_eq!(traversed(&index, query, &And(&bitmap, &NotThird)), want);
                assert_eq!(traversed(&index, query, &And(&NotThird, &bitmap)), want);
                let want = by_id(&twin, query, |id| {
                    live.contains(id) && !id.is_multiple_of(3)
                });
                assert_eq!(traversed(&index, query, &NotThird), want);
            }
        }
    }

    /// The exact scan by position scores what a lookup per record scores,
    /// the node found by a pass over the graph's origin ids rather than
    /// through the numbering, through positions that hold no record, on raw,
    /// scalar and product quantized graphs, with and without raw vectors.
    #[test]
    fn the_scan_in_increasing_order_scores_what_a_lookup_per_record_scores() {
        let (count, dim, gap) = (300usize, 8usize, 1_009u32);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 194);
        let codec = Arc::new(Int8Codec::fit(dim, &vectors).unwrap());
        let pq = Arc::new(PQ::new(dim, 4, 4, 16, None));
        pq.train(&vectors).unwrap();
        let layers = NB_LAYER_MAX as usize;
        let mut with_raw = VectorGraph::new_pq("l2", 8, 64, layers, 50, pq.clone());
        with_raw.open_raw_store(dim, 64).unwrap();
        let indexes = [
            (
                "raw",
                spread_index(
                    VectorGraph::new_raw("l2", dim, 8, 64, layers, 50),
                    None,
                    false,
                    &vectors,
                    gap,
                ),
            ),
            (
                "int8",
                spread_index(
                    VectorGraph::new_int8("l2", 8, 64, layers, 50, codec),
                    None,
                    false,
                    &vectors,
                    gap,
                ),
            ),
            (
                "pq",
                spread_index(
                    VectorGraph::new_pq("l2", 8, 64, layers, 50, pq.clone()),
                    Some(pq.clone()),
                    false,
                    &vectors,
                    gap,
                ),
            ),
            (
                "pq with raw",
                spread_index(with_raw, Some(pq), true, &vectors, gap),
            ),
        ];
        let mut positions: Vec<Position> = (0..count as u32)
            .flat_map(|k| [Position(1 + k * gap), Position(2 + k * gap)])
            .chain([Position(1 + count as u32 * gap + 7), Position(NONE)])
            .collect();
        positions.sort_unstable();
        let distance = raw_distance_fn("l2");
        let needs_unit = reconstruction_needs_unit("l2");
        for (kind, index) in &indexes {
            for q in [3usize, 77, 150] {
                let query = &vectors[q];
                let mut looked_up: Vec<(f32, u32)> = positions
                    .iter()
                    .filter(|at| {
                        at.0 != NONE && (at.0 - 1) % gap == 0 && ((at.0 - 1) / gap) % 7 != 0
                    })
                    .map(|at| {
                        let node = node_with_origin(index.graph(), at.slot()).unwrap();
                        (
                            index.exact_distance_at(query, node, distance, needs_unit),
                            at.0,
                        )
                    })
                    .collect();
                assert_eq!(looked_up.len(), count - count.div_ceil(7), "{kind}");
                looked_up.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let bits = |scored: &[(f32, u32)]| -> Vec<(u32, u32)> {
                    scored
                        .iter()
                        .map(|&(score, at)| (score.to_bits(), at))
                        .collect()
                };
                for k in [10usize, count] {
                    let page = index.score_exact(query, &positions, k, false);
                    let page: Vec<(u32, u32)> = page
                        .items
                        .iter()
                        .map(|hit| (hit.score.to_bits(), hit.position.0))
                        .collect();
                    let mut cut = looked_up.clone();
                    cut.truncate(k);
                    assert_eq!(page, bits(&cut), "{kind} query {q} k {k}");
                }
            }
        }
    }

    /// A record's raw vector, read by its position through `node_at`, is the
    /// vector its node holds, at every gap between positions, and a position
    /// that holds no record, the empty marker included, reads none.
    #[test]
    fn a_raw_vector_by_id_is_the_vector_its_node_holds() {
        let (count, dim) = (400usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 0x0195);
        let layers = NB_LAYER_MAX as usize;
        for gap in [1u32, 3, 40] {
            let graph = VectorGraph::new_raw_unreserved("l2", dim, 8, count, layers, 50);
            let mut index = DenseIndex::new(graph, "l2", dim, None, false);
            for (k, vector) in vectors.iter().enumerate() {
                put(&mut index, 1 + k as u32 * gap, vector);
            }
            let raws = index.raw_vectors();
            let past = 1 + count * gap as usize + 70_000;
            for at in (0..past).chain([NONE as usize, usize::MAX]) {
                let read = index
                    .node_of(Position::from_slot(at))
                    .and_then(|node| index.graph().raw_vector_at(node));
                assert_eq!(raws.get(at), read, "gap {gap}, position {at}");
            }
            for (k, vector) in vectors.iter().enumerate() {
                let at = 1 + k * gap as usize;
                assert_eq!(raws.get(at), Some(vector.as_slice()), "gap {gap}");
                let node = node_with_origin(index.graph(), at).unwrap();
                assert_eq!(index.graph().raw_vector_at(node), Some(vector.as_slice()));
            }
        }
    }

    /// A removal leaves a hole, a position whose `node_at` entry is
    /// [`NONE`], and leaves its node stranded with the node's raw vector
    /// still in the store. The hole reads no raw vector, on a raw graph and
    /// on a product quantized graph that keeps its raw vectors, at every gap
    /// between positions, and so does a position between two records that
    /// never held one. Every position that holds a record reads its own
    /// vector.
    #[test]
    fn a_hole_reads_no_raw_vector_though_its_node_keeps_one() {
        let (count, dim) = (300usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 0x0203);
        let pq = Arc::new(PQ::new(dim, 4, 4, 16, None));
        pq.train(&vectors).unwrap();
        let layers = NB_LAYER_MAX as usize;
        for gap in [1u32, 1_009] {
            let mut with_raw = VectorGraph::new_pq("l2", 8, 64, layers, 50, pq.clone());
            with_raw.open_raw_store(dim, 64).unwrap();
            let raw = VectorGraph::new_raw("l2", dim, 8, 64, layers, 50);
            let indexes = [
                ("raw", spread_index(raw, None, false, &vectors, gap)),
                (
                    "pq with raw",
                    spread_index(with_raw, Some(pq.clone()), true, &vectors, gap),
                ),
            ];
            for (kind, index) in &indexes {
                let mut holes = 0;
                for (k, vector) in vectors.iter().enumerate() {
                    let at = 1 + k * gap as usize;
                    let node = node_with_origin(index.graph(), at).unwrap();
                    let stored = index.graph().raw_vector_at(node);
                    assert_eq!(stored, Some(vector.as_slice()), "{kind} gap {gap} at {at}");
                    if k % 7 == 0 {
                        assert_eq!(index.node_at()[at], NONE, "{kind} gap {gap} at {at}");
                        assert_eq!(index.raw_vector(at), None, "{kind} gap {gap} hole {at}");
                        holes += 1;
                    } else {
                        assert_eq!(index.raw_vector(at), stored, "{kind} gap {gap} at {at}");
                    }
                    if gap > 1 && k + 1 < count {
                        assert_eq!(index.node_at()[at + 1], NONE, "{kind} gap {gap}");
                        assert_eq!(index.raw_vector(at + 1), None, "{kind} gap {gap}");
                    }
                }
                assert_eq!(holes, count.div_ceil(7), "{kind} gap {gap}");
            }
        }
    }

    /// The same hole on a scalar quantized graph. A removal leaves the
    /// position's `node_at` entry [`NONE`] and the stranded node's row still
    /// in the store. The hole reconstructs no vector, at every gap between
    /// positions, and nor does a position between two records that never
    /// held one. Every position that holds a record reconstructs what its
    /// node's row stands for.
    #[test]
    fn a_hole_reconstructs_no_vector_though_its_node_keeps_its_row() {
        let (count, dim) = (300usize, 8usize);
        let vectors = zeusdb_vector_core::test_support::clustered(count, dim, 0x0204);
        let codec = Arc::new(Int8Codec::fit(dim, &vectors).unwrap());
        let layers = NB_LAYER_MAX as usize;
        for gap in [1u32, 1_009] {
            let graph = VectorGraph::new_int8("l2", 8, 64, layers, 50, codec.clone());
            let index = spread_index(graph, None, false, &vectors, gap);
            let mut holes = 0;
            for k in 0..count {
                let at = 1 + k * gap as usize;
                let node = node_with_origin(index.graph(), at).unwrap();
                let stored = index.graph().int8_reconstruct_at(node);
                assert!(stored.is_some(), "gap {gap} at {at}");
                if k % 7 == 0 {
                    assert_eq!(index.node_at()[at], NONE, "gap {gap} at {at}");
                    assert_eq!(index.int8_reconstruct(at), None, "gap {gap} hole {at}");
                    holes += 1;
                } else {
                    assert_eq!(index.int8_reconstruct(at), stored, "gap {gap} at {at}");
                }
                if gap > 1 && k + 1 < count {
                    assert_eq!(index.node_at()[at + 1], NONE, "gap {gap}");
                    assert_eq!(index.int8_reconstruct(at + 1), None, "gap {gap}");
                }
            }
            assert_eq!(holes, count.div_ceil(7), "gap {gap}");
        }
    }

    /// Insert one record at position `at` under internal id `id`.
    fn put_at(index: &mut DenseIndex, at: u32, id: u32, vector: &[f32]) {
        let prepared = index.prepare(RecordId(id), vector).unwrap();
        index
            .insert(Position(at), RecordId(id), vector, prepared)
            .unwrap();
    }

    /// The internal id of the record at each position below `past`, read the
    /// long way: the node `node_at` names and that node's origin id.
    fn ids_read_twice(index: &DenseIndex, past: usize) -> Vec<Option<usize>> {
        (0..past)
            .map(|at| match index.node_at().get(at) {
                Some(&node) if node != NONE => Some(index.graph().origin_id_at(node)),
                _ => None,
            })
            .collect()
    }

    /// The reader of each position's internal id answers what the node and
    /// its origin id answer, at every position and past the last, and it
    /// takes the ids as the positions plus one exactly where they are: no
    /// hole, the first id one and the last the record count.
    #[test]
    fn each_position_names_the_internal_id_its_node_was_inserted_under() {
        let vectors: Vec<Vec<f32>> = (0..40)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let graph = || VectorGraph::new_raw("l2", 2, 4, 64, NB_LAYER_MAX as usize, 50);
        let check = |index: &DenseIndex, shifted: bool, label: &str| {
            let reader = index.ids_by_position();
            assert_eq!(reader.shifted, shifted, "{label}");
            let past = index.node_at().len() + 3;
            let read: Vec<Option<usize>> = (0..past).map(|at| reader.get(at)).collect();
            assert_eq!(read, ids_read_twice(index, past), "{label}");
        };
        let forty = || {
            let mut index = DenseIndex::new(graph(), "l2", 2, None, false);
            for k in 0..40u32 {
                put_at(&mut index, k, k + 1, &vectors[k as usize]);
            }
            index
        };

        let mut index = DenseIndex::new(graph(), "l2", 2, None, false);
        check(&index, false, "empty");
        put_at(&mut index, 0, 1, &vectors[0]);
        check(&index, true, "one record");
        let mut index = forty();
        check(&index, true, "the ids one to forty");
        index.remove(Position(3)).unwrap();
        check(&index, false, "a hole");
        let held: Vec<u32> = (0..40).filter(|&at| at != 3).collect();
        index.close_up(&held);
        assert!(index.numbering_agrees());
        check(&index, false, "closed up past a removed id");

        let mut index = forty();
        index.remove(Position(0)).unwrap();
        index.close_up(&(1..40).collect::<Vec<u32>>());
        assert!(index.numbering_agrees());
        check(&index, false, "the first id two");

        let mut index = forty();
        index.remove(Position(39)).unwrap();
        check(&index, false, "the last position a hole");

        let mut index = DenseIndex::new(graph(), "l2", 2, None, false);
        for k in 0..39u32 {
            put_at(&mut index, k, k + 2, &vectors[k as usize]);
        }
        check(&index, false, "the ids two to forty");
    }

    /// An insert files its node at the position it takes, both where
    /// `node_at` reaches that position and where it stops short of it, as it
    /// does once the records are handed back with the last positions holes.
    #[test]
    fn an_insert_files_its_node_where_node_at_reaches_and_where_it_stops_short() {
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let graph = VectorGraph::new_raw("l2", 2, 4, 64, NB_LAYER_MAX as usize, 50);
        let mut index = DenseIndex::new(graph, "l2", 2, None, false);
        for k in 0..6u32 {
            put_at(&mut index, k, k + 1, &vectors[k as usize]);
            assert_eq!(index.node_at().len(), k as usize + 1);
        }
        index.remove(Position(4)).unwrap();
        index.remove(Position(5)).unwrap();
        assert_eq!(index.set_live(0..4usize), 0);
        assert_eq!(
            index.node_at().len(),
            4,
            "node_at reaches the last record held"
        );

        put_at(&mut index, 6, 7, &vectors[6]);
        assert_eq!(index.node_at().len(), 7);
        assert_eq!(&index.node_at()[4..6], &[NONE, NONE]);
        put_at(&mut index, 7, 8, &vectors[7]);
        assert_eq!(index.node_at().len(), 8);

        assert!(index.numbering_agrees());
        assert_eq!(index.len(), 6);
        for at in [0usize, 1, 2, 3, 6, 7] {
            let node = index.node_at()[at];
            assert_eq!(index.graph().records()[node as usize], at as u32, "{at}");
            assert_eq!(index.raw_vector(at), Some(vectors[at].as_slice()), "{at}");
        }
        for at in [4usize, 5] {
            assert!(!index.holds(Position(at as u32)), "{at}");
            assert_eq!(index.raw_vector(at), None, "{at}");
        }
    }
}
