//! Insertion, replacement, removal, compaction, rebuild and clear.
//!
//! `add` is the whole of what the Python entry point does once parsing is
//! over, and the binding calls it with the interpreter lock released. The
//! three `add_*` paths below `insert_parsed_records` are chosen by whether the
//! index is quantized and whether it is still collecting for training. Removal
//! is logical, and `compact` reclaims the graph nodes that removal and
//! replacement leave behind.
//!
//! Every mutating operation the binding calls takes `writers` here, first and
//! before any guard, and the helpers under it never do.

use super::{
    validate_index_parameters, Collection, DenseIndex, LiveRecords, ParsedRecord, ParsedRecords,
    SparseHalf, StorageMode, MAX_LAYER,
};
use crate::locks::WriteGuard;
use crate::RawVectors;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace, warn};
use zeusdb_vector_core::{
    matches_filter, Bitmap, ColumnStore, Error, Filter, Prepared, RecordId, Selection,
    SparseVector, VectorGraph, VectorIndex,
};
use zeusdb_vector_sparse::PostingsIndex;

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
    id_map: WriteGuard<'a, HashMap<String, usize>>,
    rev_map: WriteGuard<'a, LiveRecords>,
    /// The dense index, so a removal drops the record from its live set.
    /// Taken as a write guard between the reverse map and the codes, which
    /// is its place in the order, and held for the removal alone.
    index: WriteGuard<'a, DenseIndex>,
    pq_codes: WriteGuard<'a, HashMap<String, Vec<u8>>>,
    /// The sparse index, where the collection declares a sparse space.
    sparse: Option<WriteGuard<'a, PostingsIndex>>,
    vector_metadata: WriteGuard<'a, HashMap<String, HashMap<String, Value>>>,
    columns: WriteGuard<'a, ColumnStore>,
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
        let live_records = self.id_map.read().unwrap().len();
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

    /// Put one record into the dense index and name it in the record set, in
    /// the two phases the graph splits an insertion into.
    ///
    /// ```text
    /// index.read()   prepare: quantize where the graph holds codes, descend
    ///                from the level the caller drew, choose the neighbour
    ///                lists
    ///   drop
    /// id_map.write()
    /// rev_map.write()
    /// index.write()  insert: append the node, install its lists, update the
    ///                reverse links, mark the record live, and name it in
    ///                both id maps
    /// ```
    ///
    /// This is the only caller that takes the index lock twice for one
    /// operation, and it is the only one that needs to. The three rebuild paths
    /// each fill a local graph nobody else can reach and swap it in under one
    /// write guard, so they insert with no lock held at all.
    ///
    /// # Why the two id maps are written here
    ///
    /// The dense index keeps its own live set and the collection keeps one
    /// beside `rev_map`, and the two are the same set. Writing the maps in
    /// their own block above and the live bit here left the invariant false
    /// for the whole of phase one, which is where an insertion spends its
    /// time, so a reader taking no mutation guard saw the collection holding a
    /// record the index did not. `get_stats` is such a reader, and on a build
    /// with debug assertions its check of the two sets fired. The two writes
    /// are one write, so they happen under one acquisition, which is what
    /// `remove_under_guards` already does with the same pair.
    ///
    /// It costs nothing on the search path. A search takes `id_map`, `rev_map`
    /// and the index read guards in that order and holds all three for its
    /// whole traversal, so a writer waiting for the index write guard was
    /// already waiting for exactly those searches. What is new is that the
    /// maps are held across phase two, being the fixed `m * 2 * m` memory
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
    /// The record is named in the maps after the index has accepted it, so an
    /// insertion the index refuses leaves the record set untouched rather than
    /// naming a record the graph does not hold.
    ///
    /// # The lock order
    ///
    /// Phase one takes the index read guard with no other guard held. Phase two
    /// takes `id_map`, `rev_map` and the index in that order, which is the
    /// order declared on `Collection` and the order `removal_guards` takes the
    /// same three in. The metadata and the columns this record has already been
    /// written to were taken and released in their own block above, and they
    /// rank below the index, so nothing here is held out of order.
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
    fn insert_one(
        &self,
        external_id: &str,
        vector: &[f32],
        internal_id: usize,
        level: usize,
    ) -> Result<Option<Vec<u8>>, Error> {
        let id = RecordId::from_slot(internal_id);
        let prepared = {
            let index = self.dense().index.read().unwrap();
            index.prepare_at_level(id, vector, level)?
        };
        let (codes, due) = {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();
            let mut index = self.dense().index.write().unwrap();
            index.insert(id, vector, prepared)?;
            id_map.insert(external_id.to_string(), internal_id);
            rev_map.insert(internal_id, external_id.to_string());
            (
                index.graph().codes_of(internal_id).map(<[u8]>::to_vec),
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
        Ok(codes)
    }

    /// Put one record into the sparse space, under its own guard, taken
    /// alone.
    fn insert_sparse(&self, internal_id: usize, vector: &SparseVector) -> Result<(), Error> {
        let space = self.sparse().ok_or(Error::NoSparseSpace)?;
        let mut index = space.index.write().unwrap();
        index.insert(
            RecordId::from_slot(internal_id),
            vector.as_ref(),
            Prepared::none(),
        )
    }

    /// Get next available internal ID
    ///
    /// Not exposed to Python. Every call takes the counter mutex and
    /// increments it, so a call from outside the insertion path burns an
    /// internal id that no record will ever hold.
    pub(super) fn get_next_id(&self) -> usize {
        let mut counter = self.id_counter.lock().unwrap();
        *counter += 1;
        *counter
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
    /// a search holds `rev_map` for its whole traversal and takes `vectors`
    /// afterwards.
    fn removal_guards(&self) -> RemovalGuards<'_> {
        RemovalGuards {
            id_map: self.id_map.write().unwrap(),
            rev_map: self.rev_map.write().unwrap(),
            index: self.dense().index.write().unwrap(),
            pq_codes: self.dense().pq_codes.write().unwrap(),
            sparse: self.sparse().map(|space| space.index.write().unwrap()),
            vector_metadata: self.vector_metadata.write().unwrap(),
            columns: self.columns.write().unwrap(),
        }
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
        let Some(internal_id) = guards.id_map.remove(id) else {
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
        guards.vector_metadata.remove(id); // Remove metadata
                                           //
                                           // The columns are addressed by internal id, so the slot is cleared
                                           // rather than reclaimed. Internal ids are never reused, which is what
                                           // the graph's own node arena already assumes, so a removed record
                                           // leaves a hole in every column exactly as it leaves a stranded node.
        guards.columns.erase(internal_id);
        guards.pq_codes.remove(id); // Remove PQ codes (if present)
        guards.rev_map.remove(internal_id); // Remove ID mapping, and the live bit with it

        // The dense index strands the node and drops the record from its
        // live set, and the sparse index unlinks the record's postings where
        // the record filled that space. Each is asked whether it holds the
        // record first, since a record may leave the sparse space empty and
        // an index refuses to remove an id it does not hold.
        let record = RecordId::from_slot(internal_id);
        debug_assert!(
            guards.index.holds(record),
            "the dense index holds every record id_map holds"
        );
        if guards.index.holds(record) {
            let _ = guards.index.remove(record);
        }
        if let Some(sparse) = guards.sparse.as_mut() {
            if sparse.holds(record) {
                let _ = sparse.remove(record);
            }
        }
        debug_assert!(
            guards.columns.tracks(guards.id_map.len()),
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
            internal_id = internal_id,
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
    pub(super) fn remove_point_internal(&self, id: String) -> Result<bool, String> {
        // Read before the guards are taken, because it reaches the graph lock and
        // the declared order puts the graph above every map held below.
        let storage_mode = self.storage_mode();
        let mut guards = self.removal_guards();
        Ok(self.remove_under_guards(&mut guards, &id, &storage_mode))
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
    pub(super) fn remove_points_internal(&self, ids: &[String]) -> Vec<String> {
        let storage_mode = self.storage_mode();
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
    /// record was read.
    pub(super) fn remove_where_locked(&self, filter: &Filter) -> usize {
        // `rev_map` before `vector_metadata` before `columns`, which is the
        // declared order, because the bitmap holds internal ids and a removal
        // names external ones. The columns guard is dropped before the metadata
        // one is taken, since the selection owns its bitmap and borrows only the
        // filter.
        let rev_map = self.rev_map.read().unwrap();
        let selection = {
            let columns = self.columns.read().unwrap();
            columns.select(filter)
        };
        let resolve = |selected: &Bitmap| {
            let mut ids = Vec::with_capacity(selected.count());
            selected.for_each(|slot| {
                if let Some(id) = rev_map.get(&slot) {
                    ids.push(id.clone());
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
                    if let Some(id) = rev_map.get(&slot) {
                        if vector_metadata
                            .get(id)
                            .is_some_and(|meta| matches_filter(meta, filter))
                        {
                            ids.push(id.clone());
                        }
                    }
                });
                ids
            }
            Selection::Whole(_) => {
                let vector_metadata = self.vector_metadata.read().unwrap();
                vector_metadata
                    .iter()
                    .filter(|(_, meta)| matches_filter(meta, filter))
                    .map(|(id, _)| id.clone())
                    .collect()
            }
        };
        drop(rev_map);

        let total = doomed.len();
        let missing = self.remove_points_internal(&doomed);
        debug_assert!(
            missing.is_empty(),
            "every id came from the metadata store under the mutation guard, so none of them can have been absent by the time it was removed"
        );
        total - missing.len()
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
    /// Existence is decided by `id_map` rather than by the metadata store,
    /// because `id_map` is the record set. Every insertion path writes a
    /// metadata entry even for a record supplied without metadata, so the two
    /// agree, and keying on the authoritative one means they cannot drift.
    pub(super) fn update_metadata_locked(
        &self,
        id: &str,
        metadata: HashMap<String, Value>,
    ) -> bool {
        let id_map = self.id_map.read().unwrap();
        let Some(&internal_id) = id_map.get(id) else {
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
        columns.write(internal_id, &metadata);
        debug_assert!(
            columns.agrees_with(internal_id, &metadata),
            "every declared field's column holds what the record's metadata holds"
        );
        vector_metadata.insert(id.to_string(), metadata);
        trace!(target: LOG_TARGET, operation = "update_metadata",
            vector_id = %id,
            "Metadata replaced"
        );
        true
    }

    /// The body of `compact`, with the interpreter lock already released
    pub(super) fn compact_locked(&self) -> Result<usize, Error> {
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();

        // The sparse space first, where there is one, since what it has to
        // reclaim is its own whatever the graph holds. A removed record
        // leaves its span in the forward arena even after the lazy rewrite
        // has taken its postings out of every list, so the test is whether
        // any record is dead rather than whether any posting is. The figure
        // `compact` reports is the graph's, as it always was.
        if let Some(space) = self.sparse() {
            let dead = space.index.read().unwrap().dead_records();
            if dead > 0 {
                space.index.write().unwrap().compact();
            }
        }

        debug_assert!(
            self.live_sets_agree(),
            "the dense index's live set is the collection's"
        );
        let live_count = self.id_map.read().unwrap().len();
        let (nodes_before, stranded) = {
            let index = self.dense().index.read().unwrap();
            (index.graph().nb_points(), index.stranded())
        };
        debug_assert_eq!(
            stranded,
            nodes_before.saturating_sub(live_count),
            "the dense index's live set is id_map's"
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
    /// `compact` does, so `id_map`, `rev_map`, the metadata store and every
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
    /// whose records could not be rebuilt from still reports the configuration
    /// its graph actually has.
    ///
    /// Returns the live record count the replacement holds.
    pub(super) fn rebuild_locked(
        &self,
        m: usize,
        expected_size: usize,
        ef_construction: usize,
    ) -> Result<usize, Error> {
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();

        let previous_m = self.dense().m();
        let previous_expected_size = self.expected_size();
        let previous_ef_construction = self.dense().ef_construction();
        let live_count = self.id_map.read().unwrap().len();
        let nodes_before = self.dense().index.read().unwrap().graph().nb_points();

        let nodes_after = self.rebuild_graph(m, expected_size, ef_construction)?;

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
    /// The caller holds `writers`.
    fn rebuild_graph(
        &self,
        m: usize,
        expected_size: usize,
        ef_construction: usize,
    ) -> Result<usize, Error> {
        let live_count = self.id_map.read().unwrap().len();
        let quantized = self.is_quantized();

        let mut new_hnsw = if quantized {
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

        // Re-insert every live record under the internal id it already holds, so the
        // two id maps stay correct without being rewritten. A record whose source data
        // is missing is collected rather than skipped, because skipping it would drop
        // it from the index silently.
        let missing: Vec<String> = {
            let id_map = self.id_map.read().unwrap();
            // The graph being replaced, which is where the vectors live. Taken
            // after `id_map` and released with it, before `replace_graph`
            // takes the write guard below.
            let old_index = self.dense().index.read().unwrap();
            let old_hnsw = old_index.graph();

            // Internal id order, which is arrival order, rather than the order a
            // hash map hands its entries out. Two compactions of the same index
            // in two processes otherwise wire the replacement graph differently
            // and answer the same query differently.
            let mut live: Vec<(&String, usize)> = id_map
                .iter()
                .map(|(id, &internal)| (id, internal))
                .collect();
            live.sort_by_key(|&(_, internal_id)| internal_id);

            if quantized {
                let pq_codes = self.dense().pq_codes.read().unwrap();
                let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(id_map.len());
                let mut missing = Vec::new();

                for (ext_id, internal_id) in live {
                    match pq_codes.get(ext_id) {
                        Some(codes) => batch.push((codes, internal_id)),
                        None => missing.push(ext_id.clone()),
                    }
                }

                if missing.is_empty() && !batch.is_empty() {
                    new_hnsw.insert_batch_pq(&batch).map_err(|e| {
                        error!(target: LOG_TARGET, operation = "rebuild_graph", error = %e, "Failed to re-insert quantized codes");
                        Error::ReinsertCodesFailed(e)
                    })?;
                }

                // The raw vectors a `quantized_with_raw` index keeps beside its
                // codes are carried over rather than rebuilt, since nothing
                // else holds them. The replacement numbers its nodes its own
                // way, so they are re-addressed node by node.
                if missing.is_empty() && old_hnsw.holds_raw() {
                    if let Some(dim) = old_hnsw.raw_dim() {
                        new_hnsw.adopt_raw_from(old_hnsw, dim).map_err(|e| {
                            error!(target: LOG_TARGET, operation = "rebuild_graph", error = %e, "Failed to carry the raw vectors over");
                            Error::AdoptRawFailed(e)
                        })?;
                    }
                }

                missing
            } else {
                let mut missing = Vec::new();

                for (ext_id, internal_id) in live {
                    match old_hnsw.raw_vector(internal_id) {
                        Some(vector) => new_hnsw.insert(vector, internal_id),
                        None => missing.push(ext_id.clone()),
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
                what: if quantized {
                    "quantized codes"
                } else {
                    "vector"
                },
            });
        }

        let nodes_after = new_hnsw.nb_points();

        // The replacement was built by insertion, so its arenas carry the same
        // geometric slack the graph being replaced carried. Compaction exists to
        // return memory, and returning it while still holding the graph outside
        // the write guard costs one copy of the live bytes against a rebuild that
        // has just re-inserted every record.
        new_hnsw.shrink_to_fit();
        self.dense().replace_graph(new_hnsw);

        Ok(nodes_after)
    }

    /// The insertion phase of `add`, run with the interpreter lock released
    ///
    /// Everything here operates on `ParsedRecords`, which is
    /// `Vec<(String, Vec<f32>, HashMap<String, Value>)>` and holds no Python
    /// object, no `Py<T>` and no borrow of anything Python owns.
    /// The caller holds the mutation guard.
    ///
    /// The complete set of functions reachable from here, verified by reading
    /// each one rather than by inference:
    ///
    /// - `remove_point_internal`, and through it `get_storage_mode`,
    ///   `has_quantization` and `can_use_quantization`
    /// - `add_single_vector`, and the three paths below it, `add_raw_vector`,
    ///   `add_with_id_collection` and `add_quantized_vector`
    /// - `get_next_id`, `is_quantized`
    /// - `maybe_trigger_training`, `train_quantization_from_ids` and
    ///   `rebuild_with_quantization_locked`
    /// - `PQ::is_trained`, `quantize`, `quantize_batch` and `train`, plus the
    ///   k-means below `train`
    /// - `VectorGraph::draw_level`, `plan_at_level`, `install`, `insert`,
    ///   `insert_batch_pq`, `new_pq` and `nb_points`, and the graph structure
    ///   in `graph::mutable` below them
    /// - `count_record_terms`, and through it the text layer's dictionary
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
        parsed_data: Vec<ParsedRecord>,
        overwrite: bool,
    ) -> (Vec<String>, Vec<InsertError>) {
        let mut inserted_ids: Vec<String> = Vec::with_capacity(parsed_data.len());
        let mut errors: Vec<InsertError> = Vec::new();

        // ENHANCED FIX: Handle overwrites properly for ALL paths (Raw, Training, PQ)
        if overwrite {
            // Phase 1: Batch identify and remove existing documents
            let (ids_to_remove, storage_analysis) = {
                let id_map = self.id_map.read().unwrap();
                let index = self.dense().index.read().unwrap();
                let pq_codes = self.dense().pq_codes.read().unwrap();
                let raws = RawVectors {
                    id_map: &id_map,
                    graph: index.graph(),
                };

                let mut ids_to_remove = Vec::new();
                let mut has_raw = 0;
                let mut has_pq = 0;
                let mut has_both = 0;

                for record in &parsed_data {
                    let id = &record.id;
                    if id_map.contains_key(id) {
                        ids_to_remove.push(id.clone());

                        // Analyze what's being replaced for logging
                        let has_raw_vector = raws.contains(id);
                        let has_pq_codes = pq_codes.contains_key(id);

                        match (has_raw_vector, has_pq_codes) {
                            (true, true) => has_both += 1,
                            (true, false) => has_raw += 1,
                            (false, true) => has_pq += 1,
                            (false, false) => {} // Shouldn't happen, but handle gracefully
                        }
                    }
                }

                (ids_to_remove, (has_raw, has_pq, has_both))
            }; // Release all read locks here

            if !ids_to_remove.is_empty() {
                info!(target: LOG_TARGET, operation = "overwrite_preparation",
                    documents_to_remove = ids_to_remove.len(),
                    storage_analysis = format!(
                        "raw_only: {}, pq_only: {}, both: {}",
                        storage_analysis.0, storage_analysis.1, storage_analysis.2
                    ),
                    "Removing existing documents for overwrite"
                );

                // Batch remove existing documents (handles both raw and PQ data)
                let mut removed_count = 0;
                let mut removal_errors = 0;

                for id in ids_to_remove {
                    match self.remove_point_internal(id.clone()) {
                        Ok(was_removed) => {
                            if was_removed {
                                removed_count += 1;
                                trace!(target: LOG_TARGET, operation = "overwrite_removal",
                                    vector_id = %id,
                                    "Removed existing vector/codes for overwrite"
                                );
                            }
                        }
                        Err(e) => {
                            removal_errors += 1;
                            warn!(target: LOG_TARGET, operation = "overwrite_removal",
                                vector_id = %id,
                                error = %e,
                                "Failed to remove existing vector for overwrite"
                            );
                            errors.push(InsertError::Counted(format!(
                                "Failed to remove existing {}: {}",
                                id, e
                            )));
                        }
                    }
                }

                info!(target: LOG_TARGET, operation = "overwrite_removal_complete",
                    removed_count = removed_count,
                    removal_errors = removal_errors,
                    "Completed removal phase for overwrite"
                );
            }
        }

        // Phase 2: Add new vectors using the correct path based on current PQ state
        debug!(target: LOG_TARGET, operation = "add_vectors_insertion_phase",
            current_state = self.storage_mode(),
            "Starting insertion phase"
        );

        for ParsedRecord {
            id,
            vector,
            sparse,
            metadata,
        } in parsed_data
        {
            let id_for_error = id.clone();

            // Use overwrite=false since we already handled removals above
            // The add_single_vector method will route to the correct path based on current PQ state
            match self.add_single_vector(id, vector, sparse, metadata, false) {
                Ok(inserted_new) => {
                    // The id is recorded here rather than counted, because the
                    // caller now needs to know which records landed and not only
                    // how many. `total_inserted` is this list's length.
                    inserted_ids.push(id_for_error.clone());
                    if inserted_new {
                        let mut count = self.vector_count.lock().unwrap();
                        *count += 1;
                    }

                    // Check training trigger (graceful failure handling)
                    if let Err(training_error) = self.maybe_trigger_training() {
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

        (inserted_ids, errors)
    }

    // 1. CORE VECTOR OPERATIONS (6 methods)
    /// 3-PATH ARCHITECTURE - Main router
    fn add_single_vector(
        &self,
        id: String,
        vector: Vec<f32>,
        sparse: Option<SparseHalf>,
        metadata: HashMap<String, Value>,
        overwrite: bool,
    ) -> Result<bool, Error> {
        // Check if this is a new vector or an overwrite
        let is_new = {
            let id_map = self.id_map.read().unwrap();
            !id_map.contains_key(&id)
        };

        if !overwrite && !is_new {
            warn!(target: LOG_TARGET, operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(Error::DuplicateId { id });
        }

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
            Some(SparseHalf::Terms(terms)) => Some(self.count_record_terms(&terms)?),
        };
        if let Some(sparse) = &sparse {
            let space = self.sparse().ok_or(Error::NoSparseSpace)?;
            space.config().weighting.validate_record(sparse.as_ref())?;
        }

        trace!(target: LOG_TARGET, operation = "add_single_vector",
            vector_id = %id,
            is_new = is_new,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        let internal_id = if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(id, vector, metadata)?
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(id, vector, metadata)?
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(id, vector, metadata)?
        };

        // The sparse half, under the same internal id, once the dense half
        // and the maps are in.
        if let Some(sparse) = sparse {
            self.insert_sparse(internal_id, &sparse)?;
        }

        Ok(is_new)
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>,
    ) -> Result<usize, Error> {
        let internal_id = self.get_next_id();
        let level = self.draw_level();

        // Store metadata, and the declared fields of it in their columns.
        //
        // One block and one order, because the two are the same write seen from
        // two directions and a search that saw one without the other would
        // return a record the filter had not admitted.
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            let mut columns = self.columns.write().unwrap();
            columns.write(internal_id, &metadata);
            debug_assert!(
                columns.agrees_with(internal_id, &metadata),
                "every declared field's column holds what the record's metadata holds"
            );
            vector_metadata.insert(id.clone(), metadata);
        }

        // Insert the processed vector into the graph and name the record in
        // both id maps, in the two phases the graph splits an insertion into,
        // at the level drawn above. See `insert_one`. The maps are written
        // there rather than here so that the record enters the collection's
        // live set and the index's under one acquisition. This is the only
        // copy of the vector the index keeps: the store the graph is addressed
        // against holds it, and there is no second map to write.
        self.insert_one(&id, &vector, internal_id, level)?; // Already normalized

        trace!(target: LOG_TARGET, operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(internal_id)
    }

    /// Path B: ID collection for consistent training
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> Result<usize, Error> {
        // 1. Store vector normally (single storage)
        let internal_id = self.add_raw_vector(id.clone(), vector, metadata)?;

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
        // what every subset of it is taken by. `train_quantization_from_ids`
        // shuffles it under a fixed seed; see `TRAINING_SAMPLE_SEED`.
        if let Some(config) = &self.dense().quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.clone());
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

        Ok(internal_id)
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(target = LOG_TARGET, level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> Result<usize, Error> {
        let internal_id = self.get_next_id();
        let level = self.draw_level();

        // Store metadata, and the declared fields of it in their columns.
        //
        // One block and one order, because the two are the same write seen from
        // two directions and a search that saw one without the other would
        // return a record the filter had not admitted.
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            let mut columns = self.columns.write().unwrap();
            columns.write(internal_id, &metadata);
            debug_assert!(
                columns.agrees_with(internal_id, &metadata),
                "every declared field's column holds what the record's metadata holds"
            );
            vector_metadata.insert(id.clone(), metadata);
        }

        // Quantize and insert, and name the record in both id maps, in the two
        // phases the graph splits an insertion into, at the level drawn above.
        // The index quantizes in the first phase, under the read guard, and
        // installs the codes in the second, carrying the raw vector beside
        // them where the storage mode keeps one, because the node the codes
        // are installed at is the node the raw has to sit at. The maps are
        // written in the second phase, so that the record enters the
        // collection's live set and the index's under one acquisition. See
        // `insert_one`.
        let codes = self
            .insert_one(&id, &vector, internal_id, level)?
            .unwrap_or_default();

        // Store quantized codes (always), keyed by external id for the paths
        // that reach a record by name.
        {
            let mut pq_codes = self.dense().pq_codes.write().unwrap();
            pq_codes.insert(id.clone(), codes.clone());
        }

        trace!(target: LOG_TARGET, operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes.len(),
            "Quantized vector added successfully"
        );

        Ok(internal_id)
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
            let _writers = self.writers.lock().unwrap();
            self.insert_parsed_records(parsed_data, overwrite)
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
    pub fn remove_point(&self, id: String) -> Result<bool, Error> {
        let _writers = self.writers.lock().unwrap();
        self.remove_point_internal(id).map_err(Error::Engine)
    }

    /// Remove a batch of records under one acquisition of the mutation guard,
    /// returning the ids that were not in the index. See
    /// `remove_points_internal`.
    pub fn remove_points(&self, ids: &[String]) -> Vec<String> {
        let _writers = self.writers.lock().unwrap();
        self.remove_points_internal(ids)
    }

    /// Remove the records these ids name and report how many went.
    ///
    /// A repeated id names one record, so it counts once whether or not it
    /// was there. `remove_points_internal` already skips a repeat rather than
    /// reporting it missing.
    pub fn delete_ids(&self, requested: &[String]) -> usize {
        let distinct: HashSet<&String> = requested.iter().collect();
        let distinct_count = distinct.len();

        let missing = {
            let _writers = self.writers.lock().unwrap();
            self.remove_points_internal(requested)
        };

        distinct_count - missing.len()
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
        let _writers = self.writers.lock().unwrap();
        Ok(self.remove_where_locked(conditions))
    }

    /// Replace one record's metadata without touching its vector. See
    /// `update_metadata_locked`.
    pub fn update_metadata(&self, id: &str, metadata: HashMap<String, Value>) -> bool {
        let _writers = self.writers.lock().unwrap();
        self.update_metadata_locked(id, metadata)
    }

    /// Rebuild the graph in memory and reclaim the nodes removal and overwrite
    /// strand. See `compact_locked`.
    pub fn compact(&self) -> Result<usize, Error> {
        self.compact_locked()
    }

    /// Return the graph's spare buffer capacity to the allocator, reporting
    /// the bytes released. See `VectorGraph::shrink_to_fit`.
    pub fn shrink_to_fit(&self) -> usize {
        let _writers = self.writers.lock().unwrap();
        let mut index = self.dense().index.write().unwrap();
        index.graph_mut().shrink_to_fit()
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
    pub fn rebuild(&self, plan: RebuildPlan) -> Result<usize, Error> {
        if plan.rearm_overgrowth {
            self.overgrowth_warned.store(false, Ordering::Release);
        }
        self.rebuild_locked(plan.m, plan.expected_size, plan.ef_construction)
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
    pub fn clear(&self) -> Result<usize, Error> {
        let quantized = self.is_quantized();
        let pq = self.dense().pq.as_ref().cloned();

        if quantized && pq.is_none() {
            return Err(Error::NoQuantizer);
        }

        let _writers = self.writers.lock().unwrap();

        // Built before any guard is taken, so the allocation happens
        // outside the write guard exactly as `compact` arranges it.
        let fresh = if let (true, Some(pq)) = (quantized, pq) {
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
            if self.dense().keeps_raw() {
                graph
                    .open_raw_store(self.dense().dim, self.expected_size())
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
        // set since the maps below are about to be empty too. Built and timed
        // before any guard is taken, so the guarded block below is a move and
        // not a construction.
        let replacement = self.dense().fresh_index(fresh);

        // The storage guards in the order declared on the struct, which is
        // the order every other multi-guard path here takes them in.
        //
        // The dense index is one of them, between `rev_map` and the codes it
        // ranks above and below. Emptying `rev_map` here and replacing the
        // index afterwards left the collection's live set empty while the
        // index's still held every record, which is the same disagreement the
        // insertion path carried and which a `get_stats` running beside a
        // `clear` observed. The two sets are one set, so they empty under one
        // acquisition.
        let (removed, old_index) = {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();
            let mut index = self.dense().index.write().unwrap();
            let mut pq_codes = self.dense().pq_codes.write().unwrap();
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            let mut columns = self.columns.write().unwrap();
            let mut training_ids = self.training_ids.write().unwrap();
            let mut id_counter = self.id_counter.lock().unwrap();
            let mut vector_count = self.vector_count.lock().unwrap();

            let removed = id_map.len();
            id_map.clear();
            rev_map.clear();
            let old_index = std::mem::replace(&mut *index, replacement);
            pq_codes.clear();
            vector_metadata.clear();
            // Keeps the declaration and drops every record, which is what
            // `clear` does to the index itself. The reservation comes back
            // too, since a cleared index is about to be filled again.
            columns.clear(self.expected_size());
            training_ids.clear();
            *id_counter = 0;
            *vector_count = 0;
            (removed, old_index)
        };

        // The graph that was replaced, dropped with every guard released. A
        // graph's drop forks to rayon, and a rayon fork under a write guard
        // deadlocks against a batch search holding the pool; see
        // `DenseSpace::replace_graph`.
        drop(old_index);

        // The sparse space, where there is one, starts again as well, under
        // its own guard taken alone. A record's terms are counted into ids
        // under `writers` as well, in `add_single_vector`, so no caller holds
        // an id the dictionary this empties issued.
        if let Some(space) = self.sparse() {
            *space.index.write().unwrap() = PostingsIndex::new(space.config().clone());
            if let Some(text) = &space.text {
                *text.dictionary.write().unwrap() = zeusdb_vector_text::TermDictionary::new();
            }
        }

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

        Ok(removed)
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
