//! Insertion, replacement, removal and compaction.
//!
//! `insert_parsed_records` is the whole of what `add` does once parsing is over,
//! and it runs with the interpreter lock released. The three `add_*` paths below
//! it are chosen by whether the index is quantized and whether it is still
//! collecting for training. Removal is logical, and `compact_locked` reclaims
//! the graph nodes that removal and replacement leave behind.

use super::{HNSWIndex, ParsedRecords, StorageMode, MAX_LAYER};
use crate::graph::VectorGraph;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace, warn};
/// Multiple of `expected_size` at which an index warns that it has outgrown its
/// declaration. Fires once per index.
const EXPECTED_SIZE_OVERGROWTH_FACTOR: usize = 2;
/// An error raised inside `add`'s insertion phase, carried out to be recorded
///
/// The insertion phase runs with the interpreter lock released, so it cannot
/// build the message for an error that arrives as a `PyErr`. `PyErr`'s
/// `Display` implementation calls `Python::attach`, which would reacquire the
/// lock in the middle of the region that exists to have released it, and would
/// do so while the mutation guard and possibly a storage guard are held.
/// `add` formats those once the lock is back.
pub(super) enum InsertError {
    /// A message Rust already holds, counted against `total_errors`
    Counted(String),

    /// A training failure. Recorded but not counted, which is what the training
    /// path has always done, because a training failure is not a rejected record
    Training(String),

    /// A `PyErr` from one of the three insert paths, with the id it belongs to.
    /// Counted against `total_errors` once formatted
    Vector { id: String, err: PyErr },
}
impl HNSWIndex {
    /// Warn once when the index holds materially more records than it declared
    ///
    /// `expected_size` is a capacity hint rather than a limit, so exceeding it is
    /// legal and the index keeps working. What it costs is not the reservation,
    /// which grows through the ordinary `Vec::push` path, but the graph degree.
    /// The Python factory derives the default `m` from `expected_size`, and `m`
    /// is fixed at construction, so an index that has outgrown its declaration by
    /// a wide margin is running at a degree chosen for a smaller index and no
    /// later `add` revises it. Nothing else tells a caller that.
    ///
    /// A warning rather than an error, because the index is correct and the only
    /// remedy is to rebuild at an honest declaration, which is the caller's call.
    ///
    /// Fires once per index. The flag is claimed with a compare and exchange, so
    /// two writers crossing the threshold together produce one line and not two.
    pub(super) fn warn_if_outgrown_expected_size(&self) {
        if self.overgrowth_warned.load(Ordering::Acquire) {
            return;
        }

        let threshold = self
            .expected_size
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

        warn!(
            operation = "expected_size_exceeded",
            live_records = live_records,
            expected_size = self.expected_size,
            m = self.m,
            "Index holds more than {}x the records its expected_size declared. \
             expected_size is a hint and not a limit, so nothing is broken, but m \
             is fixed at construction and was sized for the declaration. Recreate \
             the index with an expected_size matching what it actually holds if \
             recall matters. This warning fires once.",
            EXPECTED_SIZE_OVERGROWTH_FACTOR
        );
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

    /// Internal remove_point method that can be called without Python bindings
    /// This is the core method that properly removes all traces of a document
    /// Enhanced internal remove_point method with comprehensive PQ support
    pub(super) fn remove_point_internal(&self, id: String) -> Result<bool, String> {
        // Read before the guards are taken, because it reaches the graph lock and
        // the declared order puts the graph above every map held below.
        let storage_mode = self.get_storage_mode();

        // Every write guard, in the order declared on `HNSWIndex`. This used to
        // take `vectors` before `rev_map`, which is the reverse of what a search
        // does, and a search holds `rev_map` for its whole traversal. That pair
        // could not overlap while the receivers were exclusive. It can now.
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        let mut pq_codes = self.pq_codes.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();

        // Check if the document exists
        if let Some(internal_id) = id_map.remove(&id) {
            // Track what we're removing for logging
            let had_raw_vector = vectors.contains_key(&id);
            let had_pq_codes = pq_codes.contains_key(&id);

            // Remove from all data structures
            vectors.remove(&id); // Remove raw vectors (if present)
            vector_metadata.remove(&id); // Remove metadata
            pq_codes.remove(&id); // Remove PQ codes (if present)
            rev_map.remove(&internal_id); // Remove ID mapping

            // Enhanced training state cleanup for quantization
            if self.has_quantization() {
                // Remove from training IDs if present and not yet trained
                if !self.can_use_quantization() {
                    let mut training_ids = self.training_ids.write().unwrap();
                    let original_len = training_ids.len();
                    training_ids.retain(|training_id| training_id != &id);

                    if training_ids.len() != original_len {
                        trace!(
                            operation = "training_cleanup",
                            vector_id = %id,
                            remaining_training_vectors = training_ids.len(),
                            "Removed vector from training set"
                        );

                        // Update threshold status if we dropped below training size
                        if let Some(config) = &self.quantization_config {
                            if training_ids.len() < config.training_size {
                                self.training_threshold_reached
                                    .store(false, std::sync::atomic::Ordering::Release);
                                debug!(
                                    operation = "training_threshold_reset",
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

            debug!(
                operation = "remove_point_internal",
                vector_id = %id,
                internal_id = internal_id,
                had_raw_vector = had_raw_vector,
                had_pq_codes = had_pq_codes,
                storage_mode = storage_mode,
                note = "hnsw_graph_entries_remain_unreachable",
                "Vector completely removed from index (HNSW graph entries become unreachable)"
            );
            Ok(true)
        } else {
            trace!(
                operation = "remove_point_internal",
                vector_id = %id,
                "Vector not found for removal"
            );
            Ok(false)
        }
    }

    /// The body of `compact`, with the interpreter lock already released
    pub(super) fn compact_locked(&self) -> PyResult<usize> {
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();

        let live_count = self.id_map.read().unwrap().len();
        let nodes_before = self.hnsw.read().unwrap().nb_points();

        if nodes_before <= live_count {
            debug!(
                operation = "compact",
                graph_nodes = nodes_before,
                live_records = live_count,
                "No stranded nodes, compact is a no-op"
            );
            return Ok(0);
        }

        let quantized = self.is_quantized();

        let new_hnsw = if quantized {
            let pq = self.pq.as_ref().cloned().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Index reports a quantized graph but holds no product quantizer",
                )
            })?;
            VectorGraph::new_pq(
                &self.space,
                self.m,
                self.expected_size,
                MAX_LAYER,
                self.ef_construction,
                pq,
            )
        } else {
            VectorGraph::new_raw(
                &self.space,
                self.m,
                self.expected_size,
                MAX_LAYER,
                self.ef_construction,
            )
        };

        // Re-insert every live record under the internal id it already holds, so the
        // two id maps stay correct without being rewritten. A record whose source data
        // is missing is collected rather than skipped, because skipping it would drop
        // it from the index silently.
        let missing: Vec<String> = {
            let id_map = self.id_map.read().unwrap();

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
                let pq_codes = self.pq_codes.read().unwrap();
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
                        error!(operation = "compact", error = %e, "Failed to re-insert quantized codes");
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to re-insert quantized codes during compact: {}",
                            e
                        ))
                    })?;
                }

                missing
            } else {
                let vectors = self.vectors.read().unwrap();
                let mut missing = Vec::new();

                for (ext_id, internal_id) in live {
                    match vectors.get(ext_id) {
                        Some(vector) => new_hnsw.insert(vector, internal_id),
                        None => missing.push(ext_id.clone()),
                    }
                }

                missing
            }
        };

        if !missing.is_empty() {
            error!(
                operation = "compact",
                missing_records = missing.len(),
                live_records = live_count,
                quantized = quantized,
                "Refusing to compact, some live records have no source data to rebuild from"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Refusing to compact: {} of {} live records have no stored {} to rebuild \
                 the graph from, so compacting would drop them. The index is unchanged.",
                missing.len(),
                live_count,
                if quantized {
                    "quantized codes"
                } else {
                    "vector"
                }
            )));
        }

        let nodes_after = new_hnsw.nb_points();
        self.replace_graph(new_hnsw);

        let reclaimed = nodes_before - nodes_after;
        info!(
            operation = "compact_complete",
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
    /// - `VectorGraph::insert`, `insert_pq_codes`, `insert_batch_pq`,
    ///   `new_pq` and `nb_points`, and the vendored `hnsw_rs` graph below them
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
        parsed_data: ParsedRecords,
        overwrite: bool,
    ) -> (usize, Vec<InsertError>) {
        let mut total_inserted = 0;
        let mut errors: Vec<InsertError> = Vec::new();

        // ENHANCED FIX: Handle overwrites properly for ALL paths (Raw, Training, PQ)
        if overwrite {
            // Phase 1: Batch identify and remove existing documents
            let (ids_to_remove, storage_analysis) = {
                let id_map = self.id_map.read().unwrap();
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();

                let mut ids_to_remove = Vec::new();
                let mut has_raw = 0;
                let mut has_pq = 0;
                let mut has_both = 0;

                for (id, _, _) in &parsed_data {
                    if id_map.contains_key(id) {
                        ids_to_remove.push(id.clone());

                        // Analyze what's being replaced for logging
                        let has_raw_vector = vectors.contains_key(id);
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
                info!(
                    operation = "overwrite_preparation",
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
                                trace!(
                                    operation = "overwrite_removal",
                                    vector_id = %id,
                                    "Removed existing vector/codes for overwrite"
                                );
                            }
                        }
                        Err(e) => {
                            removal_errors += 1;
                            warn!(
                                operation = "overwrite_removal",
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

                info!(
                    operation = "overwrite_removal_complete",
                    removed_count = removed_count,
                    removal_errors = removal_errors,
                    "Completed removal phase for overwrite"
                );
            }
        }

        // Phase 2: Add new vectors using the correct path based on current PQ state
        debug!(
            operation = "add_vectors_insertion_phase",
            current_state = self.get_storage_mode(),
            "Starting insertion phase"
        );

        for (id, vector, metadata) in parsed_data {
            let id_for_error = id.clone();

            // Use overwrite=false since we already handled removals above
            // The add_single_vector method will route to the correct path based on current PQ state
            match self.add_single_vector(id, vector, metadata, false) {
                Ok(inserted_new) => {
                    total_inserted += 1;
                    if inserted_new {
                        let mut count = self.vector_count.lock().unwrap();
                        *count += 1;
                    }

                    // Check training trigger (graceful failure handling)
                    if let Err(training_error) = self.maybe_trigger_training() {
                        warn!(
                            operation = "training_trigger",
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

        (total_inserted, errors)
    }

    // 1. CORE VECTOR OPERATIONS (6 methods)
    /// 3-PATH ARCHITECTURE - Main router
    fn add_single_vector(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, Value>,
        overwrite: bool,
    ) -> PyResult<bool> {
        // Check if this is a new vector or an overwrite
        let is_new = {
            let id_map = self.id_map.read().unwrap();
            !id_map.contains_key(&id)
        };

        if !overwrite && !is_new {
            warn!(
                operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector with ID '{}' already exists",
                id
            )));
        }

        trace!(
            operation = "add_single_vector",
            vector_id = %id,
            is_new = is_new,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(id, vector, metadata)?;
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(id, vector, metadata)?;
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(id, vector, metadata)?;
        }

        Ok(is_new)
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Store processed vector directly (no additional processing)
        {
            let mut vectors = self.vectors.write().unwrap();
            vectors.insert(id.clone(), vector.clone()); // Already normalized
        }

        // Insert processed vector into HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert(&vector, internal_id); // Already normalized
        }

        trace!(
            operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(())
    }

    /// Path B: ID collection for consistent training
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        // 1. Store vector normally (single storage)
        self.add_raw_vector(id.clone(), vector, metadata)?;

        // SKIP TRAINING ID COLLECTION DURING PERSISTENCE REBUILD
        if self
            .rebuilding_from_persistence
            .load(std::sync::atomic::Ordering::Acquire)
        {
            trace!(
                operation = "add_with_id_collection",
                vector_id = %id,
                reason = "rebuilding_from_persistence",
                "Skipping training ID collection during rebuild"
            );
            return Ok(());
        }

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
        if let Some(config) = &self.quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.clone());
                    let progress = (training_ids.len() as f32 / config.training_size as f32
                        * 100.0)
                        .min(100.0);

                    trace!(
                        operation = "training_id_collection",
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
                        info!(
                            operation = "training_threshold_reached",
                            collected_count = training_ids.len(),
                            target_size = config.training_size,
                            "Training threshold reached - ready for PQ training"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Quantize the vector
        let pq = self.pq.as_ref().unwrap();
        let codes = pq.quantize(&vector).map_err(|e| {
            error!(
                operation = "add_quantized_vector",
                vector_id = %id,
                error = %e,
                "Failed to quantize vector"
            );
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to quantize vector: {}",
                e
            ))
        })?;

        // Store quantized codes (always)
        {
            let mut pq_codes = self.pq_codes.write().unwrap();
            pq_codes.insert(id.clone(), codes.clone());
        }

        // Store raw vector only if configured to keep them
        if let Some(config) = &self.quantization_config {
            if config.storage_mode == StorageMode::QuantizedWithRaw {
                let mut vectors = self.vectors.write().unwrap();
                vectors.insert(id.clone(), vector.clone());
            }
            // If QuantizedOnly mode, we don't store raw vectors (saves memory)
        }

        // Insert codes into quantized HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert_pq_codes(&codes, internal_id);
        }

        trace!(
            operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes.len(),
            "Quantized vector added successfully"
        );

        Ok(())
    }
}
