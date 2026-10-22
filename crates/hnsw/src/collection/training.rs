//! Quantizer training, and the graph rebuild that follows it.
//!
//! Training fires once, on the `add` that reaches `training_size`, once the
//! segment holding that record has committed and installed. It fits the
//! codebook or the scales, measures the rerank fetch on the sample it has
//! just fitted to where the storage mode reranks, hands the training's record
//! to the sink carrying everything it fitted and commits it alone, and only
//! then installs: the quantizer, the stamp and the calibration, the collected
//! ids cleared, and the whole graph rebuilt over the codes. A replay of the
//! record installs through the same function and fits nothing.
//! `quantized_only` sheds its raw vectors at the end of that rebuild, which is
//! the single point where the mode does so.

use super::insert::MutationGuard;
use super::{Collection, QuantizationConfig, StorageMode, MAX_LAYER};
use crate::{calibrate_rerank_from_sample, raw_distance_fn, RerankCalibration};
use chrono::Utc;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace, warn};
use zeusdb_vector_core::{
    Error, FittedRef, Int8Codec, Operation, OperationKind, Position, SeededRng, Values,
    VectorGraph, JOURNAL_MAX_PAYLOAD, PQ,
};

/// What a training fitted, held until its record has committed and then
/// installed, by the live training and by a replay of its record alike.
pub(super) enum Trained {
    /// A codebook, one list a subvector of one list a centroid, with the
    /// rerank calibration measured on it as the record carries it, being
    /// JSON, where one was measured.
    Codebook {
        centroids: Vec<Vec<Vec<f32>>>,
        calibration: Option<String>,
    },
    /// The scales.
    Scales(Int8Codec),
}

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::training";
/// Seed the training sample is shuffled with before it is used
///
/// The records the sample holds are fixed and cannot be sampled. Training fires
/// on the record that reaches `training_size`, so the index holds exactly
/// `training_size` records at that moment and every one of them is in the
/// sample. What can be drawn randomly is the order, and the order is what every
/// subset of the sample is taken by: the codebook sees the records in this
/// order, the calibration takes its queries by striding it, and the calibration
/// takes each fitting fraction as a prefix of it. Without the shuffle all three
/// are slices of insertion order, and a corpus that arrives in a meaningful
/// order makes a prefix measure something other than the whole. On ada-002
/// embeddings in DBpedia article order the first half of the sample measured a
/// fetch of 120 to 135 candidates where the second half measured 165 to 178 and
/// a random half measured 109 to 156, over three codebook draws.
///
/// It is a fixed seed rather than an entropy draw, so two builds over the same
/// records in the same order produce the same shuffle and the same calibration.
/// The k-means the codebook is fitted with draws from its own fixed seed, see
/// `PQ_TRAINING_SEED` in `pq.rs`, so the codebook is a function of the sample
/// as well and two builds over the same records produce one codebook.
const TRAINING_SAMPLE_SEED: u64 = 0x5A_EE_5D_B0_5E_ED_57_01;
impl Collection {
    /// TRAINING TRIGGER: Uses threshold flag for race condition safety
    #[instrument(target = LOG_TARGET, level = "info", skip(self, writers), fields(
        threshold_reached = self.training_threshold_reached.load(Ordering::Acquire),
        has_quantization = self.has_quantization()
    ))]
    pub(super) fn maybe_trigger_training(&self, writers: &MutationGuard<'_>) -> Result<(), String> {
        // Check atomic flag first (fast path)
        if !self.training_threshold_reached.load(Ordering::Acquire) {
            return Ok(());
        }

        // Only proceed if we have quantization config and aren't already
        // trained, on either scheme.
        let Some(config) = &self.dense().quantization_config else {
            return Ok(());
        };
        if self.can_use_quantization() {
            return Ok(());
        }
        info!(target: LOG_TARGET, operation = "training_trigger",
            scheme = config.scheme.type_name(),
            "Training threshold reached - starting quantizer training"
        );
        // The clock is read here, on the path where the quantizer is fitted
        // for the first time, and the stamp travels in the training's record,
        // so a replay installs the stamp the record carries and reads no
        // clock.
        let completed_at = Utc::now().to_rfc3339();
        let trained = self.fit_training(config)?;
        // The record carries everything the fit produced and commits alone,
        // before anything is installed. A record the sink refuses trains
        // nothing: what was fitted is dropped, the fault stands, and the
        // first call after the checkpoint fits again.
        self.record_training(&completed_at, &trained)
            .map_err(|error| error.to_string())?;
        // Once the record has committed, no valid record reaches a failure of
        // the install, so one is a defect that leaves the fault.
        self.install_training(writers, trained, completed_at)
            .inspect_err(|error| self.fault_sink(error))
    }

    /// Fit the quantizer the collection declares to the training sample,
    /// installing nothing.
    ///
    /// A codebook's rerank calibration is measured where the storage mode
    /// reranks, on a quantizer of its own holding the new codebook, since the
    /// collection's own stays untrained until the training's record commits.
    /// Every draw is seeded and every order fixed, so what this returns is a
    /// function of the records that reached the threshold, the calibration's
    /// timing aside.
    #[instrument(target = LOG_TARGET, level = "info", skip(self, config), fields(
        has_pq = self.dense().pq.is_some(),
        int8 = config.is_int8()
    ))]
    pub(super) fn fit_training(&self, config: &QuantizationConfig) -> Result<Trained, String> {
        let sample = self.training_sample(config)?;
        if config.is_int8() {
            let scale = config
                .scheme
                .int8_scale()
                .ok_or("the space is not declared with scalar quantization")?;
            info!(target: LOG_TARGET, operation = "int8_training_start",
                training_vectors = sample.len(),
                scale = scale.name(),
                "Starting scalar quantization training"
            );
            let training_start = Instant::now();
            let codec = Int8Codec::fit(self.dense().dim, &sample)?;
            info!(target: LOG_TARGET, operation = "int8_training_complete",
                dim = codec.dim(),
                duration_ms = training_start.elapsed().as_millis(),
                "Scalar quantization training completed successfully"
            );
            return Ok(Trained::Scales(codec));
        }

        let pq = self.dense().pq.as_ref().ok_or("PQ not available")?;
        let (subvectors, bits) = config.scheme.pq_shape().unwrap_or((0, 0));
        info!(target: LOG_TARGET, operation = "pq_training_start",
            training_vectors = sample.len(),
            subvectors = subvectors,
            bits = bits,
            "Starting PQ training"
        );
        let training_start = Instant::now();
        let centroids = pq.fit(&sample)?;
        info!(target: LOG_TARGET, operation = "pq_training_complete",
            training_vectors = sample.len(),
            duration_ms = training_start.elapsed().as_millis(),
            "PQ training completed successfully"
        );

        // Measured on the data the codebook was just fitted to, while that
        // data and the codebook are both in hand. See `RerankCalibration`.
        // The sample is released with this function's frame, so this is the
        // only point where the measurement is free of a second pass over the
        // records.
        let calibration = self.calibrate_rerank(pq, &centroids, &sample)?;
        let calibration = calibration
            .map(|calibration| {
                info!(target: LOG_TARGET, operation = "rerank_calibration",
                    fetch = calibration.fetch,
                    sample_records = calibration.sample_records,
                    queries = calibration.queries,
                    duration_ms = calibration.millis,
                    "Rerank fetch calibrated from the training sample"
                );
                serde_json::to_string(&calibration)
            })
            .transpose()
            .map_err(|e| format!("the rerank calibration does not serialize: {e}"))?;
        Ok(Trained::Codebook {
            centroids,
            calibration,
        })
    }

    /// Hand the training's record to the sink, carrying the stamp and
    /// everything `trained` holds, and commit it alone. An append that
    /// fails is withdrawn and leaves the fault, as a commit that fails does.
    fn record_training(&self, completed_at: &str, trained: &Trained) -> Result<(), Error> {
        let fitted = match trained {
            Trained::Codebook {
                centroids,
                calibration,
            } => {
                let first = centroids.first();
                FittedRef::Codebook {
                    subvectors: centroids.len() as u32,
                    centroids: first.map_or(0, Vec::len) as u32,
                    width: first.and_then(|c| c.first()).map_or(0, Vec::len) as u32,
                    values: Values::Nested(centroids),
                    calibration: calibration.as_deref(),
                }
            }
            Trained::Scales(codec) => FittedRef::Scales(codec.scales()),
        };
        let encode = |out: &mut Vec<u8>| {
            Operation::encode_train(completed_at, fitted, out);
            // The ceiling was derived for a codebook at the widest
            // declaration, so no payload a declaration allows passes it.
            debug_assert!(
                out.len() <= JOURNAL_MAX_PAYLOAD,
                "a training's record of {} bytes is within the ceiling",
                out.len()
            );
        };
        if let Err(error) = self.record_encoded(OperationKind::Train, encode) {
            return Err(self.withdraw_records(error));
        }
        self.commit_records()
    }

    /// Install what a training fitted: the quantizer, the stamp and the
    /// calibration, the collected ids cleared, and the graph rebuilt over the
    /// codes or the rows. The live training reaches this once its record has
    /// committed, and a replay of the record reaches it with what the record
    /// carries, so the two install the same state.
    ///
    /// A calibration that does not parse is refused before anything is
    /// installed.
    #[instrument(target = LOG_TARGET, level = "info", skip(self, writers, trained, completed_at))]
    pub(super) fn install_training(
        &self,
        writers: &MutationGuard<'_>,
        trained: Trained,
        completed_at: String,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        match trained {
            Trained::Codebook {
                centroids,
                calibration,
            } => {
                let pq = self.dense().pq.as_ref().ok_or("PQ not available")?.clone();
                let calibration = calibration
                    .map(|json| serde_json::from_str::<RerankCalibration>(&json))
                    .transpose()
                    .map_err(|e| format!("the rerank calibration does not parse: {e}"))?;
                pq.install(centroids)?;
                // The one point where the codebook goes from absent to
                // fitted, so it is the one point that stamps when that
                // happened. See the field.
                *self.dense().training_completed_at.write().unwrap() = Some(completed_at);
                if let Some(calibration) = calibration {
                    *self.dense().rerank_calibration.write().unwrap() = Some(calibration);
                }
                self.release_training_ids();

                debug!(target: LOG_TARGET, operation = "pq_rebuild_start",
                    "Rebuilding index with quantization"
                );
                let rebuild_start = Instant::now();
                let rebuilt = self
                    .rebuild_with_quantization_locked(writers)
                    .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
                if !rebuilt {
                    error!(target: LOG_TARGET, operation = "pq_rebuild", "Index rebuild returned false");
                    return Err("Index rebuild returned false".to_string());
                }
                // The code size against the vector size. A
                // `memory_savings_percent` field used to sit beside it,
                // carrying 1 - 1/compression_ratio, which is the same number in
                // another form and was labelled as a saving the index does not
                // make.
                let compression_ratio = (self.dense().dim as f64 * 4.0) / pq.subvectors() as f64;
                info!(target: LOG_TARGET, operation = "pq_complete",
                    rebuild_duration_ms = rebuild_start.elapsed().as_millis(),
                    compression_ratio = compression_ratio,
                    total_duration_ms = start_time.elapsed().as_millis(),
                    "Index successfully rebuilt with quantization"
                );
            }
            Trained::Scales(codec) => {
                // The one point where the scales go from absent to fitted,
                // stamped as the codebook's path is.
                *self.dense().training_completed_at.write().unwrap() = Some(completed_at);
                self.release_training_ids();
                // The codec becomes the space's, and its presence is the
                // trained flag every reader asks; see the field.
                if self.dense().install_int8_codec(Arc::new(codec)).is_err() {
                    error!(target: LOG_TARGET, operation = "int8_training",
                        "The scalar codec was already fitted"
                    );
                    return Err("The scalar codec was already fitted".to_string());
                }
                debug!(target: LOG_TARGET, operation = "int8_rebuild_start",
                    "Rebuilding index over scalar rows"
                );
                let rebuild_start = Instant::now();
                let rebuilt = self
                    .rebuild_with_int8_locked(writers)
                    .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
                if !rebuilt {
                    error!(target: LOG_TARGET, operation = "int8_rebuild", "Index rebuild returned false");
                    return Err("Index rebuild returned false".to_string());
                }
                info!(target: LOG_TARGET, operation = "int8_complete",
                    rebuild_duration_ms = rebuild_start.elapsed().as_millis(),
                    total_duration_ms = start_time.elapsed().as_millis(),
                    "Index successfully rebuilt over scalar rows"
                );
            }
        }
        Ok(())
    }

    /// Clear the collected training ids, and their capacity with them.
    /// Collection stops for good once `training_threshold_reached` is set,
    /// which happens on the record that fills the buffer and before training
    /// runs, so nothing pushes into it again for the life of the index and
    /// the slots it held are dead weight. At the default `training_size` of
    /// 10,000 the buffer had grown to 16,384 slots of a 24 byte `String`
    /// header, being 393,216 bytes held inside `index_bookkeeping_memory_mb`
    /// on every trained index.
    fn release_training_ids(&self) {
        let mut training_ids = self.training_ids.write().unwrap();
        training_ids.clear();
        training_ids.shrink_to_fit();
    }

    /// The training sample: every collected record's vector, in the seeded
    /// order, capped at `max_training_vectors`.
    ///
    /// Shared by both schemes, so a scalar fit and a codebook fit see the
    /// same records in the same order, and a replay of either sees what the
    /// original saw.
    fn training_sample(&self, config: &QuantizationConfig) -> Result<Vec<Vec<f32>>, String> {
        // Get consistent training set using collected IDs. The record set and
        // the graph the vectors live in are taken first and in that order, and
        // `training_ids` after both, which is the order every reader that holds
        // them takes them in.
        let training_vectors = {
            let ids = self.ids.read().unwrap();
            let index = self.dense().index.read().unwrap();
            let vectors = index.raw_vectors();
            let training_ids = self.training_ids.read().unwrap();

            // ADD EARLY CHECK:
            if training_ids.is_empty() {
                warn!(target: LOG_TARGET, operation = "pq_training",
                    reason = "no_training_ids",
                    "No training IDs available"
                );
                // Reset threshold to prevent repeated attempts
                self.training_threshold_reached
                    .store(false, Ordering::Release);
                return Err("No training IDs available for training".to_string());
            }

            let mut training_data = Vec::new();
            let mut missing_vectors = 0;

            for id in training_ids.iter() {
                if let Some(vector) = ids.slot_of(id).and_then(|slot| vectors.get(slot)) {
                    training_data.push(vector.to_vec());
                } else {
                    missing_vectors += 1;
                }
            }

            if missing_vectors > 0 {
                warn!(target: LOG_TARGET, operation = "pq_training",
                    missing_vectors = missing_vectors,
                    available_vectors = training_data.len(),
                    "Some training vectors were removed before training"
                );
            }

            debug!(target: LOG_TARGET, operation = "pq_training_dataset",
                collected_ids = training_ids.len(),
                available_vectors = training_data.len(),
                target_size = config.training_size,
                "Training dataset prepared"
            );

            training_data
        };

        if training_vectors.len() < config.training_size {
            error!(target: LOG_TARGET, operation = "pq_training",
                available = training_vectors.len(),
                required = config.training_size,
                "Insufficient vectors for training"
            );
            return Err(format!(
                "Insufficient vectors for training: need {}, have {} (some may have been removed)",
                config.training_size,
                training_vectors.len()
            ));
        }

        // Draw the sample in a seeded random order before anything reads it, so
        // that the codebook, the calibration's queries and every fraction the
        // calibration fits over are random draws rather than slices of
        // insertion order. See `TRAINING_SAMPLE_SEED`.
        let mut training_vectors = training_vectors;
        let mut sample_rng = SeededRng::seed_from_u64(TRAINING_SAMPLE_SEED);
        training_vectors.shuffle(&mut sample_rng);

        // Respect max_training_vectors limit
        let final_training_set = if let Some(max_training) = config.max_training_vectors {
            if training_vectors.len() > max_training {
                debug!(target: LOG_TARGET, operation = "pq_training_limit",
                    available = training_vectors.len(),
                    using = max_training,
                    "Limiting training set size"
                );
                training_vectors.into_iter().take(max_training).collect()
            } else {
                training_vectors
            }
        } else {
            training_vectors
        };

        Ok(final_training_set)
    }

    /// Rebuild the graph as a scalar graph over the fitted codec.
    ///
    /// Encodes every live record's raw vector where the graph being replaced
    /// holds one, which is the training transition, and carries every row
    /// over unchanged where it is already a scalar graph, which is a caller
    /// asking for the rebuild again. The rows are gathered under the storage
    /// guards and the graph is built with none held, as the codebook's
    /// rebuild does. Nothing is held twice: the rows live in the
    /// replacement's store and the raw vectors go with the graph that held
    /// them, which is the whole of how `quantized_only` sheds them. The
    /// values the encode clipped are added to the index's count.
    pub(super) fn rebuild_with_int8_locked(
        &self,
        writers: &MutationGuard<'_>,
    ) -> Result<bool, String> {
        let start_time = Instant::now();
        let Some(codec) = self.dense().int8_codec().cloned() else {
            warn!(target: LOG_TARGET, operation = "rebuild_quantization",
                reason = "int8_not_trained",
                "Cannot rebuild: the scales are not fitted"
            );
            return Ok(false);
        };

        let mut new_hnsw = VectorGraph::new_int8(
            &self.dense().metric,
            self.dense().m(),
            self.expected_size(),
            MAX_LAYER,
            self.dense().ef_construction(),
            codec.clone(),
        );

        // Every live record's row, in position order, which is internal id
        // order, so that two rebuilds of the same records wire the same graph,
        // each with the internal id its node was inserted under, taken under
        // the storage guards and released before the graph is built.
        let (rows, encoded, clipped, released) = {
            let ids = self.ids.read().unwrap();
            let old_index = self.dense().index.read().unwrap();
            let old_hnsw = old_index.graph();
            let live: Vec<usize> = ids.slots().collect();

            let mut rows: Vec<(Vec<i8>, usize)> = Vec::with_capacity(live.len());
            let mut encoded = 0usize;
            let mut clipped = 0u64;
            let mut missing = 0usize;
            let origin_ids = old_hnsw.origin_ids();
            for at in live {
                let node = old_index.node_of(Position::from_slot(at));
                let internal_id = node.map_or(0, |node| origin_ids[node as usize]);
                if let Some(row) = node.and_then(|node| old_hnsw.int8_row_at(node)) {
                    rows.push((row.to_vec(), internal_id));
                } else if let Some(vector) = node.and_then(|node| old_hnsw.raw_vector_at(node)) {
                    let row = new_hnsw.int8_encode(vector)?;
                    clipped += codec.saturated(vector) as u64;
                    rows.push((row, internal_id));
                    encoded += 1;
                } else {
                    missing += 1;
                }
            }
            if missing > 0 {
                error!(target: LOG_TARGET, operation = "int8_rebuild",
                    missing_records = missing,
                    "Refusing to rebuild, some live records hold neither a raw vector nor a scalar row"
                );
                return Err(format!(
                    "{} live records hold neither a raw vector nor a scalar row to rebuild from",
                    missing
                ));
            }
            (rows, encoded, clipped, old_hnsw.raw_count())
        };

        // Every record held has a row, so the row at index `k` is the record
        // at the `k`th position held, which is the position the close-up
        // installing the graph moves it to.
        if !rows.is_empty() {
            let batch: Vec<(&[i8], usize, Position)> = rows
                .iter()
                .enumerate()
                .map(|(k, (row, internal_id))| {
                    (row.as_slice(), *internal_id, Position::from_slot(k))
                })
                .collect();
            new_hnsw.insert_batch_int8(&batch).map_err(|e| {
                error!(target: LOG_TARGET, operation = "int8_rebuild", error = %e, "Failed to insert scalar rows");
                format!("Failed to insert scalar rows: {}", e)
            })?;
        }

        // Built in full before it is installed, and the old graph dropped
        // outside the guards; see `replace_dense_graph`. The clipped count is
        // added once the index holds the graph the rows are in.
        self.replace_dense_graph(writers, new_hnsw);
        if clipped > 0 {
            self.dense().index.read().unwrap().add_saturated(clipped);
        }

        info!(target: LOG_TARGET, operation = "int8_rebuild_complete",
            rows_inserted = rows.len(),
            rows_encoded = encoded,
            values_saturated = clipped,
            raw_vectors_released = released,
            duration_ms = start_time.elapsed().as_millis(),
            "Scalar quantization rebuild completed successfully"
        );
        Ok(true)
    }

    /// Measure how deep this index's codes bury a true neighbour
    ///
    /// Runs once, at training, over the training sample and `centroids`, the
    /// codebook just fitted to it, on a quantizer of `pq`'s shape that holds
    /// that codebook. What it measures, why the queries come from the sample
    /// itself, and how the search scales the result to a larger corpus are
    /// all recorded on `RerankCalibration`.
    ///
    /// Returns `None` where the measurement would be spent for nothing.
    /// `quantized_only` never reranks, so it is not calibrated.
    fn calibrate_rerank(
        &self,
        pq: &PQ,
        centroids: &[Vec<Vec<f32>>],
        sample: &[Vec<f32>],
    ) -> Result<Option<RerankCalibration>, String> {
        let keeps_raw = self
            .dense()
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return Ok(None);
        }

        let fitted = PQ::new(
            pq.dim(),
            pq.subvectors(),
            pq.bits(),
            pq.training_size,
            pq.max_training_vectors,
        );
        fitted.install(centroids.to_vec())?;
        Ok(calibrate_rerank_from_sample(
            &fitted,
            sample,
            raw_distance_fn(&self.dense().metric),
        ))
    }

    /// The body of `rebuild_with_quantization`, with the writers guard already held
    ///
    /// Training reaches this from inside `add`, which owns the guard for the whole
    /// call, so the two entry points are separate rather than one taking the guard
    /// twice and deadlocking on itself.
    ///
    /// Errors are `String`, which `rebuild_with_quantization` turns into the
    /// `Error::Engine` the binding raises as the `RuntimeError` it always
    /// raised, and the training path formats into its own message.
    pub(super) fn rebuild_with_quantization_locked(
        &self,
        writers: &MutationGuard<'_>,
    ) -> Result<bool, String> {
        if self.dense().declares_int8() {
            return self.rebuild_with_int8_locked(writers);
        }

        let start_time = Instant::now();

        let pq = match &self.dense().pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                warn!(target: LOG_TARGET, operation = "rebuild_quantization",
                    reason = "pq_not_trained",
                    "Cannot rebuild: PQ not trained"
                );
                return Ok(false);
            }
        };

        // Create new PQ-based HNSW index
        trace!(target: LOG_TARGET, operation = "rebuild_quantization",
            max_layer = MAX_LAYER,
            "Creating new PQ HNSW index"
        );

        // Quantize every stored raw vector and record the codes, then release
        // the storage guards. Nothing below this block holds one, which is what
        // lets the graph work take its own guards in the declared order rather
        // than under a `vectors` guard taken first.
        //
        // An empty raw store is not an error. Under QuantizedOnly the raw
        // vectors are released the moment training completes, so a trained
        // index in that mode holds codes alone, and the rebuild proceeds from
        // those stored codes exactly as `compact` does. Only an index with
        // neither raw vectors nor codes has nothing to rebuild from.
        let (vector_count, retained) = {
            let ids = self.ids.read().unwrap();
            let index = self.dense().index.read().unwrap();
            let vectors = index.raw_vectors();

            // Every live record that still has a raw vector, in position
            // order, which is internal id order and the order the store walks
            // in. The order is fixed rather than a hash map's, because the
            // codes are written back under it and a rebuild has to be a
            // function of the data alone.
            let with_raw: Vec<(&str, &[f32])> = ids
                .iter()
                .filter_map(|(at, id)| vectors.get(at).map(|vector| (id, vector)))
                .collect();

            if with_raw.is_empty() {
                let code_count = self.dense().pq_codes.read().unwrap().len();
                if code_count == 0 {
                    warn!(target: LOG_TARGET, operation = "rebuild_quantization",
                        reason = "no_vectors_or_codes",
                        "Cannot rebuild: no vectors or codes available"
                    );
                    return Ok(false);
                }
                info!(target: LOG_TARGET, operation = "quantization_rebuild_start",
                    vector_count = 0,
                    codes_retained = code_count,
                    "Starting quantization rebuild from stored codes"
                );
                (0, code_count)
            } else {
                info!(target: LOG_TARGET, operation = "quantization_rebuild_start",
                    vector_count = with_raw.len(),
                    "Starting quantization rebuild"
                );

                let vector_refs: Vec<&[f32]> = with_raw.iter().map(|&(_, vector)| vector).collect();
                let quantized_codes = pq.quantize_batch(&vector_refs).map_err(|e| {
                    error!(target: LOG_TARGET, operation = "quantization_rebuild", error = %e, "Failed to quantize vectors");
                    format!("Failed to quantize vectors: {}", e)
                })?;

                // Store quantized codes. Codes for records that have no raw vector
                // are kept rather than cleared, because under QuantizedOnly they
                // are the only copy of every record added after training completed
                // and there is nothing left to re-quantize them from. Clearing
                // dropped those records from the index outright. Removal already
                // deletes an id's codes, so nothing stale can survive here.
                let mut pq_codes = self.dense().pq_codes.write().unwrap();
                let retained = pq_codes
                    .keys()
                    .filter(|id| !ids.slot_of(id).is_some_and(|slot| vectors.contains(slot)))
                    .count();

                for (i, (id, _)) in with_raw.iter().enumerate() {
                    if i < quantized_codes.len() {
                        pq_codes.insert(id.to_string(), quantized_codes[i].clone());
                    }
                }
                debug!(target: LOG_TARGET, operation = "quantization_rebuild",
                    codes_stored = pq_codes.len(),
                    codes_retained = retained,
                    "Quantized codes stored"
                );
                (with_raw.len(), retained)
            }
        };

        // The codes are copied out so the graph is built with no lock held at
        // all, which keeps the storage guards free while the insertions run.
        // Copying costs one byte per subvector per record. Each carries the
        // record's position and the internal id its node was inserted under.
        let batch_data: Vec<(Vec<u8>, usize, usize)> = {
            let ids = self.ids.read().unwrap();
            let old_index = self.dense().index.read().unwrap();
            let pq_codes = self.dense().pq_codes.read().unwrap();
            let internal = old_index.ids_by_position();
            pq_codes
                .iter()
                .filter_map(|(id, codes)| {
                    let at = ids.slot_of(id)?;
                    let internal_id = internal.get(at)?;
                    Some((codes.clone(), at, internal_id))
                })
                .collect()
        };

        // Position order, which is internal id order and arrival order,
        // rather than the order a hash map hands its entries out. The level
        // generator is seeded and the codebook is trained under a seed, so
        // insertion order was the one remaining draw deciding how this graph
        // is wired. `compact` and the persistence rebuild take the same order
        // for the same reason.
        let mut batch_data = batch_data;
        batch_data.sort_unstable_by_key(|&(_, at, _)| at);

        let mut new_hnsw = VectorGraph::new_pq(
            &self.dense().metric,
            self.dense().m(),
            self.expected_size(),
            MAX_LAYER,
            self.dense().ef_construction(),
            pq.clone(),
        );

        // Each record at its place among the records held, counting from
        // zero, which is the position the close-up installing the graph moves
        // it to. A held record the codes did not reach is refused below.
        if !batch_data.is_empty() {
            let batch: Vec<(&Vec<u8>, usize, Position)> = batch_data
                .iter()
                .enumerate()
                .map(|(k, (codes, _, internal_id))| (codes, *internal_id, Position::from_slot(k)))
                .collect();
            new_hnsw.insert_batch_pq(&batch)
                .map_err(|e| {
                    error!(target: LOG_TARGET, operation = "quantization_rebuild", error = %e, "Failed to insert quantized vectors");
                    format!("Failed to insert quantized vectors: {}", e)
                })?;
        }

        // Every record the index holds takes a node, so a held record the
        // codes did not reach refuses the rebuild before anything is swapped,
        // rather than leaving it outside the graph.
        let held = self.ids.read().unwrap().len();
        if new_hnsw.nb_points() != held {
            error!(target: LOG_TARGET, operation = "quantization_rebuild",
                nodes = new_hnsw.nb_points(),
                held_records = held,
                "Refusing the rebuild, some held records have no codes"
            );
            return Err(format!(
                "the quantized rebuild built {} nodes for the {} records the index holds",
                new_hnsw.nb_points(),
                held
            ));
        }

        // The raw vectors survive the transition, where the storage mode keeps
        // them. This is the hardest thing the change to a node addressed store
        // has to do. The graph being replaced is a raw graph, so its store is
        // the only copy of every raw vector the index holds; the replacement is
        // a quantized graph, whose own store holds one byte per subvector. So
        // the raws are carried into a second store beside the codes, addressed
        // by the replacement's node numbering rather than the old one, before
        // the old graph is dropped.
        //
        // Under QuantizedOnly nothing is carried, and that is the whole of how
        // the mode sheds its training records: the raws go when the graph that
        // held them goes. It used to be an explicit clear of a separate map.
        let keeps_raw = self
            .dense()
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        let released = {
            let old_index = self.dense().index.read().unwrap();
            let old_hnsw = old_index.graph();
            let held = old_hnsw.raw_count();
            if keeps_raw && old_hnsw.holds_raw() {
                // Each record's raw vector from the node it held in the graph
                // being replaced, by the position the replacement files it at.
                let source: Vec<u32> = batch_data
                    .iter()
                    .map(|&(_, at, _)| old_index.node_at()[at])
                    .collect();
                new_hnsw
                    .adopt_raw_from(old_hnsw, &source, self.dense().dim)
                    .map_err(|e| format!("Failed to carry the raw vectors over: {}", e))?;
                // The store that carries them is opened at the number of
                // records present now, which is the training sample, and the
                // rest of the corpus is still coming. Every record after this
                // would push into a block sized for a fraction of the index and
                // the block would double its way up, ending at close to twice
                // what it holds: measured at 100,000 records of dimension 100,
                // a store holding 400.0 bytes a record of vectors had been
                // asked for 640.1. Reserving for the declared size makes it the
                // size the caller asked for, under the same byte budget the
                // graph's own arenas are reserved under.
                new_hnsw.reserve_raw_for(self.expected_size());
                0
            } else {
                held
            }
        };

        // The replacement is built in full before it is installed, so a search
        // running alongside this sees the old graph or the new one and never a
        // partly filled one. It used to see the empty new graph for as long as
        // the insertions took.
        //
        // The old graph is moved out and dropped after the guards are released.
        // See `replace_dense_graph`.
        self.replace_dense_graph(writers, new_hnsw);
        let released = if vector_count > 0 && !keeps_raw {
            released
        } else {
            0
        };

        // ✅ ENTERPRISE: Add duration timing with fixed compression ratio calculation
        let duration_ms = start_time.elapsed().as_millis();
        let compression_ratio = (pq.dim() as f64 * 4.0) / pq.subvectors() as f64;
        info!(target: LOG_TARGET, operation = "quantization_rebuild_complete",
            vector_count = vector_count,
            codes_inserted = batch_data.len(),
            codes_retained = retained,
            raw_vectors_released = released,
            compression_ratio = compression_ratio,
            duration_ms = duration_ms,
            "Quantization rebuild completed successfully"
        );

        Ok(true)
    }

    /// Rebuild the graph over the codes after training is complete.
    ///
    /// Re-encodes whatever raw vectors the index still holds through the
    /// trained codebook and rebuilds the graph from the stored codes. It never
    /// retrains the codebook; training runs exactly once, on the `add` that
    /// reaches `training_size`. Returns false when there is no trained
    /// quantizer or nothing stored to rebuild from. Takes the mutation guard,
    /// which the training path already holds when it reaches the body, and
    /// hands the record over before the body runs, whether or not the body
    /// turns out to rebuild, since a replay of the record decides that the
    /// same way.
    pub fn rebuild_with_quantization(&self) -> Result<bool, Error> {
        let writers = self.hold_writers();
        self.record(|| Operation::RebuildQuantized)?;
        self.commit_records()?;
        self.rebuild_with_quantization_locked(&writers)
            .map_err(Error::Engine)
    }

    /// How far the training collection has come, as a percentage, and 100 on
    /// an index whose codebook is fitted. Zero on an index declared without
    /// quantization.
    pub fn training_progress(&self) -> f32 {
        if let Some(config) = &self.dense().quantization_config {
            // If the quantizer is fitted, always return 100%
            if self.can_use_quantization() {
                return 100.0;
            }
            let training_ids = self.training_ids.read().unwrap();
            (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        if let Some(config) = &self.dense().quantization_config {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                0
            } else {
                let training_ids = self.training_ids.read().unwrap();
                config.training_size.saturating_sub(training_ids.len())
            }
        } else {
            0
        }
    }

    /// Check if training is ready to be triggered
    pub fn is_training_ready(&self) -> bool {
        self.training_threshold_reached.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::TRAINING_SAMPLE_SEED;
    use crate::{calibrate_rerank_from_sample, raw_distance_fn};
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use zeusdb_vector_core::{test_support::clustered, SeededRng, PQ};

    /// The shuffle the training sample is drawn in is fixed by its seed, so two
    /// builds over the same records produce the same sample order and two
    /// calibrations over the same codebook produce the same numbers.
    #[test]
    fn the_training_sample_shuffle_is_reproducible() {
        let sample = clustered(500, 32, 909);

        let shuffled = |seed: u64| {
            let mut copy = sample.clone();
            copy.shuffle(&mut SeededRng::seed_from_u64(seed));
            copy
        };

        // The same seed twice is the same order, and it is not the order the
        // records arrived in.
        assert_eq!(
            shuffled(TRAINING_SAMPLE_SEED),
            shuffled(TRAINING_SAMPLE_SEED)
        );
        assert_ne!(shuffled(TRAINING_SAMPLE_SEED), sample);

        // A different seed is a different order, so the fixed seed is doing the
        // work rather than the shuffle being a no-op.
        assert_ne!(
            shuffled(TRAINING_SAMPLE_SEED),
            shuffled(TRAINING_SAMPLE_SEED ^ 1)
        );

        // Every record survives it. A shuffle that dropped or duplicated one
        // would change the codebook as well as the order.
        let mut before: Vec<Vec<f32>> = sample.clone();
        let mut after = shuffled(TRAINING_SAMPLE_SEED);
        let key = |v: &Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        before.sort_by_key(key);
        after.sort_by_key(key);
        assert_eq!(before, after);

        // And the calibration over the shuffled sample is reproducible, given
        // the codebook. One codebook is fitted here and both calibrations read
        // it, so what this holds is the calibration alone.
        let pq = PQ::new(32, 8, 6, 500, None);
        pq.train(&after).unwrap();
        let first = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        let second = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        assert_eq!(first.fetch, second.fetch);
        assert_eq!(first.fit_fetches, second.fit_fetches);
        assert_eq!(first.exponent, second.exponent);
    }
}
