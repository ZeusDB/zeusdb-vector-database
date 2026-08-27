//! Product quantization training, and the graph rebuild that follows it.
//!
//! Training fires once, on the `add` that reaches `training_size`, and that call
//! does four things in order: it fits the codebook, it measures the rerank fetch
//! on the sample it has just fitted to, it clears the collected ids, and it
//! rebuilds the whole graph over the codes. `quantized_only` sheds its raw
//! vectors at the end of that rebuild, which is the single point where the mode
//! does so.

use super::{Collection, StorageMode, MAX_LAYER};
use crate::{calibrate_rerank_from_sample, raw_distance_fn, RawVectors, RerankCalibration};
use chrono::Utc;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace, warn};
use zeusdb_vector_core::{Error, SeededRng, VectorGraph, PQ};

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
    #[instrument(target = LOG_TARGET, level = "info", skip(self), fields(
        threshold_reached = self.training_threshold_reached.load(Ordering::Acquire),
        has_quantization = self.has_quantization()
    ))]
    pub(super) fn maybe_trigger_training(&self) -> Result<(), String> {
        // Check atomic flag first (fast path)
        if !self.training_threshold_reached.load(Ordering::Acquire) {
            return Ok(());
        }

        // Only proceed if we have quantization config and aren't already trained
        if let Some(_config) = &self.space.quantization_config {
            if let Some(pq) = &self.space.pq {
                if !pq.is_trained() {
                    info!(target: LOG_TARGET, operation = "training_trigger",
                        "Training threshold reached - starting PQ training"
                    );
                    return self.train_quantization_from_ids();
                }
            }
        }

        Ok(())
    }

    /// TRAINING EXECUTION: Uses collected IDs for deterministic training set
    #[instrument(target = LOG_TARGET, level = "info", skip(self), fields(
        has_pq = self.space.pq.is_some(),
        has_config = self.space.quantization_config.is_some()
    ))]
    fn train_quantization_from_ids(&self) -> Result<(), String> {
        let start_time = Instant::now();

        let pq = self.space.pq.as_ref().ok_or("PQ not available")?.clone();
        let config = self
            .space
            .quantization_config
            .as_ref()
            .ok_or("Config not available")?
            .clone();

        // Get consistent training set using collected IDs. The record set and
        // the graph the vectors live in are taken first and in that order, and
        // `training_ids` after both, which is the order every reader that holds
        // them takes them in.
        let training_vectors = {
            let id_map = self.id_map.read().unwrap();
            let hnsw = self.space.hnsw.read().unwrap();
            let vectors = RawVectors {
                id_map: &id_map,
                graph: &hnsw,
            };
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
                if let Some(vector) = vectors.get(id.as_str()) {
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

        info!(target: LOG_TARGET, operation = "pq_training_start",
            training_vectors = final_training_set.len(),
            subvectors = config.subvectors,
            bits = config.bits,
            "Starting PQ training"
        );

        // Train the PQ model
        let training_start = Instant::now();
        pq.train(&final_training_set)?;
        let training_duration = training_start.elapsed();

        // The one point where the codebook goes from absent to fitted, so it is
        // the one point that can stamp when that happened. `quantization.json`
        // used to write the save time under this name. See the field.
        *self.space.training_completed_at.write().unwrap() = Some(Utc::now().to_rfc3339());

        info!(target: LOG_TARGET, operation = "pq_training_complete",
            training_vectors = final_training_set.len(),
            duration_ms = training_duration.as_millis(),
            "PQ training completed successfully"
        );

        // Measure the rerank fetch on the data the codebook was just fitted to,
        // while that data and the codebook are both in hand. See
        // `RerankCalibration`. The training set is released with this function's
        // frame, so this is the only point where the measurement is free of a
        // second pass over the records.
        if let Some(calibration) = self.calibrate_rerank(&pq, &final_training_set) {
            info!(target: LOG_TARGET, operation = "rerank_calibration",
                fetch = calibration.fetch,
                sample_records = calibration.sample_records,
                queries = calibration.queries,
                duration_ms = calibration.millis,
                "Rerank fetch calibrated from the training sample"
            );
            *self.space.rerank_calibration.write().unwrap() = Some(calibration);
        }

        // Clear training IDs (no longer needed)
        {
            let mut training_ids = self.training_ids.write().unwrap();
            training_ids.clear();
        }

        // Rebuild index with quantization
        debug!(target: LOG_TARGET, operation = "pq_rebuild_start",
            "Rebuilding index with quantization"
        );
        let rebuild_start = Instant::now();
        let rebuild_success = self
            .rebuild_with_quantization_locked()
            .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
        let rebuild_duration = rebuild_start.elapsed();

        if rebuild_success {
            // The code size against the vector size. A `memory_savings_percent`
            // field used to sit beside it, carrying 1 - 1/compression_ratio,
            // which is the same number in another form and was labelled as a
            // saving the index does not make.
            let compression_ratio = (self.space.dim as f64 * 4.0) / pq.subvectors() as f64;

            let total_duration_ms = start_time.elapsed().as_millis();
            info!(target: LOG_TARGET, operation = "pq_complete",
                rebuild_duration_ms = rebuild_duration.as_millis(),
                compression_ratio = compression_ratio,
                total_duration_ms = total_duration_ms,
                "Index successfully rebuilt with quantization"
            );
        } else {
            error!(target: LOG_TARGET, operation = "pq_rebuild", "Index rebuild returned false");
            return Err("Index rebuild returned false".to_string());
        }

        Ok(())
    }

    /// Measure how deep this index's codes bury a true neighbour
    ///
    /// Runs once, at training completion, over the training sample and the
    /// codebook just fitted to it. What it measures, why the queries come from
    /// the sample itself, and how the search scales the result to a larger
    /// corpus are all recorded on `RerankCalibration`.
    ///
    /// Returns `None` where the measurement would be spent for nothing.
    /// `quantized_only` never reranks, so it is not calibrated.
    fn calibrate_rerank(&self, pq: &PQ, sample: &[Vec<f32>]) -> Option<RerankCalibration> {
        let keeps_raw = self
            .space
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        calibrate_rerank_from_sample(pq, sample, raw_distance_fn(&self.space.metric))
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
    pub(super) fn rebuild_with_quantization_locked(&self) -> Result<bool, String> {
        let start_time = Instant::now();

        let pq = match &self.space.pq {
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
            let id_map = self.id_map.read().unwrap();
            let hnsw = self.space.hnsw.read().unwrap();
            let vectors = RawVectors {
                id_map: &id_map,
                graph: &hnsw,
            };

            // Every live record that still has a raw vector, in internal id
            // order. The order is fixed rather than a hash map's, because the
            // codes are written back under it and a rebuild has to be a
            // function of the data alone.
            let mut live: Vec<(&String, usize)> = id_map
                .iter()
                .map(|(id, &internal)| (id, internal))
                .collect();
            live.sort_unstable_by_key(|&(_, internal_id)| internal_id);
            let with_raw: Vec<(&String, &[f32])> = live
                .iter()
                .filter_map(|&(id, internal_id)| {
                    hnsw.raw_vector(internal_id).map(|vector| (id, vector))
                })
                .collect();

            if with_raw.is_empty() {
                let code_count = self.space.pq_codes.read().unwrap().len();
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
                let mut pq_codes = self.space.pq_codes.write().unwrap();
                let retained = pq_codes
                    .keys()
                    .filter(|id| !vectors.contains(id.as_str()))
                    .count();

                for (i, (id, _)) in with_raw.iter().enumerate() {
                    if i < quantized_codes.len() {
                        pq_codes.insert((*id).clone(), quantized_codes[i].clone());
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
        // Copying costs one byte per subvector per record.
        let batch_data: Vec<(Vec<u8>, usize)> = {
            let id_map = self.id_map.read().unwrap();
            let pq_codes = self.space.pq_codes.read().unwrap();
            pq_codes
                .iter()
                .filter_map(|(id, codes)| {
                    id_map
                        .get(id)
                        .map(|&internal_id| (codes.clone(), internal_id))
                })
                .collect()
        };

        // Internal id order, which is arrival order, rather than the order a
        // hash map hands its entries out. The level generator is seeded and the
        // codebook is trained under a seed, so insertion order was the one
        // remaining draw deciding how this graph is wired. `compact` and the
        // persistence rebuild already sort for the same reason.
        let mut batch_data = batch_data;
        batch_data.sort_unstable_by_key(|&(_, internal_id)| internal_id);

        let mut new_hnsw = VectorGraph::new_pq(
            &self.space.metric,
            self.space.m(),
            self.expected_size(),
            MAX_LAYER,
            self.space.ef_construction(),
            pq.clone(),
        );

        if !batch_data.is_empty() {
            let batch: Vec<(&Vec<u8>, usize)> = batch_data
                .iter()
                .map(|(codes, internal_id)| (codes, *internal_id))
                .collect();
            new_hnsw.insert_batch_pq(&batch)
                .map_err(|e| {
                    error!(target: LOG_TARGET, operation = "quantization_rebuild", error = %e, "Failed to insert quantized vectors");
                    format!("Failed to insert quantized vectors: {}", e)
                })?;
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
            .space
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        let released = {
            let old_hnsw = self.space.hnsw.read().unwrap();
            let held = old_hnsw.raw_count();
            if keeps_raw && old_hnsw.holds_raw() {
                new_hnsw
                    .adopt_raw_from(&old_hnsw, self.space.dim)
                    .map_err(|e| format!("Failed to carry the raw vectors over: {}", e))?;
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
        // The old graph is moved out and dropped after the guard is released.
        // See `replace_graph`.
        self.space.replace_graph(new_hnsw);
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
    /// which the training path already holds when it reaches the body.
    pub fn rebuild_with_quantization(&self) -> Result<bool, Error> {
        let _writers = self.writers.lock().unwrap();
        self.rebuild_with_quantization_locked()
            .map_err(Error::Engine)
    }

    /// How far the training collection has come, as a percentage, and 100 on
    /// an index whose codebook is fitted. Zero on an index declared without
    /// quantization.
    pub fn training_progress(&self) -> f32 {
        if let Some(config) = &self.space.quantization_config {
            // If PQ is trained, always return 100%
            if let Some(pq) = &self.space.pq {
                if pq.is_trained() {
                    return 100.0;
                }
            }
            let training_ids = self.training_ids.read().unwrap();
            (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        if let Some(config) = &self.space.quantization_config {
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
        // the codebook. The codebook itself is fitted by unseeded k-means, so
        // it is trained once and both calibrations read it.
        let pq = PQ::new(32, 8, 6, 500, None);
        pq.train(&after).unwrap();
        let first = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        let second = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        assert_eq!(first.fetch, second.fetch);
        assert_eq!(first.fit_fetches, second.fit_fetches);
        assert_eq!(first.exponent, second.exponent);
    }
}
