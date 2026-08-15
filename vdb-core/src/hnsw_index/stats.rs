//! What the index reports about itself.
//!
//! `collect_stats` is the memory accounting as well as the counters, and it is
//! the one call that prices every structure the index holds. The four reports
//! beside it are the bodies of Python methods that do nothing but ask this file
//! a question, so the documented surface stays on the method and the work stays
//! here.

use super::{HNSWIndex, StorageMode};
use crate::rerank::{default_rerank_fetch, RERANK_CALIBRATION_PAGES, RERANK_CALIBRATION_TOP_K};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, info};

/// Control bytes a hashbrown table allocates past its last bucket.
///
/// The table is `buckets` entries followed by `buckets + Group::WIDTH` control
/// bytes, and the group is sixteen on every x86 target and on aarch64.
const HASH_CONTROL_TAIL: usize = 16;

/// Buckets a `HashMap` holds, from the capacity it reports.
///
/// hashbrown sizes a table as the smallest power of two at or above `cap * 8 / 7`
/// and then reports seven eighths of that back as the capacity, so undoing the
/// division and rounding up to the power of two recovers the count. Below eight
/// buckets it reports one less than the count instead, which is what the floor
/// covers: an allocated table never holds fewer than four buckets.
fn table_buckets(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    (capacity.saturating_mul(8) / 7).next_power_of_two().max(4)
}

/// Bytes a `HashMap`'s own table asks the allocator for.
///
/// The bucket array and the control bytes. What a key or a value owns on the
/// heap is not counted here, because only the caller knows what that is: the
/// `Vec` in `vectors` is a 24 byte header inside the bucket and 6,144 bytes
/// outside it at dimension 1,536, and that outside part is already reported as
/// `raw_vectors_memory_mb`.
fn table_bytes<K, V, S>(map: &HashMap<K, V, S>) -> usize {
    let buckets = table_buckets(map.capacity());
    if buckets == 0 {
        return 0;
    }
    buckets * std::mem::size_of::<(K, V)>() + buckets + HASH_CONTROL_TAIL
}

/// Bytes the record ids in a map's keys hold on the heap.
///
/// A `String` is a 24 byte header inside the bucket and its text outside it.
/// The scan is over the bucket array rather than over the heap, since a
/// `String`'s length sits in the header, so it is a linear read of a few
/// megabytes and not one pointer chase per record.
fn key_text_bytes<V, S>(map: &HashMap<String, V, S>) -> usize {
    map.keys().map(|id| id.len()).sum()
}

impl HNSWIndex {
    /// Every counter and every memory figure, as `get_stats` reports them.
    ///
    /// Every guard is taken alone, in the declared order, released within its
    /// own statement, and the map is built from the captured values with
    /// nothing held. This used to hold `vectors`, `pq_codes` and `training_ids`
    /// for the whole body, and three things inside that hold inverted the
    /// declared order. It took `id_map` for the live record count, which
    /// deadlocked against `remove_point_internal` holding `id_map` and waiting
    /// on `vectors`, and a three thread loop of stats, adds and removes froze
    /// the process inside a minute. It reached the graph lock through
    /// `is_quantized` and `get_storage_mode`, an inversion with no partner
    /// because every graph write holds nothing else. And it re-read
    /// `training_ids` inside `get_training_progress` and
    /// `training_vectors_needed`, which deadlocks the moment a writer queues
    /// between the two reads, because the standard library's lock queues
    /// readers behind waiting writers. The same loop without removes froze on
    /// exactly that.
    ///
    /// The counts are therefore point in time rather than one snapshot, which
    /// a statistics call can afford: mutations are serialised by `writers`, so
    /// the figures can disagree by at most the records one in flight operation
    /// has partially applied, and no caller reads two of these keys as a
    /// consistency check. Nothing here re-enters a helper that locks; the
    /// storage mode, the training progress and the vectors still needed are
    /// derived from the captured values by the same rules the helpers apply.
    pub(super) fn collect_stats(&self) -> HashMap<String, String> {
        // The record count and, from the same guard, what the map itself
        // occupies. Every `bookkeeping` term below is one structure's own
        // storage, and none of them is the payload the five memory keys price.
        // See `index_bookkeeping_memory_mb`.
        let (live, mut bookkeeping) = {
            let id_map = self.id_map.read().unwrap();
            (id_map.len(), table_bytes(&id_map) + key_text_bytes(&id_map))
        };

        // The reverse map holds a second copy of every id, as a value rather
        // than as a key, so its text is counted again on purpose.
        bookkeeping += {
            let rev_map = self.rev_map.read().unwrap();
            table_bytes(&rev_map) + rev_map.values().map(|id| id.len()).sum::<usize>()
        };

        // Nodes the graph holds, which exceeds the live record count by exactly the
        // number of nodes removal and overwrite have stranded. `compact` reclaims the
        // difference. The quantized flag rides along so nothing below has to go
        // back to the graph lock for it.
        let (graph_nodes, graph_memory_mb, graph_quantized) = {
            let hnsw = self.hnsw.read().unwrap();
            (
                hnsw.nb_points(),
                hnsw.memory_bytes() as f64 / (1024.0 * 1024.0),
                hnsw.is_quantized(),
            )
        };

        let raw_vector_count = {
            let vectors = self.vectors.read().unwrap();
            bookkeeping += table_bytes(&vectors) + key_text_bytes(&vectors);
            vectors.len()
        };
        let pq_code_count = {
            let pq_codes = self.pq_codes.read().unwrap();
            bookkeeping += table_bytes(&pq_codes) + key_text_bytes(&pq_codes);
            pq_codes.len()
        };

        // The metadata map holds an id and a map per record. The inner maps are
        // empty on a record added without metadata, and an empty `HashMap`
        // allocates nothing, so the outer table is the whole cost there.
        bookkeeping += {
            let vector_metadata = self.vector_metadata.read().unwrap();
            let mut bytes = table_bytes(&vector_metadata) + key_text_bytes(&vector_metadata);
            for fields in vector_metadata.values() {
                bytes += table_bytes(fields) + key_text_bytes(fields);
                // A `Value` is 32 bytes inside its bucket whatever it holds. A
                // string is the one variant that also owns text, and it is the
                // variant a filterable field usually is.
                bytes += fields
                    .values()
                    .filter_map(|value| value.as_str())
                    .map(|text| text.len())
                    .sum::<usize>();
            }
            bytes
        };

        let training_id_count = {
            let training_ids = self.training_ids.read().unwrap();
            bookkeeping += training_ids.capacity() * std::mem::size_of::<String>()
                + training_ids.iter().map(|id| id.len()).sum::<usize>();
            training_ids.len()
        };

        bookkeeping += {
            let metadata = self.metadata.lock().unwrap();
            table_bytes(&metadata)
                + key_text_bytes(&metadata)
                + metadata.values().map(|value| value.len()).sum::<usize>()
        };

        let vector_count = *self.vector_count.lock().unwrap();

        // Loaded once so the three keys derived from it agree with each other.
        let threshold_reached = self.training_threshold_reached.load(Ordering::Acquire);

        // `rerank_calibration` sits outside the declared order and is never
        // held together with another guard; see the order note on `HNSWIndex`.
        let calibration = self.get_rerank_calibration();

        // The quantizer's own locks are leaves: nothing in `pq.rs` can reach an
        // index guard, so its accessors are safe wherever they are called.
        let pq_trained = self.pq.as_ref().is_some_and(|pq| pq.is_trained());

        // What `is_quantized` answers, from the captured flag.
        let quantization_active = pq_trained && graph_quantized;

        let mut stats = HashMap::new();

        // Basic stats
        stats.insert("total_vectors".to_string(), vector_count.to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("index_type".to_string(), "HNSW".to_string());

        stats.insert("m".to_string(), self.m.to_string());
        stats.insert(
            "ef_construction".to_string(),
            self.ef_construction.to_string(),
        );
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());

        stats.insert("graph_nodes".to_string(), graph_nodes.to_string());
        stats.insert(
            "stranded_graph_nodes".to_string(),
            graph_nodes.saturating_sub(vector_count).to_string(),
        );

        // The memory keys, reported on every index rather than only on a
        // quantized one. `raw_vectors_memory_mb` used to sit inside the
        // quantization branch, so an unquantized index reported no memory at
        // all, and `graph_memory_mb` did not exist, so no index reported the
        // largest thing it holds. Both are additions. Nothing that reads a key
        // this call already returned sees a different value, which matters
        // because the langchain adapter forwards `memory_mb` from
        // `get_quantization_info` verbatim and that key is untouched.
        let raw_memory_mb = (raw_vector_count * self.dim * 4) as f64 / (1024.0 * 1024.0);
        let mut total_memory_mb = graph_memory_mb + raw_memory_mb;
        stats.insert(
            "graph_memory_mb".to_string(),
            format!("{:.2}", graph_memory_mb),
        );
        stats.insert(
            "raw_vectors_memory_mb".to_string(),
            format!("{:.2}", raw_memory_mb),
        );

        // Storage breakdown
        stats.insert(
            "raw_vectors_stored".to_string(),
            raw_vector_count.to_string(),
        );
        stats.insert(
            "quantized_codes_stored".to_string(),
            pq_code_count.to_string(),
        );

        // Training info
        if let Some(config) = &self.quantization_config {
            stats.insert("quantization_type".to_string(), "pq".to_string());
            stats.insert(
                "quantization_training_size".to_string(),
                config.training_size.to_string(),
            );

            // Storage mode information
            stats.insert(
                "storage_mode".to_string(),
                config.storage_mode.to_string().to_string(),
            );

            // Calculate actual memory usage based on storage mode. The raw
            // vector figure is reported above, on every index rather than only
            // on this branch.
            let quantized_memory_mb =
                (pq_code_count * config.subvectors) as f64 / (1024.0 * 1024.0);
            total_memory_mb += quantized_memory_mb;

            stats.insert(
                "quantized_codes_memory_mb".to_string(),
                format!("{:.2}", quantized_memory_mb),
            );

            // `memory_savings` used to sit beside `storage_strategy`, reading
            // "maximum" under QuantizedOnly. That mode is the smaller of the two
            // and it is not a maximum of anything. Measured at 3,000 records of
            // 64 dimensions it held more resident memory than the same index
            // unquantized, because the centroid distance table is a fixed 1 MB.
            // What replaces it is the fact the figures above can be checked
            // against, being which records still have a raw vector. Under
            // QuantizedOnly that is every record while the index is still
            // collecting for training and none from the moment training
            // completes, since the rebuild releases the training records once
            // their codes are stored.
            match config.storage_mode {
                StorageMode::QuantizedOnly => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "memory_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "none_once_trained".to_string(),
                    );
                }
                StorageMode::QuantizedWithRaw => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "quality_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "all_records".to_string(),
                    );
                }
            }

            // What `get_training_progress` and `training_vectors_needed`
            // compute, from the captured count rather than from a second read
            // of the guard those helpers take themselves.
            let progress = if pq_trained {
                100.0
            } else {
                (training_id_count as f32 / config.training_size as f32 * 100.0).min(100.0)
            };
            stats.insert(
                "training_progress".to_string(),
                format!(
                    "{}/{} ({:.1}%)",
                    training_id_count, config.training_size, progress
                ),
            );

            let vectors_needed = if threshold_reached {
                0
            } else {
                config.training_size.saturating_sub(training_id_count)
            };
            stats.insert(
                "training_vectors_needed".to_string(),
                vectors_needed.to_string(),
            );
            stats.insert(
                "training_threshold_reached".to_string(),
                threshold_reached.to_string(),
            );

            if let Some(pq) = &self.pq {
                stats.insert("quantization_trained".to_string(), pq_trained.to_string());
                stats.insert(
                    "quantization_active".to_string(),
                    quantization_active.to_string(),
                );

                // The two fixed costs, reported here so that the whole memory
                // question can be answered from one call. Both are independent
                // of the record count, and at small record counts the table is
                // the largest single thing a quantized index holds. They were
                // only on `get_quantization_info`, which is where a caller
                // reading the storage breakdown above would not look.
                let (centroid_mb, _) = pq.get_memory_stats();
                let sdc_mb = pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0);
                total_memory_mb += centroid_mb + sdc_mb;
                stats.insert(
                    "codebook_memory_mb".to_string(),
                    format!("{:.2}", centroid_mb),
                );
                stats.insert("sdc_table_memory_mb".to_string(), format!("{:.2}", sdc_mb));

                if pq_trained {
                    let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
                    stats.insert(
                        "quantization_compression_ratio".to_string(),
                        format!("{:.1}x", compression_ratio),
                    );
                }

                // What the default rerank fetch is derived from, and what it
                // resolves to at the record count the index holds now, so a
                // caller can see the number their searches are paying for
                // rather than deriving it. See `RerankCalibration`. Both were
                // captured at the top; the live count read here is what
                // deadlocked against removal.
                match calibration {
                    Some(calibration) => {
                        stats.insert("rerank_calibrated".to_string(), "true".to_string());
                        stats.insert(
                            "rerank_calibration_fetch".to_string(),
                            calibration.fetch.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_records".to_string(),
                            calibration.sample_records.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_queries".to_string(),
                            calibration.queries.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_target_recall".to_string(),
                            format!("{:.3}", calibration.target),
                        );
                        stats.insert(
                            "rerank_calibration_fit_fetches".to_string(),
                            calibration
                                .fit_fetches
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_exponent".to_string(),
                            format!("{:.3}", calibration.exponent),
                        );
                        stats.insert(
                            "rerank_calibration_page_fetches".to_string(),
                            calibration
                                .page_fetches
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_pages".to_string(),
                            RERANK_CALIBRATION_PAGES
                                .iter()
                                .map(|p| p.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_page_exponent".to_string(),
                            format!("{:.3}", calibration.page_exponent),
                        );
                        stats.insert(
                            "rerank_calibration_ms".to_string(),
                            calibration.millis.to_string(),
                        );
                    }
                    None => {
                        stats.insert("rerank_calibrated".to_string(), "false".to_string());
                    }
                }

                // The figure a search at the default page will actually fetch,
                // read from the same rule the search reads rather than restated
                // here. See `default_rerank_fetch`.
                stats.insert(
                    "rerank_default_fetch".to_string(),
                    default_rerank_fetch(calibration, live, RERANK_CALIBRATION_TOP_K).to_string(),
                );
            }
        } else {
            stats.insert("quantization_type".to_string(), "none".to_string());
            stats.insert("storage_mode".to_string(), "raw_only".to_string());
        }

        // What `get_storage_mode` answers, from the captured flags rather than
        // through it, because it reaches the graph lock via `is_quantized` and
        // this used to call it while holding three storage guards.
        let storage_mode_description = if self.quantization_config.is_none() {
            "raw_only"
        } else if !pq_trained {
            if threshold_reached {
                "raw_ready_for_training"
            } else {
                "raw_collecting_for_training"
            }
        } else if quantization_active {
            "quantized_active"
        } else {
            "raw_trained_not_rebuilt"
        };
        stats.insert(
            "storage_mode_description".to_string(),
            storage_mode_description.to_string(),
        );

        // What the index spends on finding a record rather than on holding one.
        //
        // Five hash tables and two id copies per record. `id_map` and `rev_map`
        // each hold the record's id, one as a key and one as a value, and
        // `vectors`, `pq_codes` and `vector_metadata` each hold it again as a
        // key. A table is a power of two bucket array with one control byte per
        // bucket, sized from the capacity the map reports, so it is between
        // eight sevenths and sixteen sevenths of the entries it currently
        // holds and it steps rather than growing smoothly. The `Vec` in
        // `vectors` and in `pq_codes` contributes only its 24 byte header here,
        // since the bytes it points at are already priced above.
        //
        // It is proportional to the record count and independent of the
        // dimension. Measured on three loaded indexes of 50,000 dbpedia-openai
        // records at dimension 1,536 it reads 12.66, 15.95 and 12.66 MiB under
        // no quantization, `quantized_with_raw` and `quantized_only`, being
        // 265, 334 and 265 bytes per record. The middle mode is the one holding
        // both stores, so it carries five tables where the others carry four.
        //
        // This is a request count and not a commitment. The allocator's own
        // headers, its rounding and its fragmentation sit outside it, the same
        // way they sit outside `graph_memory_mb`; see `graph_memory_bytes`.
        let bookkeeping_mb = bookkeeping as f64 / (1024.0 * 1024.0);
        total_memory_mb += bookkeeping_mb;
        stats.insert(
            "index_bookkeeping_memory_mb".to_string(),
            format!("{:.2}", bookkeeping_mb),
        );

        // The sum of the six memory keys above. It is what the index holds in
        // the structures this call can price, being the graph, the raw vector
        // store, the codes, the codebook, the centroid distance table and the
        // bookkeeping.
        //
        // The bookkeeping term is an addition, and it changes this figure on
        // every index. It used to be the sum of five keys and to miss the id
        // maps entirely. Nothing outside this call reads it; the `memory_mb`
        // an integration package forwards comes from `get_quantization_info`
        // and is the codebook alone, which is untouched.
        //
        // It is still not the resident set. The allocator's headers and its
        // fragmentation sit outside it. Measured on three loaded indexes of
        // 50,000 dbpedia-openai records at dimension 1,536, the index held
        // 805.4, 473.3 and 181.0 MiB of resident where the five key sum
        // reported 692.4, 400.0 and 107.0, being 0.860, 0.845 and 0.591 of it,
        // and where this six key sum reports 705.1, 415.9 and 119.7, being
        // 0.876, 0.879 and 0.661.
        //
        // What is left is 2,103, 1,202 and 1,285 bytes per record and it is
        // almost all allocator overhead on the graph, which asks for six small
        // blocks per point. It runs 1.25 times the graph figure unquantized,
        // where the per point data block is 6,144 bytes, and 1.63 and 1.67
        // times under the quantized modes, where that block is 48 bytes.
        stats.insert(
            "total_memory_mb".to_string(),
            format!("{:.2}", total_memory_mb),
        );

        stats
    }

    /// The one line description `info` returns.
    pub(super) fn info_string(&self) -> String {
        let record_count = self.id_map.read().unwrap().len();
        let base_info = format!(
            "HNSWIndex(dim={}, space={}, m={}, ef_construction={}, expected_size={}, vectors={}",
            self.dim, self.space, self.m, self.ef_construction, self.expected_size, record_count
        );

        if let Some(config) = &self.quantization_config {
            let trained_status = self
                .pq
                .as_ref()
                .map(|pq| {
                    if pq.is_trained() {
                        "trained"
                    } else {
                        "untrained"
                    }
                })
                .unwrap_or("unknown");

            let active_status = if self.is_quantized() {
                "active"
            } else {
                "inactive"
            };

            // Use cached compression ratio calculation with proper float division
            let compression_info = self
                .pq
                .as_ref()
                .map(|pq| format!("{:.1}x", (pq.dim as f64 * 4.0) / pq.subvectors as f64))
                .unwrap_or_else(|| "unknown".to_string());

            format!(
                "{}, quantization=pq(subvectors={}, bits={}, {}, {}, compression={}))",
                base_info,
                config.subvectors,
                config.bits,
                trained_status,
                active_status,
                compression_info
            )
        } else {
            format!("{}, quantization=none)", base_info)
        }
    }

    /// The quantization dictionary `get_quantization_info` returns.
    pub(super) fn quantization_info(&self) -> Option<Py<PyAny>> {
        Python::attach(|py| {
            if let Some(config) = &self.quantization_config {
                let dict = PyDict::new(py);
                dict.set_item("type", "pq").ok()?;
                dict.set_item("subvectors", config.subvectors).ok()?;
                dict.set_item("bits", config.bits).ok()?;
                dict.set_item("training_size", config.training_size).ok()?;

                if let Some(max_training) = config.max_training_vectors {
                    dict.set_item("max_training_vectors", max_training).ok()?;
                }

                if let Some(pq) = &self.pq {
                    dict.set_item("is_trained", pq.is_trained()).ok()?;

                    // Use enhanced PQ methods
                    let (memory_mb, total_centroids) = pq.get_memory_stats();
                    dict.set_item("memory_mb", memory_mb).ok()?;
                    dict.set_item("total_centroids", total_centroids).ok()?;

                    // The symmetric distance table graph construction reads.
                    // Reported separately because it is derived from the
                    // codebook rather than part of it, and because it scales
                    // with subvectors and bits alone while memory_mb scales
                    // with the dimension too.
                    dict.set_item(
                        "sdc_memory_mb",
                        pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0),
                    )
                    .ok()?;

                    // Calculate compression ratio using cached values
                    let original_bytes = pq.dim * 4; // f32
                    let compressed_bytes = pq.subvectors; // u8 per subvector
                    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
                    dict.set_item("compression_ratio", compression_ratio).ok()?;
                }

                Some(dict.into())
            } else {
                None
            }
        })
    }

    /// The characteristics `get_performance_info` returns.
    pub(super) fn performance_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("search_speedup_expected".to_string(), "1.2x-2x".to_string());
        info.insert(
            "search_bottleneck".to_string(),
            "hnsw_mutex_serialization".to_string(),
        );
        info.insert(
            "benefits".to_string(),
            "gil_release_concurrent_metadata_processing_batched_search".to_string(),
        );
        info.insert("insertion_path".to_string(), "sequential".to_string());

        // Add quantization performance info
        //
        // `quantization_memory_savings` used to sit here, reporting
        // 1 - 1/compression_ratio as a percentage. That is the share of a
        // vector a code replaces, not the memory an index saves, and under
        // QuantizedWithRaw the index saves nothing on the vectors because it
        // keeps every one of them. It carried no information
        // `quantization_compression` does not, so it is gone rather than
        // qualified. Memory belongs to `get_stats`, which measures rather than
        // projects.
        if let Some(config) = &self.quantization_config {
            let original_bytes = self.dim * 4; // f32
            let compressed_bytes = config.subvectors; // u8 per subvector
            let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

            info.insert(
                "quantization_compression".to_string(),
                format!("{:.1}x", compression_ratio),
            );
            // Measured at 0.16 recall at 10 against 1.00 for the same data
            // unquantized, so "slight" was wrong by a factor the word cannot
            // carry. Only QuantizedWithRaw can rerank, so the two modes get
            // different answers. Rerank recovers most of the loss rather than
            // all of it, and how much depends on the fetch depth, which is why
            // the default fetch is derived from the live record count rather
            // than fixed; see `DEFAULT_RERANK_CORPUS_DIVISOR`.
            info.insert(
                "quantization_accuracy_impact".to_string(),
                match config.storage_mode {
                    StorageMode::QuantizedOnly => "large_recall_loss_no_rerank_available",
                    StorageMode::QuantizedWithRaw => "large_recall_loss_unless_reranked",
                }
                .to_string(),
            );
        }

        info
    }

    /// The measurement `benchmark_concurrent_reads` returns.
    pub(super) fn benchmark_reads(
        &self,
        query_count: usize,
        max_threads: Option<usize>,
    ) -> PyResult<HashMap<String, f64>> {
        use rand::random; // Import for random number generation

        let start_time = Instant::now();

        debug!(
            operation = "benchmark_start",
            query_count = query_count,
            max_threads = max_threads,
            "Starting concurrent read benchmark"
        );

        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| random::<f32>()).collect())
            .collect();

        let mut results = HashMap::new();

        // Sequential benchmark
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        results.insert("sequential_time".to_string(), sequential_time);
        results.insert(
            "sequential_qps".to_string(),
            queries.len() as f64 / sequential_time,
        );

        // Parallel benchmark
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads
            .unwrap_or(available_threads)
            .min(available_threads);

        let start = Instant::now();
        let _: Vec<_> = queries
            .par_iter()
            .map(|query| self.raw_search_no_gil(query))
            .collect();

        let parallel_time = start.elapsed().as_secs_f64();
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert(
            "parallel_qps".to_string(),
            queries.len() as f64 / parallel_time,
        );
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);

        let total_duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "benchmark_complete",
            sequential_qps = queries.len() as f64 / sequential_time,
            parallel_qps = queries.len() as f64 / parallel_time,
            speedup = sequential_time / parallel_time,
            duration_ms = total_duration_ms,
            "Benchmark completed"
        );

        Ok(results)
    }
}
