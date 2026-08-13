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
impl HNSWIndex {
    /// Every counter and every memory figure, as `get_stats` reports them.
    pub(super) fn collect_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        // Nodes the graph holds, which exceeds the live record count by exactly the
        // number of nodes removal and overwrite have stranded. `compact` reclaims the
        // difference. Read first, because the declared lock order puts the graph
        // above every map read below it.
        let (graph_nodes, graph_memory_mb) = {
            let hnsw = self.hnsw.read().unwrap();
            (
                hnsw.nb_points(),
                hnsw.memory_bytes() as f64 / (1024.0 * 1024.0),
            )
        };

        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let training_ids = self.training_ids.read().unwrap();
        let vector_count = *self.vector_count.lock().unwrap();

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
        let raw_memory_mb = (vectors.len() * self.dim * 4) as f64 / (1024.0 * 1024.0);
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
        stats.insert("raw_vectors_stored".to_string(), vectors.len().to_string());
        stats.insert(
            "quantized_codes_stored".to_string(),
            pq_codes.len().to_string(),
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
                (pq_codes.len() * config.subvectors) as f64 / (1024.0 * 1024.0);
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

            let collected_count = training_ids.len();
            let progress = self.get_training_progress();
            stats.insert(
                "training_progress".to_string(),
                format!(
                    "{}/{} ({:.1}%)",
                    collected_count, config.training_size, progress
                ),
            );

            let vectors_needed = self.training_vectors_needed();
            stats.insert(
                "training_vectors_needed".to_string(),
                vectors_needed.to_string(),
            );
            stats.insert(
                "training_threshold_reached".to_string(),
                self.training_threshold_reached
                    .load(Ordering::Acquire)
                    .to_string(),
            );

            if let Some(pq) = &self.pq {
                let is_trained = pq.is_trained();
                stats.insert("quantization_trained".to_string(), is_trained.to_string());
                stats.insert(
                    "quantization_active".to_string(),
                    self.is_quantized().to_string(),
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

                if is_trained {
                    let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
                    stats.insert(
                        "quantization_compression_ratio".to_string(),
                        format!("{:.1}x", compression_ratio),
                    );
                }

                // What the default rerank fetch is derived from, and what it
                // resolves to at the record count the index holds now, so a
                // caller can see the number their searches are paying for
                // rather than deriving it. See `RerankCalibration`.
                let calibration = self.get_rerank_calibration();
                let live = self.id_map.read().unwrap().len();
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

        stats.insert(
            "storage_mode_description".to_string(),
            self.get_storage_mode(),
        );

        // The sum of the five memory keys above. It is what the index holds in
        // the structures this call can price, being the graph, the raw vector
        // store, the codes, the codebook and the centroid distance table.
        //
        // It is not the resident set. The id maps, the metadata map, the hash
        // table slots and the allocator's own headers and fragmentation sit
        // outside it. Measured on three loaded indexes of 50,000
        // dbpedia-openai records at dimension 1,536, the process held 805.9,
        // 474.8 and 181.4 MiB where this reports 692.4, 401.2 and 107.8, being
        // 1.16, 1.18 and 1.68 times. The share it misses is roughly 1,500 bytes
        // per record and it does not move with the dimension, so it dominates
        // the ratio on the mode that holds least.
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
