//! Running a search, on one query or on a batch.
//!
//! Four paths reach the graph: one query, a batch of five or fewer run in turn,
//! a larger batch fanned across rayon, and the benchmark. They differ only in
//! how long they hold the storage guards, so what each does with a candidate is
//! in `collect_hits` and what each hands back to Python is in `hits_to_python`.
//!
//! **Lock order.** Every path takes `rev_map` before the graph and the storage
//! maps after it, which is the order declared on `HNSWIndex` in the parent
//! module. A search holds `rev_map` for its whole traversal, so a mutation
//! taking `vectors` before `rev_map` deadlocks against it, and that is the
//! inversion `remove_point_internal` used to carry.

use super::{HNSWIndex, StorageMode};
use crate::conversion::value_map_to_python;
use crate::filter::matches_filter;
use crate::graph::GraphHit;
use crate::rerank::{raw_distance_fn, rescore_candidate, take_best, RerankPlan, SearchParams};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, error, instrument, trace};
/// Search hits for one query vector, as (external id, distance, metadata,
/// optional raw vector). The raw vector is present only when the caller asked
/// for it and the index still holds one.
pub(super) type QueryHits = Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>;
impl HNSWIndex {
    // 2. SEARCH OPERATIONS (2 methods)

    /// Decide whether a search reranks, and how far it over-fetches
    ///
    /// Rerank rescores the candidates the graph returns against raw vectors,
    /// so it needs a raw vector for every candidate. Three cases resolve to no
    /// rerank.
    ///
    /// A raw index already ranks by the raw distance, so over-fetching and
    /// rescoring would return the same page at a higher cost.
    ///
    /// A `quantized_only` index holds no raw vectors once trained, the
    /// training records included, so the only thing available to rescore any
    /// candidate against is its reconstruction, and that carries exactly the
    /// information the ADC distance already used. Measured at 10,000 records
    /// of dimension 768, recall at `top_k` 10 over code held records moved
    /// from 0.1320 to 0.1330 across one data seed and from 0.1440 to 0.1400
    /// across another, which is noise in both directions.
    ///
    /// `rerank = 0` from the caller turns it off and restores the ADC scores.
    pub(super) fn rerank_plan(&self, rerank: Option<usize>) -> Option<RerankPlan> {
        if rerank == Some(0) || !self.is_quantized() {
            return None;
        }

        let keeps_raw = self
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        Some(RerankPlan {
            factor: rerank.map(|factor| factor.max(1)),
            calibration: self.get_rerank_calibration(),
            distance: raw_distance_fn(&self.space),
        })
    }

    // 5. BATCH SEARCH METHODS (3 methods)

    /// Resolve, filter, score and cut one query's candidates
    ///
    /// The single query path and the two batch paths each held their own copy
    /// of this. The three copies agreed, and a rule that has to hold across
    /// every search path is one that must not be stated three times: the
    /// reconstruction fallback for `return_vector` was added to the batch
    /// copies in a later relay than the single one, and for a while the three
    /// disagreed about what a `quantized_only` index hands back.
    ///
    /// The guards are taken by the caller and passed in, because each path
    /// holds them for a different span. The single query path and the
    /// sequential batch path hold one set across every query, the parallel path
    /// takes its own per worker, and none of that is this function's business.
    ///
    /// Only a borrowed id and a float are held per candidate until the page is
    /// cut, so an over-fetched page pays to clone the metadata and the vector of
    /// the results it returns rather than of every candidate it considered.
    #[allow(clippy::too_many_arguments)]
    fn collect_hits(
        &self,
        graph_hits: Vec<GraphHit>,
        query: &[f32],
        rev_map: &HashMap<usize, String>,
        vectors: &HashMap<String, Vec<f32>>,
        pq_codes: &HashMap<String, Vec<u8>>,
        vector_metadata: &HashMap<String, HashMap<String, Value>>,
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
    ) -> PyResult<QueryHits> {
        let mut scored: Vec<(&String, f32)> = Vec::with_capacity(graph_hits.len());

        for neighbor in graph_hits {
            let internal_id = neighbor.internal_id;

            if let Some(ext_id) = rev_map.get(&internal_id) {
                if let Some(filter_conds) = filter_conditions {
                    if let Some(meta) = vector_metadata.get(ext_id) {
                        if !matches_filter(meta, filter_conds)? {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let score = match params.rerank.as_ref() {
                    Some(plan) => {
                        rescore_candidate(plan, query, ext_id, vectors, self.pq.as_ref(), pq_codes)
                            .unwrap_or(f32::INFINITY)
                    }
                    None => neighbor.distance,
                };

                scored.push((ext_id, score));
            }
        }

        if params.rerank.is_some() {
            take_best(&mut scored, params.top_k);
        }

        let mut results = Vec::with_capacity(scored.len());
        for (ext_id, score) in scored {
            let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
            // The raw vector where one exists and the reconstruction from the
            // codes where none does. Under `quantized_only` every record is
            // code held once training completes, so without the fallback a
            // search returns no vectors at all.
            let vector_data = if params.return_vector {
                vectors.get(ext_id).cloned().or_else(|| {
                    let codes = pq_codes.get(ext_id)?;
                    self.pq.as_ref()?.reconstruct(codes).ok()
                })
            } else {
                None
            };

            results.push((ext_id.clone(), score, metadata, vector_data));
        }

        Ok(results)
    }

    /// One query's hits as the list of dicts Python receives.
    fn hits_to_python(&self, hits: QueryHits, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut output = Vec::with_capacity(hits.len());
        for (id, score, metadata, vector_data) in hits {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("score", score)?;
            dict.set_item("metadata", value_map_to_python(&metadata, py)?)?;
            if let Some(vec) = vector_data {
                dict.set_item("vector", vec)?;
            }
            output.push(dict.into());
        }
        Ok(output)
    }

    /// A batch's hits as one list of dicts per query, in query order.
    fn batch_hits_to_python(
        &self,
        batches: Vec<QueryHits>,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let mut output = Vec::with_capacity(batches.len());
        for hits in batches {
            output.push(self.hits_to_python(hits, py)?);
        }
        Ok(output)
    }

    /// The single query search path
    ///
    /// Lifted out of the `search` entry point so that all four paths, this one,
    /// the sequential batch, the parallel batch and the benchmark, sit together
    /// and take their guards in the one documented order.
    pub(super) fn single_search_internal(
        &self,
        processed_query: &[f32],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let search_results = py.detach(|| -> PyResult<QueryHits> {
            // Check if we should use quantized search
            let use_quantized = self.is_quantized();

            trace!(
                operation = "search_method",
                use_quantized = use_quantized,
                "Selected search method"
            );

            // One read guard for the whole search, taken before the graph lock
            // and held across it. The predicate runs once per candidate the
            // traversal visits, so acquiring the guard inside it would put a
            // lock acquisition on the hot path. See the same pattern in the
            // two batch paths.
            let rev_map = self.rev_map.read().unwrap();
            let live = |internal_id: &usize| rev_map.contains_key(internal_id);

            let fetch_k = params.fetch_k(rev_map.len());

            let hnsw_results = {
                let hnsw_guard = self.hnsw.read().unwrap();

                if use_quantized {
                    // Use ADC search for quantized index
                    hnsw_guard
                        .search(processed_query, fetch_k, params.ef, Some(&live))
                        .unwrap_or_else(|e| {
                            error!(operation = "adc_search", error = %e, "ADC search failed");
                            Vec::new()
                        })
                } else {
                    // Use raw vector search
                    match hnsw_guard.search(processed_query, fetch_k, params.ef, Some(&live)) {
                        Ok(results) => results,
                        Err(e) => {
                            error!(operation = "raw_search", error = %e, "Raw search failed");
                            Vec::new()
                        }
                    }
                }
            };

            // Process results with enhanced vector retrieval
            let vectors = self.vectors.read().unwrap();
            let pq_codes = self.pq_codes.read().unwrap();
            let vector_metadata = self.vector_metadata.read().unwrap();

            self.collect_hits(
                hnsw_results,
                processed_query,
                &rev_map,
                &vectors,
                &pq_codes,
                &vector_metadata,
                filter_conditions,
                params,
            )
        })?;

        self.hits_to_python(search_results, py)
    }

    /// Internal batch search method for multiple query vectors
    #[instrument(level = "debug", skip(self, vectors, filter_conditions, params, py), fields(
        batch_size = vectors.len(),
        top_k = params.top_k,
        ef = params.ef,
        return_vector = params.return_vector,
        has_filter = filter_conditions.is_some(),
        rerank_factor = params.rerank.and_then(|plan| plan.factor)
    ), err)]
    pub(super) fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let start_time = Instant::now();

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                error!(
                    operation = "batch_search_validation",
                    vector_index = i,
                    expected_dim = self.dim,
                    actual_dim = vector.len(),
                    "Vector dimension mismatch in batch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector {}: dimension mismatch: expected {}, got {}",
                    i,
                    self.dim,
                    vector.len()
                )));
            }

            // The same value check the single query path applies. A non-finite
            // component survives normalization, because the norm of a vector
            // containing one is not greater than zero, and the search then
            // returns hits whose scores carry no distance information. The
            // message names the batch entry as well as the component, so one
            // bad vector is findable in a batch of thousands.
            for (component, &value) in vector.iter().enumerate() {
                if !value.is_finite() {
                    error!(
                        operation = "batch_search_validation",
                        vector_index = i,
                        value_index = component,
                        value = value,
                        "Vector in batch contains invalid value"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Vector {} in batch contains invalid value at index {}: {} (must be finite)",
                        i, component, value
                    )));
                }
            }
        }

        // Choose strategy based on batch size
        let result = if vectors.len() <= 5 {
            trace!(
                operation = "batch_search_strategy",
                strategy = "sequential",
                "Using sequential processing"
            );
            self.batch_search_sequential(vectors, filter_conditions, params, py)
        } else {
            trace!(
                operation = "batch_search_strategy",
                strategy = "parallel",
                "Using parallel processing"
            );
            self.batch_search_parallel(vectors, filter_conditions, params, py)
        };

        // ✅ ENTERPRISE: Add duration timing to hot path
        let duration_ms = start_time.elapsed().as_millis();
        debug!(
            operation = "batch_search_complete",
            batch_size = vectors.len(),
            duration_ms = duration_ms,
            "Batch search completed"
        );

        result
    }

    /// Sequential batch processing (for small batches)
    fn batch_search_sequential(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.detach(|| -> PyResult<Vec<QueryHits>> {
            // The read guard is taken before the graph lock and held across every
            // query in the batch, so the traversal predicate below is a hash lookup
            // rather than a lock acquisition.
            let rev_map = self.rev_map.read().unwrap();
            let live = |internal_id: &usize| rev_map.contains_key(internal_id);

            let hnsw_guard = self.hnsw.read().unwrap();
            let vector_store = self.vectors.read().unwrap();
            let code_store = self.pq_codes.read().unwrap();
            let metadata_store = self.vector_metadata.read().unwrap();

            // The same over-fetch the single query path applies, so a batch of
            // one query returns what that query returns on its own.
            let fetch_k = params.fetch_k(rev_map.len());

            let mut all_results = Vec::with_capacity(vectors.len());

            for vector in vectors {
                // FIX: Process each query vector for space
                let processed_query = self.process_vector_for_space(vector.clone());

                let neighbors = hnsw_guard
                    .search(&processed_query, fetch_k, params.ef, Some(&live))
                    .unwrap_or_else(|_| Vec::new());

                all_results.push(self.collect_hits(
                    neighbors,
                    &processed_query,
                    &rev_map,
                    &vector_store,
                    &code_store,
                    &metadata_store,
                    filter_conditions,
                    params,
                )?);
            }

            Ok(all_results)
        })?;

        self.batch_hits_to_python(rust_results, py)
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let span = tracing::Span::current();
        let rust_results = py.detach(|| -> PyResult<Vec<QueryHits>> {
            let results: PyResult<Vec<QueryHits>> = vectors
                .par_iter()
                .map(|vector| -> PyResult<QueryHits> {
                    let _entered = span.clone().entered();
                    // FIX: Process each query vector for space
                    let processed_query = self.process_vector_for_space(vector.clone());

                    // Taken before the graph lock, as in the other two search paths,
                    // so every path acquires these two locks in the same order.
                    let rev_map = self.rev_map.read().unwrap();
                    let live = |internal_id: &usize| rev_map.contains_key(internal_id);

                    // The same over-fetch the other two search paths apply.
                    let fetch_k = params.fetch_k(rev_map.len());

                    // Brief HNSW search (individual lock per query)
                    let neighbors = {
                        let hnsw_guard = self.hnsw.read().unwrap();
                        hnsw_guard
                            .search(&processed_query, fetch_k, params.ef, Some(&live))
                            .unwrap_or_else(|_| Vec::new())
                    };

                    // Concurrent data lookup
                    let vector_store = self.vectors.read().unwrap();
                    let code_store = self.pq_codes.read().unwrap();
                    let metadata_store = self.vector_metadata.read().unwrap();

                    self.collect_hits(
                        neighbors,
                        &processed_query,
                        &rev_map,
                        &vector_store,
                        &code_store,
                        &metadata_store,
                        filter_conditions,
                        params,
                    )
                })
                .collect();

            results
        })?;

        self.batch_hits_to_python(rust_results, py)
    }

    /// Raw search without Python objects (for benchmarking)
    pub(super) fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // Concurrent read access to ID mapping, taken before the graph lock so the
        // traversal predicate can consult it without acquiring anything itself.
        let rev_map = self.rev_map.read().unwrap();
        let live = |internal_id: &usize| rev_map.contains_key(internal_id);

        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard
                .search(query, 10, 100, Some(&live))
                .unwrap_or_else(|_| Vec::new())
        }; // Lock released immediately

        hnsw_results
            .into_iter()
            .filter_map(|neighbor| {
                rev_map
                    .get(&neighbor.internal_id)
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }
}
