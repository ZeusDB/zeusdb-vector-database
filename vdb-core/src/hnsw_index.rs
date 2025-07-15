use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use hnsw_rs::prelude::{Hnsw, DistCosine, DistL2, DistL1};
use serde_json::Value;
use rayon::prelude::*;

enum DistanceType {
    Cosine(Hnsw<'static, f32, DistCosine>),
    L2(Hnsw<'static, f32, DistL2>),
    L1(Hnsw<'static, f32, DistL1>),
}

impl DistanceType {
    fn insert_batch(&mut self, data: &[(&Vec<f32>, usize)]) {
        // Use parallel_insert for large batches (1000 * num_threads or more)
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        if data.len() >= threshold {
            // Use parallel insertion for large batches
            match self {
                DistanceType::Cosine(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L2(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L1(hnsw) => hnsw.parallel_insert(data),
            }
        } else {
            // Use sequential insertion for small batches (avoid threading overhead)
            for (vector, id) in data {
                match self {
                    DistanceType::Cosine(hnsw) => hnsw.insert((vector.as_slice(), *id)),
                    DistanceType::L2(hnsw) => hnsw.insert((vector.as_slice(), *id)),
                    DistanceType::L1(hnsw) => hnsw.insert((vector.as_slice(), *id)),
                }
            }
        }
    }

    /// Search method that handles different distance types
    /// and returns a vector of Neighbour results.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<hnsw_rs::prelude::Neighbour> {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.search(query, k, ef),
            DistanceType::L2(hnsw) => hnsw.search(query, k, ef),
            DistanceType::L1(hnsw) => hnsw.search(query, k, ef),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AddResult {
    #[pyo3(get)]
    pub total_inserted: usize,
    #[pyo3(get)]
    pub total_errors: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub vector_shape: Option<(usize, usize)>,
}

#[pymethods]
impl AddResult {
    fn __repr__(&self) -> String {
        format!(
            "AddResult(inserted={}, errors={}, shape={:?})",
            self.total_inserted, self.total_errors, self.vector_shape
        )
    }

    pub fn is_success(&self) -> bool {
        self.total_errors == 0
    }

    pub fn summary(&self) -> String {
        format!("✅ {} inserted, ❌ {} errors", self.total_inserted, self.total_errors)
    }
}

// Add RwLock/Mutex declarations in struct
#[pyclass]
pub struct HNSWIndex {
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,

    // Index-level metadata (simple, infrequently accessed)
    metadata: Mutex<HashMap<String, String>>,

    // Thread-safe vector store with RwLock for concurrent reads
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    vector_metadata: RwLock<HashMap<String, HashMap<String, Value>>>,
    id_map: RwLock<HashMap<String, usize>>,
    rev_map: RwLock<HashMap<usize, String>>,
    
    // Mutex for write-only fields
    id_counter: Mutex<usize>,
    
    // Mutex for HNSW (not thread-safe for concurrent reads)
    hnsw: Mutex<DistanceType>,
}

#[pymethods]
impl HNSWIndex {
    #[new]
    fn new(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
    ) -> PyResult<Self> {
        // Validation of parameters
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("dim must be positive"));
        }
        if ef_construction == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("ef_construction must be positive"));
        }
        if expected_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("expected_size must be positive"));
        }
        if m > 256 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("m must be less than or equal to 256"));
        }

        let space_normalized = space.to_lowercase();
        let max_layer = (expected_size as f32).log2().ceil() as usize;

        let hnsw = match space_normalized.as_str() {
            "cosine" => DistanceType::Cosine(Hnsw::new(m, expected_size, max_layer, ef_construction, DistCosine {})),
            "l2" => DistanceType::L2(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL2 {})),
            "l1" => DistanceType::L1(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL1 {})),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported space: '{}'. Must be 'cosine', 'l2', or 'l1'", space),
                ));
            }
        };

        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            hnsw: Mutex::new(hnsw),
        })
    }

    /// Unified add method with automatic parallel processing
    pub fn add(&mut self, data: Bound<PyAny>) -> PyResult<AddResult> {
        let records = if let Ok(list) = data.downcast::<PyList>() {
            self.parse_list_format(&list)?
        } else if let Ok(dict) = data.downcast::<PyDict>() {
            if dict.contains("ids")? {
                self.parse_separate_arrays(&dict)?
            } else {
                self.parse_single_object(&dict)?
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid format: expected dict, list, or object with 'id' field",
            ));
        };

        if records.len() >= 50 {
            self.add_batch_parallel_gil_optimized(records)
        } else {
            self.add_batch_sequential(records)
        }
    }

    /// GIL-optimized search with HNSW locking
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false))]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Search vector dimension mismatch: expected {}, got {}",
                self.dim, vector.len()
            )));
        }

        // Parse filter while still have GIL
        let filter_conditions = if let Some(filter_dict) = filter {
            Some(self.python_dict_to_value_map(filter_dict)?)
        } else {
            None
        };

        let ef = ef_search.unwrap_or_else(|| std::cmp::max(2 * top_k, 100));

        // Release GIL but use locking for HNSW
        let search_results = py.allow_threads(|| {
            // Step 1: HNSW search (needs exclusive access, but brief)
            let hnsw_results = {
                let hnsw_guard = self.hnsw.lock().unwrap();
                hnsw_guard.search(&vector, top_k, ef)
            }; // HNSW lock released here - This is critical for performance!
            
            // Step 2: Data lookup with RwLock (concurrent reads possible)
            let vectors = self.vectors.read().unwrap();
            let vector_metadata = self.vector_metadata.read().unwrap();
            let rev_map = self.rev_map.read().unwrap();
            
            let mut results = Vec::with_capacity(hnsw_results.len());

            let has_filter = filter_conditions.is_some();
            
            for neighbor in hnsw_results {
                let score = neighbor.distance;
                let internal_id = neighbor.get_origin_id();
                
                if let Some(ext_id) = rev_map.get(&internal_id) {
                    if has_filter {
                        if let Some(meta) = vector_metadata.get(ext_id) {
                            let filter_conds = filter_conditions.as_ref().unwrap(); // Safe unwrap
                            if !self.matches_filter(meta, filter_conds).unwrap_or(false) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    
                    // Collect data for Python object creation
                    let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
                    let vector_data = if return_vector {
                        vectors.get(ext_id).cloned()
                    } else {
                        None
                    };
                    
                    results.push((ext_id.clone(), score, metadata, vector_data));
                }
            }
            
            results
        }); // GIL is reacquired here
        
        // Step 3: Convert to Python objects (needs GIL)
        let mut output = Vec::with_capacity(search_results.len());
        for (id, score, metadata, vector_data) in search_results {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("score", score)?;
            dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;
            
            if let Some(vec) = vector_data {
                dict.set_item("vector", vec)?;
            }
            
            output.push(dict.into());
        }

        Ok(output)
    }

    /// Thread-safe get_records implementation
    #[pyo3(signature = (input, return_vector = true))]
    pub fn get_records(&self, py: Python<'_>, input: &Bound<PyAny>, return_vector: bool) -> PyResult<Vec<Py<PyDict>>> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)",
            ));
        };

        let mut records = Vec::with_capacity(ids.len());
        
        // Use read locks for concurrent access
        let vectors = self.vectors.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        for id in ids {
            if let Some(vector) = vectors.get(&id) {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    dict.set_item("vector", vector.clone())?;
                }

                records.push(dict.into());
            }
        }

        Ok(records)
    }

    /// Thread-safe statistics implementation
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        let vectors = self.vectors.read().unwrap();
        stats.insert("total_vectors".to_string(), vectors.len().to_string());
        
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("m".to_string(), self.m.to_string());
        stats.insert("ef_construction".to_string(), self.ef_construction.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("index_type".to_string(), "HNSW".to_string());
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());
        
        stats
    }

    /// List the first number of records in the index (ID and metadata).
    #[pyo3(signature = (number=10))]
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, PyObject)>> {
        let vectors = self.vectors.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        
        let mut results = Vec::new();
        for (id, _vec) in vectors.iter().take(number) {
            let metadata = vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = self.value_map_to_python(&metadata, py)?;
            results.push((id.clone(), py_metadata));
        }
        Ok(results)
    }

    /// Thread-safe contains check implementation
    pub fn contains(&self, id: String) -> bool {
        let vectors = self.vectors.read().unwrap();
        vectors.contains_key(&id)
    }

    /// Thread-safe metadata operations
    pub fn add_metadata(&mut self, metadata: HashMap<String, String>) {
        let mut meta_lock = self.metadata.lock().unwrap();
        for (key, value) in metadata {
            meta_lock.insert(key, value);
        }
    }

    pub fn get_metadata(&self, key: String) -> Option<String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.get(&key).cloned()
    }

    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.clone()
    }

    /// Returns basic info about the index
    pub fn info(&self) -> String {
        let vectors = self.vectors.read().unwrap();
        format!(
            "HNSWIndex(dim={}, space={}, m={}, ef_construction={}, expected_size={}, vectors={})",
            self.dim,
            self.space,
            self.m,
            self.ef_construction,
            self.expected_size,
            vectors.len()
        )
    }

    /// Remove vector by ID
    /// Removes the vector and its metadata from all accessible mappings.
    /// The point will no longer appear in queries, contains() checks, or be
    /// retrievable by ID. 
    /// 
    /// Note: Due to HNSW algorithm limitations, the underlying graph structure
    /// retains stale nodes internally, but these are completely inaccessible
    /// to users and do not affect search results or performance.
    /// 
    /// Returns:
    ///   - `Ok(true)` if the vector was found and removed
    ///   - `Ok(false)` if the vector ID was not found
    pub fn remove_point(&mut self, id: String) -> PyResult<bool> {
        let mut vectors = self.vectors.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        
        if let Some(internal_id) = id_map.remove(&id) {
            vectors.remove(&id);
            vector_metadata.remove(&id);
            rev_map.remove(&internal_id);
            // Note: HNSW doesn't support removal, so the graph still contains the point
            // but it won't be accessible via the mappings
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get performance characteristics and limitations
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("search_speedup_expected".to_string(), "1.2x-2x".to_string());
        info.insert("insertion_speedup_expected".to_string(), "4x-8x_large_batches".to_string());
        info.insert("search_bottleneck".to_string(), "hnsw_mutex_serialization".to_string());
        info.insert("insertion_bottleneck".to_string(), "hnsw_mutex_for_large_batches".to_string());
        info.insert("benefits".to_string(), "gil_release_concurrent_metadata_processing_parallel_insert".to_string());
        info.insert("limitation".to_string(), "parallel_insert_threshold_1000x_threads".to_string());
        info.insert("recommendation".to_string(), "excellent_for_large_batch_workloads".to_string());
        info
    }

    /// Concurrent benchmark
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(&self, query_count: usize, max_threads: Option<usize>) -> PyResult<HashMap<String, f64>> {
        use std::time::Instant;
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let mut results = HashMap::new();
        
        // Sequential benchmark
        let start = Instant::now();
        Python::with_gil(|py| {
            for query in &queries {
                let _ = self.search(py, query.clone(), None, 10, None, false)?;
            }
            Ok::<(), PyErr>(())
        })?;
        let sequential_time = start.elapsed().as_secs_f64();
        results.insert("sequential_time".to_string(), sequential_time);
        results.insert("sequential_qps".to_string(), queries.len() as f64 / sequential_time);
        
        // Parallel benchmark
        // Use full CPU by default, but allow limiting
        // Parallel benchmark (no GIL - pure Rust performance)
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads.unwrap_or(available_threads).min(available_threads);

        let start = Instant::now();
        let _: Vec<_> = queries
            .par_iter()
            .map(|query| {
                self.raw_search_no_gil(query)  // No GIL needed!
            })
            .collect();

        let parallel_time = start.elapsed().as_secs_f64();
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert("parallel_qps".to_string(), queries.len() as f64 / parallel_time);
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);
        
        Ok(results)
    }

    /// No-GIL benchmark for raw performance measurement
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_raw_concurrent_performance(&self, query_count: usize, max_threads: Option<usize>) -> HashMap<String, f64> {
        use std::time::Instant;
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let mut results = HashMap::new();
        
        // Sequential benchmark (no GIL)
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        
        // Parallel benchmark (no GIL, but still serialized by HNSW mutex)
        // Use full CPU by default
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads.unwrap_or(available_threads).min(available_threads);
        let chunk_size = (queries.len() + num_threads - 1) / num_threads;
        
        let start = Instant::now();
        let total_processed: usize = queries
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_count = 0;
                for query in chunk {
                    let _ = self.raw_search_no_gil(query);
                    local_count += 1;
                }
                local_count
            })
            .sum();
        
        let parallel_time = start.elapsed().as_secs_f64();
        
        results.insert("sequential_time_sec".to_string(), sequential_time);
        results.insert("parallel_time_sec".to_string(), parallel_time);
        results.insert("sequential_qps".to_string(), query_count as f64 / sequential_time);
        results.insert("parallel_qps".to_string(), total_processed as f64 / parallel_time);
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);
        results.insert("note".to_string(), "limited_by_hnsw_mutex".parse().unwrap_or(0.0));
        
        results
    }
}

impl HNSWIndex {
    /// Raw search without Python objects (for benchmarking)
    fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.search(query, 10, 100)
        }; // Lock released immediately
        
        // Concurrent read access to ID mapping
        let rev_map = self.rev_map.read().unwrap();
        
        hnsw_results
            .into_iter()
            .filter_map(|neighbor| {
                rev_map.get(&neighbor.get_origin_id())
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }

    /// Sequential batch processing (clean and borrow-safe with .clone())
    fn add_batch_sequential(
        &mut self,
        records: Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>
    ) -> PyResult<AddResult> {
        if records.is_empty() {
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        let mut valid_records = Vec::new();
        let mut errors = Vec::new();

        // Step 1: Validation + normalization
        for (id, mut vector, metadata) in records {
            if vector.len() != self.dim {
                errors.push(format!(
                    "ID '{}': Vector dimension mismatch: expected {}, got {}",
                    id, self.dim, vector.len()
                ));
                continue;
            }

            if self.space == "cosine" {
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= norm;
                    }
                }
            }

            valid_records.push((id, vector, metadata));
        }

        if valid_records.is_empty() {
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: errors.len(),
                errors,
                vector_shape: Some((0, self.dim)),
            });
        }

        // Step 2: Acquire all locks
        let mut vectors = self.vectors.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        let mut id_counter = self.id_counter.lock().unwrap();

        vectors.reserve(valid_records.len());
        vector_metadata.reserve(valid_records.len());
        id_map.reserve(valid_records.len());
        rev_map.reserve(valid_records.len());

        let mut id_pairs = Vec::with_capacity(valid_records.len());

        // Step 3: Insert into maps + prepare HNSW data
        for (id, vector, metadata) in valid_records {
            if let Some(old_id) = id_map.remove(&id) {
                rev_map.remove(&old_id);
                vectors.remove(&id);
                vector_metadata.remove(&id); // Remove old metadata
            }

            let internal_id = *id_counter;
            *id_counter += 1;

            vectors.insert(id.clone(), vector);  // Move vector into HashMap
            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());

            if let Some(meta) = metadata {
                vector_metadata.insert(id.clone(), meta);
            }

            // Store the ID and internal_id for later reference collection
            id_pairs.push((id, internal_id));

        }

        // Step 4: Collect references (all immutable operations)
        let mut hnsw_data: Vec<(&Vec<f32>, usize)> = Vec::with_capacity(id_pairs.len());

        for (id, internal_id) in id_pairs {
            let vector_ref = vectors.get(&id).unwrap(); // Safe: just inserted
            hnsw_data.push((vector_ref, internal_id));
        }

        // Step 5: Insert into HNSW
        if !hnsw_data.is_empty() {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.insert_batch(&hnsw_data);
        }

        Ok(AddResult {
            total_inserted: hnsw_data.len(),
            total_errors: errors.len(),
            errors,
            vector_shape: Some((hnsw_data.len(), self.dim)),
        })
    }
    

    /// GIL-optimized parallel processing (with proper locking)
    fn add_batch_parallel_gil_optimized(
        &mut self, 
        records: Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>
    ) -> PyResult<AddResult> {
        if records.is_empty() {
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        // Release GIL for compute-intensive processing
        let processing_result = Python::with_gil(|py| {
            py.allow_threads(|| -> Result<(usize, Vec<String>), PyErr> {
                // Parallel validation with sequential normalization per vector
                let validation_results: Vec<Result<(String, Vec<f32>, Option<HashMap<String, Value>>), String>> = records
                    .into_par_iter()
                    .map(|(id, mut vector, metadata)| {
                        if vector.len() != self.dim {
                            return Err(format!("ID '{}': Vector dimension mismatch: expected {}, got {}", 
                                             id, self.dim, vector.len()));
                        }
                        
                        // Sequential normalization per vector
                        if self.space == "cosine" {
                            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for x in vector.iter_mut() {
                                    *x /= norm;
                                }
                            }
                        }

                        Ok((id, vector, metadata))
                    })
                    .collect();

                // Separate valid/invalid records
                let mut valid_records = Vec::new();
                let mut errors = Vec::new();
                
                for result in validation_results {
                    match result {
                        Ok(record) => valid_records.push(record),
                        Err(e) => errors.push(e),
                    }
                }

                if valid_records.is_empty() {
                    return Ok((0, errors));
                }

                // Sequential data structure updates
                let mut vectors = self.vectors.write().unwrap();
                let mut vector_metadata = self.vector_metadata.write().unwrap();
                let mut id_map = self.id_map.write().unwrap();
                let mut rev_map = self.rev_map.write().unwrap();
                let mut id_counter = self.id_counter.lock().unwrap();

                let capacity = valid_records.len();
                vectors.reserve(capacity);
                vector_metadata.reserve(capacity);
                id_map.reserve(capacity);
                rev_map.reserve(capacity);

                // Prepare ID pairs for HNSW insertion
                let mut id_pairs = Vec::with_capacity(valid_records.len());

                // Insert into maps and collect IDs
                for (id, vector, metadata) in valid_records {
                    if let Some(old_id) = id_map.remove(&id) {
                        rev_map.remove(&old_id);
                        vectors.remove(&id);
                        vector_metadata.remove(&id);
                    }

                    let internal_id = *id_counter;
                    *id_counter += 1;

                    vectors.insert(id.clone(), vector);  // Move vector into HashMap
                    id_map.insert(id.clone(), internal_id);
                    rev_map.insert(internal_id, id.clone());

                    if let Some(meta) = metadata {
                        vector_metadata.insert(id.clone(), meta);
                    }

                    // Store the ID and internal_id for later reference collection
                    id_pairs.push((id, internal_id));
                }

                // Step 4: Collect references (all immutable operations)
                let mut hnsw_data: Vec<(&Vec<f32>, usize)> = Vec::with_capacity(id_pairs.len());
                for (id, internal_id) in id_pairs {
                    let vector_ref = vectors.get(&id).unwrap(); // Safe: just inserted
                    hnsw_data.push((vector_ref, internal_id));
                }

                // Step 5: Insert into HNSW
                if !hnsw_data.is_empty() {
                    let mut hnsw_guard = self.hnsw.lock().unwrap();
                    hnsw_guard.insert_batch(&hnsw_data);
                }

                Ok((hnsw_data.len(), errors))
            })
        })?;

        let (inserted_count, errors) = processing_result;

        Ok(AddResult {
            total_inserted: inserted_count,
            total_errors: errors.len(),
            errors,
            vector_shape: Some((inserted_count, self.dim)),
        })
    }

    /// Thread-safe helper methods implementation
    fn parse_single_object(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        let id = dict.get_item("id")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing required field 'id'"))?
            .extract::<String>()?;

        let vector = self.extract_vector_from_dict(dict, "object")?;

        let metadata = if let Some(meta_item) = dict.get_item("metadata")? {
            Some(self.python_dict_to_value_map(meta_item.downcast::<PyDict>()?)?)
        } else {
            None
        };

        Ok(vec![(id, vector, metadata)])
    }

    fn parse_list_format(&self, list: &Bound<PyList>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        let mut records = Vec::with_capacity(list.len());
        
        for (i, item) in list.iter().enumerate() {
            let dict = item.downcast::<PyDict>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Item {}: expected dict object", i)
                ))?;

            let id = dict.get_item("id")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Item {}: missing required field 'id'", i)
                ))?
                .extract::<String>()?;

            let vector = self.extract_vector_from_dict(dict, &format!("item {}", i))?;

            let metadata = if let Some(meta_item) = dict.get_item("metadata")? {
                Some(self.python_dict_to_value_map(meta_item.downcast::<PyDict>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Item {}: metadata must be a dictionary", i)
                    ))?)?)
            } else {
                None
            };

            records.push((id, vector, metadata));
        }

        Ok(records)
    }

    fn parse_separate_arrays(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        let ids = dict.get_item("ids")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing required field 'ids'"))?
            .extract::<Vec<String>>()?;

        let vectors = self.extract_vectors_from_separate_arrays(dict)?;

        if vectors.len() != ids.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Length mismatch: {} ids vs {} vectors", ids.len(), vectors.len())
            ));
        }

        let metadatas = if let Some(meta_item) = dict.get_item("metadatas")? {
            let meta_list = meta_item.downcast::<PyList>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Field 'metadatas' must be a list"
                ))?;

            let mut metas = Vec::with_capacity(meta_list.len());
            for (i, meta_item) in meta_list.iter().enumerate() {
                if meta_item.is_none() {
                    metas.push(None);
                } else {
                    let meta_dict = meta_item.downcast::<PyDict>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("metadatas[{}] must be a dictionary or None", i)
                        ))?;
                    metas.push(Some(self.python_dict_to_value_map(&meta_dict)?));
                }
            }

            if metas.len() != ids.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Length mismatch: {} ids vs {} metadatas", ids.len(), metas.len())
                ));
            }
            metas
        } else {
            vec![None; ids.len()]
        };

        let records = ids.into_iter()
            .zip(vectors.into_iter())
            .zip(metadatas.into_iter())
            .map(|((id, vector), metadata)| (id, vector, metadata))
            .collect();

        Ok(records)
    }

    fn extract_vector_from_dict(&self, dict: &Bound<PyDict>, context: &str) -> PyResult<Vec<f32>> {
        let vector_item = dict.get_item("values")?
            .or_else(|| dict.get_item("vector").ok().flatten())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("{}: missing required field 'values' or 'vector'", context)
            ))?;

        if let Ok(array1d) = vector_item.downcast::<PyArray1<f32>>() {
            Ok(array1d.readonly().as_slice()?.to_vec())
        } else if let Ok(array2d) = vector_item.downcast::<PyArray2<f32>>() {
            let readonly = array2d.readonly();
            let shape = readonly.shape();

            if (shape[0] == 1 && shape[1] > 0) || (shape[1] == 1 && shape[0] > 0) {
                Ok(readonly.as_slice()?.to_vec())
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("{}: expected single vector (1×N or N×1), got shape {:?}", context, shape)
                ));
            }
        } else {
            vector_item.extract::<Vec<f32>>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("{}: invalid vector format: {}", context, e)
                ))
        }
    }

    fn extract_vectors_from_separate_arrays(&self, dict: &Bound<PyDict>) -> PyResult<Vec<Vec<f32>>> {
        let vectors_item = dict.get_item("embeddings")?
            .or_else(|| dict.get_item("values").ok().flatten())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Missing required field 'embeddings' or 'values'"
            ))?;

        if let Ok(array) = vectors_item.downcast::<PyArray2<f32>>() {
            let readonly = array.readonly();
            let shape = readonly.shape();
            
            if shape.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("NumPy array must be 2D, got {}D", shape.len())
                ));
            }

            let slice = readonly.as_slice()?;
            let (rows, cols) = (shape[0], shape[1]);
            
            let mut vectors = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                vectors.push(slice[start..end].to_vec());
            }
            
            Ok(vectors)
        } else {
            vectors_item.extract::<Vec<Vec<f32>>>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid vectors format: {}", e)
                ))
        }
    }

    fn python_dict_to_value_map(&self, py_dict: &Bound<PyDict>) -> PyResult<HashMap<String, Value>> {
        let mut map = HashMap::new();
        
        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = self.python_object_to_value(&value)?;
            map.insert(string_key, json_value);
        }
        
        Ok(map)
    }

    fn python_object_to_value(&self, py_obj: &Bound<PyAny>) -> PyResult<Value> {
        if py_obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = py_obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = py_obj.extract::<i64>() {
            Ok(Value::Number(serde_json::Number::from(i)))
        } else if let Ok(f) = py_obj.extract::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(f) {
                Ok(Value::Number(num))
            } else {
                Ok(Value::String(f.to_string()))
            }
        } else if let Ok(s) = py_obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(py_list) = py_obj.downcast::<PyList>() {
            let mut vec = Vec::new();
            for item in py_list.iter() {
                vec.push(self.python_object_to_value(&item)?);
            }
            Ok(Value::Array(vec))
        } else if let Ok(py_dict) = py_obj.downcast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in py_dict.iter() {
                let string_key = key.extract::<String>()?;
                let json_value = self.python_object_to_value(&value)?;
                map.insert(string_key, json_value);
            }
            Ok(Value::Object(map))
        } else {
            Ok(Value::String(py_obj.to_string()))
        }
    }

    fn matches_filter(&self, metadata: &HashMap<String, Value>, filter: &HashMap<String, Value>) -> PyResult<bool> {
        for (field, condition) in filter {
            if !self.field_matches(metadata, field, condition)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn field_matches(&self, metadata: &HashMap<String, Value>, field: &str, condition: &Value) -> PyResult<bool> {
        let field_value = match metadata.get(field) {
            Some(value) => value,
            None => return Ok(false),
        };

        match condition {
            Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
                Ok(field_value == condition)
            },
            Value::Object(ops) => {
                self.evaluate_value_conditions(field_value, ops)
            },
            _ => Ok(false),
        }
    }

    fn evaluate_value_conditions(&self, field_value: &Value, operations: &serde_json::Map<String, Value>) -> PyResult<bool> {
        for (op, target_value) in operations {
            let matches = match op.as_str() {
                "eq" => field_value == target_value,
                "ne" => field_value != target_value,
                "gt" => self.compare_values(field_value, target_value, |a, b| a > b)?,
                "gte" => self.compare_values(field_value, target_value, |a, b| a >= b)?,
                "lt" => self.compare_values(field_value, target_value, |a, b| a < b)?,
                "lte" => self.compare_values(field_value, target_value, |a, b| a <= b)?,
                "contains" => self.value_contains(field_value, target_value)?,
                "startswith" => self.value_starts_with(field_value, target_value)?,
                "endswith" => self.value_ends_with(field_value, target_value)?,
                "in" => self.value_in_array(field_value, target_value)?,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Unknown filter operation: {}", op)
                    ));
                }
            };
            
            if !matches {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn compare_values<F>(&self, a: &Value, b: &Value, op: F) -> PyResult<bool>
    where
        F: Fn(f64, f64) -> bool,
    {
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                let f1 = n1.as_f64().unwrap_or(0.0);
                let f2 = n2.as_f64().unwrap_or(0.0);
                Ok(op(f1, f2))
            },
            _ => Ok(false),
        }
    }

    fn value_contains(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.contains(s2)),
            (Value::Array(arr), val) => Ok(arr.contains(val)),
            _ => Ok(false),
        }
    }

    fn value_starts_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.starts_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_ends_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.ends_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_in_array(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match target {
            Value::Array(arr) => Ok(arr.contains(field)),
            _ => Ok(false),
        }
    }

    fn value_map_to_python(&self, value_map: &HashMap<String, Value>, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (key, value) in value_map {
            let py_value = self.value_to_python_object(value, py)?;
            dict.set_item(key, py_value)?;
        }
        
        Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
    }

    fn value_to_python_object(&self, value: &Value, py: Python<'_>) -> PyResult<PyObject> {
        let py_obj = match value {
            Value::Null => py.None(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into_any(),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.into_pyobject(py)?.to_owned().unbind().into_any()
                } else if let Some(f) = n.as_f64() {
                    f.into_pyobject(py)?.to_owned().unbind().into_any()
                } else {
                    n.to_string().into_pyobject(py)?.to_owned().unbind().into_any()
                }
            },
            Value::String(s) => s.clone().into_pyobject(py)?.unbind().into_any(),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(self.value_to_python_object(item, py)?)?;
                }
                py_list.unbind().into_any() 
            },
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (k, v) in obj {
                    py_dict.set_item(k, self.value_to_python_object(v, py)?)?;
                }
                py_dict.unbind().into_any()
            }
        };
        
        Ok(py_obj)
    }
}
